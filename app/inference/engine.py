"""FaceEngine: InsightFace buffalo_l with batched GPU inference.

Design goals
------------
* **One model instance** shared by every camera (loaded by the
  ``app.inference.server`` process, or optionally inside a worker when
  ``INFERENCE_MODE=inprocess``).
* **Batched detection**: N frames letterboxed and pushed through the SCRFD
  session in a *single* ``session.run`` when the exported ONNX graph supports
  a dynamic batch dimension.  A startup canary verifies this; any failure
  permanently falls back to per-frame detection (still on GPU).
* **Batched embeddings**: all faces from all frames in the batch are aligned
  on CPU and embedded in ONE ``recognition.get_feat(list)`` call (chunked by
  ``MAX_EMBEDDING_BATCH``).  This is the big win for group frames.
* **Graceful CPU fallback**: provider selection walks
  ``CUDA -> CPU``; if CUDA later blows up at warmup/first inference the
  engine re-initialises on CPU instead of crashing the service.

The batched detection path replicates InsightFace's letterbox + SCRFD decode
+ NMS math exactly (see insightface/model_zoo/scrfd.py), so results match the
per-frame path.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from ..config import Settings, settings as default_settings

logger = logging.getLogger("app.inference.engine")

EMBEDDING_DIM = 512


@dataclass
class Timings:
    det_ms: float = 0.0
    align_ms: float = 0.0
    emb_ms: float = 0.0
    total_ms: float = 0.0
    frames: int = 0
    faces: int = 0
    batched_detection: bool = False

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


# ----------------------------------------------------------- numpy SCRFD ops
def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds: list[np.ndarray] = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, i % 2 + 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


# --------------------------------------------------- Windows GPU DLL plumbing
_NVIDIA_DLL_DIRS: list[str] = []
_NVIDIA_DLL_HANDLES: list = []  # os.add_dll_directory handles must stay alive


def _register_nvidia_wheel_dlls() -> None:
    """Windows: expose pip ``nvidia-*`` wheel DLL directories to the loader.

    Two mechanisms because Windows has two different DLL search modes in play:

    * **PATH prepending** - ``onnxruntime_providers_cuda.dll`` resolves its
      imports (cublasLt, cudnn, cufft, ...) with the *standard* search order,
      which honours ``PATH`` but not ``AddDllDirectory``.
    * **``os.add_dll_directory()``** - cuDNN 9 loads its sub-libraries
      (``cudnn_engines_*``, ``cudnn_ops``, ...) at runtime with
      ``LOAD_LIBRARY_SEARCH_*`` flags that honour ``AddDllDirectory`` but
      *not* PATH; without this the first Conv fails with
      ``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`` even though the DLLs exist.

    Runs before every model (re)initialisation, so a fresh shell works
    without pre-configured PATH.  Never raises: registration problems are
    logged and the normal graceful CPU fallback takes over.
    """
    if os.name != "nt" or _NVIDIA_DLL_DIRS or _NVIDIA_DLL_HANDLES:
        return
    try:
        import sysconfig
        from pathlib import Path

        root = Path(sysconfig.get_path("purelib")) / "nvidia"
        if not root.is_dir():
            return
        found: list[str] = []
        for pkg in sorted(root.iterdir()):
            for sub in ("bin", "lib"):
                d = pkg / sub
                if d.is_dir():
                    found.append(str(d))
        if not found:
            return
        # (1) standard-search loads (providers_cuda imports) read PATH.
        current = os.environ.get("PATH", "")
        have = {p.strip(os.sep).lower() for p in current.split(os.pathsep) if p}
        missing = [d for d in found if d.strip(os.sep).lower() not in have]
        if missing:
            os.environ["PATH"] = os.pathsep.join(missing + [current])
        # (2) search-flag loads (cuDNN sub-libraries) read user DLL dirs.
        for d in found:
            try:
                _NVIDIA_DLL_HANDLES.append(os.add_dll_directory(d))
            except OSError:
                pass
        _NVIDIA_DLL_DIRS.extend(found)
        logger.debug("registered nvidia wheel DLL dirs: %s", found)
    except Exception:  # registration must never break model init
        logger.warning("nvidia wheel DLL dir registration failed", exc_info=True)


class FaceEngine:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        self._force_cpu = False
        self.app = None
        self.det_model = None
        self.rec_model = None
        self._det_batch_enabled = False
        self._det_input_size: tuple[int, int] | None = None  # (w, h)
        self._det_batch_warned = False
        self._init_model()

    # ------------------------------------------------------------- lifecycle
    def _select_providers(self) -> list:
        import onnxruntime as ort

        if self.cfg.force_cpu or self._force_cpu:
            logger.warning("using CPUExecutionProvider (force_cpu)")
            return ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return [
                (
                    "CUDAExecutionProvider",
                    {"device_id": self.cfg.cuda_device_index},
                ),
                "CPUExecutionProvider",
            ]
        logger.warning(
            "CUDAExecutionProvider not available in this onnxruntime build - "
            "falling back to CPU",
            extra={"available_providers": available},
        )
        return ["CPUExecutionProvider"]

    def _init_model(self) -> None:
        """(Re)load FaceAnalysis with the selected providers."""
        _register_nvidia_wheel_dlls()
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        providers = self._select_providers()
        cuda = str(providers[0][0] if isinstance(providers[0], tuple)
                   else providers[0]) == "CUDAExecutionProvider"
        ctx_id = self.cfg.cuda_device_index if cuda else -1

        self.app = FaceAnalysis(
            name=self.cfg.face_model_name,
            root=self.cfg.insightface_root,
            providers=providers,
        )
        self.app.prepare(
            ctx_id=ctx_id,
            det_thresh=self.cfg.det_thresh,
            det_size=self.cfg.det_size_list,
        )
        self.det_model = self.app.det_model
        self.rec_model = self.app.models.get("recognition")
        if self.rec_model is None:
            raise RuntimeError(
                f"model package {self.cfg.face_model_name!r} has no recognition sub-model"
            )
        if cuda and (self._active_providers() or [None])[0] != "CUDAExecutionProvider":
            # ORT drops CUDA *silently* when its DLLs (cuBLAS/cuDNN/CUDA
            # runtime) cannot be loaded - the session then runs on CPU while
            # get_available_providers() still lists CUDA.  Make it loud.
            logger.warning(
                "CUDA was requested but the session did not initialise it "
                "(missing CUDA/cuDNN DLLs? see ORT GPU requirements) - "
                "running on %s",
                (self._active_providers() or ["unknown"])[0],
                extra={"available_providers": ort.get_available_providers()},
            )
        logger.info(
            "face model loaded",
            extra={
                "model": self.cfg.face_model_name,
                "providers": self._active_providers(),
                "ctx_id": ctx_id,
                "det_sizes": self.cfg.det_sizes,
                "det_thresh": self.cfg.det_thresh,
            },
        )
        self._probe_batched_detection()

    def reset_to_cpu(self) -> None:
        """Recover from a CUDA failure by re-initialising on CPU."""
        logger.warning("re-initialising FaceEngine on CPUExecutionProvider")
        self._force_cpu = True
        self._det_batch_enabled = False
        self._init_model()

    def _active_providers(self) -> list[str]:
        try:
            return list(self.det_model.session.get_providers())
        except Exception:  # noqa: BLE001
            return []

    # -------------------------------------------------- batched det probing
    def _probe_batched_detection(self) -> None:
        """Canary: can we push 2 frames through the detector in one run?"""
        self._det_batch_enabled = False
        self._det_input_size = None
        if not self.cfg.enable_batched_detection:
            logger.info("batched detection disabled by config")
            return
        model = self.det_model
        try:
            shape = model.session.get_inputs()[0].shape
            batch_dim = shape[0] if len(shape) == 4 else 1
            if isinstance(batch_dim, int) and batch_dim == 1:
                logger.info(
                    "detector graph has a static batch of 1 - using per-frame "
                    "detection (embeddings are still batched)",
                    extra={"input_shape": list(shape)},
                )
                return
            outputs = model.session.get_outputs()
            if len(outputs[0].shape) != 3:
                logger.info(
                    "detector outputs are not batched (rank != 3) - "
                    "using per-frame detection"
                )
                return
            sizes = model._resolve_input_sizes(None)
            if len(sizes) != 1:
                logger.info(
                    "multi-scale detection configured - using per-frame path",
                    extra={"sizes": sizes},
                )
                return

            size = (int(sizes[0][0]), int(sizes[0][1]))
            rng = np.random.default_rng(0)
            dummy = rng.integers(0, 255, (360, 480, 3), dtype=np.uint8)
            results = self._detect_batch([dummy, dummy], size)
            if len(results) != 2:
                raise RuntimeError("canary returned wrong result count")
            self._det_batch_enabled = True
            self._det_input_size = size
            logger.info(
                "batched detection ENABLED (single session.run for N frames)",
                extra={"det_input_size": size},
            )
        except Exception as exc:  # noqa: BLE001 - canary decides for us
            logger.warning(
                "batched detection unavailable, using per-frame detection: %s",
                exc,
            )

    # ---------------------------------------------------------- detection
    def _detect_single(self, img: np.ndarray):
        det, kpss = self.det_model.detect(
            img, max_num=self.cfg.max_faces_per_frame
        )
        return det, kpss

    @staticmethod
    def _letterbox(
        img: np.ndarray, input_size: tuple[int, int], model
    ) -> tuple[np.ndarray, float]:
        """InsightFace-compatible resize + pad -> (3, h, w) float32 blob."""
        import cv2

        width, height = input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(height) / width
        if im_ratio > model_ratio:
            new_height = height
            new_width = int(new_height / im_ratio)
        else:
            new_width = width
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((height, width, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized
        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / model.input_std,
            (width, height),
            (model.input_mean, model.input_mean, model.input_mean),
            swapRB=True,
        )
        return np.ascontiguousarray(blob, dtype=np.float32), det_scale

    def _decode_batch_element(
        self,
        net_outs: list[np.ndarray],
        idx: int,
        det_scale: float,
        input_size: tuple[int, int],
        model,
    ):
        """Decode one element of a batched SCRFD output (mirrors forward())."""
        width, height = input_size
        threshold = getattr(model, "det_thresh", 0.5)
        fmc = model.fmc
        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []

        for i, stride in enumerate(model._feat_stride_fpn):
            scores = net_outs[i][idx]
            bbox_preds = net_outs[i + fmc][idx] * stride
            kps_preds = None
            if model.use_kps:
                kps_preds = net_outs[i + fmc * 2][idx] * stride

            h = height // stride
            w = width // stride
            key = (h, w, stride)
            anchor_centers = model.center_cache.get(key)
            if anchor_centers is None:
                anchor_centers = np.stack(
                    np.mgrid[:h, :w][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if model._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * model._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(model.center_cache) < 100:
                    model.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if model.use_kps:
                kpss = _distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[pos_inds])

        if not scores_list or sum(s.size for s in scores_list) == 0:
            empty = np.empty((0, 5), dtype=np.float32)
            empty_kps = (
                np.empty((0, 5, 2), dtype=np.float32) if model.use_kps else None
            )
            return empty, empty_kps

        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list) / det_scale
        if model.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        order = np.argsort(-scores.ravel(), kind="stable")
        pre_det = pre_det[order, :]
        if kpss is not None:
            kpss = kpss[order, :, :]

        keep = model.nms(pre_det)
        det = pre_det[keep, :]
        if kpss is not None:
            kpss = kpss[keep, :, :]

        max_num = self.cfg.max_faces_per_frame
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = (height // 2, width // 2)
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :, :]
        return det, kpss

    def _detect_batch(
        self, images: list[np.ndarray], input_size: tuple[int, int]
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        model = self.det_model
        blobs: list[np.ndarray] = []
        scales: list[float] = []
        for img in images:
            blob, scale = self._letterbox(img, input_size, model)
            blobs.append(blob)
            scales.append(scale)
        batch = np.ascontiguousarray(np.stack(blobs))
        session = model._session_for_input_size(input_size) if hasattr(
            model, "_session_for_input_size"
        ) else model.session
        net_outs = session.run(
            model.output_names, {model.input_name: batch}
        )
        return [
            self._decode_batch_element(net_outs, i, scales[i], input_size, model)
            for i in range(len(images))
        ]

    # ------------------------------------------------------- alignment/embed
    @staticmethod
    def _image_size_of(rec_model) -> int:
        return int(rec_model.input_size[0])

    def _align_face(
        self, img: np.ndarray, kps: np.ndarray | None, bbox: np.ndarray
    ) -> np.ndarray | None:
        import cv2

        size = self._image_size_of(self.rec_model)
        if kps is not None:
            try:
                from insightface.utils import face_align

                crop = face_align.norm_crop(
                    img, landmark=kps, image_size=size
                )
                if crop is not None and crop.size:
                    return crop
            except Exception as exc:  # noqa: BLE001
                logger.debug("norm_crop failed, using bbox crop: %s", exc)
        # Degenerate-landmark fallback: centre square crop.
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1, 8.0) * 1.1
        h, w = img.shape[:2]
        left = int(max(0, min(w - 1, cx - side / 2)))
        top = int(max(0, min(h - 1, cy - side / 2)))
        right = int(max(left + 1, min(w, cx + side / 2)))
        bottom = int(max(top + 1, min(h, cy + side / 2)))
        crop = img[top:bottom, left:right]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (size, size))

    def _embed_faces(self, crops: list[np.ndarray]) -> np.ndarray:
        """ONE batched GPU call for every aligned face (chunked)."""
        if not crops:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        rec = self.rec_model
        feats: list[np.ndarray] = []
        try:
            for start in range(0, len(crops), self.cfg.max_embedding_batch):
                chunk = crops[start : start + self.cfg.max_embedding_batch]
                out = np.asarray(rec.get_feat(chunk), dtype=np.float32)
                out = out.reshape(len(chunk), -1)
                feats.append(out)
        except Exception as exc:  # noqa: BLE001 - never lose a whole batch
            logger.warning(
                "batched get_feat failed (%s); retrying per-face", exc
            )
            feats = []
            for crop in crops:
                try:
                    out = np.asarray(rec.get_feat([crop]), dtype=np.float32)
                    feats.append(out.reshape(1, -1))
                except Exception as inner:  # noqa: BLE001
                    logger.debug("per-face embed failed: %s", inner)
        if not feats:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        mat = np.vstack(feats).astype(np.float32, copy=False)
        if mat.shape[1] != EMBEDDING_DIM:
            mat = mat[:, :EMBEDDING_DIM]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-9)

    # ------------------------------------------------------------ public API
    def infer(
        self, images: list[np.ndarray]
    ) -> tuple[list[list[dict]], Timings]:
        """Detect + embed every face in every image (batched where possible)."""
        timings = Timings(frames=len(images))
        if not images:
            return [], timings
        t_start = time.perf_counter()

        # ---- detection (batched across frames when enabled/possible)
        t0 = time.perf_counter()
        used_batch = False
        dets: list[tuple[np.ndarray, np.ndarray | None]] = []
        if self._det_batch_enabled and len(images) > 1 and self._det_input_size:
            try:
                dets = self._detect_batch(images, self._det_input_size)
                used_batch = True
            except Exception as exc:  # noqa: BLE001 - permanent fallback
                if not self._det_batch_warned:
                    logger.warning(
                        "batched detection failed at runtime, permanently "
                        "falling back to per-frame: %s",
                        exc,
                    )
                    self._det_batch_warned = True
                self._det_batch_enabled = False
                dets = []
        if not dets:
            dets = [self._detect_single(img) for img in images]
        timings.det_ms = (time.perf_counter() - t0) * 1000.0

        # ---- align all faces (CPU) across the whole batch
        t1 = time.perf_counter()
        crops: list[np.ndarray] = []
        meta: list[tuple[int, np.ndarray]] = []  # (frame_index, det_row)
        for frame_idx, (det, kpss) in enumerate(dets):
            for j in range(det.shape[0]):
                kps = kpss[j] if kpss is not None else None
                crop = self._align_face(images[frame_idx], kps, det[j])
                if crop is None:
                    continue
                crops.append(crop)
                meta.append((frame_idx, det[j]))
        timings.align_ms = (time.perf_counter() - t1) * 1000.0

        # ---- ONE batched embedding call for all faces of all frames
        t2 = time.perf_counter()
        embs = self._embed_faces(crops)
        timings.emb_ms = (time.perf_counter() - t2) * 1000.0

        results: list[list[dict]] = [[] for _ in images]
        for k, (frame_idx, row) in enumerate(meta):
            if k >= embs.shape[0]:
                break
            results[frame_idx].append(
                {
                    "bbox": [round(float(v), 2) for v in row[:4]],
                    "score": round(float(row[4]), 4),
                    "embedding": [float(v) for v in embs[k]],
                }
            )
        for faces in results:
            faces.sort(key=lambda f: f["score"], reverse=True)
            del faces[self.cfg.max_faces_per_frame :]

        timings.faces = sum(len(f) for f in results)
        timings.total_ms = (time.perf_counter() - t_start) * 1000.0
        timings.batched_detection = used_batch
        return results, timings

    def warmup(self) -> Timings:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        _, timings = self.infer([img])
        return timings

    def status(self) -> dict:
        return {
            "model": self.cfg.face_model_name,
            "providers": self._active_providers(),
            "provider": (self._active_providers() or [None])[0],
            "batched_detection": self._det_batch_enabled,
            "force_cpu": self._force_cpu or self.cfg.force_cpu,
            "det_sizes": self.cfg.det_sizes,
            "det_thresh": self.cfg.det_thresh,
            "recognition_threshold": self.cfg.recognition_threshold,
            "max_embedding_batch": self.cfg.max_embedding_batch,
        }
