"""CPU-side frame reader thread.

One thread per camera:

* opens the RTSP/file source with FFmpeg (best-effort hardware decode),
* decodes frames on a CPU core,
* samples every ``sampling_rate`` frames (configurable per camera),
* pushes packets into a bounded queue (drop-oldest so latency never grows),
* reconnects automatically with exponential backoff when an RTSP stream dies.

File sources loop forever - that is the "mock CCTV camera" for testing.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("app.pipeline.reader")


@dataclass
class FramePacket:
    seq: int
    ts: float  # wall clock when the frame was decoded (UTC epoch seconds)
    image: np.ndarray


def _is_rtsp(source: str) -> bool:
    return source.lower().startswith(
        ("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://")
    )


class FrameReader(threading.Thread):
    def __init__(
        self,
        *,
        camera_id: int,
        source: str,
        sampling_rate: int,
        out_q: "queue.Queue[FramePacket]",
        stop_event: threading.Event,
        rtsp_transport: str = "tcp",
        open_timeout_ms: int = 10_000,
        read_timeout_ms: int = 10_000,
        reconnect_initial_delay_s: float = 2.0,
        reconnect_max_delay_s: float = 30.0,
    ) -> None:
        super().__init__(name=f"frame-reader-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.sampling_rate = max(1, int(sampling_rate))
        self.out_q = out_q
        self.stop_event = stop_event
        self.rtsp_transport = rtsp_transport
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.reconnect_initial_delay_s = reconnect_initial_delay_s
        self.reconnect_max_delay_s = reconnect_max_delay_s
        self.is_file = not _is_rtsp(source)
        self.frames_read = 0
        self.frames_sampled = 0
        self.dropped = 0
        self.reconnects = 0

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        import cv2

        # Must be set before the first FFmpeg-backed capture is opened.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;{self.rtsp_transport}|max_delay;500000",
        )
        delay = self.reconnect_initial_delay_s
        logger.info(
            "reader started",
            extra={
                "camera_id": self.camera_id,
                "source": self.source[:120],
                "sampling_rate": self.sampling_rate,
                "kind": "file" if self.is_file else "stream",
            },
        )

        while not self.stop_event.is_set():
            cap, opened_at = self._open(cv2)
            if cap is None:
                logger.warning(
                    "failed to open source, retrying in %.1fs",
                    delay,
                    extra={"camera_id": self.camera_id},
                )
                self.stop_event.wait(delay)
                delay = min(delay * 2, self.reconnect_max_delay_s)
                continue

            delay = self.reconnect_initial_delay_s
            if self.reconnects:
                logger.info(
                    "stream reconnected",
                    extra={"camera_id": self.camera_id},
                )
            fail_streak = 0
            index = 0
            frames_this_pass = 0

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    fail_streak += 1
                    if self.is_file:
                        # Healthy file reached EOF -> loop it immediately so
                        # the mock camera runs continuously.
                        if frames_this_pass > 5:
                            break
                        # File opens but yields nothing -> treat as failure.
                        self.stop_event.wait(delay)
                        delay = min(delay * 2, self.reconnect_max_delay_s)
                        break
                    if fail_streak >= 5:
                        logger.warning(
                            "stream read failed %d times - reconnecting",
                            fail_streak,
                            extra={"camera_id": self.camera_id},
                        )
                        self.reconnects += 1
                        break
                    continue

                fail_streak = 0
                frames_this_pass += 1
                self.frames_read += 1
                index += 1
                if (index - 1) % self.sampling_rate != 0:
                    continue  # CPU-side sampling before any GPU work
                self.frames_sampled += 1
                self._push(
                    FramePacket(seq=self.frames_sampled, ts=time.time(), image=frame)
                )

            cap.release()
            if self.is_file and frames_this_pass > 5:
                # Looping a local file needs no backoff.
                continue

        logger.info(
            "reader stopped",
            extra={
                "camera_id": self.camera_id,
                "frames_read": self.frames_read,
                "frames_sampled": self.frames_sampled,
                "dropped": self.dropped,
                "reconnects": self.reconnects,
            },
        )

    # ----------------------------------------------------------------- open
    def _open(self, cv2):  # noqa: ANN001
        cap = cv2.VideoCapture()
        try:
            if not self.is_file:
                # Two-step open lets us set FFmpeg timeouts first.
                for prop, value in (
                    ("CAP_PROP_OPEN_TIMEOUT_MSEC", self.open_timeout_ms),
                    ("CAP_PROP_READ_TIMEOUT_MSEC", self.read_timeout_ms),
                ):
                    p = getattr(cv2, prop, None)
                    if p is not None:
                        cap.set(p, value)
                # Best-effort NVDEC; silently ignored if this FFmpeg build
                # lacks hardware acceleration (CPU decode still works).
                hw = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
                any_hw = getattr(cv2, "VIDEO_ACCELERATION_ANY", None)
                if hw is not None and any_hw is not None:
                    cap.set(hw, any_hw)
                buf = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
                if buf is not None:
                    cap.set(buf, 2)
            opened = cap.open(self.source, cv2.CAP_FFMPEG)
            if not opened:
                cap.release()
                return None, time.time()
            return cap, time.time()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "VideoCapture open raised: %s", extra={"camera_id": self.camera_id}
            )
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
            return None, time.time()

    # ----------------------------------------------------------------- push
    def _push(self, packet: FramePacket) -> None:
        while not self.stop_event.is_set():
            try:
                self.out_q.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self.out_q.get_nowait()  # drop-oldest: keep latency low
                    self.dropped += 1
                    if self.dropped % 200 == 1:
                        logger.info(
                            "frame queue full - dropping oldest frames",
                            extra={
                                "camera_id": self.camera_id,
                                "dropped_total": self.dropped,
                            },
                        )
                except queue.Empty:
                    pass
