#!/usr/bin/env python
"""Google Form CSV -> student enrollment database (Tasks 2, 3, 4, 13, 14).

    python scripts/setup_student_database.py --csv "C:/path/Untitled form.csv"

For every form row this script

  1. reads name + registration number + face-video link (original CSV is
     never modified),
  2. resolves the face video: a local file in ``--videos-dir`` first
     (``<reg>.mp4`` etc.), then the Google Drive link,
  3. samples frames, keeps only single-face, high-quality, non-duplicate
     frames (5 by default) and writes them to ``data/students/<REG>/``,
  4. enrolls the student through the existing enrollment service
     (same database, same encrypted 512-d ArcFace embeddings).

Students whose Drive link is permission-gated are **reported exactly**, never
invented, and the run continues with the rest.

Exit codes: 0 = finished (see report), 1 = fatal (CSV unreadable).
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import httpx  # noqa: E402

from app.config import settings  # noqa: E402
from app.formcsv import (  # noqa: E402
    FormImportError,
    FrameCandidate,
    classify_download,
    drive_direct_url,
    largest_identity_cluster,
    parse_form_csv,
    select_reference_frames,
)

DEFAULT_REPORT = Path("eval/reports/form_import_report.csv")
DEFAULT_WORK_DIR = Path("data/enrollment")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STATUS_HELP = {
    "enrolled": "student created in the database",
    "already_enrolled": "registration number already in the database",
    "invalid_row": "row could not be used (see detail)",
    "video_not_found": "no local file and no usable Drive link",
    "inaccessible": "Drive file exists but is not shared publicly",
    "not_media": "download succeeded but the body is not an image/video",
    "download_error": "network/HTTP failure while downloading",
    "no_usable_face": "video processed but no frame passed the quality gate",
    "error": "unexpected failure (see detail)",
}


# ------------------------------------------------------------------- helpers
def _norm_stem(path: Path) -> str:
    return re.sub(r"[^a-z0-9]", "", path.stem.lower())


def find_local_video(videos_dir: Path | None, reg: str) -> Path | None:
    """Look for ``<reg>.*`` (or any file containing the reg) in --videos-dir."""
    if videos_dir is None:
        return None
    if not videos_dir.exists():
        return None
    wanted = _norm_stem(Path(reg))
    exact: Path | None = None
    loose: list[Path] = []
    for p in sorted(videos_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS | IMAGE_EXTS:
            continue
        stem = _norm_stem(p)
        if stem == wanted:
            exact = p
        elif wanted and wanted in stem:
            loose.append(p)
    return exact or (loose[0] if len(loose) == 1 else None)


def _sniff_ext(content: bytes, content_type: str) -> str:
    head = content[:64]
    if head[4:8] == b"ftyp":
        return ".mp4"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return ".avi"
    if head[:4] == b"\x1aE\xdf\xa3":
        return ".mkv"
    if head[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    ct = (content_type or "").split(";")[0].strip().lower()
    return {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/webm": ".webm",
        "image/jpeg": ".jpg",
        "image/png": ".png",
    }.get(ct, ".bin")


def download_drive(ref: str, dest: Path, timeout: float = 60.0) -> tuple[str, str | None]:
    """Download a Drive link.  Returns ``(status, detail)``.

    ``downloaded`` | ``inaccessible`` (permission/sign-in wall) |
    ``not_media`` | ``download_error``.
    """
    url = drive_direct_url(ref)
    if url is None:
        return "video_not_found", "not a Google Drive link and no local file"
    headers = {"User-Agent": "attendance-enroll/1.0"}
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
            resp = client.get(url)
            if resp.status_code in (401, 403):
                return "inaccessible", f"HTTP {resp.status_code} (access denied)"
            if resp.status_code != 200:
                return "download_error", f"HTTP {resp.status_code}"

            ct = resp.headers.get("content-type", "")
            body = resp.content
            kind = classify_download(ct, body[:512])
            if kind == "html":
                # Large files / gated files come back as an HTML interstitial.
                confirm_url = _drive_confirm_url(body.decode("utf-8", "replace"))
                if confirm_url:
                    resp2 = client.get(confirm_url)
                    ct2 = resp2.headers.get("content-type", "")
                    if classify_download(ct2, resp2.content[:512]) in ("video", "image"):
                        body, ct, kind = resp2.content, ct2, "video"
                    else:
                        return "inaccessible", "Drive asked for sign-in after confirm"
                else:
                    return "inaccessible", "Google sign-in / permission page"
            if kind not in ("video", "image"):
                return "not_media", f"content-type {ct!r}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            return "downloaded", None
    except httpx.HTTPError as exc:
        return "download_error", f"{type(exc).__name__}: {exc}"


def _drive_confirm_url(html: str) -> str | None:
    """Build the alternate download URL from Drive's interstitial form."""
    if "download-form" not in html:
        return None
    action = re.search(r'action="([^"]+)"', html)
    if not action:
        return None
    from urllib.parse import urlencode

    inputs = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html))
    base = action.group(1).replace("&amp;", "&")
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}{urlencode(inputs)}" if inputs else base


def backup_orphan_photo_dirs(db) -> list[str]:  # noqa: ANN001
    """Preserve pre-existing photo folders no student row references.

    ``enroll_student()`` writes to ``data/students/<student_id>/`` and wipes
    what is there, and the placeholder gallery used ``1..6`` - so anything
    unreferenced is moved aside instead of being overwritten.
    """
    from app.models import Student

    root = settings.photos_root / "students"
    if not root.exists():
        return []
    referenced: set[str] = set()
    for student in db.query(Student).all():
        for rel in student.photo_paths or []:
            referenced.add(Path(rel).parent.name)

    moved: list[str] = []
    stash = root / "_backup_pre_import"
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.isdigit() or d.name in referenced:
            continue
        # only canonical enrollment folders (photo_N.jpg): this never touches
        # the per-registration frame folders this script itself creates
        # (data/students/<REG>/frame_001.jpg) or the backup stash.
        if len(d.name) > 9 or not any(d.glob("photo_*")):
            continue
        if not any(d.iterdir()):
            continue
        stash.mkdir(parents=True, exist_ok=True)
        target = stash / d.name
        if target.exists():
            target = stash / f"{d.name}_{int(time.time())}"
        shutil.move(str(d), str(target))
        moved.append(f"{d.name} -> {target.relative_to(root).as_posix()}")
    return moved


# --------------------------------------------------------------- frame picking
def extract_reference_frames(
    video_path: Path,
    *,
    infer_frames,
    sample_fps: float = 2.0,
    want: int = 5,
    min_det_score: float,
    min_width: float,
    min_area: float,
    min_sharpness: float,
    brightness_range: tuple[float, float],
    duplicate_sim: float,
    identity_sim: float = 0.40,
    batch: int = 8,
) -> tuple[list[FrameCandidate], dict]:
    """Sample ``video_path`` and return quality-gated single-face frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(fps / max(sample_fps, 0.1))))

    stats = {
        "frames_read": 0,
        "frames_sampled": 0,
        "rejected_multiple_faces": 0,
        "rejected_no_face": 0,
        "rejected_small": 0,
        "rejected_low_det": 0,
        "rejected_quality": 0,
        "candidates": 0,
        "identity_clusters": 0,
        "identity_kept": 0,
        "identity_dropped_other": 0,
        "identity_dropped_no_embedding": 0,
    }
    lo, hi = brightness_range
    candidates: list[FrameCandidate] = []
    pending: list[tuple[int, object]] = []

    def _flush() -> None:
        if not pending:
            return
        jpegs = [_encode_jpeg(img) for _, img in pending]
        results = infer_frames(jpegs)
        for (frame_idx, frame), jpeg, faces in zip(pending, jpegs, results):
            stats["frames_sampled"] += 1
            if not faces:
                stats["rejected_no_face"] += 1
                continue
            if len(faces) > 1:
                stats["rejected_multiple_faces"] += 1
                continue
            face = faces[0]
            bbox = [float(v) for v in face.get("bbox", [0, 0, 0, 0])]
            score = float(face.get("score", 0.0))
            x1, y1, x2, y2 = bbox
            width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
            area = width * height
            if score < min_det_score:
                stats["rejected_low_det"] += 1
                continue
            if width < min_width or area < min_area:
                stats["rejected_small"] += 1
                continue
            crop = frame[max(0, int(y1)) : max(0, int(y2)), max(0, int(x1)) : max(0, int(x2))]
            if crop.size == 0:
                stats["rejected_small"] += 1
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())
            if sharpness < min_sharpness or not (lo <= brightness <= hi):
                stats["rejected_quality"] += 1
                continue
            stats["candidates"] += 1
            candidates.append(
                FrameCandidate(
                    index=frame_idx,
                    jpeg=jpeg,
                    embedding=[float(v) for v in face.get("embedding", [])] or None,
                    det_score=score,
                    sharpness=sharpness,
                    brightness=brightness,
                    face_width=width,
                    face_area=area,
                )
            )
        pending.clear()

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stats["frames_read"] += 1
        if idx % stride == 0:
            pending.append((idx, frame))
            if len(pending) >= batch:
                _flush()
        idx += 1
    _flush()
    cap.release()

    # identity gate: a video with more than one person must contribute frames
    # of one person only - otherwise a stranger's embedding enters the gallery
    # and later accepts them as this student.
    clustered, cluster_counts = largest_identity_cluster(
        candidates, identity_sim=identity_sim
    )
    stats["identity_clusters"] = cluster_counts["clusters"]
    stats["identity_kept"] = cluster_counts["kept"]
    stats["identity_dropped_other"] = cluster_counts["dropped_other_identity"]
    stats["identity_dropped_no_embedding"] = cluster_counts["dropped_no_embedding"]

    selected, counts = select_reference_frames(
        clustered,
        want,
        duplicate_sim=duplicate_sim,
        min_sharpness=0.0,  # already applied above (per-frame, with stats)
        brightness_range=(0.0, 255.0),
        min_det_score=0.0,
    )
    stats.update({f"select_{k}": v for k, v in counts.items()})
    stats["selected"] = len(selected)
    return selected, stats


def _encode_jpeg(frame) -> bytes:  # noqa: ANN001
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("failed to encode frame")
    return buf.tobytes()


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Import a Google Forms CSV into the student face database"
    )
    ap.add_argument("--csv", required=True, type=Path, help="Google Forms response CSV")
    ap.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Folder with locally downloaded face videos named <reg>.* "
        "(used before the Drive link - lets you bypass Drive sharing)",
    )
    ap.add_argument("--frames", type=int, default=None, help="reference frames per student")
    ap.add_argument("--sample-fps", type=float, default=2.0, help="video sampling rate")
    ap.add_argument("--min-sharpness", type=float, default=40.0, help="Laplacian variance floor")
    ap.add_argument("--min-det-score", type=float, default=None)
    ap.add_argument("--min-width", type=float, default=None)
    ap.add_argument("--min-area", type=float, default=None)
    ap.add_argument(
        "--frame-dedup", type=float, default=None,
        help="cosine above which two frames count as duplicates "
        f"(default {settings.duplicate_identity_sim}); lower it if a static "
        "video leaves you with too few reference frames",
    )
    ap.add_argument("--only", type=str, default=None, help="import just this registration number")
    ap.add_argument(
        "--identity-sim", type=float, default=0.40,
        help="minimum cosine for a frame to count as the same person as the "
        "majority of the video (frames of anyone else are discarded)",
    )
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--dry-run", action="store_true", help="parse + resolve, do not enroll")
    ap.add_argument(
        "--rpc",
        action="store_true",
        help="use the shared Redis inference service instead of loading the model here",
    )
    args = ap.parse_args()

    want = args.frames or settings.max_reference_photos
    min_det = args.min_det_score if args.min_det_score is not None else settings.face_min_det_score
    min_width = args.min_width if args.min_width is not None else settings.face_min_width
    min_area = args.min_area if args.min_area is not None else settings.face_min_area

    print("=" * 78)
    print("Google Form -> student database")
    print(f"  csv      : {args.csv}")
    print(f"  videos   : {args.videos_dir or '(Drive links only)'}")
    print(f"  frames   : {want} per student, sampled at {args.sample_fps} fps")
    print(f"  gallery  : threshold={settings.recognition_threshold} "
          f"margin={settings.recognition_margin} "
          f"dup_sim={settings.duplicate_identity_sim}")
    print("=" * 78)

    try:
        rows = parse_form_csv(args.csv)
    except FormImportError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    if args.only:
        rows = [r for r in rows if r.registration_no == args.only]
        if not rows:
            print(f"FATAL: --only {args.only} not in the CSV", file=sys.stderr)
            return 1
    print(f"rows read: {len(rows)}")

    # ---- inference: in-process FaceEngine unless --rpc ---------------------
    infer_frames = None
    local = None
    if not args.dry_run:
        if args.rpc:
            from app.inference import client as inference_client

            infer_frames = lambda jpegs: inference_client.infer_frames(jpegs)["results"]  # noqa: E731
        else:
            from app.inference.local import LocalInference

            # patch app.inference.client so enrollment's quality gate and the
            # embedding call both use this in-process FaceEngine (no Redis)
            local = LocalInference().install()
            infer_frames = lambda jpegs: local.infer_frames(jpegs)["results"]  # noqa: E731

    # ---- database ---------------------------------------------------------
    db = None
    if not args.dry_run:
        from app.db import Base, SessionLocal, engine as db_engine
        from app.services.enrollment import EnrollmentError, enroll_student

        Base.metadata.create_all(db_engine)
        db = SessionLocal()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict] = []
    enrolled = 0
    backed_up = False  # placeholder photo folders are only moved once we are
    # about to write (a run that enrolls nobody must not touch existing files)

    for row in rows:
        tag = f"row {row.row_number:02d} reg ...{row.short_reg}"
        if row.errors:
            _record(report_rows, row, status="invalid_row", detail="; ".join(row.errors))
            print(f"[INVALID ] {tag}: {'; '.join(row.errors)}")
            continue
        if db is None:
            _record(report_rows, row, status="dry_run", detail="parsed OK")
            print(f"[DRYRUN  ] {tag}: {row.name} - parsed OK")
            continue

        # 1. already enrolled?
        from app.models import Student
        from sqlalchemy import select as _select

        existing = db.scalars(
            _select(Student).where(Student.registration_no == row.registration_no)
        ).first()
        if existing is not None:
            _record(
                report_rows, row, status="already_enrolled",
                detail=f"student_id={existing.id}", student_id=existing.id,
            )
            print(f"[SKIP    ] {tag}: already enrolled (student_id={existing.id})")
            continue

        # 2. resolve the face video: local file, then Drive
        video_path, status, detail = _resolve_video(row, args)
        if status != "ok":
            _record(report_rows, row, status=status, detail=detail or "")
            print(f"[{status.upper()[:8]:8}] {tag}: {detail}")
            continue

        # 3. frames
        try:
            frames, stats = extract_reference_frames(
                video_path,
                infer_frames=infer_frames,
                sample_fps=args.sample_fps,
                want=want,
                min_det_score=min_det,
                min_width=min_width,
                min_area=min_area,
                min_sharpness=args.min_sharpness,
                brightness_range=(30.0, 225.0),
                duplicate_sim=(
                    args.frame_dedup
                    if args.frame_dedup is not None
                    else settings.duplicate_identity_sim
                ),
                identity_sim=args.identity_sim,
            )
        except Exception as exc:  # noqa: BLE001 - report, never silently skip
            _record(report_rows, row, status="error", detail=f"frame extraction: {exc}")
            print(f"[ERROR   ] {tag}: frame extraction failed: {exc}")
            continue

        if len(frames) < settings.min_reference_photos:
            detail = (
                f"only {len(frames)} usable frame(s) (need "
                f"{settings.min_reference_photos}); sampled={stats['frames_sampled']} "
                f"no_face={stats['rejected_no_face']} "
                f"multi_face={stats['rejected_multiple_faces']} "
                f"small={stats['rejected_small']} "
                f"low_det={stats['rejected_low_det']} "
                f"quality={stats['rejected_quality']} "
                f"identities={stats['identity_clusters']} "
                f"other_identity={stats['identity_dropped_other']} "
                f"near_dup={stats['select_rejected_near_duplicate']}"
            )
            _record(report_rows, row, status="no_usable_face", detail=detail, stats=stats)
            print(f"[NO FACE ] {tag}: {detail}")
            continue

        if stats["identity_dropped_other"] or stats["identity_clusters"] > 1:
            print(
                f"  identity gate: {stats['identity_clusters']} face identity(ies) in "
                f"video, kept the main one - "
                f"discarded {stats['identity_dropped_other']} frame(s) of other people"
            )

        # 4. store the selected frames under data/students/<REG>/ (Task 3)
        folder = settings.photos_root / "students" / row.registration_no
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("frame_*"):
            old.unlink()
        filenames = []
        for i, cand in enumerate(frames, start=1):
            name = f"frame_{i:03d}.jpg"
            (folder / name).write_bytes(cand.jpeg)
            filenames.append(name)

        # 5. enroll through the existing enrollment service (Task 4)
        if not backed_up:
            for note in backup_orphan_photo_dirs(db):
                print(f"  preserved existing photo folder: {note}")
            backed_up = True
        try:
            student = enroll_student(
                db,
                name=row.name,
                registration_no=row.registration_no,
                section=None,
                photos=[c.jpeg for c in frames],
                filenames=filenames,
            )
        except EnrollmentError as exc:
            status = "already_enrolled" if exc.status_code == 409 else "error"
            _record(report_rows, row, status=status, detail=exc.message, stats=stats)
            print(f"[{status.upper()[:8]:8}] {tag}: {exc.message}")
            continue
        except Exception as exc:  # noqa: BLE001
            _record(report_rows, row, status="error", detail=str(exc), stats=stats)
            print(f"[ERROR   ] {tag}: {exc}")
            continue

        enrolled += 1
        _record(
            report_rows, row, status="enrolled",
            detail=f"{len(frames)} frames -> {folder.as_posix()}",
            student_id=student.id, stats=stats,
        )
        print(f"[ENROLLED] {tag}: {row.name} -> student_id={student.id}, "
              f"{len(frames)} reference frames")

    if db is not None:
        db.close()

    _write_report(args.report, report_rows)

    # ---- summary ----------------------------------------------------------
    by_status: dict[str, int] = {}
    for r in report_rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print("\n" + "=" * 78)
    print("IMPORT SUMMARY")
    for status, count in sorted(by_status.items()):
        note = STATUS_HELP.get(status, "")
        print(f"  {status:18} {count:3d}   {note}")
    blocked = [r for r in report_rows if r["status"] == "inaccessible"]
    if blocked:
        print(f"\n  {len(blocked)} file(s) are NOT publicly shared.  In Drive open")
        print("  the file -> Share -> General access -> 'Anyone with the link' -> Viewer,")
        print("  or download it into --videos-dir as <reg>.mp4 and re-run.")
        for r in blocked:
            print(f"    reg ...{str(r['registration_no'])[-3:]}: {r['detail']}")
    print(f"\nreport: {args.report}")
    if not args.dry_run:
        print(f"frames: data/students/<REG>/   database: {settings.data_dir}/attendance.db")
    print(f"enrolled this run: {enrolled}")
    return 0


def _resolve_video(row, args) -> tuple[Path | None, str, str | None]:  # noqa: ANN001
    """Local file first, then the Drive link."""
    # a) the CSV cell itself may hold a local path
    ref = (row.video_ref or "").strip()
    if ref and Path(ref).expanduser().exists():
        p = Path(ref).expanduser()
        if p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS:
            return p, "ok", f"local file {p.name}"

    # b) --videos-dir/<reg>.<ext>
    local = find_local_video(args.videos_dir, row.registration_no)
    if local is not None:
        return local, "ok", f"local file {local.name}"

    # c) Google Drive
    if ref.startswith("http") and "drive.google.com" in ref:
        dest = args.work_dir / f"{row.registration_no}.bin"
        status, detail = download_drive(ref, dest)
        if status == "downloaded":
            ext = _sniff_ext(dest.read_bytes()[:64], "")
            final = dest.with_suffix(ext)
            if final != dest:
                dest.rename(final)
            return final, "ok", f"downloaded from Drive ({final.name})"
        return None, status, detail

    if not ref:
        return None, "video_not_found", "empty link and no local file"
    return None, "video_not_found", (
        f"no local file for reg ...{row.short_reg}"
        + (f" in {args.videos_dir}" if args.videos_dir else " (pass --videos-dir)")
        + " and the link is not a Google Drive URL"
    )


def _record(
    report_rows: list[dict],
    row,  # noqa: ANN001
    *,
    status: str,
    detail: str,
    student_id: int | None = None,
    stats: dict | None = None,
) -> None:
    report_rows.append(
        {
            "row_number": row.row_number,
            "name": row.name,
            "registration_no": row.registration_no,
            "status": status,
            "detail": detail,
            "student_id": student_id or "",
            "selected_frames": (stats or {}).get("selected", ""),
            "frames_sampled": (stats or {}).get("frames_sampled", ""),
            "rejected_multi_face": (stats or {}).get("rejected_multiple_faces", ""),
            "rejected_no_face": (stats or {}).get("rejected_no_face", ""),
            "rejected_quality": (stats or {}).get("rejected_quality", ""),
        }
    )


def _write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_number", "name", "registration_no", "status", "detail",
        "student_id", "selected_frames", "frames_sampled",
        "rejected_multi_face", "rejected_no_face", "rejected_quality",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
