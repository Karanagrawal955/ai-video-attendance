"""Seed sample data: a mock camera from a local video + optional enrollment.

Run it inside the stack (recommended, inference service must be up for
enrollment)::

    # register a mock camera that loops a local video file forever
    docker compose exec api python scripts/seed.py \
        --camera --name "Main Gate" --type entry \
        --video /samples/videos/lecture.mp4 --location "Main entrance"

    # enroll one student from a folder of 3-5 photos
    docker compose exec api python scripts/seed.py \
        --enroll /samples/faces/21CSE001 --name "Ada Lovelace" \
        --registration-no 21CSE001 --section CSE-A

    # enroll every subfolder: samples/faces/<registration_no>/*.jpg
    docker compose exec api python scripts/seed.py --enroll-dir /samples/faces

    # show what exists
    docker compose exec api python scripts/seed.py --list

Notes:
* The video file is treated as an endless RTSP substitute: the frame reader
  loops it from the start when it reaches EOF, so you can exercise
  entry/exit logic without CCTV hardware.
* After seeding, start the stream:
  POST /cameras/{id}/start
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.logging_config import configure  # noqa: E402
from app.models import AttendanceSession, Camera, RecognitionLog, Student  # noqa: E402
from app.services import enrollment  # noqa: E402

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _register_camera(db, args: argparse.Namespace) -> Camera:
    rtsp = args.rtsp
    video = args.video if args.video is not None else settings.seed_video_path
    if rtsp:
        video = None
    if not rtsp and not video:
        raise SystemExit("provide --rtsp URL or --video PATH")
    if args.name is None:
        args.name = "Mock Camera"

    existing = db.scalars(select(Camera).where(Camera.name == args.name)).first()
    if existing is not None:
        print(f"camera {args.name!r} already exists (id={existing.id})")
        return existing

    source = rtsp or str(video)
    if not rtsp and not Path(source).exists():
        print(
            f"WARNING: video file {source!r} does not exist yet - the camera "
            "will retry until you place the file there."
        )

    camera = Camera(
        name=args.name,
        location=args.location,
        rtsp_url=rtsp,
        file_path=None if rtsp else source,
        type=args.type,
        sampling_rate=args.sampling_rate,
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)
    print(
        f"registered camera id={camera.id} name={camera.name!r} type={camera.type} "
        f"source={source} sampling_rate={camera.sampling_rate}"
    )
    return camera


def _photos_in(folder: Path) -> list[tuple[bytes, str]]:
    out = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in PHOTO_EXTS:
            out.append((path.read_bytes(), path.name))
    return out


def _enroll_one(db, *, folder: Path, name: str, reg_no: str, section: str | None) -> None:
    photos = _photos_in(folder)
    if not photos:
        print(f"SKIP {folder}: no photo files found ({sorted(PHOTO_EXTS)})")
        return
    try:
        student = enrollment.enroll_student(
            db,
            name=name,
            registration_no=reg_no,
            section=section,
            photos=[p[0] for p in photos],
            filenames=[p[1] for p in photos],
        )
        print(
            f"enrolled student id={student.id} reg={student.registration_no!r} "
            f"photos={len(photos)} embeddings={len(student.embeddings)}"
        )
    except enrollment.EnrollmentError as exc:
        print(f"SKIP {reg_no}: {exc}")
    except enrollment.InferenceUnavailable as exc:
        raise SystemExit(
            f"inference service unavailable: {exc}\n"
            "Start the stack first: docker compose up -d"
        ) from exc


def _list(db) -> None:
    students = db.scalars(select(Student).order_by(Student.id)).all()
    cameras = db.scalars(select(Camera).order_by(Camera.id)).all()
    sessions = db.scalars(
        select(AttendanceSession).order_by(AttendanceSession.id.desc()).limit(10)
    ).all()
    logs = db.scalars(
        select(RecognitionLog).order_by(RecognitionLog.id.desc()).limit(5)
    ).all()

    print(f"\nstudents ({len(students)}):")
    for s in students:
        print(
            f"  #{s.id} {s.registration_no} {s.name} section={s.section} "
            f"embeddings={len(s.embeddings or [])}"
        )
    print(f"\ncameras ({len(cameras)}):")
    for c in cameras:
        src = c.rtsp_url or c.file_path
        print(f"  #{c.id} {c.name!r} type={c.type} rate={c.sampling_rate} src={src}")
    print(f"\nrecent sessions ({len(sessions)}):")
    for s in sessions:
        print(
            f"  #{s.id} student={s.student_id} date={s.date} {s.status} "
            f"in={s.entry_time} out={s.exit_time} dur={s.total_duration}"
        )
    print(f"\nrecent recognition logs ({len(logs)}):")
    for log in logs:
        print(
            f"  #{log.id} student={log.student_id} camera={log.camera_id} "
            f"ts={log.timestamp} conf={log.confidence_score} "
            f"gpu_ms={log.gpu_inference_time_ms}"
        )
    print()


def main() -> int:
    configure(settings)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--camera", action="store_true", help="register a camera")
    parser.add_argument("--name", default=None,
                        help="camera name (with --camera) or student name "
                             "(with --enroll)")
    parser.add_argument("--type", choices=["entry", "exit", "both"], default="entry")
    parser.add_argument("--location", default=None)
    parser.add_argument("--rtsp", default=None, help="RTSP URL (overrides --video)")
    parser.add_argument("--video", default=None, help="local video file path")
    parser.add_argument("--sampling-rate", type=int,
                        default=settings.default_sampling_rate)
    parser.add_argument("--enroll", default=None,
                        help="folder with 3-5 reference photos of ONE student")
    parser.add_argument("--enroll-dir", default=None,
                        help="folder of subfolders named <registration_no>")
    parser.add_argument("--registration-no", default=None)
    parser.add_argument("--section", default=None)
    parser.add_argument("--list", action="store_true", help="print current data")
    args = parser.parse_args()

    # `--name` doubles as camera name / student name depending on the mode;
    # fall back to sensible defaults per mode.
    if args.camera and args.name is None:
        args.name = "Mock Camera"
    reg_no = args.registration_no
    student_name = args.name or reg_no

    db = SessionLocal()
    try:
        if args.camera:
            _register_camera(db, args)
        if args.enroll_dir:
            root = Path(args.enroll_dir)
            if not root.is_dir():
                raise SystemExit(f"{root} is not a directory")
            for sub in sorted(p for p in root.iterdir() if p.is_dir()):
                _enroll_one(
                    db,
                    folder=sub,
                    name=sub.name,
                    reg_no=sub.name,
                    section=args.section,
                )
        if args.enroll:
            folder = Path(args.enroll)
            if not folder.is_dir():
                raise SystemExit(f"{folder} is not a directory")
            if not reg_no:
                raise SystemExit("--enroll requires --registration-no")
            _enroll_one(
                db,
                folder=folder,
                name=student_name,
                reg_no=reg_no,
                section=args.section,
            )
        if args.list or not (args.camera or args.enroll or args.enroll_dir):
            _list(db)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
