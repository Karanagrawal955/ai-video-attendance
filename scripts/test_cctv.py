#!/usr/bin/env python
"""CCTV / classroom video -> attendance + identity-level evaluation
(Tasks 5, 6, 7, 8, 9, 10, 12, 13, 16).

    python scripts/test_cctv.py --video "path/classroom.mp4"   # (see --help)

Pipeline per sampled frame:

    SCRFD detect -> quality gate -> ArcFace 512-d embed -> EmbeddingIndex
    (threshold + margin + duplicate-identity rules) -> IoU tracker ->
    K-of-N confirmation (default 3 of the last 5 within 10 s) ->
    ledger (one attendance mark per student) -> process_recognition()

Anything below threshold stays UNKNOWN and is never marked; a single frame
can never create an attendance row.

Outputs (default ``eval/reports/<video>/``):
    attendance.csv   name, registration no., status, timestamp, match
    roster.csv       Present/Absent against ``--expected``
    observations.csv every face observation with CORRECT/WRONG/... labels
    metrics.json     accuracy, precision, recall, F1, FAR, FRR, ...
    report.html      human-readable summary
    frames/*.jpg     annotated frames
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.config import settings  # noqa: E402
from app.confirm import (  # noqa: E402
    AttendanceLedger,
    BoxTracker,
    IdentityConfirmator,
    UNKNOWN,
)
from app.evalmetrics import classify_pair, identity_metrics  # noqa: E402
from app.formcsv import FormImportError, parse_form_csv  # noqa: E402


# ------------------------------------------------------------------- helpers
def mask_reg(reg: str) -> str:
    return "..." + reg[-3:] if len(reg) >= 3 else "***"


def _load_expected(path: Path | None) -> list[dict]:
    if path is None:
        return []
    rows = parse_form_csv(path, require_link=False, check_duplicates=False)
    return [
        {"name": r.name, "registration_no": r.registration_no}
        for r in rows
        if r.registration_no
    ]


def annotate(frame, faces, tracks, labels, clip_t: float):  # noqa: ANN001
    """Draw boxes + name/reg/match/status on a frame (Task 16)."""
    for face, tid in zip(faces, tracks):
        x1, y1, x2, y2 = [int(v) for v in face["bbox"][:4]]
        info = labels.get(tid)
        if info is None:
            text, colour = "UNKNOWN", (0, 0, 255)
        elif info["reg"] == UNKNOWN:
            text, colour = "UNKNOWN - NOT MARKED", (0, 0, 255)
        else:
            text = f"{info['name']} | {mask_reg(info['reg'])} | {info['score']:.2f} | {info['status']}"
            colour = (0, 200, 0) if info["status"] == "PRESENT" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            frame, text, (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA,
        )
    cv2.putText(
        frame, f"t={clip_t:05.1f}s", (10, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA,
    )
    return frame


def write_html_report(out_dir: Path, ctx: dict) -> Path:
    """Human-readable report (Task 10/16)."""

    def _rows(items: list[dict], cols: list[str]) -> str:
        return "".join(
            "<tr>" + "".join(f"<td>{_esc(r.get(c, ''))}</td>" for c in cols) + "</tr>"
            for r in items
        )

    metrics = ctx.get("metrics") or {}
    metric_rows = "".join(
        f"<tr><td>{k}</td><td>{(v * 100 if isinstance(v, float) and v <= 1 else v):.4f}"
        f"</td></tr>"
        for k, v in metrics.items()
        if isinstance(v, (int, float))
    ) or "<tr><td colspan=2>no ground truth supplied - metrics not computed</td></tr>"

    attendance_cols = ["name", "registration_no", "status", "timestamp", "match", "frames"]
    roster_cols = ["name", "registration_no", "status", "timestamp", "match"]
    obs_cols = ["frame", "time_s", "expected", "predicted", "score", "result"]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Attendance report</title>
<style>
 body{{font-family:Arial,sans-serif;margin:32px;color:#222}}
 table{{border-collapse:collapse;width:100%;margin-bottom:26px}}
 th,td{{border:1px solid #ddd;padding:8px 10px;text-align:left;font-size:14px}}
 th{{background:#2e7d32;color:#fff}}
 tr:nth-child(even){{background:#f6f6f6}}
 .ok{{color:#2e7d32;font-weight:bold}} .bad{{color:#c62828;font-weight:bold}}
 .banner{{background:#fff8e1;border:1px solid #ffe082;padding:12px 16px;border-radius:6px}}
 h2{{margin-top:28px}}
</style></head><body>
<h1>Attendance report</h1>
<p class="banner"><b>Clip:</b> {_esc(ctx['clip_label'])}<br>
<b>Gallery:</b> {ctx['students']} students / {ctx['embeddings']} embeddings -
threshold {settings.recognition_threshold}, margin {settings.recognition_margin},
confirmation {ctx['confirm_k']} of last {ctx['confirm_n']} within {ctx['confirm_window']} s<br>
<b>Note:</b> {_esc(ctx['note'])}</p>

<h2>Summary</h2>
<table><tr><th>metric</th><th>value</th></tr>
<tr><td>frames decoded</td><td>{ctx['frames_decoded']}</td></tr>
<tr><td>frames sampled</td><td>{ctx['frames_sampled']}</td></tr>
<tr><td>faces evaluated</td><td>{ctx['faces']}</td></tr>
<tr><td>marked PRESENT</td><td class="ok">{len(ctx['attendance'])}</td></tr>
<tr><td>unknown faces (not marked)</td><td class="bad">{ctx['unknown_faces']}</td></tr>
<tr><td>tracks seen</td><td>{ctx['tracks']}</td></tr>
<tr><td>ground-truth frames skipped</td><td>{ctx.get('gt_frames_skipped', 0)}</td></tr>
</table>

<h2>Attendance ({len(ctx['attendance'])})</h2>
<table><tr>{''.join(f'<th>{c}</th>' for c in attendance_cols)}</tr>
{_rows(ctx['attendance'], attendance_cols)}</table>
{'<h2>Roster: Present / Absent</h2><table><tr>' + ''.join(f'<th>{c}</th>' for c in roster_cols) + '</tr>' + _rows(ctx['roster'], roster_cols) + '</table>' if ctx['roster'] else ''}

<h2>Evaluation metrics</h2>
<table><tr><th>metric</th><th>value</th></tr>{metric_rows}</table>

<h2>Observations ({len(ctx['observations'])})</h2>
<table><tr>{''.join(f'<th>{c}</th>' for c in obs_cols)}</tr>
{_rows(ctx['observations'][:500], obs_cols)}</table>
<p>{'showing first 500 rows - full file: observations.csv' if len(ctx['observations']) > 500 else ''}</p>
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def _esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recognise enrolled students in a video and mark attendance"
    )
    ap.add_argument("--video", required=True, type=Path, help="MP4/classroom video")
    ap.add_argument(
        "--expected", type=Path, default=None,
        help="CSV of who should be there (form CSV) -> roster.csv Present/Absent",
    )
    ap.add_argument(
        "--ground-truth", type=Path, default=None,
        help="CSV with columns frame,reg (use UNKNOWN for non-students)",
    )
    ap.add_argument("--sampling-fps", type=float, default=5.0, help="recognise at N fps")
    ap.add_argument("--confirm-k", type=int, default=3, help="matches required (min 3)")
    ap.add_argument("--confirm-n", type=int, default=5, help="last N observations")
    ap.add_argument(
        "--confirm-window", type=float, default=10.0,
        help="seconds the K matches must fall in (min-frames rule: 3 frames / 10 s)",
    )
    ap.add_argument("--camera-name", default="classroom-file")
    ap.add_argument("--camera-type", default="entry", choices=["entry", "exit", "both"])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--annotate-every", type=float, default=1.0, help="seconds between saved frames")
    ap.add_argument("--start-time", default=None, help="ISO time of frame 0 (default: now - duration)")
    ap.add_argument("--rpc", action="store_true", help="use the shared Redis inference service")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"FATAL: video not found: {args.video}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or Path("eval/reports") / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"

    # ---- expected roster + ground truth -----------------------------------
    try:
        expected = _load_expected(args.expected)
    except FormImportError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    gt: dict[int, list[str]] = {}
    if args.ground_truth:
        from app.evalmetrics import load_ground_truth

        gt = load_ground_truth(args.ground_truth)

    # ---- inference ---------------------------------------------------------
    if not args.rpc:
        from app.inference.local import LocalInference

        LocalInference().install()
    from app.inference import client as inference_client

    # ---- database / gallery ----------------------------------------------
    from app.db import Base, SessionLocal, engine as db_engine
    from app.matching import EmbeddingIndex
    from app.models import Camera
    from app.services.attendance import process_recognition

    Base.metadata.create_all(db_engine)
    db = SessionLocal()

    camera = db.query(Camera).filter(Camera.name == args.camera_name).first()
    if camera is None:
        camera = Camera(name=args.camera_name, type=args.camera_type, file_path=str(args.video))
        db.add(camera)
        db.commit()
        db.refresh(camera)

    index = EmbeddingIndex()
    index.refresh()
    print("=" * 78)
    print(f"VIDEO : {args.video}")
    print(f"GALLERY: {index.student_count} students / {index.embedding_count} embeddings")
    print(f"RULES : threshold={index.threshold} margin={index.margin} "
          f"confirm {args.confirm_k} of last {args.confirm_n} in {args.confirm_window}s")
    if index.embedding_count == 0:
        print("WARNING: gallery is empty - everything will be UNKNOWN. "
              "Run scripts/setup_student_database.py first.")
    print("=" * 78)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"FATAL: cannot open video: {args.video}", file=sys.stderr)
        db.close()
        return 1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    clip_seconds = total_frames / fps if fps else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.start_time:
        start_wall = datetime.fromisoformat(args.start_time)
        if start_wall.tzinfo is None:
            start_wall = start_wall.replace(tzinfo=timezone.utc)
    else:
        start_wall = datetime.now(timezone.utc) - timedelta(seconds=clip_seconds)

    clip_label = (
        "synthetic clip built from enrolled photos: shows the flow, "
        "not real-CCTV accuracy"
        if args.video.name == "test_multi.mp4"
        else f"real camera footage ({args.video.name})"
    ) + (
        " - scored against ground truth"
        if args.ground_truth
        else " - no ground truth supplied, no accuracy claim"
    )
    print(f"CLIP : {clip_frames_line(total_frames, fps, width, height)}")

    tracker = BoxTracker()
    confirmator = IdentityConfirmator(
        k=args.confirm_k, n=args.confirm_n, window_s=args.confirm_window
    )
    ledger = AttendanceLedger()

    students: dict[int, dict] = {}
    for sid, name, reg in _load_students(db):
        students[sid] = {"name": name, "registration_no": reg}

    attendance: list[dict] = []
    observations: list[dict] = []
    track_labels: dict[int, dict] = {}
    unknown_faces = 0
    faces_total = 0
    frames_sampled = 0
    frames_decoded = 0
    stride = max(1, int(round(fps / max(args.sampling_fps, 0.1))))
    best_score: dict[int, float] = {}
    last_announce: dict[int, float] = {}
    pending_batch: list[tuple[int, float, bytes, object]] = []  # idx, t, jpeg, frame
    annotate_every = max(0.0, args.annotate_every)
    gt_mismatched = 0  # ground-truth frames whose label count != face count

    def _flush_batch() -> None:
        nonlocal faces_total, unknown_faces, frames_sampled, gt_mismatched
        if not pending_batch:
            return
        jpegs = [b for _, _, b, _ in pending_batch]
        results = inference_client.infer_frames(jpegs)["results"]
        annotate_stride = max(1, int(round(annotate_every * fps))) if annotate_every > 0 else 0
        for (frame_idx, clip_t, jpeg, frame), faces in zip(pending_batch, results):
            frames_sampled += 1
            bboxes = [f["bbox"] for f in faces]
            track_ids = tracker.update(bboxes, clip_t) if bboxes else []
            face_labels: dict[int, dict] = {}

            # ground truth is per face, ordered left -> right; a frame whose
            # label count does not match the detected faces is skipped rather
            # than guessed (counted and reported).
            expected_for: dict[int, str] = {}
            labels = gt.get(frame_idx)
            if labels:
                if len(labels) == len(faces):
                    order = sorted(
                        range(len(faces)), key=lambda i: faces[i]["bbox"][0]
                    )
                    for rank, i in enumerate(order):
                        expected_for[i] = labels[rank]
                else:
                    gt_mismatched += 1

            for i, (face, tid) in enumerate(zip(faces, track_ids)):
                faces_total += 1
                res = index.match(face["embedding"])
                sid = res.student_id
                score = float(res.score)
                if sid is None:
                    unknown_faces += 1
                elif score > best_score.get(sid, -1.0):
                    best_score[sid] = score

                # ---- observation row (identity-level, Task 10)
                stu = students.get(sid) if sid is not None else None
                predicted = stu["registration_no"] if stu else UNKNOWN
                expected = expected_for.get(i)
                row = {
                    "frame": frame_idx,
                    "time_s": round(clip_t, 3),
                    "track_id": tid,
                    "bbox": [round(float(v), 1) for v in face["bbox"][:4]],
                    "expected": expected if expected is not None else "",
                    "predicted": predicted,
                    "score": round(score, 4),
                }
                row["result"] = (
                    classify_pair(expected, predicted)
                    if expected is not None
                    else ("MATCH" if predicted != UNKNOWN else "UNKNOWN")
                )
                observations.append(row)

                # ---- K-of-N confirmation -> one attendance mark (Tasks 6+7)
                confirmed = confirmator.observe(tid, sid, clip_t, score)
                if confirmed is not None:
                    _mark_present(confirmed, tid, clip_t, score)

                if stu is not None:
                    face_labels[tid] = {
                        "name": stu["name"],
                        "reg": stu["registration_no"],
                        "score": score,
                        "status": "PRESENT" if ledger.is_marked(sid) else "PENDING",
                    }
                else:
                    face_labels[tid] = {
                        "name": UNKNOWN, "reg": UNKNOWN,
                        "score": score, "status": "NOT MARKED",
                    }

            # ---- annotated frame every annotate_every seconds
            if annotate_stride and frame_idx % annotate_stride == 0:
                frames_dir.mkdir(parents=True, exist_ok=True)
                labelled = annotate(frame.copy(), faces, track_ids, face_labels, clip_t)
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), labelled)

            # ---- live console line, at most once per 2 s per track
            for tid, info in face_labels.items():
                if info["reg"] == UNKNOWN:
                    continue
                prev = last_announce.get(tid)
                if prev is None or clip_t - prev >= 2.0:
                    last_announce[tid] = clip_t
                    print(
                        f"  [{hhmmss(start_wall, clip_t)}] {info['name']} | "
                        f"{mask_reg(info['reg'])} | match {info['score']:.2f} | "
                        f"{info['status']}"
                    )

        pending_batch.clear()

    def _mark_present(sid: int, tid: int, clip_t: float, score: float) -> None:
        stu = students.get(sid)
        if stu is None:
            return
        if not ledger.mark(sid, clip_t, score, tid):
            return  # already marked: no duplicate attendance rows (Task 7)
        ts = start_wall + timedelta(seconds=clip_t)
        try:
            outcome = process_recognition(db, student_id=sid, camera=camera, ts=ts)
            db.commit()
        except Exception as exc:  # noqa: BLE001 - never lose the run to one row
            db.rollback()
            print(f"  ! attendance write failed for student {sid}: {exc}")
            return
        rows_n = confirmator.known_observations(tid)
        attendance.append(
            {
                "name": stu["name"],
                "registration_no": stu["registration_no"],
                "status": "PRESENT",
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "time": ts.strftime("%H:%M:%S"),
                "clip_time_s": round(clip_t, 2),
                "match": f"{score:.4f}",
                "frames": rows_n,
                "track_id": tid,
                "student_id": sid,
                "db_action": outcome.action,
            }
        )
        print(
            f"  >>> PRESENT  {stu['name']} | {mask_reg(stu['registration_no'])} | "
            f"match {score:.4f} | {rows_n} frames | {ts.strftime('%H:%M:%S')}"
        )

    # ---- decode loop -------------------------------------------------------
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_decoded += 1
            if args.max_frames and frames_decoded > args.max_frames:
                break
            if idx % stride == 0:
                ok2, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality]
                )
                if ok2:
                    pending_batch.append((idx, idx / fps, buf.tobytes(), frame))
                    if len(pending_batch) >= 8:
                        _flush_batch()
            idx += 1
        _flush_batch()
    finally:
        cap.release()
        db.close()

    # ---- results -----------------------------------------------------------
    roster: list[dict] = []
    if expected:
        for stu in expected:
            match = next(
                (a for a in attendance if a["registration_no"] == stu["registration_no"]),
                None,
            )
            roster.append(
                {
                    "name": stu["name"],
                    "registration_no": stu["registration_no"],
                    "status": "PRESENT" if match else "ABSENT",
                    "timestamp": match["timestamp"] if match else "",
                    "match": match["match"] if match else "",
                }
            )

    labelled_pairs = [
        (r["expected"], r["predicted"]) for r in observations if r["expected"]
    ]
    metrics = identity_metrics(labelled_pairs) if labelled_pairs else {}

    _write_csv(
        out_dir / "observations.csv",
        observations,
        ["frame", "time_s", "track_id", "bbox", "expected", "predicted",
         "score", "result"],
    )
    _write_csv(
        out_dir / "attendance.csv",
        attendance,
        ["name", "registration_no", "status", "timestamp", "time",
         "clip_time_s", "match", "frames", "track_id", "student_id", "db_action"],
    )
    if roster:
        _write_csv(
            out_dir / "roster.csv", roster,
            ["name", "registration_no", "status", "timestamp", "match"],
        )
    if metrics:
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

    ctx = {
        "clip_label": clip_label,
        "note": (
            "Enrollment videos must never be used as the accuracy test set."
            if not gt
            else "Ground truth supplied: metrics below are identity-level. "
            "If this clip shares a recording session with the enrollment video "
            "it is easier than real CCTV - treat the numbers as an upper bound."
        ),
        "students": index.student_count,
        "embeddings": index.embedding_count,
        "confirm_k": args.confirm_k,
        "confirm_n": args.confirm_n,
        "confirm_window": args.confirm_window,
        "frames_decoded": frames_decoded,
        "frames_sampled": frames_sampled,
        "faces": faces_total,
        "unknown_faces": unknown_faces,
        "tracks": tracker.live_count,
        "attendance": attendance,
        "roster": roster,
        "observations": observations,
        "metrics": metrics,
        "gt_frames_skipped": gt_mismatched,
    }
    html_path = write_html_report(out_dir, ctx)

    # ---- console report (Task 16) ----------------------------------------
    print("\n" + "=" * 78)
    print("ATTENDANCE")
    if not attendance:
        print("  (nobody confirmed)")
    for row in attendance:
        print(f"  {row['name']}")
        print(f"  Reg No: {row['registration_no']}")
        print(f"  Match : {row['match']}")
        print(f"  Status: {row['status']}")
        print(f"  Time  : {row['time']}  ({row['frames']} frames)")
        print("-" * 40)
    if roster:
        print("\nROSTER (present/absent)")
        for row in roster:
            print(f"  {row['status']:8} {row['name']} "
                  f"({mask_reg(row['registration_no'])}) {row['timestamp']}")
    unknown_obs = sum(1 for o in observations if o["predicted"] == UNKNOWN)
    print(
        f"\nUNKNOWN: {unknown_obs} face observation(s) below threshold -> NOT MARKED "
        f"({faces_total - unknown_obs} matched observations)"
    )
    if metrics:
        print("\nMETRICS (identity-level)")
        for key in ("accuracy", "precision", "recall", "f1", "far", "frr",
                    "unknown_rejection_rate"):
            print(f"  {key:26} {metrics[key]:.4f}")
        print(f"  {'correct/total':26} {metrics['correct']}/{metrics['total']}")
        print(f"  {'false accepts (UNKNOWN->known)':26} {metrics['false_accepts']}")
        print(f"  {'false rejects (known->UNKNOWN)':26} {metrics['false_rejects']}")
        print(f"  {'wrong identities':26} {metrics['wrong_identities']}")
        if gt_mismatched:
            print(f"  {'GT frames skipped (count mismatch)':26} {gt_mismatched}")
    else:
        print("\nMETRICS: not computed - supply --ground-truth frame,reg")
    print("\nFILES")
    for p in sorted(out_dir.glob("*")):
        if p.is_file():
            print(f"  {p}")
    print(f"  {frames_dir}/  (annotated frames)")
    print(f"  {html_path}")
    return 0


def clip_frames_line(total: int, fps: float, width: int, height: int) -> str:
    return f"{total} frames @ {fps:.2f} fps = {total / max(fps, 1):.1f}s, {width}x{height}"


def hhmmss(start: datetime, offset_s: float) -> str:
    return (start + timedelta(seconds=offset_s)).strftime("%H:%M:%S")


def _load_students(db) -> list[tuple[int, str, str]]:  # noqa: ANN001
    from app.models import Student

    return [(s.id, s.name, s.registration_no) for s in db.query(Student).all()]


if __name__ == "__main__":
    raise SystemExit(main())
