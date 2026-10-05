#!/usr/bin/env python
"""
Demo script: runs the attendance pipeline completely in-process (no Redis, no workers).
- Creates 1 section, 1 classroom, 1 camera, 1 period
- Loads the pre-enrolled gallery from the database (the shipped gallery is
  SAMPLE placeholder photos + synthetic SAMPLE### ids, not enrolled students;
  see eval/enroll_and_calibrate.py --sample and README)
- Runs the sample video through detection, tracking, and matching for that period
- Generates eval/demo_report.csv and eval/demo_report.html
- Saves 6-10 annotated frames to eval/demo_frames/
"""

from __future__ import annotations

# Set environment variables BEFORE importing app modules
import os
os.environ["DATABASE_URL"] = "sqlite:///demo_real.db"
os.environ["FACE_QUALITY_ENABLED"] = "true"
os.environ["FORCE_CPU"] = "true"

import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings, settings
from app.db import Base, SessionLocal, engine
from app.models import Camera, Period, Student
from app.inference.engine import FaceEngine
from app.services.attendance import (
    process_recognition,
    student_attendance,
    period_summary,
)
from app.services.enrollment import enroll_student
from app.crypto import encrypt_embeddings, decrypt_embeddings


# ----------------------------------------------------------------------
# Demo configuration
# ----------------------------------------------------------------------
DEMO_SECTION = "CSE-A"
DEMO_CLASSROOM = "Room-101"
DEMO_PERIOD_NAME = "Morning Lecture"
DEMO_PERIOD_START = time(9, 0)
DEMO_PERIOD_END = time(10, 30)
DEMO_DAYS_BITMASK = 127  # All days
DEMO_LATE_MARGIN_MIN = 15
DEMO_EARLY_MARGIN_MIN = 15
CONFIRMATION_FRAMES = settings.confirmation_frames
CONFIRMATION_WINDOW_S = settings.confirmation_window_s

# Sample data paths
SAMPLE_VIDEO = Path(r"C:\Users\pc\OneDrive\Pictures\WhatsApp Video 2026-10-02 at 16.13.08.mp4")
STUDENT_PHOTOS_DIR = Path(settings.data_dir) / "students"

# Honest labelling of the demo clip (printed by the demo, embedded in both
# report formats, and repeated in README.md).
SYNTHETIC_CLIP_NOTE = (
    "synthetic clip built from enrolled photos (scripts/make_test_video.py): "
    "shows the flow, not real-CCTV accuracy"
)

# Output paths
DEMO_FRAMES_DIR = Path("eval") / "demo_frames"
DEMO_REPORT_CSV = Path("eval") / "demo_report.csv"
DEMO_REPORT_HTML = Path("eval") / "demo_report.html"


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def mask_name(name: str, idx: int) -> str:
    """Return masked name like P1, P2, etc."""
    return f"P{idx}"


def mask_reg_no(reg_no: str) -> str:
    """Return masked registration number (last 3 digits only)."""
    if len(reg_no) <= 3:
        return reg_no
    return "..." + reg_no[-3:] if len(reg_no) >= 3 else "***"


def mask_id(student_id: int) -> str:
    """Return masked student ID."""
    return f"ID{student_id:04d}"


def draw_boxes_and_labels(
    frame: np.ndarray,
    detections: list[dict],
    student_map: dict[int, dict],
    frame_idx: int,
) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        student_id = det.get("student_id")
        score = det.get("score", 0.0)
        
        # Draw bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label
        if student_id and student_id in student_map:
            student = student_map[student_id]
            label = f"{mask_name(student['name'], student['idx'])} ({mask_reg_no(student['registration_no'])})"
            label += f" {score:.2f}"
            color = (0, 255, 0)
        else:
            label = f"Unknown {score:.2f}"
            color = (0, 0, 255)
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Frame counter
    cv2.putText(annotated, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return annotated


def save_annotated_frame(frame: np.ndarray, frame_idx: int, output_dir: Path) -> Path:
    """Save annotated frame to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(path), frame)
    return path


# ----------------------------------------------------------------------
# Main demo function
# ----------------------------------------------------------------------
def run_demo(
    video_path: Path | None = None,
    max_frames: int | None = None,
    save_frames: bool = True,
) -> dict:
    """Run the complete demo pipeline."""
    
    # Setup paths
    video_path = video_path or SAMPLE_VIDEO
    if not video_path.exists():
        raise FileNotFoundError(f"Sample video not found: {video_path}")
    
    DEMO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # --- probe the clip: its LENGTH is what the period window must cover ----
    _probe = cv2.VideoCapture(str(video_path))
    if not _probe.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    clip_frames = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_fps = float(_probe.get(cv2.CAP_PROP_FPS) or 30.0)
    _probe.release()
    clip_seconds = clip_frames / clip_fps if clip_fps else 0.0
    is_synthetic = video_path.name == "test_multi.mp4"
    clip_label = (
        SYNTHETIC_CLIP_NOTE
        if is_synthetic
        else f"real camera footage ({video_path.name}) - accuracy not yet calibrated"
    )
    print("\n" + "=" * 78)
    print(f"CLIP: {video_path.name}  {clip_frames} frames @ {clip_fps:.2f} fps "
          f"= {clip_seconds:.2f}s")
    print(f"CLIP TYPE: {clip_label}")
    print("=" * 78)
    
    # Initialize database
    Base.metadata.create_all(engine)
    db = SessionLocal()
    
    try:
        # 1. Section is just a string field in Student and Period models
        print(f"Section: {DEMO_SECTION}")
        
        # 2. Create classroom (using Camera as proxy for classroom)
        classroom_camera = db.query(Camera).filter(Camera.name == DEMO_CLASSROOM).first()
        if not classroom_camera:
            classroom_camera = Camera(
                name=DEMO_CLASSROOM,
                location=DEMO_CLASSROOM,
                file_path=str(video_path),
                type="both",
                sampling_rate=settings.default_sampling_rate,
            )
            db.add(classroom_camera)
            db.commit()
            db.refresh(classroom_camera)
        print(f"Classroom camera: {classroom_camera.name} (id={classroom_camera.id})")
        
        # 2. Create period - window == clip length (see probe above)
        period_end_time = (
            datetime.combine(date.today(), DEMO_PERIOD_START)
            + timedelta(seconds=clip_seconds)
        ).time()
        period = db.query(Period).filter(Period.name == DEMO_PERIOD_NAME).first()
        if not period:
            period = Period(
                name=DEMO_PERIOD_NAME,
                start_time=DEMO_PERIOD_START,
                end_time=period_end_time,
                days_bitmask=DEMO_DAYS_BITMASK,
                section=DEMO_SECTION,
            )
            db.add(period)
        else:
            # keep the window in sync with the clip being demoed
            period.start_time = DEMO_PERIOD_START
            period.end_time = period_end_time
            period.section = DEMO_SECTION
        db.commit()
        db.refresh(period)
        print(f"Period: {period.name} (id={period.id}, {period.start_time}-{period.end_time}, section={period.section})")
        print(f"Period window == clip length: {clip_seconds:.2f}s "
              f"(late/early margins = 10% of the window)")
        
        # 3. Load pre-enrolled gallery records from the database
        print("\n=== Loading pre-enrolled gallery records ===")
        student_map = {}  # student_id -> {idx, name, reg_no, section}
        
        # Gallery records (sample placeholder gallery unless re-enrolled)
        students = db.query(Student).filter(Student.embeddings != "").all()
        if not students:
            print("No enrolled students found in database!")
            return {"report_csv": "", "report_html": "", "frames_dir": str(DEMO_FRAMES_DIR), "summary": {}}
        
        for idx, student in enumerate(students, 1):
            student_map[student.id] = {
                "idx": idx,
                "name": student.name,
                "registration_no": student.registration_no,
                "section": student.section,
            }
            folder = (
                student.photo_paths[0].split("/")[0]
                if student.photo_paths else "?"
            )
            print(f"  Loaded: {mask_name(student.name, idx)} "
                  f"(reg=...{student.registration_no[-3:]}, id={student.id}, "
                  f"folder=data/students/{folder}/)")
        
        print(f"\nTotal loaded: {len(student_map)} students")
        
        # 4. Initialize FaceEngine
        face_engine = FaceEngine(settings)
        warmup_timings = face_engine.warmup()
        print(f"\nEngine warmup: {warmup_timings.total_ms:.1f} ms")
        
        # 5. Run video through attendance pipeline
        print("\n=== Processing video ===")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video: {total_frames} frames @ {fps:.1f} fps")
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            print(f"Limiting to {max_frames} frames")
        
        # Process frames
        frame_idx = 0
        all_detections = []  # Store all detections for reporting
        saved_frame_count = 0
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Process frame
            results, timings = face_engine.infer([frame])
            
            # Store detections for this frame
            frame_detections = []
            for face in results[0]:
                student_id = None
                # Simple matching: find best match in enrolled students
                best_score = -1.0
                best_student_id = None
                for student_id, student_info in student_map.items():
                    # Get student embeddings from DB
                    student = db.query(Student).filter(Student.id == student_id).first()
                    if student and student.embeddings:
                        from app.crypto import decrypt_embeddings
                        try:
                            embeddings = decrypt_embeddings(student.embeddings)
                            for emb in embeddings:
                                # Compute cosine similarity
                                emb_array = np.array(emb, dtype=np.float32)
                                query_emb = np.array(face["embedding"], dtype=np.float32)
                                # Normalize
                                emb_norm = emb_array / max(np.linalg.norm(emb_array), 1e-9)
                                query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-9)
                                sim = float(np.dot(emb_norm, query_norm))
                                if sim > best_score:
                                    best_score = sim
                                    best_student_id = student_id
                        except Exception as e:
                            print(f"  Warning: Failed to match student {student_id}: {e}")
                
                if best_score >= settings.recognition_threshold:
                    student_id = best_student_id
                    score = best_score
                    print(f"  Match: student {student_id} score={best_score:.3f}")
                else:
                    student_id = None
                    score = 0.0
                    if best_score > 0.1:
                        print(f"  Below threshold: best={best_score:.3f} (threshold={settings.recognition_threshold:.2f})")
                
                face_info = {
                    "frame_idx": frame_idx,
                    "bbox": face["bbox"],
                    "score": score,
                    "student_id": student_id,
                    "embedding": face.get("embedding"),
                }
                frame_detections.append(face_info)
                
                # Track for period attendance (frame timestamp inside window)
                if student_id:
                    # Call process_recognition
                    process_recognition(
                        db,
                        student_id=student_id,
                        camera=classroom_camera,
                        ts=datetime.combine(date.today(), period.start_time)
                        + timedelta(seconds=frame_idx / max(fps, 1e-6)),
                    )
            
            all_detections.append(frame_detections)
            
            # Save annotated frames (every 10th frame + first/last)
            if save_frames and (frame_idx % 40 == 0 or frame_idx == 1 or frame_idx == total_frames):
                # Annotate frame
                face_dicts = []
                for det in frame_detections:
                    face_dicts.append({
                        "bbox": det["bbox"],
                        "student_id": det.get("student_id"),
                        "score": det.get("score", 0.0),
                    })
                annotated = draw_boxes_and_labels(frame, face_dicts, student_map, frame_idx)
                saved_path = save_annotated_frame(annotated, frame_idx, DEMO_FRAMES_DIR)
                saved_frame_count += 1
            
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...")
        
        cap.release()
        print(f"\nProcessed {frame_idx} frames, saved {saved_frame_count} annotated frames")
        
        # 6. Generate attendance report
        print("\n=== Generating attendance report ===")
        report_rows = []
        
        for student_id, student_info in student_map.items():
            # Get attendance for this student
            attendance = student_attendance(db, 
                db.query(Student).filter(Student.id == student_id).first(), 
                date.today()
            )
            
            # Get first/last seen from detections
            student_detections = [
                det for frame_dets in all_detections 
                for det in frame_dets 
                if det.get("student_id") == student_id
            ]
            
            if student_detections:
                first_seen = min(d["frame_idx"] for d in student_detections)
                last_seen = max(d["frame_idx"] for d in student_detections)
                frame_count = len(student_detections)
                best_score = max(d.get("score", 0) for d in student_detections)
                avg_score = sum(d.get("score", 0) for d in student_detections) / len(student_detections)
                
                # Convert frame index to time
                first_seen_time = timedelta(seconds=first_seen / fps)
                last_seen_time = timedelta(seconds=last_seen / fps)
                duration = timedelta(seconds=(last_seen - first_seen) / fps)
                
                # Determine status
                period_start_dt = datetime.combine(date.today(), period.start_time)
                period_end_dt = datetime.combine(date.today(), period.end_time)
                # margins scale with the window (10% of it) so an 8s clip and a
                # 90min lecture both behave sensibly
                window = period_end_dt - period_start_dt
                late_margin = window * 0.10
                early_margin = window * 0.10

                first_seen_dt = period_start_dt + timedelta(seconds=first_seen / fps)
                last_seen_dt = period_start_dt + timedelta(seconds=last_seen / fps)

                is_late = first_seen_dt > (period_start_dt + late_margin)
                left_early = last_seen_dt < (period_end_dt - early_margin)
                
                # Check if present (confirmation_frames met)
                present = frame_count >= settings.confirmation_frames
                status = "Present" if present else "Absent"
                if is_late:
                    status += " (Late)"
                if left_early:
                    status += " (Left Early)"
                
                row = {
                    "masked_name": mask_name(student_info["name"], student_info["idx"]),
                    "masked_id": mask_reg_no(student_info["registration_no"]),
                    "status": status,
                    "first_seen_frame": first_seen,
                    "last_seen_frame": last_seen,
                    "first_seen_time": str(first_seen_time),
                    "last_seen_time": str(last_seen_time),
                    "duration": str(duration),
                    "frame_count": frame_count,
                    "best_score": round(best_score, 4),
                    "avg_score": round(avg_score, 4),
                    "is_late": is_late,
                    "left_early": left_early,
                    "present": present,
                }
            else:
                row = {
                    "masked_name": mask_name(student_info["name"], student_info["idx"]),
                    "masked_id": mask_reg_no(student_info["registration_no"]),
                    "status": "Absent (Not enough footage)",
                    "first_seen_frame": "",
                    "last_seen_frame": "",
                    "first_seen_time": "",
                    "last_seen_time": "",
                    "duration": "",
                    "frame_count": 0,
                    "best_score": 0.0,
                    "avg_score": 0.0,
                    "is_late": False,
                    "left_early": False,
                    "present": False,
                }
            
            report_rows.append(row)
        
        # Unknown faces count
        unknown_count = sum(
            1 for frame_dets in all_detections 
            for d in frame_dets 
            if d.get("student_id") is None
        )
        
        # Honest labelling: every row carries what kind of clip produced it
        for r in report_rows:
            r["clip_label"] = clip_label

        # Save CSV report
        report_fields = [
            "masked_name", "masked_id", "status", "present",
            "first_seen_frame", "last_seen_frame",
            "first_seen_time", "last_seen_time", "duration",
            "frame_count", "best_score", "avg_score",
            "is_late", "left_early", "present", "clip_label"
        ]
        
        with open(DEMO_REPORT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=report_fields)
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"CSV report saved to: {DEMO_REPORT_CSV}")
        
        # Generate HTML report
        html_content = generate_html_report(
            report_rows, unknown_count, saved_frame_count, DEMO_FRAMES_DIR,
            frames_processed=frame_idx,
        )
        with open(DEMO_REPORT_HTML, "w") as f:
            f.write(html_content)
        print(f"HTML report saved to: {DEMO_REPORT_HTML}")
        
        # Print summary
        print("\n=== DEMO SUMMARY ===")
        print(f"  CLIP TYPE: {clip_label}")
        for row in report_rows:
            print(f"  {row['masked_name']} ({row['masked_id']}): {row['status']} - "
                  f"Frames: {row['frame_count']}, Best: {row['best_score']:.3f}, "
                  f"Late: {row['is_late']}, Early: {row['left_early']}")
        print(f"\nUnknown faces: {unknown_count}")
        print(f"Annotated frames saved: {saved_frame_count}")
        print(f"Frames dir: {DEMO_FRAMES_DIR}")
        
        return {
            "report_csv": str(DEMO_REPORT_CSV),
            "report_html": str(DEMO_REPORT_HTML),
            "frames_dir": str(DEMO_FRAMES_DIR),
            "summary": {
                "total_students": len(student_map),
                "present": sum(1 for r in report_rows if r["present"]),
                "absent": sum(1 for r in report_rows if not r["present"]),
                "late": sum(1 for r in report_rows if r["is_late"]),
                "left_early": sum(1 for r in report_rows if r["left_early"]),
                "unknown_faces": unknown_count,
                "frames_processed": frame_idx,
                "annotated_frames": saved_frame_count,
            }
        }
    
    finally:
        db.close()


def generate_html_report(report_rows: list, unknown_count: int, frame_count: int, frames_dir: Path, frames_processed: int = 0) -> str:
    """Generate HTML report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Demo Attendance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .present {{ color: green; font-weight: bold; }}
        .absent {{ color: red; }}
        .late {{ color: orange; }}
        .early {{ color: purple; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
        img {{ max-width: 100%; height: auto; margin: 10px; border: 1px solid #ddd; }}
        .frame-gallery {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .banner {{ background: #fff3cd; border: 1px solid #ffecb5; padding: 14px 18px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Demo Attendance Report</h1>
    <p class="banner"><strong>Clip:</strong> {report_rows[0].get('clip_label', 'n/a') if report_rows else 'n/a'}</p>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Students:</strong> {len([r for r in report_rows if r.get('present')]) + len([r for r in report_rows if not r.get('present')])}</p>
        <p><strong>Present:</strong> <span class="present">{sum(1 for r in report_rows if r.get('present'))}</span></p>
        <p><strong>Absent:</strong> <span class="absent">{sum(1 for r in report_rows if not r.get('present'))}</span></p>
        <p><strong>Late:</strong> <span class="late">{sum(1 for r in report_rows if r.get('is_late'))}</span></p>
        <p><strong>Left Early:</strong> <span class="early">{sum(1 for r in report_rows if r.get('left_early'))}</span></p>
        <p><strong>Unknown Faces:</strong> {unknown_count}</p>
        <p><strong>Frames Processed:</strong> {frames_processed}</p>
        <p><strong>Annotated Frames Saved:</strong> {frame_count}</p>
    </div>
    
    <h2>Per-Student Attendance</h2>
    <table>
        <tr>
            <th>Masked Name</th>
            <th>Masked ID</th>
            <th>Status</th>
            <th>Present</th>
            <th>First Seen</th>
            <th>Last Seen</th>
            <th>Duration</th>
            <th>Frames</th>
            <th>Best Score</th>
            <th>Avg Score</th>
            <th>Late</th>
            <th>Left Early</th>
        </tr>
"""
    
    for row in report_rows:
        status_class = ""
        if row.get("is_late"):
            status_class = "late"
        elif row.get("left_early"):
            status_class = "early"
        elif row.get("present"):
            status_class = "present"
        else:
            status_class = "absent"
        
        html += f"""
        <tr>
            <td>{row['masked_name']}</td>
            <td>{row['masked_id']}</td>
            <td class="{status_class}">{row['status']}</td>
            <td>{'Yes' if row.get('present') else 'No'}</td>
            <td>{row.get('first_seen_time', '')}</td>
            <td>{row.get('last_seen_time', '')}</td>
            <td>{row.get('duration', '')}</td>
            <td>{row.get('frame_count', 0)}</td>
            <td>{row.get('best_score', 0):.3f}</td>
            <td>{row.get('avg_score', 0):.3f}</td>
            <td>{'Yes' if row.get('is_late') else 'No'}</td>
            <td>{'Yes' if row.get('left_early') else 'No'}</td>
        </tr>
"""
    
    html += """
    </table>
    
    <h2>Annotated Frames</h2>
    <div class="frame-gallery">
"""
    
    # Add frame images if they exist
    import glob
    frame_files = sorted(glob.glob(str(DEMO_FRAMES_DIR / "*.jpg")))
    for frame_path in frame_files[:10]:
        frame_name = Path(frame_path).name
        html += f'        <img src="{frame_name}" alt="{frame_name}" width="320">\n'
    
    html += """
    </div>
</body>
</html>
"""
    return html


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run attendance demo without Redis")
    parser.add_argument("--video", type=Path, help="Path to video file (default: sample video)")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument("--no-frames", action="store_true", help="Don't save annotated frames")
    args = parser.parse_args()
    
    print("=== AI Video Attendance Demo (No Redis) ===")
    print(f"Video: {args.video or SAMPLE_VIDEO}")
    print(f"Max frames: {args.max_frames or 'all'}")
    print(f"Save frames: {not args.no_frames}")
    
    result = run_demo(
        video_path=args.video,
        max_frames=args.max_frames,
        save_frames=not args.no_frames,
    )
    
    print("\n=== DEMO COMPLETE ===")
    print(f"Report CSV: {result['report_csv']}")
    print(f"Report HTML: {result['report_html']}")
    print(f"Frames: {result['frames_dir']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())