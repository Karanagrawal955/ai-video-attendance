#!/usr/bin/env python
"""Bulk enrollment CLI: import students from CSV with photos.

Usage:
    python -m scripts.bulk_enroll --csv data/students.csv [--dry-run] [--skip-quality]
    python -m scripts.bulk_enroll --csv data/students.csv --limit 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db import SessionLocal
from app.logging_config import configure
from app.services import enrollment


def main() -> int:
    configure(settings)

    parser = argparse.ArgumentParser(
        description="Bulk enroll students from CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", required=True, help="Path to CSV file with columns: name, registration_no, section, photo_path_or_folder"
    )
    parser.add_argument(
        "--data-dir", default=None, help="Base directory for student photos (default: settings.data_dir)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of rows to process (for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate without writing to DB"
    )
    parser.add_argument(
        "--skip-quality", action="store_true", help="Skip per-photo quality checks"
    )
    parser.add_argument(
        "--report", default="import_report.csv", help="Output path for import report CSV"
    )
    parser.add_argument(
        "--create-tables", action="store_true", help="Create database tables if they don't exist"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return 1

    data_dir = Path(args.data_dir) if args.data_dir else settings.photos_root

    if args.create_tables:
        from app.db import Base, engine
        Base.metadata.create_all(engine)
        print("Database tables created/verified")

    db = SessionLocal()
    try:
        print(f"Reading {csv_path}...")
        print(f"Data directory: {data_dir}")
        if args.dry_run:
            print("DRY RUN MODE - no database changes")
        if args.skip_quality:
            print("Quality checks DISABLED")
        print()

        results, accepted, rejected = enrollment.bulk_enroll_from_csv(
            db,
            csv_path,
            data_dir=data_dir,
            skip_quality_checks=args.skip_quality,
            dry_run=args.dry_run,
        )

        if args.limit:
            results = results[:args.limit]
            accepted = sum(1 for r in results if r.status == "accepted")
            rejected = sum(1 for r in results if r.status == "rejected")

        print(f"\n=== Import Summary ===")
        print(f"Total rows: {len(results)}")
        print(f"Accepted: {accepted}")
        print(f"Rejected: {rejected}")

        for r in results:
            status_symbol = "[OK]" if r.status == "accepted" else "[FAIL]"
            reg_display = f"{r.registration_no[:3]}***" if len(r.registration_no) > 3 else r.registration_no
            print(f"  {status_symbol} Row {r.row_number}: {r.name} (reg={reg_display}) - {r.status}")
            if r.reason:
                print(f"      Reason: {r.reason}")

        enrollment.write_import_report(results, Path(args.report))
        print(f"\nImport report written to: {args.report}")

    finally:
        db.close()

    return 0 if rejected == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())