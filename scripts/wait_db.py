"""Wait until the database accepts connections (used by docker entrypoints).

    python scripts/wait_db.py [--attempts 60] [--interval 1]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text  # noqa: E402

from app.config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=60)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    last_error: Exception | None = None
    for attempt in range(1, args.attempts + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"database ready (attempt {attempt})")
            return 0
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"waiting for database ({attempt}/{args.attempts}): {exc}")
            time.sleep(args.interval)
    print(f"database never became ready: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
