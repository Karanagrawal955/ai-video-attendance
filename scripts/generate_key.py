#!/usr/bin/env python
"""Print a fresh EMBEDDING_ENCRYPTION_KEY (Fernet-compatible, 32 random bytes).

Stdlib only - safe to run before dependencies are installed:

    python scripts/generate_key.py
    python scripts/generate_key.py >> .env     # appends the assignment
"""
from __future__ import annotations

import base64
import os


def main() -> int:
    key = base64.urlsafe_b64encode(os.urandom(32)).decode()
    # sanity: must decode back to exactly 32 bytes (matches app/config.py)
    decoded = base64.urlsafe_b64decode(key + "=" * (-len(key) % 4))
    assert len(decoded) == 32, decoded
    print(f"EMBEDDING_ENCRYPTION_KEY={key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
