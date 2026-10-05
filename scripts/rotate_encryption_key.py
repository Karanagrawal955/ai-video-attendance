#!/usr/bin/env python
"""Rotate the embedding encryption key (re-encrypt every stored row in place).

    # 1. generate and store the NEW key first (keep the old one too!)
    python scripts/generate_key.py            # -> EMBEDDING_ENCRYPTION_KEY=<new>
    # 2. put <new> in .env AFTER the rotation run (see step 4 of README)

    # dry run: report what would be re-encrypted
    python scripts/rotate_encryption_key.py --new-key <new> --dry-run

    # real run (old key defaults to the currently configured one)
    python scripts/rotate_encryption_key.py --new-key <new>

    # explicit old key (e.g. rotating away from the dev key)
    python scripts/rotate_encryption_key.py --old-key <old> --new-key <new>

Rows already encrypted with the NEW key are left untouched.  Rows that can be
decrypted by neither key are reported (never silently dropped).
Exit code 0 = every decryptable row is now on the new key.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _validate_key(value: str, label: str) -> str:
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{label} is not valid base64: {exc}")
    if len(decoded) != 32:
        raise SystemExit(f"{label} must decode to exactly 32 bytes (got {len(decoded)})")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-key", required=True, help="new base64 32-byte key")
    ap.add_argument("--old-key", default=None, help="old key (default: currently configured key)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    new_key = _validate_key(args.new_key, "--new-key")

    from app.config import settings
    from app.crypto import active_key_material, fernet_from_material
    from app.db import SessionLocal
    from app.models import Student

    old_material = args.old_key or active_key_material()
    if args.old_key:
        _validate_key(args.old_key, "--old-key")

    old_f = fernet_from_material(old_material)
    new_f = fernet_from_material(new_key)
    if old_material == new_key:
        print("old key == new key: nothing to do")
        return 0

    db = SessionLocal()
    rotated = already_new = failed = 0
    try:
        rows = db.query(Student).filter(Student.embeddings.isnot(None)).all()
        print(f"[rotate] students with embeddings: {len(rows)}  dry_run={args.dry_run}")
        for s in rows:
            blob = s.embeddings or ""
            if not blob:
                continue
            plaintext = None
            try:
                plaintext = old_f.decrypt(blob.encode())
            except Exception:  # noqa: BLE001
                try:
                    plaintext = new_f.decrypt(blob.encode())
                except Exception:  # noqa: BLE001
                    failed += 1
                    print(f"  student id={s.id}: decrypt failed with old AND new key - SKIPPED")
                    continue
                already_new += 1
                continue
            if not args.dry_run:
                s.embeddings = new_f.encrypt(plaintext).decode()
            rotated += 1
        if not args.dry_run:
            db.commit()
    finally:
        db.close()

    print(
        f"[rotate] rotated={rotated} already_on_new_key={already_new} "
        f"failed={failed} dry_run={args.dry_run}"
    )
    if failed:
        print("[rotate] FAILED: some rows could not be decrypted", file=sys.stderr)
        return 1
    if not args.dry_run:
        print("[rotate] next: put the new key in .env (EMBEDDING_ENCRYPTION_KEY=<new>)")
        print("[rotate]       and restart API + workers, then verify with:")
        print("[rotate]       python -c \"from app.db import SessionLocal; from app.models "
              "import Student; from app.crypto import decrypt_embeddings; "
              "db=SessionLocal(); s=db.query(Student).filter(Student.embeddings!='').first(); "
              "print(len(s.registration_no), len(decrypt_embeddings(s.embeddings)))\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
