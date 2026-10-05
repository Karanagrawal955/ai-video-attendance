#!/usr/bin/env python
"""Download (or verify) the pretrained face-model weights.

Weights are NEVER committed to the repository - this script fetches them on
first run into INSIGHTFACE_ROOT (default ``~/.insightface``):

    python scripts/download_models.py            # idempotent, offline-friendly
    python scripts/download_models.py --force    # re-download

Source: the official InsightFace release assets
(https://github.com/deepinsight/insightface/releases) - same URL the
insightface package itself uses for ``buffalo_l``.

Exit codes: 0 = weights present, 1 = download/verification failed.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

MODELS = {
    # model pack -> (release asset, files that MUST exist afterwards)
    "buffalo_l": (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        ("det_10g.onnx", "w600k_r50.onnx"),  # detection + recognition
    ),
}


def _verify(pack_dir: Path, required: tuple[str, ...]) -> list[str]:
    """Return a list of problems (empty == good)."""
    problems = []
    for name in required:
        f = pack_dir / name
        if not f.exists():
            problems.append(f"missing {f}")
        elif f.stat().st_size < 1024:
            problems.append(f"suspiciously small {f} ({f.stat().st_size} bytes)")
    if problems:
        return problems
    # structural check: the .onnx files must parse as ONNX graphs
    try:
        import onnx  # noqa: WPS433 - optional dependency of insightface

        for name in required:
            onnx.load(str(pack_dir / name), load_external_data=False)
    except ImportError:
        print("  (onnx package not installed - skipped graph validation)")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"onnx parse failed: {exc}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="insightface root (default: settings.insightface_root)")
    ap.add_argument("--model", default="buffalo_l", help=f"one of {sorted(MODELS)}")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if args.model not in MODELS:
        print(f"unknown model {args.model!r}; known: {sorted(MODELS)}", file=sys.stderr)
        return 1
    url, required = MODELS[args.model]

    root = Path(args.root or os.path.expanduser("~/.insightface")).expanduser()
    pack_dir = root / "models" / args.model

    if pack_dir.is_dir() and not args.force:
        problems = _verify(pack_dir, required)
        if not problems:
            print(f"[models] OK - {args.model} already at {pack_dir}")
            for name in sorted(p.name for p in pack_dir.glob("*.onnx")):
                size = (pack_dir / name).stat().st_size
                print(f"           {name:<20} {size / 1e6:7.2f} MB")
            return 0
        print(f"[models] present but invalid: {'; '.join(problems)} - re-downloading")

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "models" / f"{args.model}.zip"
    print(f"[models] downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(zip_path, "wb") as out:
            total = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
                print(f"\r  {total / 1e6:8.1f} MB", end="", flush=True)
            print()
    except Exception as exc:  # noqa: BLE001
        print(f"[models] DOWNLOAD FAILED: {exc}", file=sys.stderr)
        print("[models] fix: check network/proxy, or copy an existing "
              f"{pack_dir} folder from another machine.", file=sys.stderr)
        return 1

    print(f"[models] unpacking to {pack_dir}")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root / "models")
    except Exception as exc:  # noqa: BLE001
        print(f"[models] UNPACK FAILED: {exc}", file=sys.stderr)
        return 1

    problems = _verify(pack_dir, required)
    if problems:
        print(f"[models] VERIFICATION FAILED: {'; '.join(problems)}", file=sys.stderr)
        return 1
    print(f"[models] OK - {args.model} ready at {pack_dir} "
          f"(weights are gitignored, never committed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
