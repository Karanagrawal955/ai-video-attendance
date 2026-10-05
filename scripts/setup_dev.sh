#!/usr/bin/env bash
# One-command dev setup for Linux/macOS (mirror of scripts/setup_dev.ps1).
#
#   bash scripts/setup_dev.sh
#
# Steps (7): python check -> venv -> pinned deps -> .env + encryption key
#            -> DB migration -> model weights -> test suite.
# Idempotent. No Docker and no Redis required: the demo runs in-process
# (README "Demo without Docker or Redis").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "repo root: $ROOT"

echo
echo "== 1/7 python =="
command -v python3 >/dev/null || { echo "python3 not found - install Python 3.11+" >&2; exit 1; }
PYV="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || { echo "Python $PYV found, need >= 3.11" >&2; exit 1; }
echo "python $PYV  OK"

echo
echo "== 2/7 virtualenv =="
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
  echo "created .venv"
else
  echo ".venv already exists"
fi
VPY=".venv/bin/python"
"$VPY" -m pip install --quiet --upgrade pip

echo
echo "== 3/7 pinned dependencies =="
"$VPY" -m pip install --quiet -r requirements.txt -r requirements-dev.txt
"$VPY" -c "import fastapi, sqlalchemy, insightface, cv2, pytest; print('imports OK')"

echo
echo "== 4/7 .env + encryption key =="
if [ ! -f .env ]; then
  cp .env.example .env
  echo "created .env from .env.example"
else
  echo ".env already exists - untouched"
fi
if grep -Eq '^EMBEDDING_ENCRYPTION_KEY=[[:space:]]*$' .env; then
  KEY="$("$VPY" scripts/generate_key.py | tail -n 1 | sed 's/^EMBEDDING_ENCRYPTION_KEY=//')"
  [ -n "$KEY" ] || { echo "scripts/generate_key.py produced no key" >&2; exit 1; }
  # portable in-place edit (BSD/GNU sed differ)
  tmp="$(mktemp)"
  sed "s|^EMBEDDING_ENCRYPTION_KEY=.*$|EMBEDDING_ENCRYPTION_KEY=$KEY|" .env > "$tmp" && mv "$tmp" .env
  echo "generated EMBEDDING_ENCRYPTION_KEY (.env is gitignored)"
else
  echo "EMBEDDING_ENCRYPTION_KEY already set - untouched"
fi

echo
echo "== 5/7 database migration (alembic upgrade head) =="
if [ -x ".venv/bin/alembic" ]; then ./.venv/bin/alembic upgrade head; else "$VPY" -m alembic upgrade head; fi
echo "schema is at head"

echo
echo "== 6/7 model weights =="
"$VPY" scripts/download_models.py

echo
echo "== 7/7 test suite =="
# no extra -q: pyproject addopts already supplies one, and -qq hides the summary
"$VPY" -m pytest

echo
echo "SETUP OK - next commands:"
echo "  .venv/bin/python scripts/run_demo.py --video /path/to/your-video.mp4"
echo "  .venv/bin/python -m uvicorn app.main:app --reload"
echo ""
echo "  NOTE: no demo clip ships with this archive (media-free)."
echo "  Build your own from enrolled photos with scripts/make_test_video.py."
