# ---------------------------------------------------------------------------
# AI Video Attendance - task runner
#
#   make setup     one-command teammate setup (venv, deps, .env, key, migration,
#                  model weights, tests) - picks the right script per OS
#   make test      pytest
#   make demo      run the demo clip in-process (NO docker/redis)
#                  No clip ships with the archive - point VIDEO at your own:
#                    make demo VIDEO=/path/to/your-video.mp4
#   make run-api   uvicorn with auto-reload on http://localhost:8000
#   make enroll    bulk-enroll from a CSV:  make enroll CSV=test_students.csv
#   make calibrate print the score distributions that justify the threshold
#
# Windows without GNU make?  Use the scripts directly:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_dev.ps1
#   .venv\Scripts\python.exe -m pytest
# ---------------------------------------------------------------------------

PYTHON ?= python
CSV    ?= test_students.csv
VIDEO  ?= tests/assets/test_multi.mp4

# prefer the virtualenv when it exists (Windows first, then POSIX)
ifneq (,$(wildcard .venv/Scripts/python.exe))
  VPY := .venv/Scripts/python.exe
else ifneq (,$(wildcard .venv/bin/python))
  VPY := .venv/bin/python
else
  VPY := $(PYTHON)
endif

.PHONY: setup test demo run-api enroll calibrate models migrate help

help:
	@grep -E '^#   make ' Makefile | sed 's/^#   //'

setup:
ifeq ($(OS),Windows_NT)
	powershell -ExecutionPolicy Bypass -File scripts/setup_dev.ps1
else
	bash scripts/setup_dev.sh
endif

# NOTE: no extra -q here. pyproject addopts already supplies one, and a second
# one (-qq) hides the "N passed" summary line.
test:
	$(VPY) -m pytest

# Demo WITHOUT Docker or Redis: runs fully in-process, single process,
# CPU inference, sqlite. The clip is synthetic (built from enrolled photos).
demo:
	$(VPY) scripts/run_demo.py --video $(VIDEO)

run-api:
	$(VPY) -m uvicorn app.main:app --reload --port 8000

enroll:
	$(VPY) scripts/bulk_enroll.py --csv $(CSV) --create-tables

calibrate:
	$(VPY) eval/enroll_and_calibrate.py

models:
	$(VPY) scripts/download_models.py

migrate:
	$(VPY) -m alembic upgrade head
