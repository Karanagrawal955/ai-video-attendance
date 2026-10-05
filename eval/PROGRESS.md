# PROGRESS LOG — AI Video Attendance (updated 2026-10-04)

## INCIDENT: unrelated project files appeared in this working tree
An unrelated assignment's code replaced `app/config.py`, `app/db.py`,
`app/main.py`, `app/schemas.py`, `README.md`, `requirements.txt` and deleted
attendance files (`app/models.py`, `app/matching.py`,
`app/services/enrollment.py`, `app/pipeline/camera_task.py`, `app/static/*`,
`tests/conftest.py`, `tests/test_api.py`, `tests/test_attendance.py`,
`tests/test_matching.py`, `tests/test_quality.py`).

### Recovery (user approved: restore attendance, back up the other)
- Other project's files backed up outside the repo:
  `C:\Users\pc\AppData\Local\Temp\opencode\project_split\cybercrime\`
- Attendance files restored from `ai-video-attendance-final.zip` (latest) and
  from git HEAD (static UI, test_api/test_attendance/test_matching, main.py,
  from which periods/alerts routers were re-added).
- `tests/test_alerts.py` restored from `ai-video-attendance-backend.zip`.
- Debug/enrolment scripts holding raw personal data moved out of the repo to
  `...\Temp\opencode\project_split\debug_scripts\`.
- **Lost**: `tests/test_quality.py` (13 tests) — superseded by
  `tests/test_regression.py` (8 tests, STEP 1g).

## TEST STATUS
- `python -m pytest -q` -> **51/51 passed, exit 0** (43 restored baseline + 8
  new regression tests). Baseline before incident: 43; earlier best: 56.

## REGRESSION ROOT CAUSE (STEP 1) — proven, fixed
1. Demo read `demo_real.db`, which had **0 students** (gallery actually lived
   in `demo_real2.db`), and its stored vectors matched **no photo on disk**
   (cosine of stored vector vs freshly-embedded identical photo = **0.0314**,
   must be ~1.0). -> all pairs scored ~0.15 -> "0/6 matched" at 0.25.
2. Photo folders `data/students/{1,4}`, `{2,5}`, `{3,6}` hold the **same
   person** (cosine 0.9998, identical bboxes) -> only 3 unique identities; the
   naive best-vs-second margin was ~0.002 and would have rejected every
   genuine match.
3. Fix: re-enrolled `demo_real.db` through the single `FaceEngine.infer` path
   (encrypt -> store), added a duplicate-identity-aware margin rule.

### Evidence (eval/diag_threshold.py, eval/enroll_and_calibrate.py)
- determinism cosine(run1, run2) = 1.000000
- STEP 1b: cosine(stored, re-embedded identical photo) = **1.0000** (was 0.0314)
- SAME identity photo-photo: n=90 min=0.9096 mean=0.9599 max=0.9998
- DIFFERENT identity:        n=384 min=-0.1315 mean=0.0048 max=0.1246
- genuine video faces vs gallery: n=83 min=0.5030 median=0.9553 max=0.9844
- encryption round-trip identical_to_fresh=True for all 6 students
-> single threshold **0.40** (inside 0.1246..0.5030), margin **0.10**,
   duplicate identity sim **0.90** (all in `app/config.py` with these numbers).

## DEMO RESULT (scripts/run_demo.py --video tests/assets/test_multi.mp4)
- P1 ...001 Present, 116 frames, best 0.984
- P2 ...179 Present, 85 frames, best 0.955
- P3 ...070 Present, 54 frames, best 0.966
- P4 ...178 Present, 10 frames, best 0.916
- P5 ...637 Present, 53 frames, best 0.961
- P6 ...829 Present, 10 frames, best 0.949
- Unknown faces: 0; 25 annotated frames -> `eval/demo_frames/`; exit code 0
- Reports: `eval/demo_report.csv`, `eval/demo_report.html` (masked only)

## REDIS (STEP 2)
- Docker Desktop started (server 29.7.2), `docker compose up -d redis` OK,
  plus host-published container: `docker run -d -p 6379:6379 redis:7-alpine`
- `docker exec ai-video-attendance-redis-1 redis-cli ping` -> **PONG**
- host `127.0.0.1:6379` -> **+PONG**; `app.redis_client.ping()` -> **True**
- Note: `docker-compose.yml` does not publish 6379 to the host, hence the
  second container for host-side runs.

## RESOLUTION (eval/resolution_table.py, measured)
WhatsApp video (848x478, 8 frames): 1.0->14 faces (min width 52.1px),
0.75->13 (39.4px), 0.50->11 (26.3px), 0.35/0.25/0.15->0 faces.
test_multi.mp4 (640x640): 1.0->10 (55.7px), 0.5->10 (27.8px),
0.35->8 (39.0px), 0.25->8 (28.0px), 0.15->2 (26.8px).

## PRIVACY
- All real registration numbers/names replaced by synthetic values repo-wide
  (13 files); the roster that bound real registrations to photos was deleted
  (moved to `%TEMP%\opencode\roster.local.csv.REMOVED-false-binding`).

### PLACEHOLDER GALLERY FINDING (2026-10-04) - why accuracy was never real
- `data/students/1..6` are NOT students: folder 1&4 = Peggy Whitson,
  2&5 = Barack Obama, 3&6 = Joe Biden (24 unique MD5s = crops of 3 public
  figures). Origin: `scripts/check_accuracy.py` downloads them into
  `%TEMP%/opencode/accuracy`. They are `0.9998` across 1<->4 / 2<->5 / 3<->6
  because they are the same person - which is what made the demo read as
  "identity pairs".
- Consequence: two real registration numbers had been bound to those celebrity
  photos in two gitignored local files. Fix: deleted `data/roster.local.csv`
  and added an explicit `--sample` mode to `eval/enroll_and_calibrate.py`
  (idempotent sample ids `SAMPLE001..003`, name `Sample Identity N`,
  section `SAMPLE`). `demo_real.db` re-enrolled; 22 real student face crops
  moved to `%TEMP%\opencode\quarantine_real_faces` and `eval/review/` removed.
- 22 real student face crops moved to `%TEMP%\opencode\quarantine_real_faces`
  (above); `eval/review/` no longer exists.

### HANDOFF ARCHIVE - ai-video-attendance-handoff.zip (final, verified)
- Built by `%TEMP%/opencode/build_handoff.py`: **106 files / 596182 bytes**
  (zipped 207583). `tar -tf` lists 106 entries.
- Excludes: `.git .venv __pycache__ *.db *.env *.jpg *.png *.mp4 *.npy *.onnx
  *.joblib *.zip`, `demo_report.*`, `import_report*.csv`, `test_bulk*.csv`,
  all of `data/` except `README.md` + `ground_truth.example.csv` + `.gitkeep`.
- Extracted to `%TEMP%\opencode\extracted` and scanned
  (`%TEMP%/opencode/scan.py`): `SCANNED_FILES=106  PII_MATCHES=0  MEDIA_FILES=0`.
  Scan = case-insensitive match on the 8 real registration numbers, the 7 real
  names, private-key/API-key markers, any 10+ digit run, plus a media sweep.

## SETUP (STEP 5) - clean checkout, two real failures, two fixes
Run: `%TEMP%\opencode\clean_checkout` (106 files) -> `scripts\setup_dev.ps1`.

1. **FAIL** `pip install -r requirements-dev.txt ... ResolutionImpossible`:
   `kombu[redis]==5.6.0/5.6.1/5.6.2 and redis==7.4.1`.
   Root cause: `celery[redis]` pulls `kombu[redis]`, which requires
   `redis>=4.5.2,<6.5`; this repo is verified on `redis==7.4.1`.
   **Fix (one):** dropped the extra -> `celery==5.6.3` (the extra only ever
   adds `redis`, which is already pinned; runtime unchanged).
2. **FAIL** `ModuleNotFoundError: No module named 'cryptography'` at
   `alembic upgrade head` (via `app/models.py` -> `app/crypto.py`).
   Root cause: nothing in `requirements.txt` pulls it in - it is only ever
   required by extras nobody installs (`PyJWT[crypto]`, `celery[auth]`,
   `insightface[gui]`, `redis[ocsp]`), so it existed only incidentally in the
   dev machine's global site-packages.
   **Fix (one):** declared `cryptography==50.0.2` (the verified version).
3. Third run: **all 7 steps OK, exit 0** -> `imports OK`, key generated,
   `Running upgrade 0002_add_periods -> 0003_alerts`, `schema is at head`,
   `buffalo_l` 5 weights present, `68 passed, 1 skipped in 6.37s`, `SETUP OK`.
   Re-run of the same script: `SCRIPT_EXIT=0` (idempotent).

### Gotcha: `-q` is doubled and hides the summary
`pyproject.toml` `addopts = "-q"`, so `pytest -q` == `-qq`, which **suppresses
the `N passed` line entirely** (exit code stays correct). Every command in the
docs now uses a single quiet level: `python -m pytest` ->
`69 passed, 1 warning in 4.90s`, exit 0.

## EXTRACTED-ARCHIVE VERIFICATION (STEP 7) - final numbers
- `scripts\setup_dev.ps1` from `%TEMP%\opencode\extracted`: **exit 0**, all
  7 steps, `68 passed, 1 skipped in 6.21s`, `SETUP OK`.
- `pytest` from that directory: `rootdir: ...\opencode\extracted`,
  **`68 passed, 1 skipped in 4.18s`, exit 0**.
  `SKIPPED [1] tests\test_pipeline_e2e.py:52: test video missing at
  ...\extracted\tests\assets\test_multi.mp4 - run: python scripts/make_test_video.py`
  (the clip is intentionally not shipped; the floor is > 51).
- Repo baseline unchanged after all of the above:
  `69 passed, 1 warning in 4.90s`, exit 0;
  `pytest --collect-only` -> `69 tests collected in 0.13s`,
  `test_quality.py` **18**, `test_regression.py` **8**.

## NEXT
- Accuracy on the 7 real students is still **UNMEASURED** - waiting on the
  user to download their photos into a local folder (Drive links are
  auth-gated, HTTP 401), then enroll WITHOUT `--sample` and run
  `eval/diag_threshold.py <video>` for the number.
- Roadmap: 8 prioritised tasks with one-line acceptance tests in
  `eval/ROADMAP.md` (task 1 = real-footage accuracy, task 2 = real gallery).
- Not exercised in this environment: live RTSP camera, GPU Compose stack.
- Cybercrime project files remain in the temp backup - do not delete.
