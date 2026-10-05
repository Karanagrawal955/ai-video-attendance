# Roadmap — prioritised, 8 tasks

Each task has **one line of acceptance** that can be executed and pasted.
Priority order reflects risk: the biggest unknowns come first.

---

## 1. Measure recognition accuracy on REAL footage of enrolled students  `P0`

The single most important open question. Today's only end-to-end number comes
from a *synthetic* clip built out of the enrollment photos themselves, so it
measures plumbing, not recognition.

* Enrol real people, film them with a real camera, label every appearance in
  `data/ground_truth.csv` (format: `data/ground_truth.example.csv`).
* Extend `scripts/check_accuracy.py` with a `--ground-truth <csv>` mode — today
  it only evaluates its own curated placeholder asset set (it downloads
  public-figure images into `%TEMP%/opencode/accuracy`), which is why no real
  accuracy number exists yet.
* Report per-person precision/recall and flag anyone with < 3 frames as
  "not enough footage".

**Accept:** `python scripts/check_accuracy.py --ground-truth data/ground_truth.csv`
exists, exits 0, and prints a per-person precision/recall table with no
"not enough footage" row left unexplained — its raw output pasted into
`eval/PROGRESS.md`.

---

## 2. Enrol the real roster instead of the placeholder gallery  `P0`

The shipped `data/students/*` folders hold placeholder public-figure images, so
the gallery currently uses synthetic `SAMPLE###` ids. Real registration numbers
must never be bound to placeholder photos.

* Put each student's 3–5 photos in `data/students/<reg>/`,
  write `data/roster.local.csv` (`folder,registration_no,section`),
  run `python eval/enroll_and_calibrate.py` **without** `--sample`.

**Accept:** `python eval/enroll_and_calibrate.py` exits 0 and every roster row
prints `placeholder=False pattern_ok=True`, with no `SAMPLE` id in the database.

---

## 3. Recalibrate the threshold at gallery size N ≥ 50  `P0`

The measured different-person maximum (0.1246) came from only ~75–384 pairs.
With 50 enrollees the pair count explodes and **some** impostor pair will score
higher — the threshold must be re-derived, not assumed.

**Accept:** a pasted `python eval/enroll_and_calibrate.py` run at N ≥ 50 showing
`different-person max < RECOGNITION_THRESHOLD < same-person min`, followed by
`python eval/diag_threshold.py <real-video>` printing `matches accepted: 0`
for footage containing nobody enrolled.

---

## 4. Validate the live RTSP ingestion path  `P1`

`frame_reader`, reconnect/backoff and per-camera slot scheduling are unit-tested
but no physical camera has ever been attached.

**Accept:** `python scripts/seed.py --camera --name "Main Entrance" --type entry --rtsp rtsp://<cam>/stream1`
then `GET /cameras` reports `healthy` continuously for **10 minutes**, with at
least one `RecognitionLog` row written and a forced stream drop recovered within
`RECONNECT_MAX_DELAY_S`.

---

## 5. Verify the GPU Docker Compose stack end-to-end  `P1`

`Dockerfile` and `docker-compose.yml` (postgres, redis, api, worker, inference)
are provided but were never run on GPU hardware in this environment.

**Accept:** `docker compose up -d --build` then
`curl -s localhost:8000/health` → `ok` and
`curl -s localhost:8000/system/gpu-status` reports `CUDAExecutionProvider`
active, and `docker compose ps` shows all five services `healthy`.

---

## 6. Unknown / low-confidence review queue  `P1`

Faces scoring below the threshold currently vanish into a log. Teachers need to
see them and decide.

* `GET /review/queue`, `POST /review/{id}/resolve` with actions
  accept (assign to student) / reject / flag.
* Queue everything below `RECOGNITION_THRESHOLD`, plus 0.25–0.40 as
  "low confidence".

**Accept:** `python -m pytest tests/test_review.py -q` passes and asserts that a
face with score 0.30 appears in the queue, becomes assigned after
`POST /review/{id}/resolve`, and then matches that student on the next run.

---

## 7. Disputes, RBAC and data retention  `P2`

* Dispute flow: `POST /attendance/{id}/dispute` → audit trail → accept/reject.
* Roles: admin / teacher / student enforced as JWT claims + middleware.
* Retention job: purge face crops (30 d), embeddings (1 y), logs (90 d) with a
  right-to-erasure endpoint.

**Accept:** `python -m pytest tests/test_disputes.py tests/test_rbac.py -q`
passes, including a case where a *student* token gets **403** on
`DELETE /students/{id}`, and a dry-run of the retention job reports the exact
row counts it would delete without removing them.

---

## 8. Scale the matcher past a linear scan  `P2`

`EmbeddingIndex` brute-forces every stored vector. Fine for tens of students,
unproven for hundreds across many cameras.

**Accept:** `python scripts/benchmark.py --identities 10000` reports p95
`idx.match()` latency **< 10 ms** and identical identities/scores to the linear
baseline on a 1000-vector equivalence fixture.

---

## Completed since the last review

| Item | Status |
|---|---|
| Encryption key rotation script (`scripts/rotate_encryption_key.py`) | **done** — re-encrypts in place, `--dry-run` supported |
| Production hard-fail for `EMBEDDING_ENCRYPTION_KEY` (no JWT fallback) | **done** — asserted by `tests/test_quality.py` |
| One-command setup (`scripts/setup_dev.ps1` / `.sh`) + pinned deps | **done** |
| `scripts/download_models.py` (weights fetched, never committed) | **done** |
| Single threshold + margin + duplicate-identity rule, measured | **done** — `tests/test_regression.py` (8 tests) |
| Quality suite for the 9 required cases | **done** — `tests/test_quality.py` (18 tests) |

## Ground rules for everything above

* Test count must not go **down** (currently **69**).
* Privacy by design: no real names, registration numbers, photos, databases,
  embeddings, videos or keys in logs, reports or archives.
* Thresholds change only with a pasted measurement next to the value.
* The minimum-frames rule (3 frames / 10 s) stays as-is unless explicitly
  re-decided.
