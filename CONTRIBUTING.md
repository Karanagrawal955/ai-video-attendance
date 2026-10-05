# Contributing

## Setup

```bash
bash scripts/setup_dev.sh          # Windows: scripts\setup_dev.ps1
```

That gives you a pinned environment and runs the whole suite. Re-running it is
safe — every step is idempotent.

```bash
python -m pytest                      # must stay green (addopts already = -q)
python -m pytest --collect-only -q -o addopts= | tail -3
```

> Do not add a second `-q`: `pyproject.toml` `addopts = "-q"` already provides
> one, and `-qq` suppresses the `N passed` summary line entirely.

The suite needs **no** Docker, Redis, PostgreSQL or GPU: Redis is faked, Celery
`send_task` is stubbed, and each run gets its own temp SQLite database.

## Ground rules

1. **Never commit personal data.** No `.env`, no databases, no reference photos,
   no face crops, no embeddings, no videos, no model weights. `.gitignore`
   already covers them; run `git check-ignore -v <path>` if you are unsure.
2. **Never put a real registration number or name in a tracked file**, a log
   line, a report, or an archive. Use `P1…Pn` and last-3 digits only.
3. **Placeholder photos get placeholder ids** (`SAMPLE###`), never a real
   registration number.
4. **One threshold.** Recognition behaviour comes from `RECOGNITION_THRESHOLD`,
   `RECOGNITION_MARGIN` and `DUPLICATE_IDENTITY_SIM` in `app/config.py`. Do not
   hard-code a score anywhere else, and do not change a threshold without
   re-measuring — the measured basis is documented next to each value.
5. **Do not change the minimum-frames rule** (3 frames / 10 s) without an
   explicit decision; it is part of the acceptance criteria.
6. **Raw outputs, not estimates.** If you claim a number, paste the command and
   its output. If output degrades, stop and record it in `eval/PROGRESS.md`.

## Code style

* Python 3.11+, type hints on new code, `from __future__ import annotations`.
* Match the surrounding style; keep functions small and boring.
* Tests are mandatory for new behaviour. A bug fix needs a regression test that
  fails before the fix.
* Configuration belongs in `app/config.py` + `.env.example` — document the new
  key in the README configuration table too.

## Adding an endpoint

1. Route in `app/api/<resource>.py`, schemas in `app/schemas.py`.
2. Business logic in `app/services/` — routes stay thin.
3. Tests in `tests/test_<resource>.py`, using the `client` + `auth_header`
   fixtures.
4. Document the route in the README API table.

## Changing thresholds or matching

`app/matching.py` applies, in order: threshold → margin → duplicate-identity
skip. Any change must keep `tests/test_regression.py` green and must update the
derivation comments in `app/config.py` and `docs/ARCHITECTURE.md`.

## Verifying your archive

Before handing a zip to anyone:

```bash
# 1. listing
tar -tf your-archive.zip

# 2. PII scan - must print 0
#    (case-insensitive search for every real registration number / name)

# 3. media scan - must print 0
#    *.jpg *.jpeg *.png *.mp4 *.mov *.npy *.onnx *.db *.env

# 4. it must actually work from scratch
#    extract to a clean folder and run scripts/setup_dev.ps1 there
```

## Pull request checklist

- [ ] `python -m pytest` passes and the count did not go *down*
- [ ] No personal data, media, databases, keys or weights staged
- [ ] New settings documented in `.env.example` and the README table
- [ ] Threshold changes backed by a pasted measurement
- [ ] Docs updated (`README.md`, `docs/ARCHITECTURE.md`, `eval/ROADMAP.md`)
