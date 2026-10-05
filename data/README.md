# `data/` — local, never shipped

Everything in this directory except this file and `ground_truth.example.csv`
is **gitignored and excluded from every archive**: photos, rosters, databases,
embeddings, crops, videos.

```
data/
  README.md                    ← you are here (tracked)
  ground_truth.example.csv     ← format template, FAKE data only (tracked)
  .gitkeep
  ── everything below is local-only ──────────────────────────────
  students/<folder>/photo_N.jpg    reference photos (3–5 per person)
  roster.local.csv                 REAL registration numbers → photo folders
  attendance.db                    SQLite when running without Docker
  crops/, embeddings/, videos/     runtime biometric artefacts
```

Verify any of it is really ignored:

```bash
git check-ignore -v data/students/1/photo_0.jpg data/attendance.db data/roster.local.csv
```

---

## Photo folder layout

`photo_path_or_folder` in a bulk-import CSV is resolved **relative to
`DATA_DIR`** (default `./data`):

```
data/students/STU100001/photo_0.jpg
data/students/STU100001/photo_1.jpg
data/students/STU100001/photo_2.jpg
...
```

* **3–5 photos per person** (`MIN_REFERENCE_PHOTOS` / `MAX_REFERENCE_PHOTOS`)
* exactly one face per photo
* face ≥ 80 px wide and ≥ 800 px², sharp, well lit, near-frontal

Rows that fail are reported with a reason, never silently dropped.

---

## `roster.local.csv` (create it yourself)

Header + one row per photo folder:

```csv
folder,registration_no,section
STU100001,STU100001,CSE-A
STU100002,STU100002,CSE-B
```

Rules enforced by `python eval/enroll_and_calibrate.py`:

1. It is **required** — without it the script exits with instructions rather
   than guessing a mapping.
2. Every registration number must match `REGISTRATION_NO_PATTERN`
   (`^[A-Za-z0-9_-]{6,12}$` by default).
3. Any value that looks like a placeholder (`STU…`, `TEST…`, `SAMPLE…`,
   `DEMO…`, …) causes an **abort** — a real roster must hold real ids.
4. The script prints the folder↔folder identity table and **flags folders that
   hold the same person** (cosine ≥ `DUPLICATE_IDENTITY_SIM`), then enrols only
   one record per distinct identity.

### ⚠️ Placeholder photos

The folders used for pipeline development contain **placeholder public-figure
images, not students**. Those are enrolled with:

```bash
python eval/enroll_and_calibrate.py --sample     # assigns SAMPLE001, SAMPLE002, ...
```

so that a **real registration number is never bound to a placeholder photo**.
If you run the default mode against those folders, the placeholder check stops
the run.

---

## `ground_truth.example.csv`

Template for labelling what *should* happen in a clip (used by roadmap task 1 —
`scripts/check_accuracy.py` does not consume it yet):

| column | meaning |
|---|---|
| `person_id` | `P1`, `P2`, … the masked identity label |
| `name` | display label — keep it synthetic in any file you share |
| `registration_no` | the enrolled id for that person |
| `section` | class/section |
| `photo_folder` | which photo folder holds their reference images |
| `clip_id` | which video clip the row refers to |
| `frame` | frame index where that person is visible |

One row **per visible frame** (or per sampled frame). Copy the file to
`data/ground_truth.csv` and fill it in — that copy stays local.

---

## Privacy rules for anything in this directory

* Never commit, zip or paste: photos, `roster.local.csv`, `*.db`, embeddings,
  crops, videos, `.env`.
* Reports and logs identify people as `P1…Pn` plus **last-3 registration
  digits** only.
* Before sharing an archive: run a case-insensitive PII scan (**0 matches**
  expected) and a media scan (**0 files** expected), then extract it and run
  `scripts/setup_dev.ps1` there to prove it works.
