#!/usr/bin/env python
"""STEP 3 / STEP 1f: data-honesty + re-enrolment + calibration.

Reads the LOCAL roster (``data/roster.local.csv`` - gitignored, never shipped),
embeds every photo folder through the ONE ``FaceEngine.infer`` path, then:

  1. prints the folder-vs-folder identity table and FLAGs folders that hold
     the same person (max cross-folder cosine >= ``duplicate_identity_sim``),
  2. verifies no registration number is a placeholder,
  3. enrolls only ONE record per distinct identity (demo = distinct people),
  4. proves the encrypt -> store -> decrypt round trip is lossless,
  5. prints the same/different score distributions used to justify the single
     recognition threshold and margin.

Privacy: output shows P-labels and last-3 registration digits only.

Two roster modes:
  (default)  read data/roster.local.csv - REAL registration numbers; the script
             ABORTS if any of them looks like a placeholder.
  --sample   data/students/* is a PLACEHOLDER gallery (public-figure dev
             photos): assign synthetic SAMPLE### ids, bind no real person.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///demo_real.db")
os.environ.setdefault("FORCE_CPU", "true")
os.environ.setdefault("FACE_QUALITY_ENABLED", "true")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.crypto import decrypt_embeddings, encrypt_embeddings  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402
from app.inference.engine import FaceEngine  # noqa: E402
from app.models import Student  # noqa: E402

PHOTO_ROOT = ROOT / "data" / "students"
ROSTER = ROOT / "data" / "roster.local.csv"

PLACEHOLDER_PATTERNS = (
    r"^(STU|TEST|DUMMY|PLACEHOLDER|EXAMPLE|MOCK|X|REG|DEMO|SAMPLE)",
    r"placeholder",
)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def is_placeholder(reg: str) -> bool:
    return any(re.search(p, reg, re.IGNORECASE) for p in PLACEHOLDER_PATTERNS)


def load_roster() -> list[dict]:
    if not ROSTER.exists():
        sys.exit(
            f"MISSING {ROSTER}\n"
            "This file holds the real registration numbers and is gitignored.\n"
            "Create it with header: folder,registration_no,section"
        )
    rows = []
    with open(ROSTER, newline="", encoding="utf-8") as fh:
        import csv

        for r in csv.DictReader(fh):
            rows.append(
                {
                    "folder": r["folder"].strip(),
                    "registration_no": r["registration_no"].strip(),
                    "section": r.get("section", "").strip() or None,
                }
            )
    if not rows:
        sys.exit(f"{ROSTER} is empty")
    return rows


def load_sample_roster() -> list[dict]:
    """Roster for the PLACEHOLDER gallery shipped with the pipeline.

    ``data/students/1..6`` hold public-figure development photos - they are
    NOT enrolled students, so they get explicitly synthetic ids (SAMPLE###)
    rather than anybody's real registration number.
    """
    rows: list[dict] = []
    for p in sorted(
        PHOTO_ROOT.iterdir(),
        key=lambda x: int(x.name) if x.name.isdigit() else 10**9,
    ):
        if p.is_dir():
            rows.append(
                {
                    "folder": p.name,
                    "registration_no": f"SAMPLE{len(rows) + 1:03d}",
                    "section": "SAMPLE",
                }
            )
    if not rows:
        sys.exit(f"no photo folders under {PHOTO_ROOT}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--sample",
        action="store_true",
        help="treat data/students/* as a PLACEHOLDER gallery: assign SAMPLE### "
             "ids instead of reading data/roster.local.csv (no real person is "
             "bound to these photos)",
    )
    args = ap.parse_args()

    sample_mode = bool(args.sample)
    roster = load_sample_roster() if sample_mode else load_roster()
    if sample_mode:
        print("!" * 78)
        print("SAMPLE GALLERY: data/students/* contains placeholder photos.")
        print("Ids below are synthetic (SAMPLE###). This gallery demonstrates")
        print("the pipeline; it does NOT represent enrolled students.")
        print("For a real gallery: put real photos + data/roster.local.csv and")
        print("rerun WITHOUT --sample.")
        print("!" * 78)

    eng = FaceEngine(settings)
    eng.warmup()

    # ---------------------------------------------------------- embed folders
    vecs: dict[str, list[np.ndarray]] = {}
    print("STEP 3 | folder inventory (photos embedded through FaceEngine.infer)")
    print("=" * 78)
    for row in roster:
        folder = row["folder"]
        photos = sorted((PHOTO_ROOT / folder).glob("*.jpg"))
        imgs = [cv2.imread(str(p)) for p in photos]
        imgs = [i for i in imgs if i is not None]
        res, tim = eng.infer(imgs)
        emb = [np.asarray(f[0]["embedding"], float) for f in res if f]
        vecs[folder] = emb
        print(
            f"  folder data/students/{folder}/  photos={len(photos)} embedded={len(emb)} "
            f"norms={[round(float(np.linalg.norm(e)), 4) for e in emb]} "
            f"det_ms={tim.det_ms:.1f} emb_ms={tim.emb_ms:.1f}"
        )

    # ------------------------------------------------- identity / duplicate map
    folders = [r["folder"] for r in roster]
    pair_max: dict[tuple[str, str], float] = {}
    for i, a in enumerate(folders):
        for b in folders[i + 1:]:
            best = max(
                (cos(va, vb) for va in vecs[a] for vb in vecs[b]), default=0.0
            )
            pair_max[(a, b)] = best

    dup_threshold = settings.duplicate_identity_sim
    same_group: dict[str, list[str]] = {f: [f] for f in folders}
    for (a, b), best in pair_max.items():
        if best >= dup_threshold:
            same_group[a].append(b)
            same_group[b].append(a)

    # connected components (union of >= threshold pairs)
    seen: set[str] = set()
    groups: list[list[str]] = []
    for f in folders:
        if f in seen:
            continue
        stack, comp = [f], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(x for x in same_group[cur] if x not in seen)
        groups.append(sorted(comp))

    reg_of = {r["folder"]: r["registration_no"] for r in roster}
    group_of = {f: i + 1 for i, g in enumerate(groups) for f in g}

    print("\nSTEP 3 | identity table (max cross-folder cosine decides duplicates)")
    print("=" * 78)
    print(f"  {'folder':>16} {'masked id':>10} {'id#':>4} {'photos':>6} {'flag':>28}")
    for row in roster:
        f = row["folder"]
        g = group_of[f]
        rep = groups[g - 1][0]
        flag = "distinct identity" if rep == f else f"DUPLICATE of folder {rep}"
        print(
            f"  {'data/students/' + f + '/':>16} {'...' + reg_of[f][-3:]:>10} "
            f"{g:>4} {len(vecs[f]):>6} {flag:>28}"
        )
    print("\n  pairwise max cosine (same-person flags):")
    for (a, b), best in sorted(pair_max.items(), key=lambda kv: -kv[1]):
        print(f"    folder {a} vs {b}: {best:.4f}"
              f"   {'SAME PERSON' if best >= dup_threshold else ''}")

    # ------------------------------------------------- placeholder verification
    print("\nSTEP 3 | registration-number placeholder check")
    print("=" * 78)
    if sample_mode:
        print("  mode=SAMPLE  (placeholders expected on purpose; see banner)")
    bad = []
    for row in roster:
        reg = row["registration_no"]
        ok = (not is_placeholder(reg)) and re.match(
            settings.registration_no_pattern, reg
        )
        print(f"  ...{reg[-3:]}  placeholder={is_placeholder(reg)}  "
              f"pattern_ok={bool(re.match(settings.registration_no_pattern, reg))}")
        if not ok:
            bad.append(reg)
    if bad and not sample_mode:
        sys.exit(
            f"ABORT: {len(bad)} registration number(s) look like placeholders. "
            "A real roster must carry real registration numbers - or run with "
            "--sample if these photos are placeholder data."
        )
    if bad:
        print(f"  {len(bad)} intentionally-synthetic id(s) kept (sample mode)")

    # ------------------------------------------------------ enrol distinct only
    Base.metadata.create_all(engine)
    db = SessionLocal()
    db.query(Student).delete()
    db.commit()

    enrolled = []
    print("\nSTEP 3 | enrolling ONE record per distinct identity")
    print("=" * 78)
    for g, group in enumerate(groups, 1):
        rep = group[0]  # lowest folder number in the identity group
        photos = sorted((PHOTO_ROOT / rep).glob("*.jpg"))
        imgs = [cv2.imread(str(p)) for p in photos]
        imgs = [i for i in imgs if i is not None]
        res, _ = eng.infer(imgs)
        emb = [np.asarray(f[0]["embedding"], float) for f in res if f]
        s = Student(
            name=(
                f"Sample Identity {rep}"
                if sample_mode
                else f"Volunteer-{rep}"
            ),
            registration_no=reg_of[rep],
            section=next(
                r["section"] for r in roster if r["folder"] == rep
            ),
            embeddings=encrypt_embeddings([e.tolist() for e in emb]),
            photo_paths=[f"{rep}/{p.name}" for p in photos][:3],
        )
        db.add(s)
        db.commit()
        db.refresh(s)
        enrolled.append((g, rep, s))
        extra = (
            f" (identity group also covers folder(s) {', '.join(x for x in group if x != rep)})"
            if len(group) > 1 else ""
        )
        print(
            f"  P{g} (...{reg_of[rep][-3:]}) <- data/students/{rep}/ "
            f"photos={len(emb)}{extra}"
        )

    # ------------------------------------------------- encryption round trip
    print("\n[encryption round-trip]")
    for g, rep, s in enrolled:
        stored = [np.asarray(v, float) for v in decrypt_embeddings(s.embeddings)]
        fresh = vecs[rep][: len(stored)]
        identical = all(
            np.allclose(a, b, atol=1e-9) for a, b in zip(stored, fresh)
        )
        print(f"  P{g}: stored={len(stored)} identical_to_fresh={identical}")
    db.close()

    # ------------------------------------------------------- calibration
    same_pairs, diff_pairs = [], []
    for gi in range(len(groups)):
        for gj in range(gi + 1, len(groups)):
            for va in vecs[groups[gi][0]]:
                for vb in vecs[groups[gj][0]]:
                    diff_pairs.append(cos(va, vb))
    for group in groups:  # photos inside one identity (incl. its duplicates)
        allv = [v for f in group for v in vecs[f]]
        for i, va in enumerate(allv):
            for vb in allv[i + 1:]:
                same_pairs.append(cos(va, vb))
    same = np.array(same_pairs)
    diff = np.array(diff_pairs)

    print("\n[score distributions from printed vectors]")
    print(f"  SAME identity    n={len(same):4d} min={same.min():.4f} "
          f"p05={np.percentile(same, 5):.4f} mean={same.mean():.4f} "
          f"max={same.max():.4f}")
    print(f"  DIFFERENT ident. n={len(diff):4d} min={diff.min():.4f} "
          f"mean={diff.mean():.4f} p95={np.percentile(diff, 95):.4f} "
          f"max={diff.max():.4f}")
    print(f"  configured: threshold={settings.recognition_threshold} "
          f"margin={settings.recognition_margin} "
          f"duplicate_identity_sim={settings.duplicate_identity_sim}")
    if same.min() <= diff.max():
        print("  NOT separable -> re-check photos")
        return 1
    lo, hi = diff.max(), same.min()
    print(f"  SEPARABLE: gap ({lo:.4f} .. {same.min():.4f}); "
          f"configured threshold {settings.recognition_threshold} lies inside: "
          f"{lo < settings.recognition_threshold < hi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
