#!/usr/bin/env python
"""Recognition accuracy check (TAR / FAR / rank-1 / detection / latency).

Enrolls identities through the production endpoint, runs a probe gallery of
transformed and cross-session photos through the shared GPU inference
service, and scores every probe with the *production* matcher
(``app.matching.EmbeddingIndex``):

  * genuine vs impostor cosine-similarity distributions
  * TAR / FAR at the configured threshold and across a threshold sweep
  * closed-set rank-1 identification + open-set accept/reject decisions
  * unknown-face rejection rate
  * detection coverage and RPC latency on this machine

Requires the stack up (redis + inference + api) and the same ``DATABASE_URL``
as the API so the index reads the enrolled students:

    python scripts/check_accuracy.py

Assets default to ``$TEMP/opencode/accuracy`` (override with
``ACCURACY_ASSETS``) and are downloaded automatically on first run.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
USER = os.environ.get("ADMIN_USERNAME", "admin")
PASS = os.environ.get("ADMIN_PASSWORD", "admin")
ASSETS = Path(
    os.environ.get("ACCURACY_ASSETS")
    or Path(os.environ.get("TEMP", "/tmp")) / "opencode" / "accuracy"
)

DOWNLOADS = {
    "obama_ex.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
    "biden_ex.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg",
    "two_people.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/two_people.jpg",
    "obama_wiki2.jpg": "https://commons.wikimedia.org/wiki/Special:FilePath/Official_portrait_of_Barack_Obama.jpg?width=900",
    "biden_wiki1.jpg": "https://commons.wikimedia.org/wiki/Special:FilePath/Official_portrait_of_Vice_President_Joe_Biden.jpg?width=900",
    # unenrolled identities -> unknown / rejection tests
    "unknown_einstein.jpg": "https://commons.wikimedia.org/wiki/Special:FilePath/Albert_Einstein_Head.jpg?width=800",
    "unknown_nr.jpg": "https://commons.wikimedia.org/wiki/Special:FilePath/Ronald_Reagan_portrait.jpg?width=800",
}


# --------------------------------------------------------------------- assets
def ensure_assets() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        path = ASSETS / name
        if path.exists() and path.stat().st_size > 5000:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "accuracy-check/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < 5000:
                print(f"  ! {name}: suspiciously small ({len(data)} B), skipped")
                continue
            path.write_bytes(data)
            print(f"  + downloaded {name} ({len(data)} B)")
        except Exception as exc:  # noqa: BLE001 - optional assets
            print(f"  ! {name}: {exc}")


def jpg(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buf.tobytes()


def imread(name: str) -> np.ndarray | None:
    path = ASSETS / name
    if not path.exists():
        return None
    return cv2.imread(str(path))


# ----------------------------------------------------------------- transforms
def transformed(img: np.ndarray, tag: str) -> list[tuple[str, np.ndarray]]:
    h, w = img.shape[:2]
    out: list[tuple[str, np.ndarray]] = [(f"{tag}:orig", img)]
    out.append((f"{tag}:hflip", cv2.flip(img, 1)))
    for s in (0.9, 0.8):
        ch, cw = int(h * s), int(w * s)
        y, x = (h - ch) // 2, (w - cw) // 2
        out.append((f"{tag}:crop{int(s * 100)}", img[y : y + ch, x : x + cw]))
    out.append((f"{tag}:bright", cv2.convertScaleAbs(img, alpha=1.3, beta=18)))
    out.append((f"{tag}:dark", cv2.convertScaleAbs(img, alpha=0.7, beta=-18)))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1.0)
    out.append((f"{tag}:rot12", cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)))
    out.append((f"{tag}:blur", cv2.GaussianBlur(img, (7, 7), 2)))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    if ok:
        out.append((f"{tag}:jpeg30", cv2.imdecode(buf, cv2.IMREAD_COLOR)))
    return out


# --------------------------------------------------------------------- api i/o
def login(client: httpx.Client) -> str:
    r = client.post(f"{BASE}/auth/token", json={"username": USER, "password": PASS}, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def delete_by_reg(client: httpx.Client, headers: dict, reg: str) -> None:
    data = client.get(f"{BASE}/students", params={"q": reg, "limit": 100}, headers=headers, timeout=10).json()
    for s in data.get("items", []):
        if s.get("registration_no") == reg:
            client.delete(f"{BASE}/students/{s['id']}", headers=headers, timeout=10)


def enroll(client: httpx.Client, headers: dict, reg: str, name: str, photos: list[bytes]) -> dict:
    files = [("photos", (f"{reg}_{i}.jpg", p, "image/jpeg")) for i, p in enumerate(photos)]
    r = client.post(
        f"{BASE}/students",
        data={"name": name, "registration_no": reg, "section": "ACC"},
        files=files,
        headers=headers,
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ reporting
def pct(x: float) -> str:
    return f"{100 * x:6.1f}%"


def dist(label: str, xs: list[float]) -> str:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return f"  {label:<26} n=0"
    return (
        f"  {label:<26} n={a.size:<3} mean={a.mean():.3f}  std={a.std():.3f}  "
        f"min={a.min():.3f}  max={a.max():.3f}"
    )


def main() -> int:
    print("== accuracy check ===============================================")
    ensure_assets()

    from app.config import settings  # noqa: E402
    from app.inference import client as ic  # noqa: E402
    from app.matching import EmbeddingIndex  # noqa: E402

    threshold = float(settings.recognition_threshold)
    client = httpx.Client()
    headers = {"Authorization": f"Bearer {login(client)}"}

    # ------------------------------------------------------------ enrollment
    print("\n-- enrollment (production endpoint -> GPU RPC) --")
    from skimage import data as skdata

    sources: dict[str, np.ndarray | None] = {
        "demo": cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR),
        "obama": imread("obama_wiki2.jpg"),   # official portrait (session A)
        "biden": imread("biden_wiki1.jpg"),
    }
    regs = {"demo": ("DEMO001", "Demo Student"), "obama": ("PROBE_OBA", "Probe Obama"), "biden": ("PROBE_BID", "Probe Biden")}

    sid_of: dict[str, int] = {}
    for key, img in sources.items():
        if img is None:
            print(f"  ! {key}: portrait missing, skipping identity")
            continue
        reg, name = regs[key]
        delete_by_reg(client, headers, reg)
        h, w = img.shape[:2]
        ch, cw = int(h * 0.9), int(w * 0.9)
        photos = [
            jpg(img),
            jpg(cv2.flip(img, 1)),
            jpg(img[(h - ch) // 2 : (h - ch) // 2 + ch, (w - cw) // 2 : (w - cw) // 2 + cw]),
        ]
        try:
            student = enroll(client, headers, reg, name, photos)
        except httpx.HTTPStatusError as exc:
            print(f"  ! {key}: enrollment FAILED {exc.response.status_code} {exc.response.text[:200]}")
            continue
        sid_of[key] = int(student["id"])
        print(f"  + {name:<14} reg={reg:<9} id={student['id']} embeddings={student['embedding_count']}")

    # ------------------------------------------------------------ build index
    try:
        index = EmbeddingIndex(threshold=threshold)
        index.refresh()
    except Exception as exc:  # noqa: BLE001
        print(f"\nFATAL: cannot load embedding index - point DATABASE_URL at the API's DB\n  {exc}")
        return 2
    print(f"\n  index: {index.student_count} students / {index.embedding_count} embeddings, threshold={threshold:.2f}")
    if index.flat is None:
        print("FATAL: index is empty - no students with embeddings in this DB")
        return 3
    sid_to_key = {sid: k for k, sid in sid_of.items()}

    # ---------------------------------------------------------- probe gallery
    gallery: list[tuple[str, str | None, bytes]] = []  # (label, identity_key|None, jpeg)

    if sources.get("demo") is not None:
        gallery += [(f"demo:{t}", "demo", jpg(im)) for t, im in transformed(sources["demo"], "astro")]
    # cross-session probes: candid photos (sessions B/C), NOT the enrolled portraits
    for key, fname in (("obama", "obama_ex.jpg"), ("biden", "biden_ex.jpg")):
        img = imread(fname)
        if img is None:
            print(f"  ! probe photo missing: {fname}")
            continue
        keep = {"orig", "hflip", "crop90", "bright", "rot12", "blur"}
        tag = key
        gallery += [(f"{key}:{t}", key, jpg(im)) for t, im in transformed(img, tag) if t.split(":")[1] in keep]

    unknown_count = 0
    for fname, utag in (("unknown_einstein.jpg", "einstein"), ("unknown_nr.jpg", "reagan")):
        img = imread(fname)
        if img is None:
            continue
        keep = {"orig", "hflip", "crop90", "rot12"}
        for t, im in transformed(img, utag):
            if t.split(":")[1] in keep:
                gallery.append((f"unknown:{t}", None, jpg(im)))
                unknown_count += 1

    # two_people.jpg: TWO enrolled identities in one frame (Obama left, Biden
    # right) -> detection test + two more cross-session genuine probes.
    detect_faces: list[tuple[str, np.ndarray]] = []
    detect_ms = None
    twp = imread("two_people.jpg")
    if twp is not None:
        t0 = time.perf_counter()
        det = ic.infer_frames([jpg(twp)])
        detect_ms = (time.perf_counter() - t0) * 1000
        rows = (det.get("results") or [[]])[0] or []
        faces_sorted = sorted(
            (f for f in rows if f.get("embedding")), key=lambda f: f["bbox"][0]
        )[:2]
        for i, f in enumerate(faces_sorted):
            key = "obama" if i == 0 else "biden"  # leftmost = Obama in this image
            detect_faces.append((f"{key}:twp_face{i + 1}", np.asarray(f["embedding"], dtype=np.float32)))
        print(f"  two_people.jpg -> {len(rows)} face(s) detected, detect RPC {detect_ms:.1f} ms")

    # ------------------------------------------------------------- evaluation
    print(f"\n-- probes: {len(gallery)} images + {len(detect_faces)} direct faces from 2-face frame --")
    genuine: list[float] = []
    impostor: list[float] = []
    records: list[dict] = []
    embed_ms: list[float] = []
    det_fail: list[str] = []

    def per_student(emb: np.ndarray) -> dict[int, float]:
        p = np.asarray(emb, dtype=np.float32).ravel()
        p = p / max(float(np.linalg.norm(p)), 1e-9)
        sims = index.flat @ p
        out: dict[int, float] = {}
        for sid, sc in zip(index.row_student_ids, sims):
            sid = int(sid)
            out[sid] = max(out.get(sid, -1.0), float(sc))
        return out

    def evaluate(label: str, key: str | None, emb: np.ndarray) -> None:
        scores = per_student(emb)
        top_sid = max(scores, key=scores.get)
        top = scores[top_sid]
        prod = index.match(emb)
        true_sid = sid_of.get(key) if key else None
        rec = {
            "label": label,
            "key": key,
            "top_sid": top_sid,
            "top": top,
            "prod_sid": prod.student_id,
            "true_score": scores.get(true_sid) if true_sid is not None else None,
            "correct": true_sid is not None and prod.student_id == true_sid,
        }
        records.append(rec)
        if true_sid is not None:
            genuine.append(rec["true_score"])
            impostor.extend(sc for sid, sc in scores.items() if sid != true_sid)
        else:
            impostor.extend(scores.values())

    for label, key, payload in gallery:
        t0 = time.perf_counter()
        try:
            res = ic.embed_images([payload])
        except Exception as exc:  # noqa: BLE001
            det_fail.append(f"{label} (rpc: {exc})")
            continue
        embed_ms.append((time.perf_counter() - t0) * 1000)
        row = res[0] if res else None
        emb = row.get("embedding") if isinstance(row, dict) else None
        if not emb:
            det_fail.append(label)
            continue
        evaluate(label, key, np.asarray(emb, dtype=np.float32))

    for label, emb in detect_faces:
        evaluate(label, label.split(":")[0], emb)

    # ---------------------------------------------------------------- report
    print("\n== results ==")
    print("\nsimilarity distributions (cosine, production embeddings)")
    print(dist("genuine (same identity)", genuine))
    print(dist("impostor (wrong pair)", impostor))
    if genuine and impostor:
        gmin, imax = min(genuine), max(impostor)
        print(f"  separation: impostor max {imax:.3f} vs genuine min {gmin:.3f} -> "
              f"{'CLEAN MARGIN' if imax < gmin else 'OVERLAP'} (margin {gmin - imax:+.3f})")

    print("\ngenuine per identity (max over that student's 3 enrolled photos)")
    for key in ("demo", "obama", "biden"):
        vals = [r["true_score"] for r in records if r["key"] == key and r["true_score"] is not None]
        note = "transformation variants (single source)" if key == "demo" else "cross-session photos (candid + 2-person frame)"
        print(f"  {regs[key][1]:<14} {note}")
        if vals:
            print(f"    n={len(vals)}  mean={np.mean(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}")
        else:
            print("    n=0")

    print("\nrank-1 closed-set identification (labeled probes)")
    lab = [r for r in records if r["key"]]
    hits = sum(1 for r in lab if r["top_sid"] == sid_of[r["key"]])
    print(f"  accuracy: {hits}/{len(lab)} = {pct(hits / max(len(lab), 1))}")

    correct = sum(1 for r in records if r["correct"])
    labeled_total = sum(1 for r in records if r["key"])
    false_reject = sum(1 for r in records if r["key"] and r["prod_sid"] is None)
    wrong_id = sum(1 for r in records if r["key"] and r["prod_sid"] is not None and not r["correct"])
    unk = [r for r in records if r["key"] is None]
    unk_accept = sum(1 for r in unk if r["prod_sid"] is not None)
    unk_correct = len(unk) - unk_accept
    overall = correct + unk_correct

    print(f"\nopen-set decisions @ threshold {threshold:.2f} (production matcher)")
    print(f"  correct accepts : {correct}/{labeled_total}   (enrolled person matched)")
    print(f"  false rejects   : {false_reject}   (enrolled person not matched)")
    print(f"  wrong identity  : {wrong_id}   (matched, but to the WRONG student)")
    print(f"  unknown accepted: {unk_accept}/{len(unk)}   (unenrolled person admitted)")
    print(f"  overall correct : {overall}/{len(records)} = {pct(overall / max(len(records), 1))}")

    print("\nthreshold sweep")
    print("   thr |   TAR   | FAR(pair) | open-set correct | unk accept")
    for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        tar = sum(1 for s in genuine if s >= t) / max(len(genuine), 1)
        far = sum(1 for s in impostor if s >= t) / max(len(impostor), 1)
        oc = sum(1 for r in lab if r["top"] >= t and r["top_sid"] == sid_of[r["key"]])
        ua = sum(1 for r in unk if r["top"] >= t)
        print(f"  {t:.2f} | {pct(tar)} | {pct(far)} |     {pct(oc / max(len(lab), 1))}      | {ua}/{len(unk)}")

    print("\ndetection & latency")
    print(f"  probes with no face found: {len(det_fail)}" + (f" -> {det_fail}" if det_fail else ""))
    if detect_ms is not None:
        print(f"  detect RPC (2 faces in frame): {detect_ms:.1f} ms")
    if embed_ms:
        a = np.asarray(embed_ms)
        print(f"  embed RPC per image: mean={a.mean():.1f} ms  p50={np.percentile(a, 50):.1f}  "
              f"p95={np.percentile(a, 95):.1f}  n={a.size}")

    print("\nenrollment consistency (pairwise cosine within a student's photos)")
    if index.flat is not None:
        for sid in sorted(set(index.row_student_ids.tolist())):
            rows = index.flat[index.row_student_ids == sid]
            if rows.shape[0] < 2:
                continue
            sims = rows @ rows.T
            vals = sims[np.triu_indices(rows.shape[0], k=1)]
            tag = sid_to_key.get(sid, str(sid))
            print(f"  {tag:<10} photos={rows.shape[0]}  mean={vals.mean():.3f}  min={vals.min():.3f}")

    print("\ncaveats: small curated gallery (transformation robustness for 'demo',")
    print("cross-session for obama/biden); no RTSP/encoding in the path. Point")
    print("ACCURACY_ASSETS + regs at your own dataset for production numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
