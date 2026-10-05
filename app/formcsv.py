"""Google Forms CSV import helpers (Task 2 + Task 13).

Real Forms exports have messy headers (``"your pic "``, ``"reg no."``, mixed
case, extra columns), so headers are matched on a normalised key.  Everything
in this module is pure - no database, no model, no network - so it can be
unit tested directly (``tests/test_form_pipeline.py``).

Nothing here ever invents data: a row that cannot be parsed keeps its errors
and the caller reports it instead of guessing.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "FormImportError",
    "FormRow",
    "parse_form_csv",
    "extract_drive_id",
    "drive_direct_url",
    "classify_download",
    "FrameCandidate",
    "select_reference_frames",
]


class FormImportError(Exception):
    """Fatal CSV problem (missing file, unreadable, missing columns)."""


# --------------------------------------------------------------------- rows
@dataclass
class FormRow:
    """One student row from the form.  Values are kept exactly as filed."""

    row_number: int  # 1-based data row (header is row 0)
    name: str
    registration_no: str
    video_ref: str  # Google Drive URL (or a local path when provided)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def short_reg(self) -> str:
        """Last 3 characters only - safe to print in shared logs."""
        return self.registration_no[-3:] if self.registration_no else "***"


# ------------------------------------------------------------------ headers
def _norm(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (header or "").strip().lower())


_REG_KEYS = {
    "regno",
    "regnumber",
    "regnum",
    "reg",
    "registrationnumber",
    "registrationno",
    "registration",
    "rollno",
    "rollnumber",
    "roll",
    "enrollmentno",
    "enrollmentnumber",
    "enrolmentno",
}
_NAME_KEYS = {"name", "studentname", "yourname", "fullname", "student"}
_LINK_KEYS = {
    "yourpic",
    "pic",
    "photo",
    "video",
    "link",
    "drivelink",
    "videolink",
    "facesample",
    "sample",
    "yourpicvideo",
    "yourvideo",
    "facevideo",
}


def _lookup(values: dict[str, str], header: str) -> str:
    """Fetch a cell by its original header (tolerates whitespace drift)."""
    if header in values:
        return values[header]
    wanted = header.strip()
    for key, val in values.items():
        if key.strip() == wanted:
            return val
    return ""


def _classify(header: str) -> str | None:
    """Map one CSV header onto ``name`` / ``reg`` / ``link`` / ``None``."""
    key = _norm(header)
    if not key:
        return None
    if key in _REG_KEYS or key.startswith("reg") or key.startswith("registration"):
        return "reg"
    if key in _NAME_KEYS or (key.endswith("name") and "frame" not in key):
        return "name"
    if key in _LINK_KEYS:
        return "link"
    if any(tok in key for tok in ("drive", "video", "pic", "photo")):
        return "link"
    return None


def parse_form_csv(
    path: str | Path,
    *,
    reg_pattern: str | None = None,
    require_link: bool = True,
    check_duplicates: bool = True,
) -> list[FormRow]:
    """Parse a Google Forms response CSV.

    Raises :class:`FormImportError` only for fatal problems (missing file,
    empty file, missing required columns).  Row-level problems are recorded in
    :attr:`FormRow.errors` so one bad row never hides the others.
    """
    path = Path(path)
    if not path.exists():
        raise FormImportError(f"CSV not found: {path}")
    if path.stat().st_size == 0:
        raise FormImportError(f"CSV is empty: {path}")

    if reg_pattern is None:
        from .config import settings

        reg_pattern = settings.registration_no_pattern

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise FormImportError(f"CSV has no header row: {path}")
        columns: dict[str, str] = {}  # role -> original header
        for raw in reader.fieldnames:
            role = _classify(raw)
            if role and role not in columns:
                columns[role] = raw
        missing = [r for r in ("name", "reg") if r not in columns]
        if require_link and "link" not in columns:
            missing.append("link")
        if missing:
            raise FormImportError(
                f"CSV is missing required column(s) {missing}; "
                f"found headers: {list(reader.fieldnames)}"
            )

        rows: list[FormRow] = []
        for idx, raw in enumerate(reader, start=1):
            # keys stay EXACTLY as the header was written ("your pic " has a
            # trailing space); only the values are stripped.
            values = {(k if k is not None else ""): (v or "") for k, v in raw.items()}
            if not any(v.strip() for v in values.values()):
                continue  # blank line
            name = _lookup(values, columns.get("name", "")).strip()
            reg = _lookup(values, columns.get("reg", "")).strip()
            link = _lookup(values, columns.get("link", "")).strip() if "link" in columns else ""

            errors: list[str] = []
            if not name:
                errors.append("missing name")
            if not reg:
                errors.append("missing registration number")
            elif not re.match(reg_pattern, reg):
                errors.append(
                    f"invalid registration number format (must match {reg_pattern!r})"
                )
            if require_link and not link:
                errors.append("missing video/photo link")

            rows.append(
                FormRow(
                    row_number=idx,
                    name=name,
                    registration_no=reg,
                    video_ref=link,
                    errors=errors,
                )
            )

    if not rows:
        raise FormImportError(f"CSV contains no data rows: {path}")

    if check_duplicates:
        seen: dict[str, int] = {}
        for row in rows:
            if not row.registration_no:
                continue
            if row.registration_no in seen:
                row.errors.append(
                    f"duplicate registration number (first seen in row "
                    f"{seen[row.registration_no]})"
                )
            else:
                seen[row.registration_no] = row.row_number
    return rows


# ------------------------------------------------------------------- drive
_DRIVE_ID_RE = re.compile(r"(?:[?&]id=|/file/d/|/folders/)([A-Za-z0-9_-]{10,})")


def extract_drive_id(ref: str) -> str | None:
    """Pull the file id out of any common Drive URL shape (or None)."""
    ref = (ref or "").strip()
    if not ref:
        return None
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", ref):
        return ref  # already a bare id
    m = _DRIVE_ID_RE.search(ref)
    return m.group(1) if m else None


def drive_direct_url(ref: str) -> str | None:
    """Direct-download URL for a Drive link (None when the ref has no id)."""
    file_id = extract_drive_id(ref)
    if not file_id:
        return None
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def classify_download(content_type: str, prefix: bytes) -> str:
    """Classify an HTTP download body.

    Returns one of ``video`` / ``image`` / ``html`` / ``other``.  ``html`` is
    how a permission-gated Drive file shows up (sign-in wall or "you need
    access") - it is *never* treated as a usable face sample.
    """
    ct = (content_type or "").lower()
    head = (prefix or b"")[:512].lstrip().lower()
    if ct.startswith(("video/", "application/octet-stream")):
        return "video"
    if ct.startswith("image/"):
        return "image"
    if ct.startswith("text/html") or head.startswith((b"<!doctype html", b"<html")):
        return "html"
    if head[:4] in (b"\x00\x00\x00\x18", b"\x00\x00\x00\x1c", b"\x00\x00\x00\x20") or head[
        4:8
    ] == b"ftyp":
        return "video"  # ISO-BMFF (mp4/mov) sniffed by magic bytes
    return "other"


# -------------------------------------------------------------- frame choice
@dataclass
class FrameCandidate:
    """One usable single-face frame extracted from an enrollment video."""

    index: int  # frame index inside the source video
    jpeg: bytes  # JPEG-encoded full frame (what gets stored)
    embedding: list[float] | None = None  # 512-d ArcFace embedding
    det_score: float = 0.0
    sharpness: float = 0.0  # Laplacian variance of the face crop
    brightness: float = 0.0  # mean gray value of the face crop
    face_width: float = 0.0
    face_area: float = 0.0

    @property
    def quality(self) -> float:
        """Rank: detection confidence, nudged up by a sharp crop."""
        sharp = min(self.sharpness, 1500.0) / 1500.0
        return float(self.det_score) * (0.7 + 0.3 * sharp)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def largest_identity_cluster(
    candidates: list[FrameCandidate],
    *,
    identity_sim: float = 0.40,
) -> tuple[list[FrameCandidate], dict[str, int]]:
    """Keep only the frames that show a single person (identity gate).

    Enrollment videos sometimes contain a second (or third) person - a
    roommate walking past, someone handing over the camera, a reflection.
    A frame of a stranger must never reach the gallery, otherwise that
    stranger's embedding gets stored against a real registration number and
    later *accepts* them (and their classmates) as that student.

    Candidates are walked in ``quality`` order and greedily assigned to the
    first cluster whose anchor embedding is >= ``identity_sim`` cosine away
    (ArcFace: same person in one recording scores well above 0.4, two
    different people in this footage score below 0.2).  Only the largest
    cluster is returned - that is the person occupying most usable frames,
    i.e. the subject the video was recorded for.  Frames without an
    embedding cannot be verified and are dropped.

    Returns ``(kept, counts)`` where ``counts`` explains the decision:
    ``clusters`` (how many identities were seen), ``kept``,
    ``dropped_other_identity``, ``dropped_no_embedding``.
    """
    counts = {
        "candidates": len(candidates),
        "clusters": 0,
        "kept": 0,
        "dropped_other_identity": 0,
        "dropped_no_embedding": 0,
    }

    with_emb = [c for c in candidates if c.embedding]
    if not with_emb:
        # nothing to verify against - leave the decision to the caller
        counts["clusters"] = 0
        counts["kept"] = len(candidates)
        counts["dropped_no_embedding"] = 0
        return list(candidates), counts

    counts["dropped_no_embedding"] = len(candidates) - len(with_emb)

    ordered = sorted(with_emb, key=lambda c: c.quality, reverse=True)
    clusters: list[list[FrameCandidate]] = []
    anchors: list[np.ndarray] = []
    for cand in ordered:
        vec = np.asarray(cand.embedding, dtype=np.float32)
        placed = False
        for members, anchor in zip(clusters, anchors):
            if _cosine(vec, anchor) >= identity_sim:
                members.append(cand)
                placed = True
                break
        if not placed:
            clusters.append([cand])
            anchors.append(vec)

    counts["clusters"] = len(clusters)
    # most frames wins; a tie goes to the identity that appears first in the
    # video (the person holding the camera is in front of it from frame 0)
    largest = max(
        clusters, key=lambda m: (len(m), -min(c.index for c in m))
    )
    kept = {id(m) for m in largest}
    counts["kept"] = len(largest)
    counts["dropped_other_identity"] = len(with_emb) - len(largest)

    kept_frames = [c for c in candidates if id(c) in kept]
    kept_frames.sort(key=lambda c: c.index)  # temporal order, like the selector
    return kept_frames, counts


def select_reference_frames(
    candidates: list[FrameCandidate],
    limit: int,
    *,
    duplicate_sim: float = 0.90,
    min_sharpness: float = 40.0,
    brightness_range: tuple[float, float] = (30.0, 225.0),
    min_det_score: float = 0.0,
) -> tuple[list[FrameCandidate], dict[str, int]]:
    """Pick the best ``limit`` reference frames (Task 3).

    Rejects blurry / too-dark / too-bright / low-confidence frames, then
    greedily takes the highest-quality frames while skipping near-duplicates
    (cosine >= ``duplicate_sim`` against an already chosen frame) so the
    reference set keeps pose/lighting variation instead of 5 copies of the
    same frame.

    Returns ``(selected, counts)`` where ``counts`` explains every rejection.
    """
    counts = {
        "candidates": len(candidates),
        "rejected_blurry": 0,
        "rejected_brightness": 0,
        "rejected_low_det": 0,
        "rejected_near_duplicate": 0,
        "selected": 0,
    }

    lo, hi = brightness_range
    usable: list[FrameCandidate] = []
    for c in candidates:
        if c.sharpness < min_sharpness:
            counts["rejected_blurry"] += 1
            continue
        if not (lo <= c.brightness <= hi):
            counts["rejected_brightness"] += 1
            continue
        if c.det_score < min_det_score:
            counts["rejected_low_det"] += 1
            continue
        usable.append(c)

    ordered = sorted(usable, key=lambda c: c.quality, reverse=True)
    selected: list[FrameCandidate] = []
    vectors: list[np.ndarray] = []
    for cand in ordered:
        if len(selected) >= limit:
            break
        if cand.embedding is not None:
            vec = np.asarray(cand.embedding, dtype=np.float32)
            if any(_cosine(vec, prev) >= duplicate_sim for prev in vectors):
                counts["rejected_near_duplicate"] += 1
                continue
            vectors.append(vec)
        elif selected:
            # no embedding to compare with - keep at most a couple of frames
            # that are byte-identical in index terms (defensive fallback)
            if any(cand.index == s.index for s in selected):
                counts["rejected_near_duplicate"] += 1
                continue
        selected.append(cand)

    counts["selected"] = len(selected)
    selected.sort(key=lambda c: c.index)  # store in temporal order
    return selected, counts
