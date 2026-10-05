"""Multi-frame confirmation, box tracking and one-mark-per-student ledger
(Tasks 6, 7 and 8).

The live camera pipeline keeps its own K-of-N state in
``app.pipeline.camera_task._confirm_pass``; this module is the standalone,
pure-Python equivalent used by ``scripts/test_cctv.py`` for offline video
runs, so that recognition cannot be marked from a single accidental frame.

Defaults follow the documented minimum-frames rule: **3 frames within 10 s**.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

__all__ = [
    "IdentityConfirmator",
    "BoxTracker",
    "AttendanceLedger",
    "UNKNOWN",
]

UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------- confirmation
@dataclass
class _TrackObservations:
    samples: deque = field(default_factory=deque)  # of (student_id|None, ts)
    confirmed_sid: int | None = None


class IdentityConfirmator:
    """K-of-N confirmation over a tracked face (Task 6).

    ``observe()`` records one recognition result for a track and returns the
    student id only on the transition to *confirmed* - i.e. when at least
    ``k`` of the last ``n`` observations agree on the same known identity and
    all of them happened within ``window_s`` seconds.  Everything else
    (single-frame matches, flickering identities, unknowns) stays
    unconfirmed and therefore unmarked.
    """

    def __init__(self, k: int = 3, n: int = 5, window_s: float = 10.0):
        if k < 1 or n < 1:
            raise ValueError("k and n must be >= 1")
        if k > n:
            raise ValueError(f"k ({k}) cannot exceed n ({n})")
        self.k = int(k)
        self.n = int(n)
        self.window_s = float(window_s)
        self._tracks: dict[int, _TrackObservations] = {}

    # -- introspection helpers (used by tests and reports) ------------------
    @property
    def confirmed(self) -> dict[int, int]:
        """track_id -> confirmed student id (confirmed tracks only)."""
        return {
            tid: t.confirmed_sid
            for tid, t in self._tracks.items()
            if t.confirmed_sid is not None
        }

    def known_observations(self, track_id: int) -> int:
        t = self._tracks.get(track_id)
        if t is None:
            return 0
        return sum(1 for sid, _ in t.samples if sid is not None)

    def last_observation(self, track_id: int) -> int | None:
        """Last known student id seen on this track (None = unknown/none)."""
        t = self._tracks.get(track_id)
        if t is None or not t.samples:
            return None
        return t.samples[-1][0]

    # -- main entry point ---------------------------------------------------
    def observe(
        self, track_id: int, student_id: int | None, ts: float, score: float = 0.0
    ) -> int | None:
        """Record an observation.  Returns a student id on confirmation."""
        del score  # kept for call-site clarity / future weighting
        t = self._tracks.setdefault(track_id, _TrackObservations())
        t.samples.append((student_id, ts))
        while len(t.samples) > self.n:
            t.samples.popleft()
        cutoff = ts - self.window_s
        while t.samples and t.samples[0][1] < cutoff:
            t.samples.popleft()

        if t.confirmed_sid is not None:
            return None  # already confirmed; caller decides what happens next
        if student_id is None or len(t.samples) < self.k:
            return None

        votes = [sid for sid, _ in t.samples if sid == student_id]
        if len(votes) >= self.k:
            t.confirmed_sid = student_id
            return student_id
        return None


# ---------------------------------------------------------------- tracking
def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class BoxTracker:
    """Greedy IoU tracker so a person keeps one identity across frames.

    Needed because confirmation must accumulate observations of the *same*
    face; without tracking, two people side by side would share one buffer.
    """

    def __init__(self, iou_threshold: float = 0.30, max_age_s: float = 2.0):
        self.iou_threshold = float(iou_threshold)
        self.max_age_s = float(max_age_s)
        self._next_id = 1
        self._live: dict[int, dict] = {}  # track_id -> {bbox, last_seen}

    @property
    def live_count(self) -> int:
        return len(self._live)

    def update(self, bboxes: list[list[float]], ts: float) -> list[int]:
        """Associate this frame's boxes with existing tracks.

        Returns one track id per input box (aligned by index).
        """
        for tid in [t for t, s in self._live.items() if ts - s["last_seen"] > self.max_age_s]:
            del self._live[tid]

        assigned: dict[int, int] = {}  # track_id -> box index
        unmatched_boxes = list(range(len(bboxes)))

        # score every (track, box) pair by IoU, greedily take the best
        pairs = sorted(
            (
                (_iou(self._live[tid]["bbox"], bboxes[bi]), tid, bi)
                for tid in self._live
                for bi in unmatched_boxes
            ),
            key=lambda p: p[0],
            reverse=True,
        )
        used_tracks: set[int] = set()
        for iou, tid, bi in pairs:
            if iou < self.iou_threshold:
                break
            if tid in used_tracks or bi not in unmatched_boxes:
                continue
            used_tracks.add(tid)
            unmatched_boxes.remove(bi)
            assigned[tid] = bi

        result: list[int | None] = [None] * len(bboxes)
        for tid, bi in assigned.items():
            result[bi] = tid
            self._live[tid]["bbox"] = bboxes[bi]
            self._live[tid]["last_seen"] = ts

        for bi in unmatched_boxes:
            tid = self._next_id
            self._next_id += 1
            self._live[tid] = {"bbox": bboxes[bi], "last_seen": ts}
            result[bi] = tid

        out: list[int] = []
        for bi, tid in enumerate(result):  # every box always gets a track id
            if tid is None:  # defensive: never happens in practice
                tid = self._next_id
                self._next_id += 1
                self._live[tid] = {"bbox": bboxes[bi], "last_seen": ts}
            out.append(int(tid))
        return out


# ------------------------------------------------------------------ ledger
class AttendanceLedger:
    """Marks each student at most once (Task 7).

    ``mark()`` returns True only the first time a student id is recorded, so
    later sightings of the same person can never create a second attendance
    row - the database-level entry/exit rules in
    ``app.services.attendance.process_recognition`` stay the second line of
    defence.
    """

    def __init__(self) -> None:
        self._marked: dict[int, dict] = {}

    def mark(
        self, student_id: int, ts: float, score: float = 0.0, track_id: int | None = None
    ) -> bool:
        """Returns True if this sighting created the (first) attendance mark."""
        if student_id in self._marked:
            return False
        self._marked[student_id] = {
            "ts": ts,
            "score": score,
            "track_id": track_id,
        }
        return True

    def is_marked(self, student_id: int) -> bool:
        return student_id in self._marked

    def get(self, student_id: int) -> dict | None:
        return self._marked.get(student_id)

    @property
    def marked_ids(self) -> set[int]:
        return set(self._marked)

    def __len__(self) -> int:
        return len(self._marked)
