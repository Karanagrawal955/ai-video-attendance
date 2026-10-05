"""Tests for the Google Form import, multi-frame confirmation and evaluation
metrics added for Tasks 2/5/6/7/8/10/14/15."""

from __future__ import annotations

import numpy as np
import pytest

from app.confirm import AttendanceLedger, BoxTracker, IdentityConfirmator, UNKNOWN
from app.evalmetrics import (
    ResultLabel,
    classify_pair,
    identity_metrics,
    load_ground_truth,
)
from app.formcsv import (
    FormImportError,
    FrameCandidate,
    classify_download,
    extract_drive_id,
    largest_identity_cluster,
    parse_form_csv,
    select_reference_frames,
)


# --------------------------------------------------------------------- CSV
def _write(tmp_path, content: str, name: str = "form.csv"):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_parse_messy_headers_and_preserves_values(tmp_path) -> None:
    path = _write(
        tmp_path,
        '"Timestamp","name","your pic ","reg no."\n'
        '"2026/10/02 3:26:17 PM"," Test Student ","'
        "https://drive.google.com/u/0/open?usp=forms_web&id=1AbCdEfGhIjKlMnOpQrStUvWxYz1234567890\""
        ',"900000001"\n',
    )
    rows = parse_form_csv(path)
    assert len(rows) == 1
    row = rows[0]
    assert row.errors == []
    assert row.name == "Test Student"          # exact, only outer space removed
    assert row.registration_no == "900000001"  # exact
    assert "drive.google.com" in row.video_ref


def test_parse_rejects_csv_without_required_column(tmp_path) -> None:
    path = _write(tmp_path, '"name","link"\n"Test","https://x"\n')
    with pytest.raises(FormImportError) as exc:
        parse_form_csv(path)
    assert "reg" in str(exc.value)


def test_parse_missing_file_and_empty_file(tmp_path) -> None:
    with pytest.raises(FormImportError):
        parse_form_csv(tmp_path / "nope.csv")
    empty = _write(tmp_path, "", name="empty.csv")
    with pytest.raises(FormImportError):
        parse_form_csv(empty)


def test_invalid_registration_number_is_row_error(tmp_path) -> None:
    path = _write(
        tmp_path,
        '"name","reg no.","your pic "\n"Too Short","12","http://drive.google.com/x"\n',
    )
    row = parse_form_csv(path)[0]
    assert not row.ok
    assert any("invalid registration number" in e for e in row.errors)


def test_duplicate_registration_number_flagged_not_dropped(tmp_path) -> None:
    path = _write(
        tmp_path,
        '"name","reg no.","your pic "\n'
        '"First","900000001","http://drive.google.com/a"\n'
        '"Second","900000001","http://drive.google.com/b"\n',
    )
    rows = parse_form_csv(path)
    assert rows[0].ok
    assert any("duplicate" in e for e in rows[1].errors)


def test_blank_lines_are_skipped(tmp_path) -> None:
    path = _write(
        tmp_path,
        '"name","reg no.","your pic "\n'
        '"A Student","900000001","http://drive.google.com/a"\n'
        '"","",""\n',
    )
    assert len(parse_form_csv(path)) == 1


# ------------------------------------------------------------- drive links
def test_extract_drive_id_shapes() -> None:
    assert (
        extract_drive_id(
            "https://drive.google.com/u/0/open?usp=forms_web&id=1AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"
        )
        == "1AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"
    )
    assert (
        extract_drive_id("https://drive.google.com/file/d/1abcDEF1234567890xyz/view?usp=sharing")
        == "1abcDEF1234567890xyz"
    )
    assert extract_drive_id("") is None
    assert extract_drive_id("https://example.com/photo.jpg") is None


def test_classify_download_bodies() -> None:
    assert classify_download("text/html; charset=utf-8", b"<html>Sign in</html>") == "html"
    assert classify_download("text/html", b"<!DOCTYPE html><html>") == "html"
    assert classify_download("video/mp4", b"") == "video"
    assert classify_download("", b"\x00\x00\x00\x18ftypmp42") == "video"
    assert classify_download("image/jpeg", b"\xff\xd8\xff\xe0") == "image"
    assert classify_download("application/json", b"{}") == "other"


# -------------------------------------------------------- frame selection
def _cand(i: int, emb: list[float], **kw) -> FrameCandidate:
    defaults = dict(det_score=0.9, sharpness=300.0, brightness=120.0,
                    face_width=120.0, face_area=14400.0)
    defaults.update(kw)
    return FrameCandidate(index=i, jpeg=b"jpg", embedding=emb, **defaults)


def test_select_reference_frames_picks_best_and_skips_near_duplicates() -> None:
    base = [0.1] * 8 + [0.0] * 8  # placeholder 16-dim vectors (any dim works)
    near_dup = [0.1000001] * 8 + [0.0] * 8
    other = [0.0] * 8 + [0.9] * 8
    candidates = [
        _cand(0, base, det_score=0.7),
        _cand(1, near_dup, det_score=0.99),  # near-identical to #0 but sharper
        _cand(2, other, det_score=0.8),
    ]
    selected, counts = select_reference_frames(candidates, limit=5, duplicate_sim=0.90)
    assert len(selected) == 2
    assert counts["rejected_near_duplicate"] == 1
    assert {s.index for s in selected} == {1, 2}  # best of the duplicate pair kept


def test_select_reference_frames_rejects_blurry_dark_and_low_det() -> None:
    candidates = [
        _cand(0, [1.0, 0.0], sharpness=5.0),      # blurry
        _cand(1, [0.0, 1.0], brightness=8.0),     # too dark
        _cand(2, [0.6, 0.8], det_score=0.2),      # weak detection
        _cand(3, [0.8, 0.6]),
    ]
    selected, counts = select_reference_frames(candidates, limit=5, min_det_score=0.6)
    assert [c.index for c in selected] == [3]
    assert counts["rejected_blurry"] == 1
    assert counts["rejected_brightness"] == 1
    assert counts["rejected_low_det"] == 1


def test_select_reference_frames_respects_limit() -> None:
    # orthogonal one-hot embeddings so only the limit (not dedup) applies
    candidates = [
        _cand(i, [1.0 if j == i else 0.0 for j in range(8)]) for i in range(8)
    ]
    selected, counts = select_reference_frames(candidates, limit=5)
    assert len(selected) == 5
    assert counts["rejected_near_duplicate"] == 0


def test_identity_cluster_keeps_only_the_main_person() -> None:
    """A second person in the enrollment video must never reach the gallery."""
    subject = [1.0, 0.0]
    subject2 = [0.9, 0.1]      # same person, other pose (cos ~0.99)
    stranger = [0.0, 1.0]      # different person (cos 0.0)
    candidates = [
        _cand(0, subject),
        _cand(1, stranger, det_score=0.99),  # sharper, but a different face
        _cand(2, subject2),
        _cand(3, stranger),
    ]
    kept, counts = largest_identity_cluster(candidates, identity_sim=0.40)
    assert [c.index for c in kept] == [0, 2]   # temporal order
    assert counts["clusters"] == 2
    assert counts["kept"] == 2
    assert counts["dropped_other_identity"] == 2


def test_identity_cluster_passes_a_single_person_video_through() -> None:
    candidates = [_cand(i, [1.0, 0.1 * i]) for i in range(4)]
    kept, counts = largest_identity_cluster(candidates, identity_sim=0.40)
    assert len(kept) == 4
    assert counts["clusters"] == 1
    assert counts["dropped_other_identity"] == 0


def test_identity_cluster_drops_unverifiable_frames() -> None:
    candidates = [
        _cand(0, None),          # no embedding - cannot verify who this is
        _cand(1, [1.0, 0.0]),
        _cand(2, [1.0, 0.05]),
    ]
    kept, counts = largest_identity_cluster(candidates, identity_sim=0.40)
    assert [c.index for c in kept] == [1, 2]
    assert counts["dropped_no_embedding"] == 1


def test_identity_cluster_without_any_embedding_keeps_everything() -> None:
    candidates = [_cand(i, None) for i in range(3)]
    kept, counts = largest_identity_cluster(candidates, identity_sim=0.40)
    assert len(kept) == 3
    assert counts["clusters"] == 0
    assert counts["kept"] == 3


# ------------------------------------------------------- confirmation (T5/T6)
def test_single_frame_match_never_confirms() -> None:
    """One accidental frame must not mark attendance."""
    conf = IdentityConfirmator(k=3, n=5, window_s=10.0)
    assert conf.observe(track_id=1, student_id=7, ts=1.0) is None
    assert conf.confirmed == {}


def test_k_of_n_confirmation() -> None:
    conf = IdentityConfirmator(k=3, n=5, window_s=10.0)
    assert conf.observe(1, 7, 1.0) is None
    assert conf.observe(1, None, 2.0) is None   # unknown frame in between
    assert conf.observe(1, 7, 3.0) is None
    assert conf.observe(1, 7, 4.0) == 7         # 3rd match -> confirmed
    assert conf.confirmed == {1: 7}
    # does not fire again for the same track
    assert conf.observe(1, 7, 5.0) is None


def test_confirmation_needs_k_matches_within_window() -> None:
    conf = IdentityConfirmator(k=3, n=5, window_s=10.0)
    conf.observe(1, 7, 0.0)
    conf.observe(1, 7, 1.0)
    conf.observe(1, 7, 50.0)  # outside the 10 s window -> old votes pruned
    assert conf.confirmed == {}


def test_flickering_identities_do_not_confirm() -> None:
    """Alternating identities: nobody reaches 3 of the last 5, so no mark."""
    conf = IdentityConfirmator(k=3, n=5, window_s=10.0)
    for i, sid in enumerate([7, 8, 7, 8]):
        conf.observe(1, sid, float(i))
    assert conf.confirmed == {}


def test_unknown_observations_never_confirm() -> None:
    conf = IdentityConfirmator(k=3, n=5, window_s=10.0)
    for i in range(5):
        assert conf.observe(1, None, float(i)) is None
    assert conf.confirmed == {}


def test_invalid_confirm_parameters_rejected() -> None:
    with pytest.raises(ValueError):
        IdentityConfirmator(k=5, n=3)


# ------------------------------------------------------------- tracking (T6)
def test_tracker_keeps_one_id_for_a_moving_face() -> None:
    tracker = BoxTracker(iou_threshold=0.3, max_age_s=2.0)
    first = tracker.update([[10, 10, 60, 60]], ts=0.0)
    second = tracker.update([[14, 12, 64, 62]], ts=0.2)
    assert first == second
    assert tracker.live_count == 1


def test_tracker_assigns_new_id_to_a_second_face() -> None:
    tracker = BoxTracker()
    ids = tracker.update([[10, 10, 60, 60], [300, 10, 360, 60]], ts=0.0)
    assert len(set(ids)) == 2
    # both keep their identities on the next frame
    again = tracker.update([[300, 10, 360, 60], [10, 10, 60, 60]], ts=0.2)
    assert set(again) == set(ids)


def test_tracker_expires_a_disappeared_face() -> None:
    tracker = BoxTracker(max_age_s=1.0)
    first = tracker.update([[10, 10, 60, 60]], ts=0.0)
    tracker.update([], ts=5.0)  # face gone long enough to expire
    after = tracker.update([[10, 10, 60, 60]], ts=5.1)
    assert after[0] != first[0]


# --------------------------------------------------------- ledger (T7)
def test_ledger_marks_each_student_once() -> None:
    ledger = AttendanceLedger()
    assert ledger.mark(1, ts=10.0, score=0.8) is True
    assert ledger.mark(1, ts=11.0, score=0.9) is False  # duplicate sighting
    assert ledger.mark(2, ts=12.0, score=0.7) is True
    assert len(ledger) == 2
    assert ledger.marked_ids == {1, 2}
    assert ledger.get(1)["ts"] == 10.0  # first sighting kept


def test_ledger_is_marked() -> None:
    ledger = AttendanceLedger()
    assert not ledger.is_marked(3)
    ledger.mark(3, ts=1.0)
    assert ledger.is_marked(3)


# --------------------------------------------------------- evaluation (T10)
def test_classify_pair_labels() -> None:
    assert classify_pair("900000001", "900000001") == ResultLabel.CORRECT
    assert classify_pair(UNKNOWN, UNKNOWN) == ResultLabel.CORRECT_REJECTION
    assert classify_pair(UNKNOWN, "900000001") == ResultLabel.FALSE_ACCEPT
    assert classify_pair("900000001", UNKNOWN) == ResultLabel.FALSE_REJECT
    assert classify_pair("900000001", "900000002") == ResultLabel.WRONG


def test_identity_metrics_hand_built() -> None:
    pairs = [
        ("900000001", "900000001"),  # correct
        ("900000001", "900000001"),  # correct
        ("900000001", UNKNOWN),        # false reject
        (UNKNOWN, UNKNOWN),              # correct rejection
        (UNKNOWN, UNKNOWN),              # correct rejection
        (UNKNOWN, "900000002"),        # false accept (FAR)
        ("900000002", "900000001"),  # wrong identity
    ]
    m = identity_metrics(pairs)
    assert m["total"] == 7
    assert m["correct"] == 4
    assert m["incorrect"] == 3
    assert m["accuracy"] == pytest.approx(4 / 7)
    assert m["false_accepts"] == 1
    assert m["false_rejects"] == 1
    assert m["wrong_identities"] == 1
    assert m["expected_known"] == 4
    assert m["expected_unknown"] == 3
    # micro precision: TP=2, FP = 1 false accept + 1 wrong identity
    assert m["precision"] == pytest.approx(2 / 4)
    # recall: TP=2, FN = 1 false reject + 1 wrong identity
    assert m["recall"] == pytest.approx(2 / 4)
    assert m["f1"] == pytest.approx(0.5)
    assert m["far"] == pytest.approx(1 / 3)
    assert m["frr"] == pytest.approx(1 / 4)
    assert m["unknown_rejection_rate"] == pytest.approx(2 / 3)


def test_identity_metrics_perfect_and_empty() -> None:
    perfect = identity_metrics([(UNKNOWN, UNKNOWN), ("a", "a")])
    assert perfect["accuracy"] == 1.0
    assert perfect["far"] == 0.0
    empty = identity_metrics([])
    assert empty["accuracy"] == 0.0
    assert empty["total"] == 0


def test_load_ground_truth(tmp_path) -> None:
    path = _write(tmp_path, "frame,reg\n0,900000001\n1,UNKNOWN\n2,900000002\n")
    gt = load_ground_truth(path)
    assert gt == {0: ["900000001"], 1: [UNKNOWN], 2: ["900000002"]}
    with pytest.raises(FileNotFoundError):
        load_ground_truth(tmp_path / "missing.csv")
    bad = _write(tmp_path, "nope,reg\n1,x\n", name="bad.csv")
    with pytest.raises(ValueError):
        load_ground_truth(bad)


def test_load_ground_truth_multi_face_frames(tmp_path) -> None:
    """Two rows for one frame = two faces, labelled left to right."""
    path = _write(
        tmp_path,
        "frame,reg\n10,900000001\n10,UNKNOWN\n11,900000002\n",
    )
    assert load_ground_truth(path) == {
        10: ["900000001", UNKNOWN],
        11: ["900000002"],
    }


# ------------------------------------------------- unknown rejection (T5/T8)
def test_low_similarity_probe_is_rejected_as_unknown() -> None:
    """Never force a face onto the nearest known student."""
    from app.matching import EMBEDDING_DIM, EmbeddingIndex

    rng = np.random.default_rng(4)
    known = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    known /= np.linalg.norm(known)
    stranger = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    stranger /= np.linalg.norm(stranger)

    idx = EmbeddingIndex(threshold=0.40)
    idx.flat = known.reshape(1, -1)
    idx.row_student_ids = np.array([1], dtype=np.int64)
    idx.student_count = 1
    idx.embedding_count = 1
    idx.loaded_at = 0.0
    idx._version = "test"
    idx._last_check = float("inf")
    idx._last_refresh = 0.0

    hit = idx.match(known.tolist())
    assert hit.student_id == 1
    miss = idx.match(stranger.tolist())
    assert miss.student_id is None
    assert miss.score < 0.40


def test_index_without_students_never_matches() -> None:
    from app.matching import EmbeddingIndex

    idx = EmbeddingIndex(threshold=0.40)
    idx.flat = None
    idx.row_student_ids = None
    result = idx.match([0.1] * 512)
    assert result.student_id is None
