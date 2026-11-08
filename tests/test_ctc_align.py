"""Check the CTC alignment path offline with a tiny hand-built tokenizer stub.

No model weights needed: verifies read-back logic, allocation, gap handling and
duration invariants on synthetic runs. Run: python -I tests/test_ctc_align.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))

import ctc_align  # noqa: E402


def test_allocate_exact():
    for weights, total in [([1, 1, 1], 10), ([2.4, 1.4, 0.2], 7), ([1], 3)]:
        out = ctc_align._allocate(weights, total)
        assert sum(out) == total and all(v >= 0 for v in out)
    out = ctc_align._allocate([1, 1, 1], 2)  # fewer slots than phones
    assert sum(out) == 2 and sorted(out) == [0, 1, 1]


def test_word_group_spans():
    toks = ["<bos>", "P", "AH1", "G", "IY0", "<space>", "P", "AH1", "G", "IY0", ".", "<eos>"]
    g = ctc_align.word_group_spans(toks)
    assert g == [[1, 2, 3, 4], [6, 7, 8, 9]]
    # <space> must split words; punctuation never forms a group
    assert ctc_align.word_group_spans(["<bos>", "S", "<space>", "S", ",", "<eos>"]) == [[1], [3]]


def test_group_words():
    runs = [("s", 0, 5), ("e", 5, 9), ("l", 9, 12), ("a", 12, 18), ("m", 18, 24), ("a", 24, 30),
            ("t", 30, 33), ("｜", 33, 34), ("p", 34, 38), ("a", 38, 42), ("g", 42, 46), ("i", 46, 50)]
    words = ctc_align.ctc_grouped(runs, {"|", "｜"})
    assert [w for w, _, _ in words] == ["selamat", "pagi"]
    assert words[0][1] == 0 and words[1][2] == 50


def test_resample_preserves_duration():
    pcm = np.sin(np.arange(44100) * 2 * np.pi * 440 / 44100).astype(np.float32) * 10000
    out = ctc_align.resample_to_16k(pcm.astype(np.int16), 44100)
    assert abs(len(out) - 16000) <= 2
    assert out.dtype == np.int16


def test_energy_grid_matches_features():
    pcm = np.zeros(1600, dtype=np.int16)
    e = ctc_align.frame_energy_db(pcm)
    assert e.shape == (10,)


def test_weighted_allocation_sums_to_word_span():
    # "yang" (4 letters) but 3 phones: weights path must still sum exactly.
    weights = [ctc_align._weight(p) for p in ("Y", "AH0", "NG")]
    out = ctc_align._allocate(weights, 23)
    assert sum(out) == 23 and all(x >= 1 for x in out)


def test_alignment_validate_rejects_bad_sum():
    a = ctc_align.Alignment(tokens=["<bos>", "S", "<eos>"], token_ids=[0, 1, 2],
                            durations=[0, 5, 0], start_frame=0, end_frame=7)
    try:
        a.validate()
    except AssertionError:
        pass
    else:
        raise AssertionError("validate accepted mismatched frame sum")


def test_match_letters_tolerates_merge_and_variants():
    # CTC heard "dijital" (one merge/variant); letters still line up.
    ctc = list("programstudibisnisdijital")
    spans = [(i * 2, i * 2 + 2) for i in range(len(ctc))]
    text = "programstudibisnisdigital"
    matched, cov = ctc_align.match_letters(ctc, spans, text)
    assert len(matched) == len(text)
    assert cov >= 0.9
    assert matched[0] is not None and matched[-1] is not None


def test_word_bounds_interpolate_unmatched_words():
    words = ["satu", "dua"]
    span = [None] * 7  # everything unmatched -> interpolation must cover all
    bounds, alarms = ctc_align.word_bounds_from_letters(span, words, 10, 30)
    assert bounds[0][0] >= 10 and bounds[-1][1] <= 30
    assert bounds[0][1] <= bounds[1][0]
    assert alarms and all(a.startswith("interpolated:") for a in alarms)


def test_to_frames_keeps_none():
    out = ctc_align.to_frames([(0, 5), None, (10, 12)], 0.02, 400)
    assert out[0] == (0, 10) and out[1] is None and out[2] == (20, 24)
    assert ctc_align.to_frames([(0, 999)], 0.02, 100)[0] == (0, 200)


def test_word_bounds_use_matched_letters():
    words = ["ab", "cd"]
    span = [(0, 4), (4, 9), (12, 20), (20, 26)]
    bounds, alarms = ctc_align.word_bounds_from_letters(span, words, 0, 40)
    assert bounds[0] == (0, 9) and bounds[1] == (12, 26)
    assert alarms == []


def test_align_english_requires_pivo_frontend():
    """lang='en' must not silently fall back to the Indonesian frontend."""
    try:
        ctc_align.align(Path("nonexistent.wav"), "hello", processor=None, model=None,
                        lang="en", en_phonemize=None)
    except ctc_align.AlignError as exc:
        assert "en_phonemize" in str(exc)
    else:
        raise AssertionError("align accepted lang='en' without a frontend")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASS")
