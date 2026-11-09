"""Research-only alignment prototype tests. No training data is produced."""
import json
import math
import sys
import wave

sys.path.insert(0, "src_id")

import numpy as np

from align import (
    AlignerUnavailable,
    align_wav,
    durations_energy_draft,
    durations_from_ctc,
    frame_energy_db,
    n_frames_of,
    read_wav_16k_mono,
    resample_to_16k,
    split_for_text,
    try_ctc_alignment,
)

PILOT = "Hari ini saya pergi ke pasar untuk membeli sayur segar."
WAV = "./supertonic_work/output/sample_id_f2_fp32.wav"
MANIFEST = "experiments/text/domain_manifest.jsonl"


def test_ctc_raises_without_model():
    try:
        try_ctc_alignment(WAV, (1, 2, 3))
        raise AssertionError("expected AlignerUnavailable")
    except AlignerUnavailable:
        pass


def test_energy_draft_sum_and_phones():
    pcm, sr = read_wav_16k_mono(WAV)
    pcm16 = resample_to_16k(pcm, sr)
    n = n_frames_of(pcm16)
    e = frame_energy_db(pcm16)
    from frontend import phonemize
    pr = phonemize(PILOT)
    dur, s, en, alarms = durations_energy_draft(pr.symbols, n, e)
    assert sum(dur) == en - s
    assert "energy-draft-not-for-training" in alarms
    for tok, d in zip(pr.symbols, dur):
        if tok in ("<bos>", "<eos>"):
            assert d == 0
        elif tok not in ("<space>", ".", ",", "?", "!", ";", ":"):
            assert d >= 1, tok


def test_split_preserved():
    # Pilot text is not in manifest -> 'pilot'; a manifest test text -> 'test'
    assert split_for_text(PILOT, MANIFEST) == "pilot"
    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    test_row = next(r for r in rows if r["split"] == "test")
    assert split_for_text(test_row["text"], MANIFEST) == "test"


def test_durations_from_ctc_rounding():
    tokens = ["<bos>", "AH0", "<space>", "K", "<eos>"]
    # two phones: [0.00,0.10] and [0.20,0.35], n=40 frames
    dur, s, e = durations_from_ctc(tokens, [(0.0, 0.10), (0.20, 0.35)], 40)
    assert sum(dur) == 40
    assert dur[1] == 10
    # gap 10 frames -> <space>; trailing 5 frames -> last phone (no final punct)
    assert dur[2] == 10
    assert dur[3] == 20


def test_align_wav_research_only():
    r = align_wav(WAV, PILOT, manifest_path=MANIFEST, allow_energy_draft=True)
    assert r.split == "pilot"
    assert r.method == "energy-draft"
    assert r.to_dict()["research_only_do_not_train"] is True
    r.validate()


if __name__ == "__main__":
    test_ctc_raises_without_model()
    test_energy_draft_sum_and_phones()
    test_split_preserved()
    test_durations_from_ctc_rounding()
    test_align_wav_research_only()
    print("all align tests passed")
