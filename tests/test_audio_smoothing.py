import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


def test_selective_refinement_preserves_transitions_and_removes_pitch_spike():
    from render_student import refine_pivo_features

    feats = np.zeros((12, 20), dtype=np.float32)
    feats[:, 19] = 0.9
    feats[4, 14] = 1.0
    feats[4, 18] = 2.0
    feats[7:, 1] = 20.0
    feats[7, 8] = 5.0
    original = feats.copy()
    refined = refine_pivo_features(feats)
    np.testing.assert_array_equal(feats, original)
    np.testing.assert_array_equal(refined[:, :6], feats[:, :6])
    np.testing.assert_array_equal(refined[:, 19], feats[:, 19])
    assert refined[4, 14] < feats[4, 14]
    assert refined[4, 18] == 0.0
    assert refined[7, 8] == feats[7, 8]
    unvoiced = feats.copy()
    unvoiced[3:6, 19] = -0.3
    assert refine_pivo_features(unvoiced)[4, 18] == unvoiced[4, 18]
    np.testing.assert_array_equal(refine_pivo_features(feats[:2]), feats[:2])


def test_smoothing_reduces_jitter_without_changing_pitch_or_length():
    from render_student import smooth_acoustic_features, clean_audio_16k

    features = np.zeros((20, 20), dtype=np.float32)
    features[:, 0] = np.tile([1.0, -1.0], 10)
    features[:, 18:] = np.arange(40, dtype=np.float32).reshape(20, 2)
    result = smooth_acoustic_features(features)
    assert np.std(result[2:-2, 0]) < np.std(features[2:-2, 0])
    np.testing.assert_array_equal(result[:, 18:], features[:, 18:])
    np.testing.assert_array_equal(features[:, 0], np.tile([1.0, -1.0], 10))
    for length in (0, 1, 128, 16000):
        wav = np.zeros(length, dtype=np.float32)
        cleaned = clean_audio_16k(wav)
        assert cleaned.shape == wav.shape
        assert np.isfinite(cleaned).all()
        assert not np.any(cleaned)
    t = np.arange(16000) / 16000
    wav = (0.2 * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 7500 * t)).astype(np.float32)
    cleaned = clean_audio_16k(wav)
    spectrum = np.abs(np.fft.rfft(cleaned))
    assert spectrum[7500] < spectrum[1000] * 0.2
