"""Minimal deterministic tests (no audio-quality claims; pilot never listened to).

Run from workspace root:
  .venv/Scripts/python.exe -m pytest experiments/fargan_verify_pilot_20261108_a1/test_verify_fargan_pilot_a1.py -q
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("verify_a1", EXP / "verify_fargan_pilot_a1.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def test_periods_known_values():
    feats = np.zeros((2, 20), dtype=np.float32)
    feats[0, 18] = 0.0  # 256/2^1.5 = 90.5 -> 90/91 rounding
    feats[1, 18] = -1.5  # 256/2^0 = 256 -> clipped 255
    p = v.periods_from_features(feats)
    assert p[0] in (90, 91)
    assert p[1] == 255


def test_deemphasis_unity_and_state():
    out, state = v.deemphasis(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert out[0] == 1.0
    assert abs(out[1] - 0.85) < 1e-6
    assert abs(out[2] - 0.85**2) < 1e-6
    assert abs(state - 0.85**2) < 1e-6


def test_fft_fallback_length_and_sine():
    sr_in, sr_out = 44100, 16000
    t = np.arange(sr_in) / sr_in
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = v._resample_fft_fallback(x, sr_in, sr_out)
    assert len(y) == int(round(sr_in * sr_out / sr_in))
    assert np.isfinite(y).all()
    assert 0.3 < float(np.sqrt(np.mean(y**2))) < 0.7


def test_resample_ffmpeg_or_fallback_length(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "TMP_DIR", tmp_path)
    x = np.random.default_rng(0).normal(0, 0.1, 44100).astype(np.float32)
    y, method = v.resample_44k_to_16k(x, tmp_path)
    assert method.startswith(("ffmpeg-aresample", "numpy-fft"))
    assert abs(len(y) - 16000) <= 32  # swr/soxr may differ by a few samples


def test_fresh_refuses_overwrite(tmp_path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        v.fresh(p)


def test_read_pilot_wav_contract():
    if not v.PILOT_WAV.is_file():
        pytest.skip("pilot WAV not present")
    a = v.read_pilot_wav(v.PILOT_WAV)
    assert a.shape == (387072,)
    assert np.isfinite(a).all()


def test_extract_width_is_20_on_sine(tmp_path):
    # Existing 1 s sine probe in opus-build-fargan: 16000 samples -> 100 frames x 20.
    sine = Path(r"E:\test\tts-test\pivo1-4m\opus-build-fargan\sine1s.s16")
    if not sine.is_file() or not v.FARGAN_DEMO.is_file():
        pytest.skip("sine probe / fargan_demo not present")
    audio = np.fromfile(sine, dtype="<i2").astype(np.float32) / 32768.0
    feats, width, frames = v.extract_fargan_features(audio, tmp_path)
    assert frames == 100
    assert width == 20
    assert feats.shape == (100, 20)
    assert np.isfinite(feats).all()


def test_pivo_vocoder_resynth_shape_on_sine(tmp_path):
    sine = Path(r"E:\test\tts-test\pivo1-4m\opus-build-fargan\sine1s.s16")
    if not sine.is_file() or not v.FARGAN_DEMO.is_file():
        pytest.skip("sine probe / fargan_demo not present")
    if not (v.ONNX_DIR / "conditioning.onnx").is_file():
        pytest.skip("PiVo ONNX not present")
    audio = np.fromfile(sine, dtype="<i2").astype(np.float32) / 32768.0
    feats, _, _ = v.extract_fargan_features(audio, tmp_path)
    out = v.resynth_pivo_vocoder(feats, max_frames=4)
    assert out.shape == (4 * 160,)
    assert np.isfinite(out).all()


def test_extract_rejects_scaling_truncation(tmp_path):
    # Regression: float32 audio in [-1,1] must be scaled to int16, not cast.
    rng = np.random.default_rng(1)
    audio = (0.5 * np.sin(2 * np.pi * 440 * np.arange(1600) / 16000)).astype(np.float32)
    feats, width, frames = v.extract_fargan_features(audio, tmp_path)
    assert frames == 10 and width == 20
    # Cepstral dims of a real tone must vary across frames (not all constant zero)
    assert float(feats[:, 1].std()) > 1e-4 or float(feats[:, 0].std()) > 1e-4


def test_end_to_end_smoke_max_frames_8(tmp_path):
    """Full path on the real pilot but only 8 frames of resynthesis (fast)."""
    if not v.PILOT_WAV.is_file():
        pytest.skip("pilot WAV not present")
    r = subprocess.run(
        [sys.executable, str(EXP / "verify_fargan_pilot_a1.py"), "--max-frames", "8"],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    import json

    rep = json.loads(r.stdout)
    assert rep["features"]["frames"] >= 800
    assert rep["features"]["record_width_floats"] in (20, 36)
    assert rep["pivo_vocoder_resynth"]["n_samples"] == 8 * 160
    assert Path(rep["pivo_vocoder_resynth"]["wav"]).is_file()
    assert Path(rep["opus_fargan_reference_resynth"]["wav"]).is_file()
