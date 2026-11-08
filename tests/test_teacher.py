"""Small deterministic tests for src_id/teacher.py (no model files needed)."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_id.teacher import (  # noqa: E402
    SAMPLE_RATE,
    UnicodeIndexer,
    load_voice_style,
    synthesize,
    preprocess_text,
    write_wav,
)


def _indexer(tmp_path, size=256):
    table = [-1] * size
    for ch in "<>id./,?!;:'ab cpag":
        table[ord(ch)] = ord(ch)
    p = tmp_path / "idx.json"
    import json

    p.write_text(json.dumps(table), encoding="utf-8")
    return UnicodeIndexer(p)


def test_preprocess_adds_period_and_language_tokens():
    assert preprocess_text("pagi", "id") == "<id>pagi.</id>"
    assert preprocess_text("pagi.", "id") == "<id>pagi.</id>"


def test_preprocess_normalizes_curly_quotes_and_dash():
    assert preprocess_text("“halo”—pesan", "id") == '<id>"halo"-pesan.</id>'


def test_preprocess_rejects_bad_lang():
    with pytest.raises(ValueError):
        preprocess_text("halo", "indonesian")


def test_encode_unsupported_char_raises(tmp_path):
    table = [-1] * 256
    import json

    p = tmp_path / "indexer.json"
    p.write_text(json.dumps(table), encoding="utf-8")
    idx = UnicodeIndexer(p)
    with pytest.raises(ValueError, match="unsupported"):
        idx.encode("halo", "id")


def test_encode_shape_and_mask(tmp_path):
    idx = _indexer(tmp_path)
    ids, mask = idx.encode("pagi", "id")
    assert ids.shape == (1, ids.shape[1])
    assert mask.shape == (1, 1, ids.shape[1])
    assert float(mask.sum()) == ids.shape[1]


def test_load_voice_style_shapes(tmp_path):
    import json

    style = {
        "style_ttl": {"data": [[[[0.0] * 4] * 2]], "dims": [1, 2, 4]},
        "style_dp": {"data": [[[[0.0] * 4] * 2]], "dims": [1, 2, 4]},
    }
    p = tmp_path / "style.json"
    p.write_text(json.dumps(style), encoding="utf-8")
    ttl, dp = load_voice_style(p)
    assert ttl.shape == (1, 2, 4)
    assert dp.shape == (1, 2, 4)


def test_load_voice_style_rejects_malformed(tmp_path):
    import json

    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"style_ttl": {"data": [[0.0]]}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_voice_style(p)


def test_write_wav_round_trip_and_no_overwrite(tmp_path):
    audio = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
    out = write_wav(tmp_path / "t.wav", audio)
    raw = out.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    fmt_rate = struct.unpack("<I", raw[24:28])[0]
    assert fmt_rate == SAMPLE_RATE
    with pytest.raises(FileExistsError):
        write_wav(out, audio)


def test_write_wav_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        write_wav(tmp_path / "x.wav", np.array([np.nan], dtype=np.float32))


def test_synthesize_dynamic_shapes_and_estimator_updates(tmp_path, monkeypatch):
    import src_id.teacher as teacher

    calls = []

    class Session:
        def __init__(self, name):
            self.name = name

        def run(self, _, inputs):
            calls.append((self.name, inputs))
            if self.name == "dp":
                return [np.array([1.0], dtype=np.float32)]
            if self.name == "te":
                return [np.zeros((1, 4, inputs["text_ids"].shape[1]), dtype=np.float32)]
            if self.name == "ve":
                return [inputs["noisy_latent"] + 1]
            return [np.ones((1, inputs["latent"].shape[2] * 3072 + 7), dtype=np.float32)]

    monkeypatch.setattr(teacher, "_load_session", lambda path, threads, providers: Session(path.name))
    style = np.zeros((1, 2, 4), dtype=np.float32)
    result = synthesize(dp_path=Path("dp"), te_path=Path("te"), ve_path=Path("ve"),
                        voc_path=Path("voc"), indexer=_indexer(tmp_path),
                        style_ttl=style, style_dp=style, text="pagi", speed=1.0,
                        total_steps=2)
    assert calls[0][1]["text_ids"].shape == (1, len("<id>pagi.</id>"))
    steps = [inputs for name, inputs in calls if name == "ve"]
    assert len(steps) == 2
    assert steps[0]["noisy_latent"].shape == (1, 144, 15)
    np.testing.assert_array_equal(steps[1]["noisy_latent"], steps[0]["noisy_latent"] + 1)
    np.testing.assert_array_equal(calls[-1][1]["latent"], steps[1]["noisy_latent"] + 1)
    assert result.shape == (15 * 3072 + 7,)


@pytest.mark.skipif(
    not Path(r"E:\test\tts-test").exists(), reason="local model tree not mounted"
)
def test_reference_fp32_files_present():
    base = Path(r"E:\test\tts-test\split-supertonic-v3\supertonic_work")
    for rel in [
        "hf_fresh_download_20261108/onnx/duration_predictor.onnx",
        "hf_fresh_download_20261108/onnx/text_encoder.onnx",
        "hf_fresh_download_20261108/onnx/vocoder.onnx",
        "hf_fresh_download_20261108/onnx/tts.json",
        "hf_fresh_download_20261108/onnx/unicode_indexer.json",
        "vector_estimator.onnx",
        "voice_styles/F1.json",
    ]:
        assert (base / rel).is_file(), rel
