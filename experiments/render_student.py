"""Render a student WAV from an aligned pilot row (encoder+decoder+streaming vocoder).

Sanity tool for listening checks; it does not retrain anything. The vocoder is
the pinned streaming release (`conditioning.onnx` + `vocoder_step.onnx` with
persistent state and deemphasis), not a monolithic graph.

    python -I experiments/render_student.py --row-id F1_L562 --data experiments/data/pilot_v1 \
        --model-dir PiVo-1.4M --out out.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))


def smooth_acoustic_features(features: np.ndarray) -> np.ndarray:
    """Smooth cepstral channels over 50 ms without altering pitch or energy."""
    result = np.asarray(features, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 20:
        raise ValueError("expected features with shape [frames, 20]")
    if len(result) >= 5:
        padded = np.pad(result[:, :18], ((2, 2), (0, 0)), mode="edge")
        result[:, :18] = sum(weight * padded[i:i + len(result)] for i, weight in
                             enumerate((-3/35, 12/35, 17/35, 12/35, -3/35)))
    return result


def refine_pivo_features(features: np.ndarray) -> np.ndarray:
    """Smooth high cepstra on low-flux frames and reject isolated pitch spikes."""
    out = np.asarray(features, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] != 20:
        raise ValueError("expected features with shape [frames, 20]")
    if len(out) < 5:
        return out
    diff = np.diff(out[:, 1:13], axis=0, prepend=out[:1, 1:13])
    flux = np.linalg.norm(diff, axis=1)
    steady = flux <= np.percentile(flux, 65)
    padded = np.pad(out[:, 6:18], ((1, 1), (0, 0)), mode="edge")
    filtered = 0.2 * padded[:-2] + 0.6 * padded[1:-1] + 0.2 * padded[2:]
    out[steady, 6:18] = filtered[steady]
    pitch = np.pad(out[:, 18], (1, 1), mode="edge")
    median = np.median(np.stack((pitch[:-2], pitch[1:-1], pitch[2:])), axis=0)
    # Opus feature 19 is transformed pitch correlation minus 0.5. This
    # conservative experimental threshold is a heuristic, not a trained V/UV detector.
    voiced = out[:, 19] > 0.0
    voiced_neighbors = np.pad(voiced, (1, 1), mode="edge")
    pitch_mask = steady & voiced_neighbors[:-2] & voiced_neighbors[1:-1] & voiced_neighbors[2:]
    out[pitch_mask, 18] = median[pitch_mask]
    return out


def clean_audio_16k(wav: np.ndarray) -> np.ndarray:
    """Offline gentle bandpass and high-frequency spectral gate; preserve level."""
    audio = np.asarray(wav, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("expected mono audio")
    if not len(audio):
        return audio.copy()
    size, hop = 512, 128
    padded = np.pad(audio, (size, size + (-len(audio)) % hop))
    window = np.hanning(size).astype(np.float32)
    starts = range(0, len(padded) - size + 1, hop)
    spectra = np.stack([np.fft.rfft(padded[i:i + size] * window) for i in starts])
    magnitude = np.abs(spectra)
    energy = np.mean(magnitude, axis=1)
    # Exclude padding-only frames from the noise estimate.
    valid = energy > 1e-8
    if np.any(valid):
        threshold = np.percentile(energy[valid], 15)
        noise = np.mean(magnitude[valid & (energy <= threshold)], axis=0)
        frequencies = np.fft.rfftfreq(size, 1 / 16000)
        gate = np.ones_like(magnitude)
        high = frequencies >= 5500
        gate[:, high] = np.maximum(0.5, 1 - 0.5 * noise[high] / (magnitude[:, high] + 1e-12))
        spectra *= gate
        # Smooth fourth-order Butterworth magnitude response, applied offline.
        hp = 1 / np.sqrt(1 + (70 / np.maximum(frequencies, 1e-9)) ** 8)
        lp = 1 / np.sqrt(1 + (frequencies / 7000) ** 8)
        spectra *= hp * lp
    out = np.zeros(len(padded), dtype=np.float64)
    norm = np.zeros(len(padded), dtype=np.float64)
    for i, spectrum in zip(starts, spectra):
        out[i:i + size] += np.fft.irfft(spectrum, n=size) * window
        norm[i:i + size] += window ** 2
    np.divide(out, norm, out=out, where=norm > 1e-12)
    return out[size:size + len(audio)].astype(np.float32)


def features_to_audio(model_dir: Path, feats: np.ndarray, *, threads: int = 2,
                      smooth: bool = False, selective: bool = False,
                      deemphasis_coefficient: float = 0.85) -> np.ndarray:
    """FARGAN 20-dim features [frames, 20] -> 16 kHz PCM."""
    if smooth and selective:
        raise ValueError("choose only one feature smoothing method")
    if not 0 <= deemphasis_coefficient < 1:
        raise ValueError("deemphasis coefficient must be in [0, 1)")
    import onnxruntime as ort

    sys.path.insert(0, str(ROOT / "PiVo-1.4M"))
    from pivo.audio import deemphasis  # noqa: E402
    from pivo.runtime import Runtime  # noqa: E402

    meta = json.loads((model_dir / "onnx" / "metadata.json").read_text(encoding="utf-8"))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    prov = ["CPUExecutionProvider"]
    cond = ort.InferenceSession(str(model_dir / "onnx" / "conditioning.onnx"), opts, providers=prov)
    step = ort.InferenceSession(str(model_dir / "onnx" / "vocoder_step.onnx"), opts, providers=prov)

    if selective:
        feats = refine_pivo_features(feats)
    elif smooth:
        feats = smooth_acoustic_features(feats)
    else:
        feats = np.asarray(feats, dtype=np.float32)
    n = feats.shape[0]
    state = {
        name: np.zeros((1, width), np.float32)
        for name, width in zip(meta["step_state_names"], (256, 160, 128, 128, 164))
    }
    out = np.empty(n * 160, np.float32)
    de_state = 0.0
    for k in range(n):
        previous = feats[k - 1] if k else feats[k]
        current = feats[k]
        following = feats[k + 1] if k + 1 < n else current
        context = np.stack((previous, current, following))[None].astype(np.float32)
        periods = Runtime.periods(context)
        conditioning = cond.run(None, {"features": context, "periods": periods})[0]
        frame = []
        for subframe in range(4):
            res = step.run(None, {
                "conditioning": conditioning[:, 0, subframe * 80:(subframe + 1) * 80],
                "period": periods[:, 1],
                **state,
            })
            state = dict(zip(meta["step_state_names"], res[1:]))
            frame.append(res[0][0])
        pcm, de_state = deemphasis(np.concatenate(frame), de_state, coefficient=deemphasis_coefficient)
        out[k * 160:(k + 1) * 160] = pcm
    return out


def write_wav16(path: Path, pcm: np.ndarray) -> None:
    pcm = np.clip(np.asarray(pcm, np.float32), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--row-id", required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    rows = [json.loads(l) for l in (a.data / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
    row = next((r for r in rows if r["id"] == a.row_id), None)
    if row is None:
        raise SystemExit(f"row {a.row_id} not found")
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    import onnxruntime as ort

    sys.path.insert(0, str(ROOT / "PiVo-1.4M"))
    from pivo.phase import phoneme_phase_features  # noqa: E402

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = a.threads
    opts.inter_op_num_threads = 1
    prov = ["CPUExecutionProvider"]
    enc = ort.InferenceSession(str(a.model_dir / "onnx" / "encoder.onnx"), opts, providers=prov)
    dec = ort.InferenceSession(str(a.model_dir / "onnx" / "decoder.onnx"), opts, providers=prov)

    tokens = np.array([row["tokens"]], dtype=np.int64)
    voice = np.array([row["voice_id"]], dtype=np.int64)
    meta = json.loads((a.model_dir / "onnx" / "metadata.json").read_text(encoding="utf-8"))
    encoded, _logd = enc.run(None, {"tokens": tokens, "voice": voice,
                                    "lengths": np.array([tokens.shape[1]], np.int64)})
    # Feed the *aligned* durations, not the predictor, so the render checks the
    # acoustic model against the exact timing used in training.
    dur = np.array(row["durations"], dtype=np.int64)
    if int(dur.sum()) == 0:
        raise SystemExit("row has zero total duration")
    expanded = np.repeat(encoded[0], dur, axis=0)[None]
    phase = phoneme_phase_features(dur, meta["config"]["max_token_frames"])[None]
    expanded = np.concatenate((expanded, phase), axis=-1).astype(np.float32)

    cache = {n: np.zeros((1, 2, 0, 24), np.float32) for n in ("key0", "value0", "key1", "value1")}
    feats = []
    for p in range(0, expanded.shape[1], 16):
        res = dec.run(None, {"expanded": expanded[:, p:p + 16],
                             "position": np.array(p, np.int64), **cache})
        feats.append(res[0])
        cache = dict(zip(("key0", "value0", "key1", "value1"), res[1:]))
    feats = np.concatenate(feats, axis=1).astype(np.float32)[0]

    pcm = features_to_audio(a.model_dir, feats, threads=a.threads)
    write_wav16(a.out, pcm)
    print(f"wrote {a.out} ({len(pcm)/16000:.2f}s, frames={feats.shape[0]}, peak={np.abs(pcm).max():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
