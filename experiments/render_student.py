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


def features_to_audio(model_dir: Path, feats: np.ndarray, *, threads: int = 2) -> np.ndarray:
    """FARGAN 20-dim features [frames, 20] -> 16 kHz PCM, matching pivo.runtime."""
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
        pcm, de_state = deemphasis(np.concatenate(frame), de_state)
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
