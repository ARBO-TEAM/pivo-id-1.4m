"""End-to-end check of an exported ONNX model folder: Indonesian and English text
-> ONNX encoder/decoder (+ streaming vocoder graph) -> 16 kHz WAV -> CTC readback.

This is the release-artifact gate: it proves the exported folder runs without
PyTorch acoustic weights, and that the audio still transcribes back.

    python -I experiments/verify_onnx_release.py --model experiments/models/pivo-id-1.4m \
        --aligner-dir <id CTC dir> --ctc-lang id --voice-id 0 --out out.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "PiVo-1.4M"))

import ctc_align  # noqa: E402
from render_student import features_to_audio, write_wav16  # noqa: E402

ID_CTC = ROOT / "models/hf/hub/models--indonesian-nlp--wav2vec2-indonesian-javanese-sundanese/snapshots/e5e699fa5aa5bdce999276a90a29589587b58ac9"
EN_CTC = ROOT / "models/hf/hub/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156"


def onnx_features(model_dir: Path, token_ids: list[int], voice: int, speed: float) -> np.ndarray:
    """Run the exported encoder + decoder graphs exactly as the PiVo runtime does."""
    import onnxruntime as ort

    onnx_dir = model_dir / "onnx"
    meta = json.loads((onnx_dir / "metadata.json").read_text(encoding="utf-8"))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    enc = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), opts, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(onnx_dir / "decoder.onnx"), opts, providers=["CPUExecutionProvider"])
    c = meta["config"]
    tokens = np.array([token_ids], dtype=np.int64)
    encoded, logdur = enc.run(None, {"tokens": tokens, "voice": np.array([voice], np.int64),
                                     "lengths": np.array([len(token_ids)], np.int64)})
    dur = np.rint(np.expm1(logdur.clip(0, np.log1p(c["max_token_frames"]))) / speed)
    dur = dur.clip(0, c["max_token_frames"]).astype(np.int64)
    dur = np.maximum(dur, (tokens >= 10).astype(np.int64))
    dur[tokens <= 2] = 0
    if dur.sum() > c["max_utterance_frames"]:
        raise SystemExit("text exceeds the utterance frame budget")
    expanded = np.repeat(encoded[0], dur[0], axis=0)[None]
    from acoustic import phase_features  # same phase formula as training
    import torch
    phase = phase_features(torch.from_numpy(dur[0]), c["max_token_frames"]).numpy()[None]
    expanded = np.concatenate((expanded, phase), axis=-1).astype(np.float32)

    caches = {n: np.zeros((1, c["kv_heads"], 0, c["width"] // c["heads"]), np.float32)
              for n in meta["cache_names"]}
    out = []
    for pos in range(0, expanded.shape[1], 16):
        r = dec.run(None, {"expanded": expanded[:, pos:pos + 16],
                           "position": np.array(pos, np.int64), **caches})
        out.append(r[0])
        caches = dict(zip(meta["cache_names"], r[1:]))
    return np.concatenate(out, axis=1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", choices=["id", "en"], default="id")
    ap.add_argument("--voice-id", type=int, default=0, choices=[0, 1])
    ap.add_argument("--aligner-dir", type=Path, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    if a.lang == "en":
        from pivo.text import phonemize as phon
    else:
        from frontend import phonemize as phon

    pr = phon(a.text) if a.lang == "id" else phon(a.text, strict=False)
    feats = onnx_features(a.model, list(pr.token_ids), a.voice_id, a.speed)
    if not np.isfinite(feats).all():
        raise SystemExit("ONNX model produced non-finite features")
    pcm = np.clip(features_to_audio(a.model, feats), -1, 1)
    write_wav16(a.out, pcm)

    aligner = a.aligner_dir or (ID_CTC if a.lang == "id" else EN_CTC)
    processor, ctc_model = ctc_align.load_ctc(aligner)
    pcm16, rate = ctc_align.read_wav_mono(a.out)
    runs, _n = ctc_align.ctc_runs(processor, ctc_model, pcm16)
    heard = "".join(ch.lower() for ch, _, _ in runs if ch.isalnum())
    want = "".join(ch.lower() for ch in pr.normalized_text if ch.isalnum())

    def ed(x, y):
        prev = list(range(len(y) + 1))
        for i, cx in enumerate(x, 1):
            cur = [i]
            for j, cy in enumerate(y, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy)))
            prev = cur
        return prev[-1]

    cer = ed(heard, want) / max(1, len(want))
    print(json.dumps({"out": str(a.out), "lang": a.lang, "voice": a.voice_id,
                      "frames": int(feats.shape[0]), "seconds": round(len(pcm) / 16000, 2),
                      "cer": round(cer, 4), "heard": heard, "want": want},
                     ensure_ascii=False))
    return 0 if cer < 0.2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
