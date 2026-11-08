"""Render narration candidates from a PyTorch checkpoint (not ONNX) for listening tests.

    python -I experiments/render_torch_narration.py --ckpt experiments/runs/pilot_v4_warm_reset/best.pt \
        --lang id --voice-id 0 --text "..." --out out.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from acoustic import load_pivo_checkpoint  # noqa: E402
from render_student import features_to_audio, write_wav16  # noqa: E402

MODEL_DIR = ROOT / "experiments/models/pivo-id-1.4m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", choices=["id", "en"], default="id")
    ap.add_argument("--voice-id", type=int, default=0, choices=[0, 1])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    a.out.parent.mkdir(parents=True, exist_ok=True)

    model, _ck = load_pivo_checkpoint(a.ckpt, num_voices=2)
    model.eval()
    if a.lang == "en":
        sys.path.insert(0, str(ROOT / "PiVo-1.4M"))
        from pivo.text import phonemize as phon
        pr = phon(a.text, strict=False)
    else:
        from frontend import phonemize as phon
        pr = phon(a.text)
    with torch.no_grad():
        feats, _dur = model.synthesize_features(torch.tensor(pr.token_ids), a.voice_id, speed=a.speed)
    pcm = np.clip(features_to_audio(MODEL_DIR, feats.numpy()), -1, 1)
    write_wav16(a.out, pcm)
    print(f"{a.out} {len(pcm)/16000:.2f}s frames={feats.shape[0]} peak={np.abs(pcm).max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
