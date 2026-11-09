"""Render narration WAVs from the exported ONNX folder (no PyTorch weights).

Same path the release verifier uses: text -> ONNX encoder/decoder -> FARGAN
vocoder -> 16 kHz PCM16 WAV.

    python -I experiments/render_narration.py --model experiments/models/pivo-id-1.4m \
        --lang id --voice-id 0 --text "Selamat pagi, apa kabar?" \
        --out docs/pivo-hyperframes/assets.new/pi-id.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))
sys.path.insert(0, str(ROOT / "experiments"))

from render_student import clean_audio_16k, features_to_audio, write_wav16  # noqa: E402
from verify_onnx_release import onnx_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", choices=["id", "en"], default="id")
    ap.add_argument("--voice-id", type=int, default=0, choices=[0, 1])
    ap.add_argument("--speed", type=float, default=1.0)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--smooth", action="store_true", help="apply original feature smoothing and offline audio cleanup")
    modes.add_argument("--selective", action="store_true", help="apply selective feature smoothing without offline DSP")
    ap.add_argument("--deemphasis", type=float, default=0.85, help="deemphasis coefficient (default: 0.85)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if a.lang == "en":
        from pivo.text import phonemize as phon
        pr = phon(a.text, strict=False)
    else:
        from frontend import phonemize as phon
        pr = phon(a.text)
    feats = onnx_features(a.model, list(pr.token_ids), a.voice_id, a.speed)
    if not np.isfinite(feats).all():
        raise SystemExit("non-finite features")
    pcm = features_to_audio(a.model, feats, smooth=a.smooth, selective=a.selective,
                            deemphasis_coefficient=a.deemphasis)
    if a.smooth:
        pcm = clean_audio_16k(pcm)
    pcm = np.clip(pcm, -1, 1)
    write_wav16(a.out, pcm)
    print(f"{a.out} {len(pcm) / 16000:.2f}s frames={feats.shape[0]} peak={np.abs(pcm).max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
