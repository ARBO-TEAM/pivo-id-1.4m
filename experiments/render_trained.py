"""Render speech from a fine-tuned acoustic checkpoint.

tokens -> PiVoAcoustic -> FARGAN features -> pinned PiVo streaming vocoder ->
16 kHz WAV. ``--lang id`` (default) uses the Indonesian frontend
(``src_id/frontend.py``); ``--lang en`` uses PiVo's own English frontend
(CMU pronouncing dictionary, ``PiVo-1.4M/pivo/text.py``), so one checkpoint
speaks both languages with the same ARPAbet inventory.

    python -I experiments/render_trained.py --ckpt experiments/runs/pilot_v4_warm_reset/best.pt \
        --text "Selamat pagi, apa kabar?" --voice-id 0 --model-dir PiVo-1.4M --out out.wav
    python -I experiments/render_trained.py ... --lang en --text "Good morning."
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402

from acoustic import load_pivo_checkpoint  # noqa: E402
from frontend import phonemize  # noqa: E402
from render_student import features_to_audio  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice-id", type=int, required=True, choices=[0, 1])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--lang", choices=["id", "en"], default="id",
                    help="id = Indonesian frontend (src_id/frontend.py); "
                         "en = PiVo's CMU-dictionary English frontend")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    model, ck = load_pivo_checkpoint(a.ckpt, num_voices=2)
    if ck.get("format") != "pivo-acoustic-train-v1":
        print(f"warning: checkpoint format is {ck.get('format')!r}, expected pivo-acoustic-train-v1")
    model.to(a.device).eval()

    if a.lang == "en":
        sys.path.insert(0, str(ROOT / "PiVo-1.4M"))
        from pivo.text import phonemize as en_phonemize
        pr = en_phonemize(a.text)
    else:
        pr = phonemize(a.text)
    if not any(s not in ("<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":")
               for s in pr.symbols):
        raise SystemExit("text has no spoken content")
    tokens = torch.tensor(pr.token_ids, dtype=torch.long, device=a.device)
    with torch.no_grad():
        feats, dur = model.synthesize_features(tokens, a.voice_id, speed=a.speed)
    feats_np = feats.detach().cpu().numpy()
    if not np.isfinite(feats_np).all():
        raise SystemExit("model produced non-finite features")
    pcm = features_to_audio(a.model_dir, feats_np)

    pcm = np.clip(pcm, -1, 1)
    with wave.open(str(a.out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())
    print(f"wrote {a.out} ({len(pcm)/16000:.2f}s, frames={len(feats_np)}, "
          f"dur_sum={int(dur.sum())}, peak={np.abs(pcm).max():.3f}, ckpt_step={ck.get('step')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
