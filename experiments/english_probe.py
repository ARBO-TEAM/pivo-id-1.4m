"""English retention probe: render the same English sentence with two acoustic
checkpoints and save both WAVs, so English support of the fine-tune can be
judged by listening / comparing.

    python -I experiments/english_probe.py --model-dir PiVo-1.4M \
        --ckpt-a PiVo-1.4M/checkpoints/acoustic.pt --tag-a original \
        --ckpt-b experiments/runs/pilot_v4_warm_reset/best.pt --tag-b v4 \
        --out-dir experiments/eval
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

import torch  # noqa: E402

from acoustic import load_pivo_checkpoint  # noqa: E402
from render_student import features_to_audio, write_wav16  # noqa: E402

TEXT = ("The quick brown fox jumps over the lazy dog. "
        "This voice speaks English, and now also Indonesian.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--ckpt-a", type=Path, required=True)
    ap.add_argument("--tag-a", default="a")
    ap.add_argument("--ckpt-b", type=Path, default=None)
    ap.add_argument("--tag-b", default="b")
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--voice-id", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from pivo.text import phonemize as en_phonemize  # PiVo's own English frontend

    pr = en_phonemize(a.text)
    tokens = torch.tensor(pr.token_ids, dtype=torch.long, device=a.device)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    report = {"text": a.text, "voice_id": a.voice_id, "symbols": len(pr.symbols), "runs": []}
    for ckpt, tag in ((a.ckpt_a, a.tag_a), (a.ckpt_b, a.tag_b)):
        if ckpt is None:
            continue
        model, ck = load_pivo_checkpoint(ckpt, num_voices=2)
        model.to(a.device).eval()
        with torch.no_grad():
            feats, dur = model.synthesize_features(tokens, a.voice_id)
        feats_np = feats.detach().cpu().numpy()
        pcm = np.clip(features_to_audio(a.model_dir, feats_np), -1, 1)
        out = a.out_dir / f"english_{tag}_v{a.voice_id}.wav"
        write_wav16(out, pcm)  # write_wav16 takes float [-1, 1], not int16
        report["runs"].append({
            "tag": tag, "ckpt": str(ckpt), "step": ck.get("step"),
            "frames": int(feats_np.shape[0]), "seconds": round(len(pcm) / 16000, 2),
            "peak": round(float(np.abs(pcm).max()), 3), "out": str(out)})
        print(f"{tag}: {out} ({len(pcm)/16000:.2f}s, frames={feats_np.shape[0]}, "
              f"peak={np.abs(pcm).max():.3f}, step={ck.get('step')})", flush=True)
    (a.out_dir / "english_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
