"""English intelligibility check for two acoustic checkpoints (pretrained PiVo vs
an Indonesian fine-tune), scored with openai/whisper-small.en like PiVo's own
evaluation protocol.

Renders each sentence with each checkpoint, transcribes, and reports word error
rate per run. This is a proxy for English retention, not a quality claim.

    python -I experiments/eval_english.py --model-dir PiVo-1.4M \
        --ckpt-a PiVo-1.4M/checkpoints/acoustic.pt --tag-a original \
        --ckpt-b runs/pilot_v4_warm_reset/best.pt --tag-b v4 \
        --out experiments/eval/english_wer.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "PiVo-1.4M"))

import torch  # noqa: E402

from acoustic import load_pivo_checkpoint  # noqa: E402
from render_student import features_to_audio  # noqa: E402

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your order before the end of the day.",
    "The meeting is scheduled for next Tuesday at nine in the morning.",
    "I would like to know the price of the large size.",
    "Thank you for calling, have a nice day.",
    "Our office is closed on public holidays.",
    "Could you repeat that a little more slowly?",
    "The total comes to five thousand rupiah.",
]


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--ckpt-a", type=Path, required=True)
    ap.add_argument("--tag-a", default="a")
    ap.add_argument("--ckpt-b", type=Path, default=None)
    ap.add_argument("--tag-b", default="b")
    ap.add_argument("--asr", default="openai/whisper-small.en")
    ap.add_argument("--voice-id", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from pivo.text import phonemize as en_phonemize
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(a.asr)
    asr = WhisperForConditionalGeneration.from_pretrained(a.asr).to(a.device).eval()

    @torch.no_grad()
    def transcribe(pcm: np.ndarray) -> str:
        feats = proc(pcm.astype(np.float32), sampling_rate=16000,
                     return_tensors="pt").input_features.to(a.device)
        ids = asr.generate(feats, max_new_tokens=64)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    results = {"asr": a.asr, "voice_id": a.voice_id, "runs": []}
    for ckpt, tag in ((a.ckpt_a, a.tag_a), (a.ckpt_b, a.tag_b)):
        if ckpt is None:
            continue
        model, ck = load_pivo_checkpoint(ckpt, num_voices=2)
        model.to(a.device).eval()
        rows, wer_sum, words_sum = [], 0, 0
        for text in SENTENCES:
            pr = en_phonemize(text)
            tokens = torch.tensor(pr.token_ids, dtype=torch.long, device=a.device)
            with torch.no_grad():
                feats, _dur = model.synthesize_features(tokens, a.voice_id)
            feats_np = feats.detach().cpu().numpy()
            if not np.isfinite(feats_np).all():
                rows.append({"text": text, "heard": "<non-finite features>", "wer": 1.0})
                wer_sum, words_sum = wer_sum + len(normalize_words(text)), words_sum + len(normalize_words(text))
                continue
            pcm = np.clip(features_to_audio(a.model_dir, feats_np), -1, 1)
            heard = transcribe(pcm)
            ref, hyp = normalize_words(text), normalize_words(heard)
            err = edit_distance(ref, hyp)
            rows.append({"text": text, "heard": heard,
                         "wer": round(err / max(1, len(ref)), 4)})
            wer_sum += err
            words_sum += len(ref)
            print(f"[{tag}] {text!r} -> {heard!r} (wer={err/max(1,len(ref)):.3f})", flush=True)
        summary = {"tag": tag, "ckpt": str(ckpt), "step": ck.get("step"),
                   "sentences": len(rows), "words": words_sum,
                   "wer": round(wer_sum / max(1, words_sum), 4)}
        results["runs"].append({"summary": summary, "rows": rows})
        print("SUMMARY", json.dumps(summary), flush=True)

    if a.out:
        a.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
