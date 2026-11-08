"""Objective intelligibility check: CER of trained-checkpoint renders on held-out
val rows, measured with the Indonesian character CTC model (same one used for
alignment). This is a proxy, not human quality assessment.

    python -I experiments/eval_cer.py --ckpt runs/pilot_v3_long/best.pt \
        --data data/pilot_v3 --model-dir PiVo-1.4M --limit 8
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

import torch  # noqa: E402

import ctc_align  # noqa: E402
from acoustic import load_pivo_checkpoint  # noqa: E402
from frontend import phonemize  # noqa: E402
from render_student import features_to_audio  # noqa: E402

ALIGNER = ROOT / "models/hf/hub/models--indonesian-nlp--wav2vec2-indonesian-javanese-sundanese/snapshots/e5e699fa5aa5bdce999276a90a29589587b58ac9"
ALIGNER_EN = ROOT / "models/hf/hub/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156"


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--aligner-dir", type=Path, default=ALIGNER)
    ap.add_argument("--lang", choices=["id", "en"], default="id",
                    help="id = Indonesian frontend + Indonesian CTC; en = PiVo CMU frontend "
                         "+ English CTC (facebook/wav2vec2-base-960h)")
    ap.add_argument("--id-prefix", default=None,
                    help="only evaluate rows whose id starts with this prefix (merged corpora)")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--teacher-dir", type=Path, default=None,
                    help="if set, score the teacher WAVs (<dir>/wav16/<id>.wav) instead of the model")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.lang == "en" and a.aligner_dir == ALIGNER:
        a.aligner_dir = ALIGNER_EN

    rows = [json.loads(l) for l in (a.data / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
    if a.id_prefix:
        rows = [r for r in rows if r["id"].startswith(a.id_prefix)]
    meta = {m["id"]: m for m in (json.loads(l) for l in (a.data / "build_meta.jsonl").open(encoding="utf-8") if l.strip())}
    val = [r for r in rows if r["split"] == "val"]
    by_voice: dict[int, list] = {}
    for r in val:
        by_voice.setdefault(r["voice_id"], []).append(r)
    picked = [r for v in sorted(by_voice) for r in by_voice[v][:a.limit]]

    model, ck = load_pivo_checkpoint(a.ckpt, num_voices=2) if not a.teacher_dir else (None, {"step": "teacher"})
    if model is not None:
        model.to(a.device).eval()
    processor, ctc_model = ctc_align.load_ctc(a.aligner_dir)
    if a.lang == "en":
        sys.path.insert(0, str(ROOT / "PiVo-1.4M"))
        from pivo.text import phonemize as en_phonemize
        phon = lambda t: en_phonemize(t, strict=False)
    else:
        phon = phonemize

    results = []
    for r in picked:
        text = meta.get(r["id"], {}).get("text")
        if not text:
            continue
        pr = phon(text)
        if a.teacher_dir:
            wav = a.teacher_dir / "wav16" / f"{r['id']}.wav"
            if not wav.exists() and a.id_prefix:
                wav = a.teacher_dir / "wav16" / f"{r['id'][len(a.id_prefix):]}.wav"
            pcm, rate = ctc_align.read_wav_mono(wav)
            pcm = ctc_align.resample_to_16k(pcm, rate) if rate != 16000 else pcm
        else:
            tokens = torch.tensor(pr.token_ids, dtype=torch.long, device=a.device)
            with torch.no_grad():
                feats, _dur = model.synthesize_features(tokens, r["voice_id"])
            feats_np = feats.detach().cpu().numpy()
            if not np.isfinite(feats_np).all():
                results.append({"id": r["id"], "voice": r["voice_id"], "cer": 1.0,
                                "reason": "non-finite features"})
                continue
            pcm = (np.clip(features_to_audio(a.model_dir, feats_np), -1, 1) * 32767).astype(np.int16)
        runs, _n = ctc_align.ctc_runs(processor, ctc_model, pcm)
        heard = "".join(ch.lower() for ch, _, _ in runs if ch.isalnum())
        want = "".join(c.lower() for c in pr.normalized_text if c.isalnum())
        cer = edit_distance(heard, want) / max(1, len(want))
        results.append({"id": r["id"], "voice": r["voice_id"], "text": text, "heard": heard,
                        "cer": round(cer, 4)})
        print(f"{r['id']} v{r['voice_id']} cer={cer:.3f} heard={heard[:80]!r}", flush=True)

    if not results:
        raise SystemExit("no val rows evaluated")
    summary = {"checkpoint": str(a.ckpt), "step": ck.get("step"), "lang": a.lang,
               "rows": len(results), "mean_cer": round(float(np.mean([x["cer"] for x in results])), 4)}
    for v in sorted({x["voice"] for x in results}):
        summary[f"mean_cer_voice_{v}"] = round(float(np.mean([x["cer"] for x in results if x["voice"] == v])), 4)
    print("SUMMARY", json.dumps(summary))
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "rows": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
