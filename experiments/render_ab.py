"""Side-by-side listening render for one source row id: teacher 44.1 kHz WAV,
teacher features through the FARGAN vocoder (16 kHz ceiling of the training
target), fine-tuned PyTorch student, fine-tuned ONNX student. Answers two
questions: did the ONNX export change the audio, and how far the student is
from the teacher once both go through the same 16 kHz vocoder.

    python -I experiments/render_ab.py --ckpt experiments/runs/pilot_v5c/best.pt \
        --model-dir experiments/models/pivo-id-1.4m --data experiments/data/en_pilot_v1 \
        --ids F1_E042_s0,M1_E042_s0 --out-dir experiments/eval/listen_ab_v1
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from verify_onnx_release import onnx_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="build dir with build_meta.jsonl, wav44/, feat/")
    ap.add_argument("--lang", choices=["id", "en"], default="id",
                    help="id = Indonesian frontend, en = PiVo CMU frontend")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise SystemExit(f"refusing to overwrite {a.out_dir}")
    a.out_dir.mkdir(parents=True)

    meta = {}
    for line in (a.data / "build_meta.jsonl").open(encoding="utf-8"):
        if line.strip():
            m = json.loads(line)
            meta[m["id"]] = m
    model, _ck = load_pivo_checkpoint(a.ckpt, num_voices=2)
    model.eval()
    if a.lang == "en":
        from pivo.text import phonemize as phon
    else:
        from frontend import phonemize as phon

    for rid in [s.strip() for s in a.ids.split(",") if s.strip()]:
        m = meta[rid]
        voice = m["voice_id"]
        pt = phon(m["text"], strict=False) if a.lang == "en" else phon(m["text"])
        tokens = pt.token_ids
        t_feats, _dur = model.synthesize_features(torch.tensor(tokens), voice)
        t_feats = t_feats.detach().cpu().numpy().astype(np.float32)
        o_feats = onnx_features(a.model_dir, list(tokens), voice, 1.0)

        stem = a.out_dir / rid
        write_wav16(Path(f"{stem}_student_torch16.wav"), features_to_audio(a.model_dir, t_feats))
        write_wav16(Path(f"{stem}_student_onnx16.wav"), features_to_audio(a.model_dir, o_feats))
        teacher_feats = np.load(a.data / "feat" / f"{rid}.npy").astype(np.float32)
        write_wav16(Path(f"{stem}_teacher_feats_fargan16.wav"), features_to_audio(a.model_dir, teacher_feats))
        for sub, tag in (("wav44", "teacher44"), ("wav16", "teacher16")):
            src = a.data / sub / f"{rid}.wav"
            if src.exists():
                shutil.copyfile(src, f"{stem}_{tag}.wav")

        n = min(t_feats.shape[0], o_feats.shape[0])
        print(json.dumps({
            "id": rid, "voice": voice, "text": m["text"],
            "frames_torch": int(t_feats.shape[0]), "frames_onnx": int(o_feats.shape[0]),
            "onnx_vs_torch_max_abs": float(np.abs(t_feats[:n] - o_feats[:n]).max()),
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
