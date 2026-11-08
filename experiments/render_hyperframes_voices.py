"""Regenerate the HyperFrames narration WAVs from the exported ONNX folder.

Deterministic: same model folder -> same WAV bytes. Texts are chosen for
intelligibility (CTC read-back CER, see experiments/eval/narr_matrix.json) and
match what is on screen in docs/pivo-hyperframes.

    python -I experiments/render_hyperframes_voices.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))
sys.path.insert(0, str(ROOT / "experiments"))

from render_student import features_to_audio, write_wav16  # noqa: E402
from verify_onnx_release import onnx_features  # noqa: E402

MODEL = ROOT / "experiments/models/pivo-id-1.4m"
OUT = ROOT / "docs/pivo-hyperframes/assets"

# name: (lang, voice_id, text)
CLIPS = {
    "pi-id": ("id", 0, "Selamat pagi, apa kabar?"),
    "vo-en": ("en", 1, "Hello there, ready to run on edge!"),
    "scene1": ("id", 0, "Mau pakai suara dua bahasa, tapi ukuran modelnya gede banget?"),
    "scene2": ("id", 0, "Kenalin PiVo, satu koma empat juta. Cuma lima koma enam megabita!"),
    "scene4": ("id", 0, "Model akustik dan vocoder dalam satu paket. Ringan untuk eksperimen lokal."),
    "scene5": ("id", 0, "PiVo, dua bahasa, Indonesia dan Inggris. Repositorinya ada di GitHub!"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing assets")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    from frontend import phonemize as id_phon
    sys.path.insert(0, str(ROOT / "PiVo-1.4M"))
    from pivo.text import phonemize as en_phon

    for name, (lang, voice, text) in CLIPS.items():
        out = a.out / f"{name}.wav"
        if out.exists() and not a.force:
            print(f"skip {out} (exists)")
            continue
        pr = en_phon(text, strict=False) if lang == "en" else id_phon(text)
        feats = onnx_features(MODEL, list(pr.token_ids), voice, 1.0)
        if not np.isfinite(feats).all():
            raise SystemExit(f"{name}: non-finite features")
        pcm = np.clip(features_to_audio(MODEL, feats), -1, 1)
        write_wav16(out, pcm)
        print(f"{out} {len(pcm) / 16000:.2f}s voice={voice} text={text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
