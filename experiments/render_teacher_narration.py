"""Render the HyperFrames narration WAVs with the Supertonic teacher (F1, 44.1 kHz).

The 1.4M PiVo student is a research model; its voice is rougher than the
Supertonic teacher it was distilled from. Narration uses the teacher; the two
demo cards (pi-id.wav, vo-en.wav) stay real ONNX student output, produced by
render_hyperframes_voices.py.

    python -I experiments/render_teacher_narration.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))

import teacher  # noqa: E402

MODEL_ROOT = Path("./supertonic_work")
ONNX = MODEL_ROOT / "hf_fresh_download_20261108" / "onnx"
OUT = ROOT / "docs/pivo-hyperframes/assets"

# name: (text, speed) -- speeds chosen so each clip fits its scene window
CLIPS = {
    "scene1": ("Mau suara dua bahasa tapi ukuran modelnya gede banget?", 1.05),
    "scene2": ("Kenalin PiVo. Satu koma empat juta, cuma lima koma enam megabita!", 1.25),
    "scene4": ("Model akustik dan vocoder dalam satu paket. Ringan untuk eksperimen lokal.", 1.05),
    "scene5": ("PiVo, bahasa Indonesia dan Inggris. Reponya ada di GitHub!", 1.05),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--voice", default="F1", choices=["F1", "M1"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    common = dict(
        dp_path=ONNX / "duration_predictor.onnx",
        te_path=ONNX / "text_encoder.onnx",
        ve_path=MODEL_ROOT / "vector_estimator.onnx",
        voc_path=ONNX / "vocoder.onnx",
    )
    indexer = teacher.UnicodeIndexer(ONNX / "unicode_indexer.json")
    ttl, dp = teacher.load_voice_style(MODEL_ROOT / "voice_styles" / f"{a.voice}.json")

    for name, (text, speed) in CLIPS.items():
        out = a.out / f"{name}.wav"
        if out.exists() and not a.force:
            print(f"skip {out}")
            continue
        audio = teacher.synthesize(**common, indexer=indexer, style_ttl=ttl, style_dp=dp,
                                   text=text, lang="id", seed=0, total_steps=8,
                                   speed=speed, threads=2)
        teacher.write_wav(out, audio)
        print(f"{out} {len(audio)/44100:.2f}s voice={a.voice} speed={speed} text={text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
