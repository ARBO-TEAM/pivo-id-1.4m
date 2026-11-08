"""Check that FARGAN features from real audio reproduce intelligible speech
through the pinned PiVo streaming vocoder (no training involved).

    python -I experiments/roundtrip_probe.py --wav <teacher 44.1k wav> \
        --model-dir PiVo-1.4M --out <new wav>
"""
from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))
sys.path.insert(0, str(ROOT / "experiments"))

import ctc_align  # noqa: E402
from render_student import features_to_audio, write_wav16  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--fargan", type=Path, default=ROOT / "opus-build-fargan" / "build" / "fargan_demo.exe")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    pcm, rate = ctc_align.read_wav_mono(a.wav)
    pcm16 = ctc_align.resample_to_16k(pcm, rate)
    tmp = a.out.parent / (a.out.stem + "_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    pcm_path = tmp / "in.pcm"
    pcm_path.write_bytes(pcm16.astype("<i2").tobytes())
    raw = tmp / "feat.f32"
    subprocess.run([str(a.fargan), "-features", str(pcm_path), str(raw)], check=True)
    feats = np.fromfile(raw, dtype="<f4").reshape(-1, 20).astype(np.float32)
    wav = features_to_audio(a.model_dir, feats)
    write_wav16(a.out, wav)
    pcm_path.unlink()
    raw.unlink()
    tmp.rmdir()
    print(f"wrote {a.out}: {len(wav)/16000:.2f}s, frames={feats.shape[0]}, peak={np.abs(wav).max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
