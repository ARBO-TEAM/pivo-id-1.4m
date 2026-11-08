"""Generate a small, review-pending Supertonic FP32 teacher pilot.

Run with the model environment's Python, e.g.:
python -I src_id/teacher_pilot.py --model-root MODEL_ROOT --output NEW_DIRECTORY
The output directory must not exist. Only evaluation probes are synthesized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teacher

ROOT = Path(__file__).resolve().parent.parent
PROBE_LINES = (1, 2)
SEED = 0
STEPS = 8
SPEED = 1.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voice-style", type=Path, default=None,
                        help="voice style JSON (default: <model-root>/voice_styles/F1.json)")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    model_root = args.model_root.resolve()
    onnx_dir = model_root / "hf_fresh_download_20261108" / "onnx"
    voice_path = args.voice_style or (model_root / "voice_styles" / "F1.json")
    paths = {
        "dp": onnx_dir / "duration_predictor.onnx",
        "te": onnx_dir / "text_encoder.onnx",
        "ve": model_root / "vector_estimator.onnx",
        "voc": onnx_dir / "vocoder.onnx",
        "config": onnx_dir / "tts.json",
        "indexer": onnx_dir / "unicode_indexer.json",
        "voice_style": voice_path,
    }
    cfg = json.loads(paths["config"].read_text(encoding="utf-8"))
    expected = {("ae", "sample_rate"): teacher.SAMPLE_RATE,
                ("ae", "base_chunk_size"): teacher.BASE_CHUNK_SIZE,
                ("ttl", "chunk_compress_factor"): teacher.CHUNK_COMPRESS_FACTOR,
                ("ttl", "latent_dim"): teacher.LATENT_DIM}
    for (section, key), value in expected.items():
        if cfg.get(section, {}).get(key) != value:
            raise ValueError(f"incompatible config {section}.{key}")
    probe = ROOT / "experiments" / "text" / "teacher_probe.txt"
    selected = []
    lines = probe.read_text(encoding="utf-8").splitlines()
    for line_no in PROBE_LINES:
        selected.append({"id": f"probe_{line_no:03d}", "split": "teacher_probe", "source": str(probe),
                         "source_line": line_no, "text": lines[line_no - 1]})

    model_hashes = {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}
    indexer = teacher.UnicodeIndexer(paths["indexer"])
    ttl, dp_style = teacher.load_voice_style(paths["voice_style"])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "teacher_probe").mkdir()
    voice_name = voice_path.stem
    voice_id = {"F1": 0, "M1": 1}.get(voice_name)
    metadata = {"status": "evaluation only; never use for training; pending licensing and human quality approval",
                "voice": voice_name, "voice_id": voice_id, "language": "id", "seed": SEED,
                "total_steps": STEPS, "speed": SPEED, "threads": args.threads,
                "provider": "CPUExecutionProvider", "estimator": "FP32",
                "model_files": model_hashes,
                "input_sha256": {str(probe): sha256(probe)},
                "teacher_sha256": sha256(Path(teacher.__file__)),
                "samples": []}
    for row in selected:
        audio = teacher.synthesize(dp_path=paths["dp"], te_path=paths["te"], ve_path=paths["ve"],
            voc_path=paths["voc"], indexer=indexer, style_ttl=ttl, style_dp=dp_style,
            text=row["text"], lang="id", seed=SEED, total_steps=STEPS, speed=SPEED, threads=args.threads)
        wav_path = out / row["split"] / (row["id"] + ".wav")
        teacher.write_wav(wav_path, audio)
        with wave.open(str(wav_path), "rb") as wav:
            fmt = {"channels": wav.getnchannels(), "sample_width_bytes": wav.getsampwidth(),
                   "sample_rate": wav.getframerate(), "frames": wav.getnframes(), "compression": wav.getcomptype()}
        if fmt != {"channels": 1, "sample_width_bytes": 2, "sample_rate": 44100,
                   "frames": len(audio), "compression": "NONE"}:
            raise ValueError(f"invalid WAV format: {wav_path}: {fmt}")
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if not np.isfinite(audio).all() or rms <= 0:
            raise ValueError(f"invalid or silent output: {wav_path}")
        row.update({"wav": str(wav_path.relative_to(out)), "wav_sha256": sha256(wav_path),
                    "wav_format": fmt, "duration_seconds": len(audio) / 44100,
                    "signal_float_before_pcm": {"rms": rms, "peak_abs": float(np.max(np.abs(audio))),
                    "fraction_abs_ge_1": float(np.mean(np.abs(audio) >= 1))}})
        metadata["samples"].append(row)
        print(f"Generated {wav_path}; rms={rms:.6f}", flush=True)
    with (out / "metadata.json").open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Review pending: {out}")


if __name__ == "__main__":
    main()
