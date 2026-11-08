"""Build the Indonesian pilot corpus: text -> teacher WAV -> 16 kHz -> FARGAN
features -> CTC phone durations -> training JSONL.

Every step is external-tool driven and records what it did; nothing is invented.
Refuses to reuse an existing output directory. Refuses to mix splits.

Usage:
  python -I experiments/build_pilot_corpus.py --out experiments/data/pilot_v1 \
      --model-root E:/test/tts-test/split-supertonic-v3/supertonic_work \
      --aligner-dir E:/.../wav2vec2-indonesian... --voices F1:0 M1:1 \
      --limit 40 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))

import ctc_align  # noqa: E402
import teacher  # noqa: E402

FARGAN = ROOT / "opus-build-fargan" / "build" / "fargan_demo.exe"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_wav16(path: Path, pcm16: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16.astype("<i2").tobytes())


def extract_features(wav16: Path, out_npy: Path) -> np.ndarray:
    pcm, rate = ctc_align.read_wav_mono(wav16)
    assert rate == 16000
    raw = out_npy.with_suffix(".f32")
    pcm_path = out_npy.with_suffix(".pcm")
    pcm_path.write_bytes(pcm.astype("<i2").tobytes())
    subprocess.run([str(FARGAN), "-features", str(pcm_path), str(raw)], check=True,
                   capture_output=True)
    feats = np.fromfile(raw, dtype="<f4")
    if feats.size % 20:
        raise RuntimeError(f"{wav16}: feature count {feats.size} not divisible by 20")
    feats = feats.reshape(-1, 20)
    if not np.isfinite(feats).all():
        raise RuntimeError(f"{wav16}: non-finite features")
    np.save(out_npy, feats.astype(np.float32))
    pcm_path.unlink()
    raw.unlink()
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--aligner-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=ROOT / "experiments/text/domain_manifest.jsonl")
    ap.add_argument("--voices", nargs="+", default=["F1:0", "M1:1"])
    ap.add_argument("--limit", type=int, default=40, help="train rows per voice (0 = all)")
    ap.add_argument("--val-limit", type=int, default=8, help="val rows per voice (0 = all)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="teacher seeds; each seed yields a distinct take of every text")
    ap.add_argument("--seed", type=int, default=0, help="deprecated: use --seeds")
    ap.add_argument("--lang", choices=["id", "en"], default="id",
                    help="teacher/frontend/aligner language; en uses PiVo's CMU frontend "
                         "and an English CTC aligner")
    a = ap.parse_args()

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "wav44").mkdir()
    (out / "wav16").mkdir()
    (out / "feat").mkdir()

    onnx = a.model_root / "hf_fresh_download_20261108" / "onnx"
    common = dict(
        dp_path=onnx / "duration_predictor.onnx",
        te_path=onnx / "text_encoder.onnx",
        ve_path=a.model_root / "vector_estimator.onnx",
        voc_path=onnx / "vocoder.onnx",
    )
    indexer = teacher.UnicodeIndexer(onnx / "unicode_indexer.json")
    voices = []
    for spec in a.voices:
        name, vid = spec.split(":")
        ttl, dp = teacher.load_voice_style(a.model_root / "voice_styles" / f"{name}.json")
        voices.append((name, int(vid), ttl, dp))

    rows = [json.loads(l) for l in a.manifest.open(encoding="utf-8") if l.strip()]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    picked = ([("train", r) for r in (train if a.limit == 0 else train[: a.limit])] +
              [("val", r) for r in (val if a.val_limit == 0 else val[: a.val_limit])])
    if not picked:
        raise SystemExit("no manifest rows selected")

    processor, ctc_model = ctc_align.load_ctc(a.aligner_dir)
    en_phonemize = None
    if a.lang == "en":
        sys.path.insert(0, str(ROOT / "PiVo-1.4M"))
        from pivo.text import phonemize as en_phonemize  # PiVo's CMU-dictionary frontend
    meta, records = [], []
    skipped = []
    seeds = list(dict.fromkeys(a.seeds))
    for name, vid, ttl, dp in voices:
        for seed in seeds:
            for split, r in picked:
                rid = f"{name}_{r['id']}_s{seed}"
                try:
                    audio = teacher.synthesize(
                        **common, indexer=indexer, style_ttl=ttl, style_dp=dp,
                        text=r["text"], lang=a.lang, seed=seed, total_steps=8,
                        speed=1.05, threads=2)
                    wav44 = out / "wav44" / f"{rid}.wav"
                    teacher.write_wav(wav44, audio)
                    pcm, rate = ctc_align.read_wav_mono(wav44)
                    pcm16 = ctc_align.resample_to_16k(pcm, rate)
                    wav16 = out / "wav16" / f"{rid}.wav"
                    write_wav16(wav16, pcm16)
                    feats = extract_features(wav16, out / "feat" / f"{rid}.npy")
                    al = ctc_align.align(wav16, r["text"], processor=processor,
                                         model=ctc_model, model_id=a.aligner_dir.name,
                                         lang=a.lang, en_phonemize=en_phonemize)
                except Exception as exc:  # noqa: BLE001 - record and continue the pilot
                    skipped.append({"id": rid, "split": split, "reason": f"{type(exc).__name__}: {exc}"})
                    continue
                # Clip to the aligned region so sum(durations) == feature frames.
                feat = feats[al.start_frame: al.end_frame]
                np.save(out / "feat" / f"{rid}.npy", feat.astype(np.float32))
                rec = {
                    "id": rid, "tokens": al.token_ids, "durations": al.durations,
                    "features_path": f"feat/{rid}.npy", "voice_id": vid,
                    "split": "train" if split == "train" else "val",
                }
                records.append(rec)
                meta.append({
                    "id": rid, "source_id": r["id"], "split": split, "voice": name,
                    "voice_id": vid, "text": r["text"], "seed": seed,
                    "teacher_config": {k: str(v) for k, v in common.items()},
                    "teacher_code_sha256": sha256(Path(teacher.__file__)),
                    "wav44_sha256": sha256(wav44), "wav16_sha256": sha256(wav16),
                    "features_frames": int(feat.shape[0]), "durations_sum": sum(al.durations),
                    "alarms": al.alarms, "readback": al.readback,
                    "model_id": al.model_id,
                })
                print(f"{rid}: frames={feat.shape[0]} alarms={len(al.alarms)}", flush=True)

    if not records:
        raise SystemExit(f"no rows built; first skips: {skipped[:5]}")
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with (out / "build_meta.jsonl").open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    (out / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    counts = {"train": sum(r["split"] == "train" for r in records),
              "val": sum(r["split"] == "val" for r in records), "skipped": len(skipped)}
    (out / "summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"DONE {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
