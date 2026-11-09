"""Parallel pilot corpus builder (pilot_v6_id): N OS processes synth, main aligns.

    python -I experiments/build_pilot_corpus_fast.py --out experiments/data/pilot_v6_id \
        --model-root ./supertonic_work \
        --aligner-dir ./models/hf/.../wav2vec2-indonesian.../snapshots/... \
        --manifest .tmp-out/v6_manifest.jsonl --jobs 6
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))
sys.path.insert(0, str(ROOT / "experiments"))

import ctc_align  # noqa: E402
import teacher  # noqa: E402


def worker_main(out_root, conn):
    """Runs in child: loads sessions once, synthesizes queued jobs, reports back."""
    import build_pilot_corpus as B

    out = Path(out_root)
    (out / "wav44").mkdir(parents=True, exist_ok=True)
    (out / "wav16").mkdir(parents=True, exist_ok=True)
    (out / "feat").mkdir(parents=True, exist_ok=True)

    import numpy as np

    def synth(sessions, indexer, ttl, dp, text, seed):
        text_ids, text_mask = indexer.encode(text, "id")
        duration = sessions["dp"].run(None, {"text_ids": text_ids, "style_dp": dp, "text_mask": text_mask})[0] / 1.05
        text_emb = sessions["te"].run(None, {"text_ids": text_ids, "style_ttl": ttl, "text_mask": text_mask})[0]
        chunk_size = teacher.BASE_CHUNK_SIZE * teacher.CHUNK_COMPRESS_FACTOR
        wav_len_max = duration.max() * teacher.SAMPLE_RATE
        latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)
        rng = np.random.RandomState(seed)
        latent_dim = teacher.LATENT_DIM * teacher.CHUNK_COMPRESS_FACTOR
        latent = rng.randn(1, latent_dim, latent_len).astype(np.float32)
        latent_mask = np.ones((1, 1, latent_len), dtype=np.float32)
        total_step = np.array([8.0], dtype=np.float32)
        for step in range(8):
            latent = sessions["ve"].run(
                None,
                {
                    "noisy_latent": latent,
                    "text_emb": text_emb,
                    "style_ttl": ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": np.array([float(step)], dtype=np.float32),
                    "total_step": total_step,
                },
            )[0]
        waveform = np.asarray(sessions["voc"].run(None, {"latent": latent})[0], dtype=np.float32)[0]
        return waveform.astype(np.float32)

    # Load once per child
    common = {}
    voices = {}
    while True:
        msg = conn.recv()
        if msg[0] == "init":
            common, voices, indexer = msg[1], msg[2], msg[3]
            sessions = {}
            for name in voices:
                sessions[name] = dict(
                    dp=teacher._load_session(Path(common["dp_path"]), 1, ["CPUExecutionProvider"]),
                    te=teacher._load_session(Path(common["te_path"]), 1, ["CPUExecutionProvider"]),
                    ve=teacher._load_session(Path(common["ve_path"]), 1, ["CPUExecutionProvider"]),
                    voc=teacher._load_session(Path(common["voc_path"]), 1, ["CPUExecutionProvider"]),
                )
            conn.send(("ready",))
        elif msg[0] == "job":
            name, r, seed = msg[1], msg[2], msg[3]
            rid = f"{name}_{r['id']}_s{seed}"
            try:
                if name not in voices:
                    raise RuntimeError(f"unknown voice {name}")
                ttl, dp = voices[name]
                audio = synth(sessions[name], indexer, ttl, dp, r["text"], seed)
                wav44 = out / "wav44" / f"{rid}.wav"
                teacher.write_wav(wav44, audio)
                pcm, rate = ctc_align.read_wav_mono(wav44)
                pcm16 = ctc_align.resample_to_16k(pcm, rate)
                wav16 = out / "wav16" / f"{rid}.wav"
                B.write_wav16(wav16, pcm16)
                conn.send(("done", rid, r["id"], r["text"], r["split"], name, seed))
            except Exception as exc:  # noqa: BLE001
                conn.send(("error", rid, f"{type(exc).__name__}: {exc}"[:200]))
        elif msg[0] == "stop":
            conn.close()
            return


def main():
    import json
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--aligner-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=ROOT / ".tmp-out" / "v6_manifest.jsonl")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()

    out = a.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to reuse existing {out} (delete it first)")
    out.mkdir(parents=True)
    (out / "wav44").mkdir()
    (out / "wav16").mkdir()
    (out / "feat").mkdir()

    onnx = a.model_root / "hf_fresh_download_20261108" / "onnx"
    common = dict(
        dp_path=str(onnx / "duration_predictor.onnx"),
        te_path=str(onnx / "text_encoder.onnx"),
        ve_path=str(a.model_root / "vector_estimator.onnx"),
        voc_path=str(onnx / "vocoder.onnx"),
    )
    indexer = teacher.UnicodeIndexer(onnx / "unicode_indexer.json")
    voices = {}
    for name in ("F1", "M1"):
        ttl, dp = teacher.load_voice_style(a.model_root / "voice_styles" / f"{name}.json")
        voices[name] = [ttl, dp]

    manifest_rows = [json.loads(l) for l in a.manifest.open(encoding="utf-8") if l.strip()]
    picked = [r for r in manifest_rows if r["split"] in ("train", "val")]
    jobs = []
    for name in voices:
        for seed in (0, 1):
            for r in picked:
                jobs.append((name, r, seed))
    print(f"texts: {len(picked)} train/val, jobs={len(jobs)}, workers={a.jobs}", flush=True)

    processor, ctc_model = ctc_align.load_ctc(a.aligner_dir)
    import numpy as np

    from build_pilot_corpus import extract_features

    ctx = mp.get_context("spawn")
    worker_conns = []
    procs = []
    for i in range(a.jobs):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=worker_main, args=(str(out), child_conn))
        p.start()
        parent_conn.send(("init", common, voices, indexer))
        assert parent_conn.recv()[0] == "ready"
        worker_conns.append(parent_conn)
        procs.append(p)

    records, meta, skipped = [], [], []
    recv = 0
    sent = 0
    # prefetch one job per worker
    for i in range(a.jobs):
        if sent < len(jobs):
            name, r, seed = jobs[sent]
            worker_conns[i].send(("job", name, r, seed))
            sent += 1
    while recv < len(jobs):
        progressed = False
        for i, conn in enumerate(worker_conns):
            if not conn.poll(0.02):
                continue
            msg = conn.recv()
            progressed = True
            if msg[0] == "error":
                skipped.append({"id": msg[1], "reason": msg[2]})
            else:
                _, rid, source_id, text, split, name, seed = msg
                vid = 0 if name == "F1" else 1
                wav16 = out / "wav16" / f"{rid}.wav"
                feats = extract_features(wav16, out / "feat" / f"{rid}.npy")
                try:
                    al = ctc_align.align(wav16, text, processor=processor, model=ctc_model,
                                         model_id=a.aligner_dir.name, lang="id")
                except ctc_align.AlignError as exc:
                    skipped.append({"id": rid, "reason": f"{type(exc).__name__}: {exc}"})
                    recv += 1
                    if sent < len(jobs):
                        name, r, seed = jobs[sent]
                        worker_conns[i].send(("job", name, r, seed))
                        sent += 1
                    continue
                feat = feats[al.start_frame: al.end_frame]
                np.save(out / "feat" / f"{rid}.npy", feat.astype(np.float32))
                records.append({"id": rid, "tokens": al.token_ids, "durations": al.durations,
                                "features_path": f"feat/{rid}.npy", "voice_id": vid,
                                "split": "train" if split == "train" else "val"})
                meta.append({"id": rid, "source_id": source_id, "split": split, "voice": name,
                             "voice_id": vid, "text": text, "seed": seed,
                             "features_frames": int(feat.shape[0]), "durations_sum": sum(al.durations),
                             "alarms": al.alarms, "readback": al.readback, "model_id": al.model_id})
            recv += 1
            if recv % 100 == 0:
                print(f"aligned {recv}/{len(jobs)}", flush=True)
            if sent < len(jobs):
                name, r, seed = jobs[sent]
                worker_conns[i].send(("job", name, r, seed))
                sent += 1
        if not progressed:
            # wait for any worker
            for conn in worker_conns:
                if conn.poll(0.5):
                    break

    for conn in worker_conns:
        conn.send(("stop",))
    for p in procs:
        p.join()

    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (out / "build_meta.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta), encoding="utf-8")
    (out / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    counts = {"train": sum(r["split"] == "train" for r in records),
              "val": sum(r["split"] == "val" for r in records), "skipped": len(skipped)}
    (out / "summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"DONE {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
