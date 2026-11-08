"""Merge several built corpora into one training directory.

Feature files are copied into ``<out>/feat/`` with ids prefixed by the source
corpus name, so identical teacher ids (both corpora use F1_E001_s0 style ids
for different languages) cannot collide. Splits are kept as-is; the merged
manifest is then exactly what ``src_id/train.py --data`` expects.

    python -I experiments/merge_corpora.py --out experiments/data/pilot_v5 \
        --corpora id=experiments/data/pilot_v4 en=experiments/data/en_pilot_v1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpora", nargs="+", required=True, help="label=path pairs")
    a = ap.parse_args()

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "feat").mkdir()

    records, meta_rows, counts = [], [], {}
    for spec in a.corpora:
        label, src = spec.split("=", 1)
        src = Path(src).resolve()
        rows = [json.loads(l) for l in (src / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
        counts[label] = {"rows": len(rows)}
        for m in (json.loads(l) for l in (src / "build_meta.jsonl").open(encoding="utf-8") if l.strip()):
            m = dict(m)
            m["id"] = f"{label}_{m['id']}"
            meta_rows.append(m)
        for r in rows:
            rid = f"{label}_{r['id']}"
            src_feat = src / r["features_path"]
            dst = out / "feat" / f"{rid}.npy"
            shutil.copyfile(src_feat, dst)
            n = int(np.load(dst, mmap_mode="r", allow_pickle=False).shape[0])
            if n != sum(r["durations"]):
                raise SystemExit(f"{rid}: feature frames {n} != duration sum {sum(r['durations'])}")
            records.append({
                "id": rid, "tokens": r["tokens"], "durations": r["durations"],
                "features_path": f"feat/{rid}.npy", "voice_id": r["voice_id"],
                "split": r["split"], "source": label, "source_id": r["id"],
            })
        print(f"{label}: {len(rows)} rows", flush=True)

    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids after merge")
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with (out / "build_meta.jsonl").open("w", encoding="utf-8") as f:
        for m in meta_rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    summary = {"sources": counts,
               "train": sum(r["split"] == "train" for r in records),
               "val": sum(r["split"] == "val" for r in records),
               "frames": sum(sum(r["durations"]) for r in records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"DONE {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
