"""Pre-training gate on a built pilot corpus (read-only; no training).

Checks the artifacts the trainer will load: manifest/trainer schema agreement,
duration sums vs feature frames, split integrity, voice coverage, and a concise
alarm audit. Prints PASS/FAIL with counts. Refuses nothing; exit code is 1 on
any failed check.

    python -I experiments/audit_corpus.py --data experiments/data/pilot_v3
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args()
    d = a.data
    rows = [json.loads(l) for l in (d / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
    meta = ({m["id"]: m for m in (json.loads(l) for l in (d / "build_meta.jsonl").open(encoding="utf-8") if l.strip())}
            if (d / "build_meta.jsonl").exists() else {})
    fails: list[str] = []

    req = ("id", "tokens", "durations", "features_path", "voice_id", "split")
    for r in rows:
        missing = [k for k in req if k not in r]
        if missing:
            fails.append(f"{r.get('id')}: missing {missing}")

    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        fails.append("duplicate ids")
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if not train or not val:
        fails.append(f"empty split train={len(train)} val={len(val)}")
    if {r["id"] for r in train} & {r["id"] for r in val}:
        fails.append("id leakage train/val")

    frames_total = 0
    voices = collections.Counter()
    for r in rows:
        p = d / r["features_path"]
        if not p.is_file():
            fails.append(f"{r['id']}: missing {p}")
            continue
        f = np.load(p, allow_pickle=False)
        if f.ndim != 2 or f.shape[1] != 20:
            fails.append(f"{r['id']}: feature shape {f.shape}")
            continue
        if not np.isfinite(f).all():
            fails.append(f"{r['id']}: non-finite features")
        if f.shape[0] != sum(r["durations"]):
            fails.append(f"{r['id']}: {f.shape[0]} frames != dur sum {sum(r['durations'])}")
        if len(r["tokens"]) != len(r["durations"]):
            fails.append(f"{r['id']}: tokens/durations length mismatch")
        if max(r["durations"]) > 200 or not all(0 <= x for x in r["durations"]):
            fails.append(f"{r['id']}: duration out of range")
        frames_total += f.shape[0]
        voices[r["voice_id"]] += 1

    # Source text must not overlap across splits (voice prefix / seed suffix stripped).
    def src(rid: str) -> str:
        base = rid.split("_", 1)[1] if "_" in rid else rid
        return base.rsplit("_s", 1)[0] if "_s" in base else base
    if {src(r["id"]) for r in train} & {src(r["id"]) for r in val}:
        fails.append("same source sentence in train and val")

    alarms = collections.Counter()
    for m in meta.values():
        for al in m.get("alarms", []):
            alarms[al.split(":")[0]] += 1

    print(f"rows={len(rows)} train={len(train)} val={len(val)} voices={dict(voices)} frames={frames_total}")
    print(f"audio_seconds={frames_total/100:.1f} alarm_kinds={dict(alarms)}")
    if fails:
        print("FAIL")
        for f in fails[:20]:
            print(" -", f)
        return 1
    print("PASS: corpus satisfies trainer schema and invariants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
