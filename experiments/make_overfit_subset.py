"""Build a tiny overfit subset from an existing corpus manifest (plan Task 3.1).

    python -I experiments/make_overfit_subset.py --data experiments/data/pilot_v3 \
        --out experiments/data/overfit8 --train 8 --val 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--val", type=int, default=2)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(l) for l in (a.data / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
    train = [r for r in rows if r["split"] == "train"][:a.train]
    val = [r for r in rows if r["split"] == "val"][:a.val]
    if len(train) < a.train or not val:
        raise SystemExit("not enough rows")
    out_rows = []
    for r in train + val:
        out_rows.append(dict(r, features_path=str((a.data / r["features_path"]).resolve())))
    with (a.out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
