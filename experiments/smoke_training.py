"""Exercise the training loop on invented data; not a speech-training dataset."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="pivo-invented-smoke-", dir=root / "temp") as name:
    work = Path(name)
    rng = np.random.default_rng(42)
    rows = []
    for split in ("train", "val"):
        path = work / f"{split}.npy"
        np.save(path, rng.standard_normal((12, 20), dtype=np.float32))
        rows.append({"id": split, "split": split, "tokens": [1, 2, 3],
                     "durations": [4, 4, 4], "voice_id": 0, "features_path": str(path)})
    data = work / "invented.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = work / "output"
    cmd = [sys.executable, "-I", str(root / "src_id" / "train.py"),
           "--data", str(data), "--out", str(out), "--from-scratch",
           "--steps", "2", "--batch-size", "1", "--device", "cuda",
           "--log-every", "1", "--eval-every", "1", "--save-every", "1"]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    assert (out / "best.pt").is_file() and (out / "last.pt").is_file()
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert any(x["kind"] == "train" and x["step"] == 2 for x in records)
    assert any(x["kind"] == "eval" and x["step"] == 2 for x in records)
    print("SMOKE PASS: invented features and durations only; no speech model produced")
