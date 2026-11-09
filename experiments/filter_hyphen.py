"""Saring baris tanda hubung ber-ejaan-huruf dari korpus merged (buang 553 baris
sampah), hasilkan dir korpus terfilter + manifest untuk merge ulang dengan v11.

Usage:
    python -I experiments/filter_hyphen.py --in experiments/data/pilot_v10 \
        --out experiments/data/pilot_v10_filt --exclude-text ...  (tidak dipakai)

Cara kerja: baca build_meta merged, buang record yang teksnya mengandung '-',
salin feat + tulis manifest terfilter. Baris e-KTP/OSS-RBA/kilowatt-jam juga ikut
dibuang karena teks guru-nya ikut diubah v11 (spasi), jadi tidak ada duplikat arti.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=ROOT / "experiments/data/pilot_v10")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/data/pilot_v10_filt")
    a = ap.parse_args()
    src, out = a.src.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "feat").mkdir()

    meta = {}
    for line in (src / "build_meta.jsonl").open(encoding="utf-8"):
        if line.strip():
            m = json.loads(line)
            meta[m["id"]] = m

    kept, dropped = [], 0
    for line in (src / "manifest.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        m = meta.get(r["id"])
        if m is not None and "-" in m.get("text", ""):
            dropped += 1
            continue
        kept.append(r)
        shutil.copyfile(src / r["features_path"], out / r["features_path"])

    for r in kept:
        pass
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "build_meta.jsonl").open("w", encoding="utf-8") as f:
        for r in kept:
            m = meta.get(r["id"])
            if m is not None:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    summary = {"kept": len(kept), "dropped": dropped,
               "train": sum(r["split"] == "train" for r in kept),
               "val": sum(r["split"] == "val" for r in kept),
               "frames": sum(sum(r["durations"]) for r in kept)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2),
                                       encoding="utf-8")
    print(f"kept={summary['kept']} dropped={dropped} "
          f"train={summary['train']} val={summary['val']} "
          f"frames={summary['frames']}")


if __name__ == "__main__":
    main()
