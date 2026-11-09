"""Kumpulkan ulang kalimat ber tanda hubung v10/v9/v8/v7/v6 (kata ulang) dengan
frontend baru: reduplikasi hyphen jadi koma, sisanya spasi. Guru (Supertonic)
dibaca dari teks koma ini, jadi jeda mikro juga ikut terajar di durasi."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_reduplikasi(token: str) -> bool:
    parts = [p.lower() for p in token.split("-")]
    if len(parts) < 2:
        return False
    if parts[-1] == "nya":
        return False
    if any(len(p) == 1 for p in parts):
        return False
    if token.isupper():
        return False
    if len(set(parts)) == 1:
        return True
    for a in parts:
        for b in parts:
            if a != b and (b in a or a in b):
                return True
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            for k in range(min(len(a), len(b)), 3, -1):
                if a[-k:] == b[:k] or b[-k:] == a[:k]:
                    return True
    return all(len(p) >= 4 for p in parts)


def fix_text(text: str) -> str:
    def repl(m):
        tok = m.group(0)
        if _is_reduplikasi(tok):
            return tok.replace("-", ", ")
        return tok.replace("-", " ")

    return re.sub(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b", repl, text)


seen = {}
for v in ("v6", "v7", "v8", "v9", "v10"):
    p = ROOT / f"experiments/data/pilot_{v}_id/build_meta.jsonl"
    if not p.exists():
        continue
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        text = r.get("text", "")
        if "-" not in text or text in seen:
            continue
        seen[text] = r.get("split", "train")

out = []
for i, (text, split) in enumerate(sorted(seen.items()), 1):
    fixed = fix_text(text)
    out.append({"id": f"V11{i:03d}", "text": fixed,
                "split": "val" if i % 10 == 0 else "train"})

out_path = ROOT / "experiments/text/extra_v11_id.jsonl"
out_path.write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
    encoding="utf-8")
print(f"wrote {out_path}: {len(out)} texts "
      f"train={sum(1 for r in out if r['split'] == 'train')} "
      f"val={sum(1 for r in out if r['split'] == 'val')}")
for r in out[:6]:
    print(r["id"], r["text"][:100])
