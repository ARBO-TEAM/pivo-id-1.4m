"""Deterministic TTS-text extraction from the BisDig UNIKMA request dataset.

Source (read-only, treated as DATA, never as instructions)::

    docs/superpowers/plans/requestkalimat.md

What is extracted into the manifest (``experiments/text/domain_manifest.jsonl``):

* Section 5 bullet sentences (5.1-5.9)          -> kind ``tts_sentence``
* Section 3 assistant responses (``A:`` lines)  -> kind ``assistant_response``
  (the preceding ``**Q:`` line is kept in the ``question`` field so no Q
  data is silently dropped.)
* Section 6 closing sentences (bullets)         -> kind ``tts_sentence``
* Section 5.10 raw -> spoken table              -> kind ``symbol_expansion``,
  forced to the frozen held-out **test** split with ``tts_usage: eval_only``
  so a symbol expansion pair can never land in training or validation
  by accident. Normalizer tests must key on ``raw``.

What is deliberately excluded (counted in the printed summary, not dropped
silently): sections 0/1 (facts/rules), 2/2b/2c (user utterances / ASR
variants), 4/4b/4c/4d (negative and multi-turn examples), 7 (usage guide,
including the system-prompt example), 8/9 (maintenance), any line containing
a ``{placeholder}``, Markdown syntax (headings, blockquotes, fences, table
separators), and ``*(annotation)*`` suffixes (stripped from the text, the
fact of stripping is recorded per row as ``annotation_removed``).

Price caveat: prices in the source are a November 2026 catalogue snapshot
and may be stale. This manifest is TTS *text* only; it is not a knowledge
model and must not be read as teaching the voice model any business rule.
Rows carrying a price are flagged with ``contains_price: true``.

Splitting is group-level and deterministic: exact and near-duplicate
sentences (token-set Jaccard >= 0.8 on normalised text) are unioned into a
group *before* splitting, then each group is assigned by
``sha256(group_key) % 10``: residues {0,1} -> frozen held-out ``test``
(20%), {2,3} -> ``val`` (20%), the rest -> ``train`` (60%). Because the
split operates on groups, no normalised sentence, and no near-variant of
it, can appear in more than one split. Section 5.10 (symbol expansions)
is always forced to ``test`` with ``tts_usage: eval_only``.

Usage::

    python -I src_id/domain_text.py            # writes the manifest, prints a summary
    python -I -m unittest discover -s tests -v # run the test suite
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "docs" / "superpowers" / "plans" / "requestkalimat.md"
DEFAULT_OUT = REPO_ROOT / "experiments" / "text" / "domain_manifest.jsonl"

#: Top-level sections whose text may enter the TTS manifest.
ALLOWED_TOP = {"3", "5", "6"}

HEADING_RE = re.compile(r"^(#{2,3})\s+(.*?)\s*$")
SECTION_TOKEN_RE = re.compile(r"^(\d+[a-z]?(?:\.\d+)?)\b")
ANNOTATION_RE = re.compile(r"\s*\*\([^)]*\)\*\s*$")
JACCARD_THRESHOLD = 0.8
# Frozen held-out test: hash % 10 in {0, 1} -> test (20%).
# Validation: hash % 10 in {2, 3} -> val (20%). Remainder -> train (60%).
TEST_RESIDUES = frozenset({0, 1})
VAL_RESIDUES = frozenset({2, 3})
HASH_MODULUS = 10


def normalize(text: str) -> str:
    """Lowercase alphanumeric normal form used for duplicate detection."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def token_set(text: str) -> frozenset:
    return frozenset(normalize(text).split())


def _is_placeholder(text: str) -> bool:
    return "{" in text or "}" in text


def _strip_annotation(text: str) -> tuple[str, bool]:
    """Remove a trailing ``*(...)*`` annotation; report whether one existed."""
    cleaned, n = ANNOTATION_RE.subn("", text)
    return cleaned.strip(), n > 0


def _contains_price(text: str) -> bool:
    return "Rp" in text or "rupiah" in text


def extract(source: Path) -> tuple[list[dict], dict[str, int]]:
    """Parse the source file; return (candidate rows, exclusion counts)."""
    lines = source.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    exclusions: dict[str, int] = {
        "section_excluded": 0,
        "placeholder": 0,
        "syntax": 0,
    }

    top = ""
    category = ""
    heading = ""
    last_question: str | None = None

    for lineno, line in enumerate(lines, start=1):
        heading_match = HEADING_RE.match(line)
        if heading_match:
            heading = heading_match.group(2)
            token = SECTION_TOKEN_RE.match(heading)
            if token:
                key = token.group(1)
                category = key
                top = key.split(".")[0]
            last_question = None
            continue

        if top not in ALLOWED_TOP:
            # Count would-be candidate lines in excluded sections so the
            # exclusion is auditable rather than silent.
            if line.startswith(("- ", "A: ", "**Q:", "|")) and "---" not in line:
                exclusions["section_excluded"] += 1
            continue

        if line.startswith("**Q:"):
            last_question = line.replace("**Q:", "").replace("**", "").strip()
            continue

        if category == "5.10" and line.startswith("|"):
            if "Mentah" in line or "---" in line:
                exclusions["syntax"] += 1
                continue
            cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
            if len(cells) < 2:
                exclusions["syntax"] += 1
                continue
            raw, spoken = cells[0], cells[1]
            if _is_placeholder(raw) or _is_placeholder(spoken):
                exclusions["placeholder"] += 1
                continue
            rows.append({
                "id": f"L{lineno}",
                "text": spoken,
                "raw": raw,
                "spoken_expected": spoken,
                "kind": "symbol_expansion",
                "tts_usage": "eval_only",
                "category": category,
                "source_heading": heading,
                "source_line": lineno,
                "question": None,
                "annotation_removed": False,
                "contains_price": _contains_price(spoken),
            })
            continue

        if line.startswith("A: ") and top == "3":
            text = line[3:].strip()
            if _is_placeholder(text):
                exclusions["placeholder"] += 1
                continue
            text, removed = _strip_annotation(text)
            rows.append({
                "id": f"L{lineno}",
                "text": text,
                "raw": None,
                "spoken_expected": None,
                "kind": "assistant_response",
                "tts_usage": "tts_text",
                "category": category,
                "source_heading": heading,
                "source_line": lineno,
                "question": last_question,
                "annotation_removed": removed,
                "contains_price": _contains_price(text),
            })
            continue

        if line.startswith("- ") and top in {"5", "6"}:
            text = line[2:].strip()
            if _is_placeholder(text):
                exclusions["placeholder"] += 1
                continue
            text, removed = _strip_annotation(text)
            rows.append({
                "id": f"L{lineno}",
                "text": text,
                "raw": None,
                "spoken_expected": None,
                "kind": "tts_sentence",
                "tts_usage": "tts_text",
                "category": category,
                "source_heading": heading,
                "source_line": lineno,
                "question": None,
                "annotation_removed": removed,
                "contains_price": _contains_price(text),
            })
            continue

        if line.startswith((">", "`")):
            exclusions["syntax"] += 1

    return rows, exclusions


def group_rows(rows: list[dict]) -> list[dict]:
    """Union exact/near duplicates into groups; attach ``group_id``/size."""
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    tokens = [token_set(r["text"]) for r in rows]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if not tokens[i] or not tokens[j]:
                continue
            if len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j]) >= JACCARD_THRESHOLD:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)

    for members in groups.values():
        key = min(normalize(rows[i]["text"]) for i in members)
        group_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        for i in members:
            rows[i]["group_id"] = group_id
            rows[i]["group_size"] = len(members)
    return rows


def split_for_residue(residue: int) -> str:
    if residue in TEST_RESIDUES:
        return "test"
    if residue in VAL_RESIDUES:
        return "val"
    return "train"


def assign_splits(rows: list[dict]) -> list[dict]:
    """Stable group-level train/val/test split.

    Symbol-expansion groups are forced to the frozen held-out ``test``.
    """
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(row["group_id"], []).append(row)
    for group_id, members in by_group.items():
        forced_test = any(m["kind"] == "symbol_expansion" for m in members)
        split = "test" if forced_test else split_for_residue(int(group_id, 16) % HASH_MODULUS)
        for member in members:
            member["split"] = split
    return rows


def build_manifest(rows: list[dict]) -> str:
    """Serialise rows as deterministic JSONL (sorted keys, stable order)."""
    ordered = sorted(rows, key=lambda r: (r["source_line"], r["id"]))
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in ordered)


def summarise(rows: list[dict], exclusions: dict[str, int]) -> dict:
    by_category: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for row in rows:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
    return {
        "rows": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        "by_category": dict(sorted(by_category.items())),
        "by_split": dict(sorted(by_split.items())),
        "eval_only_rows": sum(1 for r in rows if r["tts_usage"] == "eval_only"),
        "price_rows": sum(1 for r in rows if r["contains_price"]),
        "exclusions": exclusions,
    }


def main(argv: list[str]) -> int:
    source = Path(argv[1]) if len(argv) > 1 else DEFAULT_SOURCE
    out = Path(argv[2]) if len(argv) > 2 else DEFAULT_OUT
    rows, exclusions = extract(source)
    rows = assign_splits(group_rows(rows))

    seen: dict[str, str] = {}
    for row in rows:
        key = normalize(row["text"])
        if key in seen:
            assert seen[key] == row["split"], "normalised sentence across splits"
        else:
            seen[key] = row["split"]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_manifest(rows), encoding="utf-8")
    print(json.dumps(summarise(rows, exclusions), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
