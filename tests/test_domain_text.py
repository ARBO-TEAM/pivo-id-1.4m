"""Unit tests for src_id/domain_text.py.

Run from a trusted working directory (not inside the downloaded repos):

    python -I -m unittest discover -s tests -v

Stdlib only. Extract + split run against a small in-memory fixture so the
tests never need the workspace source file; one test additionally checks the
generated manifest on disk when it exists.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src_id"))

import domain_text as dt  # noqa: E402

FIXTURE = """\
# Dataset

## 3. Jawaban
**Q: Harga cireng berapa?**
A: Cireng crispy Rp1.500 per pcs. Harga per November 2026 dan dapat berubah.

**Q: Ada pesanan?**
A: Hari ini ada {jumlah} pesanan.

## 4. Bentuk salah
- Sekitar dua puluh lima ribu.

## 5. Kalimat gaya untuk TTS
### 5.1 Salam
- Halo, selamat datang.
- Halo selamat datang.

### 5.7 Nada
- Maaf, produk itu belum tersedia. *(sopan, menutup)*

### 5.10 Kalimat mentah
| Mentah | Harus diucapkan |
|---|---|
| `Rp25.000` | dua puluh lima ribu rupiah |

## 6. Penutup
- Terima kasih, semoga membantu.
"""


def load_fixture():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.md"
        src.write_text(FIXTURE, encoding="utf-8")
        rows, exclusions = dt.extract(src)
    return dt.assign_splits(dt.group_rows(rows)), exclusions


class TestExtraction(unittest.TestCase):
    def test_counts_by_kind(self):
        rows, _ = load_fixture()
        kinds = [r["kind"] for r in rows]
        self.assertEqual(kinds.count("tts_sentence"), 4)   # 5.1 x2, 5.7, 6
        self.assertEqual(kinds.count("assistant_response"), 1)
        self.assertEqual(kinds.count("symbol_expansion"), 1)

    def test_placeholder_and_negative_section_excluded(self):
        rows, exclusions = load_fixture()
        texts = [r["text"] for r in rows]
        self.assertFalse(any("{" in t for t in texts))
        self.assertNotIn("Sekitar dua puluh lima ribu.", texts)
        self.assertGreaterEqual(exclusions["placeholder"], 1)
        self.assertGreaterEqual(exclusions["section_excluded"], 1)

    def test_annotation_stripped_from_text(self):
        rows, _ = load_fixture()
        row = next(r for r in rows if r["category"] == "5.7")
        self.assertEqual(row["text"], "Maaf, produk itu belum tersedia.")
        self.assertTrue(row["annotation_removed"])

    def test_question_kept_with_response(self):
        rows, _ = load_fixture()
        row = next(r for r in rows if r["kind"] == "assistant_response")
        self.assertEqual(row["question"], "Harga cireng berapa?")
        self.assertTrue(row["contains_price"])

    def test_symbol_expansion_fields(self):
        rows, _ = load_fixture()
        row = next(r for r in rows if r["kind"] == "symbol_expansion")
        self.assertEqual(row["raw"], "Rp25.000")
        self.assertEqual(row["spoken_expected"], "dua puluh lima ribu rupiah")
        self.assertEqual(row["tts_usage"], "eval_only")

    def test_provenance_fields_present(self):
        rows, _ = load_fixture()
        for row in rows:
            self.assertTrue(row["source_heading"])
            self.assertGreaterEqual(row["source_line"], 1)
            self.assertIn("split", row)


class TestSplit(unittest.TestCase):
    def test_near_duplicates_share_group_and_split(self):
        rows, _ = load_fixture()
        dupes = [r for r in rows if r["text"] in
                 {"Halo, selamat datang.", "Halo selamat datang."}]
        self.assertEqual(len(dupes), 2)
        self.assertEqual(dupes[0]["group_id"], dupes[1]["group_id"])
        self.assertEqual(dupes[0]["split"], dupes[1]["split"])
        self.assertEqual(dupes[0]["group_size"], 2)

    def test_no_sentence_overlap_across_splits(self):
        rows, _ = load_fixture()
        split_by_text = {}
        for row in rows:
            key = dt.normalize(row["text"])
            if key in split_by_text:
                self.assertEqual(split_by_text[key], row["split"])
            split_by_text[key] = row["split"]

    def test_split_residues(self):
        self.assertEqual(dt.split_for_residue(0), "test")
        self.assertEqual(dt.split_for_residue(1), "test")
        self.assertEqual(dt.split_for_residue(2), "val")
        self.assertEqual(dt.split_for_residue(3), "val")
        self.assertEqual(dt.split_for_residue(4), "train")
        self.assertEqual(dt.split_for_residue(9), "train")

    def test_symbol_expansion_forced_test(self):
        rows, _ = load_fixture()
        for row in rows:
            if row["kind"] == "symbol_expansion":
                self.assertEqual(row["split"], "test")
                # Normalizer eval must key on the raw symbol, not the spoken form.
                self.assertEqual(row["raw"], "Rp25.000")
                self.assertEqual(row["spoken_expected"], row["text"])

    def test_deterministic(self):
        rows_a, _ = load_fixture()
        rows_b, _ = load_fixture()
        self.assertEqual(
            [(r["text"], r["group_id"], r["split"]) for r in rows_a],
            [(r["text"], r["group_id"], r["split"]) for r in rows_b],
        )


class TestManifestFile(unittest.TestCase):
    def test_manifest_if_present(self):
        manifest = (Path(dt.REPO_ROOT) / "experiments" / "text"
                    / "domain_manifest.jsonl")
        if not manifest.exists():
            self.skipTest("manifest not generated yet")
        rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l]
        self.assertTrue(rows)
        for row in rows:
            self.assertNotIn("{", row["text"])
            if row["kind"] == "symbol_expansion":
                self.assertEqual(row["split"], "test")
                self.assertEqual(row["tts_usage"], "eval_only")
        if len({r["split"] for r in rows}) > 1:
            split_by_text = {}
            for row in rows:
                key = dt.normalize(row["text"])
                if key in split_by_text:
                    self.assertEqual(split_by_text[key], row["split"])
                split_by_text[key] = row["split"]


if __name__ == "__main__":
    unittest.main()
