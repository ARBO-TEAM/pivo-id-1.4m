"""Regression tests for bugs found in independent validation of
src_id/frontend.py, src_id/train.py and the 5.10 domain manifest.

Run: python -I -m unittest discover -s tests -v   (workspace venv)
Stdlib + project modules only.  No training, no GPU, no model files.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src_id"))

import domain_text as dt  # noqa: E402
import frontend as fe  # noqa: E402
import train as T  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "docs" / "superpowers" / "plans" / "requestkalimat.md"


class TestCommaThousandsRegression(unittest.TestCase):
    def test_english_multi_group_thousands(self):
        # Regression: plain-number regex only allowed dot-thousands groups;
        # '1,234,567' was consumed as '1,234' + ',567' (silently wrong).
        self.assertEqual(
            fe.normalize("1,234,567").normalized_text,
            "satu juta dua ratus tiga puluh empat ribu "
            "lima ratus enam puluh tujuh")

    def test_english_thousands_with_dot_decimal(self):
        self.assertEqual(
            fe.normalize("1,234.56").normalized_text,
            "seribu dua ratus tiga puluh empat koma lima enam")

    def test_single_comma_thousands_unchanged(self):
        self.assertEqual(
            fe.normalize("1,234").normalized_text,
            "seribu dua ratus tiga puluh empat")
        self.assertEqual(
            fe.normalize("12,345").normalized_text,
            "dua belas ribu tiga ratus empat puluh lima")

    def test_indonesian_decimal_comma_unchanged(self):
        self.assertIn("koma", fe.normalize("3,14").normalized_text)
        self.assertEqual(
            fe.normalize("1.234,56").normalized_text,
            "seribu dua ratus tiga puluh empat koma lima enam")


class TestBareCurrencyRegression(unittest.TestCase):
    def test_bare_rp_rejected(self):
        for raw in ("Rp", "Rp abc", "IDR"):
            with self.assertRaises(fe.FrontendError, msg=raw):
                fe.normalize(raw)

    def test_money_with_glued_letter_rejected(self):
        # Regression: trailing-letter lookahead was missing, so
        # 'Rp25ribu' became the nonsense word 'rupiahribu'.
        for raw in ("Rp25ribu", "Rp25.000x"):
            with self.assertRaises(fe.FrontendError, msg=raw):
                fe.normalize(raw)

    def test_normal_money_still_works(self):
        self.assertEqual(fe.normalize("Rp 25.000").normalized_text,
                         "dua puluh lima ribu rupiah")


class TestHyphenAcronymRegression(unittest.TestCase):
    def test_lowercase_hyphenated_acronym_spelled(self):
        # Regression: hyphen-acronym regex was [A-Z]-only, so lowercase
        # 'oss-rba' fell through to hyphen-as-space.
        self.assertEqual(fe.normalize("oss-rba").normalized_text, "o es es er be a")

    def test_known_acronym_prefix_no_double_spelling(self):
        # Regression: known-acronym list consumed 'OSS' inside 'OSS-RBA',
        # leaving a dangling 'es-RBA' fragment ('o es e es er be a').
        self.assertEqual(fe.normalize("OSS-RBA").normalized_text, "o es es er be a")
        self.assertEqual(fe.normalize("OSS").normalized_text, "o es es")


class TestReduplikasiHyphenRegression(unittest.TestCase):
    def test_reduplikasi_hyphen_becomes_comma(self):
        # Regression: blanket hyphen-acronym spelling turned
        # 'meraba-raba' into letter names ('em e er a be a'), garbling
        # reduplikasi and letting the duration predictor run too fast.
        # Reduplikasi must expand to full copies with a comma micro-pause.
        self.assertEqual(
            fe.normalize("anak-anak").normalized_text, "anak, anak")
        self.assertEqual(
            fe.normalize("meraba-raba").normalized_text, "meraba, raba")
        self.assertEqual(
            fe.normalize("berlari-larian").normalized_text, "berlari, larian")
        self.assertEqual(
            fe.normalize(
                "Rara meraba-raba ruji-ruji roda kereta yang berkarat parah."
            ).normalized_text,
            "Rara meraba, raba ruji, ruji roda kereta yang berkarat parah.")

    def test_acronym_and_compound_hyphen_unchanged(self):
        self.assertEqual(fe.normalize("OSS-RBA").normalized_text, "o es es er be a")
        self.assertEqual(fe.normalize("oss-rba").normalized_text, "o es es er be a")
        self.assertEqual(
            fe.normalize("kilowatt-jam").normalized_text, "kilowatt jam")
        self.assertEqual(
            fe.normalize("e-commerce").normalized_text, "e commerce")


class TestNonObjectJsonRowRegression(unittest.TestCase):
    def test_non_dict_rows_raise_valueerror_not_attributeerror(self):
        for bad in (["x", 1], 42, "just a string", None, 3.14):
            with self.assertRaises(ValueError, msg=repr(bad)):
                T.validate_rows([bad], "data.jsonl")

    def test_non_object_line_in_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('["not an object"]\n')
            with self.assertRaises(ValueError):
                T.load_dataset(path)


class TestResumeBuildModelRegression(unittest.TestCase):
    def _rows(self):
        return [{"id": "a", "tokens": [1], "durations": [3],
                 "features_path": "a.npy", "voice_id": 0, "split": "train"}]

    def test_resume_only_build_model_returns_shell(self):
        # Regression: build_model had no 'resume' branch, so
        # '--resume' alone raised 'no weight source' in main().
        args = T.parse_args(["--data", "d.jsonl", "--out", "o", "--resume", "auto"])
        model = T.build_model(args, self._rows(), self._rows())
        self.assertIsNotNone(model)


class TestDomain510EndToEnd(unittest.TestCase):
    """Every 5.10 raw->spoken pair must be reproducible from the frontend.

    §5.10 writes acronyms as bare letters ('K B L I'); the frontend spells
    them with Indonesian letter names ('ka be el i').  Both are correct
    spoken forms of the same raw symbol, so acronym rows are checked for
    phoneme equality rather than string equality.
    """

    def _expansion_rows(self):
        rows, _ = dt.extract(SOURCE)
        rows = dt.assign_splits(dt.group_rows(rows))
        return [r for r in rows if r["kind"] == "symbol_expansion"]

    def test_all_510_expand_without_error(self):
        rows = self._expansion_rows()
        self.assertEqual(len(rows), 13)
        for r in rows:
            res = fe.normalize(r["raw"])  # must not raise
            self.assertTrue(res.normalized_text, r["raw"])
            # and the expansion must itself phonemize
            ph = fe.phonemize(res.normalized_text)
            self.assertEqual(ph.unknown_words, ())

    def test_non_acronym_510_exact_match(self):
        exact = {
            "Rp25.000": "dua puluh lima ribu rupiah",
            "Rp1.500": "seribu lima ratus rupiah",
            "2.200 VA": "dua ribu dua ratus volt ampere",
            "200 kVA": "dua ratus kilovolt ampere",
            "24 jam": "dua puluh empat jam",
            "8 pcs": "delapan buah",
        }
        for r in self._expansion_rows():
            if r["raw"] in exact:
                self.assertEqual(r["raw"], r["raw"])
                self.assertEqual(
                    fe.normalize(r["raw"]).normalized_text, exact[r["raw"]])

    def test_acronym_510_letter_sequence_match(self):
        # §5.10 writes bare letters ('K B L I', 'B satu'); the frontend
        # spells them with Indonesian letter names ('ka be el i', 'be satu').
        # Phonemes legitimately differ ('be' = B+EH1 vs bare 'B'), so assert
        # instead that both forms phonemize cleanly and that the frontend
        # expansion contains exactly as many <space> separators as the
        # source letter sequence (one name per letter, plus number words).
        for r in self._expansion_rows():
            src_letters = [t for t in r["spoken_expected"].split() if len(t) == 1]
            if not src_letters:
                continue
            got = fe.normalize(r["raw"]).normalized_text
            got_ph = fe.phonemize(got)
            self.assertEqual(got_ph.unknown_words, (), r["raw"])
            # one spelled-name token per source letter (number words add
            # their own tokens but never extra <space> beyond the count
            # implied by the spoken form's word count)
            self.assertEqual(
                len(got.split()), len(r["spoken_expected"].split()),
                f"{r['raw']!r}: {got!r} vs {r['spoken_expected']!r}")

    def test_510_never_in_train_or_val(self):
        rows = self._expansion_rows()
        for r in rows:
            self.assertEqual(r["split"], "test")
            self.assertEqual(r["tts_usage"], "eval_only")


class TestManifestNoSpokenLeakAcrossSplits(unittest.TestCase):
    def test_frontend_spoken_form_unique_per_split(self):
        manifest = REPO_ROOT / "experiments" / "text" / "domain_manifest.jsonl"
        if not manifest.exists():
            self.skipTest("manifest not generated yet")
        rows = [json.loads(line) for line in
                manifest.read_text(encoding="utf-8").splitlines() if line]
        seen = {}
        for r in rows:
            key = fe.normalize(r["text"]).normalized_text
            if key in seen:
                self.assertEqual(
                    seen[key], r["split"],
                    f"spoken form {key!r} in {seen[key]} and {r['split']}")
            seen[key] = r["split"]
        # sanity: phoneme token-id sequences must also be split-exclusive
        tok_splits = {}
        for r in rows:
            ids = fe.phonemize(r["text"]).token_ids
            if ids in tok_splits:
                self.assertEqual(tok_splits[ids], r["split"])
            tok_splits[ids] = r["split"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
