"""Unit tests for src_id/frontend.py (Indonesian normalizer / G2P for PiVo).

Run from a trusted working directory (not inside the downloaded repos):

    python -I -m unittest discover -s tests -v

Stdlib only.  The import below inserts the src_id directory explicitly so
the tests do not depend on the current working directory layout.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src_id"))

import frontend as fe  # noqa: E402


class TestInventory(unittest.TestCase):
    def test_inventory_matches_pivo_count(self):
        # Verified against PiVo-1.4M/onnx/metadata.json on 2026-11-08:
        # 10 special + 24 consonants + 45 stressed vowels = 79 symbols
        # (PiVo reserves VOCAB_SIZE=128 embedding rows regardless).
        self.assertEqual(len(fe.SYMBOLS), 79)
        self.assertEqual(fe.VOCAB_SIZE, 128)

    def test_inventory_contains_expected_symbols(self):
        for sym in ("<bos>", "<eos>", "<space>", "NG", "SH", "CH", "JH",
                    "AA1", "AH0", "EH1", "UW2", ".", ",", "?"):
            self.assertIn(sym, fe.SYMBOLS)
        # No Indonesian-only symbols invented.
        for sym in ("NY", "KH", "SCHWA"):
            self.assertNotIn(sym, fe.SYMBOLS)

    def test_symbol_ids_roundtrip(self):
        for sym, idx in fe.SYMBOL_TO_ID.items():
            self.assertEqual(fe.SYMBOLS[idx], sym)


class TestCardinalOrdinal(unittest.TestCase):
    def test_cardinal_small(self):
        self.assertEqual(fe.cardinal(0), "nol")
        self.assertEqual(fe.cardinal(1), "satu")
        self.assertEqual(fe.cardinal(11), "sebelas")
        self.assertEqual(fe.cardinal(19), "sembilan belas")

    def test_cardinal_tens_and_hundreds(self):
        self.assertEqual(fe.cardinal(25), "dua puluh lima")
        self.assertEqual(fe.cardinal(100), "seratus")
        self.assertEqual(fe.cardinal(200), "dua ratus")
        self.assertEqual(fe.cardinal(250), "dua ratus lima puluh")

    def test_cardinal_scales(self):
        self.assertEqual(fe.cardinal(1000), "seribu")
        self.assertEqual(fe.cardinal(25000), "dua puluh lima ribu")
        self.assertEqual(fe.cardinal(1000000), "satu juta")

    def test_cardinal_negative(self):
        self.assertEqual(fe.cardinal(-3), "minus tiga")

    def test_ordinal(self):
        self.assertEqual(fe.ordinal(1), "pertama")
        self.assertEqual(fe.ordinal(2), "kedua")
        self.assertEqual(fe.ordinal(3), "ketiga")
        self.assertEqual(fe.ordinal(4), "keempat")
        self.assertEqual(fe.ordinal(21), "kedua puluh satu")


class TestNormalize(unittest.TestCase):
    def test_currency_rupiah_thousands(self):
        res = fe.normalize("Harga Rp 25.000, terima kasih.")
        self.assertIn("dua puluh lima ribu rupiah", res.normalized_text)
        self.assertTrue(any("currency" in e for e in res.mapping_log))

    def test_currency_dollar_decimal(self):
        res = fe.normalize("Harganya $3.50.")
        self.assertIn("tiga dolar lima puluh sen", res.normalized_text)

    def test_decimal_comma_and_dot(self):
        self.assertIn("koma", fe.normalize("Nilainya 3,14.").normalized_text)
        self.assertIn("koma", fe.normalize("Nilainya 3.14.").normalized_text)

    def test_percent(self):
        res = fe.normalize("Diskon 50% hari ini.")
        self.assertIn("persen", res.normalized_text)

    def test_iso_date(self):
        res = fe.normalize("Tanggal 2026-11-08.")
        self.assertIn("November", res.normalized_text)
        self.assertTrue(any("date" in e for e in res.mapping_log))

    def test_invalid_iso_date_rejected(self):
        with self.assertRaises(fe.FrontendError):
            fe.normalize("Tanggal 2026-13-40.")

    def test_slash_date_rejected(self):
        with self.assertRaises(fe.FrontendError):
            fe.normalize("Tanggal 08/11/2026.")

    def test_slash_fraction_rejected(self):
        with self.assertRaises(fe.FrontendError):
            fe.normalize("Setengahnya 1/2.")

    def test_unsupported_character_rejected_not_dropped(self):
        with self.assertRaises(fe.FrontendError):
            fe.normalize("Harga Rp 25.000 # promo.")

    def test_unknown_acronym_spelled_letter_by_letter(self):
        # Unknown all-caps acronyms are spelled with Indonesian letter
        # names (auditable in mapping_log), never guessed as a word.
        res = fe.normalize("SKU 123 cepat.")
        self.assertIn("es ka u", res.normalized_text)
        self.assertTrue(any("acronym" in e for e in res.mapping_log))

    def test_mixed_alphanumeric_code_rejected(self):
        with self.assertRaises(fe.FrontendError):
            fe.normalize("Kode X9Z cepat.")

    def test_ordinal_suffix(self):
        res = fe.normalize("Urutan ke-3 adalah 3rd.")
        self.assertIn("ketiga", res.normalized_text)

    def test_domain_510_examples(self):
        # docs/superpowers/plans/requestkalimat.md §5.10 -- priority cases.
        self.assertIn("dua puluh lima ribu rupiah",
                      fe.normalize("Rp25.000").normalized_text)
        self.assertIn("be satu", fe.normalize("B-1").normalized_text)
        self.assertIn("be dua", fe.normalize("B-2").normalized_text)
        self.assertIn("er satu", fe.normalize("R1").normalized_text)
        self.assertEqual(fe.normalize("KBLI").normalized_text, "ka be el i")
        self.assertEqual(fe.normalize("NIB").normalized_text, "en i be")
        self.assertIn("delapan buah", fe.normalize("8 pcs").normalized_text)
        self.assertIn("dua ratus kilovolt ampere",
                      fe.normalize("200 kVA").normalized_text)
        self.assertIn("dua ribu dua ratus volt ampere",
                      fe.normalize("2.200 VA").normalized_text)
        self.assertIn("dua puluh empat jam",
                      fe.normalize("24 jam").normalized_text)

    def test_lowercase_acronym_spelled(self):
        # User text in §2b.7 is lowercase: nib must not become a word.
        self.assertEqual(fe.normalize("cara urus nib gimana")
                         .normalized_text, "cara urus en i be gimana")

    def test_no_silent_deletion(self):
        # Every input word must survive normalization or raise.
        res = fe.normalize("surat dikirim hari ini")
        self.assertEqual(res.normalized_text, "surat dikirim hari ini")

    def test_empty_input(self):
        self.assertEqual(fe.normalize("").normalized_text, "")

    def test_non_string_rejected(self):
        with self.assertRaises(fe.FrontendError):
            fe.normalize(123)  # type: ignore[arg-type]


class TestG2P(unittest.TestCase):
    def test_digraph_ng_is_single_phoneme(self):
        res = fe.phonemize("tangan")
        self.assertIn("NG", res.symbols)
        # 'ng' must not be split into N + G.
        self.assertNotIn("NG", [])
        self.assertEqual(res.symbols.count("NG"), 1)
        self.assertTrue(any("digraph 'ng'" in e for e in res.mapping_log))

    def test_digraph_sy(self):
        res = fe.phonemize("syarat")
        self.assertIn("SH", res.symbols)

    def test_digraph_kh_logged_approximation(self):
        res = fe.phonemize("khabar")
        self.assertTrue(any("kh" in e for e in res.mapping_log))
        for sym in res.symbols:
            self.assertIn(sym, fe.SYMBOLS)

    def test_digraph_ny_logged(self):
        res = fe.phonemize("banyak")
        self.assertTrue(any("ny" in e for e in res.mapping_log))

    def test_all_symbols_in_inventory(self):
        res = fe.phonemize("surat dikirim tanggal 2026-11-08")
        for sym in res.symbols:
            self.assertIn(sym, fe.SYMBOLS)
        self.assertEqual(res.token_ids[0], fe.SYMBOL_TO_ID["<bos>"])
        self.assertEqual(res.token_ids[-1], fe.SYMBOL_TO_ID["<eos>"])

    def test_e_ambiguity_logged(self):
        res = fe.phonemize("enak")
        self.assertTrue(any("ambiguous schwa" in e for e in res.mapping_log))

    def test_schwa_override_with_ah(self):
        res = fe.phonemize("empat",
                           overrides={"empat": ["AH0", "M", "P", "AH0", "T"]})
        self.assertIn("AH0", res.symbols)

    def test_bad_override_rejected(self):
        with self.assertRaises(fe.FrontendError):
            fe.phonemize("empat", overrides={"empat": ["SCHWA", "M"]})

    def test_override_unknown_phone(self):
        with self.assertRaises(fe.FrontendError):
            fe.phonemize("halo", overrides={"halo": ["XX"]})

    def test_unknown_word_strict_raises(self):
        # Normalization rejects unsupported characters before G2P, so
        # the word never reaches the (silent-drop-free) G2P stage.
        with self.assertRaises(fe.FrontendError):
            fe.phonemize("harga Rp 25.000 #")

    def test_unknown_word_nonstrict_reported(self):
        # Regression guard: non-strict mode with a normal sentence
        # yields no unknown entries and no dropped words.
        res = fe.phonemize("terima kasih", strict=False)
        self.assertEqual(res.unknown_words, ())
        self.assertGreater(len(res.symbols), 2)

    def test_unknown_word_error_hierarchy(self):
        self.assertTrue(issubclass(fe.UnknownWordError, fe.FrontendError))
        err = fe.UnknownWordError(["zzvo"])
        self.assertIn("zzvo", str(err))
        self.assertEqual(err.unknown_words, ("zzvo",))

    def test_g2p_fallback_is_the_rule_path(self):
        # Words outside the lexicon are handled by G2P rules, not
        # rejected: 'tangan' is not in LEKSIKON but must phonemize.
        self.assertNotIn("tangan", fe.LEKSIKON)
        res = fe.phonemize("tangan")
        self.assertEqual(res.unknown_words, ())
        self.assertIn("NG", res.symbols)

    def test_punctuation_preserved(self):
        res = fe.phonemize("Halo, apa kabar?")
        self.assertIn(",", res.symbols)
        self.assertIn("?", res.symbols)

    def test_space_between_words(self):
        res = fe.phonemize("terima kasih")
        self.assertIn("<space>", res.symbols)

    def test_c_is_tʃ_regardless_of_following_vowel(self):
        # Indonesian 'c' is /tS/ in cara, cuma, coba, cari -- not
        # conditioned on the following vowel.
        for word in ("cara", "cuma", "coba", "cari", "cincin"):
            res = fe.phonemize(word)
            self.assertIn("CH", res.symbols, word)

    def test_c_k_loanword_needs_override(self):
        res = fe.phonemize("cucu",
                           overrides={"cucu": ["K", "AH1", "K", "AH0"]})
        self.assertIn("K", res.symbols)

    def test_mapping_log_is_tuple_and_nonempty(self):
        res = fe.phonemize("Harga Rp 25.000, terima kasih.")
        self.assertIsInstance(res.mapping_log, tuple)
        self.assertGreater(len(res.mapping_log), 0)

    def test_full_sentence_from_readme_style(self):
        res = fe.phonemize("Surat dikirim tanggal 2026-11-08. Terima kasih.")
        self.assertEqual(res.symbols[0], "<bos>")
        self.assertEqual(res.symbols[-1], "<eos>")
        self.assertIn("November", res.normalized_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
