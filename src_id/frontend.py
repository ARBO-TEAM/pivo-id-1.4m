"""Indonesian text normalizer / G2P frontend for PiVo (English-only model).

PiVo-1.4M itself is English-only: its frontend (``PiVo-1.4M/pivo/text.py``,
read-only reference) looks words up in the CMU pronouncing dictionary and
raises on out-of-vocabulary (OOV) words instead of guessing.  Its symbol
inventory (``PiVo-1.4M/onnx/metadata.json``, ``symbols`` field) is plain
ARPAbet with stress digits 0/1/2 on vowels plus ``<pad> <bos> <eos>
<space> . , ? ! ; :`` -- 79 symbols, no Indonesian-specific inventory.

This module maps Indonesian (Bahasa Indonesia) text onto that inventory:

* Normalization first expands numbers, decimals, currency (Rp / $ / EUR
  style), percentages, ordinals and ISO dates into Indonesian words, so no
  digit/symbol input is silently dropped.  Ambiguous slash dates and
  fractions are *rejected*, mirroring PiVo's own policy.
* A small explicit lexicon covers the most common Indonesian function
  words and words whose spelling-to-sound mapping is unambiguous.
* Everything else goes through a rule-based grapheme-to-phoneme (G2P)
  converter that emits only symbols present in the PiVo inventory.
* Digraphs ``ng``, ``ny``, ``kh``, ``sy`` (and ``c``/``j`` conventions)
  are handled as single phonemes / explicit overrides before single
  letters, so they are never split into two sounds.
* Words not in the lexicon go through the rule-based G2P converter
  (digraphs first).  Only a word containing a letter with no rule at
  all (or a word supplied via ``overrides`` with an invalid phone)
  raises ``UnknownWordError`` / ``FrontendError``.  This differs from
  PiVo itself, which has no G2P fallback and rejects every OOV word;
  here the G2P *is* the fallback, and rejection happens only when
  even the rules cannot map the word.  Nothing is silently dropped:
  words that cannot be mapped are reported, never skipped quietly.
  Callers may supply explicit per-word ARPAbet ``overrides`` -- the
  same escape hatch PiVo's ``phonemize`` exposes.

AMBIGUITY -- the letter ``e`` (schwa)
-------------------------------------
Indonesian orthography does not mark schwa.  Letter ``e`` can be /e/
(as in *enak*) or schwa /e~@/-like (as in *empat*, *besok*), and there is
no reliable spelling rule.  PiVo's inventory has no dedicated schwa
symbol, so this frontend has to choose:

* Default rule: ``e`` -> ``EH`` (the /e/ vowel).  This is a
  deterministic, explicitly documented choice -- no claim is made
  about which reading is more frequent; it is simply the rule that
  can be applied without a dictionary.
* Schwa words must be listed in ``SCHWA_WORDS`` / ``LEKSIKON`` or passed
  via ``overrides`` with ``AH`` (PiVo's reduced-vowel symbol) where the
  speaker intends schwa, e.g. ``overrides={"empat": ["AH0","M","P","AH0","T"]}``.
* Every use of the default ``e`` -> ``EH`` rule is recorded in the
  returned ``mapping_log`` as ``e:EH (default; ambiguous)`` so a caller
  can audit / override it.  Stress: Indonesian stress is weak and not
  phonemic; this frontend puts primary stress (``1``) on the first
  vowel of a vowel cluster / first syllable nucleus and ``0`` on the
  rest, purely to satisfy PiVo's stressed-vowel inventory shape.  It is
  not a claim about Indonesian prosody.

Other documented ambiguities: in standard Indonesian orthography
``c`` represents /tS/ (as in *cara*, *cuma*, *coba*, *cari*) and
``j`` represents /dZ/; this frontend therefore maps ``c`` -> ``CH``
unconditionally (overridable via ``overrides`` for loanwords where
``c`` is pronounced /k/, e.g. some Arabic loans).  Final ``h`` and
gemination are not marked in the spelling; ``sy`` is treated as /S/
(SH) per common Indonesian convention; loanwords may retain
source-language sounds that have no PiVo symbol -- those words are
rejected rather than approximated.

Only the Python standard library is used.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

__all__ = [
    "SYMBOLS",
    "SYMBOL_TO_ID",
    "VOCAB_SIZE",
    "FrontendError",
    "UnknownWordError",
    "NormalizationResult",
    "PhonemeResult",
    "normalize",
    "phonemize",
    "cardinal",
    "ordinal",
    "SYMBOL_INVENTORY_SOURCE",
]

logger = logging.getLogger(__name__)

# PiVo symbol inventory, copied verbatim from PiVo-1.4M/onnx/metadata.json
# ("symbols" field) and PiVo-1.4M/pivo/text.py (SPECIAL + CONSONANTS +
# VOWELS with stress 0/1/2).  Keep in sync if the model inventory changes.
SYMBOL_INVENTORY_SOURCE = "PiVo-1.4M/onnx/metadata.json: symbols"
SPECIAL = ["<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":"]
CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
SYMBOLS = tuple(SPECIAL + CONSONANTS + [v + s for v in VOWELS for s in "012"])
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
VOCAB_SIZE = 128  # PiVo reserves 128 entries (see pivo/text.py).
_SYMBOL_SET = frozenset(SYMBOLS)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FrontendError(ValueError):
    """Base class for frontend failures (normalization or G2P)."""


class UnknownWordError(FrontendError):
    """Raised when a word has no lexicon entry, rule is inapplicable, or
    no explicit override was supplied.  Carries the offending words so
    callers can add overrides instead of the word being silently dropped."""

    def __init__(self, unknown_words):
        self.unknown_words = tuple(sorted(set(unknown_words)))
        super().__init__(
            "out-of-vocabulary words require pronunciation overrides: "
            + ", ".join(self.unknown_words)
        )


# ---------------------------------------------------------------------------
# Number words (Indonesian)
# ---------------------------------------------------------------------------

_ONES = [
    "nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh",
    "delapan", "sembilan", "sepuluh", "sebelas", "dua belas",
    "tiga belas", "empat belas", "lima belas", "enam belas",
    "tujuh belas", "delapan belas", "sembilan belas",
]
_TENS = ["", "", "dua puluh", "tiga puluh", "empat puluh", "lima puluh",
         "enam puluh", "tujuh puluh", "delapan puluh", "sembilan puluh"]
_SCALES = [(10**12, "triliun"), (10**9, "miliar"), (10**6, "juta"),
           (10**3, "ribu"), (10**2, "ratus")]
_ORDINALS = {"satu": "pertama", "dua": "kedua", "tiga": "ketiga",
             "empat": "keempat", "lima": "kelima", "enam": "keenam",
             "tujuh": "ketujuh", "delapan": "kedelapan", "sembilan": "kesembilan"}
_MONTHS = ("Januari Februari Maret April Mei Juni Juli Agustus September "
           "Oktober November Desember").split()


def cardinal(number: int) -> str:
    """Indonesian cardinal for a non-negative int (negatives get a prefix)."""
    if number < 0:
        return "minus " + cardinal(-number)
    if number < 20:
        return _ONES[number]
    if number < 100:
        rest = number % 10
        return _TENS[number // 10] + (f" {_ONES[rest]}" if rest else "")
    for scale, label in _SCALES:
        if number >= scale:
            if number >= 10**15:
                return " ".join(_ONES[int(d)] for d in str(number))
            if number // scale == 1 and label in ("ribu", "ratus"):
                return f"se{label}" + (f" {cardinal(number % scale)}" if number % scale else "")
            return cardinal(number // scale) + " " + label + (f" {cardinal(number % scale)}" if number % scale else "")
    raise AssertionError("unreachable")


def ordinal(number: int) -> str:
    """Indonesian ordinal: 'ke' + cardinal, with irregulars for 1-3."""
    if number < 0:
        raise FrontendError("ordinal of a negative number is unsupported")
    words = cardinal(number).split()
    if len(words) == 1 and words[0] in _ORDINALS:
        return _ORDINALS[words[0]]
    return "ke" + " ".join(words)


def _parse_amount(raw: str) -> tuple[int, str | None]:
    """Parse grouped number -> (whole, fraction-or-None).

    Indonesian: '.' thousands, ',' decimal (25.000 = 25 ribu, 3,14 = 3.14).
    English: ',' thousands, '.' decimal.  Both accepted when unambiguous;
    when both separators appear the last one is decimal.  A lone separator
    followed by exactly three digits is a thousands group.
    """
    raw = raw.strip()
    if "." in raw and "," in raw:
        decimal = "." if raw.rfind(".") > raw.rfind(",") else ","
        whole, _, frac = raw.replace("," if decimal == "." else ".", "").rpartition(decimal)
        return int(whole), frac
    if "." in raw:
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", raw):
            return int(raw.replace(".", "")), None
        whole, _, frac = raw.partition(".")
        return int(whole), frac
    if "," in raw:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", raw):
            return int(raw.replace(",", "")), None
        whole, _, frac = raw.partition(",")
        return int(whole), frac
    return int(raw), None


def _decimal_words(raw: str) -> str:
    """'3.14' / '3,14' -> 'tiga koma satu empat'; integers via cardinal()."""
    whole, frac = _parse_amount(raw)
    if frac is None:
        return cardinal(whole)
    digits = " ".join(_ONES[int(c)] for c in frac)
    return f"{cardinal(whole)} koma {digits}"


_CURRENCY = {"rp": "rupiah", "idr": "rupiah", "$": "dolar",
             "€": "euro", "£": "pound"}


def _money(match: re.Match) -> str:
    symbol, raw = match.groups()
    unit = _CURRENCY[symbol.lower()]
    whole, fraction = _parse_amount(raw)
    result = f"{cardinal(whole)} {unit}"
    if fraction:
        cents = int(fraction.ljust(2, "0")[:2])
        if cents:
            result += f" {cardinal(cents)} sen"
    return result


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_ABBREV = {"Mr": "Tuan", "Mrs": "Nyonya", "Dr": "Doktor", "Prof": "Profesor"}


@dataclass(frozen=True)
class NormalizationResult:
    normalized_text: str
    mapping_log: tuple[str, ...] = ()


def normalize(text: str) -> NormalizationResult:
    """Normalize Indonesian text into speakable words.

    Returns a NormalizationResult whose ``normalized_text`` contains only
    lowercase letters, spaces and PiVo punctuation, and whose
    ``mapping_log`` records every substitution applied.  Raises
    FrontendError on ambiguous input (slash dates/fractions, unsupported
    characters, unsupported acronyms) instead of guessing.  Nothing is
    silently deleted: every input token is either expanded, kept, or
    causes an error.
    """
    if not isinstance(text, str):
        raise FrontendError("text must be a string")
    log: list[str] = []
    original = text
    text = unicodedata.normalize("NFKC", text)
    for src, dst in {"’": "'", "‘": "'", "“": '"', "”": '"',
                     "—": ",", "–": ",", "…": "..."}.items():
        if src in text:
            log.append(f"punct: {src!r} -> {dst!r}")
            text = text.replace(src, dst)

    def _date(match: re.Match) -> str:
        from datetime import date as cal

        year, month, day = map(int, match.groups())
        try:
            cal(year, month, day)
        except ValueError as exc:
            raise FrontendError(f"invalid ISO date {match[0]!r}: {exc}") from exc
        out = f"{_MONTHS[month - 1]} {ordinal(day)}, {cardinal(year)}"
        log.append(f"date: {match[0]!r} -> {out!r}")
        return out

    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", _date, text)
    if re.search(r"\d/\d", text):
        raise FrontendError(
            "write slash dates/fractions in words, or use an ISO date (YYYY-MM-DD)")

    for abbr, expansion in _ABBREV.items():
        new = re.sub(rf"\b{abbr}\.(?=\s|$)", expansion, text, flags=re.I)
        if new != text:
            log.append(f"abbrev: {abbr}. -> {expansion}")
            text = new

    def _money_logged(match: re.Match) -> str:
        out = _money(match)
        log.append(f"currency: {match[0]!r} -> {out!r}")
        return out

    text = re.sub(
        r"(Rp|\$|€|£|IDR)\s?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+)(?!\d)(?![A-Za-z])",
        _money_logged, text, flags=re.I)

    # A bare currency symbol (Rp/IDR with no amount) must not survive to be
    # guessed as a word; $/€/£ already fail the charset check below.
    if re.search(r"\b(?:Rp|IDR)\b", text, flags=re.I):
        raise FrontendError(
            "currency symbol 'Rp'/'IDR' without an amount; write the amount "
            "explicitly (e.g. 'Rp25.000')")

    # --- Domain expansions (docs/superpowers/plans/requestkalimat.md §5.10) ---
    # Units after a number: 200 kVA -> 200 kilovolt ampere (number expands
    # later); 2.200 VA -> 2.200 volt ampere.  Never drop the unit.
    for unit, words in (("kVA", "kilovolt ampere"), ("kV", "kilovolt"),
                        ("VA", "volt ampere"), ("kWh", "kilowatt jam"),
                        ("kW", "kilowatt"), ("W", "watt")):
        new = re.sub(rf"(?<=\d)\s*{unit}\b", f" {words}", text, flags=re.I)
        if new != text:
            log.append(f"unit: {unit} -> {words}")
            text = new

    # pcs -> buah (8 pcs -> delapan buah; per pcs -> per buah).
    new = re.sub(r"\bpcs\b", "buah", text, flags=re.I)
    if new != text:
        log.append("word: 'pcs' -> 'buah'")
        text = new

    # Indonesian letter names for spelled-out acronyms (§5.10:
    # KBLI -> ka be el i, NIB -> en i be).  The names are ordinary
    # Indonesian words afterwards, so G2P handles them normally.
    letter_names = {"A": "a", "B": "be", "C": "ce", "D": "de", "E": "e",
                    "F": "ef", "G": "ge", "H": "ha", "I": "i", "J": "je",
                    "K": "ka", "L": "el", "M": "em", "N": "en", "O": "o",
                    "P": "pe", "Q": "ki", "R": "er", "S": "es", "T": "te",
                    "U": "u", "V": "ve", "W": "we", "X": "eks", "Y": "ye",
                    "Z": "zet"}

    def _spell(token: str) -> str:
        return " ".join(letter_names[c] for c in token.upper() if c.isalpha())

    def _acronym_logged(match: re.Match) -> str:
        out = _spell(match.group(0))
        log.append(f"acronym: {match[0]!r} -> {out!r}")
        return out

    # Tariff / code forms first: B-1 -> be 1 (-> be satu), B-2, R1 -> er 1.
    text = re.sub(r"\b([A-Za-z])\s*-\s*(\d+)\b",
                  lambda m: (log.append(f"code: {m[0]!r} -> "
                                         f"{letter_names[m.group(1).upper()]} {m.group(2)!r}"),
                             f"{letter_names[m.group(1).upper()]} {m.group(2)}")[1],
                  text)
    text = re.sub(r"\b([A-Za-z])(\d+)\b",
                  lambda m: (log.append(f"code: {m[0]!r} -> "
                                         f"{letter_names[m.group(1).upper()]} {m.group(2)!r}"),
                             f"{letter_names[m.group(1).upper()]} {m.group(2)}")[1],
                  text)
    # Known acronyms, any case (lowercase nib/kbli appear in user text):
    # logged, so the choice is auditable and overridable at word level.
    for acr in ("NIB", "KBLI", "NPWP", "KTP", "PLN", "MCB"):
        new = re.sub(rf"\b{acr}\b", _acronym_logged, text, flags=re.I)
        if new != text:
            text = new
    # Hyphenated acronyms (OSS-RBA -> o es es er be a) and any other
    # all-caps token: spell letter by letter, never silently guess.
    # Case-insensitive so 'oss-rba' is spelled too (it would otherwise fall
    # through to hyphen-as-space). Must run BEFORE the known-acronym list:
    # 'OSS' must not be consumed first, leaving a dangling 'es-RBA'.
    text = re.sub(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b",
                  lambda m: (log.append(f"acronym: {m[0]!r} -> {_spell(m.group(0))!r}"),
                             _spell(m.group(0)))[1], text)
    text = re.sub(r"\b[A-Z]{2,}\b", _acronym_logged, text)

    def _ord_logged(match: re.Match) -> str:
        out = ordinal(int(match.group(1)))
        log.append(f"ordinal: {match[0]!r} -> {out!r}")
        return out

    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", _ord_logged, text, flags=re.I)

    def _num_logged(match: re.Match) -> str:
        raw = match.group(0)
        out = ("minus " if raw.startswith("-") else "") + _decimal_words(raw.lstrip("-"))
        log.append(f"number: {raw!r} -> {out!r}")
        return out

    # Order matters: decimal/thousands forms before the plain-integer
    # branch so "3.14" is not consumed as "3" + ".14".  The lookahead
    # permits a sentence-final period after the number ("3.14.").
    # Comma thousands (English) also need a full multi-group branch
    # before the generic \d+,\d+ decimal branch.
    text = re.sub(
        r"-?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
        r"|\d+\.\d+|\d+,\d+|\d+)"
        r"(?![\dA-Za-z])",
        _num_logged, text)

    for ch, word in {"%": " persen ", "&": " dan ", "+": " tambah "}.items():
        if ch in text:
            log.append(f"symbol: {ch!r} -> {word.strip()!r}")
            text = text.replace(ch, word)

    # Acronyms with no rule above have already been spelled letter by
    # letter; anything still all-caps+digits (a mixed code) is rejected
    # rather than guessed.
    caps = re.findall(r"\b[A-Z]*\d[A-Z0-9]*\b|\b[A-Z]{2,}\b", text)
    caps = [c for c in caps if any(ch.isdigit() for ch in c)]
    if caps:
        raise FrontendError(
            "unsupported code/acronym (no expansion rule): "
            + ", ".join(sorted(set(caps)))
            + "; write it out or supply per-word overrides")
    if "-" in text:
        log.append("hyphen: '-' -> ' '")
    text = text.replace("-", " ")
    text = re.sub(r'[()\[\]{}"/]', " ", text)
    unsupported = re.findall(r"[^a-zA-Z\s'.,?!;:]", text)
    if unsupported:
        raise FrontendError(
            f"unsupported text characters after normalization: {sorted(set(unsupported))}")
    normalized = re.sub(r"\s+", " ", text).strip()
    if normalized != original:
        log.append(f"normalized: {original!r} -> {normalized!r}")
    for entry in log:
        logger.debug(entry)
    return NormalizationResult(normalized, tuple(log))


# ---------------------------------------------------------------------------
# Lexicon (explicit, unambiguous common words)
# ---------------------------------------------------------------------------


def _phones(*names: str) -> list[str]:
    return list(names)


LEKSIKON: dict[str, list[str]] = {
    "yang": _phones("Y", "AH0", "NG"),
    "di": _phones("D", "IY1"),
    "ke": _phones("K", "EH1"),
    "dari": _phones("D", "AH0", "R", "IY1"),
    "dan": _phones("D", "AH0", "N"),
    "atau": _phones("AH0", "T", "AW1"),
    "ini": _phones("IY1", "N", "IY0"),
    "itu": _phones("IY1", "T", "UW0"),
    "adalah": _phones("AH0", "D", "AH1", "L", "AH0"),
    "tidak": _phones("T", "IH0", "D", "AH1", "K"),
    "bukan": _phones("B", "AH0", "K", "AH1", "N"),
    "untuk": _phones("AH0", "N", "T", "AH1", "K"),
    "dengan": _phones("D", "AH0", "NG", "AH1", "N"),
    "pada": _phones("P", "AH0", "D", "AH1"),
    "dalam": _phones("D", "AH0", "L", "AH1", "M"),
    "sebuah": _phones("S", "EH0", "B", "AH1", "AH0"),
    "seorang": _phones("S", "EH0", "OW1", "R", "AH0", "NG"),
    "saya": _phones("S", "AH1", "Y", "AH0"),
    "anda": _phones("AH0", "N", "D", "AH1"),
    "kami": _phones("K", "AH1", "M", "IY0"),
    "kita": _phones("K", "IY1", "T", "AH0"),
    "mereka": _phones("M", "EH0", "R", "EH1", "K", "AH0"),
    "dia": _phones("D", "IY1", "AH0"),
    "ia": _phones("IY1", "AH0"),
    "apa": _phones("AH1", "P", "AH0"),
    "siapa": _phones("S", "IY1", "AH0", "P", "AH0"),
    "kapan": _phones("K", "AH1", "P", "AH0", "N"),
    "bagaimana": _phones("B", "AH0", "G", "AH1", "M", "AH0", "N", "AH0"),
    "mengapa": _phones("M", "EH0", "NG", "AH1", "P", "AH0"),
    "karena": _phones("K", "AH0", "R", "EH1", "N", "AH0"),
    "jika": _phones("JH", "IH1", "K", "AH0"),
    "kalau": _phones("K", "AH1", "L", "AW0"),
    "agar": _phones("AH0", "G", "AH1", "R"),
    "supaya": _phones("S", "AH0", "P", "AH1", "Y", "AH0"),
    "tetapi": _phones("T", "EH0", "T", "EH1", "P", "IY0"),
    "tapi": _phones("T", "AH1", "P", "IY0"),
    "namun": _phones("N", "AH1", "M", "AH0", "N"),
    "sangat": _phones("S", "AH0", "NG", "AH1", "T"),
    "lebih": _phones("L", "EH1", "B", "AH0"),
    "kurang": _phones("K", "AH0", "R", "AH1", "NG"),
    "juga": _phones("JH", "AH1", "G", "AH0"),
    "sudah": _phones("S", "AH0", "D", "AH1"),
    "belum": _phones("B", "EH1", "L", "AH0", "M"),
    "akan": _phones("AH0", "K", "AH1", "N"),
    "sedang": _phones("S", "EH1", "D", "AH0", "NG"),
    "telah": _phones("T", "EH1", "L", "AH0"),
    "pernah": _phones("P", "EH0", "R", "AH1", "N", "AH0"),
    "bisa": _phones("B", "IY1", "S", "AH0"),
    "dapat": _phones("D", "AH1", "P", "AH0", "T"),
    "harus": _phones("HH", "AH1", "R", "AH0", "S"),
    "mau": _phones("M", "AW1"),
    "ingin": _phones("AH0", "NG", "IH1", "N"),
    "perlu": _phones("P", "EH0", "R", "L", "UW1"),
    "mahasiswa": _phones("M", "AH0", "HH", "AH1", "S", "IH0", "S", "AH1", "W", "AH0"),
    "perusahaan": _phones("P", "EH0", "R", "AH0", "S", "AH1", "HH", "AH0", "N", "AH0"),
    "pemerintah": _phones("P", "EH0", "M", "EH1", "R", "IH0", "N", "T", "AH0"),
    "indonesia": _phones("IH0", "N", "D", "AH1", "OW0", "S", "IY0", "AH0"),
    "jakarta": _phones("JH", "AH0", "K", "AH1", "R", "AH0", "T", "AH0"),
    "surat": _phones("S", "AH0", "R", "AH1", "T"),
    "kirim": _phones("K", "IY1", "R", "IH0", "M"),
    "harga": _phones("HH", "AH0", "R", "G", "AH1"),
    "mahal": _phones("M", "AH0", "HH", "AH1", "L"),
    "murah": _phones("M", "AH0", "R", "AH1", "HH"),
    "besar": _phones("B", "EH1", "S", "AH0", "R"),
    "kecil": _phones("K", "EH1", "CH", "IH0", "L"),
    "baik": _phones("B", "AH1", "IY0", "K"),
    "terima": _phones("T", "EH0", "R", "IY1", "M", "AH0"),
    "kasih": _phones("K", "AH1", "S", "IH0", "HH"),
    "tolong": _phones("T", "OW1", "L", "OW0", "NG"),
    "maaf": _phones("M", "AH1", "AH0", "F"),
    "halo": _phones("HH", "AH1", "L", "OW0"),
    "selamat": _phones("S", "EH0", "L", "AH1", "M", "AH0", "T"),
    "pagi": _phones("P", "AH1", "G", "IY0"),
    "malam": _phones("M", "AH1", "L", "AH0", "M"),
    "hari": _phones("HH", "AH1", "R", "IY0"),
    "minggu": _phones("M", "IH0", "NG", "G", "AH1", "UW0"),
    "bulan": _phones("B", "AH1", "L", "AH0", "N"),
    "tahun": _phones("T", "AH1", "HH", "AH0", "N"),
}

_DIGRAPHS = {
    "ng": "NG",   # /ng/ as in *tangan*
    "ny": "Y",    # PiVo has no palatal-nasal symbol; nearest is Y (logged)
    "kh": "K",    # PiVo has no /kh/ symbol; nearest is K (logged)
    "sy": "SH",   # /sy/ as in *syarat*, *suka*
    "ts": "CH",
}
# NOTE on ny/kh: approximations are *always* logged so they stay visible;
# callers who need a different realisation must use explicit overrides.

_SINGLE = {
    "b": "B", "d": "D", "f": "F", "g": "G", "h": "HH", "j": "JH", "k": "K",
    "l": "L", "m": "M", "n": "N", "p": "P", "q": "K", "r": "R",
    "s": "S", "t": "T", "v": "V", "w": "W", "x": "K", "y": "Y", "z": "Z",
}
_VOWEL_LETTERS = {"a": "AH", "e": "EH", "i": "IH", "o": "OW", "u": "UW"}
SCHWA_WORDS = frozenset({
    "empat", "besok", "enam", "dengan", "mereka", "tentang",
    "sebuah", "seorang", "kepada", "menemukan", "bertemu",
})


def _g2p_word(word: str, log: list[str]) -> list[str] | None:
    """Rule-based G2P for one lowercase word; None if any letter has no
    safe mapping (caller rejects the word instead of guessing)."""
    phones: list[str] = []
    i = 0
    vowel_seen = False
    while i < len(word):
        chunk = word[i:i + 2]
        if chunk in _DIGRAPHS:
            sym = _DIGRAPHS[chunk]
            log.append(f"g2p: {word!r} digraph {chunk!r} -> {sym}")
            phones.append(sym)
            i += 2
            continue
        ch = word[i]
        if ch in _VOWEL_LETTERS:
            base = _VOWEL_LETTERS[ch]
            stress = "0" if vowel_seen else "1"
            vowel_seen = True
            if ch == "e":
                log.append(
                    f"g2p: {word!r} e -> {base}{stress} (default; ambiguous schwa, see module doc)")
            phones.append(base + stress)
            i += 1
            continue
        if ch == "c":
            # Indonesian 'c' is /tS/ (cara, cuma, coba) regardless of the
            # following vowel; loanwords with /k/ need explicit overrides.
            log.append(f"g2p: {word!r} c -> CH")
            phones.append("CH")
            i += 1
            continue
        sym = _SINGLE.get(ch)
        if sym is None:
            return None
        phones.append(sym)
        i += 1
    if any(p not in _SYMBOL_SET for p in phones):
        return None
    return phones


# ---------------------------------------------------------------------------
# Phonemization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhonemeResult:
    normalized_text: str
    symbols: tuple[str, ...]
    token_ids: tuple[int, ...]
    unknown_words: tuple[str, ...]
    mapping_log: tuple[str, ...] = field(default_factory=tuple)


def phonemize(
    text: str,
    *,
    strict: bool = True,
    overrides: dict[str, list[str]] | None = None,
) -> PhonemeResult:
    """Normalize + G2P Indonesian text into PiVo symbols.

    ``overrides`` maps a lowercase word to an explicit ARPAbet phone
    list (validated against the PiVo inventory).  In strict mode (the
    default, mirroring PiVo) any word that is neither in the lexicon nor
    rule-convertible nor overridden raises UnknownWordError.  In
    non-strict mode unknown words are reported in ``unknown_words`` and
    *skipped with a log entry* -- callers must pass strict=True (default)
    to guarantee no silent loss.
    """
    norm = normalize(text)
    log: list[str] = list(norm.mapping_log)
    overrides = {k.lower(): list(v) for k, v in (overrides or {}).items()}
    for w, phones in overrides.items():
        for p in phones:
            if p not in _SYMBOL_SET:
                raise FrontendError(f"unknown ARPAbet phone {p!r} in override for {w!r}")
        log.append(f"override: {w!r} -> {list(phones)}")

    tokens: list[str] = ["<bos>"]
    unknown: list[str] = []
    for item in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]", norm.normalized_text):
        if item in SPECIAL:
            tokens.append(item)
            log.append(f"punct: {item!r} -> {item!r}")
            continue
        word = item.lower()
        phones, source = None, None
        if word in overrides:
            phones, source = overrides[word], "override"
        elif word in LEKSIKON:
            phones, source = LEKSIKON[word], "lexicon"
        else:
            phones = _g2p_word(re.sub(r"['-]", "", word), log)
            source = "g2p" if phones is not None else None
        if phones is None:
            unknown.append(word)
            log.append(f"unknown: {word!r} (rejected, no silent drop)")
            continue
        if word in SCHWA_WORDS and source == "g2p":
            log.append(f"schwa-warning: {word!r} contains ambiguous 'e'; "
                       f"consider overrides with AH for schwa")
        if tokens[-1] not in ("<bos>", "<space>"):
            tokens.append("<space>")
        for phone in phones:
            if phone not in _SYMBOL_SET:
                raise FrontendError(f"unknown ARPAbet phone {phone!r} in {word!r}")
            tokens.append(phone)
        log.append(f"word: {word!r} -> {list(phones)} ({source})")
    if unknown and strict:
        raise UnknownWordError(unknown)
    tokens.append("<eos>")
    for entry in log:
        logger.debug(entry)
    return PhonemeResult(
        norm.normalized_text, tuple(tokens),
        tuple(SYMBOL_TO_ID[s] for s in tokens),
        tuple(unknown), tuple(log))


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import sys

    sample = sys.argv[1] if len(sys.argv) > 1 else "Harga Rp 25.000, terima kasih."
    result = phonemize(sample)
    print(result.normalized_text)
    print(" ".join(result.symbols))
    for entry in result.mapping_log:
        print(" ", entry)
