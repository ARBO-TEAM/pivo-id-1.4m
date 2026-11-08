"""CTC forced alignment of Indonesian teacher audio to PiVo phone tokens.

Strategy: align the *letter stream* of a pretrained Indonesian character CTC
model against the letter stream of the (normalized) text with ``difflib``,
then derive word spans from the matched letters. Letter-level matching is
robust to CTC word merges/splits and to small pronunciation variants
("digital" heard as "dijital"), which word-level equality wrongly rejects;
coverage is recorded and low coverage still fails.

Why not phone-level CTC: the public Indonesian wav2vec2 models emit characters,
not phones. Treating character spans as phone truth would fabricate timing the
model never produced. This module never silently invents a boundary, and every
interpolated/clamped span is recorded in the result alarms.

Output durations are integer 10 ms frames on the PiVo grid (hop 160 @ 16 kHz),
so ``sum(durations) == n_frames`` for the trimmed clip and the row can be fed to
``src_id/train.py`` once the dataset gates pass.

Usage (CLI, writes one JSON line to stdout):
    python -I src_id/ctc_align.py --wav X.wav --text "..." \
        --model-dir E:/.../models/hf/hub/models--indonesian-nlp--wav2vec2-...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HOP = 160
SR = 16000
MARGIN = 3
ENERGETIC_REL_DB = -35.0
MAX_EXTEND = 30
MAX_TOKEN_FRAMES = 200
MAX_UTTERANCE_FRAMES = 3000
MIN_COVERAGE = 0.6
SILENCE = {"", "sil", "sp", "<eps>", "silence", "|"}

# Relative phone durations used only when a word's phone count differs from its
# letter count (lexicon entries like "yang" -> Y AH0 NG). Deterministic.
_PHONE_WEIGHT = {
    "stop": 1.1, "affricate": 1.6, "fricative": 1.7, "nasal": 1.4,
    "liquid": 1.5, "glide": 1.2, "vowel": 2.4,
}
_STOPS = {"P", "B", "T", "D", "K", "G"}
_AFFRICATES = {"CH", "JH"}
_FRICATIVES = {"F", "V", "S", "Z", "SH", "TH", "DH", "HH"}
_NASALS = {"M", "N", "NG"}
_LIQUIDS = {"L", "R"}
_GLIDES = {"W", "Y"}


class AlignError(RuntimeError):
    """Audio/text/model did not permit a trustworthy alignment."""


def _weight(phone: str) -> float:
    base = re.sub(r"[0-2]$", "", phone)
    if base in _STOPS:
        return _PHONE_WEIGHT["stop"]
    if base in _AFFRICATES:
        return _PHONE_WEIGHT["affricate"]
    if base in _FRICATIVES:
        return _PHONE_WEIGHT["fricative"]
    if base in _NASALS:
        return _PHONE_WEIGHT["nasal"]
    if base in _LIQUIDS:
        return _PHONE_WEIGHT["liquid"]
    if base in _GLIDES:
        return _PHONE_WEIGHT["glide"]
    return _PHONE_WEIGHT["vowel"]


# --------------------------------------------------------------------------- audio
def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: need mono PCM16, got {w.getnchannels()}ch/{w.getsampwidth()*8}bit")
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()
    return pcm, rate


def resample_to_16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    """Deterministic FFT resampling (integer ratio enforced, no external deps)."""
    if rate == SR:
        return pcm
    g = math.gcd(rate, SR)
    up, down = SR // g, rate // g
    n_out = int(round(len(pcm) * up / down))
    spec = np.fft.rfft(pcm.astype(np.float64))
    bins = n_out // 2 + 1
    if bins <= len(spec):
        spec = spec[:bins]
    else:
        spec = np.concatenate([spec, np.zeros(bins - len(spec), dtype=spec.dtype)])
    y = np.fft.irfft(spec, n=n_out) * (n_out / len(pcm))
    return np.clip(y, -32768, 32767).astype(np.int16)


def frame_energy_db(pcm16: np.ndarray) -> np.ndarray:
    """Frame k = samples [160k-80, 160k+80), matching FARGAN feature k."""
    x = np.pad(pcm16.astype(np.float32) / 32768.0, (HOP // 2, 0))
    n = len(pcm16) // HOP
    frames = x[: n * HOP].reshape(n, HOP)
    return 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)


# --------------------------------------------------------------------------- model
def load_ctc(model_dir: Path):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(str(model_dir), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    return processor, model


def ctc_runs(processor, model, pcm16: np.ndarray) -> tuple[list, int]:
    """Collapse CTC output into character runs: [(char, start, end)], n_frames.

    Frame indices are the model's own (~50 Hz for wav2vec2); the caller derives
    seconds-per-frame from ``n_frames`` and the audio length.
    """
    import torch

    audio = pcm16.astype(np.float32) / 32768.0
    inputs = processor(audio, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits[0]
    n_frames = int(logits.shape[0])
    ids = logits.argmax(dim=-1).tolist()
    blank = model.config.pad_token_id
    vocab = processor.tokenizer.convert_ids_to_tokens(list(range(len(processor.tokenizer))))
    runs: list[tuple[str, int, int]] = []
    prev = None
    for k, cid in enumerate(ids):
        if cid == prev:
            if runs:
                c, s, _ = runs[-1]
                runs[-1] = (c, s, k + 1)
            continue
        if cid != blank:
            ch = vocab[cid] if cid < len(vocab) else ""
            if ch and ch not in ("<pad>", "<s>", "</s>", "<unk>"):
                runs.append((ch, k, k + 1))
        prev = cid
    return runs, n_frames


def ctc_grouped(runs, special_tokens: set[str]) -> list[tuple[str, float, float]]:
    """Group character runs into lowercase words."""
    words: list[tuple[str, float, float]] = []
    cur, start, prev_end = "", None, None
    for ch, s, e in runs:
        low = ch.lower()
        if not low:
            continue
        # Word delimiters (space, "|", <pad>-adjacent markers) close the word.
        if low.isspace() or low in special_tokens or low == "|":
            if cur:
                words.append((cur, start, prev_end))
                cur, start = "", None
            continue
        low_clean = "".join(c for c in low if c.isalnum())
        if not low_clean:
            continue
        if not cur:
            start = s
        cur += low_clean
        prev_end = e
    if cur:
        words.append((cur, start, prev_end))
    return words


# --------------------------------------------------------------------------- mapping
def _split_text_words(normalized: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]", normalized)


def word_group_spans(tokens: list[str]) -> list[list[int]]:
    """Token indices of each word, in text order.

    ``<space>`` closes the current word; punctuation contributes no group.
    This mirrors ``_split_text_words`` filtered to alphabetic words.
    """
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, t in enumerate(tokens):
        if t in ("<bos>", "<eos>"):
            continue
        if t == "<space>" or t in {".", ",", "?", "!", ";", ":"}:
            if cur:
                groups.append(cur)
                cur = []
        else:
            cur.append(i)
    if cur:
        groups.append(cur)
    return groups


def _allocate(weights: list[float], total: int) -> list[int]:
    """Largest-remainder integer allocation summing exactly to ``total``."""
    s = sum(weights)
    raw = [w / s * total for w in weights]
    out = [int(math.floor(x)) for x in raw]
    rest = total - sum(out)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in order[:rest]:
        out[i] += 1
    return out


def match_letters(ctc_chars: list[str], char_spans: list[tuple[int, int]],
                  text_str: str) -> tuple[list[tuple[int, int] | None], float]:
    """Map each text letter to a CTC (start, end) frame span via difflib.

    Matching the full letter stream (not word lists) tolerates CTC merging or
    splitting words and small pronunciation variants; unmatched letters stay
    ``None``. Returns (spans, coverage).
    """
    import difflib

    sm = difflib.SequenceMatcher(None, "".join(ctc_chars), text_str, autojunk=False)
    spans: list[tuple[int, int] | None] = [None] * len(text_str)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            spans[b + k] = char_spans[a + k]
    matched = sum(s is not None for s in spans)
    return spans, matched / max(1, len(text_str))


def to_frames(letter_spans, sec_per_frame: float, n_ctc_frames: int
              ) -> list[tuple[int, int] | None]:
    """CTC frame spans -> 10 ms PiVo frames, clamped to the audio, None preserved."""
    to_f = lambda t: int(round(t * 100))
    return [
        None if span is None else
        (to_f(max(0, span[0]) * sec_per_frame), to_f(min(n_ctc_frames, span[1]) * sec_per_frame))
        for span in letter_spans
    ]


def word_bounds_from_letters(letter_span, words_text: list[str],
                             speech_start: int, speech_end: int
                             ) -> tuple[list[tuple[int, int]], list[str]]:
    """Per-word frame spans from matched letters; unmatched words interpolate."""
    raw: list[tuple[int, int] | None] = []
    pos = 0
    for w in words_text:
        n = sum(1 for c in w.lower() if c.isalnum())
        spans = [s for s in letter_span[pos:pos + n] if s is not None]
        raw.append((min(s for s, _ in spans), max(e for _, e in spans)) if spans else None)
        pos += n
    bounds: list[tuple[int, int]] = []
    alarms: list[str] = []
    i, n_words = 0, len(raw)
    while i < n_words:
        if raw[i] is not None:
            bounds.append(raw[i])
            i += 1
            continue
        j = i
        while j < n_words and raw[j] is None:
            j += 1
        lo = bounds[-1][1] if bounds else speech_start
        hi = raw[j][0] if j < n_words else speech_end
        if hi <= lo:
            hi = lo + (j - i)
        step = (hi - lo) / (j - i)
        for k in range(j - i):
            bounds.append((int(round(lo + k * step)), int(round(lo + (k + 1) * step))))
        label = f"interpolated:{words_text[i]}"
        alarms.append(label + (f"..{words_text[j-1]}" if j - i > 1 else ""))
        i = j
    return bounds, alarms


@dataclass
class Alignment:
    tokens: list[str]
    token_ids: list[int]
    durations: list[int]
    start_frame: int
    end_frame: int
    alarms: list[str] = field(default_factory=list)
    readback: str = ""
    model_id: str = ""
    audio_sha256: str = ""

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    def validate(self) -> None:
        assert len(self.tokens) == len(self.durations) == len(self.token_ids)
        assert sum(self.durations) == self.n_frames
        assert 0 <= self.start_frame < self.end_frame <= MAX_UTTERANCE_FRAMES
        for tok, d in zip(self.tokens, self.durations):
            assert 0 <= d <= MAX_TOKEN_FRAMES
            if tok in ("<bos>", "<eos>"):
                assert d == 0
            elif tok not in {".", ",", "?", "!", ";", ":", "<space>"}:
                assert d >= 1, f"phone {tok} has zero frames"


def align(
    wav_path: Path,
    text: str,
    *,
    processor,
    model,
    model_id: str = "",
    frame_seconds: float | None = None,
    lang: str = "id",
    en_phonemize=None,
) -> Alignment:
    """Align one WAV to ``text`` and return per-token frame durations.

    ``lang="id"`` (default) phonemizes with the Indonesian frontend and aligns
    with the Indonesian CTC model.  ``lang="en"`` uses PiVo's own CMU-dictionary
    English frontend (passed in as ``en_phonemize``) and an English CTC model,
    so bilingual corpora keep one duration-label format.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if lang == "en":
        if en_phonemize is None:
            raise AlignError("lang='en' needs en_phonemize (PiVo pivo.text.phonemize)")
        pr = en_phonemize(text, strict=False)
    else:
        from frontend import phonemize
        pr = phonemize(text)
    tokens = list(pr.symbols)
    pcm, rate = read_wav_mono(Path(wav_path))
    pcm16 = resample_to_16k(pcm, rate)
    if rate != SR:
        alarms_prefix = [f"resampled:{rate}->{SR}"]
    else:
        alarms_prefix = []

    n_total = len(pcm16) // HOP
    energy = frame_energy_db(pcm16)

    runs, ctc_n_frames = ctc_runs(processor, model, pcm16)
    if not runs:
        raise AlignError("CTC produced no characters")
    words_text = [w for w in _split_text_words(pr.normalized_text) if w[0].isalpha()]
    text_letters = "".join(c.lower() for c in "".join(words_text) if c.isalnum())
    ctc_chars = [ch.lower() for ch, _, _ in runs if ch.isalnum()]
    ctc_spans = [(s, e) for ch, s, e in runs if ch.isalnum()]
    if not text_letters:
        raise AlignError("text has no letters to align")

    # Letter-stream match: robust to word merges/splits and pronunciation
    # variants; coverage catches truncation or wrong-language audio.
    letter_spans, coverage = match_letters(ctc_chars, ctc_spans, text_letters)
    if coverage < MIN_COVERAGE:
        raise AlignError(
            f"read-back coverage {coverage:.2f} < {MIN_COVERAGE}: "
            f"heard {' '.join(ctc_chars)[:200]!r}")

    # CTC frame duration = audio seconds / number of CTC frames.
    sec_per_frame = frame_seconds or (len(pcm16) / SR / ctc_n_frames)
    letter_frames = to_frames(letter_spans, sec_per_frame, ctc_n_frames)

    alarms = list(alarms_prefix)
    if coverage < 0.9:
        alarms.append(f"coverage:{coverage:.2f}")

    # Energetic speech region; used both as clamp and interpolation bounds.
    threshold = float(np.max(energy)) + ENERGETIC_REL_DB
    speech = energy > threshold
    if not speech.any():
        raise AlignError("no energetic frames in audio")
    speech_start = int(speech.argmax())
    speech_end = int(len(speech) - speech[::-1].argmax())  # exclusive

    word_bounds, interp_alarms = word_bounds_from_letters(
        letter_frames, words_text, speech_start, min(speech_end, n_total))
    word_bounds = [(max(0, a), min(n_total, b)) for a, b in word_bounds]
    if any(b <= a for a, b in word_bounds):
        raise AlignError("CTC word span collapsed after frame conversion")
    alarms.extend(interp_alarms)

    # CTC occasionally keeps emitting a character through trailing silence (a
    # long final "r" on "segar" measured 186->218 while audio ended at frame
    # 382). Clamp every word span into the energetic speech region so a
    # hallucinated tail cannot stretch the last word.
    clamped = []
    prev_end = 0
    for w, (b0, b1) in zip(words_text, word_bounds):
        c0 = min(max(b0, speech_start), speech_end - 1)
        c1 = min(max(b1, c0 + 1), speech_end)
        c0 = max(c0, prev_end)
        if c1 <= c0:
            c1 = min(c0 + 1, speech_end)
        if abs(c0 - b0) >= 5 or abs(c1 - b1) >= 5:
            alarms.append(f"clamped:{w}:{b0}-{b1}->{c0}-{c1}")
        clamped.append((c0, c1))
        prev_end = c1
    word_bounds = clamped

    first = word_bounds[0][0]
    last = word_bounds[-1][1]
    start = max(0, first - MARGIN)
    end = min(n_total, last + MARGIN)
    lo = max(0, start - MAX_EXTEND)
    while start > lo and energy[start - 1] > threshold:
        start -= 1
    hi = min(n_total, len(energy), end + MAX_EXTEND)
    while end < hi and energy[end] > threshold:
        end += 1
    n = end - start
    if n <= 0:
        raise AlignError("empty clip after trimming")

    group_spans = word_group_spans(tokens)
    if len(group_spans) != len(word_bounds):
        raise AlignError(f"phone-group/word count mismatch {len(group_spans)} vs {len(word_bounds)}")

    dur = [0] * len(tokens)
    for g_i, (span, bounds) in enumerate(zip(group_spans, word_bounds)):
        phones = [tokens[i] for i in span]
        b0, b1 = bounds[0] - start, bounds[1] - start
        total = max(1, b1 - b0)
        word = words_text[g_i].lower()
        if len(phones) == len(word) and word.isalpha():
            weights = [1.0] * len(phones)
        else:
            weights = [_weight(p) for p in phones]
            alarms.append(f"weighted:{word}:{len(phones)}phones/{len(word)}letters")
        for ti, d in zip(span, _allocate(weights, total)):
            dur[ti] += d

    # Gaps between words go to punctuation between them, else to <space>, else previous phone.
    for k in range(1, len(group_spans)):
        prev_last = group_spans[k - 1][-1]
        cur_first = group_spans[k][0]
        gap = (word_bounds[k][0] - start) - (word_bounds[k - 1][1] - start)
        if gap <= 0:
            continue
        between = list(range(prev_last + 1, cur_first))
        punct = [j for j in between if tokens[j] in {".", ",", "?", "!", ";", ":"}]
        sp = [j for j in between if tokens[j] == "<space>"]
        target = punct[0] if punct else (sp[0] if sp else prev_last)
        dur[target] += gap

    # Leading margin -> first phone; trailing margin -> final punctuation or last phone.
    dur[group_spans[0][0]] += word_bounds[0][0] - start
    tail = end - word_bounds[-1][1]
    after = [i for i, t in enumerate(tokens) if i > group_spans[-1][-1] and t not in ("<eos>", "<space>")]
    dur[after[0] if after else group_spans[-1][-1]] += tail
    token_pos = [i for span in group_spans for i in span]

    if any(d < 0 for d in dur):
        raise AlignError("negative duration after gap assignment")
    for k, ti in enumerate(token_pos):
        if dur[ti] == 0:
            nbrs = [token_pos[j] for j in (k - 1, k + 1) if 0 <= j < len(token_pos)]
            donor = max(nbrs, key=lambda x: dur[x], default=None)
            if donor is None or dur[donor] < 2:
                raise AlignError("zero_duration_phone with no donor")
            dur[donor] -= 1
            dur[ti] = 1
            alarms.append(f"borrowed:{tokens[ti]}")
    if sum(dur) != n:
        raise AlignError(f"duration sum {sum(dur)} != clip frames {n}")

    result = Alignment(
        tokens=tokens,
        token_ids=list(pr.token_ids),
        durations=dur,
        start_frame=start,
        end_frame=end,
        alarms=alarms,
        readback="".join(ctc_chars),
        model_id=model_id,
        audio_sha256=hashlib.sha256(Path(wav_path).read_bytes()).hexdigest(),
    )
    result.validate()
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Word-level CTC alignment to PiVo phone frames")
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    processor, model = load_ctc(a.model_dir)
    res = align(a.wav, a.text, processor=processor, model=model, model_id=str(a.model_dir))
    print(json.dumps({
        "tokens": res.tokens,
        "token_ids": res.token_ids,
        "durations": res.durations,
        "n_frames": res.n_frames,
        "start_frame": res.start_frame,
        "end_frame": res.end_frame,
        "alarms": res.alarms,
        "readback": res.readback,
        "audio_sha256": res.audio_sha256,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
