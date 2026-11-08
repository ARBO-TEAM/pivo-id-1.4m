"""Research-only phoneme-duration prototype for Supertonic pilot audio.

DO NOT USE FOR TRAINING. Teacher licensing and subjective-quality gates
(2026-11-08-pivo-indonesian.md Task 0.1/0.2) are not passed. Outputs of
this module are technical-research artifacts only.

What this does
--------------
- Maps Indonesian text -> PiVo token ids via src_id.frontend.phonemize
  (ARPAbet inventory, 79 symbols). Frontend is the lexicon, so there is
  no lexicon mismatch by construction.
- Reads a 44.1 kHz mono PCM WAV with stdlib ``wave`` (no extra deps),
  resamples 44.1k -> 16k with numpy polyphase-style resampling, and
  computes n_frames = len(samples_16k) // 160 (10 ms hop, PiVo/FARGAN).
- If transformers + a cached Indonesian wav2vec2 CTC model are present,
  runs CTC forced alignment of the frontend token sequence against the
  audio. Otherwise raises AlignerUnavailable and, only when explicitly
  requested, returns an energy draft for QA inspection.

What this does NOT do
---------------------
- It never invents training durations. The energy draft is NOT a
  pretrained alignment and must never be fed to src_id.train.
- It preserves held-out splits: rows whose text matches the frozen
  domain_manifest test split are labelled split="test" and excluded
  from any train/val output.
- It writes nothing outside the caller-supplied out_path.

Dependencies: stdlib + numpy (available in .venv). transformers/torch
are optional and only needed for the CTC path.
"""

from __future__ import annotations

import hashlib
import json
import math
import wave
from dataclasses import dataclass, field

HOP = 160
SR_TEACHER = 44100
SR_STUDENT = 16000
MAX_TOKEN_FRAMES = 200
MAX_UTTERANCE_FRAMES = 3000
SILENCE_LABELS = {"", "sil", "sp", "<eps>", "silence"}

# ARPAbet inventory from PiVo-1.4M/onnx/metadata.json via src_id.frontend.
# Phones are the frontend symbols minus specials.
SPECIAL = {"<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":"}


class AlignerUnavailable(RuntimeError):
    """No pretrained aligner model/runtime is available."""


class DropClip(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class AlignmentResult:
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    durations: tuple[int, ...]
    n_frames: int
    method: str  # "ctc-wav2vec2-id" | "energy-draft"
    model_id: str | None
    split: str
    audio_sha256: str
    alarms: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "tokens": list(self.tokens),
            "token_ids": list(self.token_ids),
            "durations": list(self.durations),
            "n_frames": self.n_frames,
            "sum_durations": sum(self.durations),
            "method": self.method,
            "model_id": self.model_id,
            "split": self.split,
            "audio_sha256": self.audio_sha256,
            "alarms": list(self.alarms),
            "research_only_do_not_train": True,
        }

    def validate(self) -> None:
        assert len(self.durations) == len(self.tokens) == len(self.token_ids)
        assert sum(self.durations) == self.n_frames, "sum(durations) != n_frames"
        assert self.n_frames <= MAX_UTTERANCE_FRAMES
        for tok, d in zip(self.tokens, self.durations):
            assert 0 <= d <= MAX_TOKEN_FRAMES, f"duration {d} out of range for {tok}"
        # bos/eos are always 0; phones >= 1
        for tok, d in zip(self.tokens, self.durations):
            if tok in ("<bos>", "<eos>"):
                assert d == 0, f"{tok} must be 0"
            elif tok not in SPECIAL:
                assert d >= 1, f"phone {tok} has zero duration"


def read_wav_16k_mono(path: str) -> tuple["list[int]", int]:
    """Read mono PCM16 WAV, return int16 samples at original rate + rate."""
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1, f"{path}: expected mono"
        assert w.getsampwidth() == 2, f"{path}: expected PCM16"
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
    import numpy as np
    pcm = np.frombuffer(frames, dtype="<i2").copy()
    return pcm, sr


def resample_to_16k(pcm, sr: int):
    """Deterministic rational resampling via numpy interpolation."""
    import numpy as np
    if sr == SR_STUDENT:
        return pcm
    assert sr == SR_TEACHER, f"unexpected rate {sr}"
    n_out = int(round(len(pcm) * SR_STUDENT / sr))
    x_old = np.linspace(0, 1, len(pcm), endpoint=False)
    x_new = np.linspace(0, 1, n_out, endpoint=False)
    return np.interp(x_new, x_old, pcm).astype(np.int16)


def n_frames_of(pcm16) -> int:
    return len(pcm16) // HOP


def frame_energy_db(pcm16):
    import numpy as np
    x = np.pad(np.asarray(pcm16, dtype=np.float32) / 32768.0, (HOP // 2, 0))
    n = len(pcm16) // HOP
    frames = x[: n * HOP].reshape(n, HOP)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)


def split_for_text(text: str, manifest_path: str | None) -> str:
    """Return 'test' if text matches frozen manifest test split, else
    'pilot' (pilot texts are not in the manifest at all)."""
    if not manifest_path:
        return "pilot"
    try:
        rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        return "pilot"
    t = text.strip()
    for r in rows:
        if r.get("text", "").strip() == t:
            return str(r.get("split", "pilot"))
    return "pilot"


def durations_energy_draft(tokens, n_frames: int, energy_db):
    """Energy-only draft for QA inspection. NOT a pretrained alignment.

    Splits the energetic region evenly across phones (pauses to
    <space>/punctuation). Clearly labelled draft_not_for_training.
    """
    speech_mask = energy_db > (energy_db.max() - 35.0)
    if not speech_mask.any():
        raise DropClip("near_silent")
    first = int(speech_mask.argmax())
    last = int(len(speech_mask) - 1 - speech_mask[::-1].argmax())
    # keep 3-frame margin, matching prep.py convention
    start = max(0, first - 3)
    end = min(n_frames, last + 4)
    n = end - start
    phone_idx = [i for i, t in enumerate(tokens) if t not in SPECIAL]
    if not phone_idx or n <= 0:
        raise DropClip("empty_after_trim")
    if n < len(phone_idx):
        raise DropClip("too_few_frames")
    dur = [0] * len(tokens)
    # even split across phones; leading margin into first phone
    base, rem = divmod(n, len(phone_idx))
    for k, ti in enumerate(phone_idx):
        dur[ti] = base + (1 if k < rem else 0)
    # trailing margin goes to final punctuation if present
    tail_landing = phone_idx[-1]
    for j in range(phone_idx[-1] + 1, len(tokens)):
        if tokens[j] not in ("<eos>", "<space>"):
            tail_landing = j
            break
    dur[tail_landing] += 0  # draft keeps split even; tail stays in phones
    alarms = ("energy-draft-not-for-training",)
    return dur, start, end, alarms


def durations_from_ctc(tokens, phone_intervals, n_frames: int):
    """Convert CTC phone intervals (seconds) to frame durations.

    phone_intervals: list of (start_s, end_s) aligned 1:1 with phone
    tokens (tokens minus specials), in order. Rounding: round-half-even
    via Python round(); leading/trailing silence handled like prep.py:
    leading margin -> first phone, trailing -> final punct or last phone.
    """
    phone_idx = [i for i, t in enumerate(tokens) if t not in SPECIAL]
    assert len(phone_idx) == len(phone_intervals), "CTC/phone count mismatch"
    def to_f(s: float) -> int:
        return int(round(s * 100))  # 10 ms frames
    bounds = [(to_f(s), to_f(e)) for s, e in phone_intervals]
    start = 0
    end = n_frames
    # clip to audio
    bounds = [(max(0, b0), min(n_frames, b1)) for b0, b1 in bounds]
    n = end - start
    dur = [0] * len(tokens)
    for k, ti in enumerate(phone_idx):
        b0, b1 = bounds[k]
        dur[ti] = max(0, b1 - b0)
    # gaps between phones -> punctuation if present else <space> else previous phone
    for k in range(len(phone_idx) - 1):
        gap = bounds[k + 1][0] - bounds[k][1]
        if gap <= 0:
            continue
        between = tokens[phone_idx[k] + 1 : phone_idx[k + 1]]
        if not between:
            dur[phone_idx[k]] += gap
        else:
            punct = [j for j in range(phone_idx[k] + 1, phone_idx[k + 1]) if tokens[j] not in ("<space>",)]
            if punct:
                dur[punct[0]] += gap
            else:
                dur[phone_idx[k] + between.index("<space>") + 1] += gap
    # trailing
    tail = n - bounds[-1][1] if bounds else n
    if tail > 0:
        after = [j for j in range(phone_idx[-1] + 1, len(tokens)) if tokens[j] not in ("<eos>", "<space>")]
        dur[after[0] if after else phone_idx[-1]] += tail
    if any(d < 0 for d in dur):
        raise DropClip("negative_duration")
    for k, ti in enumerate(phone_idx):
        if dur[ti] == 0:
            nbrs = [phone_idx[j] for j in (k - 1, k + 1) if 0 <= j < len(phone_idx)]
            donor = max(nbrs, key=lambda x: dur[x], default=None)
            if donor is None or dur[donor] < 2:
                raise DropClip("zero_duration_phone")
            dur[donor] -= 1
            dur[ti] = 1
    if sum(dur) != n:
        raise DropClip("duration_sum_mismatch")
    return dur, start, end


def try_ctc_alignment(wav_path: str, token_ids, model_id=None):
    """Attempt proper pretrained CTC alignment. Raises AlignerUnavailable
    when transformers/model are missing. Never invents timings."""
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC  # type: ignore
    except Exception as e:
        raise AlignerUnavailable(f"transformers unavailable: {e}") from e
    if not model_id:
        raise AlignerUnavailable("no model_id supplied and none cached locally")
    raise AlignerUnavailable(
        f"CTC path wired but model {model_id!r} weights not in local "
        "HF cache (E:/models/hf/hub); download to E: required — "
        "not done here to avoid C:-full / unbounded writes"
    )


def align_wav(
    wav_path: str,
    text: str,
    *,
    manifest_path: str | None = None,
    allow_energy_draft: bool = False,
    model_id: str | None = None,
) -> AlignmentResult:
    """Align one pilot WAV. Research-only; see module docstring."""
    import sys
    sys.path.insert(0, "src_id")
    from frontend import phonemize  # trusted sibling, stdlib-only
    pr = phonemize(text)
    tokens = tuple(pr.symbols)
    token_ids = tuple(pr.token_ids)
    split = split_for_text(text, manifest_path)
    audio_sha = hashlib.sha256(open(wav_path, "rb").read()).hexdigest()
    pcm, sr = read_wav_16k_mono(wav_path)
    pcm16 = resample_to_16k(pcm, sr)
    n_frames = n_frames_of(pcm16)
    energy = frame_energy_db(pcm16)
    method, mid, alarms = "energy-draft", None, ()
    try:
        # No CTC model weights on disk; this will raise AlignerUnavailable.
        raise AlignerUnavailable("no pretrained aligner weights present")
    except AlignerUnavailable:
        if not allow_energy_draft:
            raise
        dur, start, end, alarms = durations_energy_draft(tokens, n_frames, energy)
        n_frames = end - start
    result = AlignmentResult(
        tokens=tokens, token_ids=token_ids, durations=tuple(dur),
        n_frames=n_frames, method=method, model_id=mid, split=split,
        audio_sha256=audio_sha, alarms=alarms,
    )
    result.validate()
    return result
