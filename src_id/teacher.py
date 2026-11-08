"""Standalone Indonesian pilot synthesizer over local Supertonic FP32 ONNX files.

NOTICE / LICENSE
----------------
This script is an independent reimplementation of the *behavior* observed in
locally inspected Supertonic reference materials. No code from the
``supertonic`` Python package, the Reza2kn LiteRT fork, or the local
``split-supertonic-v3`` research workspace is copied here. Only the model
contracts (tensor names, shapes, sample rate) needed to run the supplied ONNX
graphs are applied, from first principles.

Model weights and voice styles are subject to the upstream BigScience
Open RAIL-M License (see the ``LICENSE`` file shipped with the model files,
e.g. ``supertonic_work/hf_fresh_download_20261108/LICENSE``). Under Section
1(e), a model trained on synthetic output of Supertonic is a Derivative of the
Model and remains bound by the OpenRAIL-M use restrictions. This script is
for lawful personal evaluation only; it does not grant any distribution
rights. Do not redistribute the model files or derived student weights in
violation of those terms. This is not legal advice; consult the license text
and, where needed, the rights holder.

No secrets are read or required. No network access is performed. Nothing is
downloaded. Generated WAV files are never overwritten.

Usage
-----
    python -I teacher.py synthesize \
        --dp PATH --te PATH --ve PATH --voc PATH \
        --config PATH --indexer PATH --voice-style PATH \
        --text "..." --lang id --seed 0 --output out.wav

All paths are explicit arguments; there are no sensible defaults that could
silently pick up the wrong weights. Run from any directory; the script only
reads model files and writes the requested output.

Dependencies: numpy, onnxruntime (>=1.20 with MatMulNBits support when using
INT4 estimators; the FP32 estimator needs plain CPUExecutionProvider).
Set TEMP/TMP to a volume with free space before running on a full disk.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import unicodedata
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100
BASE_CHUNK_SIZE = 512
CHUNK_COMPRESS_FACTOR = 6
LATENT_DIM = 24
DEFAULT_TOTAL_STEPS = 8
DEFAULT_SPEED = 1.05

# --------------------------------------------------------------------------
# Text front end (independent reimplementation of observed behavior)
# --------------------------------------------------------------------------

_SYMBOL_REPLACEMENTS = {
    "–": "-",
    "‑": "-",
    "—": "-",
    "¯": " ",
    "_": " ",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "´": "'",
    "`": "'",
    "[": " ",
    "]": " ",
    "|": " ",
    "/": " ",
    "#": " ",
    "→": " ",
    "←": " ",
}

_ENDING_PATTERN = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")
_PUNCT_SPACING = [
    (re.compile(r" ,"), ","),
    (re.compile(r" \."), "."),
    (re.compile(r" !"), "!"),
    (re.compile(r" \?"), "?"),
    (re.compile(r" ;"), ";"),
    (re.compile(r" :"), ":"),
    (re.compile(r" '"), "'"),
]
_DUP_QUOTES = re.compile(r'(["\'\`])\1+')
_WS = re.compile(r"\s+")


def _strip_emoji(text: str) -> str:
    out = []
    for ch in text:
        o = ord(ch)
        if o >= 0x1F600 or 0x2600 <= o <= 0x27BF or 0x1F1E6 <= o <= 0x1F1FF:
            continue
        out.append(ch)
    return "".join(out)


def preprocess_text(text: str, lang: str) -> str:
    """Normalize text and wrap with ``<lang>...</lang>`` tokens."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not re.fullmatch(r"[a-z]{2}", lang or ""):
        raise ValueError(f"lang must be a 2-letter code, got {lang!r}")
    t = unicodedata.normalize("NFKD", text)
    t = _strip_emoji(t)
    for old, new in _SYMBOL_REPLACEMENTS.items():
        t = t.replace(old, new)
    for bad in "♥☆♡©\\":
        t = t.replace(bad, "")
    t = t.replace("@", " at ")
    t = t.replace("e.g.,", "for example, ").replace("i.e.,", "that is, ")
    for pat, rep in _PUNCT_SPACING:
        t = pat.sub(rep, t)
    t = _DUP_QUOTES.sub(r"\1", t)
    t = _WS.sub(" ", t).strip()
    if not t or not _ENDING_PATTERN.search(t):
        t += "."
    return f"<{lang}>{t}</{lang}>"


class UnicodeIndexer:
    """Maps characters to model token ids via a JSON lookup table.

    The table is a list indexed by code point; ``-1`` marks unsupported
    characters, which are rejected loudly instead of being dropped silently.
    """

    def __init__(self, path: Path | str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("unicode indexer must be a non-empty JSON list")
        self.table = [int(v) for v in data]

    def encode(self, text: str, lang: str) -> tuple[np.ndarray, np.ndarray]:
        prepared = preprocess_text(text, lang)
        ids = []
        missing = []
        for ch in prepared:
            o = ord(ch)
            if o >= len(self.table) or self.table[o] < 0:
                missing.append(repr(ch))
            else:
                ids.append(self.table[o])
        if missing:
            raise ValueError(
                "unsupported characters for model: " + ", ".join(sorted(set(missing)))
            )
        if not ids:
            raise ValueError("input produced no encodable characters")
        text_ids = np.array([ids], dtype=np.int64)
        mask = np.ones((1, 1, len(ids)), dtype=np.float32)
        return text_ids, mask


def load_voice_style(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load style_ttl / style_dp arrays from a Supertonic voice JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        ttl_part = data["style_ttl"]
        dp_part = data["style_dp"]
        ttl = np.asarray(ttl_part["data"], dtype=np.float32)
        dp = np.asarray(dp_part["data"], dtype=np.float32)
        dims_ttl = ttl_part.get("dims")
        dims_dp = dp_part.get("dims")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"voice style file is malformed: {path}") from exc
    if dims_ttl is not None and tuple(dims_ttl) != ttl.shape:
        ttl = ttl.reshape(*dims_ttl)
    if dims_dp is not None and tuple(dims_dp) != dp.shape:
        dp = dp.reshape(*dims_dp)
    if ttl.ndim != 3 or ttl.shape[0] != 1:
        raise ValueError(f"unexpected style_ttl shape {ttl.shape} in {path}")
    if dp.ndim != 3 or dp.shape[0] != 1:
        raise ValueError(f"unexpected style_dp shape {dp.shape} in {path}")
    if not (np.isfinite(ttl).all() and np.isfinite(dp).all()):
        raise ValueError(f"voice style contains non-finite values: {path}")
    return ttl, dp


# --------------------------------------------------------------------------
# ONNX orchestration
# --------------------------------------------------------------------------


def _load_session(path: Path, threads: int, providers: list[str]):
    import onnxruntime as ort

    if not path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, int(threads))
    opts.inter_op_num_threads = 1
    available = ort.get_available_providers()
    chosen = [p for p in providers if p in available]
    if not chosen:
        raise RuntimeError(
            f"none of the requested providers {providers} is available; "
            f"available: {available}"
        )
    return ort.InferenceSession(str(path), sess_options=opts, providers=chosen)


def synthesize(
    *,
    dp_path: Path,
    te_path: Path,
    ve_path: Path,
    voc_path: Path,
    indexer: UnicodeIndexer,
    style_ttl: np.ndarray,
    style_dp: np.ndarray,
    text: str,
    lang: str = "id",
    seed: int = 0,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    speed: float = DEFAULT_SPEED,
    threads: int = 1,
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
) -> np.ndarray:
    """Run the full pipeline deterministically and return float32 PCM."""
    if not 0.7 <= float(speed) <= 2.0:
        raise ValueError("speed must be in [0.7, 2.0]")
    if not 1 <= int(total_steps) <= 100:
        raise ValueError("total_steps must be in [1, 100]")

    dp = _load_session(Path(dp_path), threads, list(providers))
    te = _load_session(Path(te_path), threads, list(providers))
    ve = _load_session(Path(ve_path), threads, list(providers))
    voc = _load_session(Path(voc_path), threads, list(providers))

    text_ids, text_mask = indexer.encode(text, lang)
    duration = dp.run(None, {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask})[0] / speed
    if duration.size != 1 or not np.isfinite(duration).all() or float(duration[0]) <= 0:
        raise ValueError(f"duration predictor returned invalid value: {duration}")
    text_emb = te.run(None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask})[0]

    chunk_size = BASE_CHUNK_SIZE * CHUNK_COMPRESS_FACTOR
    # Match the reference's float32 arithmetic and integer truncation exactly.
    wav_len_max = duration.max() * SAMPLE_RATE
    latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)
    if latent_len < 1:
        raise ValueError(f"duration is too short to produce a latent frame: {duration}")
    rng = np.random.RandomState(seed)
    latent_dim = LATENT_DIM * CHUNK_COMPRESS_FACTOR
    latent = rng.randn(1, latent_dim, latent_len).astype(np.float32)
    latent_mask = np.ones((1, 1, latent_len), dtype=np.float32)
    latent = latent * latent_mask

    total_step = np.array([float(total_steps)], dtype=np.float32)
    for step in range(int(total_steps)):
        latent = ve.run(
            None,
            {
                "noisy_latent": latent,
                "text_emb": text_emb,
                "style_ttl": style_ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": np.array([float(step)], dtype=np.float32),
                "total_step": total_step,
            },
        )[0]

    waveform = np.asarray(voc.run(None, {"latent": latent})[0], dtype=np.float32)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(f"unexpected vocoder output shape {waveform.shape}")
    waveform = waveform[0]
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("vocoder produced empty or non-finite audio")
    return waveform.astype(np.float32)


# --------------------------------------------------------------------------
# Audio I/O (stdlib only, PCM16)
# --------------------------------------------------------------------------


def write_wav(path: Path | str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("refusing to save empty or non-finite audio")
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with path.open("xb") as fh:
        fh.write(b"RIFF")
        fh.write(struct.pack("<I", 36 + 2 * pcm.size))
        fh.write(b"WAVEfmt ")
        fh.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        fh.write(b"data")
        fh.write(struct.pack("<I", 2 * pcm.size))
        fh.write(pcm.tobytes())
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp", type=Path, required=True, help="duration_predictor.onnx")
    parser.add_argument("--te", type=Path, required=True, help="text_encoder.onnx")
    parser.add_argument("--ve", type=Path, required=True, help="vector_estimator.onnx (FP32 preferred)")
    parser.add_argument("--voc", type=Path, required=True, help="vocoder.onnx")
    parser.add_argument("--config", type=Path, required=True, help="tts.json (checked for sample rate)")
    parser.add_argument("--indexer", type=Path, required=True, help="unicode_indexer.json")
    parser.add_argument("--voice-style", type=Path, required=True, help="voice style JSON, e.g. F1.json")
    parser.add_argument("--text", required=True, help="text to synthesize")
    parser.add_argument("--lang", default="id", help="2-letter language code (default: id)")
    parser.add_argument("--seed", type=int, default=0, help="deterministic RNG seed (default: 0)")
    parser.add_argument("--steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--provider", action="append", default=["CPUExecutionProvider"],
                        help="execution provider, repeatable")
    parser.add_argument("--output", type=Path, required=True, help="new WAV path; never overwritten")
    # Allow the bare "synthesize" sub-command word seen in the docs.
    import sys as _sys

    _args_list = list(argv) if argv is not None else _sys.argv[1:]
    if _args_list and _args_list[0] == "synthesize":
        _args_list = _args_list[1:]
    args = parser.parse_args(_args_list)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    expected = {
        ("ae", "sample_rate"): SAMPLE_RATE,
        ("ae", "base_chunk_size"): BASE_CHUNK_SIZE,
        ("ttl", "chunk_compress_factor"): CHUNK_COMPRESS_FACTOR,
        ("ttl", "latent_dim"): LATENT_DIM,
    }
    for (section, key), value in expected.items():
        actual = cfg.get(section, {}).get(key)
        if actual != value:
            raise ValueError(f"unexpected config {section}.{key}: {actual!r} (expected {value})")

    ttl, dp_style = load_voice_style(args.voice_style)
    indexer = UnicodeIndexer(args.indexer)
    audio = synthesize(
        dp_path=args.dp, te_path=args.te, ve_path=args.ve, voc_path=args.voc,
        indexer=indexer, style_ttl=ttl, style_dp=dp_style,
        text=args.text, lang=args.lang, seed=args.seed,
        total_steps=args.steps, speed=args.speed,
        threads=args.threads, providers=tuple(args.provider),
    )
    out = write_wav(args.output, audio, SAMPLE_RATE)
    print(f"Saved {out} ({SAMPLE_RATE} Hz, {len(audio)/SAMPLE_RATE:.2f} s, "
          f"peak {float(np.max(np.abs(audio))):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
