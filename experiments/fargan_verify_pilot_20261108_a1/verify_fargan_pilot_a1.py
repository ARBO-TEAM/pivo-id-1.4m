"""Verify pinned FARGAN extractor + PiVo vocoder resynthesis on the provisional
teacher pilot WAV. Read-only w.r.t. teacher/frontend/trainer; every output is a
new uniquely-named file under this experiment directory (no overwrites).

Pilot (provisional, never listened to in this session):
  E:/test/tts-test/split-supertonic-v3/output/teacher_fp32_indonesia_pilot.wav
  44.1 kHz mono PCM16, 387072 frames (~8.78 s).

Pinned FARGAN extractor (what PiVo's pipeline pins, see
  pivo-v1-multilingual/training/data_pipeline/pivo_kn_v3/prep.py:extract_features):
  opus-build-fargan/fargan_demo.exe -features <in.s16> <out.f32>
  -> 20 float32 per full 160-sample frame (= NB_FEATURES in opus-reference/dnn/lpcnet.h;
     NB_TOTAL_FEATURES=36 is the internal width, the demo writes only the first 20).
  Vocoder provenance in PiVo-1.4M/checkpoints/vocoder.pt: kind=pivo-fargan-v1,
  converted from Xiph/Opus FARGAN rev bafd40dc... checkpoint far..._50.pth
  (trained_steps=0 for the PiVo wrapper; the FARGAN weights themselves are the
  upstream conversion). ONNX conditioning/vocoder_step are the resynthesis path.

Resampling 44.1k -> 16k: ffmpeg libswresample (aresample) is the primary path
(soxr resampler if the ffmpeg build provides it, else the default swr resampler;
the chosen one is recorded in the report). A pure-numpy FFT band-limited
fallback is included only for environments without ffmpeg and is flagged in the
report. No scipy / soxr / torchaudio is installed in .venv, hence this choice.

Feature/sample alignment (PiVo + Opus conventions):
  * hop = 160 samples (10 ms) @ 16 kHz; frames = floor(n16 / 160); the trailing
    partial frame (<160 samples) is discarded by the extractor (it only reads
    full frames) and is NOT resynthesized.
  * feature k is centred on source samples [160k-80, 160k+80) (PiVo audio.py
    native_feature_audio: 80-sample half-frame delay). The vocoder consumes
    feature frames directly (runtime.py: context = previous/current/next frame),
    so no extra audio shift is applied here.
  * resynthesis output is frames*160 samples @ 16 kHz, de-emphasized (0.85,
    state carried across frames, exactly like PiVo runtime.py).

TEMP/TMP: forced onto this experiment directory on E: (C: is full per audit).
No input file is modified; outputs are asserted non-existent before writing.

Usage:
  .venv/Scripts/python.exe experiments/fargan_verify_pilot_20261108_a1/verify_fargan_pilot_a1.py [--max-frames N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
TMP_DIR = EXP_DIR / "tmp"
PILOT_WAV = Path(
    r"E:\test\tts-test\split-supertonic-v3\output\teacher_fp32_indonesia_pilot.wav"
)
FARGAN_DEMO = Path(r"E:\test\tts-test\pivo1-4m\opus-build-fargan\fargan_demo.exe")
ONNX_DIR = Path(r"E:\test\tts-test\pivo1-4m\PiVo-1.4M\onnx")
SR_IN = 44100
SR_OUT = 16000
HOP = 160
FEATURE_DIM = 20
OPUS_TOTAL_FEATURES = 36
PREEMPH = 0.85

# Force all temp on E: before tempfile / subprocesses use it.
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TEMP", str(TMP_DIR))
os.environ.setdefault("TMP", str(TMP_DIR))
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)
tempfile.tempdir = str(TMP_DIR)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fresh(path: Path) -> Path:
    """No-overwrite guard: refuse to clobber an existing experiment file."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    return path


def read_pilot_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        info = {
            "nchannels": w.getnchannels(),
            "sampwidth": w.getsampwidth(),
            "framerate": w.getframerate(),
            "nframes": w.getnframes(),
        }
        if info["nchannels"] != 1 or info["sampwidth"] != 2 or info["framerate"] != SR_IN:
            raise ValueError(f"unexpected pilot WAV format: {info}")
        audio = np.frombuffer(w.readframes(info["nframes"]), dtype="<i2")
    return audio.astype(np.float32) / 32768.0


def _ffmpeg_has_soxr() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-h", "filter=aresample"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return "soxr" in (r.stdout + r.stderr).lower()


def _resample_fft_fallback(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Pure-numpy FFT band-limited resample (fallback only, flagged in report)."""
    n_out = int(round(len(audio) * sr_out / sr_in))
    spectrum = np.fft.rfft(audio)
    out_spectrum = np.zeros(n_out // 2 + 1, dtype=complex)
    keep = min(len(spectrum), len(out_spectrum))
    out_spectrum[:keep] = spectrum[:keep]
    return (np.fft.irfft(out_spectrum, n=n_out) * (n_out / len(audio))).astype(
        np.float32
    )


def _fresh_work_dir(base: Path) -> Path:
    """Per-run work dir under E: experiment tmp, so re-runs never overwrite."""
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    d = base / f"run_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resample_44k_to_16k(audio: np.ndarray, work: Path) -> tuple[np.ndarray, str]:
    """Return (16 kHz float32 mono, method). Primary: ffmpeg aresample."""
    raw = fresh(work / "pilot_44k.s16")
    (audio * 32767).round().astype("<i2").tofile(raw)
    out = fresh(work / "pilot_16k.s16")
    method = "ffmpeg-aresample-default"
    if shutil.which("ffmpeg") is not None:
        af = "aresample=16000:resampler=soxr:precision=28" if _ffmpeg_has_soxr() else "aresample=16000"
        if _ffmpeg_has_soxr():
            method = "ffmpeg-aresample-soxr-VHQ-ish(precision=28)"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", str(SR_IN), "-ac", "1", "-i", str(raw),
            "-af", af, "-f", "s16le", "-ar", str(SR_OUT), "-ac", "1", str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        pcm16 = np.fromfile(out, dtype="<i2")
        return (pcm16.astype(np.float32) / 32768.0), method
    # Fallback path (no ffmpeg): FFT band-limited.
    return _resample_fft_fallback(audio, SR_IN, SR_OUT), "numpy-fft-fallback"


def extract_fargan_features(audio: np.ndarray, work: Path) -> tuple[np.ndarray, int, int]:
    """Run pinned fargan_demo -features; return ((frames,20) float32, width, frames).

    Input is float32 mono in [-1, 1] (16 kHz). It is scaled to int16 PCM16
    (x*32767, rounded) before writing -- casting float directly to int16 would
    truncate every sample in (-1, 1) to 0. A silent-input guard below catches
    that failure mode.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("non-finite audio passed to extractor")
    frames = len(audio) // HOP
    src = fresh(work / "pilot_16k_in.s16")
    dst = fresh(work / "pilot_feats_raw.f32")
    pcm16 = (audio.clip(-1, 1) * 32767).round().astype("<i2")
    if np.max(np.abs(audio)) > 1e-4 and not np.any(pcm16):
        raise ValueError("extractor PCM16 is all-zero despite non-silent input (scaling bug)")
    pcm16.tofile(src)
    subprocess.run(
        [str(FARGAN_DEMO), "-features", str(src), str(dst)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    flat = np.fromfile(dst, dtype="<f4")
    width = None
    for w in (FEATURE_DIM, OPUS_TOTAL_FEATURES):
        if frames and flat.size == frames * w:
            width = w
    if width is None:
        raise ValueError(
            f"feature file size {flat.size} floats does not match {frames} frames "
            f"at width 20 or 36"
        )
    feats = flat.reshape(frames, width)[:, :FEATURE_DIM].copy()
    if not np.isfinite(feats).all():
        raise ValueError("non-finite FARGAN features")
    return feats, width, frames


def periods_from_features(feats: np.ndarray) -> np.ndarray:
    """Pinned FARGAN feature 18: log2(256/period) - 1.5 (PiVo audio.py)."""
    return (
        np.rint(256 / np.power(2.0, np.clip(feats[..., 18], -10, 10) + 1.5))
        .clip(32, 255)
        .astype(np.int64)
    )


def deemphasis(audio: np.ndarray, state: float = 0.0) -> tuple[np.ndarray, float]:
    out = np.empty_like(audio, dtype=np.float32)
    for i, sample in enumerate(audio):
        state = float(sample) + PREEMPH * state
        out[i] = state
    return out, state


def resynth_pivo_vocoder(feats: np.ndarray, max_frames: int | None) -> np.ndarray:
    """PiVo ONNX vocoder resynthesis, replicating PiVo-1.4M runtime.py lines for
    conditioning + vocoder_step + deemphasis (decoder/acoustic are NOT used:
    the teacher features replace the decoder output)."""
    import onnxruntime as ort

    n = len(feats) if max_frames is None else min(len(feats), max_frames)
    feats = feats[:n]
    periods_all = periods_from_features(feats)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    conditioning_sess = ort.InferenceSession(
        str(ONNX_DIR / "conditioning.onnx"), sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    step_sess = ort.InferenceSession(
        str(ONNX_DIR / "vocoder_step.onnx"), sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    state_names = ("memory", "gru1", "gru2", "gru3", "frame_state")
    state_widths = (256, 160, 128, 128, 164)
    state = {name: np.zeros((1, w), dtype=np.float32) for name, w in zip(state_names, state_widths)}
    deemph_state = 0.0
    chunks: list[np.ndarray] = []
    for i in range(n):
        prev = feats[max(i - 1, 0)]
        cur = feats[i]
        nxt = feats[min(i + 1, n - 1)]
        context = np.stack((prev, cur, nxt))[None].astype(np.float32)
        periods = periods_all[max(i - 1, 0) : min(i + 1, n - 1) + 1][None]
        # context has 3 frames; periods must match context_frames=3
        periods = np.array([[periods_all[max(i - 1, 0)], periods_all[i], periods_all[min(i + 1, n - 1)]]], dtype=np.int64)
        conditioning = conditioning_sess.run(None, {"features": context, "periods": periods})[0]
        frame = []
        for sub in range(4):
            seg = conditioning[:, 0, sub * 80 : (sub + 1) * 80]
            result = step_sess.run(
                None, {"conditioning": seg, "period": periods[:, 1], **state}
            )
            state = dict(zip(state_names, result[1:]))
            frame.append(result[0][0])
        pcm, deemph_state = deemphasis(np.concatenate(frame).astype(np.float32), deemph_state)
        chunks.append(pcm)
    return np.concatenate(chunks).astype(np.float32) if chunks else np.empty(0, dtype=np.float32)


def resynth_opus_fargan(feats: np.ndarray, work: Path, max_frames: int | None) -> tuple[np.ndarray, int]:
    """Reference resynthesis with the ORIGINAL Opus FARGAN in fargan_demo
    (-fargan-synthesis). Note its framing: it prepends 5 copies of frame 0 as
    context and skips the first 80 output samples, then emits full frames plus
    a final half frame. Length is therefore ~(frames+1)*160-80ish; we report
    the exact sample count instead of forcing equality with the PiVo path."""
    n = len(feats) if max_frames is None else min(len(feats), max_frames)
    src = fresh(work / "pilot_feats20_in.f32")
    dst = fresh(work / "pilot_opus_fargan_out.s16")
    np.ascontiguousarray(feats[:n], dtype="<f4").tofile(src)
    subprocess.run(
        [str(FARGAN_DEMO), "-fargan-synthesis", str(src), str(dst)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    pcm = np.fromfile(dst, dtype="<i2").astype(np.float32) / 32768.0
    return pcm, len(pcm)


def write_wav(path: Path, audio: np.ndarray, rate: int = SR_OUT) -> None:
    path = fresh(path)
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite audio for {path}")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((audio.clip(-1, 1) * 32767).round().astype("<i2").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None, help="limit frames (tests)")
    args = ap.parse_args()

    for p in (PILOT_WAV, FARGAN_DEMO, ONNX_DIR / "conditioning.onnx", ONNX_DIR / "vocoder_step.onnx"):
        if not p.is_file():
            print(f"missing required input: {p}", file=sys.stderr)
            return 2

    work = _fresh_work_dir(TMP_DIR)  # per-run dir on E:; re-runs never overwrite
    report: dict = {
        "pilot_wav": str(PILOT_WAV),
        "pilot_sha256": sha256(PILOT_WAV),
        "fargan_demo": str(FARGAN_DEMO),
        "fargan_demo_sha256": sha256(FARGAN_DEMO),
        "onnx_conditioning_sha256": sha256(ONNX_DIR / "conditioning.onnx"),
        "onnx_vocoder_step_sha256": sha256(ONNX_DIR / "vocoder_step.onnx"),
        "provisional": True,
        "listened_to": False,
        "temp_dir": str(TMP_DIR),
        "work_dir": str(work),
    }

    audio44 = read_pilot_wav(PILOT_WAV)
    report["input"] = {
        "sample_rate": SR_IN,
        "n_samples": int(len(audio44)),
        "duration_s": float(len(audio44) / SR_IN),
        "rms": float(np.sqrt(np.mean(audio44 ** 2))),
        "peak": float(np.max(np.abs(audio44))),
    }

    audio16, method = resample_44k_to_16k(audio44, work)
    report["resample"] = {
        "method": method,
        "n_samples_16k": int(len(audio16)),
        "expected_16k": float(len(audio44) * SR_OUT / SR_IN),
        "rms": float(np.sqrt(np.mean(audio16 ** 2))),
        "peak": float(np.max(np.abs(audio16))),
    }

    feats, width, frames = extract_fargan_features(audio16, work)
    periods = periods_from_features(feats)
    report["features"] = {
        "frames": int(frames),
        "hop": HOP,
        "record_width_floats": int(width),
        "feature_dim_used": FEATURE_DIM,
        "trailing_partial_samples_discarded": int(len(audio16) - frames * HOP),
        "finite": bool(np.isfinite(feats).all()),
        "per_dim_min": [float(x) for x in feats.min(axis=0)],
        "per_dim_max": [float(x) for x in feats.max(axis=0)],
        "per_dim_mean": [float(x) for x in feats.mean(axis=0)],
        "period_min": int(periods.min()),
        "period_max": int(periods.max()),
        "period_median": float(np.median(periods)),
    }
    npy_path = fresh(EXP_DIR / "pilot_feats20_a1.npy")
    np.save(npy_path, feats)
    report["features_npy"] = str(npy_path)

    synth = resynth_pivo_vocoder(feats, args.max_frames)
    synth_wav = EXP_DIR / "pilot_resynth_pivo_vocoder_a1.wav"
    write_wav(synth_wav, synth)
    used_frames = len(feats) if args.max_frames is None else min(len(feats), args.max_frames)
    report["pivo_vocoder_resynth"] = {
        "wav": str(synth_wav),
        "frames_used": int(used_frames),
        "n_samples": int(len(synth)),
        "expected_samples_frames_x_160": int(used_frames * HOP),
        "sample_rate": SR_OUT,
        "finite": bool(np.isfinite(synth).all()),
        "rms": float(np.sqrt(np.mean(synth ** 2))) if len(synth) else 0.0,
        "peak": float(np.max(np.abs(synth))) if len(synth) else 0.0,
        "notes": "conditioning+vocoder_step ONNX, context=prev/cur/next frame, deemphasis 0.85 state carried; no acoustic/decoder used",
    }

    opus_pcm, opus_n = resynth_opus_fargan(feats, work, args.max_frames)
    opus_wav = EXP_DIR / "pilot_resynth_opus_fargan_a1.wav"
    write_wav(opus_wav, opus_pcm)
    report["opus_fargan_reference_resynth"] = {
        "wav": str(opus_wav),
        "frames_used": int(used_frames),
        "n_samples": int(opus_n),
        "finite": bool(np.isfinite(opus_pcm).all()),
        "rms": float(np.sqrt(np.mean(opus_pcm ** 2))) if len(opus_pcm) else 0.0,
        "peak": float(np.max(np.abs(opus_pcm))) if len(opus_pcm) else 0.0,
        "notes": "original fargan_demo -fargan-synthesis; prepends frame0 x5 context, skips first 80 samples, appends final half-frame on flush",
    }

    report_path = fresh(EXP_DIR / "report_fargan_verify_pilot_a1.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
