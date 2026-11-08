# FARGAN verification on provisional pilot — technical fixture only

**Status: technical audio fixture, NOT a valid teacher gold sample.**
Per coordinator note, the pilot was generated with an earlier runner that has
since been corrected. Do not use it (or anything derived here) as teacher
ground truth, training data, or a quality reference. Nothing in this session
listened to any audio; no audio-quality claim is made.

## What was run

Pinned extractor = PiVo's pipeline extractor: `opus-build-fargan/fargan_demo.exe
-features` (Opus `dnn/fargan_demo.c`, writes `NB_FEATURES=20` floats/frame;
`NB_TOTAL_FEATURES=36` is internal only). Vocoder = PiVo-1.4M ONNX
`conditioning.onnx` + `vocoder_step.onnx` (converted from Xiph/Opus FARGAN
rev `bafd40dc...`, per `PiVo-1.4M/checkpoints/vocoder.pt`), plus the original
`fargan_demo -fargan-synthesis` as reference. Resampling 44.1k→16k via
ffmpeg `aresample` with SoXR (`precision=28`) — recorded in the report; a
pure-numpy FFT fallback exists for no-ffmpeg environments and is flagged.

Pilot: `split-supertonic-v3/output/teacher_fp32_indonesia_pilot.wav`
(44.1 kHz mono PCM16, 387072 samples, 8.78 s, sha256 `82f9ec2e...`).
→ 140434 samples @16k → **877 frames × 20 dims**, record width **20** (not 36),
finite, trailing 114 samples discarded (partial frame).

## Commands (all TEMP/TMP forced to `tmp/` on E:; C: is full)

```bash
cd /e/test/tts-test/pivo1-4m
TEMP=E:/test/tts-test/pivo1-4m/experiments/fargan_verify_pilot_20261108_a1/tmp \
TMP=E:/test/tts-test/pivo1-4m/experiments/fargan_verify_pilot_20261108_a1/tmp \
  ./.venv/Scripts/python.exe -m pytest experiments/fargan_verify_pilot_20261108_a1/test_verify_fargan_pilot_a1.py -q -p no:cacheprovider
TEMP=... TMP=... \
  ./.venv/Scripts/python.exe experiments/fargan_verify_pilot_20261108_a1/verify_fargan_pilot_a1.py
# re-runs refuse to overwrite (fresh() guard + per-run tmp/run_<timestamp>/)
```

Tests: **10 passed**. Full run exit 0. No teacher/frontend/trainer file was
changed (this dir only).

## Feature/sample alignment

- Hop 160 @16 kHz (10 ms). Feature k is centred on source samples
  `[160k-80, 160k+80)` (PiVo `audio.py` half-frame delay). Vocoder consumes
  frames directly with prev/cur/next context — no extra audio shift.
- PiVo resynthesis output = `frames×160` samples (140320), de-emphasis 0.85
  with state carried across frames, exactly like PiVo `runtime.py`.
- Original FARGAN synthesis prepends frame-0 ×5 as context, skips the first
  80 output samples, appends a final half-frame on flush — so its frame/sample
  mapping differs at the edges by design (both still 140320 samples here by
  coincidence of flush behaviour; see script docstring).

## Results (technical, not perceptual)

- Resynthesis RMS/peak match the 16 kHz source closely (PiVo 0.0490/0.279,
  Opus 0.0499/0.277, source 0.0495/0.259) and both are finite.
- Waveform correlation of resynthesis vs the 16 kHz source is low
  (PiVo −0.05, Opus +0.02; PiVo-vs-Opus 0.34, SNR −1.3 dB). This is
  expected for a neural vocoder: it regenerates phase/excitation, it does not
  copy the waveform. Low correlation alone says nothing about perceived
  quality either way.

## Honest limitations

1. **Fixture invalid as gold**: pilot came from a corrected runner; treat as
   arbitrary 44.1 kHz audio. Feature sanity (finite, varying, plausible
   cepstra, period median 89) only proves the extractor ran, not that the
   audio was the intended Indonesian utterance, the intended voice/text, or
   even speech at all. Per-second spectra show a strong ~160–280 Hz tone —
   consistent with voiced speech/buzz, but **not listened to or transcribed**.
2. **No listening / no intelligibility / no MOS** — no quality claim.
3. The earlier silent-feature run was a **bug in this experiment script**
   (float→int16 truncation fed all-zero PCM to the extractor); it was caught
   by the zero-variance features, fixed with explicit `×32767` scaling, a
   silent-input guard, and a regression test. The first `report_*.json`
   (877 frames of constant features, ~silent resynthesis) was deleted and
   replaced; the current report is the corrected one.
4. Resampling choice (ffmpeg SoXR) differs from the training pipeline's
   `soxr`/`scipy resample_poly` options; anti-aliasing/phase details may
   shift features slightly. PiVo vocoder `trained_steps=0` refers to the
   PiVo wrapper (weights are the upstream FARGAN conversion).
5. Edge frames: extractor discards the trailing partial frame; Opus
   synthesis has its own 80-sample skip/flush behaviour — first/last ~80
   samples are not sample-aligned across the two resynthesis paths.

## Files (all new, in this directory)

- `verify_fargan_pilot_a1.py` — main script (resample → extract → PiVo + Opus resynthesis)
- `test_verify_fargan_pilot_a1.py` — 10 tests
- `report_fargan_verify_pilot_a1.json` — full numbers + input/model sha256
- `pilot_feats20_a1.npy` — (877, 20) float32 features
- `pilot_resynth_pivo_vocoder_a1.wav`, `pilot_resynth_opus_fargan_a1.wav` — 16 kHz resyntheses (unlistened)
- `README_fargan_verify_a1.md` — this file; `tmp/` — per-run intermediates on E:
