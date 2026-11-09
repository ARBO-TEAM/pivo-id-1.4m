# Model Card — PiVo-ID v11

Indonesian fine-tune of [PiVo-1.4M](https://huggingface.co/empero-ai/PiVo-1.4M)
(1,401,562 parameters: 584,965 acoustic + 816,597 FARGAN vocoder, unchanged).
Mono 16 kHz, PiVo 20-dim FARGAN features (18 Bark cepstra, log-pitch,
pitch correlation), 10 ms frames. Two voices: **Pi** (female, voice 0) and
**Vo** (male, voice 1).

This is **research**, not a production release. It is a 1.4M-parameter
student trained entirely on synthetic speech from a single teacher;
its output quality is bounded by both its capacity and its teacher.

## Intended use / out-of-scope

- Local, experimentation-only Indonesian TTS on edge / consumer hardware.
- Not for production TTS, public announcements, or voice cloning of real
  people. Output must be marked as machine-generated where applicable
  (see teacher license below).
- Out of scope: code-switching within one sentence, streaming server
  deployment, and any use requiring guaranteed pronunciation of numbers
  or names (see limitations).

## Training data (synthetic, single teacher)

All training audio is synthetic speech rendered by the **Supertonic v3**
teacher (`Supertone/supertonic-3`, BigScience Open RAIL-M) in Indonesian:
44.1 kHz teacher audio resampled to 16 kHz, FARGAN feature extraction,
CTC forced alignment (Indonesian wav2vec2), JSONL manifest
(id/tokens/durations/features_path/voice_id/split). No real human speech
was recorded or used.

Corpus v11: 4,386 rows (train 3,915 / val 471, 1,154,961 frames) —
3,804 non-hyphen rows from the v10 merged corpus + 582 rebuilt rows for
147 unique comma-fixed reduplikasi texts (2 voices × 2 seeds, builder
`experiments/build_pilot_corpus_fast.py`, merge `experiments/merge_corpora.py`,
filter `experiments/filter_hyphen.py`).

Earlier lineage: v6 bilingual (1,857 rows) → v7 ID-only (2,732) →
v8/v9/v10 incremental vocabulary (perusahaan, evakuasi, reduplikasi,
R-trill) → v11 comma-reduplikasi fix.

## Training recipe (v11)

Warm-start from v10 `best.pt` (`--init-ckpt`, no reset of feature stats),
AdamW lr 1e-5 → 1e-6 cosine, warmup 100, **batch 8**, grad clip 1.0,
seed 42, 3,000 steps, loss = masked SmoothL1 on normalized features
+ 0.3·MSE(log duration) + 0.5·masked delta loss. Final validation:
**feature 0.1884, duration 0.0747** (step 3000). ONNX parity:
encoder 2.21e-06, log-durations 9.54e-07, decoder 7.15e-05
(gate 2e-04). Exporter: `experiments/export_onnx.py`.

## Evaluation

- CER (CTC read-back, Indonesian wav2vec2) measures intelligibility
  only, not naturalness; final judgment is by listening.
- Sample WAVs: `experiments/eval/v11_listen/` (rendered with
  `--selective --deemphasis 0.88`).
- No human MOS / blind test has been run.

## Limitations & biases

- Student inherits the teacher's mispronunciations; numbers, prices and
  foreign words are still sometimes wrong in both.
- 1.4M parameters cap quality well below the teacher; expect robotic
  timbre and occasional prosody errors.
- Reduplikasi is now comma-separated (micro-pause between copies);
  letter-by-letter spelling of hyphenated acronyms is preserved.

## Licensing

- PiVo code/weights: Apache-2.0. FARGAN/Opus BSD-3-Clause (retained in
  `LICENSES/`). CMU dictionary permissive notice retained.
- Training audio is Supertonic v3 output; under its Open RAIL-M §1(e)
  this student may be a **Derivative of the Model**. See
  [`research/teacher-audit.md`](research/teacher-audit.md) — do not
  redistribute without reviewing those terms and carrying the license.
  This card is not legal advice.
