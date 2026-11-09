# PiVo-ID 1.4M — bilingual (Indonesian + English) PiVo fine-tune

Experimental Indonesian fine-tune of [PiVo-1.4M](https://huggingface.co/empero-ai/PiVo-1.4M)
that **keeps English working**. This is research, not an official PiVo release and not
a production-ready TTS package. The original model, architecture, vocoder and English
frontend are PiVo's; this repository contains the Indonesian pipeline, the fine-tuned
weights, the ONNX export and the measurements behind every claim below.

- **Acoustic**: 584,965 parameters (+ 816,597 unchanged FARGAN vocoder = 1,401,562 total)
- **Audio**: mono 16 kHz, PiVo's 20-dim FARGAN features, 10 ms frames
- **Voices**: voice 0 = **Pi** (female, teacher F1), voice 1 = **Vo** (male, teacher M1)
- **Languages**: Indonesian and English, same 79-symbol ARPAbet inventory
- **Training audio**: Supertonic v3 synthetic speech only (`E:/test/tts-test/split-supertonic-v3`)

## What is in this repo

| Path | Contents |
|---|---|
| `src_id/` | Training + inference code: `acoustic.py` (PiVo architecture), `train.py`, `frontend.py` (Indonesian G2P), `ctc_align.py` (CTC forced alignment), `teacher.py` (Supertonic driver) |
| `experiments/` | Corpus builder, ONNX exporter, renderers, and the CER/WER evaluation scripts |
| `experiments/text/` | Text manifests: `domain_manifest.jsonl` (Indonesian domain sentences), `english_manifest.jsonl` (English rehearsal), plus the v6 expansion `extra_id_v6.jsonl` (215 demo/domain sentences) and `extra_id_time_v6.jsonl` (85 clock-form sentences) |
| `tests/` | 138 tests (`python -I -m pytest tests -q`) |
| `research/` | Measured results: `training-run-report.md` (main report), `ab-warm-scratch.md`, `pivo-parity.md`, `teacher-audit.md` (license audit) |
| `docs/superpowers/plans/` | The plan this work followed |

The fine-tuned checkpoint and ONNX model are attached to the **GitHub Release**, not
committed to the tree (checkpoints are build artifacts; corpus audio/features are not
redistributed).

| Release asset | Size | What it is |
|---|---:|---|
| `pivo-id-1.4m-onnx.zip` | 5.6 MB | Ready-to-run PiVo-format model folder (see below) |
| `pilot_v5c_best.pt` | 6.8 MB | Raw training checkpoint (weights + AdamW state, for resume/further training) |
| `pivo-id-v6-onnx.zip` | 5.6 MB | v6 model folder: warm-start retrain on the expanded corpus (see Measured results) |

## The ONNX model folder

`pivo-id-1.4m/` has the same layout and contract as PiVo-1.4M, so **PiVo's own runtime
loads it unchanged**:

```
pivo-id-1.4m/
  onnx/encoder.onnx        tokens, voice, lengths -> encoded, log_durations
  onnx/decoder.onnx        streaming decoder (attention cache), PiVo cache names
  onnx/conditioning.onnx   copied unchanged from PiVo-1.4M (vocoder not retrained)
  onnx/vocoder_step.onnx   copied unchanged from PiVo-1.4M
  onnx/metadata.json       symbols, voices, config, parity numbers
  model_manifest.json      sha256 + bytes of the five runtime files
  LICENSE, NOTICE, THIRD_PARTY.md, LICENSES/
```

Verified two ways:

1. **PiVo's reference runtime** (`pivo.engine.PiVo`) passes its manifest verification and
   synthesises English with both voices.
2. **PyTorch↔ONNX parity of the exported graphs** (the exporter refuses to write output
   above tolerance): encoder `2.44e-06`, log-durations `1.19e-06`, streaming decoder
   `7.20e-05`, all under `2e-04`. Chunk sizes 1, 16 and 47, both voices, Indonesian and
   English sequences.

For Indonesian text, PiVo's English CMU dictionary is not enough: feed tokens from
`src_id/frontend.py` to the *same* ONNX graphs. `experiments/verify_onnx_release.py`
does exactly that end-to-end (text → ONNX encoder/decoder → FARGAN vocoder → WAV → CTC
read-back) and is the release gate.

```bash
# English through PiVo's own runtime
python -I -c "import sys; sys.path.insert(0,'<PiVo-1.4M>');
from pivo.engine import PiVo; a=PiVo('pivo-id-1.4m').synthesize('Hello there.', voice='Pi')"

# Indonesian through the same folder + this repo's frontend
python -I experiments/verify_onnx_release.py --model pivo-id-1.4m --lang id --voice-id 0 \
  --text "Selamat pagi, apa kabar?" --out out.wav
```

## Measured results

All numbers below were produced on this machine (RTX 5060 Ti, PyTorch 2.11.0+cu128,
CUDA). Full detail: [`research/training-run-report.md`](research/training-run-report.md).

### Indonesian intelligibility (CER, held-out val rows, Indonesian CTC)

Lower is better. Teacher = the Supertonic WAVs themselves through the vocoder (an ASR
floor, not a quality ceiling).

| Checkpoint | rows | CER | voice 0 | voice 1 |
|---|---:|---:|---:|---:|
| Teacher (Supertonic F1/M1 audio) | 40 | 0.1398 | 0.1253 | 0.1542 |
| v4 — Indonesian only | 40 | 0.2482 | 0.2338 | 0.2626 |
| **v5c — bilingual (final)** | 40 | **0.2636** | 0.2506 | 0.2766 |

Long domain sentences (>60 chars) are much closer to teacher than short foreign
phrases and numbers: v4 0.136 vs teacher 0.111; v5c 0.148 vs 0.111. Short phrases
(≤10 chars: "OSS", "Naashop", "prabayar") are the weak spot for both student runs.

#### v6 — expanded corpus retrain (demo-word coverage)

v5c's weakness was unseen demo vocabulary ("Indonesia", "Inggris", "megabita",
"vocoder", "GitHub", "pagi", "koma", "ringan", clock forms like `08:00`).
v6 retrains from the PiVo original on a 1,857-row merged corpus: 1,569 new
Indonesian rows (424 unique texts × 2 voices × 2 seeds, 127 rows rejected by a
95% CTC read-back gate and duration checks) + 288 English rehearsal rows.
9000 steps, batch 8, AdamW 1e-4 → 1e-5, delta-weight 0.15 (frame-to-frame
smoothing loss), RTX 5060 Ti. Final val: feature 0.2016, duration 0.0923.

Demo + held-out sentences (`experiments/eval/v6_listen/`, ONNX model
`experiments/models/pivo-id-v6`, parity encoder 1.88e-06, log-durations
2.15e-06, decoder 7.34e-05):

| Sentence | CER |
|---|---:|
| "Selamat pagi, apa kabar?" | **0.000** |
| "Ada sesuatu hari ini, semuanya jelas dan halus." | **0.000** |
| "Waktu sekarang pukul 08:15 pagi." | **0.000** |
| "PiVo adalah model suara dua bahasa, Indonesia dan Inggris." | 0.021 |
| "Selamat pagi, apa kabar? Saya baik, terima kasih." | 0.026 |
| "Model akustik dan vocoder dalam satu paket..." | 0.065 |
| "Mau suara dua bahasa tapi ukuran modelnya gede banget?" | 0.089 |
| "PiVo, bahasa Indonesia dan Inggris. Reponya ada di GitHub!" | 0.149 |
| "Hello there, ready to run on edge!" (English, M1) | 0.154 |
| "Buka pukul 08:00 dan tutup pukul 20:00." | 0.163 |
| "Kenalin PiVo. Satu koma empat juta, cuma lima koma enam megabita!" | 0.269 |

Weakest: spoken numbers ("lima koma enam megabita" heard as "lima konga
kenem noumezewita") and clock digits ("nol" dropped in `h2`). Every previously
unseen demo word is now read back correctly.

#### v6-cont — 1,000 more steps (negative result)

Resuming from v6 `last.pt` for +1,000 steps (batch 32, lr 3e-5 → 3e-6,
delta-weight 0.15, same corpus) lowered train feature to 0.170 but **raised**
val feature 0.2016 → 0.2030 at step 9,500 and 10,000 — mild overfitting.
CER on the same 11 sentences (`experiments/eval/v6cont_listen/`) was mixed:
`h3` improved 0.021 → 0.000, but `scene1` 0.089 → 0.133, `scene2` 0.269 →
0.288, `h2` 0.163 → 0.233 worsened. Verdict: keep **v6 at 9,000 steps**;
extra steps do not help at 1.4M parameters. The fixed `--resume` shell-shape
bug (vocab taken from checkpoint config, not the dataset) is in `src_id/train.py`.

### English retention (CER, 16 held-out English rows, English CTC)

| Checkpoint | CER | Note |
|---|---:|---|
| PiVo original | 0.0095 | baseline |
| Teacher Supertonic `en` WAVs | 0.0080 | ASR floor |
| v4 — Indonesian only | **0.2916** | English breaks without rehearsal |
| v5 — bilingual, 2 EN seeds | 0.0398 | English restored |
| **v5c — bilingual, 3 EN seeds (final)** | **0.0335** | best of the bilingual runs |

English WER with `openai/whisper-small.en` (8 sentences, same protocol as PiVo's own
evaluation) agrees: PiVo original 0.0556, v4 0.0833/0.1111, v5c 0.0417 (voice 0) and
0.0694 (voice 1).

### Model size

| Asset | Parameters | Size |
|---|---:|---:|
| Acoustic (fine-tuned, same architecture) | 584,965 | 2.26 MB fp32 |
| FARGAN vocoder (unchanged) | 816,597 | 3.12 MB |
| Total inference | 1,401,562 | ≈ 5.4 MB |
| Training checkpoint `best.pt` | — | 6.8 MB (includes AdamW state; training only) |

### Training recipe

Warm-start from `PiVo-1.4M/checkpoints/acoustic.pt`, `--reset-feature-stats`
(overwrite PiVo's English feature mean/std with Indonesian train-split statistics),
AdamW lr 1e-4 → 1e-5 cosine, warmup 200, batch 8, grad clip 1.0, seed 42, 6000 steps,
loss = masked SmoothL1 on normalized features + 0.1·MSE(log duration)
(+ optional `--delta-weight`, a frame-to-frame difference SmoothL1 that
suppresses temporal jitter).

Corpus v5c: 719 rows / 263,054 frames (2,630 s) — 431 Indonesian rows from corpus v4
plus 288 English rows (48 sentences × 2 voices × 3 seeds) synthesised by Supertonic
`lang=en`, aligned with `facebook/wav2vec2-base-960h` and PiVo's CMU frontend.

```bash
python -I src_id/train.py --data experiments/data/pilot_v5c/manifest.jsonl \
  --out experiments/runs/pilot_v5c --pivo-ckpt PiVo-1.4M/checkpoints/acoustic.pt \
  --reset-feature-stats --steps 6000 --batch-size 8 --lr 1e-4 --min-lr 1e-5 \
  --warmup 200 --seed 42
```

Corpus v6: 1,857 rows / 551,260 frames (5,513 s) — 1,569 new Indonesian rows
(424 texts × 2 voices × 2 seeds, parallel builder
`experiments/build_pilot_corpus_fast.py`) + 288 English rows, merged with
`experiments/merge_corpora.py`. Same recipe at 9,000 steps, `--delta-weight 0.15`:

## Honest limitations

- **Everything is synthetic, single-teacher.** The student inherits Supertonic's
  mispronunciations; numbers/prices are still wrong in both the student and the teacher.
- **Small eval samples.** 40 Indonesian rows and 16 English rows; differences of ~0.02
  between neighbouring runs are within noise. No human MOS / blind listening was done.
  CER measures intelligibility, not naturalness.
- **Bilingual trade-off.** Adding English rehearsal costs a little Indonesian CER
  (v4 0.2482 → v5c 0.2636 on the same 40 rows; 18 rows worse, 14 better). More English
  data helped both languages here (v5 → v5c), but this is one seed and one budget.
- **ONNX parity is PiVo-runtime parity, not perceptual equality.** PiVo's original
  PyTorch↔ONNX comparison (`research/pivo-parity.md`) still reports cache differences
  up to `1.46e-05`; this export's own encoder/decoder parity is `7.2e-05` max.
- **Language switching is per-utterance.** English text goes through PiVo's CMU frontend;
  Indonesian through `src_id/frontend.py`. Code-switching inside one sentence is not
  handled by either frontend.
- **Licensing.** PiVo code/weights are Apache-2.0; the FARGAN vocoder carries Opus
  BSD-3-Clause notices; the English frontend uses `cmudict.dict` (CMU notice). Training
  audio is Supertonic-3 output, whose license (BigScience OpenRAIL-M) treats models
  trained on its synthetic output as potential **derivatives**. Do not redistribute this
  model without reviewing [`research/teacher-audit.md`](research/teacher-audit.md) and
  carrying the included license/notice files.

## Reproducing

1. Obtain PiVo-1.4M (Hugging Face) and Supertonic v3 locally; install
   `torch`, `onnxruntime`, `transformers`, `numpy`.
2. Download the CTC models used: `indonesian-nlp/wav2vec2-indonesian-javanese-sundanese`
   (alignment + Indonesian CER), `facebook/wav2vec2-base-960h` (English alignment/CER),
   `openai/whisper-small.en` (English WER).
3. Build the corpora: `experiments/build_pilot_corpus.py` (`--lang id` / `--lang en`)
   or the parallel `experiments/build_pilot_corpus_fast.py`, merge with
   `experiments/merge_corpora.py`, audit with `experiments/audit_corpus.py`.
4. Train with `src_id/train.py`, evaluate with `experiments/eval_cer.py`
   (`--lang id|en`) and `experiments/eval_english.py`, export and gate with
   `experiments/export_onnx.py` + `experiments/verify_onnx_release.py`.

Paths in the scripts default to this machine's layout
(`E:/test/tts-test/pivo1-4m/models/hf/...`); pass `--aligner-dir`/`--model-dir`
explicitly elsewhere.

## Credits

PiVo-1.4M by empero-ai (Apache-2.0). FARGAN vocoder from Xiph/Opus. English frontend
from `PiVo-1.4M/pivo/text.py` + CMU dictionary. Indonesian frontend and the
alignment/training pipeline in this repo were written for this project. Multilingual
reference: [Manojvoxal/pivo-v1-multilingual](https://github.com/Manojvoxal/pivo-v1-multilingual).
