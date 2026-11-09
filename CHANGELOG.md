# Changelog — PiVo-ID

Releases are the ONNX zip + training checkpoint on the GitHub Releases page;
the repo holds code, docs, and listening samples.

## v1.1.0 — 2026-11-09 — PiVo-ID v11 (Indonesian-only, recommended)

- Frontend reduplikasi fix: "meraba-raba" now expands to full copies
  with a comma micro-pause ("meraba, raba") instead of letter names;
  true acronyms ("OSS-RBA") still spell letter-by-letter.
- Corpus v11: 4,386 rows (train 3,915 / val 471, 1,154,961 frames) —
  3,804 non-hyphen rows from v10 + 582 rebuilt comma-fixed reduplikasi
  rows (147 unique texts × 2 voices × 2 seeds).
- Fine-tune from v10 `best.pt`, 3,000 steps, batch 8:
  final val feature **0.1884**, duration **0.0747** (v10: 0.1904 / 0.0786).
- ONNX parity: encoder 2.21e-06, log-durations 9.54e-07, decoder 7.15e-05.
- Listening samples: `experiments/eval/v11_listen/` (rendered with
  `--selective --deemphasis 0.88`).

## v1.0.0 — 2026-11-08 — PiVo-ID v6 (bilingual id+en)

- First expanded-corpus retrain: 1,857 rows (1,569 ID + 288 EN), 9,000 steps.
- Demo-word coverage fix ("Indonesia", "megabita", clock "08:00", …).
- Final val feature 0.2016, duration 0.0923. CER 0.000 on four demo
  sentences (see README v6 table).
- Later ID-only fine-tunes v7–v11 continue from v6; the bilingual v6
  checkpoint is preserved untouched.
