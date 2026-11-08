# Laporan eksperimen training PiVo Indonesia

Tanggal: 2026-11-08. Semua angka di bawah berasal dari run yang benar-benar
dijalankan pada mesin ini (RTX 5060 Ti, PyTorch 2.11.0+cu128, CUDA).

## Ringkasan status

| Tahap | Status | Bukti |
|---|---|---|
| Korpus pilot v3 | 91 baris valid (76 train/15 val), audit PASS | `experiments/data/pilot_v3/summary.json`, `audit_corpus.py` |
| Korpus pilot v4 | 431 baris valid (307 train/124 val), 192.009 frame = 1920 s, audit PASS | `experiments/data/pilot_v4/summary.json` |
| A/B warm-start vs from-scratch | Warm-start menang (CER 0.19–0.18 vs 0.35) | `research/ab-warm-scratch.md` |
| Overfit 8 contoh | Loss turun konsisten (0.340 → 0.281), render terbaca CTC | `research/ab-warm-scratch.md`, `experiments/eval/trained_overfit8_f1.wav` |
| Training corpus v4 | `experiments/runs/pilot_v4_warm_reset` (6000 step, best val 0.2431 @5000) | `train_log.jsonl` |
| CER vs teacher floor | v4: student 0.2482 vs teacher 0.1398 pada 40 baris yang sama | `experiments/eval/cer_*_v4_n20.json` |
| Penilaian manusia | **Belum dilakukan** | — |

## Konfigurasi training

- Warm-start dari `PiVo-1.4M/checkpoints/acoustic.pt` + `--reset-feature-stats`
  (statistik feature dari train split Indonesia, karena buffer PiVo berbahasa
  Inggris membuat normalisasi tidak cocok; lihat `ab-warm-scratch.md`).
- AdamW lr 1e-4 → 1e-5 cosine, warmup 200, batch 8, grad clip 1.0, seed 42,
  loss SmoothL1 fitur ternormalisasi + 0.1·MSE log durasi, pada CUDA.
- Checkpoint `best.pt` (val terbaik) dan `last.pt` (tiap 500 step), disimpan
  atomik; `train_log.jsonl` mencatat setiap 250 step.

## Hasil terukur

CER dihitung dengan model CTC `indonesian-nlp/wav2vec2-indonesian-javanese-sundanese`
(Apache-2.0) pada 12 baris val (6 per suara), teks yang sama untuk semua entri.

| Konfigurasi | Checkpoint | CER rata-rata | v0 (F1) | v1 (M1) |
|---|---|---|---|---|
| Teacher (WAV guru → vocoder FARGAN) | — | 0.1557 | 0.1587 | 0.1528 |
| Student pilot v3, 1200 step, warm | `runs/pilot_v3_warm/best.pt` | 0.2083 | 0.2114 | 0.2053 |
| Student pilot v3, 6000 step | `runs/pilot_v3_long/best.pt` | 0.1805 | 0.1635 | 0.1976 |
| Student pilot v3, warm + reset stats | `runs/pilot_v3_warm_reset/best.pt` | 0.1944 | 0.1832 | 0.2056 |
| Student pilot v3, from-scratch | `runs/pilot_v3_scratch/best.pt` | 0.3458 | 0.3457 | 0.3459 |
| Student pilot v4 (12 baris, subset termudah) | `runs/pilot_v4_warm_reset/best.pt` | 0.1584 | 0.1604 | 0.1565 |
| Student pilot v4 (40 baris) | `runs/pilot_v4_warm_reset/best.pt` | **0.2482** | 0.2338 | 0.2626 |
| Teacher pada 40 baris yang sama | — | **0.1398** | 0.1253 | 0.1542 |
| Student bilingual v5 (ID v4 + EN Supertonic, 40 baris) | `runs/pilot_v5_warm_reset_en/best.pt` | 0.2766 | 0.2742 | 0.2790 |

Val loss terbaik per run (tidak sebanding lintas statistik normalisasi kecuali
disebut): v3 warm 0.2977, v3 scratch 0.2526, v3 warm_reset 0.2406,
v4 warm_reset 0.2431 (komposit val_feature + 0.1·val_duration; val_feature
mentah sekitar 0.227).

### CER per panjang teks (40 baris val v4, student v4 → v5)

| Panjang teks (karakter alfanumerik) | n | Student v4 | Student v5 | Teacher |
|---|---|---|---|---|
| ≤10 (kata/frasa pendek: "OSS", "Naashop", "prabayar") | 9 | 0.477 | 0.545 | 0.184 |
| 11–30 | 2 | 0.225 | 0.325 | 0.050 |
| 31–60 | 13 | 0.231 | 0.241 | 0.159 |
| >60 | 16 | 0.136 | 0.148 | 0.111 |

Kalimat panjang (>60 karakter): gap student–teacher hanya 0.025 (v4) /
0.037 (v5). Kelemahan utama ada pada frasa pendek/asing dan angka, bukan pada
kalimat domain panjang. Contoh render: `experiments/eval/v4_f1_probe.wav`,
`v4_m1_probe.wav`, `v5_f1_probe_id.wav`, `v5_f1_probe_en.wav`.

## Ukuran model

| Aset | Parameter | Ukuran |
|---|---|---|
| Akustik PiVo asli (`checkpoints/acoustic.pt`) | 585.005 (buffer ikut terhitung) / 584.965 trainable | 2,26 MB fp32 |
| Akustik hasil fine-tune (arsitektur sama, tanpa parameter tambahan) | 585.005 | bobot 2,34 MB fp32; `best.pt` 6,81 MB karena menyertakan state optimizer AdamW untuk resume |
| Vocoder FARGAN (`vocoder.pt`) — tidak diubah | 816.597 | 3,12 MB |
| Total inferensi (akustik + vocoder) | 1.401.602 | ≈ 5,4 MB fp32 |

Tidak ada parameter atau lapisan yang ditambahkan pada fine-tune; checkpoint
hasil bisa diekspor dengan format ONNX yang sama seperti rilis asli.

## Retensi bahasa Inggris

PiVo memang sudah berbahasa Inggris; yang perlu diuji adalah apakah
fine-tuning Indonesia merusaknya. Diuji dengan `experiments/eval_english.py`
(8 kalimat Inggris, `openai/whisper-small.en` — model ASR yang juga dipakai
PiVo di `evaluation.json`), merender lewat frontend Inggris asli PiVo
(CMU dict, `--lang en`):

| Checkpoint | Suara | WER |
|---|---|---|
| PiVo asli | Pi (0) | 0,0556 |
| Fine-tune v4 | Pi (0) | 0,0833 |
| PiVo asli | Vo (1) | 0,0556 |
| Fine-tune v4 | Vo (1) | 0,1111 |
| Fine-tune v5 (bilingual rehearsal) | Pi (0) | **0,0417** |
| Fine-tune v5 (bilingual rehearsal) | Vo (1) | **0,0417** |

Inggris tetap berfungsi pada kedua suara setelah fine-tuning Indonesia; WER
naik ~0,03–0,05 pada sampel 8 kalimat (kecil, tetapi konsisten). Kemungkinan
penyebab: tidak ada data Inggris dalam korpus latihan, sehingga bobot bergeser
ke domain Indonesia. Ini dapat dikurangi dengan rehearsal Inggris pada data
sintetis (Supertonic juga mendukung `lang=en`), tetapi melanggar batas user
"audio training hanya Supertonic" bila memakai audio Inggris nyata — audio
Supertonic `en` masih memenuhi batasan itu.

### Rehearsal bilingual v5 (sudah dijalankan)

Korpus v5 = pilot_v4 (431 baris Indonesia) + 192 baris Inggris dari teacher
Supertonic `lang=en` (48 kalimat × 2 suara × 2 seed, 474 s audio), di-align
dengan CTC Inggris `facebook/wav2vec2-base-960h` dan frontend CMU PiVo.
Audit PASS; total 623 baris (467 train / 156 val).

CTC Inggris memberi sinyal retensi yang jauh lebih tajam daripada WER whisper
(16 baris val Inggris). Angka CER di bawah diukur dengan CTC Inggris yang sama
untuk semua checkpoint:

| Checkpoint | CER Inggris (16 baris) | Catatan |
|---|---|---|
| PiVo asli | 0,0095 | baseline |
| v4 (Indonesia saja) | 0,2916 | regresi besar — rehearsal memang perlu |
| **v5 (bilingual)** | **0,0398** | Inggris pulih hampir setara asli |
| Teacher Supertonic `en` (lantai ASR) | 0,0080 | batas bawah |

Di sisi Indonesia, v5 = 0,2766 vs v4 = 0,2482 pada 40 baris val yang sama:
18 baris memburuk, 13 membaik, 9 stabil. Penurunan kecil ini adalah harga dari
rehearsal; run v5b (data Inggris separuh) menguji apakah harga itu bisa ditekan.

Catatan bug: probe WAV English pertama ditulis dengan skala ganda (int16
dimasukkan ke writer float), sehingga terdengar kasar/pecah; file sudah
dirender ulang dengan benar. WER tidak terpengaruh karena dihitung dari float
sebelum penulisan WAV. Atribusi frontend Inggris: `PiVo-1.4M/pivo/resources/cmudict.dict`,
lisensi CMU (lihat `PiVo-1.4M/LICENSES/CMUDICT_LICENSE`).

## Batasan yang harus disertakan dalam klaim akhir

- Corpus seluruhnya audio sintetis dari satu teacher (Supertonic v3 F1/M1),
  termasuk 192 baris rehearsal Inggris (`lang=en`). Student mewarisi salah
  baca teacher — angka/harga masih salah pada kedua student dan teacher
  sendiri (lihat `cer_teacher_v3.json`).
- 16 baris CER Inggris dan 40 baris CER Indonesia adalah sampel kecil; selisih
  antar run yang berdekatan (mis. 0.1805 vs 0.1944, atau 0.2482 vs 0.2766)
  sebagian ada dalam rentang noise. Angka WER whisper (8 kalimat) lebih kasar
  lagi; gunakan CTC sebagai sinyal utama.
- Tidak ada penilaian manusia (MOS/blind listening). CER bukan ukuran
  naturalness.
- Lisensi: output teacher tunduk pada OpenRAIL-M Supertonic; model hasil
  berpotensi dianggap derivative. Jangan distribusikan tanpa meninjau
  `research/teacher-audit.md` dan menyertakan lisensi/notice.
- Parity PyTorch–ONNX asli masih gagal pada cache dekoder (69 perbandingan,
  `research/pivo-parity.md`); training memakai implementasi PyTorch sendiri,
  bukan jalur ONNX.
