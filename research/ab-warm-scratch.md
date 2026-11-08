# A/B: PiVo acoustic warm-start vs from-scratch (plan Task 3.1)

Tanggal: 2026-11-08. Data: `experiments/data/pilot_v3` (91 baris, 76 train / 15 val,
61.126 frame FARGAN, suara F1+M1 dari Supertonic v3). Semua run: 6000 step,
batch 8, AdamW lr 1e-4, warmup 200, seed 42, GPU RTX 5060 Ti, loss
SmoothL1 fitur ternormalisasi + 0.1·MSE log durasi.

| Run | Init | feature_mean/std | val_feature terbaik | CER val (12 baris) |
|---|---|---|---|---|
| `pilot_v3_long` | PiVo acoustic.pt | statistik PiVo (Inggris) | 0.2977 @1750 | 0.1805 |
| `pilot_v3_scratch` | random | statistik train split | 0.2526 @2750 | 0.3458 |
| `pilot_v3_warm_reset` | PiVo acoustic.pt | statistik train split | 0.2284 @1750 (best 0.2406 @2250) | 0.1944 |
| teacher (`acoustic.pt` → vocoder FARGAN pada WAV guru) | — | — | — | 0.1557 |

## Temuan

1. **Val loss lintas run tidak sebanding bila statistik normalisasi berbeda.**
   Run warm-start memakai buffer mean/std milik PiVo (bahasa Inggris), sehingga
   fitur Indonesia ternormalisasi tidak pada skala yang sama dengan run
   from-scratch. Karena itu 0.2977 vs 0.2526 bukan bukti scratch lebih baik;
   run `warm_reset` (bobot PiVo + statistik data Indonesia) adalah pembanding
   yang setara.
2. **CER adalah metrik keputusan utama**, karena dihitung setelah denormalisasi
   dan vocoder nyata. Scratch (0.3458) jauh lebih buruk daripada warm-start
   (0.1805) dan teacher (0.1557) pada data dan anggaran step yang sama.
3. Teacher CER 0.1557 adalah lantai ASR: WAV guru sendiri hanya terbaca
   seakurat itu oleh model CTC Indonesia; batas bawah realistis untuk student.

## Keputusan

Warm-start dipakai untuk training korpus yang lebih besar. From-scratch tidak
memberi keuntungan CER pada anggaran ini dan membutuhkan lebih banyak data
agar parity arsitektur bisa dipelajari dari nol.

Hasil korpus v4 (431 baris) dengan resep warm-start + statistik Indonesia:
best val 0.2431 @5000, CER 0.2482 pada 40 baris val (teacher 0.1398 pada baris
yang sama). Rincian per panjang teks menunjukkan gap terbesar pada frasa
pendek/angka, bukan kalimat domain panjang. Lihat `training-run-report.md`.

## Uji overfit 8 contoh (plan Task 3.1)

`experiments/data/overfit8` (8 train + 2 val dari pilot_v3, semua suara F1),
warm-start + statistik Indonesia, 1000 step: val_feature 0.340 → 0.281
(best 0.2962 pada checkpoint terpilih), train loss 0.13, tidak ada loss
non-finite, durasi tidak runtuh. Render kalimat train (`trained_overfit8_f1.wav`)
dibaca balik oleh CTC sebagai "programstudibisnisdibitalmempengajaribagaimanateknologidipakai..."
— dapat dikenali, dengan beberapa kata meleset ("mempengajari", "menggaan",
"memasakkan"), jadi bukan bukti kualitas, hanya bukti model belajar memetakan
teks → fitur → audio.

## Batasan yang harus ditulis di laporan akhir

- Corpus hanya 91 baris / ~10 menit audio; val 15 baris dan CER 12 baris.
  Angka ini indikatif, bukan klaim kualitas.
- Corpus dan audio evaluasi berasal dari satu teacher (Supertonic v3, F1/M1).
  Student mewarisi batas dan kesalahan teacher; tidak ada penilaian manusia,
  tidak ada MOS.
- Cer diukur dengan satu model ASR (wav2vec2 ID/Jawa/Sunda, Apache-2.0); salah
  baca ASR ikut terhitung (terlihat pada kata harga/angka).
