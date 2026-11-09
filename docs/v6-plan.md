# Plan v6 — korpus diperluas, word-coverage, delta loss

Status:korpus Indonesia (pilot_v6_id) sedang disintesis di background.

## Konteks

Model v5c hanya 163 kalimat unik (115 ID + 48 EN); 22 kata demo HyperFrames tidak pernah dilihat
("Indonesia", "megabita", "vocoder", "GitHub", "pagi", "koma", "ringan", ...). Itu penyebab
utama ketidakjelasan, bukan smoothing.

## Isi v6

- `experiments/text/extra_id_v6.jsonl` — 215 kalimat teks demo + domain requestkalimat
  (baseline: L562-L1159 -> domain_manifest 185 baris, dedup -> 177).
- `experiments/text/extra_id_time_v6.jsonl` — 85 kalimat bentuk jam ("08:00").
- Gabungan dinormalisasi di `.tmp-out/v6_manifest.jsonl` (481 baris: 365 train / 59 val / 57 test; test tidak dipakai build_pilot_corpus, jadi yang dibangun = 424 teks).
- Coverage gate: semua kata demo (ID) sudah ada sebagai token-run di korpus -> lolos.
  Kata Inggris demo dicakup oleh english_manifest (E001-E048).
- Held-out: 5 kalimat demo H001-H005 disimpan di manifest generator, tidak dilatih
  (`DEMO_EVAL`), untuk eval CER adil.

## Korpus dan merge

- `pilot_v6_id` — teacher Supertonic F1/M1, seed 0+1, aligner CTC Indonesia. 424 teks x 2 suara x 2 seed = ~1.696 baris.
- Merge final: `pilot_v6_id` + `en_pilot_v1` (192) + `en_pilot_s2` (96) ->
  `pilot_v6` ~2.212 baris, Inggris 288 baris = 13% (cukup untuk rehearsal).

## Training

Satu run utama (bukan A/B dua run penuh — budget waktu): warm-start PiVo asli,
`--reset-feature-stats`, lr 1e-4 -> 1e-5 cosine, warmup 200, batch 8, grad-clip 1.0,
seed 42, steps 9000, `--delta-weight 0.15` (delta loss baru di src_id/train.py),
`--eval-every 500`, `--log-every 20`. GPU: RTX 5060 Ti (~17-18 langkah/detik -> ~9 menit).

Catatan: (a) delta_weight>0 belum ada fallback; bila val lebih buruk dibanding baseline
yang diterima (delta=0), checkpoint hasil run ini disimpan saja dan eksperimen
dilanjutkan dengan delta=0 — keputusan setelah melihat log val.
(b) Feature stats di-reset dari korpus baru sesuai resep v5c.

## Ekspor dan eval

1. `experiments/export_onnx.py --ckpt experiments/runs/pilot_v6/best.pt --pivo-dir PiVo-1.4M
   --out experiments/models/pivo-id-v6` (parity check ID+EN, 2 suara, chunk 1/16/47).
2. Render paket dengar HyperFrames dengan teks demo asli (render_narration.py) dan eval
   CER untuk ID + EN. Bandingkan dengan hasil v5c (scene1 CER 0.178 asli).
3. Bila lolos, putuskan apakah asset HyperFrames diganti (narasi tetap teacher; yang
   diganti hanya kartu demo ONNX bila diinginkan) dan rilis ke GitHub Release.

## Risiko

- Teacher synth 481 teks x 4 = ~1.924 render; ~30-45 menit (background).
- English rehearsal 13% dari korpus; CER Inggris dipantau saat eval.
- Delta-loss hanya intervensi kecil; seharusnya tidak mengubah karakter suara jauh.
