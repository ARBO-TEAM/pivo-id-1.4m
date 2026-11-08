# PiVo Bahasa Indonesia Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to execute task-by-task. Checkboxes track work; never bypass a STOP gate.

**Goal:** Menghasilkan satu suara narasi bahasa Indonesia yang natural, dapat dimengerti, dan sekecil mungkin dari student PiVo, menggunakan audio Supertonic sebagai satu-satunya sumber audio pelatihan.

**Architecture:** Evaluasi dan verifikasi lisensi teacher terlebih dahulu; setelah lulus, sintesis korpus satu suara dengan teks Indonesia yang telah dipisah sebelum generasi. Rekonstruksi model akustik PiVo dan buktikan kesetaraan dengan ONNX asli sebelum adaptasi frontend, alignment, ekstraksi fitur, dan fine-tuning; vocoder FARGAN asli dipertahankan terlebih dahulu. Bandingkan student dengan teacher melalui listening test dan evaluasi terukur, kemudian ekspor hanya model yang lolos.

**Tech Stack:** Python terisolasi, PyTorch, ONNX/ONNX Runtime, NumPy, audio WAV PCM, tool alignment yang kompatibel dengan bahasa Indonesia dan lisensinya, NVIDIA GPU bila kompatibel. Versi library dibekukan setelah uji kompatibilitas, bukan diasumsikan tersedia.

## Batasan global dan prinsip

- Target: **satu suara**, narasi umum, audio training **hanya** berasal dari Supertonic; penilaian oleh manusia diperbolehkan, tetapi rekaman manusia bukan data training.
- Kualitas dan pelafalan Indonesia lebih penting daripada mempertahankan kemampuan Inggris. Target awal adalah ukuran asli 1.401.562 parameter termasuk vocoder; model yang lebih besar hanya dicoba jika ada bukti peningkatan kualitas.
- GPU boleh dipakai untuk inferensi. Mesin terdeteksi RTX 5060 Ti, 16.311 MiB; kompatibilitas driver/PyTorch/CUDA serta kecepatan belum diverifikasi.
- PiVo menghasilkan audio mono 16 kHz; keterbatasan bandwidth dan tekstur FARGAN tidak boleh disebut setara audio full-band.
- Jangan menyebut supervised training pada audio teacher sebagai distilasi fitur/internal teacher: metode awal ini adalah **pelatihan pada data sintetis**. Distilasi internal hanya mungkin jika antarmuka teacher benar-benar tersedia dan terbukti berguna.
- Dataset, output audio, checkpoints, direktori hasil dan informasi privat tidak diunggah. Jangan mengubah repo Supertonic sumber.
- Tambahan user: `docs/superpowers/plans/requestkalimat.md` menjadi sumber domain TTS; pilih contoh positif (terutama §5 dan respons relevan), sisihkan evaluasi sebelum sintesis, jangan pakai contoh negatif/khusus ASR/placeholder sebagai target ujaran. Harga dan aturan percakapan berasal dari dokumen dan dapat basi; model TTS tidak menjadi sumber fakta atau penjamin perilaku asisten. Pastikan UNIKMA, Naashop, cireng, KBLI, NIB, Rp, kVA dan kata Inggris domain diuji pelafalannya.
- File model eksternal adalah input tidak tepercaya: jangan menjalankan skrip dari direktori unduhan; jalankan skrip sendiri dari direktori terpisah dan Python `-I`. `.pt` berbasis pickle: gunakan `torch.load(..., weights_only=True, map_location='cpu')`; jangan gunakan `weights_only=False` terhadap checkpoint yang belum dipercaya. Jika format tidak kompatibel, STOP dan audit sebelum memuat.
- Jangan membuat commit atau push kecuali diminta; repo clone tetap bersih. Produk kerja lokal diletakkan di luar direktori clone.

## Dasar temuan yang sudah diverifikasi

- Repo `PiVo-1.4M/` diklon dari https://huggingface.co/empero-ai/PiVo-1.4M pada revisi `515d84fc3420da6ae138f1b81017fcbc24d622a8`; Git LFS aktif, hash `acoustic.pt` dan `vocoder.pt` cocok dengan model card. `hg` tidak terpasang; Hugging Face menyediakan repositori Git, sehingga clone menggunakan Git.
- `PiVo-1.4M/README.md`: akustik 584.965 parameter + vocoder 816.597; bahasa Inggris; ONNX untuk inferensi; FARGAN 16 kHz. Checkpoint PyTorch ada, tetapi definisi training dan trainer tidak disertakan dalam rilis ini.
- `PiVo-1.4M/onnx/metadata.json`, `pivo/text.py`, `pivo/phase.py`, `pivo/runtime.py`, `pivo/audio.py`: sumber kebenaran untuk konfigurasi, kosakata, timing, state, dan konvensi fitur. Jangan mengganti konvensi fitur 20 dimensi dengan mel-spectrogram biasa.
- `E:/test/tts-test/split-supertonic-v3/README.md`: kompresi Supertonic eksperimental; tiap versi dilaporkan menghasilkan WAV Indonesia singkat, **belum membuktikan kualitas/kelengkapan**. Sepuluh voice style F1–F5 dan M1–M5 dan aset untuk konfigurasi inferensi referensi ada secara lokal; operasional dan kualitas masing-masing belum diuji. Konfigurasi teacher menyebut 44.100 Hz dan harus di-resample ke 16 kHz untuk student.
- Model card upstream [Supertonic-3](https://huggingface.co/Supertone/supertonic-3) secara eksplisit mencantumkan `id` (Indonesian), namun kualitas versi lokal belum diuji. [Lisensi upstream](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE) §1(e) secara eksplisit memasukkan model yang dilatih pada data sintetis ke definisi derivative, dan §5 memberlakukan pembatasan penggunaan. README proyek lokal menyatakan lisensi kode proyeknya belum dipilih. Periksa kewajiban distribusi, versi aset aktual, dan tujuan penggunaan sebelum training/distribusi; jangan menyimpulkan derivative bebas dari pembatasan.
- Klaim listening dan WER Inggris PiVo asli tidak membuktikan kualitas Indonesia. Teacher, student dan perangkat belum diuji menghasilkan audio dalam sesi perencanaan ini.

## Peta lokasi kerja

Lokasi di bawah adalah keluaran yang akan dibuat hanya jika gerbang sebelumnya lulus. Tidak memodifikasi source clone.

| Lokasi | Tanggung jawab |
| --- | --- |
| `research/teacher-audit.md` | bukti lisensi, inventaris, parameter model dan hasil listening teacher |
| `research/pivo-parity.md` | bukti state dict, operasi graf, toleransi dan hasil parity |
| `experiments/text/` | sumber teks yang diizinkan, manifest pembagian dan daftar uji yang dibekukan |
| `experiments/data/` | WAV sintetis, metadata, alignment, fitur dan laporan QA; tidak dimasukkan ke Git |
| `experiments/runs/` | konfigurasi dan checkpoint eksperimen; tidak dimasukkan ke Git |
| `experiments/eval/` | audio buta, skor agregat, error analisis dan benchmark |
| `src_id/frontend.py` | normalisasi teks dan konversi teks ke fonem ID dengan aturan eksplisit |
| `src_id/acoustic.py` | implementasi PyTorch akustik yang cocok dengan checkpoint dan ONNX |
| `src_id/data.py` | manifest, fitur FARGAN, alignment dan batching |
| `src_id/train.py` | pelatihan dan checkpoint reproducible |
| `src_id/export.py` | ekspor dan verifikasi ONNX |
| `tests/` | self-check deterministik untuk frontend, fitur, alignment, parity, training dan ekspor |

Nama file di atas adalah batas tanggung jawab, bukan alasan membuat scaffolding kosong. Lakukan implementasi minimal setelah bukti tahap sebelumnya tersedia.

## Tahap 0 — Kelayakan teacher, hak penggunaan, lingkungan

### Task 0.1: Inventaris dan hak penggunaan

**Files:** `research/teacher-audit.md`; baca `E:/test/tts-test/split-supertonic-v3/README.md`, lisensi upstream, kartu model dan ketentuan model/voice style persis yang dipakai.

- [ ] Catat path, versi/revisi, checksum setiap komponen teacher, voice style, contoh inferensi, konfigurasi, dan apakah file betul-betul tersedia lokal. Bedakan hasil observasi dari yang hanya tercantum di README; tidak mengasumsikan akses ke repo privat.
- [ ] Periksa teks lisensi model, batas penggunaan, lisensi kode wrapper, hak penggunaan output untuk training student, distribusi hasil, tujuan penggunaan dan atribusi. Catat URL, revisi/tanggal serta kutipan klausul relevan. Bila tidak jelas minta klarifikasi pemegang hak; jangan menganggap tidak adanya pernyataan di README sebagai pelarangan eksplisit.
- [ ] **STOP** generasi dataset/training atau distribusi jika izin untuk penggunaan yang direncanakan tidak dapat dibuktikan. Evaluasi pribadi hanya dilakukan jika diizinkan oleh ketentuan yang berlaku.
- [ ] Catat versi OS, GPU, driver, ruang disk, Python, ONNX Runtime dan PyTorch; jalankan pemeriksaan kompatibilitas dari direktori kerja tepercaya tanpa mengeksekusi skrip repo unduhan. Simpan hasil dan error; jangan mengklaim waktu training sebelum mengukur.

**Lulus:** inventaris lengkap dan dasar izin terdokumentasi untuk penggunaan yang dipilih. Jika tidak, laporkan penyebab dan berhenti; user telah membatasi audio ke Supertonic saja.

### Task 0.2: Seleksi suara dan gate kualitas teacher

**Files:** `research/teacher-audit.md`, `experiments/eval/teacher/`; input daftar 60 kalimat tetap dalam `experiments/text/teacher_probe.txt`.

- [ ] Susun 60 kalimat unik dan berizin: 15 berita/informasi, 10 kalimat panjang, 10 angka/tanggal/mata uang, 10 nama/tempat, 10 tanda baca/pertanyaan, 5 campuran kata Inggris. Pastikan tidak mengandung data pribadi. Versi teks dan hash dicatat.
- [ ] Generate set dengan semua voice style yang tersedia menggunakan parameter reproducible; catat kode bahasa aktual, model/varian, seed jika didukung, speed, jumlah langkah, rate dan format WAV. Sertakan referensi FP32 bila tersedia dan diizinkan; jangan mengasumsikan quantized lebih baik atau setara.
- [ ] Dua atau lebih penutur fasih Indonesia menilai secara buta 1–5 untuk naturalness dan pelafalan; tandai angka, kata hilang, kata tambahan, jeda dan artefak. Acak urutan dengan seed; simpan per-kalimat dan agregat, bukan hanya rata-rata.
- [ ] **Gate awal operasional:** pilih satu suara bila median naturalness dan pelafalan masing-masing ≥4/5, sedikitnya 90% kalimat tanpa kesalahan pembacaan material berdasarkan dua penilai, dan tidak ada kegagalan sistematis angka/tanggal. Ini target proyek, bukan jaminan atau standar industri. Tinjau ketidaksetujuan dengan mendengar ulang.
- [ ] **STOP** bila tidak ada suara yang lulus. Jangan membuat dataset skala besar atau menjanjikan bahwa student akan memperbaiki kesalahan bahasa teacher; user memilih Supertonic saja.

## Tahap 1 — Parity arsitektur PiVo sebelum fine-tuning

### Task 1.1: Model akustik dan uji ekuivalensi

**Files:** `src_id/acoustic.py`, `tests/test_acoustic_parity.py`, `research/pivo-parity.md`; referensi `PiVo-1.4M/onnx/{metadata.json,encoder.onnx,decoder.onnx}`, `checkpoints/acoustic.pt`, `pivo/runtime.py`, `pivo/phase.py`.

- [ ] Audit state dict secara aman dan catat nama/shape tensor, buffers, default config, input/output ONNX, mask, RoPE, normalisasi, cache, durasi, konvensi phase serta urutan operasi. Periksa graf ONNX dan hasil pengujian, jangan menyimpulkan fungsi hanya dari nama tensor.
- [ ] Implementasikan hanya arsitektur yang dibutuhkan untuk memuat seluruh tensor akustik secara `strict=True`; tidak ada tensor hilang/tambahan. Dekoder diuji secara step-by-step dengan state awal nol dan cache yang benar. Implementasi model adalah rekonstruksi karena source train asli tidak tersedia.
- [ ] Gunakan seed tetap dan kasus panjang token/frame 1, pendek dan panjang, plus kedua voice ID. Bandingkan PyTorch `eval()` dengan ONNX untuk durasi/logit encoder, keluaran fitur, cache dan state yang bersesuaian; gunakan `rtol=1e-4`, `atol=1e-5` untuk FP32 CPU, catat max/mean absolute error dan sumber perbedaan. Terapkan mask dan preprocessing identik.
- [ ] **STOP** bila `strict=True` gagal atau parity melebihi toleransi. Jangan melanjutkan dengan model yang hanya kebetulan memiliki ukuran tensor sama; debug layer pertama yang berbeda. Toleransi hanya dapat diubah dengan bukti kuantisasi atau numerik yang ditulis jelas, bukan untuk menyembunyikan bug.
- [ ] Simpan baseline output dan audio Inggris PiVo asli untuk regresi teknis; kualitas Inggris setelah adaptasi bukan target utama.

### Task 1.2: Frontend bahasa Indonesia

**Files:** `src_id/frontend.py`, `tests/test_frontend.py`, `experiments/text/normalization_cases.jsonl`.

- [ ] Tetapkan inventaris fonem Indonesia dan pemetaan deterministik ke maksimum 128 indeks embedding yang tersedia. Pertahankan special token; gunakan kembali embedding untuk fonem yang benar-benar ekuivalen, inisialisasi simbol lain secara terkontrol. Catat bahwa mempertahankan indeks lama yang berbeda makna adalah kesalahan.
- [ ] Terapkan normalisasi angka, tanggal, desimal, nominal, singkatan umum, tanda baca dan kata serapan; definisikan kebijakan fallback untuk nama/OOV, code-switching dan simbol yang tidak dikenal. Jangan diam-diam membuang karakter atau mengganti dengan fonem tidak berhubungan.
- [ ] Sediakan tabel kasus dengan input dan keluaran yang ditinjau penutur: `17 Agustus 1945`, `Rp12.500`, `3,5`, `Jakarta`, `ng`, `ny`, `kh`, `sy`, `?`, `!` dan teks kosong. Uji determinisme, stabilitas ID token, batas panjang, karakter tak didukung dan setiap aturan normalisasi.
- [ ] Karena ambiguitas `e`/schwa dan nama tidak selalu dapat diselesaikan dengan aturan, dukung override leksikon tervalidasi; tandai kata ambigu dalam laporan, jangan mengklaim G2P sempurna.

## Tahap 2 — Dataset sintetis yang terukur dan bebas kebocoran

### Task 2.1: Teks, manifest, dan pilot teacher

**Files:** `experiments/text/{train,validation,test}.jsonl`, `experiments/data/manifest.jsonl`, `src_id/data.py`, `tests/test_data.py`.

- [ ] Kumpulkan teks yang hak penggunaan/privasi memungkinkan, normalisasi dan deduplikasi exact serta near-duplicate. Teks evaluasi teacher gate tidak masuk training. Hindari formula berulang, nama nyata sensitif, dan materi tanpa hak.
- [ ] Pisahkan berdasarkan sumber/topik dan kemiripan sebelum generasi; target 90/5/5 berdasarkan teks unik. Test dibekukan dan tidak dipakai memilih checkpoint; validation untuk pemilihan. Catat seed, hash manifest, panjang dan cakupan fenomena.
- [ ] Manifest setiap baris memuat `id`, `split`, `text_raw`, `text_normalized`, `phonemes`, `wav_path`, `sample_rate`, `duration_seconds`, `teacher_revision`, `voice_id`, `generation_config`, `audio_sha256`, `status`, `rejection_reason`. Setiap ID unik; WAV hasil satu kalimat tidak boleh muncul di lebih dari satu split.
- [ ] Mulai pilot 100 kalimat training + 30 validasi; lanjutkan ke target awal 5–10 jam audio training **hanya jika gate kualitas, alignment dan overfit lulus**. Ini anggaran eksperimen, bukan angka kualitas yang terjamin. Test terpisah ≥100 kalimat yang mencakup semua kategori teacher gate.
- [ ] Hasilkan file sementara kemudian rename atomik setelah sukses; rekam input/config/error agar restart tidak menduplikasi atau salah mengaitkan teks dan audio. Audio PCM mono, sample rate asli dicatat; resampling deterministik ke 16 kHz untuk FARGAN, tanpa clipping atau normalisasi destruktif.
- [ ] QA durasi, silence di awal/akhir, clipping, file rusak, sampel duplikat, speech rate ekstrem; sampling manual dan ASR Indonesia untuk menandai kemungkinan salah baca. ASR tidak dijadikan ground truth; audio yang salah atau diragukan ditolak atau dikoreksi, tidak otomatis diloloskan oleh skor.
- [ ] Uji otomatis manifest: ID duplikat, file hilang, hash salah, kebocoran split dan data gagal QA harus menghentikan training.

### Task 2.2: Fitur FARGAN dan alignment durasi

**Files:** `src_id/data.py`, `tests/test_features_alignment.py`, `research/pivo-parity.md`.

- [ ] Ikuti implementasi dan revisi Opus/FARGAN yang cocok dengan vocoder PiVo untuk mengekstrak 18 cepstral + pitch + pitch-correlation setiap 10 ms; catat pre-emphasis, pitch scaling, frame hop 160, offset/delay dan satuan. Jangan mengganti fitur dengan mel. Periksa lisensi tool dan catat versi/build.
- [ ] Lakukan resynthesis teacher WAV → fitur → vocoder asli pada ≥30 kalimat beragam. Dengarkan berdampingan dengan WAV asli; ukur durasi dan keterpahaman. Ini memisahkan batas vocoder dari kesalahan acoustic model. **STOP** bila output rusak, pitch/delay salah, atau pelafalan memburuk secara sistematis.
- [ ] Pilih forced aligner yang secara eksplisit mendukung bahasa/telepon ID dan memiliki lisensi yang sesuai; dokumentasikan model, dictionary, versi, dan peta fonemnya. Gunakan teks yang benar-benar sesuai dengan audio teacher; tandai audio yang salah baca dan buang, bukan memaksa alignment.
- [ ] Ubah timestamp alignment ke durasi integer frame 10 ms menggunakan aturan rounding deterministik; jumlah frame per utterance wajib sama dengan jumlah frame fitur setelah memperhitungkan offset terdokumentasi. Tangani fonem berdurasi nol secara konsisten, cek monotonisitas, batas, leading/trailing silence dan mismatch. Tolak contoh di luar toleransi yang ditetapkan berdasarkan audit pilot.
- [ ] Uji kasus frame boundary, phoneme tanpa durasi, total frame salah, timestamp terbalik, input kosong, dan audio pendek. Audit manual minimal 30 alignment, termasuk angka dan kata multi-suku kata; gate awal ≥95% batas kata/fonem masuk akal, tanpa systematic offset. Jika aligner ID tidak layak, STOP dan bandingkan alternatif; jangan melatih durasi dari label yang dibuat-buat.

## Tahap 3 — Training dan pemilihan model

### Task 3.1: Smoke training dan eksperimen terkendali

**Files:** `src_id/train.py`, `tests/test_training_smoke.py`, `experiments/runs/`.

- [ ] Muat checkpoint akustik melalui implementasi parity; bekukan vocoder. Latih prediksi fitur dan durasi menggunakan label hasil Tahap 2; tentukan loss awal dan bobotnya dari skala/normalisasi fitur dengan merekam setiap komponen, bukan mengklaim mengetahui loss PiVo asli. Bandingkan baseline validation sebelum/sesudah.
- [ ] Mulai dengan optimizer AdamW, seed 42, FP32, batch kecil yang muat GPU, gradient clipping 1.0 dan learning rate awal `1e-4`; ini hipotesis eksperimen awal, bukan hiperparameter original. Catat nilai aktual, parameter trainable, durasi, batch, driver dan penggunaan VRAM. Hentikan bila loss tidak finite.
- [ ] Smoke test menyimpan serta memuat ulang checkpoint dengan optimizer, scheduler, seed/RNG, vocab dan config; cek prediksi cocok setelah reload dan resume melanjutkan step, tidak memulai ulang diam-diam. Simpan checkpoint melalui file sementara lalu rename atomik.
- [ ] Overfit 8 contoh selama maksimal 1.000 step; evaluasi loss, alignment dan audio, bukan loss saja. Target loss fitur dan durasi turun konsisten, ucapan dapat dikenali dan durasi tidak runtuh. Jika tidak, STOP dan periksa teacher labels, normalisasi fitur, frame alignment, gradients dan parity.
- [ ] Bandingkan paling sedikit: A) pretrained acoustic + frontend ID, B) arsitektur sama dengan acoustic baru/random init. Gunakan data, kondisi evaluasi dan anggaran step sama; baseline B tidak menggantikan teacher audio. Simpan seed dan configs; catat trade-off kualitas, waktu dan ukuran.
- [ ] Jika kualitas belum memadai, perbaiki masalah yang terukur terlebih dahulu (salah label, coverage, duration, vocoder). Eksperimen model lebih besar hanya setelah A/B terdokumentasi; catat tambahan parameter, VRAM, ukuran dan manfaat nyata. Jangan fine-tune vocoder sebelum resynthesis membuktikan vocoder sebagai bottleneck.
- [ ] Gunakan validation untuk early stopping dan pemilihan checkpoint. Satu kali evaluasi test setelah model dipilih; jika test digunakan untuk tuning, buat test baru yang independen sebelum membuat klaim final.

## Tahap 4 — Evaluasi kualitas dan delivery

### Task 4.1: Blind listening dan akurasi narasi

**Files:** `experiments/eval/{protocol.md,results.csv,report.md}`.

- [ ] Buat pasangan teacher/student untuk test yang sama; atur loudness dan jeda pemutaran secara konsisten tanpa mengubah file master. Randomisasi ID dengan seed; penyaji dan penilai tidak melihat identitas model. Nilai minimal dua penutur Indonesia, dengan target lima atau lebih jika tersedia; jumlah kecil dilaporkan apa adanya.
- [ ] Catat skor naturalness, pelafalan, prosodi, artefak 1–5 dan kesalahan material per kalimat; tampilkan sebaran dan interval ketidakpastian bila ukuran sampel memadai. Jangan menyebut nilai dari satu pendengar sebagai MOS representatif.
- [ ] Target rilis awal: median student ≥4/5 untuk naturalness dan pelafalan, selisih median terhadap teacher ≤0,5 poin untuk masing-masing, ≥95% kalimat tanpa salah baca material menurut penilai, dan tidak ada kegagalan sistematis kategori angka/tanggal. Semua target merupakan target internal yang harus dilaporkan termasuk kegagalan.
- [ ] Hitung CER/WER dengan ASR yang mendukung bahasa Indonesia pada teacher dan student memakai aturan normalisasi teks yang sama; catat nama/revisi model dan hasil mentah. Tinjau salah ASR secara manual; CER/WER adalah indikator keterpahaman, bukan kualitas suara dan bukan pengganti penilaian manusia.
- [ ] Pisahkan hasil berdasarkan panjang kalimat, angka, nama dan tanda baca; laporkan contoh kegagalan, rate, artefak FARGAN dan batas kemampuan. Bila target tidak lulus, jangan labeli siap produksi.

### Task 4.2: Ekspor, parity dan benchmark

**Files:** `src_id/export.py`, `tests/test_export.py`, `experiments/eval/report.md`, paket hasil terpisah.

- [ ] Ekspor encoder/dekoder beserta metadata vocab, voice, normalisasi fitur, rate, durasi dan versi format. Jangan menimpa artefak asli; buat manifest baru berisi SHA256 dan ukuran untuk semua aset yang benar-benar didistribusikan.
- [ ] Uji parity PyTorch–ONNX pada input pendek/panjang, cache awal/nonzero dan beberapa durasi; FP32 `rtol=1e-4`, `atol=1e-5` bila numerik memungkinkan. Uji end-to-end WAV valid, finite, mono 16 kHz, durasi masuk akal dan output bukan seluruhnya diam. Catat toleransi dan error aktual.
- [ ] Ukur RTF, waktu first audio jika streaming benar-benar tersedia, peak VRAM/RAM dan ukuran model pada perangkat aktual; cantumkan provider ORT, driver, hardware, jumlah run dan panjang teks. Inferensi CPU boleh dibandingkan, bukan syarat wajib; tidak mengasumsikan GPU lebih cepat.
- [ ] Sertakan lisensi dan notice PiVo, Opus/FARGAN, frontend/aligner dan dependensi, serta hak yang berlaku untuk teacher/output; jangan mendistribusikan model atau audio sampai izin dan atribusi diperiksa. Hindari menyertakan data privat.

## Keputusan dan kriteria berhenti

1. **Izin atau akses teacher tidak dapat dibuktikan:** berhenti pada audit, jangan membuat korpus.
2. **Teacher Indonesia gagal listening gate:** berhenti; dengan batas Supertonic saja, kualitas bagus tidak dapat dijanjikan.
3. **Checkpoint tidak dapat dipetakan dengan parity:** berhenti; fine-tuning dianggap belum terbukti secara teknis.
4. **Vocoder resynthesis/alignment buruk:** perbaiki pipeline atau lakukan eksperimen terbatas; jangan menskalakan dataset/training.
5. **Student tidak lolos evaluasi buta:** laporkan hasil negatif dan bottleneck; jangan menyamakan penurunan loss atau WER rendah dengan suara bagus.
6. **Kualitas lolos tetapi ukuran naik:** laporkan parameter, ukuran, latensi dan keputusan kualitas vs ukuran secara transparan.

## Hasil akhir yang diharapkan

Laporan audit teacher dan legal, reproducible manifests, model/checkpoint dan metadata jika semua gerbang lulus, hasil parity, laporan listening + CER/WER, benchmark perangkat aktual, dan panduan menjalankan inferensi sesuai izin. Tidak ada janji bahwa seluruh gerbang dapat dilewati sebelum eksperimen dilakukan.
