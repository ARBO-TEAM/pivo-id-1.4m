# Audit awal Supertonic sebagai teacher Indonesia

Tanggal: 2026-11-08. Audit ini bukan nasihat hukum dan belum menyatakan proyek siap training.

## Bahasa dan kualitas

- [Model card Supertonic-3](https://huggingface.co/Supertone/supertonic-3/blob/main/README.md) secara eksplisit mencantumkan `id` = Indonesian dalam supported languages. Ini berlaku untuk model upstream; kualitas dan perilaku build lokal tetap harus diuji.
- `E:/test/tts-test/split-supertonic-v3/README.md` melaporkan setiap varian menghasilkan WAV Indonesia singkat, tetapi secara eksplisit mengatakan bahwa itu tidak membuktikan kualitas atau kelengkapan.
- Sepuluh file voice style `F1.json`–`F5.json` dan `M1.json`–`M5.json` ada di `supertonic_work/voice_styles/`. Aset lokal yang cocok dengan konfigurasi referensi upstream juga ada: encoder, duration predictor dan vocoder FP32 TFLite serta `vector_estimator.onnx`; kombinasi INT4 encoder/duration dan INT8 vocoder juga tersedia. Keberadaan file dan konfigurasi loader bukan bukti inferensi berhasil pada mesin ini.
- `supertonic_work/tts.json:175` dan `supertonic_work/upstream_reference/inference.py:46` menetapkan sample rate teacher 44.100 Hz. Student FARGAN menggunakan 16.000 Hz sehingga perlu resampling terkontrol.
- Audio terdahulu termasuk `output/test_indonesia.wav`, `output/indonesia_original_fp32.wav` dan `output/indonesia_original_int8.wav` tersedia. Hasil tersebut belum didengarkan atau diverifikasi kecocokan teks, parameter dan voice-nya dalam sesi ini.
- Tidak ada listening test penutur Indonesia yang dilakukan dalam sesi ini; kualitas, kemampuan membaca angka, prosodi, dan suara terbaik belum terbukti.

## Lisensi dan kewajiban

- [Lisensi upstream](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE), BigScience Open RAIL-M, §1(e), secara eksplisit mendefinisikan model lain yang dibuat menggunakan output atau data sintetis untuk menirukan model sebagai **Derivatives of the Model**. Jadi student PiVo yang dilatih pada output teacher harus diperlakukan sebagai kemungkinan derivative di bawah ketentuan tersebut; tidak otomatis hanya mengikuti Apache-2.0 PiVo.
- §5 dan Attachment A memuat pembatasan penggunaan bagi model dan derivative. Perlu periksa juga kewajiban distribusi dan notice berdasarkan teks lisensi lengkap; tujuan penggunaan dan syarat model serta kode versi lokal harus dicocokkan. Audit ini tidak memutuskan implikasi hukumnya.
- `E:/test/tts-test/split-supertonic-v3/README.md:117-119` menyebut lisensi kode proyek kompresi lokal belum dipilih. Ini tidak otomatis membatasi semua output, tetapi penggunaan dan distribusi kode tersebut perlu dasar izin yang jelas. Komponen model, kode dan data harus ditinjau terpisah.
- §2 memberikan lisensi untuk membuat derivative, dengan syarat lisensi; §5 menyatakan penggunaan dapat meliputi training dan evaluation. Dengan demikian pelatihan student menggunakan output model tidak dilarang secara umum oleh klausul tersebut, tetapi tetap tunduk pada penggunaan sah dan pembatasan yang berlaku.
- Audit menemukan `MODEL_CARD.md:18` menautkan `LICENSE` lokal yang tidak ditemukan di root proyek. Jangan menganggap tautan itu sebagai salinan lisensi yang tersedia; gunakan sumber upstream dan verifikasi lisensi setiap komponen.
- Pemeriksaan `supertonic_work/hf_fresh_download_20261108/LICENSE` lokal pada 2026-11-08: §4(a)–(d) mewajibkan pembatasan penggunaan yang dapat ditegakkan dan pemberitahuan kepada penerima, salinan lisensi, penandaan perubahan berkas, serta mempertahankan pemberitahuan hak dan atribusi yang relevan. Definisi distribusi §1(g) juga mencakup layanan jarak jauh. §6 menyatakan pemberi lisensi tidak mengklaim hak atas output kecuali ketentuan lain dalam lisensi; hal itu tidak menghapus definisi derivative pada §1(e).
- Attachment A melarang, antara lain, peniruan seseorang tanpa izin, keluaran yang dipakai melukai atau menipu, dan penyebaran konten tanpa pemberitahuan yang jelas bahwa konten dihasilkan mesin; periksa teks lengkap untuk semua pembatasan. Jangan menyimpulkan penggunaan komersial selalu terlarang atau bebas syarat.
- **Status:** teks lisensi lokal sudah diperiksa, tetapi kecocokan setiap aset dan voice style serta kewajiban pihak ketiga dan tujuan penggunaan akhir belum selesai diverifikasi. Jangan distribusikan model atau kode berdasarkan asumsi Apache-2.0 saja. Ini bukan pendapat hukum.

## Lingkungan teramati

- NVIDIA GeForce RTX 5060 Ti, driver 610.88, 16.311 MiB VRAM; drive E sekitar 224 GB ruang kosong pada saat pengukuran.
- Python 3.11.15; metadata paket `torch 2.11.0+cu128`, `onnxruntime 1.31.1`, `numpy 2.4.6`, `onnx 1.22.0`.
- Python global gagal mengimpor torch: `ModuleNotFoundError: No module named 'torch._C'`. Lingkungan terisolasi `.venv` di E: kemudian diverifikasi: PyTorch 2.11.0+cu128 mendeteksi RTX 5060 Ti dan berhasil melakukan perkalian matriks CUDA; ONNX Runtime 1.31.1 mencantumkan CUDA dan CPU providers. Instalasi pip sempat melaporkan kekurangan ruang dan exit 4, sehingga integritas dependensi lainnya tetap harus diuji sebelum digunakan. TEMP/TMP/cache diarahkan ke E: karena drive C: penuh.
- Repo PiVo pada revisi `515d84fc3420da6ae138f1b81017fcbc24d622a8` bersih; SHA-256 kedua checkpoint sesuai model card.

## Gerbang selanjutnya

1. Lengkapi inventaris berkas model, varian, revision dan checksum; verifikasi apakah konfigurasi lokal lengkap dan dapat berjalan.
2. Tentukan cara inferensi yang lisensinya jelas dan validasi setup secara aman. Jangan menjalankan skrip unduhan dari dalam direktorinya.
3. Evaluasi sistematis teacher dan pilih satu suara; belum ada dasar memilih voice atau menghasilkan dataset besar.
4. Perbaiki instalasi PyTorch terisolasi dan verifikasi CUDA sebelum merencanakan angka waktu training.
