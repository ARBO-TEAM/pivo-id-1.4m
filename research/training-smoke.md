# Uji loop training — data buatan, bukan ucapan

Tanggal: 2026-11-08.

`experiments/smoke_training.py` membuat dua matriks fitur acak dan durasi buatan, masing-masing satu untuk train dan validasi, di direktori temporer. Dua langkah optimisasi dan evaluasi dijalankan pada CUDA memakai `.venv/Scripts/python.exe -I`; skrip memeriksa adanya `best.pt`, `last.pt`, dan metrik langkah kedua. Direktori temporer dan checkpoint uji dihapus setelah skrip selesai.

Hasil observasi: proses exit 0, kedua langkah selesai, loss train terbatas, dan `SMOKE PASS` tercetak. Ini hanya membuktikan loop, GPU, evaluasi, dan penyimpanan checkpoint dapat berjalan pada data kecil buatan. Ini **bukan** dataset Supertonic, fine-tuning Bahasa Indonesia, bukti kualitas model, ataupun pembuktian bahwa resume identik.

Jalankan ulang dari workspace tepercaya:

```powershell
$env:TEMP = 'E:\test\tts-test\pivo1-4m\temp'
$env:TMP = $env:TEMP
& 'E:\test\tts-test\pivo1-4m\.venv\Scripts\python.exe' -I 'E:\test\tts-test\pivo1-4m\experiments\smoke_training.py'
```

Model akustik asli dan dataset nyata belum dilatih pada pengujian ini. Gerbang hak penggunaan, kualitas teacher, fitur, dan alignment tetap berlaku.
