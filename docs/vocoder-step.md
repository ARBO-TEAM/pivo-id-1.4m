# PiVo ONNX vocoder step

PiVo menghasilkan audio 16 kHz dari fitur akustik 20 dimensi per frame (10 ms). Implementasi lengkap ada di [`experiments/render_student.py`](../experiments/render_student.py), fungsi `features_to_audio`.

## Alur

1. Decoder ONNX menghasilkan `features` berbentuk `[jumlah_frame, 20]`.
2. Untuk setiap frame, ambil konteks frame sebelumnya, sekarang, dan berikutnya. Frame tepi diulang.
3. Hitung `periods` dari konteks, lalu jalankan `conditioning.onnx`.
4. Jalankan `vocoder_step.onnx` empat kali per frame, sambil meneruskan state dari setiap subframe ke subframe berikutnya.
5. Gabungkan keempat subframe dan terapkan deemphasis dengan state yang dipertahankan antar-frame. Hasilnya 160 sampel per frame.

Potongan inti implementasi:

```python
state = {
    name: np.zeros((1, width), np.float32)
    for name, width in zip(meta["step_state_names"], (256, 160, 128, 128, 164))
}
de_state = 0.0
for k in range(len(feats)):
    previous = feats[k - 1] if k else feats[k]
    current = feats[k]
    following = feats[k + 1] if k + 1 < len(feats) else current
    context = np.stack((previous, current, following))[None].astype(np.float32)
    periods = Runtime.periods(context)
    conditioning = cond.run(None, {"features": context, "periods": periods})[0]
    frame = []
    for subframe in range(4):
        res = step.run(None, {
            "conditioning": conditioning[:, 0, subframe * 80:(subframe + 1) * 80],
            "period": periods[:, 1],
            **state,
        })
        state = dict(zip(meta["step_state_names"], res[1:]))
        frame.append(res[0][0])
    pcm, de_state = deemphasis(np.concatenate(frame), de_state)
    out[k * 160:(k + 1) * 160] = pcm
```

State awal adalah nol. Tidak ada input noise scale atau temperature pada jalur ini. Frame fitur nol belum tentu merepresentasikan silence; karena itu warm-up dengan frame nol tidak diterapkan tanpa validasi.

## Opsi smoothing eksperimental

`render_narration.py --smooth` menerapkan filter Savitzky–Golay 5 frame pada kanal 0–17 sebelum vocoder, lalu bandpass lembut dan spectral gate frekuensi tinggi pada audio. Proses DSP dilakukan **offline**, bukan janji latensi kurang dari 1 ms. Mode ini tidak mengubah kanal 18 dan 19. Tidak ada normalisasi peak otomatis supaya tingkat suara tetap sebanding.

`--selective` menawarkan alternatif tanpa DSP offline: kelancaran tiga frame hanya pada kanal 6–17 ketika flux rendah, ditambah median pitch tiga frame pada daerah yang diperkirakan bersuara. Kanal 19 adalah korelasi pitch yang ditransformasi, bukan energi; nilai > 0 selama tiga frame dipakai sebagai heuristik eksperimental untuk voicing, bukan detektor terlatih. Kanal 19 tetap utuh. `--deemphasis 0.88` dapat dipakai untuk uji suara lebih hangat; default tetap 0.85 sesuai runtime asli. Mengubahnya juga dapat mengubah level audio.

```powershell
.\.venv\Scripts\python.exe -I experiments/render_narration.py --model experiments/models/pivo-id-1.4m --lang id --voice-id 0 --text "Selamat pagi, apa kabar?" --smooth --out smooth.wav
```

Gunakan `--selective --deemphasis 0.88` sebagai alternatif, atau hilangkan keduanya untuk hasil asli; `--voice-id 1` memilih suara pria, dan `--lang en` memakai frontend Inggris. `--speed` sudah tersedia, tetapi peningkatan durasi tidak dipaksakan karena dapat mengubah ritme dan waktu video. Opsi ini hanya memengaruhi renderer tersebut, bukan file model ONNX atau aplikasi lain.

Perlu uji dengar A/B: smoothing dapat mengurangi detail atau mengubah karakter suara, dan skor pengenalan teks saja tidak membuktikan kualitas suara lebih baik. Model siswa tetap tidak setara dengan teacher Supertonic.
