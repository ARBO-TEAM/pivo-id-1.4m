"""Render v7-cont4 demo + new words, both voices."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/Scripts/python.exe"
MODEL = ROOT / "experiments/models/pivo-id-v7-cont4"
OUT = ROOT / "experiments/eval/v7cont4_listen"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    ("pi-id", 0, "Selamat pagi, apa kabar?"),
    ("pi-id-m1", 1, "Selamat pagi, apa kabar?"),
    ("scene1", 0, "Mau suara dua bahasa tapi ukuran modelnya gede banget?"),
    ("scene2", 0, "Kenalin PiVo. Satu koma empat juta, cuma lima koma enam megabita!"),
    ("scene4", 0, "Model akustik dan vocoder dalam satu paket. Ringan untuk eksperimen lokal."),
    ("scene5", 0, "PiVo, bahasa Indonesia dan Inggris. Reponya ada di GitHub!"),
    ("h1", 0, "Selamat pagi, apa kabar? Saya baik, terima kasih."),
    ("h2", 0, "Buka pukul 08:00 dan tutup pukul 20:00."),
    ("h3", 0, "PiVo adalah model suara dua bahasa, Indonesia dan Inggris."),
    ("h3-m1", 1, "PiVo adalah model suara dua bahasa, Indonesia dan Inggris."),
    ("h4", 0, "Ada sesuatu hari ini, semuanya jelas dan halus."),
    ("h5", 0, "Waktu sekarang pukul 08:15 pagi."),
    ("req-halo", 0, "Halo kak, boleh tanya-tanya? Saya mau tanya tentang kampus."),
    ("req-halo-m1", 1, "Halo kak, boleh tanya-tanya? Saya mau tanya tentang kampus."),
    ("req-nib", 0, "Cara bikin NIB gimana? Syaratnya apa saja?"),
    ("req-cireng", 0, "Harga cireng crispy berapa? Cireng jamur harganya berapa?"),
    ("req-pln", 0, "Cara pindah ke tarif listrik bisnis gimana?"),
    ("req-karier", 0, "Lulusannya bisa kerja apa? Prospek kerja bisnis digital apa saja?"),
    ("req-harga", 0, "Jaket rajut knit zip up berapa? Buket bunga snack biru berapa?"),
    ("req-bantuke", 0, "Halo, selamat datang di Program Studi Bisnis Digital UNIKMA Cilacap."),
]

for name, voice, text in CASES:
    out = OUT / f"{name}.wav"
    if out.exists():
        out.unlink()
    r = subprocess.run([str(PY), "-I", "experiments/render_narration.py",
                        "--model", str(MODEL), "--lang", "id", "--voice-id", str(voice),
                        "--text", text, "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.strip() or (r.stderr.strip().splitlines()[-1] if r.stderr else ""))
    if r.returncode != 0:
        print("FAILED", name, r.stderr[-400:])
print("done")
