"""Render v9 demo + perusahaan + evakuasi, smooth mode."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/Scripts/python.exe"
MODEL = ROOT / "experiments/models/pivo-id-v9"
OUT = ROOT / "experiments/eval/v9_listen"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    ("pi-id", 0, "Selamat pagi, apa kabar?"),
    ("pi-id-m1", 1, "Selamat pagi, apa kabar?"),
    ("scene1", 0, "Mau suara dua bahasa tapi ukuran modelnya gede banget?"),
    ("scene2", 0, "Kenalin PiVo. Satu koma empat juta, cuma lima koma enam megabita!"),
    ("h3", 0, "PiVo adalah model suara dua bahasa, Indonesia dan Inggris."),
    ("h3-m1", 1, "PiVo adalah model suara dua bahasa, Indonesia dan Inggris."),
    ("test-perusahaan", 0, "Perusahaan itu mengumumkan hasil rapat pemegang saham hari ini."),
    ("test-perusahaan2", 0, "Direktur perusahaan menyampaikan laporan keuangan kuartal pertama."),
    ("test-perusahaan3", 0, "Perusahaan teknologi asal Bandung memproduksi perangkat elektronik untuk ekspor."),
    ("test-perusahaan4", 0, "Perusahaan manufaktur itu memberikan pelatihan untuk karyawan baru."),
    ("test-perusahaan5", 0, "Kantor perusahaan itu terletak di gedung tinggi pusat kota."),
    ("test-perusahaan6", 0, "Perusahaan itu mendapatkan sertifikat halal untuk produknya."),
    ("test-evakuasi", 0, "Masyarakat ikhlas membantu evakuasi korban bencana tanah longsor."),
    ("test-evakuasi2", 0, "Tim evakuasi membawa korban banjir ke tempat pengungsian."),
    ("test-evakuasi3", 0, "Evakuasi udara dilakukan untuk korban gempa di pegunungan."),
    ("test-evakuasi4", 0, "Evakuasi warga di pesisir dilakukan karena ancaman tsunami."),
    ("test-kemenkes", 0, "Pada hari rabu, dua puluh lima Oktober dua ribu dua puluh tiga pukul dua belas tiga puluh waktu Indonesia barat, Kementerian Kesehatan mengumumkan anggaran sebesar satu koma lima triliun rupiah untuk program vaksinasi massal."),
    ("test-kemenkes2", 0, "Apakah target delapan puluh persen masyarakat sudah realistis? Tentu saja, tegas juru bicara tersebut dengan optimis di hadapan wartawan."),
]

for name, voice, text in CASES:
    out = OUT / f"{name}.wav"
    if out.exists():
        out.unlink()
    r = subprocess.run([str(PY), "-I", "experiments/render_narration.py",
                        "--model", str(MODEL), "--lang", "id", "--voice-id", str(voice),
                        "--text", text, "--out", str(out), "--selective", "--deemphasis", "0.88"],
                       cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.strip() or (r.stderr.strip().splitlines()[-1] if r.stderr else ""))
    if r.returncode != 0:
        print("FAILED", name, r.stderr[-400:])
print("done")
