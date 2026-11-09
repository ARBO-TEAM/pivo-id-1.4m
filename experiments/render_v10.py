"""Render v10 reduplikasi + R + stress, smooth mode."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/Scripts/python.exe"
MODEL = ROOT / "experiments/models/pivo-id-v10"
OUT = ROOT / "experiments/eval/v10_listen"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    ("pi-id", 0, "Selamat pagi, apa kabar?"),
    ("pi-id-m1", 1, "Selamat pagi, apa kabar?"),
    ("t-redux-1", 0, "Anak-anak berlari-larian mengejar layang-layang putus di lapangan yang berumput tebal."),
    ("t-redux-2", 0, "Ibu memasak gulai dengan mengaduk-aduk bumbu secara perlahan-lahan agar santannya tidak pecah."),
    ("t-redux-3", 0, "Daun-daun kering itu tampak melayang-layang tertiup angin sore yang sepoi-sepoi."),
    ("t-redux-4", 0, "Gerak-gerik orang tak dikenal itu mencurigakan saat mondar-mandir di depan gang."),
    ("t-glotal-3", 0, "Gerak bolak-balik bebek di tepi selokan itu membuat anak kecil itu tertawa terbahak-bahak."),
    ("t-r-1", 0, "Raja rimba itu mengaum keras meruntuhkan reruntuhan ranting kering di lereng curam."),
    ("t-r-2", 0, "Rara meraba-raba ruji-ruji roda kereta yang berkarat parah."),
    ("t-r-3", 0, "Pencuri itu berlari tergesa-gesa menyusuri lorong pasar yang ramai pembeli."),
    ("t-r-4", 0, "Rudi merapikan rak-rak buku di ruang tamu."),
    ("t-r-5", 0, "Rina merawat rumput liar di kebun belakang rumah."),
    ("t-stress-1", 0, "Kakek duduk-duduk di teras belakang sambil memperhatikan gerak-gerik katak yang melompat-lompat di atas petak sawah."),
    ("t-stress-2", 0, "Tak berselang lama, terdengar suara letupan ketel air panas dari dapur."),
    ("t-stress-3", 0, "Ibu berteriak-teriak meminta tolong agar kompor minyak tanah segera dimatikan sebelum apinya merambat ke bilik bambu."),
    ("t-perusahaan", 0, "Perusahaan itu mengumumkan hasil rapat pemegang saham hari ini."),
    ("t-evakuasi", 0, "Masyarakat ikhlas membantu evakuasi korban bencana tanah longsor."),
    ("t-kemenkes", 0, "Pada hari rabu, dua puluh lima Oktober dua ribu dua puluh tiga pukul dua belas tiga puluh waktu Indonesia barat, Kementerian Kesehatan mengumumkan anggaran sebesar satu koma lima triliun rupiah untuk program vaksinasi massal."),
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
