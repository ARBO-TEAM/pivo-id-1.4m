"""Generate GitHub Pages TTS demo (docs/index.html) — no runtime dependency."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "index.html"

# (wav path under docs/, title, subtitle)
SAMPLES = [
    ("assets/pi-id.wav", "Sapaan (Pi, v11)",
     "Selamat pagi, apa kabar? — suara Pi (perempuan)."),
    ("assets/t-r-2.wav", "R-trill + reduplikasi (v11)",
     "Rara meraba-raba ruji-ruji roda kereta yang berkarat parah."),
    ("assets/t-redux-1.wav", "Reduplikasi berantai (v11)",
     "Anak-anak berlari-larian mengejar layang-layang putus …"),
    ("assets/t-stress-1.wav", "Kalimat panjang (v11)",
     "Kakek duduk-duduk di teras belakang sambil memperhatikan …"),
    ("assets/t-perusahaan.wav", "Suffix -an (v11)",
     "Perusahaan itu mengumumkan hasil rapat pemegang saham hari ini."),
    ("assets/t-r-2-v10.wav", "R-trill + reduplikasi (v10)",
     "Reduplikasi yang sama di v10 — bandingkan dengan v11 di atas."),
    ("assets/t-r-2-fixed.wav", "v10 + frontend baru",
     "Model v10 dengan frontend koma baru, tanpa retraining."),
]

cards = "\n".join(
    f'    <div class="card"><h2>{t}</h2><p class="sub">{s}</p>'
    f'<audio controls preload="none" src="{w}"></audio>'
    f'<p class="src"><a href="{w}">{w}</a></p></div>'
    for w, t, s in SAMPLES)

rows = "\n".join(
    f"      <tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>"
    for a, b, c in [
        ("v7", "2,732 rows (ID-only, from v6)", "0.2016 / 0.0960"),
        ("v8–v10", "incremental vocabulary", "0.1924 → 0.1904"),
        ("v11", "4,386 rows, comma-fixed reduplikasi", "0.1884 / 0.0747"),
    ])

html = f"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PiVo-ID 1.4M — Demo TTS Bahasa Indonesia</title>
<meta name="description" content="Indonesian TTS 1.4M-parameter ONNX demo. Listen to v11 sample WAVs.">
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 1rem; }}
.card {{ border: 1px solid #8884; border-radius: 8px; padding: 1rem; }}
.card h2 {{ font-size: 1rem; margin: 0 0 .25rem; }}
.sub {{ color: #666; font-size: .9rem; margin: 0 0 .5rem; }}
.src {{ font-size: .75rem; overflow-wrap: anywhere; }}
audio {{ width: 100%; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #8884; padding: .35rem .6rem; text-align: left; }}
code {{ background: #8882; padding: .1rem .3rem; border-radius: 4px; }}
</style>
</head>
<body>
<h1>PiVo-ID 1.4M — Demo TTS Bahasa Indonesia</h1>
<p>Fine-tune 1.4M parameter <a href="https://huggingface.co/empero-ai/PiVo-1.4M">PiVo-1.4M</a>
untuk Bahasa Indonesia (16&nbsp;kHz mono, ONNX 5,6&nbsp;MB, suara Pi &amp; Vo).
Bagian v6 bilingual (Indonesia + Inggris) masih tersimpan; model
sekarang fokus Indonesia saja karena Inggris menurun saat retrained.
Unduh dari <a href="https://github.com/ARBO-TEAM/pivo-id-1.4m/releases/tag/v1.1.0">GitHub Release v1.1.0</a>.</p>

<h2>Dengarkan</h2>
<div class="grid">
{cards}
</div>

<h2>Coba sendiri (5 baris)</h2>
<pre><code>pip install -r requirements.txt
unzip pivo-id-v11-onnx.zip
python -I experiments/render_narration.py \\
  --model pivo-id-v11 --lang id --voice-id 0 \\
  --text "Selamat pagi, apa kabar?" --out coba.wav</code></pre>

<h2>Hasil ukur (validasi)</h2>
<table><tr><th>Run</th><th>Korpus</th><th>Val feature / duration</th></tr>
{rows}
</table>
<p>ONNX parity v11: encoder 2.21e-06, log-durations 9.54e-07,
decoder 7.15e-05 (gate 2e-04). Model card:
<a href="https://github.com/ARBO-TEAM/pivo-id-1.4m/blob/main/MODEL_CARD.md">MODEL_CARD.md</a>
· Lisensi: <a href="https://github.com/ARBO-TEAM/pivo-id-1.4m/blob/main/LICENSE">LICENSE</a>
(Apache-2.0; derivatif Supertonic Open RAIL-M — lihat model card).</p>
</body>
</html>
"""
OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({len(html)} bytes, {len(SAMPLES)} samples)")
