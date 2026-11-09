"""Kumpulkan kalimat baru untuk v8: uji yang dikirim user + perusahaan + singkatan."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# kalimat yang user kirim sebagai uji (persis) + perusahaan + variasi akronim/angka
texts = [
    "Pejabat teras itu duduk santai di teras rumah sambil memakan apel setelah apel pagi selesai.",
    "Dia memakai kemeja abu-abu saat menghadiri rapat di gedung Kementerian.",
    "Khusus hari ini, pihak syahbandar mengumumkan jadwal keberangkatan kapal laut.",
    "Nyanyian burung gereja terdengar merdu di antara pepohonan yang rimbun.",
    "Masyarakat ikhlas membantu evakuasi korban bencana tanah longsor.",
    "Tagihan listrik pada tanggal 17 Agustus 2024 mencapai dua juta tiga ratus lima puluh ribu lima ratus rupiah, naik dua belas koma lima persen dari bulan lalu.",
    "Pukul delapan empat puluh lima waktu Indonesia barat, suhu di ruangan ini tercatat dua puluh delapan koma lima derajat Celsius dengan kelembapan enam puluh persen.",
    "Nomor resi pengiriman paket Anda adalah I D nol satu sembilan dua delapan tiga tujuh empat enam lima.",
    "Petugas K P K dan Polri bekerja sama dengan pihak B U M N untuk memeriksa laporan P B B.",
    "Pada hari rabu, dua puluh lima Oktober dua ribu dua puluh tiga pukul dua belas tiga puluh waktu Indonesia barat, Kementerian Kesehatan mengumumkan anggaran sebesar satu koma lima triliun rupiah untuk program vaksinasi massal.",
    "Apakah target delapan puluh persen masyarakat sudah realistis? Tentu saja, tegas juru bicara tersebut dengan optimis di hadapan wartawan.",
    "Perusahaan itu mengumumkan hasil rapat pemegang saham hari ini.",
    "Perusahaan teknologi asal Bandung memproduksi perangkat elektronik untuk ekspor.",
    "Karyawan perusahaan itu menerima tunjangan hari raya tahun ini.",
    "Direktur perusahaan menyampaikan laporan keuangan kuartal pertama.",
    "Perusahaan penerbangan itu menurunkan harga tiket ke Bali.",
    "Perusahaan properti itu membangun perumahan baru di pinggiran kota.",
    "Perusahaan farmasi itu memproduksi obat-obatan untuk rumah sakit.",
    "Perusahaan manufaktur itu memberikan pelatihan untuk karyawan baru.",
    "Perusahaan retail itu membuka gerai baru di mal kota.",
    "Perusahaan jasa itu menawarkan paket wisata untuk keluarga.",
    "Kantor perusahaan itu terletak di gedung tinggi pusat kota.",
    "Perusahaan itu mencari mitra bisnis untuk ekspansi ke luar negeri.",
    "Perusahaan leasing itu menyediakan pembiayaan untuk kendaraan bermotor.",
    "Perusahaan asuransi itu memberikan polis perlindungan jiwa.",
    "Perusahaan tambang itu beroperasi di Kalimantan Timur.",
    "Perusahaan kopi itu mengekspor biji kopi ke Eropa.",
    "Perusahaan tekstil itu memproduksi kain untuk ekspor.",
    "Perusahaan logistik itu mengirim barang ke seluruh Indonesia.",
    "Perusahaan makanan itu memproduksi camilan kemasan untuk anak-anak.",
    "Perusahaan telekomunikasi itu meluncurkan layanan internet cepat.",
    "Perusahaan perakitan itu membangun bendungan untuk irigasi sawah.",
    "Perusahaan otomotif itu memproduksi mobil listrik murah.",
    "Perusahaan minyak itu mengebor sumur baru di lepas pantai.",
    "Perusahaan konstruksi itu membangun jembatan penghubung dua kota.",
    "Perusahaan media itu menayangkan berita lokal setiap malam.",
    "Perusahaan perhotelan itu membuka resort baru di lombok.",
    "Perusahaan perikanan itu mengekspor ikan tuna ke Jepang.",
    "Perusahaan perkebunan itu memanen sawit di Sumatera.",
    "Perusahaan kertas itu memproduksi kertas daur ulang.",
    "Perusahaan plastik itu mengurangi penggunaan kantong sekali pakai.",
    "Perusahaan farmasi itu memproduksi vitamin untuk anak.",
    "Perusahaan perhiasan itu memamerkan koleksi emas di pameran.",
    "Perusahaan sepatu itu memproduksi sepatu olahraga lokal.",
    "Perusahaan tas itu memproduksi tas kulit asli.",
    "Perusahaan furnitur itu memproduksi meja kayu jati.",
    "Perusahaan mainan itu memproduksi boneka untuk anak-anak.",
    "Korupsi di perusahaan itu diselidiki oleh K P K.",
    "Karyawan perusahaan B U M N itu menerima THR tahun ini.",
    "Dana pensiun P B B itu dikelola oleh perusahaan asuransi.",
    "Polri mengamankan kerusuhan di depan kantor perusahaan itu.",
    "Perusahaan itu terdaftar di bursa efek Indonesia.",
    "Perusahaan itu memenangkan tender proyek jalan tol.",
    "Perusahaan itu memperluas jaringan distribusi ke Jawa Barat.",
    "Perusahaan itu meluncurkan produk baru untuk pasar ASEAN.",
    "Perusahaan itu mengurangi karyawan karena efisiensi biaya.",
    "Perusahaan itu membuat komitmen untuk mengurangi emisi karbon.",
    "Perusahaan itu mendapatkan sertifikat halal untuk produknya.",
    "Perusahaan itu membuka lowongan kerja untuk lulusan baru.",
    "Perusahaan itu memberikan beasiswa untuk mahasiswa berprestasi.",
    "Perusahaan itu menyumbang untuk korban bencana banjir.",
    "Perusahaan itu mengadakan seminar wirausaha untuk pemuda.",
    "Perusahaan itu memproduksi kemasan ramah lingkungan.",
    "Perusahaan itu menggunakan tenaga surya untuk pabriknya.",
    "Perusahaan itu menjalin kerja sama dengan universitas lokal.",
    "Perusahaan itu mendapatkan penghargaan perusahaan terbaik tahun ini.",
    "Perusahaan itu berekspansi ke pasar Asia Tenggara.",
    "Perusahaan itu memproduksi sari buah untuk ekspor ke Singapura.",
    "Perusahaan itu membangun pabrik baru di Jawa Tengah.",
    "Perusahaan itu menyediakan fasilitas kesehatan untuk karyawan.",
    "Perusahaan itu mengadakan buka puasa bersama untuk karyawan.",
    "Perusahaan itu merayakan ulang tahun ke dua puluh lima.",
    "Perusahaan itu memproduksi jamu tradisional untuk kesehatan.",
    "Perusahaan itu mengekspor rempah-rempah ke Timur Tengah.",
    "Perusahaan itu memproduksi batik untuk pasar Eropa.",
    "Perusahaan itu membangun sekolah untuk anak-anak karyawan.",
    "Perusahaan itu memberikan tunjangan pendidikan untuk anak karyawan.",
    "Perusahaan itu memproduksi susu segar untuk anak-anak.",
    "Perusahaan itu membuka toko online untuk penjualan daring.",
    "Perusahaan itu menggunakan platform digital untuk pemasaran.",
    "Perusahaan itu mengadakan pameran produk di Jakarta.",
    "Perusahaan itu memproduksi lampu LED hemat energi.",
    "Perusahaan itu membangun apartemen untuk hunian karyawan.",
    "Perusahaan itu menyediakan pelatihan bahasa Inggris untuk karyawan.",
    "Perusahaan itu membuat aplikasi mobile untuk pelanggan.",
    "Perusahaan itu mendapatkan dana investasi dari luar negeri.",
    "Perusahaan itu memproduksi kerajinan tangan untuk ekspor.",
    "Perusahaan itu membuka warung kopi untuk karyawan.",
    "Perusahaan itu memproduksi sabun mandi untuk hotel.",
    "Perusahaan itu menggunakan bahan baku lokal untuk produksi.",
    "Perusahaan itu mengadakan lomba inovasi untuk karyawan.",
    "Perusahaan itu memproduksi tas kresek ramah lingkungan.",
    "Perusahaan itu membangun kolam ikan untuk pemberdayaan masyarakat.",
    "Perusahaan itu memberikan pinjaman lunak untuk usaha kecil.",
    "Perusahaan itu memproduksi minyak goreng untuk kebutuhan dalam negeri.",
    "Perusahaan itu mengekspor teh ke Rusia.",
    "Perusahaan itu memproduksi baju batik untuk seragam sekolah.",
    "Perusahaan itu membuka klinik kesehatan untuk masyarakat sekitar.",
    "Perusahaan itu memberikan bantuan untuk renovasi masjid.",
    "Perusahaan itu memproduksi kecap untuk pasar nasional.",
    "Perusahaan itu membangun lapangan bola untuk pemuda desa.",
    "Perusahaan itu menyediakan perpustakaan untuk anak-anak.",
    "Perusahaan itu memproduksi gula aren untuk ekspor.",
    "Perusahaan itu mengadakan donor darah untuk karyawan.",
    "Perusahaan itu memproduksi tisu untuk kebutuhan rumah tangga.",
    "Perusahaan itu membuka bakery untuk menjual roti segar.",
    "Perusahaan itu memproduksi deterjen cair untuk rumah tangga.",
    "Perusahaan itu memberikan pelatihan keterampilan untuk ibu-ibu.",
    "Perusahaan itu memproduksi kaos untuk komunitas motor.",
    "Perusahaan itu membangun sumur bor untuk desa kering.",
    "Perusahaan itu mengekspor jambu biji ke Malaysia.",
    "Perusahaan itu memproduksi kecantikan untuk pasar remaja.",
    "Perusahaan itu mengadakan festival kuliner untuk masyarakat.",
    "Perusahaan itu memproduksi sampo untuk rambut rontok.",
    "Perusahaan itu membuka toko buku untuk pelajar.",
    "Perusahaan itu memberikan sertifikat untuk keamanan pangan.",
    "Perusahaan itu memproduksi es krim untuk anak-anak.",
    "Perusahaan itu membangun pusat pelatihan kerja.",
    "Perusahaan itu menyediakan asuransi kesehatan untuk keluarga karyawan.",
    "Perusahaan itu memproduksi pakan ternak untuk peternak.",
    "Perusahaan itu mengadakan pasar murah untuk masyarakat.",
    "Perusahaan itu memproduksi lilin aromaterapi untuk relaksasi.",
    "Perusahaan itu membuka laundry untuk karyawan.",
    "Perusahaan itu memberikan beasiswa untuk anak karyawan berprestasi.",
    "Perusahaan itu memproduksi handuk untuk kebutuhan hotel.",
    "Perusahaan itu membangun rumah susun untuk pekerja.",
    "Perusahaan itu memproduksi bumbu instan untuk masak cepat.",
    "Perusahaan itu mengadakan pelatihan digital untuk UMKM.",
    "Perusahaan itu memproduksi jus buah untuk kendaraan umum.",
    "Perusahaan itu membuka warung kopi untuk karyawan.",
    "Perusahaan itu memberikan pinjaman untuk petani lokal.",
    "Perusahaan itu memproduksi lilin untuk acara adat.",
    "Perusahaan itu membangun taman bermain untuk anak-anak.",
    "Perusahaan itu memproduksi pelembut kain untuk rumah tangga.",
    "Perusahaan itu mengadakan lomba masak untuk ibu-ibu.",
    "Perusahaan itu memproduksi kopi bubuk untuk ekspor.",
]

# dedup + saring yang sudah ada di v6/v7 build_meta
meta = Path("experiments/data/pilot_v6_id/build_meta.jsonl").read_text(encoding="utf-8")
known = set()
for line in meta.splitlines():
    try: known.add(json.loads(line).get("text", "").strip().lower())
    except Exception: pass
meta2 = Path("experiments/data/pilot_v7_id/build_meta.jsonl").read_text(encoding="utf-8")
for line in meta2.splitlines():
    try: known.add(json.loads(line).get("text", "").strip().lower())
    except Exception: pass

out = []
seen = set()
for t in texts:
    t = t.strip()
    if len(t) < 5: continue
    low = t.lower()
    if low in seen or low in known: continue
    seen.add(low)
    out.append({"id": f"V8{len(out):03d}", "text": t,
                "split": "val" if len(out) % 10 == 9 else "train"})

Path("experiments/text/extra_v8_id.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
print(f"new texts: {len(out)} train={sum(1 for r in out if r['split']=='train')} val={sum(1 for r in out if r['split']=='val')}")
for r in out[:8]: print(r["id"], r["text"][:80])
