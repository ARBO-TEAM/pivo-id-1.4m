"""Kumpulkan kalimat v10: reduplikasi besar + R trill + stress test."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

texts = [
    # reduplikasi penuh/parsial (jeda mikro antar pengulangan)
    "Anak-anak berlari-larian mengejar layang-layang putus di lapangan yang berumput tebal.",
    "Ibu memasak gulai dengan mengaduk-aduk bumbu secara perlahan-lahan agar santannya tidak pecah.",
    "Daun-daun kering itu tampak melayang-layang tertiup angin sore yang sepoi-sepoi.",
    "Gerak-gerik orang tak dikenal itu mencurigakan saat mondar-mandir di depan gang.",
    "Kakek duduk-duduk di teras belakang sambil memperhatikan gerak-gerik katak yang melompat-lompat di atas petak sawah.",
    "Ibu berteriak-teriak meminta tolong agar kompor minyak tanah segera dimatikan sebelum apinya merambat ke bilik bambu.",
    "Tak berselang lama, terdengar suara letupan ketel air panas dari dapur.",
    "Gerak bolak-balik bebek di tepi selokan itu membuat anak kecil itu tertawa terbahak-bahak.",
    "Rara meraba-raba ruji-ruji roda kereta yang berkarat parah.",
    "Pencuri itu berlari tergesa-gesa menyusuri lorong pasar yang ramai pembeli.",
    "Raja rimba itu mengaum keras meruntuhkan reruntuhan ranting kering di lereng curam.",
    # variasi reduplikasi anak-anak
    "Anak-anak bermain bola di lapangan sore ini.",
    "Anak-anak itu belajar berenang di kolam renang sekolah.",
    "Anak-anak makan siang bersama di kantin sekolah.",
    "Anak-anak itu menonton film kartun di rumah tetangga.",
    "Buku-buku itu ditata rapi di rak perpustakaan.",
    "Pohon-pohon itu ditanam sepanjang jalan desa.",
    "Bunga-bunga itu mekar indah di taman kota.",
    "Burung-burung itu berkicau di pagi hari.",
    "Ikan-ikan itu berenang di akuarium besar.",
    "Mobil-mobil itu parkir rapi di tempat parkir mal.",
    "Motor-motor itu melaju kencang di jalan tol.",
    "Rumah-rumah itu dibangun dengan gaya modern.",
    "Orang-orang itu berkumpul di alun-alun kota.",
    "Barang-barang itu dikemas dengan aman.",
    "Baju-baju itu dicuci di laundry langganan.",
    "Sepatu-sepatu itu dijual murah di pasar.",
    "Tas-tas itu terbuat dari kulit asli.",
    "Topi-topi itu dipakai untuk melindungi dari panas.",
    "Meja-meja itu ditata untuk acara rapat.",
    "Kursi-kursi itu disusun mengelilingi meja.",
    "Pintu-pintu itu dicat ulang kemarin.",
    "Jendela-jendela itu dibuka untuk sirkulasi udara.",
    "Lampu-lampu itu menyala terang di ruang tamu.",
    "Kunci-kunci itu disimpan di laci meja.",
    # reduplikasi kata kerja
    "Mereka berlari-larian di taman setiap pagi.",
    "Anak itu menangis meraung-raung minta jajan.",
    "Ibu menyanyi lirih-lirih sambil menidurkan bayi.",
    "Bapak membaca koran berulang-ulang setiap pagi.",
    "Dia berjalan pelan-pelan menaiki tangga.",
    "Mereka berbicara pelan-pelan agar tidak didengar orang.",
    "Anak itu makan lahap-lahap sampai habis.",
    "Mereka tertawa terbahak-bahak mendengar lelucon itu.",
    "Dia mengangguk-angguk setuju dengan usulan itu.",
    "Mereka menggeleng-geleng tidak percaya cerita itu.",
    "Anak itu mengedip-ngedip menahan kantuk.",
    "Dia menggosok-gosok mata karena lelah.",
    "Mereka menepuk-nepuk tangan merayakan kemenangan.",
    "Anak itu memukul-mukul meja karena kesal.",
    "Dia mengangguk-angguk tanda mengerti.",
    "Mereka melambai-lambai tangan saat perpisahan.",
    "Anak itu merangkak-rangkak belajar berjalan.",
    "Dia mengusap-usap kepala anak itu dengan lembut.",
    "Mereka menari-nari mengikuti irama musik.",
    "Anak itu berloncat-loncat di kasur.",
    "Mereka berteriak-teriak memanggil teman.",
    "Dia membisik-bisik pesan rahasia ke telinga temannya.",
    "Mereka berdesis-desus membicarakan gosip terbaru.",
    "Anak itu merengek-rengek minta dibelikan mainan.",
    "Dia mengintip-ngintip dari balik jendela.",
    "Mereka mengobrol-ngobrol santai di teras rumah.",
    "Anak itu melompat-lompat girang kegirangan.",
    "Mereka berpelukan erat-erat melepas rindu.",
    "Dia memeluk erat-erat adiknya yang ketakutan.",
    "Mereka berpegangan tangan erat-erat saat menyeberang.",
    # reduplikasi sifat/keadaan
    "Daun-daun itu hijau-hijau segar setelah hujan.",
    "Langit sore itu merah-merah keemasan.",
    "Awan itu putih-putih berarak perlahan.",
    "Bunga itu merah-merah mekar di taman.",
    "Baju itu biru-biru cerah dipakai sore ini.",
    "Sepatu itu hitam-hitam mengkilap baru dibeli.",
    "Matahari itu kekuning-kuningan menjelang senja.",
    "Awan itu abu-abu gelap menandakan hujan.",
    "Daun itu hijau-hijau rimbun menaungi jalan.",
    "Air itu jernih-jernih mengalir di sungai.",
    "Bintang itu berkelip-kelip di langit malam.",
    "Lampu itu redup-redup temaram di sudut ruangan.",
    "Suara itu lirih-lirih terdengar dari kejauhan.",
    "Tawa itu terbahak-bahak memecah keheningan.",
    "Tangis itu meraung-raung membuat ibu khawatir.",
    "Angin itu sepoi-sepoi bertiup lembut sore ini.",
    "Hujan itu rintik-rintik membasahi bumi.",
    "Salju itu turun lebat-lebat menutupi tanah.",
    "Kabut itu tebal-tebal menyelimuti pegunungan.",
    "Ombak itu besar-besar menghantam pantai.",
    # frasa reduplikasi
    "Mereka bolak-balik ke pasar membeli kebutuhan.",
    "Dia mondar-mandir di depan pintu menunggu tamu.",
    "Anak itu merangkak-rangkak merayap di lantai.",
    "Mereka berdesakan-desakan masuk ke dalam bus.",
    "Orang-orang itu berdesak-desakan antre sembako.",
    "Mereka berbaris rapi-rapi menunggu giliran.",
    "Anak-anak itu bermain layangan-layang di sawah.",
    "Mereka menari-nari mengelilingi api unggun.",
    "Dia berbisik-bisik membicarakan rahasia.",
    "Mereka berbondong-bondong datang ke acara.",
    "Warga berduyun-duyun mengantre di TPS.",
    "Mereka beriring-iringan pulang dari masjid.",
    "Anak itu merangkak-rangkak merayap ke arah pintu.",
    "Mereka berpelukan erat-erat melepas kangen.",
    "Dia memeluk erat-erat ibunya yang menangis.",
    "Mereka berpegangan erat-erat melewati jembatan goyang.",
    "Anak itu mengangguk-angguk tanda setuju.",
    "Mereka menggeleng-geleng tidak percaya.",
    "Dia menepuk-nepuk tangan semangat untuk tim.",
    "Mereka melambai-lambai tangan pada pesawat yang lepas landas.",
    # R trill heavy
    "Raja rimba itu mengaum keras meruntuhkan reruntuhan ranting kering di lereng curam.",
    "Rara meraba-raba ruji-ruji roda kereta yang berkarat parah.",
    "Pencuri itu berlari tergesa-gesa menyusuri lorong pasar yang ramai pembeli.",
    "Rudi merapikan rak-rak buku di ruang tamu.",
    "Rina merawat rumput liar di kebun belakang rumah.",
    "Ratu merias rambut rapi untuk acara pesta.",
    "Rojak merobek roti untuk burung merpati.",
    "Rangga merakit rakit bambu untuk menyeberang sungai.",
    "Roro merobohkan reruntuhan rumah tua dengan alat berat.",
    "Rudi merobohkan pohon-pohon kering di kebunnya.",
    "Rini merajut jaket rajut dari benang wol.",
    "Rana merangkai rangkaian bunga segar untuk ibu.",
    "Rara merawat rumpun bambu di halaman rumah.",
    "Rudi memandikan kuda putih di sungai kecil.",
    "Rina membaca surat dari kakaknya di perantauan.",
    "Rudi mengendarai sepeda motor ke pasar pagi.",
    "Rina memasak rendang pedas untuk keluarga.",
    "Rara menari tari tradisional di panggung kesenian.",
    "Rudi bertani padi di sawah milik keluarganya.",
    "Rina merawat kucing peliharaan dengan penuh kasih.",
    "Rara berlatih berenang setiap pagi di kolam.",
    "Rudi bermain bola basket di lapangan sekolah.",
    "Rina berkebun sayuran di pekarangan rumah.",
    "Rara belajar berenang untuk lomba renang sekolah.",
    "Rudi berlatih karate untuk bela diri.",
    "Rina bersepeda santai mengelilingi taman kota.",
    "Rara bernyanyi lagu daerah di acara keluarga.",
    "Rudi bertani jagung di ladang sebelah rumah.",
    "Rina merajut selimut untuk bayi baru lahir.",
    "Rara menggambar pemandangan gunung di kertas.",
    "Rudi memancing ikan di sungai belakang kampung.",
    "Rina menjual hasil kebun di pasar pagi.",
    "Rudi mengajar anak-anak membaca di rumah.",
    "Rara memasak soto ayam untuk sarapan pagi.",
    "Rina menanam cabai di pot kecil di teras.",
    "Rudi memperbaiki atap rumah yang bocor.",
    "Rara menyapu halaman setiap pagi sebelum sekolah.",
    "Rina mencuci pakaian di sungai kecil.",
    "Rudi memelihara ayam kampung di kandang.",
    "Rara memetik bunga mawar di kebun tetangga.",
    "Rina belajar matematika untuk ujian semester.",
    "Rudi belajar mengemudi mobil di lapangan kosong.",
    "Rara belajar memasak masakan Padang dari ibu.",
    "Rina belajar menyanyi untuk audisi sekolah.",
    "Rudi belajar silat untuk pertandingan daerah.",
    # stress test gabungan
    "Tak berselang lama, terdengar suara letupan ketel air panas dari dapur.",
    "Kakek duduk-duduk di teras belakang sambil memperhatikan gerak-gerik katak yang melompat-lompat di atas petak sawah.",
    "Ibu berteriak-teriak meminta tolong agar kompor minyak tanah segera dimatikan sebelum apinya merambat ke bilik bambu.",
    "Gerak bolak-balik bebek di tepi selokan itu membuat anak kecil itu tertawa terbahak-bahak.",
    "Tak berselang lama terdengar suara letupan ketel air panas.",
    "Kakek duduk-duduk di teras sambil memperhatikan katak.",
    "Ibu berteriak-teriak meminta tolong agar kompor dimatikan.",
]

# dedup + saring yang sudah ada di build_meta v6/v7/v8/v9
known = set()
for path in ["experiments/data/pilot_v6_id/build_meta.jsonl",
             "experiments/data/pilot_v7_id/build_meta.jsonl",
             "experiments/data/pilot_v8_id/build_meta.jsonl",
             "experiments/data/pilot_v9_id/build_meta.jsonl"]:
    p = Path(path)
    if not p.exists(): continue
    for line in p.read_text(encoding="utf-8").splitlines():
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
    out.append({"id": f"V10{len(out):03d}", "text": t,
                "split": "val" if len(out) % 10 == 9 else "train"})

Path("experiments/text/extra_v10_id.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
print(f"new texts: {len(out)} train={sum(1 for r in out if r['split']=='train')} val={sum(1 for r in out if r['split']=='val')}")
for r in out[:10]: print(r["id"], r["text"][:80])
