// frontend.js — JS port of src_id/frontend.py (normalize + phonemize, Indonesian)
// Faithful port for browser TTS demo. Reproduces token_ids exactly for the
// demo sentences; throws FrontendError on unsupported input like Python.
'use strict';

const SPECIAL = ["<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":"];
const CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split(' ');
const VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split(' ');
const SYMBOLS = [...SPECIAL, ...CONSONANTS, ...VOWELS.flatMap(v => ['0','1','2'].map(s => v + s))];
const SYMBOL_TO_ID = Object.fromEntries(SYMBOLS.map((s, i) => [s, i]));
const _SYMBOL_SET = new Set(SYMBOLS);

class FrontendError extends Error {}
class UnknownWordError extends FrontendError {
  constructor(words) { super("out-of-vocabulary words: " + [...new Set(words)].sort().join(', ')); this.unknownWords = words; }
}

const _ONES = ["nol","satu","dua","tiga","empat","lima","enam","tujuh","delapan","sembilan","sepuluh","sebelas","dua belas","tiga belas","empat belas","lima belas","enam belas","tujuh belas","delapan belas","sembilan belas"];
const _TENS = ["", "", "dua puluh", "tiga puluh", "empat puluh", "lima puluh", "enam puluh", "tujuh puluh", "delapan puluh", "sembilan puluh"];
const _SCALES = [[1e12, "triliun"], [1e9, "miliar"], [1e6, "juta"], [1e3, "ribu"], [1e2, "ratus"]];
const _ORDINALS = {"satu":"pertama","dua":"kedua","tiga":"ketiga","empat":"keempat","lima":"kelima","enam":"keenam","tujuh":"ketujuh","delapan":"kedelapan","sembilan":"kesembilan"};
const _MONTHS = "Januari Februari Maret April Mei Juni Juli Agustus September Oktober November Desember".split(' ');
const _CURRENCY = {"rp":"rupiah","idr":"rupiah","$":"dolar","€":"euro","£":"pound"};
const _ABBREV = {"Mr":"Tuan","Mrs":"Nyonya","Dr":"Doktor","Prof":"Profesor"};

function cardinal(n) {
  if (n < 0) return "minus " + cardinal(-n);
  if (n < 20) return _ONES[n];
  if (n < 100) { const rest = n % 10; return _TENS[Math.floor(n/10)] + (rest ? " " + _ONES[rest] : ""); }
  for (const [scale, label] of _SCALES) {
    if (n >= scale) {
      if (n >= 1e15) return String(n).split('').map(d => _ONES[+d]).join(' ');
      if (Math.floor(n/scale) === 1 && (label === "ribu" || label === "ratus")) {
        return `se${label}` + (n % scale ? " " + cardinal(n % scale) : "");
      }
      return cardinal(Math.floor(n/scale)) + " " + label + (n % scale ? " " + cardinal(n % scale) : "");
    }
  }
  throw new Error("unreachable");
}
function ordinal(n) {
  if (n < 0) throw new FrontendError("ordinal of a negative number is unsupported");
  const words = cardinal(n).split(' ');
  if (words.length === 1 && words[0] in _ORDINALS) return _ORDINALS[words[0]];
  return "ke" + words.join(' ');
}
function _parseAmount(raw) {
  raw = raw.trim();
  if (raw.includes('.') && raw.includes(',')) {
    const decimal = raw.lastIndexOf('.') > raw.lastIndexOf(',') ? '.' : ',';
    const other = decimal === '.' ? ',' : '.';
    const [whole, frac] = raw.replaceAll(other, '').split(decimal);
    return [parseInt(whole, 10), frac];
  }
  if (raw.includes('.')) {
    if (/^\d{1,3}(\.\d{3})+$/.test(raw)) return [parseInt(raw.replaceAll('.', ''), 10), null];
    const [whole, frac] = raw.split('.'); return [parseInt(whole, 10), frac];
  }
  if (raw.includes(',')) {
    if (/^\d{1,3}(,\d{3})+$/.test(raw)) return [parseInt(raw.replaceAll(',', ''), 10), null];
    const [whole, frac] = raw.split(','); return [parseInt(whole, 10), frac];
  }
  return [parseInt(raw, 10), null];
}
function _decimalWords(raw) {
  const [whole, frac] = _parseAmount(raw);
  if (frac === null || frac === undefined) return cardinal(whole);
  return `${cardinal(whole)} koma ${frac.split('').map(c => _ONES[+c]).join(' ')}`;
}
function _money(symbol, raw) {
  const unit = _CURRENCY[symbol.toLowerCase()];
  const [whole, fraction] = _parseAmount(raw);
  let result = `${cardinal(whole)} ${unit}`;
  if (fraction) {
    const cents = parseInt((fraction + '00').slice(0, 2), 10);
    if (cents) result += ` ${cardinal(cents)} sen`;
  }
  return result;
}

const LEKSIKON = {
  "yang":["Y","AH0","NG"],"di":["D","IY1"],"ke":["K","EH1"],"dari":["D","AH0","R","IY1"],"dan":["D","AH0","N"],
  "atau":["AH0","T","AW1"],"ini":["IY1","N","IY0"],"itu":["IY1","T","UW0"],"adalah":["AH0","D","AH1","L","AH0"],
  "tidak":["T","IH0","D","AH1","K"],"bukan":["B","AH0","K","AH1","N"],"untuk":["AH0","N","T","AH1","K"],
  "dengan":["D","AH0","NG","AH1","N"],"pada":["P","AH0","D","AH1"],"dalam":["D","AH0","L","AH1","M"],
  "sebuah":["S","EH0","B","AH1","AH0"],"seorang":["S","EH0","OW1","R","AH0","NG"],"saya":["S","AH1","Y","AH0"],
  "anda":["AH0","N","D","AH1"],"kami":["K","AH1","M","IY0"],"kita":["K","IY1","T","AH0"],"mereka":["M","EH0","R","EH1","K","AH0"],
  "dia":["D","IY1","AH0"],"ia":["IY1","AH0"],"apa":["AH1","P","AH0"],"siapa":["S","IY1","AH0","P","AH0"],
  "kapan":["K","AH1","P","AH0","N"],"bagaimana":["B","AH0","G","AH1","M","AH0","N","AH0"],"mengapa":["M","EH0","NG","AH1","P","AH0"],
  "karena":["K","AH0","R","EH1","N","AH0"],"jika":["JH","IH1","K","AH0"],"kalau":["K","AH1","L","AW0"],
  "agar":["AH0","G","AH1","R"],"supaya":["S","AH0","P","AH1","Y","AH0"],"tetapi":["T","EH0","T","EH1","P","IY0"],
  "tapi":["T","AH1","P","IY0"],"namun":["N","AH1","M","AH0","N"],"sangat":["S","AH0","NG","AH1","T"],
  "lebih":["L","EH1","B","AH0"],"kurang":["K","AH0","R","AH1","NG"],"juga":["JH","AH1","G","AH0"],
  "sudah":["S","AH0","D","AH1"],"belum":["B","EH1","L","AH0","M"],"akan":["AH0","K","AH1","N"],
  "sedang":["S","EH1","D","AH0","NG"],"telah":["T","EH1","L","AH0"],"pernah":["P","EH0","R","AH1","N","AH0"],
  "bisa":["B","IY1","S","AH0"],"dapat":["D","AH1","P","AH0","T"],"harus":["HH","AH1","R","AH0","S"],
  "mau":["M","AW1"],"ingin":["AH0","NG","IH1","N"],"perlu":["P","EH0","R","L","UW1"],
  "mahasiswa":["M","AH0","HH","AH1","S","IH0","S","AH1","W","AH0"],
  "perusahaan":["P","EH0","R","AH0","S","AH1","HH","AH0","N","AH0"],
  "pemerintah":["P","EH0","M","EH1","R","IH0","N","T","AH0"],
  "indonesia":["IH0","N","D","AH1","OW0","S","IY0","AH0"],
  "jakarta":["JH","AH0","K","AH1","R","AH0","T","AH0"],
  "surat":["S","AH0","R","AH1","T"],"kirim":["K","IY1","R","IH0","M"],"harga":["HH","AH0","R","G","AH1"],
  "mahal":["M","AH0","HH","AH1","L"],"murah":["M","AH0","R","AH1","HH"],"besar":["B","EH1","S","AH0","R"],
  "kecil":["K","EH1","CH","IH0","L"],"baik":["B","AH1","IY0","K"],"terima":["T","EH0","R","IY1","M","AH0"],
  "kasih":["K","AH1","S","IH0","HH"],"tolong":["T","OW1","L","OW0","NG"],"maaf":["M","AH1","AH0","F"],
  "halo":["HH","AH1","L","OW0"],"selamat":["S","EH0","L","AH1","M","AH0","T"],"pagi":["P","AH1","G","IY0"],
  "malam":["M","AH1","L","AH0","M"],"hari":["HH","AH1","R","IY0"],"minggu":["M","IH0","NG","G","AH1","UW0"],
  "bulan":["B","AH1","L","AH0","N"],"tahun":["T","AH1","HH","AH0","N"],
};
const _DIGRAPHS = {"ng":"NG","ny":"Y","kh":"K","sy":"SH","ts":"CH"};
const _SINGLE = {"b":"B","d":"D","f":"F","g":"G","h":"HH","j":"JH","k":"K","l":"L","m":"M","n":"N","p":"P","q":"K","r":"R","s":"S","t":"T","v":"V","w":"W","x":"K","y":"Y","z":"Z"};
const _VOWEL_LETTERS = {"a":"AH","e":"EH","i":"IH","o":"OW","u":"UW"};
const SCHWA_WORDS = new Set(["empat","besok","enam","dengan","mereka","tentang","sebuah","seorang","kepada","menemukan","bertemu"]);

function _isReduplikasi(token) {
  const parts = token.split('-').map(p => p.toLowerCase());
  if (parts.length < 2) return false;
  if (parts[parts.length - 1] === "nya") return false;
  if (parts.some(p => p.length === 1)) return false;
  if (token === token.toUpperCase() && token !== token.toLowerCase()) return false;
  if (new Set(parts).size === 1) return true;
  for (const a of parts) for (const b of parts) if (a !== b && (b.includes(a) || a.includes(b))) return true;
  for (let i = 0; i < parts.length; i++) for (let j = i + 1; j < parts.length; j++) {
    const a = parts[i], b = parts[j];
    for (let k = Math.min(a.length, b.length); k >= 4; k--) {
      if (a.slice(-k) === b.slice(0, k) || b.slice(-k) === a.slice(0, k)) return true;
    }
  }
  return parts.every(p => p.length >= 4);
}

function normalize(text) {
  if (typeof text !== 'string') throw new FrontendError("text must be a string");
  const log = [];
  const original = text;
  text = text.normalize('NFKC');
  for (const [src, dst] of [["’","'"],["‘","'"],["“",'"'],["”",'"'],["—",","],["–",","],["…","..."]]) {
    if (text.includes(src)) { log.push(`punct: '${src}' -> '${dst}'`); text = text.split(src).join(dst); }
  }
  text = text.replace(/\b(\d{4})-(\d{2})-(\d{2})\b/g, (_, y, m, d) => {
    const out = `${_MONTHS[+m - 1]} ${ordinal(+d)}, ${cardinal(+y)}`;
    log.push(`date: '${y}-${m}-${d}' -> '${out}'`); return out;
  });
  if (/\d\/\d/.test(text)) throw new FrontendError("write slash dates/fractions in words, or use an ISO date (YYYY-MM-DD)");
  for (const [abbr, expansion] of Object.entries(_ABBREV)) {
    const re = new RegExp(`\\b${abbr}\\.(?=\\s|$)`, 'gi');
    const next = text.replace(re, expansion);
    if (next !== text) { log.push(`abbrev: ${abbr}. -> ${expansion}`); text = next; }
  }
  text = text.replace(/(Rp|\$|€|£|IDR)\s?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+)(?!\d)(?![A-Za-z])/gi,
    (m, sym, raw) => { const out = _money(sym, raw); log.push(`currency: '${m}' -> '${out}'`); return out; });
  if (/\b(?:Rp|IDR)\b/i.test(text)) throw new FrontendError("currency symbol 'Rp'/'IDR' without an amount; write the amount explicitly (e.g. 'Rp25.000')");
  for (const [unit, words] of [["kVA","kilovolt ampere"],["kV","kilovolt"],["VA","volt ampere"],["kWh","kilowatt jam"],["kW","kilowatt"],["W","watt"]]) {
    const re = new RegExp(`(?<=\\d)\\s*${unit}\\b`, 'gi');
    const next = text.replace(re, ` ${words}`);
    if (next !== text) { log.push(`unit: ${unit} -> ${words}`); text = next; }
  }
  { const next = text.replace(/\bpcs\b/gi, "buah"); if (next !== text) { log.push("word: 'pcs' -> 'buah'"); text = next; } }
  const letterNames = {A:"a",B:"be",C:"ce",D:"de",E:"e",F:"ef",G:"ge",H:"ha",I:"i",J:"je",K:"ka",L:"el",M:"em",N:"en",O:"o",P:"pe",Q:"ki",R:"er",S:"es",T:"te",U:"u",V:"ve",W:"we",X:"eks",Y:"ye",Z:"zet"};
  const _spell = tok => tok.toUpperCase().split('').filter(c => /[A-Z]/.test(c)).map(c => letterNames[c]).join(' ');
  const _acronymLogged = (m, g0) => { const out = _spell(g0); log.push(`acronym: '${g0}' -> '${out}'`); return out; };
  text = text.replace(/\b([A-Za-z])\s*-\s*(\d+)\b/g, (m, a, d) => {
    const out = `${letterNames[a.toUpperCase()]} ${d}`; log.push(`code: '${m}' -> '${out}'`); return out; });
  text = text.replace(/\b([A-Za-z])(\d+)\b/g, (m, a, d) => {
    const out = `${letterNames[a.toUpperCase()]} ${d}`; log.push(`code: '${m}' -> '${out}'`); return out; });
  for (const acr of ["NIB","KBLI","NPWP","KTP","PLN","MCB"]) {
    const re = new RegExp(`\\b${acr}\\b`, 'gi');
    text = text.replace(re, (m) => { const out = _spell(m); log.push(`acronym: '${m}' -> '${out}'`); return out; });
  }
  text = text.replace(/\b[A-Za-z]+(?:-[A-Za-z]+)+\b/g, (m) => {
    if (_isReduplikasi(m)) { const out = m.replaceAll('-', ', '); log.push(`reduplikasi: '${m}' -> '${out}'`); return out; }
    const parts = m.split('-');
    if (m === m.toUpperCase() || parts.every(p => p.length <= 4)) { const out = _spell(m); log.push(`acronym: '${m}' -> '${out}'`); return out; }
    const out = m.replaceAll('-', ' '); log.push(`hyphen-word: '${m}' -> '${out}'`); return out;
  });
  text = text.replace(/\b[A-Z]{2,}\b/g, (m) => { const out = _spell(m); log.push(`acronym: '${m}' -> '${out}'`); return out; });
  text = text.replace(/\b(\d+)(?:st|nd|rd|th)\b/gi, (m, d) => { const out = ordinal(parseInt(d, 10)); log.push(`ordinal: '${m}' -> '${out}'`); return out; });
  text = text.replace(/-?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+,\d+|\d+)(?![\dA-Za-z])/g,
    (m) => { const out = ((m.startsWith('-') ? 'minus ' : '') + _decimalWords(m.replace(/^-/, ''))); log.push(`number: '${m}' -> '${out}'`); return out; });
  for (const [ch, word] of [["%", " persen "], ["&", " dan "], ["+", " tambah "]]) {
    if (text.includes(ch)) { log.push(`symbol: '${ch}' -> '${word.trim()}'`); text = text.split(ch).join(word); }
  }
  const caps = (text.match(/\b[A-Z]*\d[A-Z0-9]*\b|\b[A-Z]{2,}\b/g) || []).filter(c => /\d/.test(c));
  if (caps.length) throw new FrontendError("unsupported code/acronym (no expansion rule): " + [...new Set(caps)].sort().join(', ') + "; write it out or supply per-word overrides");
  if (text.includes('-')) log.push("hyphen: '-' -> ' '");
  text = text.replaceAll('-', ' ');
  text = text.replace(/[()\[\]{}"/]/g, ' ');
  const unsupported = (text.match(/[^a-zA-Z\s'.,?!;:]/g) || []);
  if (unsupported.length) throw new FrontendError(`unsupported text characters after normalization: ${[...new Set(unsupported)].sort()}`);
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (normalized !== original) log.push(`normalized: '${original}' -> '${normalized}'`);
  return { normalizedText: normalized, mappingLog: log };
}

function _g2pWord(word) {
  const phones = [];
  let i = 0, vowelSeen = false;
  while (i < word.length) {
    const chunk = word.slice(i, i + 2);
    if (chunk in _DIGRAPHS) { phones.push(_DIGRAPHS[chunk]); i += 2; continue; }
    const ch = word[i];
    if (ch in _VOWEL_LETTERS) {
      const stress = vowelSeen ? "0" : "1"; vowelSeen = true;
      phones.push(_VOWEL_LETTERS[ch] + stress); i += 1; continue;
    }
    if (ch === 'c') { phones.push("CH"); i += 1; continue; }
    if (!(ch in _SINGLE)) return null;
    phones.push(_SINGLE[ch]); i += 1;
  }
  if (phones.some(p => !_SYMBOL_SET.has(p))) return null;
  return phones;
}

function phonemize(text, { strict = true } = {}) {
  const norm = normalize(text);
  const tokens = ["<bos>"];
  const unknown = [];
  const items = norm.normalizedText.match(/[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]/g) || [];
  for (const item of items) {
    if (SPECIAL.includes(item)) { tokens.push(item); continue; }
    const word = item.toLowerCase();
    let phones = null;
    if (word in LEKSIKON) phones = LEKSIKON[word];
    else phones = _g2pWord(word.replace(/['-]/g, ''));
    if (phones === null) { unknown.push(word); continue; }
    if (tokens[tokens.length - 1] !== "<bos>" && tokens[tokens.length - 1] !== "<space>") tokens.push("<space>");
    for (const p of phones) tokens.push(p);
  }
  if (unknown.length && strict) throw new UnknownWordError(unknown);
  tokens.push("<eos>");
  return {
    normalizedText: norm.normalizedText,
    symbols: tokens,
    tokenIds: tokens.map(s => SYMBOL_TO_ID[s]),
    unknownWords: unknown,
    mappingLog: norm.mappingLog,
  };
}

// Browser + Node/ESM compat: export named if available, else attach to window/global.
const api = { SYMBOLS, SYMBOL_TO_ID, FrontendError, UnknownWordError, normalize, phonemize, cardinal, ordinal };
if (typeof module !== 'undefined' && module.exports) module.exports = api;
if (typeof window !== 'undefined') Object.assign(window, api);
export { SYMBOLS, SYMBOL_TO_ID, FrontendError, UnknownWordError, normalize, phonemize, cardinal, ordinal };
