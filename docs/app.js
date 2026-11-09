// app.js — interactive demo UI: type text, pick voice, generate live in browser.
// Imports ./frontend.js (normalize+phonemize) and ./pivo.js (ONNX inference).
import { phonemize } from './frontend.js';
import { loadModels, synthesize, pcmToWavBlob } from './pivo.js';

const $ = (s) => document.querySelector(s);
const els = {
  text: $('#text'), voice: $('#voice'), deemph: $('#deemph'),
  btn: $('#generate'), ratio: $('#token-ratio'), output: $('#output'),
  status: $('#status'), tokens: $('#tokens'), frames: $('#frames'),
  normalized: $('#normalized'),
};

async function load() {
  els.status.textContent = 'Memuat model ONNX (5,6 MB)…';
  try {
    await loadModels({ onLog: (m) => { els.status.textContent = m; } });
    els.status.textContent = 'Model siap. Ketik kalimat, lalu klik Generate.';
    els.btn.disabled = false;
  } catch (e) {
    els.status.textContent = 'Gagal memuat model: ' + e.message + ' — coba refresh (model di-load dari docs/models).';
    els.output.innerHTML = '<p class="err">Tidak bisa load ONNX. Pastikan online (CDN onnxruntime-web) + file di <code>docs/models/</code>.</p>';
  }
}

async function onGenerate() {
  const text = els.text.value.trim();
  if (!text) { els.status.textContent = 'Tulis dulu kalimatnya.'; return; }
  els.btn.disabled = true;
  els.output.innerHTML = '';
  try {
    const pr = phonemize(text);
    els.tokens.textContent = `${pr.tokenIds.length} tokens`;
    els.frames.textContent = `norm: “${pr.normalizedText}”`;
    els.normalized.textContent = pr.mappingLog.join('\n');
    els.status.textContent = 'Menjalankan encoder + decoder + vocoder… (beberapa detik pertama kali)';
    const voice = parseInt(els.voice.value, 10); // 0 Pi, 1 Vo
    const coef = parseFloat(els.deemph.value || '0.88');
    const { pcm, frames } = await synthesize(pr.tokenIds, voice, {
      deemphasisCoef: coef,
      onProgress: (p) => { els.status.textContent = `Synthesizing… ${Math.round(p * 100)}%`; },
    });
    const blob = pcmToWavBlob(pcm);
    const url = URL.createObjectURL(blob);
    els.frames.textContent = `${frames} frame (${(frames / 100).toFixed(2)}s @16kHz) · ${pr.tokenIds.length} tokens`;
    els.output.innerHTML = `<audio controls autoplay src="${url}"></audio><p><a href="${url}" download="pivo-tts.wav">Download WAV</a></p>`;
    els.status.textContent = 'Selesai — audio di atas diputar otomatis.';
    // perkiraan token ratio (frontend->frames headroom)
    els.ratio.textContent = `${pr.tokenIds.length} token → ${frames} frame (${(frames / Math.max(1, pr.tokenIds.length)).toFixed(1)} frame/token)`;
  } catch (e) {
    els.output.innerHTML = `<p class="err">Gagal: ${e.message}</p>`;
    els.status.textContent = 'Gagal — lihat pesan di atas / console.';
    console.error(e);
  } finally {
    els.btn.disabled = false;
  }
}

els.btn.addEventListener('click', onGenerate);
load();
