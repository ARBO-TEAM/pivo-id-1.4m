// app.js — interactive demo UI: type text, pick voice, generate live in browser.
// Model load shows real download progress (4 ONNX files, ~5.6 MB total).
import { phonemize } from './frontend.js';
import { loadModels, synthesize, pcmToWavBlob } from './pivo.js';

const $ = (s) => document.querySelector(s);
const els = {
  text: $('#text'), voice: $('#voice'), deemph: $('#deemph'),
  btn: $('#generate'), ratio: $('#token-ratio'), output: $('#output'),
  status: $('#status'), bar: $('#bar'), fill: $('#bar-fill'),
  tokens: $('#tokens'), frames: $('#frames'), normalized: $('#normalized'),
};

function showProgress(frac, label) {
  els.bar.hidden = false;
  els.fill.style.width = `${Math.round(Math.min(1, Math.max(0, frac)) * 100)}%`;
  if (label) els.status.textContent = label;
}

async function load() {
  els.btn.disabled = true;
  els.btn.textContent = '⏳ Memuat model…';
  try {
    await loadModels({
      onProgress: showProgress,
    });
    els.status.textContent = 'Model siap. Ketik kalimat, lalu klik Generate.';
    els.bar.hidden = true;
    els.btn.disabled = false;
    els.btn.textContent = '🔊 Generate';
  } catch (e) {
    console.error('[PiVo] model load failed', e);
    els.status.textContent = 'Gagal memuat model: ' + e.message;
    els.btn.textContent = '🔁 Coba lagi';
    els.btn.disabled = false;
    els.btn.onclick = () => location.reload();
  }
}

async function onGenerate() {
  const text = els.text.value.trim();
  if (!text) { els.status.textContent = 'Tulis dulu kalimatnya.'; return; }
  els.btn.disabled = true;
  els.btn.textContent = '⏳ Generate…';
  els.output.innerHTML = '';
  try {
    const pr = phonemize(text);
    els.tokens.textContent = `${pr.tokenIds.length} token`;
    els.normalized.textContent = pr.mappingLog.join('\n');
    const voice = parseInt(els.voice.value, 10);
    const coef = parseFloat(els.deemph.value || '0.88');
    const { pcm, frames, seconds } = await synthesize(pr.tokenIds, voice, {
      deemphasisCoef: coef,
      onProgress: showProgress,
    });
    const url = URL.createObjectURL(pcmToWavBlob(pcm));
    els.frames.textContent = `${frames} frame · ${seconds.toFixed(2)} s @16 kHz · norm: “${pr.normalizedText}”`;
    els.ratio.textContent = `${(frames / pr.tokenIds.length).toFixed(1)} frame/token`;
    els.output.innerHTML = `<audio controls autoplay src="${url}"></audio><p><a href="${url}" download="pivo-tts.wav">⬇ Download WAV</a></p>`;
    els.status.textContent = 'Selesai.';
    els.bar.hidden = true;
  } catch (e) {
    console.error(e);
    els.output.innerHTML = `<p class="err">Gagal: ${e.message}</p>`;
    els.status.textContent = 'Gagal — lihat pesan di atas.';
    els.bar.hidden = true;
  } finally {
    els.btn.disabled = false;
    els.btn.textContent = '🔊 Generate';
  }
}

els.btn.addEventListener('click', onGenerate);
load();
