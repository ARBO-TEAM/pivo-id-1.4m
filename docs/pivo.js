// pivo.js — browser inference for PiVo-ID v11 (encoder + decoder + conditioning + vocoder_step)
// Faithful port of PiVo-1.4M/pivo/runtime.py + src_id/acoustic.py phase_features.
// Runs entirely in the browser via onnxruntime-web (wasm). No server, no PyTorch.
'use strict';

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.wasm.min.mjs';

const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
const MODEL_URLS = {
  encoder: { url: 'models/encoder.onnx', bytes: 1783481 },
  decoder: { url: 'models/decoder.onnx', bytes: 721228 },
  conditioning: { url: 'models/conditioning.onnx', bytes: 282407 },
  vocoder_step: { url: 'models/vocoder_step.onnx', bytes: 3001580 },
};
const TOTAL_BYTES = Object.values(MODEL_URLS).reduce((a, m) => a + m.bytes, 0);

let metadata = null;
let sessions = null;
let loadPromise = null;

async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get('Content-Length')) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (onBytes) onBytes(got, total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

/** Load the four ONNX graphs. onProgress(p 0..1, label). Cached across calls. */
export function loadModels({ onProgress, onLog } = {}) {
  if (loadPromise) return loadPromise;
  const log = onLog || (() => {});
  const prog = onProgress || (() => {});
  loadPromise = (async () => {
    prog(0, 'Mengambil metadata…');
    metadata = await (await fetch('models/metadata.json')).json();
    ort.env.wasm.wasmPaths = ORT_DIST;
    // GitHub Pages is not cross-origin isolated: threaded wasm would fail, pin 1.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    let done = 0;
    sessions = {};
    for (const [name, m] of Object.entries(MODEL_URLS)) {
      log(`Mengunduh ${name} (${Math.round(m.bytes / 1024)} KB)…`);
      prog(done / TOTAL_BYTES, `Mengunduh ${name}…`);
      const buf = await fetchWithProgress(m.url, (got, total) => {
        const frac = Math.min(1, got / (total || m.bytes));
        prog((done + m.bytes * frac) / TOTAL_BYTES, `Mengunduh ${name} ${Math.round(frac * 100)}%`);
      });
      done += m.bytes;
      prog(done / TOTAL_BYTES, `Menyiapkan ${name}…`);
      sessions[name] = await ort.InferenceSession.create(buf, opts);
    }
    prog(1, 'Model siap.');
  })().catch((e) => { loadPromise = null; throw e; });
  return loadPromise;
}

function periods(features) {
  // features Float32Array [3*20] -> BigInt64Array [3]
  const out = new BigInt64Array(3);
  for (let c = 0; c < 3; c++) {
    const p = 256 / Math.pow(2, Math.max(-10, Math.min(10, features[c * 20 + 18])) + 1.5);
    out[c] = BigInt(Math.max(32, Math.min(255, Math.round(p))));
  }
  return out;
}

function phaseFeatures(durations, maxTokenFrames) {
  // durations Int32Array -> Float32Array [F*4], port of PiVo-1.4M/pivo/phase.py
  let F = 0;
  for (let i = 0; i < durations.length; i++) F += durations[i];
  const counts = new Float32Array(F);
  const starts = new Float32Array(F);
  let idx = 0;
  for (let i = 0; i < durations.length; i++) {
    const d = durations[i];
    for (let k = 0; k < d; k++) { counts[idx] = d; starts[idx] = idx - k; idx++; }
  }
  const out = new Float32Array(F * 4);
  for (let i = 0; i < F; i++) {
    const frac = (i - starts[i] + 0.5) / counts[i];
    out[i * 4] = 2 * frac - 1;
    out[i * 4 + 1] = Math.sin(Math.PI * frac);
    out[i * 4 + 2] = Math.cos(Math.PI * frac);
    out[i * 4 + 3] = Math.log1p(counts[i]) / Math.log1p(maxTokenFrames);
  }
  return out;
}

function deemphasis(x, state, coef) {
  const out = new Float32Array(x.length);
  let s = state;
  for (let i = 0; i < x.length; i++) { s = x[i] + coef * s; out[i] = s; }
  return [out, s];
}

/**
 * tokenIds -> { pcm: Float32Array, frames, seconds }.
 * Mirrors Runtime.iter_audio: decoder KV cache in 16-frame chunks, vocoder
 * streaming one 10 ms frame at a time with persistent state + deemphasis.
 */
export async function synthesize(tokenIds, voice = 0, { speed = 1.0, deemphasisCoef = 0.88, onProgress } = {}) {
  if (!sessions) await loadModels();
  const c = metadata.config;
  const T = tokenIds.length;
  const prog = onProgress || (() => {});

  const enc = await sessions.encoder.run({
    tokens: new ort.Tensor('int64', BigInt64Array.from(tokenIds.map(BigInt)), [1, T]),
    voice: new ort.Tensor('int64', BigInt64Array.from([BigInt(voice)]), [1]),
    lengths: new ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1]),
  });
  const encoded = enc.encoded.data;  // Float32Array [T*96]
  const logDur = enc.log_durations.data; // Float32Array [T]

  const dur = new Int32Array(T);
  let total = 0;
  const maxTok = Math.log1p(c.max_token_frames);
  for (let i = 0; i < T; i++) {
    let d = Math.round(Math.expm1(Math.max(0, Math.min(maxTok, logDur[i]))) / speed);
    d = Math.max(0, Math.min(c.max_token_frames, d));
    if (tokenIds[i] >= 10) d = Math.max(d, 1);
    if (tokenIds[i] <= 2) d = 0;
    dur[i] = d;
    total += d;
  }
  if (total > c.max_utterance_frames) throw new Error('teks melebihi batas 3000 frame — potong jadi klausa');
  if (total === 0) throw new Error('tidak ada frame (teks kosong?)');

  // expanded [1, F, 100] = encoder vector repeated by duration + 4 phase features
  const expData = new Float32Array(total * 100);
  let f = 0;
  for (let i = 0; i < T; i++) {
    for (let k = 0; k < dur[i]; k++, f++) expData.set(encoded.subarray(i * 96, i * 96 + 96), f * 100);
  }
  const phase = phaseFeatures(dur, c.max_token_frames);
  for (let i = 0; i < total; i++) expData.set(phase.subarray(i * 4, i * 4 + 4), i * 100 + 96);

  // decoder (causal, window 64) with KV cache
  const chunk = 16;
  const empty = () => new ort.Tensor('float32', new Float32Array(0), [1, 2, 0, 24]);
  let cache = { key0: empty(), value0: empty(), key1: empty(), value1: empty() };
  const feats = new Float32Array(total * 20);
  for (let pos = 0; pos < total; pos += chunk) {
    const n = Math.min(chunk, total - pos);
    const res = await sessions.decoder.run({
      expanded: new ort.Tensor('float32', expData.slice(pos * 100, (pos + n) * 100), [1, n, 100]),
      position: new ort.Tensor('int64', BigInt64Array.from([BigInt(pos)]), []),
      ...cache,
    });
    cache = { key0: res.key0_out, value0: res.value0_out, key1: res.key1_out, value1: res.value1_out };
    feats.set(res.features.data, pos * 20);
    prog(0.35 * (pos + n) / total, 'Decoder…');
  }

  // vocoder: streaming 1 frame, context = [prev, cur, next] (edge-clamped)
  const state = {
    memory: new ort.Tensor('float32', new Float32Array(256), [1, 256]),
    gru1: new ort.Tensor('float32', new Float32Array(160), [1, 160]),
    gru2: new ort.Tensor('float32', new Float32Array(128), [1, 128]),
    gru3: new ort.Tensor('float32', new Float32Array(128), [1, 128]),
    frame_state: new ort.Tensor('float32', new Float32Array(164), [1, 164]),
  };
  const pcm = new Float32Array(total * 160);
  let deState = 0;
  const ctx = new Float32Array(60);
  for (let i = 0; i < total; i++) {
    const ip = Math.max(0, i - 1) * 20, ic = i * 20, ifo = Math.min(total - 1, i + 1) * 20;
    ctx.set(feats.subarray(ip, ip + 20), 0);
    ctx.set(feats.subarray(ic, ic + 20), 20);
    ctx.set(feats.subarray(ifo, ifo + 20), 40);
    const per = periods(ctx);
    const cond = (await sessions.conditioning.run({
      features: new ort.Tensor('float32', ctx, [1, 3, 20]),
      periods: new ort.Tensor('int64', per, [1, 3]),
    })).conditioning.data; // Float32Array [320]
    const period1 = new ort.Tensor('int64', BigInt64Array.from([per[1]]), [1]);
    const frame = new Float32Array(160);
    for (let sub = 0; sub < 4; sub++) {
      const r = await sessions.vocoder_step.run({
        conditioning: new ort.Tensor('float32', cond.slice(sub * 80, (sub + 1) * 80), [1, 80]),
        period: period1,
        memory: state.memory, gru1: state.gru1, gru2: state.gru2, gru3: state.gru3, frame_state: state.frame_state,
      });
      state.memory = r.memory_out;
      state.gru1 = r.gru1_out;
      state.gru2 = r.gru2_out;
      state.gru3 = r.gru3_out;
      state.frame_state = r.frame_state_out;
      frame.set(r.excitation.data, sub * 40);
    }
    const [out, next] = deemphasis(frame, deState, deemphasisCoef);
    deState = next;
    pcm.set(out, i * 160);
    if ((i & 15) === 0) prog(0.35 + 0.65 * i / total, 'Vocoder…');
    await new Promise((res) => setTimeout(res, 0)); // keep UI responsive
  }
  prog(1, 'Selesai.');
  return { pcm, frames: total, seconds: total / 100 };
}

export function pcmToWavBlob(pcm, sampleRate = 16000) {
  const n = pcm.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); ws(8, 'WAVE'); ws(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ws(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    v.setInt16(44 + i * 2, Math.round(s * 32767), true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}
