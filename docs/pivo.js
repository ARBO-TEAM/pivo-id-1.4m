// pivo.js — browser inference for PiVo-ID v11 (encoder + decoder + conditioning + vocoder_step)
// Faithful port of PiVo-1.4M/pivo/runtime.py + src_id/acoustic.py phase_features.
// No retraining; runs entirely in the browser via onnxruntime-web.
'use strict';

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.0/dist/ort.wasm.min.js';

const MODEL_URLS = {
  encoder: 'models/encoder.onnx',
  decoder: 'models/decoder.onnx',
  conditioning: 'models/conditioning.onnx',
  vocoder_step: 'models/vocoder_step.onnx',
};
const CHA = 16; // decoder chunk like Python chunk_frames=16; chunk value does not change math
let metadata = null;
let sessions = null;

async function loadModels({ onLog } = {}) {
  if (sessions) return;
  metadata = await (await fetch('models/metadata.json')).json();
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.0/dist/';
  const opts = { executionProviders: ['wasm'], intraOpNumThreads: Math.min(4, (navigator.hardwareConcurrency || 2)) };
  const log = onLog || (() => {});
  sessions = {};
  for (const [name, url] of Object.entries(MODEL_URLS)) {
    log(`Memuat ${name}…`);
    sessions[name] = await ort.InferenceSession.create(url, opts);
  }
  log('Model siap.');
}

function periods(features) {
  // features Float32Array [3,20] -> Int64Array [3]: 256 / 2^(clip(f18,-10,10)+1.5), clip 32..255
  const out = [];
  for (let c = 0; c < 3; c++) {
    let p = 256 / Math.pow(2, Math.max(-10, Math.min(10, features[c * 20 + 18])) + 1.5);
    out.push(Math.max(32, Math.min(255, Math.round(p))));
  }
  return [out.map(BigInt)];
}
function phaseFeatures(durations, maxTokenFrames) {
  // durations Int32Array -> Float32Array [F,4], port of pivo/phase.py
  const counts = [];
  for (let i = 0; i < durations.length; i++) for (let k = 0; k < durations[i]; k++) counts.push(durations[i]);
  const F = counts.length;
  const starts = [];
  for (let i = 0; i < durations.length; i++) for (let k = 0; k < durations[i]; k++) starts.push(0); // recomputed below
  let cum = 0; const startArr = [];
  for (let i = 0; i < durations.length; i++) { for (let k = 0; k < durations[i]; k++) startArr.push(cum); cum += durations[i]; }
  const out = new Float32Array(F * 4);
  for (let i = 0; i < F; i++) {
    const frac = (i - startArr[i] + 0.5) / counts[i];
    out[i * 4] = 2 * frac - 1;
    out[i * 4 + 1] = Math.sin(Math.PI * frac);
    out[i * 4 + 2] = Math.cos(Math.PI * frac);
    out[i * 4 + 3] = Math.log1p(counts[i]) / Math.log1p(maxTokenFrames);
  }
  return out;
}
function deemphasisStep(x, state, coef = 0.85) {
  const out = new Float32Array(x.length);
  let s = state;
  for (let i = 0; i < x.length; i++) { s = x[i] + coef * s; out[i] = s; }
  return [out, s];
}
function refineSelective(feats) {
  // Port of render_student.refine_pivo_features (deemphasis 0.88 path uses selective? No — default demo = raw+0.88)
  return feats;
}

export async function synthesize(tokenIds, voice = 0, { speed = 1.0, deemphasisCoef = 0.88, onProgress } = {}) {
  if (!sessions) await loadModels();
  const c = metadata.config;
  const tokens = new BigInt64Array([tokenIds.length]);
  for (let i = 0; i < tokenIds.length; i++) tokens[i] = BigInt(tokenIds[i]);
  const voiceArr = [BigInt(voice)];
  const lengths = [BigInt(tokenIds.length)];

  const enc = await sessions.encoder.run({
    tokens: new ort.Tensor('int64', tokens, [1, tokenIds.length]),
    voice: new ort.Tensor('int64', BigInt64Array.from(voiceArr), [1]),
    lengths: new ort.Tensor('int64', BigInt64Array.from(lengths), [1]),
  });
  const encoded = enc.encoded; // Float32Tensor [1,T,96]
  const logDur = enc.log_durations; // [1,T]
  const T = tokenIds.length;
  // durations: expm1(clip(log,0,log1p(200)))/speed, clip 0..200, max with (tok>=10), zero for tok<=2
  const dur = new Int32Array(T);
  let total = 0;
  for (let i = 0; i < T; i++) {
    let d = Math.expm1(Math.max(0, Math.min(Math.log1p(c.max_token_frames), logDur.data[i]))) / speed;
    d = Math.round(d);
    d = Math.max(0, Math.min(c.max_token_frames, d));
    if (tokenIds[i] >= 10) d = Math.max(d, 1);
    if (tokenIds[i] <= 2) d = 0;
    dur[i] = d; total += d;
  }
  if (total > c.max_utterance_frames) throw new Error('teks melebihi budget 3000 frame, potong jadi klausa');
  if (total === 0) throw new Error('tidak ada frame durasi (teks kosong?)');

  // expanded [1, F, 100]: repeat encoded by dur + phase
  const expData = new Float32Array(total * 100);
  let write = 0;
  for (let i = 0; i < T; i++) {
    for (let k = 0; k < dur[i]; k++) {
      for (let w = 0; w < 96; w++) expData[write * 100 + w] = encoded.data[i * 96 + w];
      write++;
    }
  }
  const phase = phaseFeatures(dur, c.max_token_frames);
  for (let f = 0; f < total; f++) for (let w = 0; w < 4; w++) expData[f * 100 + 96 + w] = phase[f * 4 + w];
  const chunk = 16;
  // decoder with KV cache
  let cache = {
    key0: new ort.Tensor('float32', new Float32Array(1 * 2 * 0 * 24), [1, 2, 0, 24]),
    value0: new ort.Tensor('float32', new Float32Array(1 * 2 * 0 * 24), [1, 2, 0, 24]),
    key1: new ort.Tensor('float32', new Float32Array(1 * 2 * 0 * 24), [1, 2, 0, 24]),
    value1: new ort.Tensor('float32', new Float32Array(1 * 2 * 0 * 24), [1, 2, 0, 24]),
  };
  const allFeats = [];
  for (let pos = 0; pos < total; pos += chunk) {
    const slice = expData.slice(pos * 100, Math.min(total, pos + chunk) * 100);
    const nFr = Math.min(chunk, total - pos);
    const res = await sessions.decoder.run({
      expanded: new ort.Tensor('float32', slice, [1, nFr, 100]),
      position: new ort.Tensor('int64', BigInt64Array.from([BigInt(pos)]), []),
      ...cache,
    });
    cache = { key0: res.key0_out, value0: res.value0_out, key1: res.key1_out, value1: res.value1_out };
    allFeats.push(res.features.data); // Float32Array [nFr,20]
    if (onProgress) onProgress(pos / total);
  }
  const feats = new Float32Array(total * 20);
  let off = 0;
  for (const a of allFeats) { feats.set(a, off); off += a.length; }

  // vocoder streaming 1 frame at a time (same as Runtime.iter_audio)
  const pad = (n) => new Float32Array(1 * n).fill(0);
  const state = {
    memory: new ort.Tensor('float32', pad(256), [1, 256]),
    gru1: new ort.Tensor('float32', pad(160), [1, 160]),
    gru2: new ort.Tensor('float32', pad(128), [1, 128]),
    gru3: new ort.Tensor('float32', pad(128), [1, 128]),
    frame_state: new ort.Tensor('float32', pad(164), [1, 164]),
  };
  const pcm = new Float32Array(total * 160);
  let deState = 0;
  const queue = [];
  for (let f = 0; f < total; f++) {
    const frame20 = feats.slice(f * 20, f * 20 + 20);
    queue.push(frame20);
    // emit when queue>=2 or last frame
    while (queue.length >= 2 || (f === total - 1 && queue.length)) {
      const cur = queue.shift();
      const foll = queue.length ? queue[0] : cur;
      const prev = f === 0 && queue.length === total ? cur : (queue.length ? allFrame(feats, f - 1) : cur);
      // simpler: previous/current/following from feats directly
      const ctx = new Float32Array(3 * 20);
      const idxPrev = Math.max(0, f - 1), idxCur = f, idxFol = Math.min(total - 1, f + 1);
      ctx.set(feats.slice(idxPrev * 20, idxPrev * 20 + 20), 0);
      ctx.set(feats.slice(idxCur * 20, idxCur * 20 + 20), 20);
      ctx.set(feats.slice(idxFol * 20, idxFol * 20 + 20), 40);
      const per = periods(ctx);
      const condRes = await sessions.conditioning.run({
        features: new ort.Tensor('float32', ctx, [1, 3, 20]),
        periods: new ort.Tensor('int64', BigInt64Array.from(per[0]), [1, 3]),
      });
      const cond = condRes.conditioning.data; // [1,1,320]
      const period1 = per[0][1];
      const frame160 = [];
      for (let sub = 0; sub < 4; sub++) {
        const c80 = cond.slice(sub * 80, (sub + 1) * 80);
        const r = await sessions.vocoder_step.run({
          conditioning: new ort.Tensor('float32', c80, [1, 80]),
          period: new ort.Tensor('int64', BigInt64Array.from([period1]), [1]),
          memory: state.memory, gru1: state.gru1, gru2: state.gru2, gru3: state.gru3, frame_state: state.frame_state,
        });
        state = { memory: r.memory_out, gru1: r.gru1_out, gru2: r.gru2_out, gru3: r.gru3_out, frame_state: r.frame_state_out };
        frame160.push(r.excitation.data);
      }
      const concat = new Float32Array(160);
      concat.set(frame160[0], 0); concat.set(frame160[1], 40); concat.set(frame160[2], 80); concat.set(frame160[3], 120);
      const [dec, newState] = deemphasisStep(concat, deState, deemphasisCoef);
      deState = newState;
      pcm.set(dec, f * 160);
      if (onProgress) onProgress(0.6 + 0.4 * (f / total));
    }
  }
  // clip peak like python write_wav16 does not hard-clip before write except write path clips. Keep raw, clip at encode.
  return { pcm, frames: total };
}
function allFrame(feats, i) { return feats.slice(i * 20, i * 20 + 20); }

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
export { loadModels };
