"""Compare the PyTorch acoustic model with the official ONNX encoder and decoder.

Run from the workspace root:
    python -I tests/test_parity.py --model-dir PiVo-1.4M
    python -I tests/test_parity.py --model-dir PiVo-1.4M --tokens "1,21,38"

All tensors are checked per element with rtol=1e-4 and atol=1e-5. The 511-token
case exercises near the reference server's default 512-token frontend limit;
it is not a limit imposed by the acoustic model itself.

Attribution: PiVo-1.4M original (Empero Research, Leon Lehmann) Apache-2.0.
See src_id/acoustic.py header and PiVo-1.4M/NOTICE. Quantization/vocoder not in scope.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_id"))
from acoustic import load_pivo_checkpoint, phase_features  # noqa: E402

SENTENCES = {
    "nasa": [1, 21, 38, 25, 40, 3, 62, 32, 3, 24, 53, 12, 64, 3, 14, 44, 24, 3, 13, 40, 3, 21, 53, 18, 25, 27,
             3, 20, 62, 26, 40, 21, 4, 2],
    "hello": [1, 16, 40, 19, 68, 5, 3, 20, 50, 3, 21, 59, 20, 3, 62, 32, 3, 29, 68, 4, 3, 16, 47, 3, 20, 53, 21,
              64, 3, 23, 55, 38, 20, 40, 27, 55, 32, 3, 12, 41, 32, 3, 40, 3, 29, 71, 25, 3, 21, 65, 12, 6, 2],
}
CACHE_NAMES = ("key0", "value0", "key1", "value1")
RTOL, ATOL = 1e-4, 1e-5


def compare(actual, expected, label, failures):
    """Keep running after discrepancies so every case and output is diagnosed."""
    try:
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False, err_msg=label)
    except AssertionError as exc:
        failures.append(label)
        print(f"FAIL {label}\n{exc}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens", default=None, help="comma-separated token ids (optional)")
    ap.add_argument("--no-optimizations", action="store_true", help="disable ONNX graph and PyTorch MKLDNN optimizations")
    a = ap.parse_args()
    import onnxruntime as ort

    # Prevent excessive CPU oversubscription on machines with many cores.
    torch.set_num_threads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    if a.no_optimizations:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        torch.backends.mkldnn.enabled = False
    model, _ = load_pivo_checkpoint(os.path.join(a.model_dir, "checkpoints", "acoustic.pt"), num_voices=2)
    model.eval()
    enc_s = ort.InferenceSession(os.path.join(a.model_dir, "onnx", "encoder.onnx"), options,
                                 providers=["CPUExecutionProvider"])
    dec_s = ort.InferenceSession(os.path.join(a.model_dir, "onnx", "decoder.onnx"), options,
                                 providers=["CPUExecutionProvider"])

    # The long case uses one frame per phone to bound quadratic full-decoder memory.
    token_sets = {
        "length-1": [21],
        **SENTENCES,
        "near-512": [1] + (SENTENCES["nasa"][1:-1] * 16)[:509] + [2],
    }
    if a.tokens:
        token_sets["custom"] = [int(x) for x in a.tokens.split(",")]
    failures = []
    for name, toks in token_sets.items():
        for voice in (0, 1):
            label = f"{name} voice={voice}"
            tokens = np.array([toks], dtype=np.int64)
            o_enc, o_logd = enc_s.run(None, {"tokens": tokens, "voice": np.array([voice], np.int64),
                                             "lengths": np.array([len(toks)], np.int64)})
            with torch.no_grad():
                t_enc, t_logd = model.encode(torch.from_numpy(tokens), torch.tensor([voice]), torch.tensor([len(toks)]))
            compare(t_enc.numpy(), o_enc, f"{label} encoded", failures)
            compare(t_logd.numpy(), o_logd, f"{label} log_durations", failures)

            dur = model.predict_durations(torch.from_numpy(o_logd), torch.from_numpy(tokens))[0]
            if name == "near-512":
                dur = (torch.from_numpy(tokens[0]) >= 10).long()
            if not dur.sum():
                print(f"{label}: no frames from predicted durations; decoder skipped", flush=True)
                continue
            expanded = np.repeat(o_enc[0], dur.numpy(), axis=0)[None]
            phase = phase_features(dur).numpy()[None]
            expanded = np.concatenate((expanded, phase), axis=-1).astype(np.float32)

            cache = {n: np.zeros((1, 2, 0, 24), np.float32) for n in CACHE_NAMES}
            t_caches = [torch.from_numpy(cache[n].copy()) for n in CACHE_NAMES]
            o_feats, t_feats = [], []
            with torch.no_grad():
                t_full = model.denormalize(model.decode(torch.from_numpy(expanded))).numpy()
                for p in range(0, expanded.shape[1], 16):
                    chunk = expanded[:, p:p + 16]
                    result = dec_s.run(None, {"expanded": chunk, "position": np.array(p, np.int64), **cache})
                    t_feat, t_caches = model.decode_step(torch.from_numpy(chunk), torch.tensor(p), t_caches)
                    o_feats.append(result[0])
                    t_feats.append(t_feat.numpy())
                    cache = dict(zip(CACHE_NAMES, result[1:]))
                    for n, t_cache in zip(CACHE_NAMES, t_caches):
                        compare(t_cache.numpy(), cache[n], f"{label} position={p} {n}", failures)
            o_feats = np.concatenate(o_feats, axis=1)
            compare(t_full, o_feats, f"{label} full features", failures)
            compare(np.concatenate(t_feats, axis=1), o_feats, f"{label} stream features", failures)
            print(f"{label}: tokens={len(toks)} frames={expanded.shape[1]}", flush=True)

    # Independently seeded nonzero caches ensure both decoders see exactly the same
    # history, rather than letting prior backend differences propagate into the input.
    rng = np.random.default_rng(1234)
    seeded_cache = {n: rng.standard_normal((1, 2, 17, 24)).astype(np.float32) for n in CACHE_NAMES}
    chunk = rng.standard_normal((1, 7, model.config["width"] + 4)).astype(np.float32)
    result = dec_s.run(None, {"expanded": chunk, "position": np.array(17, np.int64), **seeded_cache})
    with torch.no_grad():
        t_feat, t_cache = model.decode_step(torch.from_numpy(chunk), torch.tensor(17),
                                            [torch.from_numpy(seeded_cache[n].copy()) for n in CACHE_NAMES])
    compare(t_feat.numpy(), result[0], "seeded nonzero cache features", failures)
    for n, actual, expected in zip(CACHE_NAMES, t_cache, result[1:]):
        compare(actual.numpy(), expected, f"seeded nonzero cache {n}", failures)

    print(f"PARITY {'FAILED' if failures else 'OK'}: {len(failures)} failed comparisons "
          f"(rtol={RTOL:g}, atol={ATOL:g})", flush=True)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
