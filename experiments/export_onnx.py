"""Export a fine-tuned acoustic checkpoint to PiVo's ONNX runtime format.

The exported folder has the same layout as PiVo-1.4M (onnx/encoder.onnx,
decoder.onnx, conditioning.onnx, vocoder_step.onnx, metadata.json,
model_manifest.json), so the PiVo runtime can load it for English, and this
repo's Indonesian frontend can drive the same graphs.

Exports both graphs from src_id/acoustic.py (the training code, not a copy),
then refuses to finish unless ONNX and PyTorch agree within --tol on Indonesian
and English sequences, both voices, chunk sizes 1/16/47.

    python -I experiments/export_onnx.py --ckpt experiments/runs/pilot_v5c/best.pt \
        --pivo-dir PiVo-1.4M --out experiments/models/pivo-id-1.4m
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))

from acoustic import CONFIG, phase_features  # noqa: E402
from acoustic import load_pivo_checkpoint  # noqa: E402

CACHE_NAMES = ["key0", "value0", "key1", "value1"]

TEST_TEXT = {
    "en": "Hello, my name is Vo. How many parameters does a voice need?",
    "id": "Selamat pagi, apa kabar? Layanan administrasi kampus dibuka pada hari Senin.",
}


class EncoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, voice, lengths):
        return self.model.encode(tokens, voice, lengths)


class DecoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, expanded, position, key0, value0, key1, value1):
        features, caches = self.model.decode_step(expanded, position, [key0, value0, key1, value1])
        return (features, *caches)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(model, onnx_dir: Path) -> None:
    tokens = torch.tensor([[1, 21, 38, 25, 40, 3, 62, 32, 4, 2]])
    dyn_enc = {"tokens": {0: "batch", 1: "tokens"}, "voice": {0: "batch"},
               "lengths": {0: "batch"},
               "encoded": {0: "batch", 1: "tokens"}, "log_durations": {0: "batch", 1: "tokens"}}
    torch.onnx.export(
        EncoderGraph(model).eval(), (tokens, torch.tensor([0]), torch.tensor([tokens.shape[1]])),
        str(onnx_dir / "encoder.onnx"), dynamo=False, opset_version=17,
        input_names=["tokens", "voice", "lengths"],
        output_names=["encoded", "log_durations"], dynamic_axes=dyn_enc)

    c = model.config
    hd = c["width"] // c["heads"]
    caches = [torch.zeros(1, c["kv_heads"], 5, hd) for _ in CACHE_NAMES]
    dyn = {"expanded": {0: "batch", 1: "frames"}, "features": {0: "batch", 1: "frames"}}
    for n in CACHE_NAMES:
        dyn[n] = {0: "batch", 2: "history"}
        dyn[n + "_out"] = {0: "batch", 2: "new_history"}
    torch.onnx.export(
        DecoderGraph(model).eval(), (torch.zeros(1, 7, c["width"] + 4), torch.tensor(5), *caches),
        str(onnx_dir / "decoder.onnx"), dynamo=False, opset_version=17,
        input_names=["expanded", "position", *CACHE_NAMES],
        output_names=["features", *[n + "_out" for n in CACHE_NAMES]], dynamic_axes=dyn)
    model.eval()  # the exporter restores the wrapper's train/eval flag; force eval


def token_sets(symbols: list[str]) -> dict[str, list[int]]:
    sys.path.insert(0, str(ROOT / "PiVo-1.4M"))
    from pivo.text import phonemize as en_phonemize
    from frontend import phonemize as id_phonemize

    out = {}
    for lang, phon in (("en", lambda t: en_phonemize(t, strict=False)), ("id", id_phonemize)):
        front = phon(TEST_TEXT[lang])
        if [symbols[i] for i in front.token_ids] != list(front.symbols):
            raise SystemExit(f"checkpoint symbol table does not match the {lang} frontend")
        out[lang] = list(front.token_ids)
    return out


def check(model, onnx_dir: Path, toks_by_lang: dict[str, list[int]], voices, tol: float) -> dict:
    import onnxruntime as ort

    enc = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(onnx_dir / "decoder.onnx"), providers=["CPUExecutionProvider"])
    c = model.config
    hd = c["width"] // c["heads"]
    worst = {"encoder": 0.0, "log_durations": 0.0, "decoder": 0.0}
    for toks in toks_by_lang.values():
        for voice in voices:
            feeds = {"tokens": np.array([toks], np.int64), "voice": np.array([voice], np.int64),
                     "lengths": np.array([len(toks)], np.int64)}
            o_enc, o_logd = enc.run(None, feeds)
            with torch.no_grad():
                t_enc, t_logd = model.encode(
                    torch.tensor([toks]), torch.tensor([voice]), torch.tensor([len(toks)]))
            worst["encoder"] = max(worst["encoder"], float(np.abs(t_enc.numpy() - o_enc).max()))
            worst["log_durations"] = max(worst["log_durations"],
                                         float(np.abs(t_logd.numpy() - o_logd).max()))
            dur = model.predict_durations(torch.from_numpy(o_logd), torch.from_numpy(feeds["tokens"]))[0]
            expanded = np.repeat(o_enc[0], dur.numpy(), axis=0)[None]
            expanded = np.concatenate((expanded, phase_features(dur).numpy()[None]), -1).astype(np.float32)
            with torch.no_grad():
                ref = model.denormalize(model.decode(torch.from_numpy(expanded))).numpy()
            for chunk in (1, 16, 47):
                cache = {n: np.zeros((1, c["kv_heads"], 0, hd), np.float32) for n in CACHE_NAMES}
                out = []
                for p in range(0, expanded.shape[1], chunk):
                    r = dec.run(None, {"expanded": expanded[:, p:p + chunk],
                                       "position": np.array(p, np.int64), **cache})
                    out.append(r[0])
                    cache = dict(zip(CACHE_NAMES, r[1:]))
                worst["decoder"] = max(worst["decoder"],
                                       float(np.abs(np.concatenate(out, 1) - ref).max()))
    print("ONNX vs PyTorch max abs diff:", {k: f"{v:.2e}" for k, v in worst.items()})
    if max(worst.values()) > tol:
        raise SystemExit(f"PARITY FAILED (tol {tol})")
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pivo-dir", type=Path, required=True,
                    help="PiVo-1.4M folder (unchanged vocoder graphs + base metadata)")
    ap.add_argument("--out", type=Path, required=True, help="new model folder")
    ap.add_argument("--name", default="PiVo-ID-1.4M")
    ap.add_argument("--tol", type=float, default=2e-4)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    model, ck = load_pivo_checkpoint(a.ckpt, num_voices=2)
    model.eval()
    onnx_dir = a.out / "onnx"
    onnx_dir.mkdir(parents=True)
    export(model, onnx_dir)
    for n in ("conditioning.onnx", "vocoder_step.onnx"):
        shutil.copyfile(a.pivo_dir / "onnx" / n, onnx_dir / n)

    base = json.loads((a.pivo_dir / "onnx" / "metadata.json").read_text(encoding="utf-8"))
    symbols = list(base["symbols"])
    voices = sorted({0, 1})
    worst = check(model, onnx_dir, token_sets(symbols), voices, a.tol)

    acoustic = sum(p.numel() for p in model.parameters())
    vocoder = base["parameters"]["components"]["vocoder"]["parameters"]
    meta = dict(base)
    # Runtime keys (sample_rate, hop_length, phoneme_phase, ...) stay from PiVo;
    # only the acoustic architecture fields come from this checkpoint.
    meta["config"] = dict(base["config"], **{k: v for k, v in model.config.items()
                                             if k != "dropout"})
    meta["config"].update({"head_dim": model.config["width"] // model.config["heads"],
                           "dropout": base["config"]["dropout"]})
    meta["voices"] = {"Pi": 0, "Vo": 1}
    meta["voice_info"] = {
        "Pi": {"id": 0, "speaker": "F1", "language": ["id", "en"], "gender": "female"},
        "Vo": {"id": 1, "speaker": "M1", "language": ["id", "en"], "gender": "male"},
    }
    meta["languages"] = ["id", "en"]
    meta["frontend"] = "pivo-id-en-v1"
    meta["source_checkpoint"] = {"file": a.ckpt.name, "sha256": sha256(a.ckpt),
                                 "kind": ck.get("format"), "step": ck.get("step")}
    meta["parameters"] = dict(base["parameters"], provisional_name=a.name,
                              total_parameters=acoustic + vocoder,
                              within_1_5m=acoustic + vocoder <= 1_500_000)
    meta["parameters"]["components"] = dict(
        base["parameters"]["components"],
        acoustic={"parameters": acoustic, "trainable": acoustic, "fp32_bytes": 4 * acoustic})
    meta["verification"] = dict(base["verification"])
    meta["verification"]["encoder"] = {
        "max_absolute_error": max(worst["encoder"], worst["log_durations"]),
        "bytes": (onnx_dir / "encoder.onnx").stat().st_size}
    meta["verification"]["decoder"] = {
        "max_absolute_error": worst["decoder"],
        "bytes": (onnx_dir / "decoder.onnx").stat().st_size}
    (onnx_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    files = []
    for p in sorted(onnx_dir.iterdir()):
        files.append({"path": f"onnx/{p.name}", "bytes": p.stat().st_size, "sha256": sha256(p)})
    manifest = {"format": "pivo-model-manifest-v1", "model_name": a.name,
                "total_inference_parameters": acoustic + vocoder,
                "sample_rate": base["config"]["sample_rate"], "files": files}
    (a.out / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")
    for extra in ("LICENSE", "NOTICE", "THIRD_PARTY.md"):
        if (a.pivo_dir / extra).exists():
            shutil.copyfile(a.pivo_dir / extra, a.out / extra)
    if (a.pivo_dir / "LICENSES").is_dir():
        shutil.copytree(a.pivo_dir / "LICENSES", a.out / "LICENSES", dirs_exist_ok=True)
    print(f"wrote {a.out}: {acoustic:,} acoustic + {vocoder:,} vocoder = "
          f"{acoustic + vocoder:,} params")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
