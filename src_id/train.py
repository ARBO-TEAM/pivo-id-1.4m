"""Stage-1 supervised training for the PiVo acoustic model (minimal, genuine).

Trains ``PiVoAcoustic`` (``src_id/acoustic.py``) on an explicit JSONL dataset of
precomputed FARGAN (Opus) feature matrices -- no synthetic / invented targets:
every target frame comes from a ``.npy`` file produced by the FARGAN feature
extractor.  The loss is the stage-1 recipe from the PiVo technical report,

    loss = SmoothL1(normalized features) + 0.1 * MSE(log durations, log(1 + frames))

masked over padded frames/tokens (see ``masked_losses``).

Dataset format (one JSON object per line)::

    {"id": str, "tokens": [int, ...], "durations": [int, ...],
     "features_path": "<path to .npy, [frames, 20] float32>",
     "voice_id": int, "split": "train" | "val"}

``features_path`` is resolved relative to the JSONL file (or as an absolute
path).  Rows are validated strictly at load time (see ``validate_rows``):
token/duration lengths match, ``sum(durations) == frames`` (rows of the
``.npy``), features are 2-D ``[frames, 20]`` and finite, both ``train`` and
``val`` splits are non-empty, ids are unique, and no id or feature file is
shared between splits (no leakage).  ``test`` rows are refused outright:
.model selection here is by on ``val`` only and ``test`` is never read.

Coordination / import safety
----------------------------
Training uses ``PiVoAcoustic`` directly through its plain public API
(``config`` / ``feature_mean`` / ``feature_std`` / ``parameters()`` /
``train()`` / ``eval()`` / ``state_dict()`` / ``load_state_dict()`` /
``forward(...)``) with no abstraction layer, so the acoustic module can be
edited independently.  The only import from inside this project is the
sibling ``acoustic`` module
in this same trusted directory (inserted explicitly into ``sys.path``, so
``python -I`` runs work).  Nothing is imported or executed from the read-only
reference checkouts (``PiVo-1.4M/``, ``pivo-v1-multilingual/``): model
weights are only ever loaded as data via ``load_pivo_checkpoint`` when the
caller passes ``--pivo-ckpt``.

Modes
-----
* default (from scratch): fresh ``PiVoAcoustic`` sized from the dataset
  (``vocab_size`` / ``num_voices`` derived from the rows, ``--width`` /
  ``--dropout`` from the CLI); feature mean/std are computed empirically
  from the train split and stored in the model's buffers / checkpoints.
* ``--pivo-ckpt PATH``: warm-start weights from a PiVo acoustic checkpoint
  via ``load_pivo_checkpoint`` (voice embedding grown to fit the dataset).
* ``--init-ckpt PATH``: fresh model, weights only from one of this script's
  own checkpoints.  ``--resume PATH`` (or ``auto`` = ``<out>/last.pt``):
  full resume -- weights, optimizer state and step/epoch are restored.
  ``--pivo-ckpt`` / ``--init-ckpt`` / ``--resume`` are mutually exclusive.

Checkpoints are written atomically (temporary file + ``os.replace``) as
``<out>/best.pt`` (best val loss) and ``<out>/last.pt`` (every
``--save-every`` steps); a JSONL metrics log is appended at
``<out>/train_log.jsonl``.  If the dataset is absent the script exits
non-zero before building any model -- it never trains on missing data.

Attribution / license
---------------------
Training-loop structure (masked SmoothL1 + log-duration MSE, warmup/cosine
LR, AdamW decay split, grad clipping, atomic ``path + ".tmp"`` + replace
checkpointing) is adapted from ``pivo-v1-multilingual/training/pivo_train/``
(``train.py`` / ``data.py`` / ``model.py``), which is Apache-2.0 (see
``pivo-v1-multilingual/LICENSE`` and ``PiVo-1.4M/NOTICE``).  The acoustic
model itself is the Apache-2.0 original (Empero Research, Leon Lehmann
<lehmann@empero.org>); see ``src_id/acoustic.py``.  This file retains those
notices and must keep ``LICENSE``/``NOTICE``/``THIRD_PARTY.md`` with any
redistribution.  The JSONL dataset schema above is specific to this script.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Import sibling module from this trusted directory only (works under `python -I`,
# where neither the script directory nor the cwd is on sys.path).  No reference
# checkout is ever added to sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from acoustic import PiVoAcoustic, load_pivo_checkpoint  # noqa: E402

FEATURE_DIM = 20
MAX_TOKEN_FRAMES = 200
MAX_UTTERANCE_FRAMES = 3000
REQUIRED_KEYS = ("id", "tokens", "durations", "features_path", "voice_id", "split")
SPLITS = ("train", "val")
CKPT_FORMAT = "pivo-acoustic-train-v1"


# --------------------------------------------------------------------------- dataset
def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def resolve_features_path(jsonl_path, row):
    p = row["features_path"]
    if os.path.isabs(p):
        return p
    return os.path.join(os.path.dirname(os.path.abspath(jsonl_path)), p)


def validate_rows(rows, jsonl_path, vocab_size=None):
    """Strict validation of every row; raises ValueError on the first problem.

    Split-specific checks (non-empty train/val, no id/feature leakage) live in
    ``load_dataset``; per-row checks (keys, types, lengths, duration sum,
    feature file shape/finiteness) run here.  Note: feature-file checks need
    the dataset dir, so they run in ``load_dataset`` via ``validate_features``;
    this function covers everything that does not touch the filesystem.
    """
    if not rows:
        raise ValueError("dataset is empty: no rows in JSONL")
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"row {i}: not a JSON object")
        where = f"row {i} (id={r.get('id')!r})"
        for k in REQUIRED_KEYS:
            if k not in r:
                raise ValueError(f"{where}: missing required key {k!r}")
        if not isinstance(r["id"], str) or not r["id"]:
            raise ValueError(f"{where}: 'id' must be a non-empty string")
        if r["split"] == "test":
            raise ValueError(
                f"{where}: split 'test' is refused -- this trainer does model "
                "selection on 'val' only and never reads test data"
            )
        if r["split"] not in SPLITS:
            raise ValueError(f"{where}: split must be one of {SPLITS}, got {r['split']!r}")
        tokens, durations = r["tokens"], r["durations"]
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"{where}: 'tokens' must be a non-empty list")
        if not isinstance(durations, list) or len(durations) != len(tokens):
            raise ValueError(
                f"{where}: len(durations)={len(durations) if isinstance(durations, list) else 'n/a'} "
                f"!= len(tokens)={len(tokens)}"
            )
        for t in tokens:
            if not _is_int(t) or t < 0:
                raise ValueError(f"{where}: token ids must be ints >= 0, got {t!r}")
            if vocab_size is not None and t >= vocab_size:
                raise ValueError(f"{where}: token id {t} >= vocab_size {vocab_size}")
        for d in durations:
            if not _is_int(d) or d < 0:
                raise ValueError(f"{where}: durations must be ints >= 0, got {d!r}")
            if d > MAX_TOKEN_FRAMES:
                raise ValueError(f"{where}: duration {d} > max_token_frames {MAX_TOKEN_FRAMES}")
        frames = sum(durations)
        if frames <= 0:
            raise ValueError(f"{where}: sum(durations) must be > 0, got {frames}")
        if frames > MAX_UTTERANCE_FRAMES:
            raise ValueError(f"{where}: sum(durations)={frames} > max_utterance_frames {MAX_UTTERANCE_FRAMES}")
        if not _is_int(r["voice_id"]) or r["voice_id"] < 0:
            raise ValueError(f"{where}: 'voice_id' must be an int >= 0, got {r['voice_id']!r}")
        if not isinstance(r["features_path"], str) or not r["features_path"]:
            raise ValueError(f"{where}: 'features_path' must be a non-empty string")
    return rows


def load_features(path, expected_frames):
    """Load and fully validate one precomputed feature matrix (finite, [frames, 20])."""
    if not os.path.exists(path):
        raise ValueError(f"features file not found: {path}")
    feats = np.load(path, allow_pickle=False)
    if feats.ndim != 2 or feats.shape[1] != FEATURE_DIM:
        raise ValueError(f"{path}: expected shape [frames, {FEATURE_DIM}], got {tuple(feats.shape)}")
    if feats.shape[0] != expected_frames:
        raise ValueError(
            f"{path}: {feats.shape[0]} feature frames != sum(durations)={expected_frames}"
        )
    if not np.all(np.isfinite(feats)):
        raise ValueError(f"{path}: features contain NaN or inf")
    return feats.astype(np.float32)


def validate_features(rows, jsonl_path):
    """Eagerly check every referenced feature file exists and matches its row."""
    for r in rows:
        path = resolve_features_path(jsonl_path, r)
        feats = load_features(path, sum(r["durations"]))
        del feats  # freed; PiVoDataset re-loads (or preloads) as configured


def load_dataset(jsonl_path, vocab_size=None):
    """Read + validate the JSONL dataset; returns (train_rows, val_rows).

    Fails (ValueError) on any invalid row, on empty train or val splits, on
    duplicate ids, and on train/val sharing an id or a features file.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"dataset not found: {jsonl_path}")
    rows = validate_rows(read_jsonl(jsonl_path), jsonl_path, vocab_size=vocab_size)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if not train:
        raise ValueError("dataset has no 'train' rows")
    if not val:
        raise ValueError("dataset has no 'val' rows -- model selection requires validation data")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        dup = sorted({x for x in ids if ids.count(x) > 1})
        raise ValueError(f"duplicate id(s) in dataset: {dup}")
    feat_of = {r["id"]: os.path.normpath(resolve_features_path(jsonl_path, r)) for r in rows}
    train_ids, val_ids = {r["id"] for r in train}, {r["id"] for r in val}
    if train_ids & val_ids:
        raise ValueError(f"id(s) in both train and val (leakage): {sorted(train_ids & val_ids)}")
    train_feats = {feat_of[i] for i in train_ids}
    val_feats = {feat_of[i] for i in val_ids}
    if train_feats & val_feats:
        raise ValueError(f"feature file(s) shared between train and val (leakage): {sorted(train_feats & val_feats)}")
    validate_features(rows, jsonl_path)
    return train, val


class PiVoDataset(Dataset):
    """Minimal dataset: yields token/duration tensors + normalized features.

    Normalization uses the model's feature_mean/feature_std buffers (passed in
    as arrays), exactly like the reference pipeline.  With ``preload=True``
    (default) every feature file is loaded and validated once at construction;
    otherwise files are loaded (and validated) on access.
    """

    def __init__(self, jsonl_path, rows, feature_mean, feature_std, preload=True):
        self.jsonl_path = jsonl_path
        self.rows = list(rows)
        self.mean = np.asarray(feature_mean, dtype=np.float32)
        self.std = np.asarray(feature_std, dtype=np.float32)
        if self.mean.shape != (FEATURE_DIM,) or self.std.shape != (FEATURE_DIM,):
            raise ValueError("feature_mean/std must have shape (20,)")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)) or np.any(self.std <= 0):
            raise ValueError("feature_mean/std must be finite with std > 0")
        self.cache = [load_features(resolve_features_path(jsonl_path, r), sum(r["durations"]))
                      for r in self.rows] if preload else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        feats = self.cache[i] if self.cache is not None else load_features(
            resolve_features_path(self.jsonl_path, r), sum(r["durations"]))
        normalized = torch.from_numpy((feats - self.mean) / self.std)
        return {
            "id": r["id"],
            "tokens": torch.tensor(r["tokens"], dtype=torch.long),
            "durations": torch.tensor(r["durations"], dtype=torch.long),
            "voice": int(r["voice_id"]),
            "features": normalized,
        }


def collate(batch):
    b = len(batch)
    t_max = max(len(x["tokens"]) for x in batch)
    f_max = max(len(x["features"]) for x in batch)
    tokens = torch.zeros(b, t_max, dtype=torch.long)  # 0 = <pad>
    durations = torch.zeros(b, t_max, dtype=torch.long)
    features = torch.zeros(b, f_max, batch[0]["features"].shape[1])
    token_lengths = torch.zeros(b, dtype=torch.long)
    frame_lengths = torch.zeros(b, dtype=torch.long)
    for i, x in enumerate(batch):
        n, f = len(x["tokens"]), len(x["features"])
        tokens[i, :n] = x["tokens"]
        durations[i, :n] = x["durations"]
        features[i, :f] = x["features"]
        token_lengths[i] = n
        frame_lengths[i] = f
    return {
        "ids": [x["id"] for x in batch],
        "voices": torch.tensor([x["voice"] for x in batch], dtype=torch.long),
        "tokens": tokens, "durations": durations, "features": features,
        "token_lengths": token_lengths, "frame_lengths": frame_lengths,
    }


def build_model(args, train_rows, val_rows):
    """Build the model (directly ``PiVoAcoustic`` -- no abstraction layer).

    Training depends on ``PiVoAcoustic`` only through its plain public API
    (``config`` / ``feature_mean`` / ``feature_std`` / ``parameters()`` /
    ``train()`` / ``eval()`` / ``state_dict()`` / ``load_state_dict()`` /
    ``forward(tokens, voice, token_lengths, durations, frame_lengths)`` ->
    ``(predicted normalized features, log durations)``), so the acoustic
    module can be edited independently.  Sizes for from-scratch come from
    the dataset, and random initialization happens ONLY under the explicit
    ``--from-scratch`` flag: the default explicit sources are ``--pivo-ckpt``
    (warm-start weights), ``--init-ckpt`` (weights from our own checkpoint)
    and ``--resume`` (full resume) -- never an unflagged random init.
    """
    all_rows = train_rows + val_rows
    num_voices = max(r["voice_id"] for r in all_rows) + 1
    if args.pivo_ckpt:
        num_voices = max(num_voices, 2)  # PiVo's voice embedding has Pi+Vo rows
    want_stats = args.compute_feature_stats or args.reset_feature_stats
    stats = None
    if want_stats:
        feats = np.concatenate([load_features(resolve_features_path(args.data, r), sum(r["durations"]))
                                for r in train_rows])
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)  # constant dim: fall back to unit scale
        stats = (mean.astype(np.float32), std.astype(np.float32))
    if args.pivo_ckpt:
        model, _ = load_pivo_checkpoint(args.pivo_ckpt, num_voices=num_voices)
        if args.reset_feature_stats:
            model.feature_mean.copy_(torch.from_numpy(stats[0]))
            model.feature_std.copy_(torch.from_numpy(stats[1]))
        return model
    vocab_size = args.vocab_size or (max(max(r["tokens"]) for r in all_rows) + 1)
    if args.width % 8:
        raise ValueError("--width must be a multiple of 8 (4 heads, even RoPE dim)")
    model = PiVoAcoustic(
        config={"width": args.width, "ff_width": 2 * args.width, "vocab_size": vocab_size,
                "num_voices": num_voices, "num_langs": 0, "dropout": args.dropout},
        num_voices=num_voices,
    )
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["state_dict"], strict=True)
    elif args.from_scratch:
        # Empirical train-split statistics (genuine data statistics, stored in
        # the buffers and checkpoints) -- same role as feature_stats.json in
        # the reference pipeline.
        model.feature_mean.copy_(torch.from_numpy(stats[0]))
        model.feature_std.copy_(torch.from_numpy(stats[1]))
    elif args.resume:
        # Weights/stats are loaded from the checkpoint in main(); just return
        # the fresh shell with the dataset-derived shapes.  Without this,
        # '--resume' alone raised 'no weight source' before ever reaching
        # the resume path in main().
        pass
    else:
        raise ValueError("no weight source: pass --pivo-ckpt, --init-ckpt, or --resume "
                         "(random init requires the explicit --from-scratch flag)")
    return model


# --------------------------------------------------------------------------- training
def masked_losses(pred, target, frame_lengths, log_dur, durations, token_lengths):
    """Masked SmoothL1 on normalized features + masked MSE on log durations."""
    dev = pred.device
    fmask = (torch.arange(pred.shape[1], device=dev)[None] < frame_lengths[:, None]).float()
    feat = (F.smooth_l1_loss(pred, target, reduction="none").mean(-1) * fmask).sum() / fmask.sum()
    tmask = (torch.arange(log_dur.shape[1], device=dev)[None] < token_lengths[:, None]).float()
    dur = ((log_dur - torch.log1p(durations.float())).pow(2) * tmask).sum() / tmask.sum()
    return feat, dur


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


def create_optimizer(model, args):
    emb = ("voices", "tokens", "langs")
    decay = [p for n, p in model.named_parameters()
             if p.dim() >= 2 and not any(e in n for e in emb)]
    no_decay = [p for n, p in model.named_parameters()
                if not (p.dim() >= 2 and not any(e in n for e in emb))]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.98))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (TypeError, AttributeError):
        pass


def batch_to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    model.eval()
    tot_f = tot_d = n = 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        b = batch_to(batch, device)
        pred, logd = model(b["tokens"], b["voices"], b["token_lengths"],
                           b["durations"], b["frame_lengths"])
        f, d = masked_losses(pred, b["features"], b["frame_lengths"],
                             logd, b["durations"], b["token_lengths"])
        tot_f += f.item()
        tot_d += d.item()
        n += 1
    model.train()
    return {"val_feature": tot_f / max(n, 1), "val_duration": tot_d / max(n, 1)}


def _torchify_np_rng_state(state):
    """Convert np.random.get_state() to a dict of torch tensors (weights_only-safe)."""
    name, keys, pos, has_gauss, cached_gauss = state
    return {"name": name, "keys": torch.from_numpy(keys.copy()),
            "pos": int(pos), "has_gauss": int(has_gauss), "cached_gauss": float(cached_gauss)}


def _restore_np_rng_state(state):
    keys = state["keys"].numpy() if torch.is_tensor(state["keys"]) else np.asarray(state["keys"])
    np.random.set_state((state["name"], keys, int(state["pos"]),
                         int(state["has_gauss"]), float(state["cached_gauss"])))


def save_checkpoint(path, model, opt, step, epoch, best, args):
    """Atomic save: write to a sibling temp file, then os.replace over target."""
    tmp = path + ".tmp"
    # Only tensors + primitives + plain containers: reloadable with
    # weights_only=True (no pickle payloads, no unsafe fallback).
    # RNG states are numpy arrays/tuples (numpy's internal types); persist them
    # as plain byte tensors so the file stays weights_only=True loadable.
    save_obj = {
        "format": CKPT_FORMAT,
        "step": step, "epoch": epoch, "best_val": best,
        "config": dict(model.config),
        "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
        "args": {k: v for k, v in vars(args).items()
                 if isinstance(v, (str, int, float, bool)) or v is None},
        "py_rng_state": random.getstate(),
        "np_rng_state": _torchify_np_rng_state(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(save_obj, tmp)
    os.replace(tmp, path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="explicit JSONL dataset (id/tokens/durations/features_path/voice_id/split)")
    p.add_argument("--out", required=True, help="output directory for checkpoints + train_log.jsonl")
    p.add_argument("--from-scratch", action="store_true",
                   help="EXPLICITLY request random init (otherwise a weight source is required)")
    p.add_argument("--pivo-ckpt", default=None, help="warm-start from PiVo acoustic.pt (weights only)")
    p.add_argument("--init-ckpt", default=None, help="init weights from this script's own checkpoint")
    p.add_argument("--resume", default=None, help="full resume: checkpoint path or 'auto' (= <out>/last.pt)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--duration-weight", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--width", type=int, default=96, help="from-scratch model width (multiple of 8)")
    p.add_argument("--vocab-size", type=int, default=None, help="from-scratch vocab (default: derived from data)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=50)
    p.add_argument("--amp", choices=["none", "bf16"], default="none")
    p.add_argument("--no-preload", action="store_true", help="load feature files lazily instead of preloading")
    p.add_argument("--no-compute-feature-stats", action="store_true",
                   help="from scratch: keep zero/unit buffers instead of train-split stats")
    p.add_argument("--reset-feature-stats", action="store_true",
                   help="warm-start: overwrite the checkpoint's feature_mean/std buffers with "
                        "train-split statistics (PiVo's English stats misnormalize Indonesian features)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    sources = [bool(args.from_scratch), bool(args.pivo_ckpt), bool(args.init_ckpt), bool(args.resume)]
    if sum(sources) != 1:
        p.error("exactly one weight source is required: --from-scratch (explicit random init), "
                "--pivo-ckpt, --init-ckpt, or --resume")
    args.compute_feature_stats = args.from_scratch and not args.no_compute_feature_stats
    return args


def main(argv=None):
    args = parse_args(argv)

    # Never train on an absent dataset: fail before any model/optimizer exists.
    if not os.path.exists(args.data):
        print(f"error: dataset not found: {args.data}", file=sys.stderr)
        return 1
    try:
        train_rows, val_rows = load_dataset(args.data)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: invalid dataset: {exc}", file=sys.stderr)
        return 1

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.jsonl")

    model = build_model(args, train_rows, val_rows)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = args.dropout
    model.to(args.device).train()

    step, epoch, best = 0, 0, float("inf")
    opt = create_optimizer(model, args)
    resume = os.path.join(args.out, "last.pt") if args.resume == "auto" else args.resume
    if resume and not os.path.exists(resume):
        print(f"error: resume checkpoint not found: {resume}", file=sys.stderr)
        return 1
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["state_dict"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        step, epoch, best = ck["step"], ck.get("epoch", 0), ck.get("best_val", best)
        if "py_rng_state" in ck:
            random.setstate(ck["py_rng_state"])
        if "np_rng_state" in ck:
            _restore_np_rng_state(ck["np_rng_state"])
        if "torch_rng_state" in ck:
            torch.set_rng_state(ck["torch_rng_state"])
        if torch.cuda.is_available() and "torch_cuda_rng_state" in ck:
            torch.cuda.set_rng_state_all(ck["torch_cuda_rng_state"])
        print(f"resumed from {resume} at step {step}")

    # Normalisation stats must come from the (possibly resumed) model, not a
    # fresh one -- snapshotting them before resume would normalise every
    # resumed batch with the wrong mean/std.
    mean = model.feature_mean.detach().cpu().numpy()
    std = model.feature_std.detach().cpu().numpy()

    train_ds = PiVoDataset(args.data, train_rows, mean, std, preload=not args.no_preload)
    val_ds = PiVoDataset(args.data, val_rows, mean, std, preload=not args.no_preload)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=min(2, args.num_workers))

    def log(record):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def run_eval(s):
        ev = evaluate(model, val_loader, args.device, args.max_val_batches)
        ev.update({"kind": "eval", "step": s})
        print("eval:", json.dumps(ev), flush=True)
        log(ev)
        score = ev["val_feature"] + args.duration_weight * ev["val_duration"]
        return score

    if step == 0:
        ev0 = evaluate(model, val_loader, args.device, args.max_val_batches)
        ev0.update({"kind": "eval", "step": 0})
        print("before training:", json.dumps(ev0), flush=True)
        log(ev0)

    t0, run_f, run_d, run_n = time.time(), 0.0, 0.0, 0
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, args)
            b = batch_to(batch, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.amp == "bf16" and args.device.startswith("cuda")):
                pred, logd = model(b["tokens"], b["voices"], b["token_lengths"],
                                   b["durations"], b["frame_lengths"])
            f, d = masked_losses(pred.float(), b["features"], b["frame_lengths"],
                                 logd.float(), b["durations"], b["token_lengths"])
            loss = f + args.duration_weight * d
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1
            run_f, run_d, run_n = run_f + f.item(), run_d + d.item(), run_n + 1

            if step % args.log_every == 0:
                rec = {"kind": "train", "step": step, "epoch": epoch,
                       "feature": run_f / run_n, "duration": run_d / run_n,
                       "loss": run_f / run_n + args.duration_weight * run_d / run_n,
                       "lr": opt.param_groups[0]["lr"], "grad_norm": float(gn),
                       "steps_per_s": run_n / max(time.time() - t0, 1e-9)}
                print(json.dumps(rec), flush=True)
                log(rec)
                t0, run_f, run_d, run_n = time.time(), 0.0, 0.0, 0

            if step % args.eval_every == 0 or step == args.steps:
                score = run_eval(step)
                if score < best:
                    best = score
                    save_checkpoint(os.path.join(args.out, "best.pt"), model, opt, step, epoch, best, args)

            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(os.path.join(args.out, "last.pt"), model, opt, step, epoch, best, args)
        epoch += 1

    print(f"done: {step} steps, best val {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
