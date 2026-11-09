"""Unit tests for src_id/train.py (supervised PiVo acoustic training).

Run from a trusted directory with the project venv, isolated mode, e.g.:
    python -I tests/test_training.py
    python -I -m unittest tests.test_training -v   (from workspace root)

These are pure unit tests: dataset/loader validation, masked losses, LR
schedule, optimizer group split, atomic checkpointing, and the
"no dataset -> no training" exit path.  They never run a training loop and
never read any training dataset present on disk.  Tiny .npy files written
into a temp dir are shape-validation fixtures only -- they are not training
targets and must never be treated as FARGAN features.

Attribution: training-loop structure adapted from pivo-v1-multilingual
(Apache-2.0); see src_id/train.py header and PiVo-1.4M/NOTICE.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_id"))
import train as T  # noqa: E402


def write_row(tmp, rid, tokens, durations, feats, voice_id=0, split="train"):
    """Write a feature .npy fixture + return a dataset row dict."""
    path = os.path.join(tmp, f"{rid}.npy")
    np.save(path, np.asarray(feats, dtype=np.float32))
    return {"id": rid, "tokens": tokens, "durations": durations,
            "features_path": os.path.basename(path), "voice_id": voice_id, "split": split}


def write_jsonl(tmp, rows, name="data.jsonl"):
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def feats(frames, fill=0.0):
    return np.full((frames, T.FEATURE_DIM), fill, dtype=np.float32)


class TestRowValidation(unittest.TestCase):
    def test_missing_key(self):
        row = {"id": "a", "tokens": [1], "durations": [3],
               "features_path": "x.npy", "voice_id": 0}  # no 'split'
        with self.assertRaises(ValueError):
            T.validate_rows([row], "data.jsonl")

    def test_token_duration_length_mismatch(self):
        row = write_row(".", "a", [1, 2], [3], feats(3))
        with self.assertRaises(ValueError):
            T.validate_rows([row], "data.jsonl")

    def test_empty_tokens(self):
        row = write_row(".", "a", [], [], feats(0))
        with self.assertRaises(ValueError):
            T.validate_rows([row], "data.jsonl")

    def test_negative_duration_and_negative_token(self):
        for bad in (write_row(".", "a", [1], [-1], feats(0)),
                    write_row(".", "b", [-1], [3], feats(3))):
            with self.assertRaises(ValueError):
                T.validate_rows([bad], "data.jsonl")

    def test_duration_over_max_token_frames(self):
        row = write_row(".", "a", [1], [T.MAX_TOKEN_FRAMES + 1], feats(T.MAX_TOKEN_FRAMES + 1))
        with self.assertRaises(ValueError):
            T.validate_rows([row], "data.jsonl")

    def test_absurd_total_frames(self):
        row = write_row(".", "a", [1, 2], [T.MAX_UTTERANCE_FRAMES, 1], feats(T.MAX_UTTERANCE_FRAMES + 1))
        with self.assertRaises(ValueError):
            T.validate_rows([row], "data.jsonl")

    def test_bad_split_type(self):
        row = write_row(".", "a", [1], [3], feats(3), split="holdout")
        with self.assertRaises(ValueError):
            T.validate_rows([row], "data.jsonl")

    def test_test_split_refused(self):
        row = write_row(".", "a", [1], [3], feats(3), split="test")
        with self.assertRaises(ValueError) as cm:
            T.validate_rows([row], "data.jsonl")
        self.assertIn("test", str(cm.exception))

    def test_vocab_bound(self):
        row = write_row(".", "a", [9], [3], feats(3))
        with self.assertRaises(ValueError):
            T.validate_rows([row], "data.jsonl", vocab_size=9)


class TestLoadFeatures(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(ValueError):
            T.load_features("/nonexistent/x.npy", 3)

    def test_wrong_dims(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.npy")
            np.save(p, np.zeros((3, 19), dtype=np.float32))
            with self.assertRaises(ValueError):
                T.load_features(p, 3)

    def test_frames_must_match_duration_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.npy")
            np.save(p, feats(4))
            with self.assertRaises(ValueError):
                T.load_features(p, 3)

    def test_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (np.nan, np.inf):
                p = os.path.join(tmp, "x.npy")
                f = feats(3)
                f[0, 0] = bad
                np.save(p, f)
                with self.assertRaises(ValueError):
                    T.load_features(p, 3)

    def test_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.npy")
            np.save(p, feats(3, fill=1.5))
            out = T.load_features(p, 3)
            self.assertEqual(out.shape, (3, 20))
            self.assertTrue(np.all(out == 1.5))


class TestLoadDataset(unittest.TestCase):
    def _two_rows(self, tmp, train_voice=0, val_voice=0):
        train = write_row(tmp, "tr0", [1, 2], [2, 1], feats(3, 0.1), voice_id=train_voice, split="train")
        val = write_row(tmp, "va0", [1, 2], [1, 2], feats(3, 0.2), voice_id=val_voice, split="val")
        return write_jsonl(tmp, [train, val])

    def test_ok_and_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._two_rows(tmp)
            train, val = T.load_dataset(path)
            self.assertEqual([r["id"] for r in train], ["tr0"])
            self.assertEqual([r["id"] for r in val], ["va0"])

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            T.load_dataset("/nonexistent/data.jsonl")

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(tmp, [])
            with self.assertRaises(ValueError):
                T.load_dataset(path)

    def test_train_only_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(tmp, [write_row(tmp, "tr0", [1], [3], feats(3), split="train")])
            with self.assertRaises(ValueError):
                T.load_dataset(path)

    def test_val_only_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(tmp, [write_row(tmp, "va0", [1], [3], feats(3), split="val")])
            with self.assertRaises(ValueError):
                T.load_dataset(path)

    def test_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(tmp, [
                write_row(tmp, "dup", [1], [3], feats(3), split="train"),
                write_row(tmp, "dup", [1], [3], feats(3), split="val"),
            ])
            with self.assertRaises(ValueError):
                T.load_dataset(path)

    def test_id_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            # same id used in both splits (different files so the leak is the id)
            a = write_row(tmp, "same", [1], [3], feats(3), split="train")
            b = write_row(tmp, "same2", [1], [3], feats(3), split="val")
            b["id"] = "same"
            with self.assertRaises(ValueError):
                T.load_dataset(write_jsonl(tmp, [a, b]))

    def test_feature_file_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = write_row(tmp, "tr0", [1], [3], feats(3), split="train")
            b = write_row(tmp, "va0", [1], [3], feats(3), split="val")
            b["features_path"] = a["features_path"]  # same file in both splits
            with self.assertRaises(ValueError):
                T.load_dataset(write_jsonl(tmp, [a, b]))

    def test_invalid_json_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json}\n")
            with self.assertRaises(ValueError):
                T.load_dataset(path)


class TestDatasetAndCollate(unittest.TestCase):
    def test_shapes_padding_and_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [write_row(tmp, "a", [1, 2], [2, 1], feats(3, 1.0), voice_id=0, split="train"),
                    write_row(tmp, "b", [1], [4], feats(4, 2.0), voice_id=1, split="train")]
            path = write_jsonl(tmp, rows)
            mean = np.zeros(20, dtype=np.float32)
            std = np.ones(20, dtype=np.float32)
            ds = T.PiVoDataset(path, rows, mean, std, preload=True)
            self.assertEqual(len(ds), 2)
            batch = T.collate([ds[0], ds[1]])
            self.assertEqual(batch["tokens"].shape, (2, 2))
            self.assertEqual(batch["features"].shape, (2, 4, 20))
            self.assertEqual(batch["token_lengths"].tolist(), [2, 1])
            self.assertEqual(batch["frame_lengths"].tolist(), [3, 4])
            self.assertEqual(batch["voices"].tolist(), [0, 1])
            # padded positions are zeros
            self.assertTrue(torch.all(batch["features"][0, 3:] == 0))
            self.assertTrue(torch.all(batch["durations"][1, 1:] == 0))

    def test_normalization_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [write_row(tmp, "a", [1], [3], feats(3, 2.0))]
            path = write_jsonl(tmp, rows)
            ds = T.PiVoDataset(path, rows, np.full(20, 1.0, np.float32),
                               np.full(20, 2.0, np.float32))
            self.assertTrue(torch.all(ds[0]["features"] == 0.5))

    def test_bad_stats_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [write_row(tmp, "a", [1], [3], feats(3))]
            path = write_jsonl(tmp, rows)
            with self.assertRaises(ValueError):
                T.PiVoDataset(path, rows, np.zeros(19, np.float32), np.ones(20, np.float32))
            with self.assertRaises(ValueError):
                T.PiVoDataset(path, rows, np.zeros(20, np.float32), np.zeros(20, np.float32))


class TestMaskedLosses(unittest.TestCase):
    def test_padded_positions_do_not_affect_loss(self):
        torch.manual_seed(0)
        pred = torch.randn(2, 5, 20)
        target = torch.randn(2, 5, 20)
        logd = torch.randn(2, 3)
        durs = torch.tensor([[2, 1, 0], [1, 1, 0]])
        flen = torch.tensor([4, 3])
        tlen = torch.tensor([2, 2])
        f1, d1 = T.masked_losses(pred, target, flen, logd, durs, tlen)
        target2 = target.clone()
        target2[0, 4] = 999.0  # padded frame of item 0
        logd2 = logd.clone()
        logd2[0, 2] = 999.0    # padded token of item 0
        f2, d2 = T.masked_losses(pred, target2, flen, logd2, durs, tlen)
        self.assertAlmostEqual(f1.item(), f2.item(), places=6)
        self.assertAlmostEqual(d1.item(), d2.item(), places=6)

    def test_zero_error(self):
        pred = torch.randn(2, 4, 20)
        durs = torch.tensor([[2, 2], [1, 3]])
        logd = torch.log1p(durs.float())
        f, d = T.masked_losses(pred, pred.clone(), torch.tensor([4, 4]), logd, durs, torch.tensor([2, 2]))
        self.assertAlmostEqual(f.item(), 0.0, places=6)
        self.assertAlmostEqual(d.item(), 0.0, places=6)

    def test_delta_loss_zero_for_identical_and_ignored_on_padding(self):
        pred = torch.randn(2, 5, 20)
        target = torch.randn(2, 5, 20)
        flen = torch.tensor([4, 3])
        self.assertAlmostEqual(T.masked_delta_loss(pred, pred.clone(), flen).item(), 0.0, places=6)
        target2 = target.clone()
        target2[0, 4] = 999.0  # padded frame and diff
        self.assertAlmostEqual(T.masked_delta_loss(pred, target, flen).item(),
                               T.masked_delta_loss(pred, target2, flen).item(), places=6)
        jitter_pred = pred.clone()
        jitter_pred[0, 1] += 10.0
        self.assertGreater(T.masked_delta_loss(jitter_pred, target, flen).item(),
                           T.masked_delta_loss(pred, target, flen).item())


class TestLRAndOptimizer(unittest.TestCase):
    def _args(self, **kw):
        base = dict(steps=100, warmup=10, lr=3e-4, min_lr=3e-5, weight_decay=0.01)
        base.update(kw)
        return type("A", (), base)()

    def test_lr_warmup_then_cosine(self):
        a = self._args()
        self.assertAlmostEqual(T.lr_at(0, a), a.lr / a.warmup, places=10)
        self.assertAlmostEqual(T.lr_at(a.warmup - 1, a), a.lr, places=10)
        self.assertAlmostEqual(T.lr_at(a.steps, a), a.min_lr, places=9)

    def test_optimizer_decay_split(self):
        from acoustic import PiVoAcoustic
        model = PiVoAcoustic(config={"width": 32, "ff_width": 64}, num_voices=2)
        opt = T.create_optimizer(model, self._args())
        self.assertEqual(len(opt.param_groups), 2)
        self.assertEqual(opt.param_groups[0]["weight_decay"], 0.01)
        self.assertEqual(opt.param_groups[1]["weight_decay"], 0.0)
        decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
        for name, p in model.named_parameters():
            if "tokens.weight" in name:
                self.assertNotIn(id(p), decay_ids)

    def test_weight_source_required(self):
        # random init must never be the silent default: --from-scratch is explicit
        with self.assertRaises(SystemExit):
            T.parse_args(["--data", "d.jsonl", "--out", "o"])
        with self.assertRaises(SystemExit):
            T.parse_args(["--data", "d.jsonl", "--out", "o", "--from-scratch", "--pivo-ckpt", "x.pt"])
        ok = T.parse_args(["--data", "d.jsonl", "--out", "o", "--from-scratch"])
        self.assertTrue(ok.from_scratch)
        self.assertTrue(ok.compute_feature_stats)

    def test_seed_determinism(self):
        T.set_seed(7)
        a = torch.rand(3)
        T.set_seed(7)
        b = torch.rand(3)
        self.assertTrue(torch.equal(a, b))


class TestCheckpoint(unittest.TestCase):
    def test_atomic_save_roundtrip(self):
        from acoustic import PiVoAcoustic
        model = PiVoAcoustic(config={"width": 32, "ff_width": 64}, num_voices=2)
        opt = T.create_optimizer(model, type("A", (), dict(weight_decay=0.01, lr=1e-3))())
        args = type("A", (), {"data": "x", "out": "y"})()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "last.pt")
            T.save_checkpoint(path, model, opt, step=5, epoch=1, best=0.25, args=args)
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))  # os.replace moved it
            # strictly weights_only=True: no pickle payloads allowed (unsafe-free)
            ck = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(ck["format"], T.CKPT_FORMAT)
            self.assertEqual(ck["step"], 5)
            self.assertIn("optimizer", ck)
            self.assertIn("torch_rng_state", ck)
            fresh = PiVoAcoustic(config={"width": 32, "ff_width": 64}, num_voices=2)
            fresh.load_state_dict(ck["state_dict"], strict=True)

    def test_rng_state_saved_and_reproducible(self):
        from acoustic import PiVoAcoustic
        model = PiVoAcoustic(config={"width": 32, "ff_width": 64}, num_voices=2)
        opt = T.create_optimizer(model, type("A", (), dict(weight_decay=0.01, lr=1e-3))())
        args = type("A", (), {"data": "x", "out": "y"})()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "last.pt")
            T.set_seed(123)
            T.save_checkpoint(path, model, opt, step=1, epoch=0, best=0.0, args=args)
            a = torch.rand(3)
            ck = torch.load(path, map_location="cpu", weights_only=True)
            torch.set_rng_state(ck["torch_rng_state"])
            b = torch.rand(3)  # rewind to the state saved in the checkpoint
            self.assertTrue(torch.equal(a, b))


class TestNoTrainingOnMissingData(unittest.TestCase):
    def test_missing_resume_checkpoint_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [write_row(tmp, "a", [1], [3], feats(3)),
                    write_row(tmp, "b", [1], [3], feats(3), split="val")]
            data = write_jsonl(tmp, rows)
            out = os.path.join(tmp, "out")
            rc = T.main(["--data", data, "--out", out, "--steps", "5",
                         "--resume", os.path.join(tmp, "nope.pt"), "--device", "cpu"])
            self.assertEqual(rc, 1)  # explicit resume must never silently restart

    def test_missing_dataset_exits_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            rc = T.main(["--data", os.path.join(tmp, "absent.jsonl"),
                         "--out", out, "--steps", "5", "--from-scratch"])
            self.assertEqual(rc, 1)
            # no checkpoints, no log: training never started
            self.assertFalse(os.path.exists(out))

    def test_invalid_dataset_exits_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.jsonl")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write('{"id": "x", "split": "train"}\n')  # missing keys
            out = os.path.join(tmp, "out")
            rc = T.main(["--data", bad, "--out", out, "--steps", "5", "--from-scratch"])
            self.assertEqual(rc, 1)
            self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
