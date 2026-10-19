import ast
import copy
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from scripts import train_prefix_view as t
from scripts import episode_training_dataset as d
from test_episode_training_dataset import Fixture


class IDDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return str(i)


class TestTrainPrefixView(unittest.TestCase):
    def test_cpu_synthetic_shapes_all_six_prefix_models_no_state_updates(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No steps")):
                for classes in (32, 42):
                    for view in d.VIEWS:
                        with self.subTest(classes=classes, view=view), torch.random.fork_rng(devices=[]):
                            torch.manual_seed(42)
                            model = t.make_model(view, classes).eval()
                            before = {k: v.clone() for k, v in model.state_dict().items()}
                            with torch.no_grad():
                                logits = model(torch.zeros(2, 1, 64, 64))
                                single = model(torch.zeros(1, 1, 64, 64))
                                loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, classes - 1]))
                            self.assertEqual(logits.shape, (2, classes))
                            self.assertEqual(single.shape, (1, classes))
                            self.assertTrue(torch.isfinite(loss).item())
                            self.assertTrue(all(p.grad is None for p in model.parameters()))
                            t.assert_equal_state(before, model.state_dict())
        finally:
            torch.set_num_threads(previous)

    def test_optimizer_defaults_no_step_and_environment(self):
        model = nn.Linear(2, 2)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No steps")):
            optimizer = t.make_optimizer(model)
        self.assertEqual(optimizer.defaults, t.EXPECTED_ADAM)
        self.assertEqual(len(optimizer.state), 0)
        t.assert_equal_state(before, model.state_dict())
        self.assertTrue(t.environment()["version_parity"])
        with patch.object(t, "EXPECTED_ADAM", {"lr": 0.1}):
            with self.assertRaises(ValueError):
                t.make_optimizer(model)

    def test_deterministic_multi_epoch_shuffle_coverage_and_tails(self):
        for n, batches, tail in ((7034, 220, 26), (9803, 307, 11)):
            loader1, _ = t.make_loader(IDDataset(n), pin_memory=False)
            loader2, _ = t.make_loader(IDDataset(n), pin_memory=False)
            previous = None
            for _ in range(2):
                first, second = list(loader1), list(loader2)
                self.assertEqual(first, second)
                self.assertEqual(len(first), batches)
                self.assertEqual(len(first[-1]), tail)
                order = [x for b in first for x in b]
                digest = t.check_coverage(order, [str(i) for i in range(n)])
                if previous is not None:
                    self.assertNotEqual(previous, digest)
                previous = digest
        with self.assertRaises(ValueError):
            t.make_loader(IDDataset(33), pin_memory=False)

    def test_bad_coverage(self):
        for observed in (["a"], ["a", "a"], ["a", "c"]):
            with self.assertRaises(ValueError):
                t.check_coverage(observed, ["a", "b"])

    def test_synthetic_checkpoint_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = nn.Linear(2, 3)
            optimizer = t.make_optimizer(model)
            payload = {"epoch": 100, "config": dict(t.RUN_SPEC),
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(), "python_rng_state": (3, (1, 2), None)}
            path = Path(tmp) / "epoch_100.pt"
            hash_value = t.save_checkpoint(path, payload)
            self.assertEqual(hash_value, d.sha256(path.read_bytes()))
            self.assertFalse(path.with_suffix(".pt.tmp").exists())
            with self.assertRaises(FileExistsError):
                t.save_checkpoint(path, payload)

    def test_checkpoint_corruption_and_nonfinite_cannot_be_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            with self.assertRaises(ValueError):
                t.save_checkpoint(path, {"value": torch.tensor([float("nan")])})
            self.assertFalse(path.exists())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            with patch.object(torch, "load", return_value={"epoch": 1}):
                with self.assertRaises(ValueError):
                    t.save_checkpoint(path, {"epoch": 100})
            self.assertFalse(path.exists())

    def test_preflight_roundtrip_requires_all_views_and_current_bindings(self):
        with Fixture() as f, patch.object(t, "source_hashes", return_value={"fixture": "hash"}):
            receipt = d.preflight("E1", f.root)
            datasets = t.verify_preflight("E1", receipt, f.root)
            self.assertEqual(set(datasets), set(d.VIEWS))
            for field, value in (("case", "E2"), ("status", "FAIL"), ("code_sha256", {})):
                bad = copy.deepcopy(receipt)
                bad[field] = value
                with self.assertRaises(ValueError):
                    t.verify_preflight("E1", bad, f.root)
            bad = copy.deepcopy(receipt)
            del bad["views"]["entropy"]
            with self.assertRaises(ValueError):
                t.verify_preflight("E1", bad, f.root)
            bad = copy.deepcopy(receipt)
            bad["views"]["sbsmi"]["binding"]["class_to_idx"] = {"a": 1, "b": 0}
            with self.assertRaises(ValueError):
                t.verify_preflight("E1", bad, f.root)

    def test_import_and_execution_guard_never_train(self):
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No optimizer step")), \
             patch.object(torch.Tensor, "backward", side_effect=AssertionError("No backward")), \
             patch.object(torch, "load", side_effect=AssertionError("No checkpoint load")), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("No CUDA probe")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("No input reads")):
            importlib.reload(t)
            importlib.reload(d)
            with self.assertRaises(PermissionError):
                t.train("E1", "sbsmi", "nonexistent", "0" * 64)

    def test_no_configurable_epochs_or_warm_start(self):
        self.assertEqual(t.RUN_SPEC["epochs"], 100)
        self.assertEqual(t.RUN_SPEC["selected_epoch"], 100)
        self.assertFalse(t.RUN_SPEC["resume"])
        self.assertFalse(t.RUN_SPEC["warm_start"])
        tree = ast.parse(Path(t.__file__).read_text(encoding="utf-8"))
        # Only the explicitly invoked train function contains optimizer.step.
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name != "train":
                for call in ast.walk(node):
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                        self.assertNotEqual(call.func.attr, "step")

    def test_failure_record_preserved_before_any_model_or_optimizer_execution(self):
        # Mock readiness; inject a constructor failure before .to(cuda) or any step.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt_path = root / "data/derived/prefix_preflight_v1/fixture.json"
            receipt_path.parent.mkdir(parents=True)
            receipt_path.write_bytes(d.json_bytes({"synthetic": True}))
            with patch.object(t, "ROOT", root), \
                 patch.object(t, "verify_preflight", return_value={"sbsmi": type("D", (), {"num_classes": 32})()}), \
                 patch.object(t, "source_hashes", return_value={}), \
                 patch.object(torch.cuda, "is_available", return_value=True), \
                 patch.object(torch.cuda, "manual_seed_all"), \
                 patch.object(t, "make_model", side_effect=RuntimeError("synthetic injected failure")), \
                 patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No steps")):
                old_benchmark = torch.backends.cudnn.benchmark
                old_deterministic = torch.backends.cudnn.deterministic
                try:
                    with self.assertRaisesRegex(RuntimeError, "synthetic injected failure"):
                        t.train("E1", "sbsmi", receipt_path, d.sha256(receipt_path.read_bytes()),
                                execute_training=True)
                finally:
                    torch.backends.cudnn.benchmark = old_benchmark
                    torch.backends.cudnn.deterministic = old_deterministic
            runs = list((root / "data/derived/prefix_training_v1/E1/sbsmi").iterdir())
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0] / "STARTED.json").is_file())
            failure = d.parse_json((runs[0] / "FAILED.json").read_bytes())
            self.assertFalse(failure["automatic_retry"])
            self.assertFalse((runs[0] / "COMPLETE.json").exists())
            self.assertFalse(list(runs[0].glob("*.pt")))


if __name__ == "__main__":
    unittest.main()
