"""Prefix trainer implementation. Training requires a separate owner approval.

Import is inert. The train CLI additionally requires --execute-training and a
hash-pinned, current all-view image-preflight receipt. No resume/warm-start path.
The explicit flag is an operator acknowledgement, not proof of owner approval.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.compact_view_net import CompactViewNet
from models.malsbslcnet import MalSBSLCNet
from scripts.episode_training_dataset import (
    ROOT, EPISODES, VIEWS, EpisodeTrainingDataset, json_bytes, parse_json,
    read_bound, require_hash, sha256,
)

RUN_SPEC = {"seed": 42, "batch_size": 32, "epochs": 100,
    "selected_epoch": 100, "learning_rate": 1e-4, "weight_decay": 1e-6,
    "loss": "unweighted CrossEntropyLoss", "scheduler": None,
    "augmentation": None, "mixed_precision": False, "shuffle": True,
    "num_workers": 0, "pin_memory": True, "drop_last": False,
    "initialization": "fresh", "resume": False, "warm_start": False,
    "selection": "fixed epoch100; no metric-based selection"}
EXPECTED_ADAM = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8,
    "weight_decay": 1e-6, "amsgrad": False, "maximize": False, "foreach": None,
    "capturable": False, "differentiable": False, "fused": None,
    "decoupled_weight_decay": False}
MODEL_HASHES = {
    "models/malsbslcnet.py": "dd897abb80f41de30bfde7b54a0ef7da32a1fcecd24149ef4838fef6271e11eb",
    "models/compact_view_net.py": "a555e8e81c596f8853ccd0cf2fef4120a5931ca3df23c6096cfc9f9103c29bfa",
}
PROTOCOL_HASHES = {
    "docs/RESEARCH_PROTOCOL_V1.md": "e64664c4c499ef762b4a52c45889f0d05db6741ec5021b249f2072585f816546",
}
SOURCE_FILES = ("scripts/episode_training_dataset.py", "scripts/train_prefix_view.py",
    "models/malsbslcnet.py", "models/compact_view_net.py",
    "tests/test_episode_training_dataset.py", "tests/test_train_prefix_view.py",
    "docs/RESEARCH_PROTOCOL_V1.md")


def source_hashes(root=ROOT):
    anchors = {**MODEL_HASHES, **PROTOCOL_HASHES}
    return {name: sha256(read_bound(root, name, anchors.get(name))) for name in SOURCE_FILES}


def environment():
    # No CUDA initialization or device queries here.
    result = {"python": platform.python_version(), "torch": str(torch.__version__),
              "numpy": np.__version__, "cuda_runtime": torch.version.cuda}
    result["version_parity"] = result == {"python": "3.11.9", "torch": "2.13.0+cu126",
                                         "numpy": "2.4.6", "cuda_runtime": "12.6"}
    return result


def make_model(view, classes):
    if view not in VIEWS or classes not in (32, 42):
        raise ValueError("Expected prefix view and K32/K42")
    model = MalSBSLCNet(classes) if view == "sbsmi" else CompactViewNet(classes)
    width = 2048 if view == "sbsmi" else 512
    if tuple(model.classifier.weight.shape) != (classes, width):
        raise ValueError("Classifier architecture mismatch")
    return model


def make_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    if optimizer.defaults != EXPECTED_ADAM:
        raise ValueError("Effective Adam defaults changed: owner review required")
    return optimizer


def make_loader(dataset, pin_memory=True):
    if len(dataset) % 32 == 1:
        raise ValueError("BatchNorm singleton final batch")
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False,
                        num_workers=0, pin_memory=pin_memory, generator=generator)
    return loader, generator


def check_coverage(observed, expected):
    if len(observed) != len(expected) or len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise ValueError("Epoch coverage/duplicate/unknown ID error")
    return sha256(json_bytes(observed))


def write_new_json(path, value):
    with Path(path).open("xb") as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def assert_equal_state(a, b):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.detach().cpu(), b.cpu()):
            raise ValueError("Checkpoint tensor mismatch")
        if not torch.isfinite(a).all().item():
            raise ValueError("Non-finite checkpoint state")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError("Checkpoint keys mismatch")
        for key in a:
            assert_equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError("Checkpoint sequence mismatch")
        for x, y in zip(a, b):
            assert_equal_state(x, y)
    elif type(a) is not type(b) or a != b:
        raise ValueError("Checkpoint metadata mismatch")


def save_checkpoint(path, payload):
    """Only round-trip newly written state; never load a registered checkpoint."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    saved = torch.load(temporary, map_location="cpu", weights_only=True)
    assert_equal_state(payload, saved)
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)
    return sha256(path.read_bytes())


def verify_preflight(case, receipt, root=ROOT):
    if (receipt.get("schema") != 1 or receipt.get("status") != "PASS"
            or receipt.get("purpose") != "prefix_train_image_preflight"
            or receipt.get("case") != case or set(receipt.get("views", {})) != set(VIEWS)
            or receipt.get("code_sha256") != source_hashes(root)):
        raise ValueError("Stale/incomplete/wrong-case preflight receipt")
    datasets = {}
    for view in VIEWS:
        item = receipt["views"][view]
        dataset = EpisodeTrainingDataset(case, view, root, image_receipts=item["images"])
        if item["binding"] != dataset.binding:
            raise ValueError("Preflight binding mismatch")
        for image, sample in zip(item["images"], dataset.samples):
            if image["content_sha256"] != sample.content_sha256:
                raise ValueError("Preflight content identity mismatch")
            require_hash(image["image_sha256"])
            require_hash(image["report_sha256"])
        datasets[view] = dataset
    if any(d.samples != datasets["sbsmi"].samples for d in datasets.values()):
        raise ValueError("Preflight cross-view alignment mismatch")
    return datasets


def git_snapshot(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)
    return {"head": git("rev-parse", "HEAD").decode().strip(),
            "status": git("status", "--short").decode(),
            "tracked_diff_sha256": sha256(git("diff", "HEAD", "--", "scripts", "models", "tests"))}


def train(case, view, receipt_path, receipt_hash, *, execute_training=False):
    """Not executed during implementation verification; explicit operator opt-in."""
    if not execute_training:
        raise PermissionError("Separate owner training approval and explicit execution required")
    if case not in EPISODES or view not in VIEWS:
        raise ValueError("Unknown prefix case/view")
    require_hash(receipt_hash)
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt_root = (ROOT / "data/derived/prefix_preflight_v1").resolve()
    if not receipt_path.is_relative_to(receipt_root):
        raise ValueError("Expected a prefix_preflight_v1 receipt")
    receipt = parse_json(read_bound(ROOT, receipt_path.relative_to(ROOT), receipt_hash))
    datasets = verify_preflight(case, receipt)
    env = environment()
    if not env["version_parity"]:
        raise RuntimeError("Environment parity unresolved; stop before training")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    dataset = datasets[view]
    code = source_hashes()
    # Every run is new, including after interruption; no resume or overwrite.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "data/derived/prefix_training_v1" / case / view / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "STARTED.json", {"case": case, "view": view, "run": run_id})
    try:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = make_model(view, dataset.num_classes).to("cuda")
        optimizer = make_optimizer(model)
        criterion = nn.CrossEntropyLoss()
        loader, generator = make_loader(dataset)
        config = {"run_spec": RUN_SPEC, "case": case, "view": view,
            "dataset": dataset.binding, "preflight_sha256": receipt_hash,
            "code_sha256": code, "environment": env, "git": git_snapshot(ROOT),
            "adam_defaults": optimizer.defaults, "parameters": sum(p.numel() for p in model.parameters()),
            "gpu": torch.cuda.get_device_name(0), "cudnn_deterministic": True,
            "cudnn_benchmark": False, "future_test_used": False,
            "design_reference": "docs/g3cp_prefix_training_design.md; D025",
            "owner_training_approval": "operator acknowledged; external approval required"}
        config_hash = sha256(json_bytes(config))
        write_new_json(output / "config.json", config)
        write_new_json(output / "preflight.json", receipt)
        write_new_json(output / "train_samples.json", [vars(s) for s in dataset.samples])
        # Copy source text, not merely HEAD (the worktree may be dirty/untracked).
        for name, expected in code.items():
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(read_bound(ROOT, name, expected))
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        initial_hash = save_checkpoint(output / "initial_state.pt", {"model_state_dict": initial,
            "config_sha256": config_hash, "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "loader_generator_state": generator.get_state(), "python_rng_state": random.getstate()})
        del initial
        write_new_json(output / "initial_state.json", {"sha256": initial_hash})
        expected_ids = [s.sha for s in dataset.samples]
        registry = []
        for epoch in range(1, 101):
            model.train()
            observed, losses, correct = [], 0.0, 0
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = perf_counter()
            for images, labels, identifiers in loader:
                images, labels = images.to("cuda", non_blocking=True), labels.to("cuda", non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                optimizer.step()
                observed.extend(identifiers)
                losses += loss.detach().item() * len(labels)
                correct += (logits.detach().argmax(1) == labels).sum().item()
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            order_hash = check_coverage(observed, expected_ids)
            # Detect source/input metadata mutation before registering each epoch.
            if source_hashes() != code:
                raise RuntimeError("Source changed during training")
            current = EpisodeTrainingDataset(case, view)
            if current.binding != dataset.binding or current.samples != dataset.samples:
                raise RuntimeError("TRAIN binding changed during training")
            numpy_state = np.random.get_state()
            payload = {"epoch": epoch, "config": config, "config_sha256": config_hash,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                "loader_generator_state": generator.get_state(), "order_sha256": order_hash}
            path = output / f"epoch_{epoch:03d}.pt"
            checkpoint_hash = save_checkpoint(path, payload)
            info = {"epoch": epoch, "samples": len(observed), "steps": len(loader),
                "loss": losses / len(observed), "online_accuracy": correct / len(observed),
                "loop_seconds": elapsed, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "order_sha256": order_hash, "checkpoint": path.name, "sha256": checkpoint_hash}
            write_new_json(output / f"epoch_{epoch:03d}.json", info)
            registry.append(info)
            print(f"{case} {view} epoch {epoch}/100: saved {path.name}", flush=True)
        write_new_json(output / "COMPLETE.json", {"epochs": registry, "selected": registry[-1],
                                                   "config_sha256": config_hash})
    except BaseException as exc:
        write_new_json(output / "FAILED.json", {"error_type": type(exc).__name__, "message": str(exc),
            "automatic_retry": False, "resume_supported": False})
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("environment", help="CPU-only versions and Adam defaults; no step")
    run = sub.add_parser("train", help="Requires separate owner training approval")
    run.add_argument("--case", choices=tuple(EPISODES), required=True)
    run.add_argument("--view", choices=VIEWS, required=True)
    run.add_argument("--preflight-receipt", type=Path, required=True)
    run.add_argument("--preflight-sha256", required=True)
    run.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    if args.command == "environment":
        optimizer = make_optimizer(nn.Linear(1, 1))
        print(json.dumps({"environment": environment(), "adam": optimizer.defaults}, indent=2))
    else:
        train(args.case, args.view, args.preflight_receipt, args.preflight_sha256,
              execute_training=args.execute_training)


if __name__ == "__main__":
    main()
