import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.compact_view_net import CompactViewNet
from scripts.float_image_dataset import FloatImageDataset


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train-only engineering run for a compact image branch."
    )
    parser.add_argument(
        "--view",
        choices=("raw-byte", "entropy"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    seed = 42
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-6

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = FloatImageDataset(
        ROOT,
        view=args.view,
        split="train",
    )

    if len(dataset) != 18061:
        raise ValueError("Expected 18,061 training samples.")
    if len(dataset) % batch_size == 1:
        raise ValueError("The final batch would contain only one sample.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )

    device = torch.device("cuda")
    model = CompactViewNet(num_classes=dataset.num_classes).to(device)

    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 59059:
        raise RuntimeError("Unexpected compact model parameter count.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        ROOT / "data/derived/float_view_training" / args.view / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "models/compact_view_net.py",
        ROOT / "models/malsbslcnet.py",
        ROOT / "scripts/float_image_dataset.py",
    ]

    config = {
        "purpose": "train-only engineering run",
        "near_duplicate_policy_status": "not finalized",
        "view": args.view,
        "model": "CompactViewNet",
        "feature_dim": model.feature_dim,
        "parameter_count": parameter_count,
        "num_classes": dataset.num_classes,
        "class_to_idx": dataset.class_to_idx,
        "train_samples": len(dataset),
        "dataset_config_sha256": dataset.config_id,
        "input_shape": [1, 64, 64],
        "input_normalization": "already normalized; no additional scaling",
        "initialization": "fresh random initialization",
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": False,
        "num_workers": 0,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss": "unweighted CrossEntropyLoss",
        "scheduler": None,
        "augmentation": None,
        "mixed_precision": False,
        "checkpoint_rule": "save at each completed epoch",
        "resume_supported": False,
        "validation_used": False,
        "future_test_used": False,
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): file_sha256(path)
            for path in source_paths
        },
    }
    save_json(output_dir / "config.json", config)

    print("Purpose:", config["purpose"])
    print("View:", args.view)
    print("GPU:", config["gpu"])
    print("Parameters:", parameter_count)
    print("Train samples:", len(dataset))
    print("Batch size:", batch_size)
    print("Batches per epoch:", len(loader))
    print("Epochs:", args.epochs)
    print("Output folder:", output_dir)
    print("Starting training...", flush=True)

    history_path = output_dir / "history.csv"

    with history_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "samples_seen",
                "online_train_loss",
                "online_train_accuracy",
                "epoch_seconds",
                "peak_gpu_allocated_mib",
                "checkpoint",
                "checkpoint_sha256",
            ],
        )
        writer.writeheader()
        file.flush()

        for epoch in range(1, args.epochs + 1):
            model.train()
            classifier_before = (
                model.classifier.weight.detach().cpu().clone()
            )

            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = perf_counter()

            loss_sum = 0.0
            correct = 0
            seen = 0

            for step, (images, labels) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits = model(images)
                loss = criterion(logits, labels)

                if not torch.isfinite(loss).item():
                    raise RuntimeError(
                        f"Non-finite loss: epoch {epoch}, batch {step}"
                    )

                loss.backward()

                # Reject non-finite gradients without imposing clipping.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float("inf"),
                    error_if_nonfinite=True,
                )
                optimizer.step()

                count = labels.size(0)
                loss_sum += loss.detach().item() * count
                correct += (
                    logits.detach().argmax(dim=1) == labels
                ).sum().item()
                seen += count

                if step == 1 or step % 100 == 0 or step == len(loader):
                    print(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"batch {step}/{len(loader)} | "
                        f"loss={loss_sum / seen:.4f} | "
                        f"online train accuracy={correct / seen:.2%}",
                        flush=True,
                    )

            torch.cuda.synchronize(device)
            elapsed = perf_counter() - started
            peak_gpu_mib = (
                torch.cuda.max_memory_allocated(device) / 1024**2
            )

            if seen != len(dataset):
                raise RuntimeError("Not all training samples were processed.")

            if torch.equal(
                classifier_before,
                model.classifier.weight.detach().cpu(),
            ):
                raise RuntimeError("Classifier weights did not change.")

            for name, value in model.state_dict().items():
                if not torch.isfinite(value).all().item():
                    raise RuntimeError(f"Non-finite model state: {name}")

            checkpoint_path = output_dir / f"epoch_{epoch:03d}.pt"
            temporary_path = checkpoint_path.with_suffix(".pt.tmp")

            checkpoint = {
                "epoch": epoch,
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
                "loader_generator_state": generator.get_state(),
            }

            torch.save(checkpoint, temporary_path)
            temporary_path.replace(checkpoint_path)

            saved = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            if saved["epoch"] != epoch or saved["config"] != config:
                raise RuntimeError("Checkpoint metadata mismatch.")

            current_state = model.state_dict()
            saved_state = saved["model_state_dict"]

            if set(current_state) != set(saved_state):
                raise RuntimeError("Checkpoint parameter names differ.")

            for name, value in current_state.items():
                if not torch.equal(value.detach().cpu(), saved_state[name]):
                    raise RuntimeError(f"Checkpoint mismatch: {name}")

            del saved, saved_state, current_state, checkpoint

            writer.writerow({
                "epoch": epoch,
                "samples_seen": seen,
                "online_train_loss": loss_sum / seen,
                "online_train_accuracy": correct / seen,
                "epoch_seconds": elapsed,
                "peak_gpu_allocated_mib": peak_gpu_mib,
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": file_sha256(checkpoint_path),
            })
            file.flush()

            print(f"Epoch finished in {elapsed:.1f}s")
            print("Samples processed:", seen)
            print("Optimizer changed classifier weights: PASS")
            print("Checkpoint save/load: PASS")
            print("Checkpoint:", checkpoint_path, flush=True)

    print("\nTraining run completed.")
    print("History:", history_path)
    print("No validation or future-test evaluation performed.")


if __name__ == "__main__":
    main()