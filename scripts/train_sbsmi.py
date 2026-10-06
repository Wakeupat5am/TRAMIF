import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.malsbslcnet import MalSBSLCNet
from scripts.sbsmi_dataset import SBSMIDataset


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train-only engineering run for the SBSMI baseline."
    )
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    project_root = Path(__file__).resolve().parents[1]

    seed = 42
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-6

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Record repeatability settings; do not claim cross-machine identity.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = SBSMIDataset(project_root, split="train")

    if len(dataset) != 18061:
        raise ValueError("Expected 18,061 training samples.")

    # BatchNorm1d cannot train on a batch containing only one sample.
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
    model = MalSBSLCNet(num_classes=dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        project_root
        / "data"
        / "derived"
        / "sbsmi_training"
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    source_paths = [
        Path(__file__).resolve(),
        project_root / "models" / "malsbslcnet.py",
        project_root / "scripts" / "sbsmi_dataset.py",
    ]

    config = {
        "purpose": "train-only engineering run",
        "model": "MalSBSLCNet",
        "num_classes": dataset.num_classes,
        "train_samples": len(dataset),
        "dataset_config_sha256": dataset.config_id,
        "class_to_idx": dataset.class_to_idx,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "drop_last": False,
        "shuffle": True,
        "num_workers": 0,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss": "unweighted CrossEntropyLoss",
        "scheduler": None,
        "mixed_precision": False,
        "checkpoint_rule": "save at each completed epoch",
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
            str(path.relative_to(project_root)): file_sha256(path)
            for path in source_paths
        },
    }
    save_json(output_dir / "config.json", config)

    print("Purpose:", config["purpose"])
    print("GPU:", config["gpu"])
    print("Train samples:", len(dataset))
    print("Batch size:", batch_size)
    print("Batches per epoch:", len(loader))
    print("Epochs:", args.epochs)
    print("Output folder:", output_dir)
    print("Starting training...", flush=True)

    history_path = output_dir / "history.csv"
    torch.cuda.reset_peak_memory_stats(device)

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
            ],
        )
        writer.writeheader()
        file.flush()

        for epoch in range(1, args.epochs + 1):
            model.train()
            start = perf_counter()
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

                # Check gradient norm without imposing a clipping limit.
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
            elapsed = perf_counter() - start

            if seen != len(dataset):
                raise RuntimeError("Not all training samples were processed.")

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

            # Confirm that the saved model state can be loaded unchanged.
            saved = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            if saved["epoch"] != epoch:
                raise RuntimeError("Checkpoint epoch mismatch.")

            for name, value in model.state_dict().items():
                if not torch.equal(
                    value.detach().cpu(),
                    saved["model_state_dict"][name],
                ):
                    raise RuntimeError(f"Checkpoint mismatch: {name}")

            del saved, checkpoint

            writer.writerow({
                "epoch": epoch,
                "samples_seen": seen,
                "online_train_loss": loss_sum / seen,
                "online_train_accuracy": correct / seen,
                "epoch_seconds": elapsed,
                "peak_gpu_allocated_mib": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                ),
                "checkpoint": checkpoint_path.name,
            })
            file.flush()

            print(f"Epoch finished in {elapsed:.1f}s")
            print("Checkpoint save/load: PASS")
            print("Checkpoint:", checkpoint_path, flush=True)

    print()
    print("Training run completed.")
    print("History:", history_path)
    print("No validation or future-test evaluation performed.")


if __name__ == "__main__":
    main()