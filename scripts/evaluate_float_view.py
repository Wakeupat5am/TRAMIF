import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.compact_view_net import CompactViewNet
from scripts.historical_policy_dataset import HistoricalPolicyDataset


ROOT = Path(__file__).resolve().parents[1]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed compact branch on filtered validation."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument(
        "--month", choices=("2020-02", "2020-03"), required=True
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = sha256_bytes(checkpoint_bytes)

    # Check the checkpoint against the training history.
    with (checkpoint_path.parent / "history.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        matches = [
            row for row in csv.DictReader(handle)
            if row["checkpoint"] == checkpoint_path.name
        ]

    if len(matches) != 1:
        raise ValueError("Checkpoint must have one training-history entry.")
    if matches[0]["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Checkpoint checksum differs from training history.")

    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location="cpu",
        weights_only=True,
    )
    del checkpoint_bytes

    config = checkpoint["config"]
    if checkpoint["epoch"] != 100 or config["epochs"] != 100:
        raise ValueError("Expected the fixed epoch-100 checkpoint.")
    if config.get("evaluation_checkpoint_epoch") != 100:
        raise ValueError("Checkpoint selection rule is not epoch 100.")
    if config.get("model") != "CompactViewNet":
        raise ValueError("Expected CompactViewNet.")

    for key in (
        "validation_used_for_weight_updates",
        "validation_evaluated_during_training",
        "future_test_used",
    ):
        if config.get(key) is not False:
            raise ValueError(f"Unsupported training configuration: {key}")

    view = config["view"]
    if view not in {"raw-byte", "entropy"}:
        raise ValueError("Unexpected checkpoint view.")

    dataset = HistoricalPolicyDataset(
        project_root=ROOT,
        policy_dir=args.policy_dir,
        view=view,
        split="validation",
        month=args.month,
        threshold=0,
    )
    expected_counts = {"2020-02": 3799, "2020-03": 4332}
    if len(dataset) != expected_counts[args.month]:
        raise ValueError("Unexpected filtered validation count.")

    if dataset.preprocessing_config_id != config["dataset_config_sha256"]:
        raise ValueError("Preprocessing differs from checkpoint.")
    if dataset.class_to_idx != config["class_to_idx"]:
        raise ValueError("Class mapping differs from checkpoint.")
    if dataset.num_classes != config["num_classes"]:
        raise ValueError("Class count differs from checkpoint.")

    # Verify the original training selection without reading training images.
    train_dataset = HistoricalPolicyDataset(
        project_root=ROOT,
        policy_dir=args.policy_dir,
        view=view,
        split="train",
        threshold=0,
    )
    if train_dataset.selection != config["train_selection"]:
        raise ValueError("Training selection differs from checkpoint.")
    if train_dataset.selection_id != config["train_selection_sha256"]:
        raise ValueError("Training selection checksum mismatch.")
    del train_dataset

    required_sources = (
        "models/compact_view_net.py",
        "models/malsbslcnet.py",
        "scripts/float_image_dataset.py",
        "scripts/historical_policy_dataset.py",
    )
    for relative_path in required_sources:
        actual = sha256_bytes((ROOT / relative_path).read_bytes())
        if actual != config["source_sha256"][relative_path]:
            raise ValueError(f"Source differs from training: {relative_path}")

    model = CompactViewNet(num_classes=dataset.num_classes)
    if sum(p.numel() for p in model.parameters()) != config["parameter_count"]:
        raise ValueError("Model parameter count mismatch.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint

    device = torch.device("cuda")
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    print("Purpose: filtered historical validation")
    print("View:", view)
    print("Month:", args.month)
    print("Policy threshold: 0")
    print("Checkpoint epoch: 100")
    print("Samples:", len(dataset))
    print("Model mode: eval")
    print("Temperature: 1.0 (uncalibrated)")
    print("Starting evaluation...", flush=True)

    logits_parts = []
    label_parts = []
    seen = 0
    torch.cuda.synchronize(device)
    started = perf_counter()

    with torch.inference_mode():
        for step, (images, labels) in enumerate(loader, start=1):
            scores = model(images.to(device, non_blocking=True))
            if scores.shape != (len(labels), dataset.num_classes):
                raise RuntimeError("Unexpected logits shape.")
            if not torch.isfinite(scores).all().item():
                raise RuntimeError("Non-finite logits.")

            logits_parts.append(scores.cpu())
            label_parts.append(labels.clone())
            seen += len(labels)

            if step % 25 == 0 or step == len(loader):
                print(f"Evaluated: {seen}/{len(dataset)}", flush=True)

    torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    if seen != len(dataset):
        raise RuntimeError("Evaluation sample count mismatch.")

    logits = torch.cat(logits_parts)
    labels = torch.cat(label_parts)
    predictions = logits.argmax(dim=1)

    expected_labels = torch.tensor([
        dataset.class_to_idx[row["family"]]
        for row in dataset.samples
    ], dtype=torch.int64)
    if not torch.equal(labels, expected_labels):
        raise RuntimeError("Labels differ from manifest order.")

    classes = dataset.num_classes
    confusion = torch.bincount(
        labels * classes + predictions,
        minlength=classes * classes,
    ).reshape(classes, classes)

    if confusion.sum().item() != seen:
        raise RuntimeError("Confusion matrix count mismatch.")

    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted_count = matrix.sum(dim=0)
    supported = support > 0

    recall = true_positive / support.clamp_min(1)
    f1 = 2 * true_positive / (support + predicted_count).clamp_min(1)
    accuracy = true_positive.sum().item() / seen
    macro_supported = f1[supported].mean().item()
    macro_fixed = f1.mean().item()
    balanced_accuracy = recall[supported].mean().item()
    nll = F.cross_entropy(logits.to(torch.float64), labels).item()

    inverse_mapping = {
        index: family for family, index in dataset.class_to_idx.items()
    }
    class_names = [inverse_mapping[index] for index in range(classes)]
    absent = [
        class_names[index] for index in range(classes)
        if not supported[index].item()
    ]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = checkpoint_path.parent / (
        f"validation_filtered_{args.month}_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    write_csv(
        output_dir / "confusion_matrix.csv",
        ["true_family", *class_names],
        [
            [class_names[index], *row]
            for index, row in enumerate(confusion.tolist())
        ],
    )
    write_csv(
        output_dir / "per_family.csv",
        [
            "class_index", "family", "true_support",
            "predicted_count", "recall", "f1_zero_if_undefined",
        ],
        [
            [
                index,
                class_names[index],
                int(support[index].item()),
                int(predicted_count[index].item()),
                recall[index].item() if supported[index].item() else "",
                f1[index].item(),
            ]
            for index in range(classes)
        ],
    )

    predictions_path = output_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        sample_identifiers=np.array([
            row["sha"] for row in dataset.samples
        ]),
        logits=logits.numpy(),
        labels=labels.numpy(),
        predictions=predictions.numpy(),
        class_names=np.array(class_names),
    )

    report = {
        "purpose": "filtered historical validation of fixed compact branch",
        "view": view,
        "split": "validation",
        "month": args.month,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": 100,
        "training_seed": config["seed"],
        "dataset_config_sha256": dataset.preprocessing_config_id,
        "selection": dataset.selection,
        "selection_sha256": dataset.selection_id,
        "class_to_idx": dataset.class_to_idx,
        "samples": seen,
        "supported_class_count": int(supported.sum().item()),
        "absent_families": absent,
        "accuracy": accuracy,
        "macro_f1_supported_classes": macro_supported,
        "macro_f1_fixed_51_zero_if_undefined": macro_fixed,
        "balanced_accuracy_supported_classes": balanced_accuracy,
        "uncalibrated_nll": nll,
        "temperature": 1.0,
        "batch_size": 64,
        "model_mode": "eval",
        "weight_updates": False,
        "future_test_used": False,
        "elapsed_inference_pipeline_seconds": elapsed,
        "predictions_sha256": sha256_bytes(predictions_path.read_bytes()),
        "evaluator_source_sha256": sha256_bytes(
            Path(__file__).resolve().read_bytes()
        ),
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "note": (
            "Development validation for one seed. "
            "Timing includes data loading and is not pure model latency."
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("View:", view)
    print("Month:", args.month)
    print("Evaluated samples:", seen)
    print("Supported classes:", report["supported_class_count"])
    print("Absent families:", absent)
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Macro F1 (supported classes): {macro_supported:.4f}")
    print(f"Macro F1 (fixed 51 classes): {macro_fixed:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.2%}")
    print(f"Uncalibrated NLL: {nll:.4f}")
    print("Report:", report_path)
    print("Outputs:", output_dir)
    print("No weight updates. No future-test data used.")


if __name__ == "__main__":
    main()