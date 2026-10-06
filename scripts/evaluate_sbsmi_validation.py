import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.malsbslcnet import MalSBSLCNet
from scripts.sbsmi_dataset import SBSMIDataset


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed epoch-100 SBSMI baseline."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--month",
        choices=["2020-02", "2020-03"],
        default="2020-03",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]
    epoch = checkpoint["epoch"]

    if epoch != 100 or config["epochs"] != 100:
        raise ValueError("This evaluation requires the epoch-100 baseline.")

    if config.get("validation_used") is not False:
        raise ValueError("Checkpoint does not confirm train-only fitting.")

    if config.get("future_test_used") is not False:
        raise ValueError("Checkpoint does not confirm exclusion of future test.")

    dataset = SBSMIDataset(
        project_root,
        split="validation",
        month=args.month,
    )

    expected_counts = {"2020-02": 3825, "2020-03": 4438}

    if len(dataset) != expected_counts[args.month]:
        raise ValueError("Unexpected validation sample count.")

    if dataset.config_id != config["dataset_config_sha256"]:
        raise ValueError("Dataset configuration differs from checkpoint.")

    if dataset.class_to_idx != config["class_to_idx"]:
        raise ValueError("Class mapping differs from checkpoint.")

    if dataset.num_classes != config["num_classes"]:
        raise ValueError("Class count differs from checkpoint.")

    for relative_path, expected_hash in config["source_sha256"].items():
        source_path = project_root / relative_path

        if source_path.name in {"malsbslcnet.py", "sbsmi_dataset.py"}:
            if file_sha256(source_path) != expected_hash:
                raise ValueError(f"Source changed: {relative_path}")

    device = torch.device("cuda")
    model = MalSBSLCNet(num_classes=dataset.num_classes)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    del checkpoint

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    print("Purpose: historical validation of the fixed baseline")
    print("Month:", args.month)
    print("Checkpoint epoch:", epoch)
    print("Samples:", len(dataset))
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model mode: eval")
    print("Temperature: 1.0 (uncalibrated)")
    print("Starting evaluation...", flush=True)

    logits_parts = []
    label_parts = []
    seen = 0
    start = perf_counter()

    with torch.inference_mode():
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            logits = model(images)

            if logits.shape != (labels.size(0), dataset.num_classes):
                raise RuntimeError("Unexpected logits shape.")

            if not torch.isfinite(logits).all().item():
                raise RuntimeError(f"Non-finite logits at batch {step}.")

            logits_parts.append(logits.cpu())
            label_parts.append(labels.clone())
            seen += labels.size(0)

            if step % 25 == 0 or step == len(loader):
                print(
                    f"Evaluated: {seen}/{len(dataset)}",
                    flush=True,
                )

    torch.cuda.synchronize(device)
    elapsed = perf_counter() - start

    if seen != len(dataset):
        raise RuntimeError("Evaluation sample count mismatch.")

    scores = torch.cat(logits_parts, dim=0)
    targets = torch.cat(label_parts, dim=0)
    predictions = scores.argmax(dim=1)
    num_classes = dataset.num_classes

    # Confirm that outputs follow the original manifest order.
    expected_labels = torch.tensor(
        [
            dataset.class_to_idx[row["family"]]
            for row in dataset.samples
        ],
        dtype=torch.int64,
    )

    if not torch.equal(targets, expected_labels):
        raise RuntimeError("Prediction order differs from manifest.")

    confusion = torch.bincount(
        targets * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    if confusion.sum().item() != seen:
        raise RuntimeError("Confusion-matrix count mismatch.")

    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    true_support = matrix.sum(dim=1)
    predicted_support = matrix.sum(dim=0)
    supported = true_support > 0

    f1_denominator = true_support + predicted_support
    f1 = 2 * true_positive / f1_denominator.clamp_min(1)
    recall = true_positive / true_support.clamp_min(1)

    accuracy = true_positive.sum().item() / seen
    macro_f1_supported = f1[supported].mean().item()
    macro_f1_fixed = f1.mean().item()
    balanced_accuracy = recall[supported].mean().item()

    # Cross-entropy on raw logits is uncalibrated negative log likelihood.
    nll = F.cross_entropy(scores.to(torch.float64), targets).item()

    idx_to_class = {
        index: family
        for family, index in dataset.class_to_idx.items()
    }
    class_names = [
        idx_to_class[index]
        for index in range(num_classes)
    ]
    supported_families = [
        class_names[index]
        for index in range(num_classes)
        if supported[index].item()
    ]
    absent_families = [
        class_names[index]
        for index in range(num_classes)
        if not supported[index].item()
    ]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        checkpoint_path.parent
        / f"validation_{args.month}_{run_id}"
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

    per_family_rows = []

    for index, family in enumerate(class_names):
        support = int(true_support[index].item())
        predicted_count = int(predicted_support[index].item())

        # Blank recall means undefined because no true examples exist.
        family_recall = recall[index].item() if support > 0 else ""

        per_family_rows.append([
            index,
            family,
            support,
            predicted_count,
            family_recall,
            f1[index].item(),
        ])

    write_csv(
        output_dir / "per_family.csv",
        [
            "class_index",
            "family",
            "true_support",
            "predicted_count",
            "recall",
            "f1_zero_if_undefined",
        ],
        per_family_rows,
    )

    # Preserve sample order and raw logits for reproducible later analysis.
    np.savez_compressed(
        output_dir / "predictions.npz",
        sample_identifiers=np.array([
            row["sha"] for row in dataset.samples
        ]),
        logits=scores.numpy(),
        labels=targets.numpy(),
        predictions=predictions.numpy(),
        class_names=np.array(class_names),
    )

    report = {
        "purpose": "historical validation of fixed SBSMI baseline",
        "split": "validation",
        "month": args.month,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": epoch,
        "training_seed": config["seed"],
        "dataset_config_sha256": dataset.config_id,
        "class_to_idx": dataset.class_to_idx,
        "samples": seen,
        "supported_class_count": len(supported_families),
        "supported_families": supported_families,
        "absent_families": absent_families,
        "accuracy": accuracy,
        "macro_f1_supported_classes": macro_f1_supported,
        "macro_f1_fixed_51_zero_if_undefined": macro_f1_fixed,
        "balanced_accuracy_supported_classes": balanced_accuracy,
        "uncalibrated_nll": nll,
        "temperature": 1.0,
        "batch_size": 64,
        "model_mode": "eval",
        "weight_updates": False,
        "elapsed_inference_pipeline_seconds": elapsed,
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "evaluator_source_sha256": file_sha256(Path(__file__).resolve()),
        "grouping_status": (
            "Exact disarmed-content duplicates audited; "
            "near-duplicate policy pending."
        ),
        "note": (
            "Development validation for one training seed. "
            "Not future-test performance."
        ),
    }

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("Month:", args.month)
    print("Evaluated samples:", seen)
    print("Supported classes:", len(supported_families))
    print("Absent families:", absent_families)
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Macro F1 (supported classes): {macro_f1_supported:.4f}")
    print(f"Macro F1 (fixed 51 classes): {macro_f1_fixed:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.2%}")
    print(f"Uncalibrated NLL: {nll:.4f}")
    print(f"Evaluation seconds: {elapsed:.1f}")
    print("Report:", report_path)
    print("Outputs:", output_dir)
    print("No weight updates. No future-test data used.")


if __name__ == "__main__":
    main()