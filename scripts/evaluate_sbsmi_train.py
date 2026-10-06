import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.malsbslcnet import MalSBSLCNet
from scripts.sbsmi_dataset import SBSMIDataset


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SBSMI checkpoint on training data."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    training_config = checkpoint["config"]

    dataset = SBSMIDataset(project_root, split="train")

    if len(dataset) != training_config["train_samples"]:
        raise ValueError("Training sample count differs from checkpoint.")

    if dataset.config_id != training_config["dataset_config_sha256"]:
        raise ValueError("Dataset configuration differs from checkpoint.")

    if dataset.class_to_idx != training_config["class_to_idx"]:
        raise ValueError("Class mapping differs from checkpoint.")

    if dataset.num_classes != training_config["num_classes"]:
        raise ValueError("Number of classes differs from checkpoint.")

    # Ensure the model and loader still match the training implementation.
    for relative_path, expected_hash in training_config["source_sha256"].items():
        source_path = project_root / relative_path

        if source_path.name in {"malsbslcnet.py", "sbsmi_dataset.py"}:
            if file_sha256(source_path) != expected_hash:
                raise ValueError(f"Source changed: {relative_path}")

    device = torch.device("cuda")

    model = MalSBSLCNet(num_classes=dataset.num_classes)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    # No optimizer is created; evaluation does not update weights.
    del checkpoint

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss(reduction="sum")
    num_classes = dataset.num_classes
    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = checkpoint_path.parent / f"train_evaluation_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    print("Purpose: evaluate a fixed checkpoint on train")
    print("Checkpoint:", checkpoint_path)
    print("Samples:", len(dataset))
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model mode: eval")
    print("Starting evaluation...", flush=True)

    total_loss = 0.0
    seen = 0
    start = perf_counter()

    with torch.inference_mode():
        for step, (images, labels_cpu) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels_cpu.to(device, non_blocking=True)

            logits = model(images)

            if not torch.isfinite(logits).all().item():
                raise RuntimeError(f"Non-finite logits at batch {step}.")

            loss = criterion(logits, labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss at batch {step}.")

            predictions = logits.argmax(dim=1).cpu()

            # Rows are true labels; columns are predicted labels.
            pairs = labels_cpu * num_classes + predictions
            confusion += torch.bincount(
                pairs,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)

            total_loss += loss.item()
            seen += labels_cpu.size(0)

            if step % 100 == 0 or step == len(loader):
                print(
                    f"Evaluated: {seen}/{len(dataset)}",
                    flush=True,
                )

    torch.cuda.synchronize(device)
    elapsed = perf_counter() - start

    if seen != len(dataset) or confusion.sum().item() != seen:
        raise RuntimeError("Evaluation sample count mismatch.")

    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    true_support = matrix.sum(dim=1)
    predicted_support = matrix.sum(dim=0)

    supported = true_support > 0
    denominator = true_support + predicted_support

    f1 = torch.where(
        denominator > 0,
        2 * true_positive / denominator.clamp_min(1),
        torch.zeros_like(denominator),
    )
    recall = true_positive / true_support.clamp_min(1)

    accuracy = true_positive.sum().item() / seen
    mean_loss = total_loss / seen
    macro_f1 = f1[supported].mean().item()
    balanced_accuracy = recall[supported].mean().item()

    idx_to_class = {
        index: family
        for family, index in dataset.class_to_idx.items()
    }

    confusion_path = output_dir / "confusion_matrix.csv"

    with confusion_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["true_family"]
            + [idx_to_class[index] for index in range(num_classes)]
        )

        for index, row in enumerate(confusion.tolist()):
            writer.writerow([idx_to_class[index], *row])

    report = {
        "purpose": "fixed-checkpoint evaluation on training data",
        "split": "train",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "dataset_config_sha256": dataset.config_id,
        "class_to_idx": dataset.class_to_idx,
        "samples": seen,
        "supported_classes": int(supported.sum().item()),
        "model_mode": "eval",
        "batch_size": 64,
        "cross_entropy": mean_loss,
        "accuracy": accuracy,
        "macro_f1_supported_classes": macro_f1,
        "balanced_accuracy_supported_classes": balanced_accuracy,
        "elapsed_seconds": elapsed,
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "evaluator_source_sha256": file_sha256(Path(__file__).resolve()),
        "note": (
            "The model was fitted on these training samples. "
            "These metrics do not measure generalization."
        ),
    }

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("Evaluated samples:", seen)
    print("Supported classes:", report["supported_classes"])
    print(f"Train cross-entropy: {mean_loss:.4f}")
    print(f"Train accuracy: {accuracy:.2%}")
    print(f"Train macro F1: {macro_f1:.4f}")
    print(f"Train balanced accuracy: {balanced_accuracy:.2%}")
    print(f"Evaluation seconds: {elapsed:.1f}")
    print("Report:", report_path)
    print("Confusion matrix:", confusion_path)
    print("No weight updates. No validation or future-test data used.")


if __name__ == "__main__":
    main()