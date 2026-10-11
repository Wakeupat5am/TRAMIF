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

from models.malsbslcnet import MalSBSLCNet
from scripts.historical_policy_dataset import HistoricalPolicyDataset


ROOT = Path(__file__).resolve().parents[1]

# Fixed development checkpoint, recorded by the earlier SBSMI evaluation.
EXPECTED_CHECKPOINT_SHA256 = (
    "0c7d76ec3e077e5f88dd6d455b14a041ee2d701a2dc170d5718928695bcf2c4d"
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the existing SBSMI baseline on filtered validation."
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

    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("This is not the previously recorded SBSMI checkpoint.")

    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location="cpu",
        weights_only=True,
    )
    del checkpoint_bytes

    config = checkpoint["config"]
    if checkpoint["epoch"] != 100 or config["epochs"] != 100:
        raise ValueError("Expected the fixed epoch-100 checkpoint.")
    if config.get("model") != "MalSBSLCNet":
        raise ValueError("Expected MalSBSLCNet.")
    if config.get("validation_used") is not False:
        raise ValueError("Training must exclude validation.")
    if config.get("future_test_used") is not False:
        raise ValueError("Training must exclude future test.")

    dataset = HistoricalPolicyDataset(
        project_root=ROOT,
        policy_dir=args.policy_dir,
        view="sbsmi",
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

    # The old checkpoint predates the policy. Reuse is valid here because
    # the policy retains exactly the same original training rows.
    train_dataset = HistoricalPolicyDataset(
        project_root=ROOT,
        policy_dir=args.policy_dir,
        view="sbsmi",
        split="train",
        threshold=0,
    )
    if train_dataset.samples != train_dataset.base.samples:
        raise ValueError("Policy changed the original training selection.")
    if len(train_dataset) != config["train_samples"]:
        raise ValueError("Training sample count differs from checkpoint.")
    train_selection = dict(train_dataset.selection)
    del train_dataset

    # Older training configs used Windows separators in source paths.
    source_hashes = {
        path.replace("\\", "/"): value
        for path, value in config["source_sha256"].items()
    }
    for relative_path in (
        "models/malsbslcnet.py",
        "scripts/sbsmi_dataset.py",
    ):
        actual = sha256_bytes((ROOT / relative_path).read_bytes())
        if actual != source_hashes[relative_path]:
            raise ValueError(f"Source differs from training: {relative_path}")

    model = MalSBSLCNet(num_classes=dataset.num_classes)
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
    print("View: sbsmi")
    print("Month:", args.month)
    print("Policy threshold: 0")
    print("Checkpoint epoch: 100")
    print("Original training selection unchanged: PASS")
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
        "purpose": "filtered historical validation of fixed SBSMI baseline",
        "view": "sbsmi",
        "split": "validation",
        "month": args.month,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": 100,
        "training_seed": config["seed"],
        "dataset_config_sha256": dataset.preprocessing_config_id,
        "selection": dataset.selection,
        "selection_sha256": dataset.selection_id,
        "verified_train_selection": train_selection,
        "original_train_selection_unchanged": True,
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
        "policy_loader_source_sha256": sha256_bytes(
            (ROOT / "scripts/historical_policy_dataset.py").read_bytes()
        ),
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "note": (
            "Existing development checkpoint reused after verifying unchanged "
            "training rows. Policy was defined after initial unfiltered "
            "SBSMI validation. Old temperature is not applied. "
            "Timing includes data loading."
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("View: sbsmi")
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