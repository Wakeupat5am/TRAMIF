# cho học trên tập nhỏ trong train 
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from models.malsbslcnet import MalSBSLCNet
from scripts.sbsmi_pilot_dataset import SBSMIPilotDataset


def evaluate(model, images, labels, criterion):
    model.eval()

    with torch.no_grad():
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()

    if not torch.isfinite(loss).item():
        raise RuntimeError("Evaluation loss is not finite.")

    return loss.item(), accuracy.item(), predictions.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    project_root = Path(__file__).resolve().parents[1]
    mapping_path = project_root / "docs" / "class_mapping.json"

    seed = 42
    max_steps = 300
    evaluation_interval = 25
    learning_rate = 1e-4
    weight_decay = 1e-6

    torch.manual_seed(seed)

    dataset = SBSMIPilotDataset(args.run_dir, mapping_path)

    if len(dataset) != 10:
        raise ValueError("Expected exactly 10 pilot samples.")

    # The complete tiny dataset fits into one batch.
    items = [dataset[index] for index in range(len(dataset))]
    images = torch.stack([image for image, _ in items])
    labels = torch.tensor(
        [label for _, label in items],
        dtype=torch.long,
    )

    device = torch.device("cuda")
    images = images.to(device)
    labels = labels.to(device)

    model = MalSBSLCNet(num_classes=dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        project_root / "data" / "derived" / "tiny_training" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    print("Purpose: memorize 10 training samples as a pipeline check")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Batch shape:", tuple(images.shape))
    print("Maximum optimizer steps:", max_steps)
    print("Output folder:", output_dir)

    history = []
    start = perf_counter()

    loss_value, accuracy, predictions = evaluate(
        model, images, labels, criterion
    )
    history.append({
        "step": 0,
        "eval_loss_on_training_samples": loss_value,
        "accuracy_on_training_samples": accuracy,
    })
    print(
        f"Step 000 | eval loss={loss_value:.4f} | "
        f"train accuracy={accuracy:.1%}",
        flush=True,
    )

    passed = False
    completed_steps = 0

    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite training loss at step {step}.")

        loss.backward()

        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"Missing gradient: {name}")

            if not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError(f"Non-finite gradient: {name}")

        optimizer.step()
        completed_steps = step

        if step % evaluation_interval == 0 or step == max_steps:
            loss_value, accuracy, predictions = evaluate(
                model, images, labels, criterion
            )
            history.append({
                "step": step,
                "eval_loss_on_training_samples": loss_value,
                "accuracy_on_training_samples": accuracy,
            })

            print(
                f"Step {step:03d} | eval loss={loss_value:.4f} | "
                f"train accuracy={accuracy:.1%}",
                flush=True,
            )

            # Engineering criterion only, not a research performance target.
            passed = accuracy == 1.0 and loss_value < 0.1

            if passed:
                break

    elapsed = perf_counter() - start

    with (output_dir / "history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    true_labels = labels.cpu().tolist()
    mapping_hash = hashlib.sha256(mapping_path.read_bytes()).hexdigest()

    report = {
        "purpose": "tiny training memorization check",
        "source_pilot": str(args.run_dir.resolve()),
        "sample_identifiers": [row["sha"] for row in dataset.samples],
        "sample_count": len(dataset),
        "class_mapping_sha256": mapping_hash,
        "num_classes": dataset.num_classes,
        "seed": seed,
        "python_torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": len(dataset),
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "elapsed_seconds": elapsed,
        "final_eval_loss_on_training_samples": loss_value,
        "final_accuracy_on_training_samples": accuracy,
        "true_labels": true_labels,
        "predicted_labels": predictions,
        "criterion": "train accuracy 100% and eval loss below 0.1",
        "status": "PASS" if passed else "NOT_REACHED",
        "checkpoint_saved": False,
        "note": (
            "Evaluation uses the same 10 training samples in model.eval() "
            "mode. This does not measure generalization."
        ),
    }

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("True labels:", true_labels)
    print("Predicted labels:", predictions)
    print("Tiny training check:", report["status"])
    print("Report:", report_path)
    print("No validation/future-test data used. No checkpoint saved.")


if __name__ == "__main__":
    main()