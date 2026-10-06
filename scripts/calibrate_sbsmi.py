import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.sbsmi_dataset import SBSMIDataset


T_MIN = 0.05
T_MAX = 20.0
ECE_BINS = 15


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_predictions(folder, project_root, month):
    folder = folder.resolve()
    report_path = folder / "report.json"
    prediction_path = folder / "predictions.npz"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset = SBSMIDataset(
        project_root,
        split="validation",
        month=month,
    )

    expected_report = {
        "split": "validation",
        "month": month,
        "checkpoint_epoch": 100,
        "samples": len(dataset),
        "temperature": 1.0,
        "model_mode": "eval",
        "weight_updates": False,
        "dataset_config_sha256": dataset.config_id,
        "class_to_idx": dataset.class_to_idx,
    }

    for key, expected in expected_report.items():
        if report.get(key) != expected:
            raise ValueError(f"Report mismatch: {month}, {key}")

    with np.load(prediction_path, allow_pickle=False) as data:
        logits = data["logits"].copy()
        labels = data["labels"].copy()
        predictions = data["predictions"].copy()
        identifiers = data["sample_identifiers"].tolist()
        class_names = data["class_names"].tolist()

    expected_names = [
        family
        for family, index in sorted(
            dataset.class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]
    expected_identifiers = [row["sha"] for row in dataset.samples]
    expected_labels = [
        dataset.class_to_idx[row["family"]]
        for row in dataset.samples
    ]

    if class_names != expected_names:
        raise ValueError("Class order mismatch.")

    if identifiers != expected_identifiers:
        raise ValueError("Sample order or identity mismatch.")

    if logits.shape != (len(dataset), dataset.num_classes):
        raise ValueError("Invalid logits shape.")

    if not np.issubdtype(logits.dtype, np.floating):
        raise ValueError("Logits must be floating-point values.")

    if not np.isfinite(logits).all():
        raise ValueError("Non-finite logits.")

    if labels.shape != (len(dataset),):
        raise ValueError("Invalid labels shape.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be integers.")

    if labels.tolist() != expected_labels:
        raise ValueError("Labels differ from the frozen manifest.")

    if not np.array_equal(predictions, logits.argmax(axis=1)):
        raise ValueError("Saved predictions differ from logits.")

    scores = torch.tensor(logits, dtype=torch.float64)
    targets = torch.tensor(labels, dtype=torch.int64)

    reproduced_nll = F.cross_entropy(scores, targets).item()

    if abs(reproduced_nll - report["uncalibrated_nll"]) > 1e-8:
        raise ValueError("Saved NLL does not match the logits.")

    provenance = {
        "folder": str(folder),
        "report_sha256": file_sha256(report_path),
        "predictions_sha256": file_sha256(prediction_path),
    }

    return scores, targets, report, provenance


def fit_temperature(logits, labels):
    """Minimize mean NLL using only the supplied calibration samples.

    NLL is convex in inverse temperature beta = 1 / T.
    Its derivative can therefore be solved by bisection.
    """
    centered = logits - logits.max(dim=1, keepdim=True).values
    true_logits = centered.gather(1, labels[:, None]).squeeze(1)

    def gradient(beta):
        probabilities = torch.softmax(centered * beta, dim=1)
        expected_logits = (probabilities * centered).sum(dim=1)
        return (expected_logits - true_logits).mean().item()

    lower_beta = 1.0 / T_MAX
    upper_beta = 1.0 / T_MIN

    if abs(gradient(1.0)) < 1e-12:
        return 1.0, False

    if gradient(lower_beta) >= 0:
        return T_MAX, True

    if gradient(upper_beta) <= 0:
        return T_MIN, True

    for _ in range(60):
        middle = (lower_beta + upper_beta) / 2.0

        if gradient(middle) > 0:
            upper_beta = middle
        else:
            lower_beta = middle

    beta = (lower_beta + upper_beta) / 2.0
    return 1.0 / beta, False


def probability_metrics(logits, labels, temperature):
    scaled = logits / temperature
    probabilities = torch.softmax(scaled, dim=1)
    predictions = scaled.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    correct = (predictions == labels).to(torch.float64)

    one_hot = F.one_hot(
        labels,
        num_classes=logits.shape[1],
    ).to(torch.float64)

    # Multiclass Brier: sum over classes, then mean over samples.
    brier = (
        (probabilities - one_hot).square().sum(dim=1).mean().item()
    )

    # Bins: [0, 1/15), ..., [14/15, 1].
    bin_ids = torch.floor(confidence * ECE_BINS).long()
    bin_ids = bin_ids.clamp(max=ECE_BINS - 1)

    ece = 0.0
    bin_rows = []

    for index in range(ECE_BINS):
        mask = bin_ids == index
        count = int(mask.sum().item())

        if count:
            mean_confidence = confidence[mask].mean().item()
            bin_accuracy = correct[mask].mean().item()
            ece += (
                count / len(labels)
                * abs(mean_confidence - bin_accuracy)
            )
        else:
            mean_confidence = None
            bin_accuracy = None

        bin_rows.append({
            "lower": index / ECE_BINS,
            "upper": (index + 1) / ECE_BINS,
            "count": count,
            "mean_confidence": mean_confidence,
            "accuracy": bin_accuracy,
        })

    return {
        "nll": F.cross_entropy(scaled, labels).item(),
        "brier": brier,
        "ece_15": ece,
        "accuracy": correct.mean().item(),
        "mean_confidence": confidence.mean().item(),
        "bins": bin_rows,
    }


def self_check():
    # Four identical predictions, three correct.
    # logits = [log(9), 0] gives confidence 0.9 at T=1.
    # Optimal confidence is 3/4, obtained at T=2.
    logits = torch.tensor(
        [[math.log(9.0), 0.0]] * 4,
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 0, 0, 1], dtype=torch.int64)

    temperature, at_boundary = fit_temperature(logits, labels)
    metrics = probability_metrics(logits, labels, temperature)

    if at_boundary or abs(temperature - 2.0) > 1e-8:
        raise RuntimeError("Known-temperature check failed.")

    if abs(metrics["brier"] - 0.375) > 1e-8:
        raise RuntimeError("Brier calculation check failed.")

    if metrics["ece_15"] > 1e-8:
        raise RuntimeError("ECE calculation check failed.")

    print("Known-answer calibration check: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--february-dir", type=Path, required=True)
    parser.add_argument("--march-dir", type=Path, required=True)
    args = parser.parse_args()

    # Small saved-logit matrices do not require GPU execution.
    torch.set_num_threads(1)
    self_check()

    project_root = Path(__file__).resolve().parents[1]

    feb_logits, feb_labels, feb_report, feb_source = load_predictions(
        args.february_dir,
        project_root,
        "2020-02",
    )

    print("Calibration month: 2020-02")
    print("Calibration samples:", len(feb_labels))

    # Temperature fitting receives February data only.
    temperature, at_boundary = fit_temperature(feb_logits, feb_labels)

    feb_before = probability_metrics(feb_logits, feb_labels, 1.0)
    feb_after = probability_metrics(feb_logits, feb_labels, temperature)

    if feb_after["nll"] > feb_before["nll"] + 1e-10:
        raise RuntimeError("Fitted temperature increased calibration NLL.")

    print(f"Fitted temperature: {temperature:.6f}")
    print("Optimum at search boundary:", at_boundary)

    # March is loaded only after T has been determined.
    mar_logits, mar_labels, mar_report, mar_source = load_predictions(
        args.march_dir,
        project_root,
        "2020-03",
    )

    for key in (
        "checkpoint_sha256",
        "checkpoint_epoch",
        "training_seed",
        "dataset_config_sha256",
        "class_to_idx",
    ):
        if feb_report[key] != mar_report[key]:
            raise ValueError(f"February/March mismatch: {key}")

    mar_before = probability_metrics(mar_logits, mar_labels, 1.0)
    mar_after = probability_metrics(mar_logits, mar_labels, temperature)

    for name, logits in (
        ("February", feb_logits),
        ("March", mar_logits),
    ):
        unchanged = torch.equal(
            logits.argmax(dim=1),
            (logits / temperature).argmax(dim=1),
        )
        if not unchanged:
            raise RuntimeError(f"Predictions changed: {name}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        args.february_dir.resolve().parent
        / f"temperature_scaling_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    temperature_record = {
        "schema_version": 1,
        "method": "scalar temperature scaling",
        "temperature": temperature,
        "temperature_bounds": [T_MIN, T_MAX],
        "optimum_at_boundary": at_boundary,
        "optimizer": "60 bisection steps on inverse-temperature derivative",
        "objective": "unweighted mean February NLL",
        "fit_month": "2020-02",
        "fit_samples": len(feb_labels),
        "checkpoint_sha256": feb_report["checkpoint_sha256"],
        "dataset_config_sha256": feb_report["dataset_config_sha256"],
        "class_to_idx": feb_report["class_to_idx"],
        "calibration_source": feb_source,
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
    }
    save_json(output_dir / "temperature.json", temperature_record)

    results = {
        "temperature": temperature,
        "ece_definition": "15 equal-width top-label confidence bins",
        "brier_definition": "mean over samples of sum over classes",
        "february": {
            "role": "calibration fit data; metrics are in-sample for T",
            "before": feb_before,
            "after": feb_after,
        },
        "march": {
            "role": "forward validation; not used to fit T",
            "before": mar_before,
            "after": mar_after,
        },
        "sources": {
            "february": feb_source,
            "march": mar_source,
        },
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "predictions_unchanged": True,
        "future_test_used": False,
        "note": (
            "Development calibration for one SBSMI checkpoint. "
            "Near-duplicate policy remains pending."
        ),
    }
    save_json(output_dir / "report.json", results)

    for name, before, after in (
        ("February (fit data)", feb_before, feb_after),
        ("March (forward validation)", mar_before, mar_after),
    ):
        print()
        print(name)
        print(f"NLL: {before['nll']:.4f} -> {after['nll']:.4f}")
        print(f"Brier: {before['brier']:.4f} -> {after['brier']:.4f}")
        print(f"ECE-15: {before['ece_15']:.4f} -> {after['ece_15']:.4f}")
        print(
            f"Accuracy: {before['accuracy']:.2%}"
            f" -> {after['accuracy']:.2%}"
        )

    print()
    print("Predictions unchanged: PASS")
    print("Temperature:", output_dir / "temperature.json")
    print("Report:", output_dir / "report.json")
    print("No network weight updates. No future-test data used.")


if __name__ == "__main__":
    main()