import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.calibrate_sbsmi import (
    ECE_BINS,
    T_MAX,
    T_MIN,
    fit_temperature,
    probability_metrics,
    self_check,
)
from scripts.historical_policy_dataset import HistoricalPolicyDataset


ROOT = Path(__file__).resolve().parents[1]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_filtered_predictions(folder, policy_dir, view, month):
    folder = Path(folder).resolve()
    report_bytes = (folder / "report.json").read_bytes()
    report = json.loads(report_bytes)

    dataset = HistoricalPolicyDataset(
        project_root=ROOT,
        policy_dir=policy_dir,
        view=view,
        split="validation",
        month=month,
        threshold=0,
    )
    expected_counts = {"2020-02": 3799, "2020-03": 4332}
    if len(dataset) != expected_counts[month]:
        raise ValueError("Unexpected filtered sample count.")

    expected_report = {
        "view": view,
        "split": "validation",
        "month": month,
        "checkpoint_epoch": 100,
        "samples": len(dataset),
        "temperature": 1.0,
        "model_mode": "eval",
        "weight_updates": False,
        "future_test_used": False,
        "dataset_config_sha256": dataset.preprocessing_config_id,
        "class_to_idx": dataset.class_to_idx,
        "selection": dataset.selection,
        "selection_sha256": dataset.selection_id,
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            raise ValueError(f"Report mismatch: {view}, {month}, {key}")

    prediction_bytes = (folder / "predictions.npz").read_bytes()
    prediction_hash = sha256_bytes(prediction_bytes)
    if prediction_hash != report["predictions_sha256"]:
        raise ValueError("Prediction file checksum mismatch.")

    # Hash and decode the same bytes.
    with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as data:
        logits = data["logits"].copy()
        labels = data["labels"].copy()
        predictions = data["predictions"].copy()
        identifiers = data["sample_identifiers"].tolist()
        class_names = data["class_names"].tolist()

    expected_names = [
        family
        for family, index in sorted(
            dataset.class_to_idx.items(), key=lambda item: item[1]
        )
    ]
    expected_ids = [row["sha"] for row in dataset.samples]
    expected_labels = [
        dataset.class_to_idx[row["family"]] for row in dataset.samples
    ]

    if identifiers != expected_ids:
        raise ValueError("Prediction identifiers or order differ from policy.")
    if class_names != expected_names:
        raise ValueError("Prediction class order differs from class mapping.")
    if logits.shape != (len(dataset), dataset.num_classes):
        raise ValueError("Invalid logits shape.")
    if not np.issubdtype(logits.dtype, np.floating):
        raise ValueError("Logits must be floating-point values.")
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite logits.")
    if labels.shape != (len(dataset),):
        raise ValueError("Invalid label shape.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be integers.")
    if labels.tolist() != expected_labels:
        raise ValueError("Labels differ from policy manifest.")
    if predictions.shape != labels.shape:
        raise ValueError("Invalid prediction shape.")
    if not np.array_equal(predictions, logits.argmax(axis=1)):
        raise ValueError("Saved predictions differ from logits.")

    scores = torch.tensor(logits, dtype=torch.float64)
    targets = torch.tensor(labels, dtype=torch.int64)
    reproduced_nll = F.cross_entropy(scores, targets).item()
    if not np.isclose(
        reproduced_nll,
        report["uncalibrated_nll"],
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError("Saved NLL differs from prediction logits.")

    source = {
        "folder": str(folder),
        "report_sha256": sha256_bytes(report_bytes),
        "predictions_sha256": prediction_hash,
        "selection_sha256": dataset.selection_id,
    }
    return scores, targets, report, source


def main():
    parser = argparse.ArgumentParser(
        description="Fit temperature on filtered February; evaluate March."
    )
    parser.add_argument(
        "--view", choices=("sbsmi", "raw-byte", "entropy"), required=True
    )
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--february-dir", type=Path, required=True)
    parser.add_argument("--march-dir", type=Path, required=True)
    args = parser.parse_args()

    # Saved logits are small enough for CPU calibration.
    torch.set_num_threads(1)
    self_check()

    feb_logits, feb_labels, feb_report, feb_source = (
        load_filtered_predictions(
            args.february_dir, args.policy_dir, args.view, "2020-02"
        )
    )

    print("View:", args.view)
    print("Policy threshold: 0")
    print("Calibration month: 2020-02")
    print("Calibration samples:", len(feb_labels))

    # Fit receives February only.
    temperature, at_boundary = fit_temperature(feb_logits, feb_labels)
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Invalid fitted temperature.")

    feb_before = probability_metrics(feb_logits, feb_labels, 1.0)
    feb_after = probability_metrics(feb_logits, feb_labels, temperature)
    if feb_after["nll"] > feb_before["nll"] + 1e-10:
        raise RuntimeError("Calibration increased February NLL.")

    print(f"Fitted temperature: {temperature:.6f}")
    print("Optimum at search boundary:", at_boundary)

    # March is loaded after fitting T and never passed to the optimizer.
    mar_logits, mar_labels, mar_report, mar_source = (
        load_filtered_predictions(
            args.march_dir, args.policy_dir, args.view, "2020-03"
        )
    )

    for key in (
        "view",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "training_seed",
        "dataset_config_sha256",
        "class_to_idx",
    ):
        if feb_report[key] != mar_report[key]:
            raise ValueError(f"February/March mismatch: {key}")

    for key in (
        "threshold",
        "policy_config_sha256",
        "policy_report_sha256",
        "retained_manifest_sha256",
    ):
        if feb_report["selection"][key] != mar_report["selection"][key]:
            raise ValueError(f"February/March policy mismatch: {key}")

    mar_before = probability_metrics(mar_logits, mar_labels, 1.0)
    mar_after = probability_metrics(mar_logits, mar_labels, temperature)

    for name, logits in (("February", feb_logits), ("March", mar_logits)):
        if not torch.equal(
            logits.argmax(dim=1),
            (logits / temperature).argmax(dim=1),
        ):
            raise RuntimeError(f"Predictions changed: {name}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        args.february_dir.resolve().parent
        / f"temperature_filtered_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    source_hashes = {
        path: sha256_bytes((ROOT / path).read_bytes())
        for path in (
            "scripts/calibrate_filtered_view.py",
            "scripts/calibrate_sbsmi.py",
            "scripts/historical_policy_dataset.py",
        )
    }

    temperature_record = {
        "schema_version": 1,
        "view": args.view,
        "method": "scalar temperature scaling",
        "temperature": temperature,
        "temperature_bounds": [T_MIN, T_MAX],
        "optimum_at_boundary": at_boundary,
        "optimizer": "bisection on inverse-temperature derivative",
        "objective": "unweighted mean filtered February NLL",
        "fit_month": "2020-02",
        "fit_samples": len(feb_labels),
        "checkpoint_sha256": feb_report["checkpoint_sha256"],
        "checkpoint_epoch": feb_report["checkpoint_epoch"],
        "training_seed": feb_report["training_seed"],
        "dataset_config_sha256": feb_report["dataset_config_sha256"],
        "class_to_idx": feb_report["class_to_idx"],
        "fit_selection": feb_report["selection"],
        "fit_selection_sha256": feb_report["selection_sha256"],
        "calibration_source": feb_source,
        "source_code_sha256": source_hashes,
        "future_test_used": False,
    }
    save_json(output_dir / "temperature.json", temperature_record)

    result = {
        "view": args.view,
        "temperature": temperature,
        "optimum_at_boundary": at_boundary,
        "policy_threshold": 0,
        "checkpoint_sha256": feb_report["checkpoint_sha256"],
        "ece_definition": (
            f"{ECE_BINS} equal-width top-label confidence bins"
        ),
        "brier_definition": "mean over samples of sum over classes",
        "february": {
            "samples": len(feb_labels),
            "role": "fit data; metrics are in-sample for temperature",
            "before": feb_before,
            "after": feb_after,
        },
        "march": {
            "samples": len(mar_labels),
            "role": "forward validation; not used to fit temperature",
            "before": mar_before,
            "after": mar_after,
        },
        "sources": {"february": feb_source, "march": mar_source},
        "source_code_sha256": source_hashes,
        "predictions_unchanged": True,
        "network_weight_updates": False,
        "future_test_used": False,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "note": (
            "Development calibration for one checkpoint. "
            "February fit metrics are not independent reliability estimates. "
            "Historical policy was defined after initial SBSMI validation."
        ),
    }
    save_json(output_dir / "report.json", result)

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