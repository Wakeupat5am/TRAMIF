"""CPU-only fixed March fusion evaluation from the pinned historical register."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.calibrate_filtered_view import load_filtered_predictions
from scripts.uniform_fusion import load_bundle, fuse_uniform, classification_metrics

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "data/derived/historical_near_duplicates_v1/20261009_143301_921802"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    torch.set_num_threads(1)
    register_path = ROOT / "docs/baseline_artifacts.json"
    register = json.loads(register_path.read_bytes())
    entries = register["entries"]
    require([entry["view"] for entry in entries] == ["sbsmi", "raw-byte", "entropy"], "Unexpected views/order")
    bundles, provenance = [], []
    common_policy = None
    for entry in entries:
        view = entry["view"]
        temperature_path = ROOT / entry["temperature_file"]
        require(sha(temperature_path) == entry["temperature_file_sha256"], "Temperature checksum")
        temperature = json.loads(temperature_path.read_bytes())
        require(temperature["view"] == view and temperature["fit_month"] == "2020-02"
                and temperature["fit_samples"] == 3799 and temperature["future_test_used"] is False,
                "Temperature scope")
        require(sha(ROOT / entry["checkpoint"]) == entry["checkpoint_sha256"]
                == temperature["checkpoint_sha256"], "Checkpoint binding")
        for path, expected in temperature["source_code_sha256"].items():
            require(sha(ROOT / path) == expected, "Calibration source changed: " + path)
        month_reports = {}
        for month in ("2020-02", "2020-03"):
            selected = [item for item in entry["evaluations"] if item["month"] == month]
            require(len(selected) == 1, "Ambiguous month")
            saved = selected[0]
            report_path = ROOT / saved["report"]
            require(sha(report_path) == saved["report_sha256"], "Report checksum")
            _, labels, report, source = load_filtered_predictions(report_path.parent, POLICY, view, month)
            require(report["predictions_sha256"] == saved["predictions_sha256"], "Prediction registry binding")
            for key in ("checkpoint_sha256", "checkpoint_epoch", "training_seed", "dataset_config_sha256", "class_to_idx"):
                require(report[key] == temperature[key], "Temperature/evaluation mismatch: " + key)
            policy = {key: report["selection"][key] for key in
                      ("threshold", "policy_config_sha256", "policy_report_sha256", "retained_manifest_sha256")}
            if common_policy is None:
                common_policy = policy
            require(policy == common_policy, "Views/months use different policies")
            month_reports["february" if month == "2020-02" else "march"] = source
            if month == "2020-02":
                require(source == temperature["calibration_source"], "February calibration source mismatch")
                require(report["selection"] == temperature["fit_selection"]
                        and report["selection_sha256"] == temperature["fit_selection_sha256"], "Calibration selection")
            else:
                batch = load_bundle(report_path.parent / "predictions.npz", source["predictions_sha256"], temperature["temperature"])
                require(np.array_equal(batch["labels"], labels.numpy()), "Decoded labels mismatch")
                bundles.append(batch)
        calibration_report_path = temperature_path.parent / "report.json"
        calibration_report = json.loads(calibration_report_path.read_bytes())
        require(calibration_report["sources"] == month_reports, "Calibration report sources differ")
        require(calibration_report["temperature"] == temperature["temperature"], "Calibration temperature differs")
        provenance.append({"view": view, "registry_entry": entry,
                           "calibration_report_sha256": sha(calibration_report_path),
                           "previous_march_metrics": calibration_report["march"]["after"]})
        print(view + ": canonical manifest, checkpoint, temperature and saved predictions verified")

    single_logp, fused_logp = fuse_uniform(bundles)
    labels = bundles[0]["labels"]
    names = [entry["view"] for entry in entries] + ["uniform-calibrated"]
    all_logp = single_logp + [fused_logp]
    results = {}
    for name, logp in zip(names, all_logp):
        results[name] = classification_metrics(logp, labels)
    for source in provenance:
        actual = results[source["view"]]
        for key in ("accuracy", "nll", "brier", "ece_15"):
            require(np.isclose(actual[key], source["previous_march_metrics"][key], rtol=0, atol=1e-10),
                    "Single-view metric regression: " + source["view"] + " " + key)

    output = ROOT / "data/derived/uniform_fusion" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "predictions.npz"
    np.savez_compressed(predictions_path, sample_identifiers=bundles[0]["ids"],
                        class_names=bundles[0]["classes"], labels=labels,
                        method_names=np.array(names), log_probabilities=np.stack(all_logp),
                        predictions=np.stack([values.argmax(1) for values in all_logp]))
    fields = ["method", "accuracy", "macro_f1_supported", "macro_f1_fixed", "balanced_accuracy", "nll", "brier", "ece_15"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name in names:
            writer.writerow({"method": name, **{key: results[name][key] for key in fields[1:]}})
    source_paths = ["scripts/evaluate_uniform_fusion.py", "scripts/uniform_fusion.py",
                    "scripts/calibrate_filtered_view.py", "scripts/historical_policy_dataset.py",
                    "scripts/sbsmi_dataset.py", "scripts/float_image_dataset.py"]
    report = {"schema_version": 1, "month": "2020-03", "samples": len(labels),
              "weights": {name: 1/3 for name in names[:3]}, "temperature_fit_month": "2020-02",
              "evaluation_labels_used_for_optimization": False, "future_test_used": False,
              "network_weight_updates": False, "alignment": "exact IDs/order/classes/labels; canonical manifest checked",
              "policy": common_policy, "sources": provenance, "metrics": results,
              "register_sha256": sha(register_path), "predictions_sha256": sha(predictions_path),
              "source_sha256": {path: sha(ROOT / path) for path in source_paths},
              "numpy_version": np.__version__, "torch_version": str(torch.__version__),
              "limitations": "Single-seed historical development validation already observed; no significance test or future claim."}
    save_json(output / "report.json", report)
    # Verify persisted payload, without rerunning any model.
    with np.load(predictions_path, allow_pickle=False) as saved:
        require(np.array_equal(saved["sample_identifiers"], bundles[0]["ids"]), "Output ID round-trip")
        require(np.array_equal(saved["log_probabilities"], np.stack(all_logp)), "Output probability round-trip")
    for name in names:
        metrics = results[name]
        print(f"{name}: accuracy={metrics['accuracy']:.6f} macro-F1={metrics['macro_f1_supported']:.6f} NLL={metrics['nll']:.6f}")
    print("Report:", output / "report.json")
    print("CPU historical evaluation only; no fitting or future-test access.")


if __name__ == "__main__":
    main()
