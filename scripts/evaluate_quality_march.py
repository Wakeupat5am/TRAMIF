"""Fixed March development metrics; no fitting or uncertainty estimation."""
import io
import json
from datetime import datetime, timezone

import numpy as np
import torch

from scripts.check_quality_freeze import verify_freeze
from scripts.calibrate_filtered_view import load_filtered_predictions
from scripts.february_group_audit import ROOT, POLICY, require, sha, save_json, save_csv
from scripts.quality_fusion import weighted_fusion
from scripts.uniform_fusion import load_bundle, fuse_uniform, classification_metrics

FREEZE = ROOT / "data/derived/quality_weight_freeze/20261013_124031_275438"
PREVIOUS = ROOT / "data/derived/uniform_fusion/20261013_082927_879263"
VIEWS = ["sbsmi", "raw-byte", "entropy"]
METRICS = ["accuracy", "macro_f1_supported", "macro_f1_fixed", "balanced_accuracy", "nll", "brier", "ece_15"]


def compare(bundles, weights):
    # fuse_uniform validates all IDs, their order, class order and labels FIRST.
    single, uniform = fuse_uniform(bundles)
    quality = weighted_fusion(np.stack(single), weights)
    names = VIEWS + ["uniform-calibrated", "quality-only-calibrated"]
    logs = np.stack(single + [uniform, quality])
    metrics = {name: classification_metrics(lp, bundles[0]["labels"]) for name, lp in zip(names, logs)}
    return names, logs, metrics


def main():
    torch.set_num_threads(1)
    freeze_hash = sha(FREEZE / "weights.json")
    frozen = verify_freeze(FREEZE)
    require(frozen["views"] == VIEWS and len(frozen["inference_calibration"]) == 3, "Frozen view order")
    registry_path = ROOT / "docs/baseline_artifacts.json"
    entries = json.loads(registry_path.read_bytes())["entries"]
    require([e["view"] for e in entries] == VIEWS, "Registry views")
    bundles, sources = [], []
    for entry, inference, feb in zip(entries, frozen["inference_calibration"], frozen["february_provenance"]):
        matches = [r for r in entry["evaluations"] if r["month"] == "2020-03"]
        require(len(matches) == 1, "Ambiguous March source")
        saved = matches[0]
        path = ROOT / saved["report"]
        require(sha(path) == saved["report_sha256"], "March report hash")
        _, labels, report, source = load_filtered_predictions(path.parent, POLICY, entry["view"], "2020-03")
        require(report["checkpoint_sha256"] == inference["checkpoint_sha256"] == entry["checkpoint_sha256"], "Checkpoint mismatch")
        temperature_path = ROOT / inference["temperature_file"]
        require(sha(temperature_path) == inference["temperature_file_sha256"], "Temperature hash")
        t = json.loads(temperature_path.read_bytes())
        require(t["fit_month"] == "2020-02" and t["fit_samples"] == 3799 and not t["future_test_used"], "Temperature scope")
        for key in ("checkpoint_sha256", "checkpoint_epoch", "training_seed", "dataset_config_sha256", "class_to_idx"):
            require(report[key] == t[key], "Evaluation/calibration mismatch: " + key)
        selection = {key: report["selection"][key] for key in feb["selection"]}
        require(selection == feb["selection"] and selection["threshold"] == 0, "Primary policy changed")
        require(source["predictions_sha256"] == saved["predictions_sha256"], "Prediction registry hash")
        bundle = load_bundle(path.parent / "predictions.npz", source["predictions_sha256"], inference["temperature"])
        require(np.array_equal(bundle["labels"], labels.numpy()) and bundle["classes"].tolist() == frozen["all_classes"], "Labels/classes mismatch")
        bundles.append(bundle)
        sources.append({"view": entry["view"], **source, "checkpoint_sha256": report["checkpoint_sha256"],
                        "inference_calibration": inference, "selection": report["selection"]})
    require(bundles[0]["logits"].shape == (4332, 51), "March sample count")
    names, logs, results = compare(bundles, frozen["weights"])
    # Regression against the already recorded four-method March experiment.
    old = json.loads((PREVIOUS / "report.json").read_bytes())
    require(sha(PREVIOUS / "predictions.npz") == old["predictions_sha256"], "Previous prediction hash")
    with np.load(io.BytesIO((PREVIOUS / "predictions.npz").read_bytes()), allow_pickle=False) as data:
        require(data["method_names"].tolist() == names[:4], "Previous method order")
        for old_key, key in (("sample_identifiers", "ids"), ("class_names", "classes"), ("labels", "labels")):
            require(np.array_equal(data[old_key], bundles[0][key]), "Previous evaluation alignment")
        require(np.array_equal(data["log_probabilities"], logs[:4]), "Previous probability regression")
    for name in names[:4]:
        for key in METRICS:
            require(np.isclose(results[name][key], old["metrics"][name][key], rtol=0, atol=1e-12), "Baseline regression")
    require(sha(FREEZE / "weights.json") == freeze_hash, "Weights changed during evaluation")
    output = ROOT / "data/derived/quality_march_evaluation" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "predictions.npz", sample_identifiers=bundles[0]["ids"],
                        class_names=bundles[0]["classes"], labels=bundles[0]["labels"],
                        method_names=np.array(names), log_probabilities=logs, predictions=logs.argmax(2))
    rows = [{"method": name, **{key: results[name][key] for key in METRICS}} for name in names]
    save_csv(output / "comparison.csv", rows, ["method"] + METRICS)
    deltas = {key: results[names[-1]][key] - results[names[-2]][key] for key in METRICS}
    source_paths = ["scripts/evaluate_quality_march.py", "scripts/check_quality_freeze.py", "scripts/quality_fusion.py",
                    "scripts/uniform_fusion.py", "scripts/calibrate_filtered_view.py", "scripts/historical_policy_dataset.py"]
    result = {"status": "MARCH_DEVELOPMENT_METRICS_COMPLETE", "samples": 4332, "classes": 51,
              "supported_classes": results[names[-1]]["supported_classes"], "month": "2020-03",
              "weights": frozen["weights"], "views": VIEWS, "freeze_directory": str(FREEZE),
              "weights_sha256": freeze_hash, "registry_sha256": sha(registry_path),
              "previous_uniform_report_sha256": sha(PREVIOUS / "report.json"),
              "sources": sources, "source_sha256": {p: sha(ROOT / p) for p in source_paths},
              "metrics": results, "quality_minus_uniform": deltas,
              "future_test_used": False, "march_labels_used_for_optimization": False,
              "weights_changed": False, "training_performed": False, "bootstrap_performed": False,
              "alignment": "Canonical retained March IDs/classes/labels and exact three-view order checked",
              "baseline_regression": "All four previous methods: identical probabilities, seven metrics reproduced",
              "limitations": "Single-seed development evaluation already observed; no final holdout or statistical superiority claim.",
              "numpy_version": np.__version__, "torch_version": torch.__version__,
              "artifact_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}
    save_json(output / "report.json", result)
    save_json(output / "checksums.json", {p.name: sha(p) for p in sorted(output.iterdir())})
    with np.load(output / "predictions.npz", allow_pickle=False) as data:
        require(np.array_equal(data["log_probabilities"], logs), "Prediction round-trip")
    print(json.dumps(rows, indent=2))
    print("Quality minus uniform:", deltas)
    print("Report:", output / "report.json")


if __name__ == "__main__":
    main()
