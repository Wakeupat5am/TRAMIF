"""Task 2: CPU-only calibration using February artifacts and one audited manifest."""
import argparse
import csv
import json
from datetime import datetime, timezone

import numpy as np
import torch

from scripts.calibrate_filtered_view import load_filtered_predictions
from scripts.calibrate_sbsmi import T_MIN, T_MAX
from scripts.february_group_audit import ROOT, POLICY, CONFIG, require, save_json, sha
from scripts.quality_fusion import cross_fit
from scripts.uniform_fusion import load_bundle, validate_alignment

VIEWS = ["sbsmi", "raw-byte", "entropy"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def checked_audit(folder):
    folder = folder.resolve()
    require(folder.parent == (ROOT / "data/derived/february_group_audit").resolve(), "Unexpected audit location")
    audit = json.loads((folder / "report.json").read_bytes())
    require(audit["status"] == "FEASIBLE_ENGINEERING_ONLY" and all(audit["gates"].values()), "Audit did not pass")
    require(audit["config"] == CONFIG and not audit["future_test_accessed"] and not audit["march_used_for_assignment"], "Audit config/scope changed")
    for filename, digest in audit["artifact_sha256"].items():
        require(sha(folder / filename) == digest, "Audit artifact changed: " + filename)
    for path, digest in audit["source_sha256"].items():
        require(sha(ROOT / path) == digest, "Audit source changed: " + path)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=type(ROOT), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit_dir = args.audit_dir.resolve()
    audit = checked_audit(audit_dir)
    manifest = read_csv(audit_dir / "fold_manifest.csv")
    registry_path = ROOT / "docs/baseline_artifacts.json"
    entries = json.loads(registry_path.read_bytes())["entries"]
    require([e["view"] for e in entries] == VIEWS, "View order mismatch")
    bundles, provenance = [], []
    common_policy = None
    for entry in entries:
        view = entry["view"]
        feb = [item for item in entry["evaluations"] if item["month"] == "2020-02"]
        require(len(feb) == 1, "Ambiguous February source")
        saved = feb[0]
        report_path = ROOT / saved["report"]
        require(sha(report_path) == saved["report_sha256"], "February report checksum")
        _, _, report, source = load_filtered_predictions(report_path.parent, POLICY, view, "2020-02")
        require(sha(ROOT / entry["checkpoint"]) == entry["checkpoint_sha256"] == report["checkpoint_sha256"], "Checkpoint provenance mismatch")
        require(source["predictions_sha256"] == saved["predictions_sha256"], "February prediction hash")
        selection = {key: report["selection"][key] for key in
                     ("threshold", "policy_config_sha256", "policy_report_sha256", "retained_manifest_sha256")}
        if common_policy is None:
            common_policy = selection
        require(selection == common_policy and selection["threshold"] == 0, "Primary policy mismatch")
        require(selection["policy_config_sha256"] == sha(POLICY / "config.json")
                and selection["policy_report_sha256"] == sha(POLICY / "report.json")
                and selection["retained_manifest_sha256"] == sha(POLICY / "threshold_00/validation_keep.csv"), "Policy binding")
        batch = load_bundle(report_path.parent / "predictions.npz", source["predictions_sha256"], 1.0)
        bundles.append(batch)
        provenance.append({"view": view, "checkpoint": entry["checkpoint"],
                           "checkpoint_sha256": entry["checkpoint_sha256"], "february_source": source,
                           "selection": selection, "training_seed": report["training_seed"]})
    validate_alignment(bundles)
    require(bundles[0]["logits"].shape == (3799, 51), "February shape mismatch")
    logs, folds, temperatures = cross_fit(bundles, manifest)
    output = ROOT / "data/derived/february_crossfit" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "oof_predictions.npz", log_probabilities=logs,
                        sample_identifiers=bundles[0]["ids"], class_names=bundles[0]["classes"],
                        labels=bundles[0]["labels"], folds=folds, views=np.array(VIEWS))
    sources = [__file__, ROOT / "scripts/quality_fusion.py", ROOT / "scripts/calibrate_sbsmi.py",
               ROOT / "scripts/calibrate_filtered_view.py", ROOT / "scripts/uniform_fusion.py",
               ROOT / "scripts/historical_policy_dataset.py"]
    # Recheck sources shared with Task 1 after fitting; no model files are written.
    checked_audit(audit_dir)
    result = {"status": "OOF_CALIBRATION_COMPLETE", "views": VIEWS, "samples": 3799, "classes": 51,
              "month": "2020-02", "objective": "standard mean NLL", "bounds": [T_MIN, T_MAX],
              "folds": 5, "oof_predictions_per_sample_per_view": 1, "temperatures": temperatures,
              "audit_directory": str(audit_dir), "audit_report_sha256": sha(audit_dir / "report.json"),
              "fold_manifest_sha256": sha(audit_dir / "fold_manifest.csv"),
              "source_sha256": {str(type(ROOT)(p).relative_to(ROOT).as_posix()): sha(p) for p in sources},
              "registry_sha256": sha(registry_path), "provenance": provenance,
              "prediction_sha256": sha(output / "oof_predictions.npz"),
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "future_test_used": False, "march_used": False, "cnn_retrained": False,
              "limits": "OOF scalar calibration conditional on fixed models and February grouping; not temporal episodes."}
    save_json(output / "report.json", result)
    for v in range(3):
        print(VIEWS[v], [round(t["temperature"], 6) for t in temperatures if t["view_index"] == v])
    print("Exactly one OOF prediction per sample/view: PASS")
    print("Report:", output / "report.json")


if __name__ == "__main__":
    main()
