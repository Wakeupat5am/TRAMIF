"""Read-only verification of the frozen quality weights before later evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.february_group_audit import ROOT, require, sha
from scripts.freeze_quality_weights import verify_oof
from scripts.quality_fusion import estimate_quality


def verify_checksums(folder, checksums):
    folder = folder.resolve()
    for name, expected in checksums.items():
        require(Path(name).name == name and not Path(name).is_absolute(), "Only local filenames allowed")
        require(sha(folder / name) == expected, "Frozen artifact changed: " + name)


def verify_freeze(folder):
    folder = folder.resolve()
    require(folder.parent == (ROOT / "data/derived/quality_weight_freeze").resolve(), "Unexpected freeze location")
    checksums = json.loads((folder / "checksums.json").read_bytes())
    require(set(checksums) == {"weights.json", "fold_manifest.csv", "class_support.csv"}, "Incomplete freeze checksums")
    verify_checksums(folder, checksums)
    frozen = json.loads((folder / "weights.json").read_bytes())
    require(frozen["status"] == "FROZEN_BEFORE_THIS_MARCH_COMPARISON"
            and frozen["quality_coefficient"] == 1 and frozen["drift_coefficient"] == 0
            and frozen["uncertainty_penalty"] == 0 and not frozen["march_labels_used"]
            and not frozen["future_test_used"], "Unapproved freeze settings")
    require(sha(folder / "fold_manifest.csv") == frozen["fold_manifest_sha256"]
            and sha(folder / "class_support.csv") == frozen["support_manifest_sha256"], "Frozen manifests mismatch")
    oof_dir = Path(frozen["oof_directory"])
    require(sha(oof_dir / "report.json") == frozen["oof_report_sha256"], "OOF report changed")
    report, audit, audit_dir, _, arrays = verify_oof(oof_dir)
    require(report["prediction_sha256"] == frozen["oof_predictions_sha256"]
            and report["audit_report_sha256"] == frozen["audit_report_sha256"], "OOF binding")
    for name in ("fold_manifest.csv", "class_support.csv"):
        require(sha(folder / name) == sha(audit_dir / name), "Copied manifest changed")
    require(frozen["all_classes"] == arrays["class_names"].tolist()
            and frozen["support_classes"] == audit["support_classes"]
            and [frozen["all_classes"][i] for i in frozen["support_indices"]] == frozen["support_classes"], "Support/class binding")
    reproduced = estimate_quality(arrays["log_probabilities"], arrays["labels"], frozen["support_indices"])
    for key, value in reproduced.items():
        require(np.array_equal(value, frozen[key]), "Frozen numerical mismatch: " + key)
    for path, digest in frozen["source_sha256"].items():
        require(sha(ROOT / path) == digest, "Freeze source changed: " + path)
    require(frozen["february_provenance"] == report["provenance"], "February provenance changed")
    for entry, source in zip(frozen["inference_calibration"], report["provenance"]):
        path = ROOT / entry["temperature_file"]
        require(sha(path) == entry["temperature_file_sha256"], "Inference temperature changed")
        t = json.loads(path.read_bytes())
        require(entry["view"] == source["view"] == t["view"]
                and entry["temperature"] == t["temperature"]
                and entry["checkpoint_sha256"] == source["checkpoint_sha256"] == t["checkpoint_sha256"]
                and t["calibration_source"] == source["february_source"], "Inference provenance changed")
    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-dir", type=Path, required=True)
    args = parser.parse_args()
    frozen = verify_freeze(args.freeze_dir)
    print("Audit inputs/artifacts and five-fold manifest: PASS")
    print("OOF probabilities reproduced without refitting: PASS")
    print("Quality and weights reproduced exactly: PASS")
    print("Inference temperature / checkpoint provenance: PASS")
    print("Standard OOF NLL (all February):", frozen["standard_nll_all_february"])
    print("Frozen weights:", frozen["weights"])
    print("Weights SHA-256:", sha(args.freeze_dir / "weights.json"))
    print("Read-only; no March predictions or Future Test accessed.")


if __name__ == "__main__":
    main()
