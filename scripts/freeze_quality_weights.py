"""Task 3: verify February OOF artifacts, estimate quality, write immutable weights."""
import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.crossfit_february import VIEWS, checked_audit, read_csv
from scripts.february_group_audit import ROOT, require, save_json, sha
from scripts.quality_fusion import aligned_folds, estimate_quality
from scripts.uniform_fusion import calibrated_log_probabilities, load_bundle


def verify_oof(folder):
    folder = folder.resolve()
    require(folder.parent == (ROOT / "data/derived/february_crossfit").resolve(), "Unexpected OOF location")
    report = json.loads((folder / "report.json").read_bytes())
    require(report["status"] == "OOF_CALIBRATION_COMPLETE" and report["month"] == "2020-02"
            and report["views"] == VIEWS and not report["march_used"] and not report["future_test_used"]
            and not report["cnn_retrained"] and report["bounds"] == [0.05, 20.0]
            and report["objective"] == "standard mean NLL" and report["folds"] == 5, "OOF scope/config")
    audit_dir = Path(report["audit_directory"])
    audit = checked_audit(audit_dir)
    require(sha(audit_dir / "report.json") == report["audit_report_sha256"]
            and sha(audit_dir / "fold_manifest.csv") == report["fold_manifest_sha256"], "Audit binding")
    for path, digest in report["source_sha256"].items():
        require(sha(ROOT / path) == digest, "Crossfit source changed: " + path)
    raw = (folder / "oof_predictions.npz").read_bytes()
    import hashlib
    require(hashlib.sha256(raw).hexdigest() == report["prediction_sha256"], "OOF checksum")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    require(arrays["views"].tolist() == VIEWS and arrays["log_probabilities"].shape == (3, 3799, 51), "OOF shape/order")
    manifest = read_csv(audit_dir / "fold_manifest.csv")
    bundles = []
    for source in report["provenance"]:
        require(source["view"] == VIEWS[len(bundles)], "Provenance order")
        require(sha(ROOT / source["checkpoint"]) == source["checkpoint_sha256"], "Checkpoint changed")
        feb = source["february_source"]
        require(sha(Path(feb["folder"]) / "report.json") == feb["report_sha256"], "February report changed")
        bundles.append(load_bundle(Path(feb["folder"]) / "predictions.npz", feb["predictions_sha256"], 1.0))
    folds = aligned_folds(bundles, manifest)
    for saved, expected in (("sample_identifiers", bundles[0]["ids"]), ("class_names", bundles[0]["classes"]),
                            ("labels", bundles[0]["labels"]), ("folds", folds)):
        require(np.array_equal(arrays[saved], expected), "OOF alignment: " + saved)
    require(len(report["temperatures"]) == 15, "Expected 15 temperature fits")
    keys = set()
    for t in report["temperatures"]:
        v, f = t["view_index"], t["fold"]
        require(v in range(3) and f in range(5) and (v, f) not in keys, "Invalid/duplicate temperature")
        keys.add((v, f))
        mask = folds == f
        require(t["score_samples"] == int(mask.sum()) and t["fit_samples"] == int((~mask).sum()), "OOF fit/score counts")
        require(np.isfinite(t["temperature"]) and .05 <= t["temperature"] <= 20, "Temperature outside bounds")
        expected = calibrated_log_probabilities(bundles[v]["logits"][mask], t["temperature"])
        require(np.array_equal(arrays["log_probabilities"][v, mask], expected), "OOF probabilities do not reproduce")
    return report, audit, audit_dir, manifest, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.oof_dir.resolve()
    report, audit, audit_dir, manifest, arrays = verify_oof(folder)
    classes = arrays["class_names"].tolist()
    support_rows = read_csv(audit_dir / "class_support.csv")
    support = []
    for c, family in enumerate(classes):
        selected = [r for r in manifest if r["family"] == family]
        groups = len({r["group_id"] for r in selected})
        include = len(selected) >= 20 and groups >= 5
        saved = [r for r in support_rows if r["family"] == family]
        require(len(saved) == 1 and int(saved[0]["samples"]) == len(selected)
                and int(saved[0]["groups"]) == groups
                and (saved[0]["in_support"] == "True") == include, "Support manifest mismatch")
        if include:
            support.append(c)
    require([classes[c] for c in support] == audit["support_classes"], "Fixed support changed")
    quality = estimate_quality(arrays["log_probabilities"], arrays["labels"], support)
    output = ROOT / "data/derived/quality_weight_freeze" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("fold_manifest.csv", "class_support.csv"):
        (output / name).write_bytes((audit_dir / name).read_bytes())
    registry = json.loads((ROOT / "docs/baseline_artifacts.json").read_bytes())
    inference = []
    require([e["view"] for e in registry["entries"]] == VIEWS, "Inference view order")
    for entry, source in zip(registry["entries"], report["provenance"]):
        path = ROOT / entry["temperature_file"]
        require(sha(path) == entry["temperature_file_sha256"], "Inference temperature checksum")
        t = json.loads(path.read_bytes())
        require(t["view"] == entry["view"] and t["fit_month"] == "2020-02" and t["fit_samples"] == 3799
                and t["checkpoint_sha256"] == source["checkpoint_sha256"] == entry["checkpoint_sha256"]
                and t["calibration_source"] == source["february_source"]
                and t["temperature"] == entry["temperature"] and not t["future_test_used"], "Inference calibration binding")
        inference.append({"view": entry["view"], "checkpoint_sha256": source["checkpoint_sha256"],
                          "temperature": t["temperature"], "temperature_file": entry["temperature_file"],
                          "temperature_file_sha256": entry["temperature_file_sha256"]})
    frozen = {"status": "FROZEN_BEFORE_THIS_MARCH_COMPARISON", "created_utc": datetime.now(timezone.utc).isoformat(),
              "formula": "w = softmax(-class_balanced_OOF_NLL)", "quality_coefficient": 1,
              "drift_coefficient": 0, "uncertainty_penalty": 0, "views": VIEWS,
              **quality, "support_indices": support, "support_classes": [classes[c] for c in support],
              "support_samples": audit["support_samples"], "all_classes": classes,
              "oof_directory": str(folder), "oof_report_sha256": sha(folder / "report.json"),
              "oof_predictions_sha256": report["prediction_sha256"],
              "audit_report_sha256": report["audit_report_sha256"],
              "fold_manifest_sha256": sha(output / "fold_manifest.csv"),
              "support_manifest_sha256": sha(output / "class_support.csv"),
              "inference_calibration": inference, "february_provenance": report["provenance"],
              "source_sha256": {p: sha(ROOT / p) for p in ("scripts/quality_fusion.py", "scripts/freeze_quality_weights.py")},
              "march_labels_used": False, "future_test_used": False,
              "bootstrap_uncertainty": {"status": "NOT_ESTIMATED", "reason": "Optional diagnostic deferred; no uncertainty penalty. Fixed-S missing-class replicates need explicit treatment."},
              "limitations": ["31-class historical quality support; not quality of every one of the 51 outputs.",
                              "Single seed and February period; overlapping temperature-fit subsets are not independent trials.",
                              "March single-view and uniform results were already observed; not retrospective preregistration."]}
    save_json(output / "weights.json", frozen)
    save_json(output / "checksums.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    for v, loss, w in zip(VIEWS, quality["class_balanced_nll"], quality["weights"]):
        print(f"{v}: balanced OOF NLL={loss:.9f} | frozen weight={w:.9f}")
    print("OOF recomputation from saved fold temperatures: PASS (no refitting)")
    print("Frozen weights:", output / "weights.json")


if __name__ == "__main__":
    main()
