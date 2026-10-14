"""Verify the existing historical baseline chain without loading models or images."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "sbsmi": ("sbsmi_training/20261005_105325_352508", "temperature_filtered_20261011_021110_098860"),
    "raw-byte": ("float_view_training/raw-byte/20261009_145102_408184", "temperature_filtered_20261011_021130_959168"),
    "entropy": ("float_view_training/entropy/20261011_001259_203865", "temperature_filtered_20261011_021150_219503"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-register", action="store_true")
    args = parser.parse_args()
    entries = []
    for view, (run_name, temperature_name) in RUNS.items():
        run = ROOT / "data/derived" / run_name
        temperature_path = run / temperature_name / "temperature.json"
        temperature = json.loads(temperature_path.read_bytes())
        checkpoint = run / "epoch_100.pt"
        require(digest(checkpoint) == temperature["checkpoint_sha256"], "Checkpoint hash: " + view)
        require(temperature["view"] == view and temperature["fit_month"] == "2020-02"
                and temperature["fit_samples"] == 3799, "Calibration scope: " + view)
        require(temperature["future_test_used"] is False, "Future flag: " + view)
        for name, expected in temperature["source_code_sha256"].items():
            require(digest(ROOT / name) == expected, "Calibration source changed: " + name)
        evaluations = []
        for month, count in (("2020-02", 3799), ("2020-03", 4332)):
            folders = list(run.glob("validation_filtered_" + month + "_*"))
            require(len(folders) == 1, "Ambiguous evaluation: " + view + " " + month)
            folder = folders[0]
            report_path = folder / "report.json"
            report = json.loads(report_path.read_bytes())
            require(digest(folder / "predictions.npz") == report["predictions_sha256"], "Prediction hash")
            require(report["checkpoint_sha256"] == temperature["checkpoint_sha256"], "Checkpoint binding")
            require(report["class_to_idx"] == temperature["class_to_idx"], "Class mapping")
            require(report["view"] == view and report["month"] == month
                    and report["samples"] == count and report["selection"]["threshold"] == 0,
                    "Evaluation scope")
            require(report["future_test_used"] is False, "Evaluation future flag")
            if month == "2020-02":
                source = temperature["calibration_source"]
                require(digest(report_path) == source["report_sha256"], "Calibration report binding")
                require(report["predictions_sha256"] == source["predictions_sha256"], "Calibration prediction binding")
            evaluations.append({
                "month": month, "samples": count,
                "report": report_path.relative_to(ROOT).as_posix(),
                "report_sha256": digest(report_path),
                "predictions_sha256": report["predictions_sha256"],
                "accuracy": report["accuracy"],
                "macro_f1_supported": report["macro_f1_supported_classes"],
            })
        entries.append({
            "view": view, "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": digest(checkpoint),
            "temperature_file": temperature_path.relative_to(ROOT).as_posix(),
            "temperature_file_sha256": digest(temperature_path),
            "temperature": temperature["temperature"], "evaluations": evaluations,
        })
        print(view + ": checkpoint, calibration provenance and prediction hashes PASS")
    if args.write_register:
        register = {
            "schema_version": 1,
            "scope": "Historical artifact hash/binding checks only; metrics copied, not recomputed. Not a backup.",
            "entries": entries,
        }
        output = ROOT / "docs/baseline_artifacts.json"
        output.write_text(json.dumps(register, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("Saved docs/baseline_artifacts.json")


if __name__ == "__main__":
    main()
