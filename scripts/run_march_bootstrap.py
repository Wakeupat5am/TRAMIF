"""Approved 2,000 paired whole-group draws, seed 42, fixed March predictions."""
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.check_quality_freeze import verify_freeze, verify_checksums
from scripts.february_group_audit import ROOT, require, sha, save_json, save_csv
from scripts.march_group_feasibility import support_matrix
from scripts.paired_group_bootstrap import (
    METRICS, sample_metrics, aggregate_groups, draw_multiplicities, resampled_metrics,
    paired_differences, percentile_interval, distribution_diagnostic, influence_diagnostics,
)

EVAL = ROOT / "data/derived/quality_march_evaluation/20261013_125046_074164"
AUDIT = ROOT / "data/derived/march_group_feasibility/20261013_125459_149338"
FREEZE = ROOT / "data/derived/quality_weight_freeze/20261013_124031_275438"
METHODS = ["sbsmi", "raw-byte", "entropy", "uniform-calibrated", "quality-only-calibrated"]
REFERENCES = [3, 0, 1, 2]  # Primary first, then the three secondary references.
CONFIG = {"replicates": 2000, "seed": 42, "rng": "numpy default_rng / PCG64",
          "resampling": "G whole active groups with replacement, multinomial multiplicities shared by all methods",
          "metrics": METRICS, "primary": "quality-only minus uniform", "secondary": ["sbsmi", "raw-byte", "entropy"],
          "percentiles": [.025, .975], "quantile_method": "linear", "grouping_threshold": 10,
          "primary_retention_threshold": 0, "drop_missing_class_replicates": False,
          "simultaneous_coverage_or_multiplicity_adjustment": False}


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    torch.set_num_threads(1)
    require(sha(EVAL / "report.json") == "6a4d9191deedabf859bd7594d3addbe56a4f0d8048f824cda02e56ec66c0d922", "Unrecognized March report")
    require(sha(AUDIT / "report.json") == "70cbcac4d34819291593a719599a11ff3b3637ab0b067db3060b539c6a3f523d", "Unrecognized grouping report")
    frozen = verify_freeze(FREEZE)
    evaluation, audit = [json.loads((p / "report.json").read_bytes()) for p in (EVAL, AUDIT)]
    for folder in (EVAL, AUDIT):
        verify_checksums(folder, json.loads((folder / "checksums.json").read_bytes()))
    for source in (evaluation, audit):
        for p, digest in source["source_sha256"].items():
            require(sha(ROOT / p) == digest, "Source changed: " + p)
    require(evaluation["weights_sha256"] == sha(FREEZE / "weights.json") and evaluation["weights"] == frozen["weights"], "Weight binding")
    require(audit["threshold"] == 10 and audit["primary_policy_threshold"] == 0 and audit["retained_samples"] == 4332
            and not audit["sample_selection_changed"] and not audit["future_test_used"], "Audit scope")
    with np.load(io.BytesIO((EVAL / "predictions.npz").read_bytes()), allow_pickle=False) as data:
        logs, labels, ids, classes = [data[k].copy() for k in ("log_probabilities", "labels", "sample_identifiers", "class_names")]
        require(data["method_names"].tolist() == METHODS and logs.shape == (5, 4332, 51), "Method/shape mismatch")
    require(len(set(ids.tolist())) == 4332 and classes.tolist() == frozen["all_classes"], "ID/class order mismatch")
    members = csv_rows(AUDIT / "members.csv")
    for r in members:
        require(r["retained"] in ("True", "False") and r["timestamp"].startswith("2020-03-"), "Member scope")
        r["retained"] = r["retained"] == "True"
    retained = {r["sha"]: r for r in members if r["retained"]}
    require(len(retained) == sum(r["retained"] for r in members) == 4332 and set(retained) == set(ids.tolist()), "Retained coverage")
    sample_groups = []
    for sid, y in zip(ids, labels):
        require(retained[str(sid)]["family"] == classes[y], "Group/prediction label mismatch")
        sample_groups.append(retained[str(sid)]["group_id"])
    values = sample_metrics(logs, labels)
    groups, inverse, sizes, sums = aggregate_groups(values, np.asarray(sample_groups))
    support_groups, group_class_counts = support_matrix(members, classes.tolist())
    require(groups.tolist() == support_groups and len(groups) == 2234 and np.array_equal(group_class_counts.sum(1), sizes), "Group counts mismatch")
    point = values.mean(0)
    for method, scores in zip(METHODS, point):
        for metric, score in zip(METRICS, scores):
            require(np.isclose(score, evaluation["metrics"][method][metric], rtol=0, atol=1e-12), "Point metric regression")
    print("Freeze, March predictions, group manifest and original metrics: PASS", flush=True)
    source_paths = [EVAL / "report.json", EVAL / "predictions.npz", AUDIT / "report.json", AUDIT / "members.csv",
                    AUDIT / "pilot_counts.npz", FREEZE / "weights.json", ROOT / "docs/class_mapping.json",
                    Path(__file__), ROOT / "scripts/paired_group_bootstrap.py", ROOT / "scripts/check_quality_freeze.py",
                    ROOT / "scripts/march_group_feasibility.py"]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    draws = draw_multiplicities(len(groups), CONFIG["replicates"], CONFIG["seed"])
    # Recreating the same multiplicities verifies determinism; not extra draws
    # used to select an estimate or an alternative seed.
    require(np.array_equal(draws, draw_multiplicities(len(groups), 2000, 42)), "RNG reproducibility")
    with np.load(AUDIT / "pilot_counts.npz", allow_pickle=False) as pilot:
        require(np.array_equal(groups, pilot["group_ids"]) and np.array_equal(draws[:500], pilot["sampled_group_multiplicities"]), "Seed-42 pilot prefix mismatch")
    sample_counts, boot = resampled_metrics(draws, sizes, sums)
    # Independent expanded-sample check: all methods use the same repeated IDs.
    expanded_checks = sorted({0, 1, 1999, int(sample_counts.argmin()), int(sample_counts.argmax())})
    for b in expanded_checks:
        expanded = np.repeat(np.arange(len(ids)), draws[b, inverse])
        require(len(expanded) == sample_counts[b], "Expanded sample count")
        require(np.allclose(values[expanded].mean(0), boot[b], rtol=0, atol=1e-12), "Expanded paired metric mismatch")
    class_counts = draws @ group_class_counts
    missing = (class_counts == 0) & (group_class_counts.sum(0) > 0)[None, :]
    differences = paired_differences(boot, 4, REFERENCES)
    intervals = percentile_interval(differences)
    point_deltas = point[4, None, :] - point[REFERENCES, :]
    distribution = {METHODS[r]: {metric: distribution_diagnostic(differences[:, c, j])
                    for j, metric in enumerate(METRICS)} for c, r in enumerate(REFERENCES)}
    influence_rows, influence = influence_diagnostics(sizes, sums, groups, draws, differences[:, 0])
    # These flags request interpretation, not filtering, seed changes or CI changes.
    flags = []
    for ref, stats in distribution.items():
        for metric, d in stats.items():
            if d["skewness_moment"] is not None and abs(d["skewness_moment"]) > 2:
                flags.append(f"{ref}/{metric}: |moment skewness| > 2")
            if d["excess_kurtosis_moment"] is not None and d["excess_kurtosis_moment"] > 10:
                flags.append(f"{ref}/{metric}: excess kurtosis > 10")
    comparisons = [{"comparison": "primary" if c == 0 else "secondary", "reference": METHODS[r], "metric": metric,
                    "point_difference": float(point_deltas[c, j]), "ci_lower": float(intervals[0, c, j]),
                    "ci_upper": float(intervals[1, c, j]), "improvement_direction": "positive" if j == 0 else "negative"}
                   for c, r in enumerate(REFERENCES) for j, metric in enumerate(METRICS)]
    reps = []
    for b in range(2000):
        row = {"replicate": b, "sample_count": int(sample_counts[b]), "distinct_groups": int((draws[b] > 0).sum()),
               "missing_supported_classes": int(missing[b].sum()),
               "missing_families": "|".join(classes[c] for c in np.flatnonzero(missing[b]))}
        row.update({f"{name}_{metric}": float(boot[b, m, j]) for m, name in enumerate(METHODS) for j, metric in enumerate(METRICS)})
        row.update({f"quality_minus_{METHODS[r]}_{metric}": float(differences[b, c, j]) for c, r in enumerate(REFERENCES) for j, metric in enumerate(METRICS)})
        reps.append(row)
    require(all(sha(ROOT / p) == digest for p, digest in before.items()), "Inputs changed during bootstrap")
    output = ROOT / "data/derived/march_paired_bootstrap" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", CONFIG)
    (output / "grouping_manifest.csv").write_bytes((AUDIT / "members.csv").read_bytes())
    save_csv(output / "sample_group_alignment.csv", [{"sha": str(ids[i]), "group_id": str(groups[inverse[i]]),
              "group_index": int(inverse[i]), "label": int(labels[i])} for i in range(len(ids))], ["sha", "group_id", "group_index", "label"])
    save_csv(output / "comparisons.csv", comparisons, list(comparisons[0]))
    save_csv(output / "replicates.csv", reps, list(reps[0]))
    save_csv(output / "group_influence.csv", influence_rows, list(influence_rows[0]))
    np.savez_compressed(output / "draws_and_metrics.npz", group_ids=groups, sample_identifiers=ids, sample_group_index=inverse,
                        group_sizes=sizes, group_metric_sums=sums, sampled_group_multiplicities=draws, sample_counts=sample_counts,
                        method_names=np.array(METHODS), metric_names=np.array(METRICS), bootstrap_metrics=boot,
                        comparison_references=np.array([METHODS[r] for r in REFERENCES]), paired_differences=differences,
                        class_names=classes, group_class_counts=group_class_counts, replicate_class_counts=class_counts,
                        missing_supported_classes=missing)
    diagnostic = {"distribution": distribution, "group_influence": influence, "review_flags": flags,
                  "diagnostic_flag_thresholds": "Descriptive flags only: absolute skewness >2 or excess kurtosis >10; not significance tests or proofs of validity.",
                  "sample_count_quantiles": {str(q): float(np.quantile(sample_counts, q)) for q in (0, .025, .5, .975, 1)},
                  "replicates_missing_classes": int(missing.any(1).sum()), "replicates_kept": 2000,
                  "replicates_discarded": 0, "nonfinite_metrics": 0,
                  "largest_group": int(sizes.max()), "largest_group_fraction": float(sizes.max()/sizes.sum()),
                  "size_concentration_effective_groups": float(sizes.sum()**2 / np.square(sizes).sum()),
                  "point_estimates_replaced_with_bootstrap_means": False}
    save_json(output / "diagnostics.json", diagnostic)
    report = {"status": "COMPLETE_CONDITIONAL_INTERVALS_REVIEW_REQUIRED" if flags else "COMPLETE_CONDITIONAL_INTERVALS",
              "config": CONFIG, "samples": 4332, "active_groups": 2234, "views": METHODS,
              "original_point_metrics": evaluation["metrics"], "comparisons": comparisons,
              "provenance": before, "input_reports": {"evaluation": str(EVAL), "grouping": str(AUDIT), "freeze": str(FREEZE)},
              "numpy_version": np.__version__, "rng_bit_generator": "PCG64",
              "verification": {"source_hashes": "PASS", "whole_group_pairing": "Shared multiplicity per group for every member and all five methods",
                               "point_metrics": "PASS", "deterministic_regeneration": "PASS", "same_seed_pilot_prefix": "PASS",
                               "explicit_sample_expansion_replicates": expanded_checks, "expanded_metric_check": "PASS"},
              "no_intervals_for": ["macro_f1_supported", "macro_f1_fixed", "balanced_accuracy", "ece_15"],
              "future_test_used": False, "weights_changed": False, "training_performed": False,
              "limitations": ["Conditional on fixed checkpoints, temperatures and weights; March is development data.",
                              "TLSH components do not establish independent/exchangeable sampling units.",
                              "Does not capture training-seed or temporal variability, or all calibration/weight-estimation uncertainty.",
                              "Pointwise percentile CIs, no multiple-comparison adjustment; no simultaneous superiority claim.",
                              "Seed 42 shares first 500 support-only pilot draws; no pilot performance outcomes were used to select draws.",
                              "Size imbalance and finite 2000-draw Monte Carlo approximation remain; inspect diagnostics."],
              "artifact_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}
    save_json(output / "report.json", report)
    save_json(output / "checksums.json", {p.name: sha(p) for p in sorted(output.iterdir())})
    with np.load(output / "draws_and_metrics.npz", allow_pickle=False) as saved:
        require(np.array_equal(saved["sampled_group_multiplicities"], draws)
                and np.array_equal(saved["paired_differences"], differences), "Artifact round-trip")
    print(json.dumps(comparisons, indent=2))
    print("Sample counts:", diagnostic["sample_count_quantiles"])
    print("Missing-class replicates retained:", diagnostic["replicates_missing_classes"])
    print("Diagnostic review flags:", flags)
    print("Report:", output / "report.json")


if __name__ == "__main__":
    main()
