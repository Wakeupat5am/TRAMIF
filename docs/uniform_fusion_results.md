# Uniform calibrated fusion — historical development result

Date: 2026-10-13. Base commit: 6b65430 plus uncommitted fusion implementation.
Run: data/derived/uniform_fusion/20261013_082927_879263/.
Artifacts: report.json, comparison.csv, predictions.npz (four methods' log
probabilities, labels, sample IDs, class order, predictions). Confusion matrices,
per-class scores and ECE bins are stored in report.json.

## Contract and verification
- March 2020, 4,332 retained samples, threshold 0, 51 fixed output classes,
  47 supported classes. Existing three seed-42 epoch-100 checkpoints unchanged.
- Temperatures fitted on February only. Weights fixed at 1/3 before evaluation.
- IDs, their exact order, class order and labels agree across all views; each
  input also passes the existing canonical HistoricalPolicyDataset validation.
- Checkpoint/report/prediction/temperature hashes and February fitting provenance
  checked. Single-view accuracy/NLL/Brier/ECE reproduced to absolute 1e-10.
- No intersection/subset selection, fitted weights, temperature refit, GPU
  inference/training or future access. March labels only check integrity and score.
- Initial execution rejected a reader schema mismatch (february/march versus
  2020-02/2020-03 dictionary keys), before any output was written. Fixed reader;
  no input artifacts changed. Subsequent real-data integration succeeded.

## Results
| Method | Accuracy | Macro-F1 supported | Macro-F1 fixed 51 | Balanced accuracy | NLL | Brier | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SBSMI calibrated | 0.702678 | 0.727771 | 0.670691 | 0.727025 | 1.070787 | 0.388283 | 0.106310 |
| Raw-byte calibrated | 0.673130 | 0.650515 | 0.599495 | 0.657894 | 1.302756 | 0.460390 | 0.047595 |
| Entropy calibrated | 0.632733 | 0.659303 | 0.607593 | 0.650162 | 1.308594 | 0.483171 | 0.060347 |
| Uniform calibrated | 0.709603 | 0.743854 | 0.685513 | 0.728113 | 1.107323 | 0.407841 | 0.077073 |

Versus SBSMI: accuracy +0.6925 percentage points; supported macro-F1 +0.016084;
NLL and Brier worsen. ECE improves versus SBSMI but not versus both compact views.
These are observed differences, not statistical significance or general superiority.
March has already been used for development; one seed, no confidence intervals,
no future evidence. Reliability weighting is not evaluated by this experiment.

## Reproduce this bounded evaluation
`python -m scripts.evaluate_uniform_fusion`
Uses docs/baseline_artifacts.json and the pinned historical policy directory.
CPU only; each run creates a new directory. It does not refit anything.
Tests: `python -m unittest discover -s tests -p "test*.py" -v`:
81 passed after evaluation (0.649 seconds).
