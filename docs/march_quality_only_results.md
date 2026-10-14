# March quality-only development evaluation — 2026-10-13

Task 4A COMPLETE. This is development evaluation, not a final unseen holdout.
No March labels used for fitting or optimization; no Future Test or CNN training.

## Frozen inputs and verification
Same 4,332 primary-retained March samples, 51 output classes, 47 supported classes.
Absent: allaple, ausiv, socks, tescrypt. Canonical IDs, ID order, class order and
labels verified across the three views. Checkpoints and near-duplicate policy 0
unchanged. All February OOF/weight-freeze provenance checks pass before evaluation.

Weights read at full precision from
data/derived/quality_weight_freeze/20261013_124031_275438/weights.json:
- SBSMI: 0.4017215600259309
- Raw-byte: 0.2866850774030112
- Entropy: 0.3115933625710579

Full-February inference temperatures: 2.5202034447760657, 2.211872879462139,
1.9354895383696529, respectively. No OOF temperature averaging or refitting.
The three single-view and uniform predictions reproduce the previous March
artifact exactly; all seven previous metrics also match (tolerance 1e-12).

## All requested point metrics
Accuracy and balanced accuracy below are percentages. F1, NLL, Brier and ECE are
raw values. NLL uses natural logarithms; Brier is the sum over 51 classes; ECE uses
15 fixed equal-width top-confidence bins. Supported F1 averages over the original
47 classes, fixed F1 over all 51 with zero for undefined 0/0 class F1.

| Method | Accuracy % | Macro-F1 (47) | Macro-F1 (51) | Balanced accuracy % | NLL | Brier | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SBSMI calibrated | 70.2678 | 0.727771 | 0.670691 | 72.7025 | 1.070787 | 0.388283 | 0.106310 |
| Raw-byte calibrated | 67.3130 | 0.650515 | 0.599495 | 65.7894 | 1.302756 | 0.460390 | 0.047595 |
| Entropy calibrated | 63.2733 | 0.659303 | 0.607593 | 65.0162 | 1.308594 | 0.483171 | 0.060347 |
| Uniform calibrated | 70.9603 | 0.743854 | 0.685513 | 72.8113 | 1.107323 | 0.407841 | 0.077073 |
| Quality-only calibrated | 71.4681 | 0.748931 | 0.690191 | 73.4081 | 1.094385 | 0.401633 | 0.076168 |

Quality minus uniform: accuracy +0.507849 percentage points (22 more correct
predictions), supported F1 +0.005076144, fixed F1 +0.004678015, balanced accuracy
+0.596792 percentage points, NLL -0.012938074, Brier -0.006207650, ECE -0.000905249.
All seven point estimates move favorably versus uniform. This is not a significance
claim and does not establish future reliability. Versus SBSMI, quality-only has
better accuracy/F1/balanced accuracy/ECE but worse NLL (+0.023598422) and Brier
(+0.013350321). Raw-byte and entropy individually still have lower ECE than fusion.
No unfavorable metrics suppressed; no weights adjusted in response.

## Artifacts
Run: data/derived/quality_march_evaluation/20261013_125046_074164/.
- predictions.npz: five methods' log probabilities and class predictions, IDs,
  class names, true labels; shape (5,4332,51) for log probabilities.
- comparison.csv: all seven metrics at full precision.
- report.json: per-class scores, confusion matrices, ECE bins, deltas, source
  hashes, frozen weights binding, input/policy provenance and limitations.
- checksums.json: artifact digests including report.json.

Report SHA-256: `6a4d9191deedabf859bd7594d3addbe56a4f0d8048f824cda02e56ec66c0d922`.
Weights SHA-256 unchanged:
`3fbaf039b608c4df195922b6fc29a2ff141b6fbdd1a4527a5a8c87f96eecc0ba`.

Entry point: scripts/evaluate_quality_march.py; numeric/alignment tests:
tests/test_quality_march.py, plus existing fusion and calibration tests.
No bootstrap or confidence intervals in this evaluation. March grouping/support
pilot is separately reported in docs/march_grouping_results.md.
