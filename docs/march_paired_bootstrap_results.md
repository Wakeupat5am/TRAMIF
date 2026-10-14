# March development — paired whole-group bootstrap

Date: 2026-10-13. Authorization: owner approval, DECISION_LOG.md D012.
Status: IMPLEMENTED, UNIT TESTED, INTEGRATION VERIFIED; bounded conditional
uncertainty experiment completed. This is not final holdout validation.

## Question and protocol

Does frozen quality-only fusion improve the fixed March results relative to uniform
fusion, allowing samples in an audited similarity component to move together?

- Same 4,332 retained March samples, 51 output classes, 47 supported classes.
- Same five frozen prediction sets, checkpoints, full-February temperatures and
  full-precision quality weights. No fitting or selection using March labels.
- Existing content-SHA equality OR TLSH distance <=10 connected components:
  4,438 graph vertices, 106 bridge-only excluded samples, 2,236 total components,
  2,234 active retained components. Bridge-only records never enter metric scoring.
- 2,000 whole-group draws with replacement, seed 42, NumPy default_rng/PCG64.
  Each draw selects G=2,234 groups; all methods share the same group multiplicities.
  Every retained member of a selected group is repeated by the same multiplicity.
- Metrics are sample means with the actual draw sample count as denominator.
  Brier is the sum of squared probability errors over all 51 classes, then the
  sample mean. NLL uses natural logarithms.
- Pointwise 95% percentile intervals use quantiles 0.025/0.975, linear interpolation.
  Original full-sample point differences are retained; bootstrap means do not
  replace them. No simultaneous or multiplicity-adjusted coverage is claimed.
- CI scope: Accuracy, NLL and Brier only. All seven original point metrics remain
  in report.json. Macro-F1, balanced accuracy and ECE receive no intervals here.
- The first 500 draws exactly match the prior support-only pilot because seed and
  generator match. This was verified and disclosed; no seed or draw was selected
  using performance differences. Every one of the 2,000 draws is retained.

## Primary comparison: quality-only minus uniform

Accuracy entries below are percentages; its difference and interval use percentage
points (pp). NLL/Brier retain their original units. Negative loss differences improve.

| Metric | Uniform point | Quality-only point | Q - Uniform point | Pointwise 95% CI |
| --- | ---: | ---: | ---: | --- |
| Accuracy | 70.960295% | 71.468144% | +0.507849 pp | [+0.119380, +0.947476] pp |
| NLL | 1.107323171 | 1.094385097 | -0.012938074 | [-0.018487435, -0.007689166] |
| Brier | 0.407840699 | 0.401633049 | -0.006207650 | [-0.008577202, -0.004074026] |

Each individual interval excludes zero in the favorable direction under this
resampling model. This is evidence of a modest conditional March improvement, not
a joint 95% statement across metrics, nor proof of improvement in future months.

## Secondary comparisons: quality-only minus each single view

These use exactly the same draws. Intervals are unadjusted and pointwise. Do not
claim simultaneous significant improvement over all baselines or across all metrics.

| Reference | Metric | Point difference | Pointwise 95% CI |
| --- | --- | ---: | --- |
| SBSMI | Accuracy | +1.200369 pp | [+0.273201, +2.165386] pp |
| SBSMI | NLL | +0.023598422 | [-0.025844785, +0.070621756] |
| SBSMI | Brier | +0.013350321 | [+0.001149644, +0.026544441] |
| Raw-byte | Accuracy | +4.155125 pp | [+0.734215, +6.850633] pp |
| Raw-byte | NLL | -0.208370493 | [-0.282161808, -0.143514867] |
| Raw-byte | Brier | -0.058757086 | [-0.078196221, -0.040402970] |
| Entropy | Accuracy | +8.194829 pp | [+5.804205, +10.961746] pp |
| Entropy | NLL | -0.214209333 | [-0.286975606, -0.149006162] |
| Entropy | Brier | -0.081537913 | [-0.100351060, -0.064023754] |

The unfavorable SBSMI comparison is retained: quality-only has worse Brier, with
its pointwise interval entirely positive. The SBSMI NLL interval includes zero.
Quality-only therefore does not dominate the strongest single view on every metric.

## Integrity, support and distribution diagnostics

- Alignment is an ID join against the saved grouping and canonical March artifact,
  followed by exact class/label checks. No unverified positional prediction mixing.
- Whole-group sums agree with explicit shared repeated sample indices on fixed
  draws and draws with extreme sample counts. Unit tests independently check this
  equivalence. No group is split; all methods receive the same sample multiplicities.
- All 2,000 draws have finite metrics. 1,182 (59.1%) miss at least one originally
  supported class. All are retained; no retry, filtering or changing class mapping.
- Actual draw sample count: min 3,512; 2.5th percentile 3,781.95; median 4,326;
  97.5th percentile 4,978.125; max 5,622. Interpolated count percentiles need not
  be integers; each actual draw count is an integer. Point estimates still use 4,332.
- Largest retained component: 166 samples (3.832%); 28 components have mixed labels.
  Size concentration N^2 / sum(n_g^2) = 174.514, far below 2,234 nominal groups.
  This is a descriptive concentration index, NOT a verified independent sample size.

Primary difference moments (empirical diagnostics, not validity tests):

| Difference | Skewness | Excess kurtosis |
| --- | ---: | ---: |
| Accuracy | 0.3583 | 0.2146 |
| NLL | -0.0656 | -0.2772 |
| Brier | -0.2433 | -0.0994 |

No predeclared moment review flag (absolute skewness >2 or excess kurtosis >10)
was triggered; histograms and tail quantiles are saved in diagnostics.json. These
findings do not prove normality, thin population tails or group independence.
Some Accuracy draws are negative despite its central 95% interval being positive.

## Sensitivity to groups

Descriptive leave-one-group-out calculations cover all 2,234 active groups.
No deletion reverses the sign of any PRIMARY point difference. This does not
establish that the confidence intervals would retain their signs after deletion:
no alternative-group bootstrap or altered estimator was run.

- Most influential Accuracy deletion: M000621 (23 samples), reducing the point
  improvement from 0.507849 pp to 0.394523 pp, a shift of -0.113325 pp.
- Most influential NLL deletion: M000092 (46 samples), changing -0.012938074 to
  -0.011552884, a shift of +0.001385189.
- Most influential Brier deletion: M000002 (131 samples), changing -0.006207650
  to -0.005449366, a shift of +0.000758284.
- M000002 multiplicity has correlation approximately -0.599 with the Brier
  difference. M000001 (50 samples) multiplicity has correlation approximately
  +0.522 with the Accuracy difference. These are descriptive associations.

Thus group composition affects the magnitude/distribution materially, even though
single-group removal does not reverse the primary point differences. The largest
group is not automatically the most influential. No group was removed from the
reported original estimates or confidence intervals.

## Reproduction and evidence

Command: `.\.venv\Scripts\python.exe -B -m scripts.run_march_bootstrap`.
CPU-only run exited successfully. No CNN or malware sample was executed.
11 new bootstrap tests PASS; complete suite **120/120 PASS (0.790s)**.
Tests cover determinism, whole-group integrity, paired sampling, metric agreement,
known percentile answers, missing-class retention, non-finite rejection, influence
arithmetic and saved NPZ reconstruction. Frozen source hashes checked before/after.

Run: `data/derived/march_paired_bootstrap/20261013_131843_476287/`.

| File | Purpose |
| --- | --- |
| config.json | Fixed scope, seed, generator, quantiles and conventions |
| grouping_manifest.csv | Exact copy of audited memberships including bridge-only flags |
| sample_group_alignment.csv | Canonical retained IDs, group IDs/indices and labels |
| draws_and_metrics.npz | All group multiplicities, group sums/counts, per-draw metrics/deltas and class support |
| replicates.csv | 2,000 rows with actual N, missing classes, five-method metrics and paired differences |
| comparisons.csv | Original differences and primary/secondary percentile intervals |
| group_influence.csv | All leave-one-group-out descriptive results |
| diagnostics.json | Distribution moments, histograms, counts and influence summaries |
| report.json | Original point metrics, comparisons, complete provenance and limitations |
| checksums.json | Persisted artifact hashes |

Immutable source runs:
- weights: `data/derived/quality_weight_freeze/20261013_124031_275438/weights.json`
  SHA256 `3fbaf039b608c4df195922b6fc29a2ff141b6fbdd1a4527a5a8c87f96eecc0ba`.
- March predictions: `data/derived/quality_march_evaluation/20261013_125046_074164/`;
  report SHA256 `6a4d9191deedabf859bd7594d3addbe56a4f0d8048f824cda02e56ec66c0d922`.
- March grouping: `data/derived/march_group_feasibility/20261013_125459_149338/`;
  report SHA256 `70cbcac4d34819291593a719599a11ff3b3637ab0b067db3060b539c6a3f523d`.
- Bootstrap report SHA256
  `6b89d6d202ba61e7dd697c3e28656c4c67b505eb94880d5a05b83ff77e759962`.
- Draw NPZ SHA256
  `275bfe526ee99fc89ce52657831a3d52d2c6d079e190c2910ae5c3d2981e4d0b`.

Git does not back up ignored data/ artifacts. No commit or push was performed.

## Interpretation and remaining research work

Uncertainty is conditional on the existing checkpoints, fitted temperatures,
estimated quality weights, selected March sample set and audited grouping. It
does not propagate training-seed, calibration-estimation, weight-estimation or
between-month variability. TLSH connected components are an operational dependence
control and do not establish independence between components or absence of every
near-duplicate. Group-size imbalance limits interpretation. No alarming numerical
pathology was found that required changing the approved estimator; its assumptions
remain limitations, not empirically proven facts.

March was already observed during development. These intervals support a bounded
development finding, not final unseen future generalization or full TRAMIF reliability.
No Future Test data was accessed. No quality weights, graph, primary selection or
class mapping changed. Drift and uncertainty coefficients remain zero.

Before an authorized Future Test evaluation:
1. Review and freeze the exact primary research endpoints, comparisons, multiplicity
   interpretation and uncertainty scope. Keep class-sensitive point metrics visible;
   this batch does not solve their missing-class interval problem.
2. Decide whether the final claim is explicitly single-seed quality-only or requires
   additional approved training-seed experiments and duplicate-policy sensitivity.
   This bootstrap substitutes for neither; threshold 10 here is a resampling group
   definition, not a change to the primary dataset policy.
3. Freeze the future evaluation specification: family eligibility, chronology,
   cross-period duplicate handling, representation settings, checkpoint/temperature/
   weight hashes, monthly metrics, and rules against adapting to future outcomes.
4. Preserve the complete ignored artifact chain separately from Git and verify it
   before evaluation. Obtain explicit authorization before opening Future Test.
5. Treat any drift/uncertainty-weighted extension as a new methodology proposal,
   requiring honest historical validation episodes and approval. It is not needed
   to relabel this completed quality-only experiment as full reliability.

Batch stopped after reporting. No follow-on estimator, training or evaluation started.
