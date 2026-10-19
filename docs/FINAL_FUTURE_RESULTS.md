# D062 — Final Future Results

2026-10-18. **FINAL_FUTURE_SCORING_VERIFIED / Engineering PASS.**

Owner explicitly authorized final scoring after D060 freeze and D061 input verification.
All five registered seeds and seven methods evaluated without retraining, inference,
calibration refits, weight changes, beta retuning or seed selection. This is the final
fixed April–September 2020 horizon of the approved reduced Q/Q+D study.

## Main finding

**The predeclared practical target for adding descriptive Drift is NOT MET.**
Q+D minus New Q mean-month Macro-F1 = +0.0000789850145024 proportion
(+0.0078985 percentage points); worst-month = +0.000171204140653
(+0.0171204 percentage points). Required mean gain was >=0.02 proportion
(+2 percentage points), together with worst gain >0. Positive worst alone is insufficient.
Mean gain is negative in seeds43/44; all seeds are retained. No CI, p-value or
statistical significance claim. Engineering PASS is not confirmation of the hypothesis.

## Frozen contracts and verified coverage

- D060 bundle SHA: `f52f70c1c23dc815258c71f7f19c578ef7592b76c8e84b9df1361dab8ae5c1b3`.
- D061 receipt SHA: `116f994f5b3bf4372de248fe34b855e01c87f037955795a8156fa34d433ac771`.
- K51 outputs; per-seed February temperatures, March B/S25 Q, descriptive d=MMD2/2.
- beta=1, gamma=0; U remains diagnostic-only. Q/QD and learned-static weights frozen.
- Metrics use exact frozen definitions, full K51 probabilities and fixed class order.
- Each seed: mean of six monthly supported Macro-F1; worst is minimum of six.
  Then average five seed outcomes. No pooled F1 substitution or cross-seed ensemble.

| Month | retention0 N / classes | retention10 N / classes | retention20 N / classes |
|---|---:|---:|---:|
| 2020-04 | 4032 / 47 | 1962 / 45 | 1597 / 43 |
| 2020-05 | 4311 / 43 | 2318 / 40 | 1909 / 37 |
| 2020-06 | 3483 / 41 | 2414 / 37 | 2221 / 36 |
| 2020-07 | 3058 / 38 | 1990 / 34 | 1807 / 33 |
| 2020-08 | 2686 / 37 | 1493 / 33 | 1324 / 30 |
| 2020-09 | 2793 / 41 | 1625 / 36 | 1423 / 29 |

Class support is the true-supported set of each reported retained month, common to
all seeds/arms in that condition; no prediction-based selection and no S25 evaluation filter.
Sensitivity support/counts can differ; preserve these changes when interpreting scores.

## Primary retention0 outcomes

Macro-F1 in percent. SD is sample SD over five seeds (ddof=1), not standard error or CI.
All per-seed outcomes and ranges are preserved in primary_outcomes.csv.

| Method | Mean-month | SD | Min–max | Worst-month | SD | Min–max |
|---|---:|---:|---:|---:|---:|---:|
| sbsmi | 64.272882 | 0.490768 | 63.571774–64.710200 | 57.456785 | 0.985760 | 56.114772–58.454233 |
| raw-byte | 54.674800 | 1.439157 | 52.680206–56.475582 | 47.956444 | 1.361237 | 45.725449–49.213138 |
| entropy | 52.291846 | 0.847351 | 50.909131–53.125704 | 45.263350 | 1.684797 | 43.363885–46.939572 |
| uniform | 64.377877 | 0.803300 | 63.368657–65.291392 | 56.235290 | 0.721247 | 55.486538–57.098649 |
| new-quality-only | 64.749422 | 0.818242 | 63.456118–65.430422 | 56.630088 | 0.566625 | 56.028378–57.431215 |
| quality-plus-drift | 64.757320 | 0.827705 | 63.456457–65.467027 | 56.647208 | 0.553227 | 56.089631–57.431215 |
| learned-static | 64.936615 | 0.862132 | 63.997052–66.052821 | 57.518346 | 1.137991 | 55.891882–58.506208 |

Learned-static has the highest aggregate Mean and Worst among the seven listed
methods on primary retention0; this descriptive ranking is not best-seed selection.
Q+D has higher Mean than SBSMI but lower Worst, and is below learned-static on both.

## Monthly supported Macro-F1

Five-seed means (%), not alternative primary aggregates.

| Method | April | May | June | July | August | September |
|---|---:|---:|---:|---:|---:|---:|
| sbsmi | 69.4282 | 70.9767 | 66.7700 | 62.4418 | 57.5307 | 58.4900 |
| raw-byte | 60.9347 | 58.8381 | 59.9850 | 50.0630 | 50.2137 | 48.0143 |
| entropy | 59.6501 | 56.7159 | 55.4124 | 47.8584 | 48.3082 | 45.8062 |
| uniform | 69.5482 | 70.1291 | 71.1160 | 62.0276 | 56.6839 | 56.7624 |
| new-quality-only | 69.8955 | 70.9291 | 71.0790 | 62.6747 | 56.9662 | 56.9520 |
| quality-plus-drift | 69.8951 | 70.9495 | 71.0656 | 62.6760 | 56.9724 | 56.9853 |
| learned-static | 70.2489 | 72.4073 | 67.9212 | 62.6927 | 57.5959 | 58.7537 |

## All eight fixed contrasts

Paired left-minus-right differences in percentage points; both positive and negative
results included. SD/range describe paired seed deltas. No multiplicity-adjusted
inference is claimed because no hypothesis tests/intervals are performed.

| Priority | Left − right | Outcome | Mean delta | SD | Min–max |
|---|---|---|---:|---:|---:|
| primary | quality-plus-drift − new-quality-only | mean_monthly_macro_f1 | +0.0078985 | 0.0170417 | -0.0039723–+0.0366046 |
| primary | quality-plus-drift − new-quality-only | worst_month_macro_f1 | +0.0171204 | 0.0260502 | +0.0000000–+0.0612537 |
| secondary | new-quality-only − uniform | mean_monthly_macro_f1 | +0.3715445 | 0.2782649 | +0.0874612–+0.7093113 |
| secondary | new-quality-only − uniform | worst_month_macro_f1 | +0.3947980 | 0.2222332 | +0.1886317–+0.7283436 |
| secondary | quality-plus-drift − uniform | mean_monthly_macro_f1 | +0.3794430 | 0.2683244 | +0.0877998–+0.7053390 |
| secondary | quality-plus-drift − uniform | worst_month_macro_f1 | +0.4119184 | 0.2264648 | +0.2091397–+0.7301223 |
| secondary | quality-plus-drift − learned-static | mean_monthly_macro_f1 | -0.1792951 | 1.1626644 | -1.5991708–+1.3687366 |
| secondary | quality-plus-drift − learned-static | worst_month_macro_f1 | -0.8711376 | 1.0867132 | -2.2255666–+0.6423520 |

Q-only exceeds Uniform on Mean (+0.3715445 pp) and Worst (+0.3947980 pp)
for this fixed cohort; all five paired seed differences are positive. This does
not establish population significance. Q+D improvements over Uniform largely
track Q rather than an appreciable incremental benefit of D.
Q+D-versus-three-single-view descriptive deltas are fully saved separately,
outside the fixed eight-contrast reporting family.

## Retention10/20 sensitivity

Same frozen predictions/weights, chronological nested sample subsets; no retraining.
All methods shown. Mean/Worst are five-seed aggregate percentages.

| Threshold | N | Method | Mean-month | Worst-month |
|---|---:|---|---:|---:|
| 0 | 20363 | sbsmi | 64.272882 | 57.456785 |
| 0 | 20363 | raw-byte | 54.674800 | 47.956444 |
| 0 | 20363 | entropy | 52.291846 | 45.263350 |
| 0 | 20363 | uniform | 64.377877 | 56.235290 |
| 0 | 20363 | new-quality-only | 64.749422 | 56.630088 |
| 0 | 20363 | quality-plus-drift | 64.757320 | 56.647208 |
| 0 | 20363 | learned-static | 64.936615 | 57.518346 |
| 10 | 11802 | sbsmi | 59.775561 | 52.146622 |
| 10 | 11802 | raw-byte | 48.191066 | 41.241960 |
| 10 | 11802 | entropy | 46.812649 | 40.624573 |
| 10 | 11802 | uniform | 59.337126 | 48.570238 |
| 10 | 11802 | new-quality-only | 60.019026 | 49.918293 |
| 10 | 11802 | quality-plus-drift | 60.029187 | 49.918575 |
| 10 | 11802 | learned-static | 60.349007 | 52.026473 |
| 20 | 10281 | sbsmi | 53.940706 | 46.763413 |
| 20 | 10281 | raw-byte | 43.786575 | 37.633536 |
| 20 | 10281 | entropy | 41.710741 | 35.688254 |
| 20 | 10281 | uniform | 54.364612 | 44.985534 |
| 20 | 10281 | new-quality-only | 55.050747 | 46.272658 |
| 20 | 10281 | quality-plus-drift | 55.062566 | 46.278408 |
| 20 | 10281 | learned-static | 54.756849 | 47.105737 |

Primary Q+D−Q sensitivity deltas (percentage points):

| Threshold | Mean-month delta | Worst-month delta |
|---|---:|---:|
| 0 | +0.0078985 | +0.0171204 |
| 10 | +0.0101610 | +0.0002826 |
| 20 | +0.0118188 | +0.0057498 |

No threshold changes the practical-target conclusion. Absolute F1 falls under
stricter retention, but sample/class composition also changes; this is not a causal
estimate of duplication bias. At20, Q+D Mean exceeds learned-static Mean, while
Worst remains lower. Do not substitute this favorable sensitivity for primary0.

## Required file-size descriptive analysis

Cutoffs fitted only on 18,061 TRAIN files:86414/254672/721840 bytes.
Bins are right-closed; no Future fitting or adjustment. Reported below are
five-seed means of six monthly within-bin supported Macro-F1 scores (%).
These diagnostics do not replace primary outcomes. Original whole-month-support
F1, fixed51 metrics, monthly counts/classes and every seed are in file_size_metrics.csv.

| TRAIN-size bin | Total N | SBSMI | Raw-byte | Entropy | Uniform | Q | Q+D | Learned-static |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2877 | 61.65998 | 51.78956 | 55.53326 | 59.79666 | 60.08433 | 60.12873 | 62.23247 |
| 2 | 3288 | 61.64228 | 55.01072 | 57.26218 | 63.59628 | 64.08752 | 64.09364 | 62.49188 |
| 3 | 4264 | 66.47821 | 54.06858 | 51.83027 | 66.32970 | 67.18942 | 67.15713 | 66.70034 |
| 4 | 9934 | 54.43372 | 45.74413 | 41.22271 | 55.84318 | 55.86776 | 55.87088 | 54.89499 |

9,934/20,363 samples (48.8%) exceed the TRAIN upper quartile cutoff. Q+D−Q
within-bin Mean differences vary in sign (negative in bin3); the gain is not
uniform across file sizes. Families and size strata overlap unevenly; stratification
does not remove packing, capacity or class-composition confounding.

## Secondary six-month pooled metrics

Means over five seeds on primary0. This is a sample-pooled secondary horizon,
not the primary mean-month or worst-month estimand. All monthly metrics and
pooled1/3/6-month horizons for all thresholds are retained in CSV/JSON.

| Method | Accuracy % | Fixed51 F1 % | Balanced accuracy % | NLL | Brier | ECE15 |
|---|---:|---:|---:|---:|---:|---:|
| sbsmi | 66.0954 | 60.2206 | 65.4440 | 1.218055 | 0.443934 | 0.084000 |
| raw-byte | 59.2869 | 50.3901 | 57.1045 | 1.500416 | 0.533733 | 0.048130 |
| entropy | 53.1935 | 49.2011 | 55.5463 | 1.634026 | 0.578556 | 0.079439 |
| uniform | 67.0599 | 60.0650 | 66.6175 | 1.227758 | 0.461396 | 0.087309 |
| new-quality-only | 67.0628 | 60.5309 | 66.8438 | 1.204266 | 0.451125 | 0.072154 |
| quality-plus-drift | 67.0667 | 60.5385 | 66.8502 | 1.203905 | 0.450964 | 0.071970 |
| learned-static | 66.3291 | 61.1120 | 66.2479 | 1.177851 | 0.437350 | 0.083605 |

Learned-static improves pooled NLL/Brier relative to Q+D but has lower pooled
accuracy; Q+D does not dominate every metric or baseline. No unfavorable metric omitted.

## Verification, artifacts and reproduction

- 37/37 targeted synthetic/contract tests PASS (15.864s).
- Independent raw-logit calibration/mixture readback for all35 seed-method arrays PASS.
- All630 monthly metric sets,2520 size cells and315 pooled metric sets independently
  recomputed from saved predictions/log probabilities, including per-class and ECE bins.
- All66 outcome/contrast aggregates across three thresholds independently verified.
- CSV readback:630 monthly,2520 size,315 pooled,42 outcome,24 fixed-contrast and18
  descriptive single-view rows verified against full-precision JSON; no duplicate keys.
- All47 hashed scoring artifacts verified; source, D060/D061 binding and protocol hashes unchanged.
- CPU scoring plus numerical verification took 56.932 seconds; no CNN inference or optimizer.

Run: `data/derived/final_future_scoring_v1/20261018_141835_165405/`.
Completion receipt SHA-256: `53c3c885851136549eacb8756ccd3c44debec3fea5e24bc722efb6a730840bc6`.
Independent verification: `independent_verification.json`.
Saved outputs: `results.json` (all confusion matrices,per-class metrics,ECE bins),
`log_probabilities.npy`, `predictions.npz`, canonical scoring/retention/support manifests,
six CSV tables, full source/config snapshots and hashes.
Supplementary CSV/test readback: `data/derived/final_future_scoring_readback/20261018_142254_109840/`.

Synthetic checks:
```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_final_future_scoring tests.test_d060_integration tests.test_final_point_statistics -q
```
A deliberate CPU reproduction creates a fresh run; do not rerun to choose favorable outputs:
```powershell
.\.venv\Scripts\python.exe -B -m scripts.score_final_future --owner-approved-final-scoring
```

## Research interpretation and remaining limits

RQ1 remains descriptive view–encoder drift measurement; no standardized significance
or intrinsic representation claim. RQ2 historical predictive screening and negative
E2 are preserved. RQ3 final frozen evaluation is now observed: quality weighting
has a modest descriptive improvement over uniform, but added D fails its practical
target and does not establish useful extra temporal reliability.

Five seeds share training/development data and are not independent temporal experiments.
Six months form a fixed horizon; no general future-year, causal, SOTA or significance claim.
U synthetic negative findings remain diagnostic-only; QU/QDU are not evaluated arms.
No full original Q+D+U framework validation is claimed. Capacity differs: SBSMI140467
parameters versus59059 per compact branch;three-view system258585. Existing CPU
synthetic latency is not end-to-end preprocessing/inference latency. Capacity-matched
ensembles,broad baselines,packing adjustment,family/block inference and statistical
coverage remain unperformed/deferred under approved reduced scope.

Input limits remain: original ZIP CRC unavailable, current disarmed bytes pinned;
benign/out-of-K families are outside known-family scoring. New D061/D062 artifacts
have not been verified off-device. Paper polishing/figures and independent backup
are deliverable work, not permission to retune after seeing Future.

**D062 is complete; the primary practical target is not achieved.** Preserve the
weak/negative incremental-D result and all saved arms/seeds/months. Further model
selection on these months cannot be called untouched-holdout confirmation.
