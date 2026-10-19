# G5 U synthetic simulation — results

2026-10-16. **Engineering PASS. U_NOT_APPROVED_FOR_FUSION.**
All approved scenarios executed unchanged. No estimator modification, rerun to
improve results, real data, CNN inference/training, C/Future access or fusion.

## Evidence and budget

- 7 scenarios ×300 independent synthetic datasets =2,100 primary fits.
- Each fit uses2,000 exponential group-weight replicates, PCG64 seed42.
- 210 additional seed43 diagnostics, exactly first30 datasets per scenario.
- 9 targeted tests PASS (0.173s): layouts, mixed-family IDs, exact stream and
  hidden-effect construction, all-scenario independent numerical reference,
  expectation algebra, jackknife/summary calculations, fail-closed guards.
- Runner elapsed4.677s on this environment, not a runtime guarantee.
- Readback verified36 output files' hashes, all21 rows, saved losses/point means,
  replicate SDs, ratios, jackknife ratio precision and secondary-seed counts.
- Existing estimator hash matches D039 synthetic receipt: unchanged.

Run: `data/derived/quality_uncertainty_simulation/20261016_165327_417140/`.
Report SHA256: `7278f005d0cdd006c136b5f5b9bf1c778172456d6d2a42a6ea425fa8e5561130`.
Independent readback:
`data/derived/quality_uncertainty_simulation/20261016_165327_417140_verification/receipt.json`.

The run stores config/plan/source hashes, NumPy/BLAS configuration, layouts, exact
group draws, all generated losses, primary and seed43 replicate losses, point L/U,
summary metrics and leave-group-out diagnostics. No malformed replicate was removed.
NumPy emitted an optional `pyyaml` formatting warning while recording build config;
it did not affect computation. No dependency was installed.

## All 21 scenario/view results

Views0/1/2 are synthetic, NOT the three named TRAMIF models. S_emp is between-
dataset SD of original L. Ratio=mean(U)/S_emp; below1 means average underestimation.
JK-SE is delete-one-dataset jackknife SE of this ratio, a simulation-precision
diagnostic, not an interval for real data or a formal significance test.

| Scenario | View | S_emp | Mean U | Ratio | JK-SE ratio | Review trigger |
|---|---:|---:|---:|---:|---:|---|
|1 Equal independent groups|0|.048359|.045215|.934997|.039289|No|
|1|1|.084035|.079583|.947026|.040767|No|
|1|2|.122503|.113294|.924830|.051233|No|
|2 Unequal independent groups|0|.078134|.050247|.643084|.024245|Yes|
|2|1|.150870|.087829|.582150|.029159|Yes|
|2|2|.216417|.129702|.599316|.028125|Yes|
|3 Strong within-group dependence|0|.069113|.064809|.937727|.038326|No|
|3|1|.116257|.110254|.948363|.038762|No|
|3|2|.159947|.156951|.981269|.039258|No|
|4 Hidden cross-group dependence|0|.067098|.044141|.657857|.026909|Yes|
|4|1|.116459|.074570|.640311|.026323|Yes|
|4|2|.176387|.113918|.645841|.024247|Yes|
|5 Rare family, five groups|0|.067183|.053255|.792695|.037121|Yes|
|5|1|.135021|.097638|.723130|.039298|Yes|
|5|2|.176384|.140497|.796542|.037742|Yes|
|6 Dominant independent group|0|.085682|.044172|.515531|.026006|Yes|
|6|1|.133995|.076812|.573243|.021419|Yes|
|6|2|.211232|.108787|.515009|.025584|Yes|
|7 Multi-family groups|0|.072286|.063535|.878930|.037849|No|
|7|1|.129109|.108160|.837742|.035414|No|
|7|2|.182573|.163240|.894111|.040034|No|

All630 view-level seed43 comparisons (210 datasets ×3 views) stay below the fixed
max(0.005nat,10% larger SD) trigger. No seed was chosen from these results.
Full expected/empirical means, biases, mean-L MCSE, U quantiles, RMS U, variance
ratios and all three jackknife SEs are retained in report.json and case summaries.

## Interpretation, including negative evidence

12/21 ratio cells violate the predeclared[0.8,1.25] review range. All mean-U ratios
are below1; none shows average overestimation by this metric. Scenario3/view2 has
variance ratio1.013 despite mean-U ratio.981: RMS and mean SD answer different
questions, so both are retained rather than presenting only favorable values.

Scenarios1/3 are closest to empirical variability (roughly2–8% underestimation).
Scenario7 remains inside review bounds but underestimates about11–16%; passing
that engineering range is not statistical calibration. Scenario5 views0/2 are
near the.8 boundary relative to jackknife precision; keep their triggered status
without claiming a sharp distinction from.8. Its view1 shows greater deficit.

Scenarios2/4/6 show substantial underestimation, approximately34–48.5%; discrepancies
are large relative to reported simulation SEs. This is a descriptive assessment,
not a multiplicity-adjusted significance statement. Crucially, scenarios2/6 have
independent operational groups: the limitation cannot be attributed only to the
deliberately missed dependence in scenario4. Few heavily weighted groups and the
fixed-layout target can limit how well conditional perturbations approximate
repeated-dataset variation. These simulations do not isolate every causal factor.

Leave-one-group-out diagnostics on the first30 datasets of scenarios5/6/7 produce
no missing-family/undefined cases. Maximum absolute original-loss change is
0.625415nat. This is influence evidence, not a new acceptance threshold and not
V1's pending induced-fusion-weight check. Groups were not removed from estimates.

## Decision and next step

**STATISTICAL_LIMITATION, not IMPLEMENTATION_FAILURE.** All scenarios completed;
the estimator was not tuned or repaired to obtain a favorable ratio. Stable seed
results show low observed inner-randomness sensitivity for these fixtures, not
correct uncertainty estimation. Outer precision is conditional on fixed layouts
and common inner draws, with only300 datasets per scenario.

The simulation does not establish coverage, group independence on TRAMIF, training/
calibration uncertainty, temporal variability or prediction of future model error.
U remains **U_NOT_APPROVED_FOR_FUSION**; no QU/QDU coefficient decision follows.

There is sufficient engineering evidence to prepare a read-only historical input
readiness audit, but NOT to approve empirical U execution or its use in weights.
Recommended next task: verify availability/provenance of complete E1/E2 A/B logits
and A-only temperatures plus registered Final logits/temperatures, with frozen
support and A/B-only group bindings. Use all retained B in S, not capped Drift
samples. Report missing prefix inputs; do not infer absent artifacts from missing
registry entries. If missing, propose bounded A/B prediction/calibration preparation.

Owner review should decide whether to retain U as a diagnostic only for now or
authorize a justified estimator/methodology change. Do not open another synthetic
search to obtain PASS. Q/calibration input preparation can proceed independently
after scoped authorization; empirical U and fusion eligibility remain separate.
