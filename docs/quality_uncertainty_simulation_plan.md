# G5 — Finite synthetic simulation plan

Date: 2026-10-16. **PROPOSED; not executed.** Scope: estimator diagnostics only.
Source contract: RESEARCH_PROTOCOL_V1.md §7 and historical_quality_uncertainty_design.md.
Implementation already unit tested: scripts/quality_uncertainty.py; evidence in
quality_uncertainty_synthetic_results.md. This plan does not change that estimator.

## 1. Question and target

Does conditional group-weight SD U approximate the sampling spread of the same
sample-level, equal-family loss L in controlled synthetic settings? Where does
it underestimate uncertainty when operational groups fail to capture dependence?

Do not confuse two loops:
- Outer loop: generate independent datasets with a fixed layout; their original
  L values estimate repeated-dataset variability.
- Inner loop: apply the approved 2,000 positive group-weight draws to one dataset;
  their SD is U. Never divide U by sqrt(2,000).

The target is conditional on fixed family counts and group layout. No CNN,
temperature fitting, uncertain labels, cohort selection or time shift is simulated.
Thus even successful diagnostics cannot establish future reliability or coverage
on real malware. No synthetic hypothesis is presented as verified TRAMIF behavior.

## 2. Fixed scenarios (no search for a passing design)

Three families c=0,1,2; three anonymous views v=0,1,2. Do not associate these
synthetic view strengths with SBSMI/raw-byte/entropy. Every family remains in S.
Groups below are disjoint across families unless scenario7 explicitly shares IDs.

| ID | Layout per family | Within-group latent variance alpha | Purpose |
|---|---|---:|---|
|1|20 groups ×4 samples|0.5|Independent equal groups, reference scenario|
|2|20 groups: ten×1, six×2, three×8, one×40|0.5|Unequal independent groups, sample-level target preserved|
|3|Same layout as1|0.9|Strong dependence correctly captured by group IDs|
|4|Same layout as1; blocks of4 operational groups share a hidden effect|0.5|Missed dependence across nominal groups|
|5|Families0/1 as1; family2 has5 groups ×4 samples|0.5|Rare family at the five-group support boundary|
|6|Each family: one group×80 plus19 groups×4|0.5|Dominant group, no removal from main estimate|
|7|20 joint groups; each contains4 samples from EACH family|0.5|Multi-label groups and cross-family paired weights|

Layouts are deterministic, not chosen using outcomes. Scenario4 hidden blocks
stay within family; group indices0–3,4–7,... share one hidden effect. The estimator
receives only the original operational group IDs, deliberately not hidden blocks.
Scenario7 uses60 within-family memberships but only20 unique group IDs. No
stratified redraw per class is permitted.

## 3. Exact nonnegative loss generator

Generate independent standard-normal latent variables; correlation across views
is constructed explicitly rather than chosen from outputs. For any latent unit u,
let vector V_u,v = sqrt(0.5)*Z_u,shared + sqrt(0.5)*Z_u,v, where all Z are independent.
Each coordinate has unit variance; different views share half its variance.

Draw group vectors G_g and sample vectors E_i independently. Except scenario4,
set C_g=G_g. In scenario4 draw hidden-block vectors H_b independently and set:

    C_g = sqrt(0.75)*H_block(g) + sqrt(0.25)*G_g.
    X_i,v = sqrt(alpha)*C_g(i),v + sqrt(1-alpha)*E_i,v.
    ell_i,v = mu_c,v + s_v * X_i,v^2.
    mu_c,v = 0.3 + 0.2*c + 0.1*v; s = (0.4,0.7,1.0).

In scenario7 the same G_g is shared by samples in all three families. The sample
vectors remain independent. Every X has marginal variance1, so known expected
class loss is mu_c,v+s_v, and expected balanced L is mean_c(mu_c,v)+s_v.
All losses are nonnegative, hence mathematically admissible NLL values; no
synthetic logits/calibration are needed. Do not claim this generator models
actual malware loss distributions or every heavy-tail mechanism.

Implementation must fix array draw order: group shared normals(G), group-specific
normals(G,3), sample shared normals(N), sample-specific normals(N,3), then hidden
shared/block-specific normals only for scenario4. Samples ordered family, group,
member; scenario7 also uses family, joint-group, member. Persist layout hashes.

## 4. Finite budget and random streams

- Outer independent datasets R=300 per scenario, exactly7 scenarios:2,100 datasets.
- Dataset r (zero-based) uses Generator(PCG64(SeedSequence([20261016,scenario_id,r]))).
  No global mutable generator or seed retries. Save generator/version metadata.
- Each dataset uses the estimator's canonical sorted active group IDs and2,000
  PCG64 seed42 exponential draws, paired across all views/classes. Reuse the same
  draw matrix for identical group layouts. This is common random numbers across
  datasets, not a claim that inner Monte Carlo errors are independent.
- For datasets r=0..29 ONLY in every scenario, repeat using seed43 and report the
  predefined V1 trigger abs(U42-U43)>max(0.005nat,0.10*max(U42,U43)). Never select a
  preferred seed.210 extra fits, total2,310 estimator calls; no other seed search.
- CPU only, sequential, one numerical-library thread. No GPU or real-data paths.
  Store each distinct draw matrix once. Maximum group count60; each2,000×60
  float64 matrix uses960,000 bytes. Runtime UNKNOWN until measured; no guaranteed
  wall-clock estimate. Interruption produces PARTIAL, not a reduced-R result.

## 5. Report all results, including failure modes

For every scenario/view report:

    S_emp = SD_r(L_r), ddof=1
    U_mean = mean_r(U_r)
    U_rms = sqrt(mean_r(U_r^2))
    ratio_SD = U_mean/S_emp
    ratio_variance = mean_r(U_r^2)/(S_emp^2)

Report known expected L, empirical mean/bias, S_emp, U distribution (min, median,
5th/95th percentiles, max), both ratios, seed43 discrepancies/trigger counts and
all nonfinite/undefined cases. S_emp=0 means ratio undefined; do not replace with
epsilon. With only300 datasets, S_emp is itself estimated, not ground truth.

For simulation precision, report mean-L Monte Carlo SE=S_emp/sqrt(300), and
delete-one-outer-dataset jackknife SE of S_emp and both ratios. This is a numeric
precision diagnostic over independent generated datasets, not a confidence
interval for actual TRAMIF or proof of estimator consistency. No percentile
interval coverage claim or p-value is part of this batch.

For r=0..29 of scenarios5/6/7, record leave-one-operational-group-out changes in
original L only. Preserve S. If a class vanishes, mark undefined; never shrink S.
Do not recompute/select fusion coefficients or call loss influence a substitute
for V1's still-pending induced-fusion-weight diagnostic.

## 6. Acceptance and stop rules, fixed before results

Engineering completion requires all2,100 outer datasets and210 seed43 diagnostics,
exact IDs/layouts, shared draws, independent reference agreement on a small fixed
fixture from each scenario, deterministic replay, correct known expectations in
the generator algebra, and complete artifacts. Numerical comparison tolerance:
rtol1e-12, atol1e-12 for float64 reference calculations; no output clipping to pass.

Any nonfinite losses/draws/denominators/metrics, missing family, identity mismatch
or reference discrepancy stops execution, preserves failure details and requires
review. A finite unfavorable ratio is NOT a computational error; finish reporting
all predeclared scenarios without retries or estimator changes.

PROPOSED scientific review triggers: ratio_SD outside[0.8,1.25] in any scenario,
or any V1 seed43 trigger, mandates explicit owner review before empirical U is
used in weights. These bounds are transparent engineering tolerances, not a
statistical theorem or automatic pass threshold. Report jackknife precision so
boundary crossings are not overstated. A violation in the deliberately dependent
scenario documents a limit; do not redesign that scenario until it passes.

Even all ratios inside bounds yields only COMPATIBLE_WITH_TESTED_SCENARIOS, not
RESEARCH_VALIDATED. Dependence assumptions on TRAMIF remain unresolved. Persistent
underestimation in independent reference cases is especially concerning; stop
progression to real U pending review. Do not silently increase replicates, change
groups, use seed43 instead, or substitute predictive entropy.

## 7. Implementation and artifacts proposed for the next approved batch

1. Separate synthetic-only runner importing unchanged quality_uncertainty.py;
   test generator dimensions/layout/covariance construction, mixed-label IDs,
   exact stream keys, reference formulas and jackknife calculations. No data loader.
2. Save config, plan/source hashes and deterministic layout before execution.
   Run fixed grid; preserve generated synthetic losses, point L, U, replicate
   summaries, distinct draw matrices, seed43 diagnostics and influence results.
   Record all source/config/output hashes, NumPy/BLAS versions, time and failures
   in a unique data/derived/quality_uncertainty_simulation/ directory.
3. Independent readback of hashes and summary recomputation. Publish all21
   scenario/view rows and limitations; update progress/decisions; stop for review.

No existing model/manifest/protocol changes; no historical losses, CNN, C/Future,
Q/U on real data, fusion weights, commit or push. Approval needed for this exact
simulation budget and proposed review tolerances before implementation/execution.
