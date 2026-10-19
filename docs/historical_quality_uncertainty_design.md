# Historical Quality uncertainty — proposed G5 contract

Date: 2026-10-16. Status: DESIGN REVIEW ONLY; not estimator execution or approval
for new Q/U/weights. V1 remains unchanged. D038 records this bounded design task.

## 1. Why this measurement is needed

Quality L is an average classification loss. Two views can have similar L while
one estimate depends much more heavily on a few historical groups. U describes
that sensitivity conditional on the observed history. It does not establish that
the view with larger U will predict worse next month. Penalizing U remains a
research hypothesis, requiring later held-forward comparisons against New Q.

## 2. Evidence and current input readiness

| Item | Status and direct evidence |
|---|---|
| Q/U definition, chronology | VERIFIED specification: RESEARCH_PROTOCOL_V1.md §§4,5,7; implementation still PROPOSED |
| E1/E2 operational A/B groups and retained IDs | Existing G2 evidence: historical_episode_feasibility_report.md; immutable run historical_episode_feasibility/20261014_233013_930021, E1/E2 A_keep.csv, B_keep.csv, AB_10_members.csv, class_mapping.json, support.csv. No graph rebuild needed |
| Final joint groups/support | balanced_phase1_results.md: 25 supported families, 4,064 retained March samples in S*, out of4,332 total; joint graph excludes C/Future. These are recorded audit counts, not newly recomputed here |
| Final logits/calibration reuse path | evaluate_quality_march.py main binds baseline_artifacts.json, March predictions and February temperature hashes; calibrate_filtered_view.py validates provenance. Current bytes require a bounded input verification before execution |
| Prefix logits and A-only temperatures | UNKNOWN: not verified/registered by this review. Existing frozen_encoder_extraction.py persists feature matrices, not a complete episode calibration/prediction bundle |
| Existing Quality implementation | quality_fusion.py estimate_quality is February OOF-specific and also produces fusion weights; do not call it unchanged to implement this new estimator |
| Stable log probabilities / temperature solver | uniform_fusion.py calibrated_log_probabilities; calibrate_sbsmi.py fit_temperature, bounds[0.05,20]. Reuse numerical contracts, not historical dataset assumptions |
| U implementation and empirical validity | Not established by this review. No U values, bootstrap runs or model predictions computed |

This review inspected relevant source/contracts, not every derived directory.
UNKNOWN prefix prediction availability means unverified, not proof no file exists.
No outcome prediction files were opened.

## 3. Frozen data contract proposed for approval

| Case | K | A: fit T | B: Q/U | Frozen support |
|---|---:|---|---|---:|
| E1 |32| November2019 | December2019 |17|
| E2 |42| December2019 | January2020 |25|
| Final |51| February2020 | March2020 |25|

Use all primary-retained B samples belonging to frozen S for class-balanced
Q/U. Do NOT reuse cap128/cap64 Drift subsets or group-balanced MMD sample masses.
Q/U keeps the sample-level empirical target within each class. Record standard
NLL across all retained B and across S separately. Calibration uses all retained
A with its episode mapping and standard NLL. No B fitting or temperature refit
inside bootstrap. Final seed42 preserves the registered February temperatures.
Probabilities retain K outputs; never renormalize over S.

Bind immutable group memberships from the A/B-only graph, including approved
excluded bridges. Only retained B observations enter losses. Shared-month and
multi-label components keep their original IDs. No train/C nodes may be added as
bridges. Never substitute whole_episode groups for AB_10_members.csv.

Before execution: verify exact sample set, ID join/order, labels/families, UTC
dates, retention, mapping K, checkpoint hash/cutoff, preprocessing, calibration
fit IDs, support hashes and joint graph source hashes across three views.
Reject missing/extra/duplicate IDs; no silent intersection. Final registry
verification must use an explicit historical allowlist, not scan Future folders.

## 4. Estimator (V1 §7, proposed implementation details)

Let ell_iv = -logsoftmax(z_iv / T_v)[y_i], in nats. z_iv is the K-class logit
vector for sample i/view v; T_v is fixed from A. Use float64 stable log-softmax,
not probability clipping that hides zero/underflow. Fail on non-finite losses.

For each supported family c, n_c counts retained B samples. Point loss:

    L_v = mean over c in S of [sum_i_in_Bc ell_iv / n_c].

For replicate b, draw a_bg independently from Exponential(rate=1) for each active
B group g. Active means at least one retained B sample in S. Sort original group
IDs lexicographically and persist this ordered list. Use NumPy Generator(PCG64(42)),
2,000 replicates. Persist draws and NumPy version, not only the seed. Use exactly
the same a_bg across views and across families belonging to that group.

    L_v^(b) = mean_c [sum_i_in_Bc a_b,g(i)*ell_iv / sum_i_in_Bc a_b,g(i)]
    U_v = sqrt(sum_b (L_v^(b) - mean_b L_v^(b))^2 / (2000-1)).

U is the sample SD across perturbations; do not divide by sqrt(2000).
Original L remains the point estimate. The group weights multiply every member's
loss and count: this is NOT equal-group point estimation. Larger groups retain
their original sample influence. Report replicate mean, bias relative to L, SD,
quantiles and paired covariance across views; quantiles are diagnostic, not
automatically calibrated confidence intervals.

Strictly positive draws preserve observed class support mathematically. Check
actual floating draws >0, finite, and every weighted denominator >0 and finite.
Failure writes a receipt and stops; no dropping, replacing or retrying replicates.
Multiplying all draws within a replicate by one common positive constant should
leave the estimator unchanged. No additional randomness for each view/class.

## 5. Meaning, limitations and stability review

This is conditional uncertainty of historical L with fixed CNN, temperature,
labels, S, cohort and operational groups. It is not predictive entropy, neural
epistemic uncertainty, label-error uncertainty, training-seed variation or future
temporal variation. No refitting of CNN/T occurs, so their uncertainty is omitted.
Operational TLSH components are not proven independent or exhaustive dependence
units. Cross-month linkage to A can make B less independent of calibration; this
bootstrap does not correct that dependence or validate temporal generalization.

Keep V1's proposed seed43 diagnostic (another2,000 draws). Review if U differs
from seed42 by more than max(0.005nat, 0.10*larger SD). Do not select the preferred
seed or average away a failure. This is an engineering trigger, not a theorem.

Leave one active group out at a time to diagnose point-loss influence. Preserve
the original S denominator; if a class disappears, mark the diagnostic undefined
and STOP for review. Report group size, within-class mass and each change in L.
Do not delete influential groups from the main estimate.

V1 also proposes a >0.05 absolute induced-fusion-weight change trigger over a
locked coefficient grid. That part is BLOCKED until coefficients/grid and the
precise leave-group-out recomputation contract are approved. Do not generate
weights or substitute a new loss threshold to claim that gate passed. Thus G5
numerical readiness can pass separately while eligibility to penalize U remains
pending. This explicitly preserves, rather than bypasses, V1 §7.2.

## 6. Finite verification plan, before real Q/U execution

1. Pure estimator tests: constant loss within each class gives U=0; independent
   reference loop matches vectorized sums; common draws preserve pairing and
   mixed-label groups; all-one draws recover L; scaling invariance; deterministic
   group-order canonicalization; invalid/zero/nonfinite inputs fail closed.
2. Synthetic simulation only: independent equal clusters; independent unequal
   clusters; highly correlated samples within clusters; dependence across nominal
   groups; rare families and one dominant group. Fixed scenario list and seeds
   must be saved before execution. Compare bootstrap SD with empirical repeated-
   dataset spread and report discrepancies, including underestimation. Do not
   declare universal validity from a favorable scenario or invent coverage claims.
3. Metadata/provenance readiness: establish complete A/B logits/temperature
   registry. No inference in this design gate. If prefix bundles are missing,
   request a separate bounded historical inference/calibration gate. Do not derive
   predictions from normalized Drift embeddings as an undocumented shortcut.
4. Only after review: execute Q/U on frozen B, save full losses/group draws,
   source/config hashes, original/per-class L, replicate values, U and diagnostics.
   These artifacts do not authorize coefficients, fusion or C/Future evaluation.

Simulation design is a further review deliverable, not an assertion of validated
coverage. Computational tests and source verification alone cannot satisfy G5's
scientific assumptions.

## 7. Approval checklist and stopping rules

Recommended next approval is ONLY task1 above: a new pure synthetic estimator
module and targeted tests, no real losses, inference, Q/U or fusion weights.
Confirm the complete-population B target (not Drift cap), shared exponential
weights, PCG64/2,000/seed42, ddof1, and diagnostic-only interpretation first.
Seed43/simulation execution and real-data preparation remain separate bounded
tasks; coefficient-dependent dominance gate remains pending.

STOP for binding/chronology mismatch, unavailable prefix predictions/T, support
failure, undefined influence, nonfinite draws/losses/denominators or stability
trigger. Do not change S, grouping, seeds, replication count or temperature
objective to get a pass. A failed validity review permits diagnostic reporting
only; QU/QDU stays blocked unless owner approves a justified scope change.

No existing code, V1, checkpoints or immutable artifacts changed in this review.
No tests were rerun: this batch changes documentation only. No U was computed.
