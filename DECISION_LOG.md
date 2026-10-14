# Decision log

## D015 — BALANCED Phase 1 observed findings, no estimator approval (2026-10-14)
Scope D014 fulfilled. Run balanced_phase1_audit/20261013_233825_921873 (UTC);
docs/balanced_phase1_results.md records full support and limitations. Joint <=10
graph: 4,048 active groups, 178 across retained months, 41 mixed-label groups;
no exact-content edges. S* candidate 25 families (3,380 February/4,064 March).
Primary threshold0 selection, 51 outputs and all legacy checkpoint/temperature/
quality/March/bootstrap files preserved. No new Q/D/U weights.

Methodology gate remains: group-separated within-period nulls do not automatically
match observed cross-month dependence. Even agent's best February binary group
allocation only permits 9 matched samples per side. No support reduction, group
splitting, shared-group deletion, null/normalization fallback or coefficient selected.
Positive group weights would preserve observed support, not provide independent
units or training/temporal uncertainty. U implementation and interpretation still
need approval. Single seed/no honest prefix episodes constrain all future claims.

One new-runner provenance check initially confused canonical JSON identity with
file-byte checksum. Corrected to existing loader semantics, both identities recorded;
regression tested. No legacy file repaired/replaced, no results bypassed.
130/130 tests PASS, including 10 new tests. Counts audit INTEGRATION VERIFIED;
statistical assumptions and D/U predictive usefulness NOT RESEARCH VALIDATED.
Stop for owner Task 3 decision; Tasks 4/5 need later scope/resource/future gates.

## D014 — BALANCED v2 Phase 1 approved (2026-10-14, owner)
Only scope/information budget and February-March feasibility audit. March may be
historical development for v2; preserve February quality and old March artifacts.
Nine arms, four new ablations share historical budget/support/checkpoints/February
temperatures. Candidate graph threshold <=10 and support >=20 files/>=5 groups in
each month are audited, not silently revised or promoted to validated methodology.
Audit can write new reports/tests/code; source datasets remain read-only. No new
quality weights, embedding extraction, MMD/null estimator, positive-weight U,
beta/gamma, fallback, GPU/training or future evaluation approved. Task 1 scope and
pre-result audit contract: docs/balanced_research_scope.md. D/U requires next gate.

## D013 — Approved bootstrap completed (2026-10-13, observed)
Run: data/derived/march_paired_bootstrap/20261013_131843_476287/.
All 2,000 seed-42 paired draws retained, including 1,182 missing-class draws.
Primary pointwise intervals favor quality-only over uniform for Accuracy/NLL/Brier;
secondary SBSMI Brier favors SBSMI and secondary SBSMI NLL spans zero. Do not turn
these into a joint, all-baseline or future-generalization superiority claim.
The first 500 draws exactly reproduce the earlier support-only pilot because its
generator and seed were also 42. This prefix reuse is disclosed; no draws or seed
were chosen using metric outcomes. Original point estimates remain the reported
estimates; bootstrap means are diagnostic only. No replicate was discarded.
Moment review flags were not triggered. Group-deletion diagnostics preserve the
primary point-difference signs but show magnitude sensitivity; no group was removed
from estimates or intervals. Group independence remains unverified, size concentration
is substantial, and these intervals omit training/calibration/weight-estimation and
temporal variability. No estimator change is justified or performed in this batch.
120/120 tests PASS. Findings and provenance: docs/march_paired_bootstrap_results.md.
Stop here. Any broader uncertainty, additional training, drift weighting or future
evaluation needs a separately authorized task. D012 supersedes the old proposal.

## D012 — Paired group bootstrap approved (2026-10-13, user-authorized)
2,000 March whole-group paired replicates, seed 42 (supersedes proposed seed
20261013). Use the audited graph, frozen selection/predictions/weights. One common
draw for all five methods. Primary Q-uniform; secondary Q-SBSMI/raw/entropy.
Only Accuracy/NLL/multiclass Brier get 95% pointwise percentile intervals; all
other metrics remain point estimates. Keep every class-missing draw. No simultaneous
coverage/multiple-comparison adjustment or overall superiority claim. No training,
future access, weight changes or Git publication. Save all multiplicities and N.
Implementation uses per-group sums of additive metrics; verify against explicit
sample repetition. Quantiles use NumPy linear interpolation. Group leave-one-out,
multiplicity associations and moment/histogram diagnostics do not alter the estimator.
Descriptive review flags fixed before run: |skewness|>2 or excess kurtosis>10;
these are neither validity guarantees nor significance tests. Investigate pathologies
and stop before changing estimator; never prune replicates or retune the seed.

## D011 — March outcome and uncertainty feasibility (2026-10-13, observed)
Quality-only improves all seven March point metrics versus uniform, but still has
worse NLL/Brier than SBSMI and worse ECE than raw/entropy. No significance claim.
Weights hash unchanged. March remains previously observed development validation.
The approved grouping audit found 2,234 active groups, 28 mixed-label groups and
60.8% support-missing pilot draws. Do not use only the 196 complete-support draws,
change macro denominators, split mixed-label groups, or replace March support with
February S. A reduced final uncertainty scope is PROPOSED in
docs/march_uncertainty_decision.md; formal bootstrap still requires owner approval.

## D010 — March metrics and grouping pilot only (2026-10-13, user-approved)
Task 4A: five methods on fixed 4,332 March samples; use full-precision frozen
weights and full-February temperatures. No fitting, selection or reweighting.
Task 4B: March SHA/TLSH<=10 component feasibility audit, analogous to February;
excluded March records are bridge-only vertices, primary retained set unchanged.
Up to 500 pilot replicates allowed for support feasibility only; use 500 and seed
42, saving all draws/counts and missing-class flags. No model metric intervals or
final bootstrap performed. Group-size effective counts are descriptive concentration
indices, not proof of independent units. Report missing support without discarding
replicates or changing 47/51 denominators. Final bootstrap requires another approval.

## D009 — Quality weights frozen; March uncertainty decision pending (2026-10-13)
Task 3 applied the approved formula without optimization: weights SBSMI/raw/entropy
0.4017215600259309 / 0.2866850774030112 / 0.3115933625710579. Source is February OOF
class-balanced NLL on fixed 31-class S. Artifact and formula reproductions PASS.
No optional bootstrap penalty or diagnostic uncertainty was introduced.
Before Task 4, request explicit March resampling-group and missing-class rules:
threshold <=10 approval was scoped to February feasibility, not March bootstrap.
docs/march_uncertainty_decision.md is a proposal only. Do not substitute an IID file
bootstrap or silently discard class-missing replicates. Frozen weights are not
reselected after March results, regardless of the eventual uncertainty decision.

## D008 — Feasibility passed; shared OOF calibration (2026-10-13, observed)
Task 1 has 2,032 active groups, largest 144, support 31 classes / 3,646 samples;
all five fit/score partitions separated and support present. Threshold adequacy
and independence are limitations, not inferred from PASS. No threshold/seed search.
Proceed under D007 with the one saved partition, standard NLL, bounds [0.05,20].
Task 2 fits temperatures only on other February folds; existing full-February
temperatures remain the inference temperatures. Never average the five fitted
temperatures or apply OOF temperatures to March. No CNN retraining.

## D007 — Conditional quality-only approval (2026-10-13, user-authorized)
Existing checkpoints, primary policy 0, temporal split and 51 outputs unchanged.
Audit February components using content equality OR TLSH <=10, including excluded
February vertices only as bridges. Five grouped folds; support >=20 retained files
and >=5 components. If feasible, fit standard-NLL temperatures with existing bounds
out of fold, calculate class-balanced OOF NLL, freeze softmax(-L) with coefficient 1.
Drift=0 and uncertainty penalty=0. March only after freeze, development evaluation.
No threshold/fold/support changes on failure; no future data, CNN training or Git
publication. Older D006/proposal alternatives remain unapproved outside this scope.
The grouping audit and predeclared engineering review gates are documented in
docs/february_feasibility_protocol.md. Deterministic February-only stratification
uses no predictions. Operational TLSH groups are not proven independent groups.

## D001 — Controlled autonomy (2026-10-13, accepted)
User authorized bounded technical implementation and light tests, with at most four
tasks per batch. No commit/push or expensive training without approval. Methodology
and future-test decisions require explicit review. Source: user request attached
2026-10-13. This supersedes the previous read-only working mode for routine tasks.

## D002 — Preserve existing research state (2026-10-13, accepted)
Do not retrain or change cohort/splits/near-duplicate thresholds in this batch.
The three filtered February calibrations are the current baseline artifacts.
Validation results were already observed before this batch, including before the
near-duplicate policy was defined. No claim of preregistration is made.

## D003 — Verification before fusion (2026-10-13, accepted engineering choice)
Add CPU regression tests for existing calibration mathematics and verify the saved
artifact chain before combining probabilities. This changes neither the fitting
objective nor any model weights. Evidence and outcomes: PROJECT_PROGRESS.md.
Uniform fusion is the next baseline; reliability formulas/episode design are not
newly approved by this entry.

## D004 — Uniform calibrated baseline (2026-10-13, user-authorized)
Equal 1/3 weights average each view's February-calibrated probabilities, not logits.
No weight fitting, no temperature refit; evaluate March on the established primary
selection (4,332 samples). Reject ordering differences rather than silently join,
sort or intersect samples. Existing loader also checks the canonical manifest.
March labels are used only for integrity checks and scoring. March has already
been observed during development; results are not an untouched research test.

## D005 — Preserve mixed fusion outcome (2026-10-13, observed)
Uniform improves March accuracy/F1 but worsens NLL/Brier versus SBSMI. Retain all
metrics and all single-view baselines; no weight tuning in response to this result.
Evidence: docs/uniform_fusion_results.md and its immutable artifact run.

## D006 — Reliability methodology (2026-10-13, PROPOSED ONLY)
docs/reliability_protocol_proposal.md compares quality, drift and uncertainty
options. Recommend a bounded cross-fitted class-balanced quality stage before full
drift work. The current encoder's training membership confounds January/February
drift, and using final encoders in earlier forecast episodes leaks training time.
Owner approval is required for support/cross-fitting/bootstrap choices and honest
drift windows/episode fitting; expensive training requires separate authorization.
No coefficients selected, Q/D/U computed, or reliability weights implemented.
