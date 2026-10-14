# TRAMIF BALANCED v2 — scope and information budget

Date: 2026-10-14 (Asia/Saigon). Status: Phase 1 scope APPROVED by owner;
signal estimators, coefficients and implementation remain NOT APPROVED.
Authorization: pasted request 19d21a9d-c0dc-4494-b3e0-d5676e34d156.

## Research objective and evidence boundary
Test one static historical quality/drift/uncertainty fusion rule, using the existing
three CNN checkpoints, against matched-history ablations. A negative result is a
valid completed experiment. No assertion that drift implies error, nor that lower
uncertainty implies a more accurate model. No CNN retraining in this phase.

Train remains August 2019-January 2020. February and March become pre-April
historical development data for v2. April-September remains unopened Future Test.
Original 51-class eligibility/mapping, chronology, primary TLSH retention distance 0
and all image selections remain unchanged. TLSH <=10 is an audit grouping proposal,
not a new primary retention rule or proven statistical independence threshold.

## Immutable legacy
Preserve all checkpoints, February temperatures, February OOF weights, historical
prediction files, March metrics, March bootstrap and prior grouping/fold manifests.
v2 groups cannot retrospectively alter February cross-fitting or March intervals.
New versions will have unique artifact paths, explicit parent hashes and their own
approval records. This phase computes no new quality score or mixture weight.

## Information budget of nine arms
All arms share the same August-January checkpoints, image preprocessing and ordered
51 outputs. All use the existing full-February temperatures; no March temperature
refit is authorized. New arms are defined here, not implemented.

| Arm | Historical information used | Status |
| --- | --- | --- |
| SBSMI calibrated | Train CNN; February temperature | Legacy implemented |
| Raw-byte calibrated | Same dates, its own checkpoint/temperature | Legacy implemented |
| Entropy calibrated | Same dates, its own checkpoint/temperature | Legacy implemented |
| Uniform | Same three calibrated predictions, fixed 1/3 weights | Legacy implemented |
| Legacy Quality-only (February) | February grouped OOF quality/support and frozen weights | Legacy implemented; no March-based changes |
| New Quality-only (March) | Shared v2 historical budget; March held-forward calibrated quality on S* | PLANNED |
| Q+D | Exactly the same quality as new Q; February-March representation distributions | PLANNED |
| Q+U | Exactly the same quality as new Q; group uncertainty of that estimate | PLANNED |
| Q+D+U | Same Q, D and U definitions as the corresponding ablations | PLANNED |

All four new arms have equal access to February/March IDs, labels, saved predictions,
operational groups and (only after approval) embeddings. Ablation intentionally
disables a signal; it does not withhold a newer labeled window from the Q control.
They must share S*, samples, group policy, temperatures and checkpoints. S* is a
quality/drift estimation support set, not a new classifier or evaluation cohort.
Final S*/null rules need Task 3 approval after the counts audit.

Legacy Q versus full v2 conflates newer quality information with D/U effects; report
it as a historical reference, never as the sole incremental-mechanism comparison.
Primary mechanism contrasts to finalize before Future Test: QD-newQ, QU-newQ,
QDU-newQ (and QDU-QD/QU if predeclared). No significance/menu selection yet.

## What each evidence category means
- Development evidence: the old February -> March experiment and feasibility of
  v2 signals. March labels may estimate v2 quality; v2 scoring back on March is
  not independent validation of its weights.
- Exploratory diagnostics: entropy/disagreement, drift-error associations,
  group concentration and null feasibility. These do not prove forecasting value.
- Confirmatory comparisons: only predeclared hypotheses, endpoints, method arms,
  uncertainty/multiplicity rules frozen before any future result. Current approval
  does not authorize them to run.
- Final future evaluation: all frozen arms on the identical authorized monthly
  retained sets, without coefficient/model selection using April-September labels
  OR unlabeled future batches. No monthly test-driven reweighting.

March label availability before April is a retrospective research assumption;
real-time label availability is UNKNOWN. Both months are after CNN training, but
February labels were used for calibration. March calibrated quality is held-forward
with respect to CNN and temperature fitting, not untouched after method development.

## Framework requirements crosswalk
Source: C:/Users/hoang/Desktop/Capstone/Temporal_Reliability_Malware_Research_Framework.docx.
Sections and roles below are requirements/proposals, not evidence of completed work.

| Framework requirement | Current evidence / BALANCED disposition |
| --- | --- |
| 51 eligible known families selected only from train; chronological split (§11.2) | VERIFIED existing manifests; preserved |
| Three binary views; calibrated probabilities (§4, §7) | IMPLEMENTED and historical artifacts available |
| Held-out historical quality (§6-7) | February OOF legacy VERIFIED; March v2 quality PLANNED |
| Conditional/marginal fixed-encoder MMD, historical normalization, matched null (§5) | NOT IMPLEMENTED; Phase 1 audits prerequisites only |
| Quality-estimate uncertainty (§6) | NOT IMPLEMENTED; positive group weights are a proposal, not ordinary bootstrap already approved |
| Static Q/D/U weighting and mechanism ablations (§7, §13) | Scope accepted, formulas/coefficients NOT APPROVED |
| Honest prefix episodes and held-forward next-error predictor (RQ2, §12.2) | NOT DONE, outside current BALANCED implementation scope; cannot claim independently validated predictive mechanism |
| Five training seeds (§11.4) | One existing seed only; unmet in BALANCED, not silently declared complete |
| Mean/worst monthly supported Macro-F1, pooled fixed-family F1, all calibration metrics (§11.4) | Future outcomes NOT EVALUATED; retain as planned endpoints; CI/multiplicity protocol unresolved |
| Historical near-duplicate primary/sensitivity (§11.2) | Primary historical policy implemented; sensitivity performance/future application still missing |
| Strong/cost-matched alternatives: learned fusion, gating, 1D/MLP/PE baselines (§11.3) | Not covered by the nine approved arms; no claim of superiority to all these methods |
| Capacity, total byte-to-decision costs and reproducibility (§12, §15) | Partial pilot evidence; systematic cost controls and artifact backup remain incomplete |
| Cross-time supervised contrastive training (RQ4, §8) | Optional, outside this phase; not necessary for testing the approved static rule |

## Permitted and impermissible claims
If a future authorized experiment is favorable: this particular fixed rule improves
the specified outcomes over specified controls on this BODMAS known-family cohort,
time horizon and existing seed, subject to the recorded uncertainty assumptions.
Do not claim universal temporal robustness, causal effect of drift, Bayesian model
uncertainty, unknown-family detection, multi-seed stability or superiority to unrun
cost/capacity baselines. The existing view architectures differ; results concern
view-model pairs, not intrinsic representation quality with capacity controlled.
BALANCED is a narrower experiment than completion of every original framework RQ.

## Phase 1 audit contract (before numerical results)
- Input graph: original February/March cohort inventory, including primary-excluded
  records only as bridge vertices. No raw binaries, images, future rows or logits
  used to choose graph edges. Equality of content SHA OR standard TLSH diff <=10.
- Retained IDs must match the existing 3,799 February and 4,332 March selection.
- Enumerate all pairs; report within/cross-month exact and TLSH edges, components,
  mixed labels, bridges and changes from within-month grouping.
- Candidate S*: >=20 retained samples AND >=5 JOINT operational groups in EACH
  month. Report every family and failures; do not lower thresholds on failure.
- For each family/month compute the maximum equal sample budget that two disjoint
  sets of whole groups could supply (subset-sum capacity). This allows subsequent
  subsampling within each assigned side; it is not a null estimator or its approval.
- Also construct ONE deterministic multi-family disjoint-group allocation witness
  per month to expose simultaneous-support limitations. It is not an optimized or
  repeated null design; failure is not a proof no better partition exists.
- No required cap, null count, null variance threshold, U estimator, beta/gamma or
  fallback is selected. Count-only diagnostics inform Task 3; no D/U draws occur.

## Gates
Task 1 scope complete; Task 2 produces immutable counts/provenance artifacts and
tests. Owner then approves/rejects the exact Task 3 estimator/support/null design.
Feature extraction, GPU use, signal calculation and weights require later approval.
Future Test requires an additional final protocol freeze and explicit authorization.
No commit/push, training, legacy mutation or fallback selection in this phase.
