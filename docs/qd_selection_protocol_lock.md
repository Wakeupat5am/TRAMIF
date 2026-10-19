# D048 — Pre-outcome Q/QD selection lock

2026-10-17. Owner approved locking D selection under V1 before E1-C metrics.
**LOCKED_PRE_OUTCOME_SCORING.** No real coefficient selected or C metric computed.

## Fixed rule and scope

Q persistence forecast for family c/view v is its saved calibrated B mean NLL.
QD forecast is `L_v,c,B + beta*d_v`, with `d_v=primary descriptive MMD²/2`.
Use exactly beta={0,0.5,1}, a=1, gamma=0. U remains diagnostic-only and unavailable
for fusion: QU/QDU are not silently represented as valid full-reliability arms.

This operationalizes V1 §8.2's beta grid and §6.4's approved descriptive fallback;
reducing the original nine QDU candidates to three QD candidates follows the
owner's exclusion of U. It explicitly approves this bounded descriptive scale/grid
for selection preparation, not statistical equivalence to null-standardized D.
No change to V1 or original amendment file; this later decision is recorded here.

Select beta on E1 by minimum MAE between saved per-family forecasts and actual
calibrated C per-family NLL. Average equally over S_eval families, then three views.
S_eval is frozen S intersect true-supported C classes; it cannot refit signals or
support. If empty, stop NOT_ESTIMABLE. All S forecasts are saved before scoring.
Ties within1e-12 prefer fewer nonzero terms, then smaller beta+gamma, then beta.
Gamma0 reduces the latter ordering to smaller beta. Beta0 is a valid negative result.

After E1 selection, freeze beta before E2 outcome scoring; do not retune on E2,
Final March or Future. Corresponding fusion rule is `softmax(-L_v-beta*d_v)`,
averaging fixed-A-temperature probabilities. No actual QD fusion weights generated
in this batch. E1 fusion metrics are model-selection development evidence, not an
unbiased post-selection evaluation. E2 remains an overlapping retrospective
held-forward check, not an independent temporal replication or Future Test.

## Why this step precedes evaluation

Locking the candidate grid/objective prevents changing the definition of success
after seeing January results. Forecast MAE directly addresses the limited RQ2
question of D adding information beyond Q; choosing whichever fusion has highest
January accuracy would answer a different selection question. Saving forecasts
before outcome scoring makes this ordering auditable. Do not change h/cap or use
group-balanced/cap64 sensitivity winners to alter these primary signals.

D is a bounded numerical distance, not a significance score or empirically
calibrated reliability penalty. The largest permitted beta1 adjustment is about
.0211nat in E1 and.0299nat across all three cases. A small grid may have little
effect; retain weak/negative results rather than expanding it after selection.
Positive penalties cannot forecast quality improvements (loss decreases).
Multi-seed and learned-static controls remain required for broader framework claims.

## Verification and evidence

5 synthetic tests PASS (0.006s): reference forecasts, beta0 recovery, selector MAE,
tie/zero-winner behavior, family/view averaging invariance and invalid inputs.
Saved forecasts for E1/E2/Final have shapes(3,3,17)/(3,3,25)/(3,3,25), indexed by
candidate, view, family. Frozen Q and primary descriptive D only; no new MMD,
inference, T fitting, U, C scores, Future access or coefficients tuned.

Artifacts:`data/derived/qd_selection_protocol_v1/20261017_033146_321989/`.
Contains protocol.json, three forecasts.json files, completion/checksums.
Completion SHA256:
`3f50e4aec6ec37cc7fe9fcceb912a27b499f4be3517790d0d0553b73d023e152`.
All output/source hashes and forecast arithmetic independently rechecked PASS.
Primary MMD report was hash-pinned directly; its previously documented bad
self-checksum entry is not used or repaired. Q inputs/selection/support remain pinned.

Next bounded execution: implement and run E1-C scoring from the already exported
logits, compare all fixed beta forecasts using this lock, record the selected beta
and all candidate MAEs, plus three single-view/Uniform/NewQ point metrics. Any
selected-QD E1 metrics must be labeled in-selection development. Then stop before
E2 inference/evaluation or Future access. No additional candidate/grid design gate
is needed for this exact approved rule. No commit/push.
