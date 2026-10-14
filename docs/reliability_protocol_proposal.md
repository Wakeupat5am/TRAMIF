# Historical reliability protocol — PROPOSAL, NOT APPROVED

Date: 2026-10-13. No reliability weights, bootstrap or MMD implemented/run here.
Source: Temporal_Reliability_Malware_Research_Framework.docx, sections 5-7 and
evaluation/ablation plan (local source: C:/Users/hoang/Desktop/Capstone/).
The March uniform result is already observed; this is not retrospective preregistration.

## Research question and controls
Does historically measured drift add predictive information beyond recent quality
for a view's next-period usefulness? Uniform improvement alone cannot answer this.
Keep cohort, main train split, primary/sensitivity duplicate policies and models
unchanged unless separately approved. Preserve all results, including negative ones.

## Options
| Component | Option | Advantage | Limitation / recommendation |
| --- | --- | --- | --- |
| Quality | Accuracy / macro-F1 | Interpretable | Discards probability quality; not framework's primary Q |
| Quality | Unweighted negative NLL | Available and smooth | Dominated by common families |
| Quality | Class-balanced negative NLL | Matches framework; every supported family contributes equally | Rare classes noisy; recommend support disclosure and cross-fitted calibration |
| Drift | TLSH distance | Already computed | Measures binary similarity, not fixed-encoder distribution shift; do not substitute |
| Drift | Marginal embedding MMD | Simple distribution discrepancy | Confounds within-family change with family mix; diagnostic only |
| Drift | Family-conditioned, null-standardized MMD | Controls family composition and sampling noise | Requires adequate support and consistent frozen encoder; preferred full-method candidate |
| Uncertainty | Predictive entropy | Easy per-sample confidence | Not uncertainty of estimated Q; not a substitute |
| Uncertainty | File bootstrap SE of Q | Simple | Underestimates dependence for related files |
| Uncertainty | Near-duplicate-group bootstrap SE of Q | Respects recorded groups | Mixed-label/large groups and missing-class replicates need explicit handling; preferred |

## Proposed Q and U (approval needed)
Let S be the same set of supported families for all three views in the historical
reliability window. Q_v = -(1/|S|) sum_c (1/n_c) sum_{i:y_i=c} [-log p_v(y_i|x_i)].
Higher Q is better. Report S, counts and excluded families; never change the 51
classification outputs just because a reliability estimate lacks support.

Do not score the same examples used to fit their temperature and call that held-out
quality. Candidate low-cost solution: deterministic five-fold group cross-fitting
within February. Fit T on four folds, score the fifth; assemble out-of-fold
probabilities for Q. Models were trained before February. Group assignments must
prevent near-related samples crossing calibration/scoring folds, including mixed
labels. Cross-fitting is within past data, not a random train/future-test split.
Final inference still uses the existing temperature fitted on all February.

Provisional support rule to approve after counts-only audit: >=20 files and >=5
groups per family, with all scored folds' fitting portions containing that family.
Do not tune this rule to March performance. If five folds cannot satisfy support,
stop and report; do not silently switch folds or remove classification classes.

U_v is the bootstrap standard error (standard deviation of replicate Q values).
Proposed 500 replicates, seed 42, same group draws for all views. Resample whole
groups, recomputing per-family means. Never split a mixed-family group to stratify.
Keep S fixed; replicates missing a required family are invalid, not assigned zero
loss. Report invalid rate; >5% invalid is an unavailable-U condition requiring review.
This estimates uncertainty conditional on fitted models/cross-fit calibration;
it is not training-seed uncertainty. If U unavailable but Q valid, quality-only
fallback; if Q itself unavailable, uniform fallback, recorded explicitly.

## Drift specification candidate (not ready to execute)
Use the same frozen view encoder on both compared windows, features before dropout,
and normalization fit only on the episode's training prefix. Never compare feature
coordinates from different independently trained encoders.
Use nonnegative biased RBF MMD squared. For each shared supported family, match
sample counts (candidate cap 128/family) and average family scores equally. Set RBF
bandwidth by a training-only median-distance sample, bounded and fixed per episode.
Compare to 100 same-period null splits with identical support/sample counts;
Dtilde=max(0,(Dconditional-null_mean)/max(null_sd,1e-8)). The epsilon, counts and caps
are proposals, not established defaults. Null splits must also respect groups.
If support/variance cannot support an estimate, mark unavailable, never drift=0.
Compute marginal MMD only as a composition diagnostic. Approximate CPU/RAM budget
must be measured on a bounded historical subset before full extraction/resampling.

## A real design conflict to resolve before implementation
Current final encoders were trained through January. January-vs-February embeddings
are comparable coordinates, but January is in-sample and February is out-of-sample.
MMD can reflect this training membership difference, not just temporal change.
Using those encoders to simulate honest October/November forecasts also leaks later
training information. Stored final-model logits cannot reconstruct honest earlier
encoder episodes. These limitations block a strong incremental-drift claim today.

Two routes for approval:
1. **Staged route (recommended now):** implement cross-fitted Q/group-U and a
   quality-only baseline first using historical saved logits. Keep monthly
   January/February MMD only as explicitly confounded exploratory diagnostics until
   an honest reference-window design is approved. No claim that full reliability
   is complete. This is CPU-bounded and needs no model retraining.
2. **Full mechanism route:** train episode encoders using only each training prefix,
   then use two later windows A/B to estimate drift/reliability and a later C to
   evaluate. Illustrative episodes: train Aug-Oct, A=Nov, B=Dec, C=Jan; train Aug-Nov,
   A=Dec, B=Jan, C=Feb. Each episode derives eligibility/mapping from its own prefix,
   not the final January-selected 51-family list. Compare methods on the same
   episode-specific cohort and disclose changing support. Only two episodes provide
   weak selection evidence; group bootstrap does not create more temporal episodes.

For the final full model, two truly held-out pre-March windows are not already
available at monthly resolution with the current train split. Possible choices are
early/late February (less support; submonthly sensitivity), or a reserved training
reference period with new fits (changes training selection), or retaining a stated
in-sample-reference limitation. These are alternative methodology decisions,
not interchangeable implementation details. Do not select one without approval.

## Weights, selection and leakage boundaries (design only)
r_v=a Q_v-b Dtilde_v-c U_v; w=softmax(r); coefficients nonnegative.
Suggested small candidate grid, pending approval: a in {1,2}, b in {0,0.5,1},
c in {0,1}; include uniform separately. Evaluate Q-only, Q+D, Q+U and full fairly.
Do not expand the grid after seeing results. Freeze signal definitions/scales before
coefficient comparisons. Log unavailable signals and the same fallback in every arm.
Use only earlier held-forward episode outcomes to narrow choices; March may serve
the framework's final development selection, but then cannot be called untouched
evaluation of the selected method. Report all candidate scores and selection rule.
Primary selection metric/tie rule must be approved (candidate: mean supported
macro-F1, then simpler rule); retain NLL/Brier even if they worsen.

Allowed: designated train prefixes, held-out historical A/B windows, historical
forward outcome labels for disclosed development selection. Prohibited: future
labels, future batches (even unlabeled) for fitting, April-September tuning or
changing cohort/support/grid after observing future results.
Freeze model weights, BN statistics, temperatures, support policy and one fusion
weight vector before future access. No monthly future-driven reweighting.

## Finite outputs and stop conditions
Phase Q: fold/group manifest, out-of-fold probabilities, Q/U/support report,
known-answer tests, fixed quality-only comparison. Stop on insufficient support.
Phase D (separate approval): episode specification, resource estimate, embeddings
with encoder hashes, conditional/marginal/null scores and fallback report.
Phase selection: complete candidate/ablation table, held-forward evidence, chosen
rule and a freeze manifest. No repeated ad-hoc neighbor or threshold exploration.
Still required later: five seeds and paired group-bootstrap comparisons as in the
framework, capacity/cost controls, sensitivity thresholds, and future evaluation.

## Decisions requested from owner
- Approve staged cross-fitted class-balanced Q and group-U design, including support
  and bootstrap choices, or request a different calibration/quality separation.
- Choose how the full method will obtain honest drift reference windows; approve
  episode training/resources separately. No costly training is authorized here.
- Approve coefficient grid and selection criterion only after episode/support
  feasibility is documented; do not treat this proposal as a final research protocol.
