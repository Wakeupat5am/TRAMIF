# Quality-only reliability — February evidence and weight freeze

Subsequent checkpoint: the separately approved March metrics and grouping pilot
are now complete; see docs/march_quality_only_results.md and
docs/march_grouping_results.md. The historical Task 4 pause recorded below was
lifted for those scopes only. Formal statistical uncertainty remains unapproved.

Date: 2026-10-13. Tasks 1-3 complete; Task 4 NOT RUN in this batch.
No CNN retraining, March fitting, Future Test access, commit or push.

## Evidence chain
- Audit: data/derived/february_group_audit/20261013_090516_568369/.
- OOF: data/derived/february_crossfit/20261013_091117_010698/.
- Freeze: data/derived/quality_weight_freeze/20261013_124031_275438/.
- weights.json SHA-256:
  `3fbaf039b608c4df195922b6fc29a2ff141b6fbdd1a4527a5a8c87f96eecc0ba`.
- check_quality_freeze verifies the audit input/artifact chain, saved OOF payload,
  exact probability recomputation from each fold's temperature, quality/weights,
  copied manifests and inference-temperature/checkpoint provenance. PASS.
  Verification reads February artifacts without refitting or model execution.

## Why cross-fitting is needed
Fitting a temperature and estimating its quality on exactly the same labels can
give an optimistic quality estimate. Each sample's OOF probability instead uses
a temperature fitted on the other four groups of folds. Encoders remain frozen.
All 3,799 retained February samples contribute to standard-NLL calibration;
26 excluded samples are graph vertices only. No excluded sample gets an OOF row.

All three arrays have shape (3799, 51); saved combined shape is (3, 3799, 51).
The fold manifest is explicitly joined by sample ID, and class/label/checkpoint
and primary-policy provenance are checked. Exactly one OOF row per sample/view.

| View | T fold 0 | T fold 1 | T fold 2 | T fold 3 | T fold 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SBSMI | 2.571574 | 2.360868 | 2.566188 | 2.573520 | 2.507900 |
| Raw-byte | 2.249621 | 2.099542 | 2.283247 | 2.266285 | 2.148046 |
| Entropy | 1.950348 | 1.864075 | 1.984284 | 1.982943 | 1.886883 |

Objective standard mean NLL, original bounds [0.05,20]; no boundary optimum.
Five temperatures are for OOF estimation only. March inference will use the
existing full-February temperatures 2.520203445 / 2.211872879 / 1.935489538.

## Frozen quality and weights
S = 31 supported families, 3,646 retained samples, fixed by >=20 samples and >=5
operational components. Every S class appears in fitting and scoring for all folds.
The remaining retained samples stay in calibration; all 51 model outputs remain.

For view v, sample i in fold f(i), calibrated OOF probability is
`p_vi = softmax(z_vi / T_v,-f(i))`. Here z is the frozen model's saved logits;
T is fitted without that sample's entire component. Let n_c be sample count of
class c in S. The positive loss is
`L_v = (1/|S|) sum_{c in S} (1/n_c) sum_{i:y_i=c} -log(p_vi[y_i])`.
Pool sample losses per class, not equally weighted fold means. Lower L is better.
`w_v = exp(-L_v) / sum_u exp(-L_u)`; coefficient 1, drift 0, uncertainty penalty 0.

| View | Class-balanced OOF NLL on S | Standard OOF NLL, all February | Frozen weight |
| --- | ---: | ---: | ---: |
| SBSMI | 0.858230315 | 0.827768707 | 0.401721560 |
| Raw-byte | 1.195605205 | 1.089332581 | 0.286685077 |
| Entropy | 1.112290514 | 1.032135872 | 0.311593363 |

All weights finite, nonnegative, sum to one. Artifact verification reproduces the
numbers exactly. Unit tests cover a hand-computed class-balanced loss/softmax,
temperature fit/score separation, canonical alignment and file-level integration.
Changing a separate March-label fixture cannot change the February computation;
production crossfit/freeze APIs take no March prediction/label input. Shared
historical inventory files are filtered to February before grouping.
Final full suite: 99/99 PASS (0.699s), including immutable-artifact tamper checks.

## Limits and current stop
These are historical quality estimates, not proof of future improvement. Quality
is estimated only over S; 31-class support does not imply verified quality for all
51 families. Mixed-label/transitive groups and TLSH false negatives remain. Some
classes have highly unequal group sizes. Five folds are not temporal replications;
temperature fit subsets overlap. The experiment uses one existing training seed.
Optional February bootstrap uncertainty was NOT computed and is not in weighting.
Existing March baseline/uniform results were already observed; this is development,
not retrospective preregistration or a final unseen test.

Task 4 paused before new March evaluation because its paired-uncertainty grouping
and missing-class replicate rules need approval. The owner restricted threshold
10 approval to February feasibility. It must not silently become a March bootstrap
protocol. See docs/march_uncertainty_decision.md. Weights remain locked regardless
of that choice; no March-driven reweighting is permitted.

## Recheck without fitting
`python -B -m scripts.check_quality_freeze --freeze-dir data/derived/quality_weight_freeze/20261013_124031_275438`

Frozen artifacts live under ignored data/. This report is not an artifact backup.
