# TRAMIF project checkpoint

Updated: 2026-10-14 (Asia/Saigon). Base commit: 6b65430.
Working mode: controlled autonomous development; no automatic commit/push.

## Current batch — BALANCED Phase 1 (D014)
Only Task 1 scope/information budget and Task 2 February-March support/group
feasibility authorized. Task 1 COMPLETE: docs/balanced_research_scope.md defines
nine arms, shared v2 information budget, immutable legacy, framework crosswalk,
claim limits and pre-result count-audit contract. No executable methodology yet.
Task 2 COMPLETE: data/derived/balanced_phase1_audit/20261013_233825_921873/.
34,134,453 pairs; 4,048 active joint components; 0 exact-content edges;
12,280 cross-month TLSH edges; 178 retained cross-month groups (2,197 samples);
41 multi-label groups; largest 278. Local graph partitions reproduced exactly.
Candidate S*: 25 classes, February 3,380 / March 4,064 samples. Primary selection
remains 3,799/4,332 and all 51 outputs; thresholds unchanged. Shared historical
budget and nine arms are documented; no new weights computed.
Null design gate: agent February permits at most 9 samples per disjoint group
side; observed months share components while naive within-month null sides do not.
Count feasibility is not null exchangeability/power. Positive group-weight U is
count-feasible but concentrated groups limit interpretation; no U draws performed.
Full CPU suite 130/130 PASS (1.343s); 10 new tests. Six prediction alignments,
checkpoint/temperature/preprocessing identity and legacy artifact hashes PASS.
Results: docs/balanced_phase1_results.md; decision D015. Feature extraction, MMD,
U estimator, coefficients, fallback, new weights and Future Test remain unapproved.
BATCH STOPPED after Tasks 1–2. Next is owner review of Task 3 methodology design.
No inference/training, primary-data change, commit or push.

## Previous batch — approved paired bootstrap (D012)
Task B1: implement/test additive group metrics, shared deterministic draws and
percentile CIs; verify frozen artifact chain. Task B2: 2,000 paired draws, seed42,
diagnostics and bounded conditional report. No other-metric CIs or fitting. Stop
after this batch; full reliability/Future Test remain outside authorization.

B1 COMPLETE: scripts/paired_group_bootstrap.py and scripts/run_march_bootstrap.py;
11 new tests cover shared deterministic draws, explicit whole-group sample expansion,
metric consistency, missing-class retention, quantiles and persisted reproducibility.
B2 COMPLETE: data/derived/march_paired_bootstrap/20261013_131843_476287/.
2,000 draws, seed 42, 2,234 active groups; all 4,332 retained IDs aligned by ID.
Primary Q-uniform point differences and pointwise 95% percentile CIs:
- Accuracy +0.507849 percentage points [0.119380, 0.947476].
- NLL -0.0129381 [-0.0184874, -0.00768917].
- Brier -0.00620765 [-0.00857720, -0.00407403].
All 1,182 missing-class replicates retained; draw sizes 3,512-5,622; no non-finite
metrics. Leave-one-group-out diagnostics never reverse these point-difference signs,
but some groups materially affect magnitude. No altered-group CIs were computed.
Full suite 120/120 PASS (0.790s). Frozen inputs verified before and after execution.
Report: docs/march_paired_bootstrap_results.md. Secondary SBSMI Brier favors SBSMI;
secondary NLL interval includes zero. These are conditional development intervals,
not simultaneous coverage or proof of future generalization. Macro-F1, balanced
accuracy and ECE remain point estimates. BATCH STOPPED; no further evaluation or
methodology changes authorized. Historical checkpoint sections below retain their
then-current approval state; D012/D013 and this section supersede pending-bootstrap
statements. Final persisted artifact checksum check: 9/9 PASS; frozen weights hash
unchanged; source/document whitespace scan clean. HEAD remains 6b65430, nothing
staged. No CNN training, Future Test access, commit or push.

## Previous batch — March metrics and grouping feasibility (D010)
Task 4A authorized: verify freeze/provenance, evaluate the five fixed methods on
4,332 canonical March samples, reproduce previous four baselines, save predictions
and every requested metric. Task 4B authorized: exhaustive March grouping <=10,
distribution/class support, at most 500 support-only pilot draws. No final bootstrap,
Future Test, training, weight changes or Git publication. Previous pause is lifted
only for these scopes; final uncertainty methodology still awaits approval.

Task 4A COMPLETE: data/derived/quality_march_evaluation/20261013_125046_074164/.
Freeze/provenance and canonical 4,332 IDs/class ordering verified; previous four
methods' probabilities reproduced exactly and all seven metrics match. Quality-only:
accuracy .714681440, macro-F1 supported .748930605, fixed .690190950, balanced
accuracy .734080914, NLL 1.094385097, Brier .401633049, ECE15 .076167751.
All seven point metrics improve versus uniform; NLL/Brier remain worse than SBSMI.
Three new Task 4A tests PASS (0.008s); no statistical superiority claim. Task 4B
follows with support-only pilot, not confidence intervals or significance tests.

Task 4B COMPLETE: data/derived/march_group_feasibility/20261013_125459_149338/.
4,438 original March vertices / 4,332 retained, 2,236 total / 2,234 active groups;
largest retained 166 (3.832%), 28 mixed-label groups. Size-only effective count
174.514 (descriptive, NOT independent sample size). Ten supported classes have
2-3 groups. 304/500 pilot draws lack >=1 originally supported class; all draws saved,
none retried/discarded. No metric bootstrap or CI. Raw selection/weights unchanged.
Results: docs/march_quality_only_results.md, docs/march_grouping_results.md.
Final full suite: 109/109 PASS (0.713s). Stop after this batch. Updated proposal in
docs/march_uncertainty_decision.md awaits owner approval; no final bootstrap yet.
Final persisted checksums PASS for freeze, March evaluation and March audit runs.
No staged/tracked changes, commit or push; pre-existing and new working files remain
untracked. HEAD remains 6b65430. Source/document trailing-whitespace scan clean.

## Previous batch — conditional quality-only (2026-10-13)
Task 1 first: February component/support/fold audit at fixed TLSH <=10. No fitting
until feasibility review. Tasks 2-4 are conditional: OOF calibration, quality/weight
freeze, then March development comparison. No future access or retraining.
Predeclared gates: docs/february_feasibility_protocol.md; authorization: D007.
Task 1 COMPLETE: seven new audit tests PASS (0.016s), real integration PASS.
Run data/derived/february_group_audit/20261013_090516_568369: 2,032 active groups,
largest 144; S=31 classes / 3,646 samples; five score folds 763/752/761/763/760;
21 mixed-label groups; no fit/score overlap. Detailed limits:
docs/february_feasibility_results.md. Computational feasibility permits Task 2;
group independence and threshold statistical adequacy are not claimed verified.

Task 2 COMPLETE: data/derived/february_crossfit/20261013_091117_010698/.
CPU-only 15 standard-NLL fits, bounds [0.05,20], none at boundary. Exactly one OOF
prediction for each of 3,799 samples per view, unchanged 51 outputs. Shared manifest
is joined by ID and checked against canonical February predictions and checkpoints.
Nine quality/crossfit tests PASS (0.089s); Task 1 full-suite checkpoint 88/88 PASS.
No March predictions or labels used for fitting. Task 3 will independently verify
saved OOF probabilities from the saved temperatures without repeating fitting.

Task 3 COMPLETE: data/derived/quality_weight_freeze/20261013_124031_275438/.
Class-balanced OOF NLL: SBSMI 0.858230315; raw-byte 1.195605205; entropy 1.112290514.
Frozen weights: 0.401721560 / 0.286685077 / 0.311593363. Full-February inference
temperatures and model provenance pinned. check_quality_freeze PASS: complete audit
chain, exact OOF probability recomputation, exact loss/weight reproduction and
inference bindings, without redoing calibration. Full suite before freeze 97/97
PASS (1.462s); final suite including freeze-integrity tests 99/99 PASS (0.699s).
Durable evidence: docs/quality_only_results.md. Optional February bootstrap not run.

Task 4 PAUSED FOR OWNER METHODOLOGY DECISION, not executed: March bootstrap group
definition and missing-class handling are not approved by the February-only audit
threshold authorization. Proposal: docs/march_uncertainty_decision.md. No March
comparison or paired interval from this new method exists yet. No future access.
Current resume point: obtain that decision; verify frozen weights via
scripts/check_quality_freeze.py; then run only the approved March evaluation scope.

## Definition of Done
Reproducible three-view pipeline; validated historical reliability and fusion
comparisons; frozen protocol before future evaluation; temporal results and
limitations documented. Existing single-seed validation does not satisfy this yet.

## Verified starting state
- Three generated historical datasets: 18,061 train + 8,263 validation per view.
- Primary policy retains 3,799 February and 4,332 March samples; train unchanged.
- Three 100-epoch runs (seed 42), filtered evaluations and February temperatures.
- Uniform calibrated fusion is now evaluated on March (see current checkpoint).
  Q/drift/uncertainty estimator, reliability fusion, multi-seed conclusions and
  future evaluation are not completed.

## Previous governance batch
| Task | Acceptance | Status/evidence |
| --- | --- | --- |
| Persistent rules and control documents | Preserve established methodology; explicit approval boundaries and next tasks | IMPLEMENTED: AGENTS.md, this file, DECISION_LOG.md, TECHNICAL_WALKTHROUGH.md |
| Calibration regression coverage | Known-answer metrics/temperature, argmax invariance, objective and boundary checks pass on CPU | UNIT TESTED: six new tests/test_calibration.py tests passed |
| Baseline artifact register and suite checkpoint | Three checkpoint hashes and six prediction hashes agree with calibration provenance; current suite passes | INTEGRATION VERIFIED (artifact bindings only): scripts/check_baseline_artifacts.py; docs/baseline_artifacts.json; 72/72 suite tests passed |

## Batch verification checkpoint — 2026-10-13
- `.venv/Scripts/python.exe -m unittest discover -s tests -p "test*.py" -v`:
  72 tests, 1.479 seconds, OK. CPU only, no dataset processing/training.
- `.venv/Scripts/python.exe -m scripts.check_baseline_artifacts --write-register`:
  PASS for all three views; checks checkpoint bytes, six prediction files,
  calibration code hashes and February report/prediction bindings.
- Verification initially could not launch the Windows Store Python interpreter in
  sandbox; both commands succeeded outside sandbox through the approval flow.
- Registry metrics are copied from reports, NOT independently recomputed. This
  check does not inspect NPZ row alignment; the upcoming fusion loader must do so.
- Added governance files, calibration tests, artifact checker and registry only.
  No methodology/model/source calibration changes, no future access, no commit/push.
- Tasks above meet their bounded acceptance criteria. Overall research is not
  validated; subsequent fusion batch results are recorded below.

## Remaining issues
- Raw-byte memory gate reports false under its process-commit 256 MiB criterion;
  image correctness and full generation success do not resolve that criterion.
- README and older status documents lag implementation; use dated artifacts and
  this checkpoint for current state, not their old 'not trained' statements.
- Historical near-duplicate application is done; future application and performance
  sensitivity remain. Do not reopen open-ended neighbor exploration.
- Artifact backup outside ignored data/ is not confirmed.
- Reliability episode design, fit/evaluation separation and coefficients must be
  agreed before dependent experiments. Future test remains closed.

## Next tasks
1. Review/approve the proposed reliability methodology; implementation awaits approval.
2. After approval, audit support/groups and bounded CPU cost without future data.
3. Implement approved quality-only baseline before drift/uncertainty extensions.

## Fusion batch — task 1 checkpoint (2026-10-13)
- User explicitly requested uniform calibrated fusion and historical evaluation.
- scripts/uniform_fusion.py rejects mismatched/duplicate IDs, class order, labels,
  invalid temperatures and non-finite logits before averaging probabilities.
- Nine tests including NPZ file integration passed (0.063s); fixed 1/3 weights,
  log-space averaging, labels cannot affect predictions. No future access.
- Task 1 UNIT TESTED / synthetic INTEGRATION VERIFIED. Real artifact integration
  and March comparison follow in task 2. Reliability remains design-only task 3.

## Fusion batch — task 2 checkpoint (2026-10-13)
- EXPERIMENTALLY VERIFIED: fixed uniform calibrated March evaluation, 4,332 samples.
  All three views pass canonical selection checks and exact IDs/classes/labels order.
- Artifact run: data/derived/uniform_fusion/20261013_082927_879263/.
  Durable summary: docs/uniform_fusion_results.md. Four methods' predictions and
  metrics saved; single-view accuracy/NLL/Brier/ECE agree with prior calibration.
- Uniform accuracy 0.709603, supported macro-F1 0.743854, NLL 1.107323.
  SBSMI accuracy 0.702678, macro-F1 0.727771, NLL 1.070787. Mixed outcome:
  accuracy/F1 improve; NLL/Brier worsen. No significance/generalization claim.
- Full suite: 81/81 PASS (0.649s). Initial schema-key reader failure corrected
  before output creation. No fitting/training, future access, commit or push.

## Current batch — final checkpoint
| Task | Acceptance and evidence | Status |
| --- | --- | --- |
| Uniform calibrated fusion | Strict three-view IDs/order/classes/labels; fixed probability mean; numerical and file integration tests | UNIT TESTED and INTEGRATION VERIFIED |
| Historical evaluation | Canonical March selection, pinned February T, reproduce single-view metrics, save comparison and predictions | EXPERIMENTALLY VERIFIED, not RESEARCH VALIDATED |
| Reliability protocol design | Options, Q/D/U definitions, data permissions, leakage analysis, finite outputs, explicit approval questions | PROPOSAL COMPLETE; implementation NOT AUTHORIZED |

- Task 3 document: docs/reliability_protocol_proposal.md, checked against framework
  sections 5-7 and evaluation plan. Final suite again 81/81 PASS (0.649s).
- Key unresolved methodology: honest historical encoder episodes and drift reference
  windows; existing final encoders cannot stand in for earlier-prefix models.
- Proposed staged Q/U cross-fitting, support rules, bootstrap and later full drift
  design require user approval. No reliability calculations or new training run.
- Changed this batch: scripts/uniform_fusion.py, scripts/evaluate_uniform_fusion.py,
  tests/test_uniform_fusion.py, docs/uniform_fusion_results.md,
  docs/reliability_protocol_proposal.md and the three root tracking documents.
- Final new-file whitespace check clean (Git emits only LF/CRLF conversion notices).
  All files remain uncommitted, including previous batch files. Stop after this batch.
