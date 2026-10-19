# TRAMIF architecture map

## D062 — Authorized saved-logit scoring and independent numerical readback (2026-10-18)
score_final_future consumes hash-pinned D061 receipts/manifests/15raw logit arrays,
checks frozen D060 sources/package and canonical IDs/classes/checkpoint/T bindings,
then calls final_multiseed_evaluator.evaluate_retained unchanged. No model forward,
optimizer or parameter estimation. Saves full(5,7,N,51)float64 calibrated mixture
log-probabilities,predicted labels,per-class/confusion/ECE metrics,retention/size/
pooled tables,and frozen statistical aggregates in one new immutable run.
verify_final_future_scoring independently recomputes calibration/mixtures,confusion
via bincount,Brier via norm identity,ECE bins and statistics-based aggregation.
It checks all630monthly sets,2520size cells,315pooled sets and66aggregates.
37targeted tests andreal Future readback PASS. Final report and scope limits:
docs/FINAL_FUTURE_RESULTS.md. Engineering completion does not confirm D hypothesis.

## D061 — Protected input preparation separated from final scoring (2026-10-18)
prepare_protected_future binds D060 bundle/references/dependencies,selects authorized
months,hashes current known-family bytes,applies frozen direct-earlier retention,
and creates exact SBSMI/raw-byte/entropy images. Four CPU workers;unique run only.
After full image verification,it loads each registered seed/view encoder and saves
full K51 raw logits with canonical IDs/classes/labels;eval/inference_mode,no updates,
state hashes unchanged. It does not calibrate,fit weights or call scoring functions.
verify_protected_future independently reconstructs canonical selection,checks all
exclusion witnesses/bounded KEEP queries,all image hashes/decoders and15logit arrays,
then emits input-only completion receipt. No final evaluator invoked.
12 targeted tests + real input/inference readback PASS. See D061 results report;
READY_FOR_FINAL_SCORING requires a separate authorization before scoring.

## D060 — Combined five-seed seven-arm integration (2026-10-18)
final_multiseed_evaluator validates canonical IDs/classes/labels and checkpoint/T
bindings;reuses calibrated log probabilities,uniform and frozen weighted mixtures.
It reports primary/pooled metrics,retention0/10/20 subsets and TRAIN-quartile strata.
No input-file reader,training or optimizer path exists in that numerical module.
build_combined_freeze reuses D054/D058 metadata and source byte hashes;it does not
load registered checkpoint tensors. replay_combined_freeze generates synthetic
logits bound to package identities. backup_combined_freeze restores explicit assets
then runs that synthetic replay under the original dependency environment.
freeze_retention_references adds the original26324historical TLSH rows/config/
checksum chain as a separate immutable component;no graph/pairs recomputed.
All final components are bound by a bundle index;none opens Future implicitly.
Evidence and limitations:docs/D060_PRE_FUTURE_INTEGRATION_RESULTS.md.

## D059 — Pure original-data statistics contract (2026-10-18)
final_point_statistics.aggregate_monthly accepts aligned(5seeds,7methods,6months)
F1 proportions and explicit ordered IDs. It computes each seed's Mean/Worst first,
then descriptive seed mean/SD/range,and paired differences from the same seeds.
It never reads files,loads models,resamples or calculates CIs/p-values. Candidate
metadata requires owner adoption.13 new tests plus6 existing statistics tests PASS.
Caller must independently verify sample/class/provenance alignment; this utility
does not turn the old six-method/seed42 Future consumer into a five-seed evaluator.
Exact proposal/limits:docs/FINAL_STATISTICAL_PROTOCOL.md; real integration pending.

## D058 — Version-bound learned-static production adapter (2026-10-18)
fit_final_learned_static_v2.py pins the unchanged D057 solver_v2 SHA and reuses
the verified historical logits/ID/class/full-K temperature path. All-five preflight
and targeted tests precede sequential solves. The numerical API remains historically
labeled candidate;the explicitly owner-approved adapter promotes only after
independent Decimal60 projected KKT,objective,simplex and provenance checks.
Each seed writes immutable inputs/verification/weights/COMPLETE;root source snapshots
and receipt bind all five results. Original runner and D056 remain reproducible.
No model forward or Future loader is used. Results:docs/LEARNED_STATIC_V2_RESULTS.md.

## Historical evaluation to pre-Future package (2026-10-17)
evaluate_historical_episode.py reuses pinned E1 raw C logits or explicitly exports
E2 C with correct prefix encoders; canonical IDs/classes checked before mixture.
forecast_scores uses frozen S_eval only for scoring. E1 select_beta saves an
immutable beta_selection before E2 can run; E2 never invokes selector. Six-method
probabilities, seven metrics/confusions and full forecast residuals are saved.
build_pre_future_package.py freezes existing Final checkpoint/T/Q/D and E1 beta.
evaluate_frozen_future.py is a dormant six-month seed42 point consumer with an
explicit owner flag before any reads; actual Future input builder/retention and
full five-seed statistics remain unimplemented/unapproved. No Future executed.
summarize_historical_research.py exports CSVs and Vega-Lite spec (not rendered).
backup_research_snapshot.py excludes raw malware/images/Future and verifies a
curated local restore without executing archived files. See reproduction/readiness.

## Pre-outcome QD selection lock (2026-10-17)
freeze_qd_selection_protocol.py binds completed Q and primary descriptive D and
writes immutable candidate forecasts before outcome scoring. forecasts returns
(candidate,view,family) loss predictions. select_beta is synthetic-tested equal-
family/view MAE with fixed tolerance/tie ordering; no outcome loader in this CLI.
Real selection remains unexecuted. U unavailable; details in
docs/qd_selection_protocol_lock.md. This does not create QD fusion weights.

## E1 outcome preparation (2026-10-17)
prepare_e1_outcome.frozen_plan binds G2 C_keep to E1 K32 mapping, original source
metadata and D045 frozen weights/T. inspect reads metadata/checkpoint bytes only;
extract uses reused decoder/frozen loader to persist full C classifier logits.
No metrics CLI is exposed yet. compare_fixed is a pure helper reusing strict
uniform alignment, calibrated probability averaging and existing metrics; tested
with synthetic arrays. No outcome labels enter weights or temperature fitting.

## New-Q-only static baseline freeze (2026-10-17)
freeze_historical_quality_weights.py consumes hash-pinned D044 completion and
stored losses, verifies IDs/labels/support means, and serializes named-view
softmax(-L) weights with mapping/checkpoint/calibration provenance. No prediction
evaluation or model loading. Reuses existing probability-mixture semantics as a
saved recipe; outcome execution is separate. See docs/new_quality_only_weights.md.

## Historical full-logit preparation (2026-10-17)
prepare_historical_logits.selection binds complete retained G2 A/B rows to the
A/B-only operational graph, canonical metadata and episode mapping. It is separate
from capped MMD selection. infer_logits calls the frozen classifier, not a feature
endpoint. prepare exports A, fits existing scalar-temperature solver on A only,
then exports B; no B metrics or Q/U. Image preprocessing/load_encoder are reused.
audit reads saved Final arrays/calibration and static checkpoint hashes without
model loading. New-run receipts bind input/source/config hashes. No Final refit or
automatic inference on import. Details:docs/historical_logits_calibration_preparation.md.

## Fixed U simulation runner (2026-10-16)
simulate_quality_uncertainty.py generates seven predeclared layouts and nonnegative
losses using explicit outer SeedSequence keys. It imports unchanged estimator,
shares canonical exponential draws across views/classes and saves full synthetic
losses/replicate outputs. summarize compares U with between-dataset SD and computes
delete-one-dataset jackknife precision. Statistical review triggers are report
fields, never reasons to drop a replicate or stop remaining scenarios. Numerical
failure writes FAILED.json; interrupted runs are PARTIAL. No real-data loader.
Results demonstrate important underestimation limits; see
docs/quality_uncertainty_simulation_results.md. No fusion eligibility implied.

## Conditional Quality uncertainty numerical primitive (2026-10-16)
quality_uncertainty.estimate_uncertainty accepts (3,N) calibrated NLLs and explicit
sample/family/group/support identities. For each class it aggregates sample counts
and loss sums by group, multiplies both by shared positive draws, then averages
class means equally. Active group order is lexicographic; default draws are
PCG64 seed42 exponential(rate1),2,000 replicates. Returned U is ddof1 SD across
replicate means. Covariance retains paired-view information; diagnostic quantiles
are not claimed calibrated intervals. No model, file access or fusion calculation.
Historical binding adapter and statistical simulation review remain separate gates.
Synthetic tests/evidence:docs/quality_uncertainty_synthetic_results.md.

## Sampling diagnostics (2026-10-16)
weighted_descriptive_mmd.group_weights assigns equal mass to represented groups
inside each family/month. weighted_mmd2 computes a'KAAa+b'KBBb-2a'KABb with
diagonals and fixed V1 kernel exp(-squared_distance/p/h/2). Float64 block32 direct
differences avoid Gram-distance cancellation. Input mass tolerance64eps; quadratic
kernel bounds allow4 times this tolerance without clipping kernels. Only final
negative roundoff within64eps is clipped. Larger errors fail explicitly.
check_drift_sampling_sensitivity binds exact cap64 subset and primary prepared
arrays by hashes, validates IDs/metadata/order across views, and preserves primary
normalization/h/support. Both diagnostics use equal-family outer averaging.
sample_weights.json contains diagnostic empirical weights, NOT fusion weights.
No model execution or fit occurs. Results/limitations: docs/drift_sampling_sensitivity_results.md.

## Fixed bandwidth diagnostic (2026-10-16)
check_drift_bandwidth_sensitivity.py loads pinned primary and prepared A/B arrays,
holds samples/support/normalizer fixed, runs only h/2 and2h with unchanged MMD
primitive. Saves18 aggregates/402 family scores, differences and descriptive
orders; no selected variant or replacement of primary. Tests cover known-answer
kernel scaling and zero-primary differences. Sampling diagnostics remain pending
review; see drift_sampling_sensitivity_plan.md.

## Observed descriptive Drift wrapper (2026-10-16)
scripts/run_frozen_descriptive_drift.py binds saved float64 A/B to locked sample
plans/support, frozen TRAIN parameters and original image-report file sizes.
Calls unchanged class_conditional_mmd2; saves per-family kernel means/diagonal-
inclusive MMD2, counts/groups/concentration/file-size quartiles and equal-family
aggregates. d_desc=MMD2/2 is numeric scaling only. No model/weights/null fitting.
Nine observed computations complete; original manifest self-entry defect retained
and covered by supplemental receipt; writer regression now fixed/tested.

## Frozen measurement preparation (2026-10-16)
scripts/prepare_frozen_drift_measurements.py consumes hash-pinned verified extraction
receipts and reuses G3-B TRAIN-only normalization/bandwidth APIs. Each case/view
saves float64 mean/std, boolean mask, rank/conditioning, h/pair distances/IDs and
normalized TRAIN-reference/A/B arrays with original sample order/provenance.
NPZ parameter restore and transform parity verified before applying A/B. Parameters
remain unchanged after transforms. It never calls MMD, CNN or fits on A/B.
All9 fits/27 arrays PASS; see docs/frozen_drift_preparation_results.md.

## G3-C2 runtime verified on bounded TRAIN inputs (2026-10-16)
The extractor now compares checkpoint/JSON config using trainer canonical bytes;
tuple/list serialization no longer creates a false mismatch. 9 targeted tests
and all nine trained encoders PASS on three TRAIN samples per case/view. Frozen
eval/inference_mode, endpoint dimensions, exact repeatability and full-state/BN
invariance verified. Bulk paths remain unexecuted. The older BLOCKED entry below
is historical; current receipt/commands are in g3c2_minimal_verification.md.

## G3-C2 draft extractor (BLOCKED, 2026-10-16)
scripts/frozen_encoder_extraction.py pins nine checkpoints and G3-B manifests,
reuses image decoding, freezes eval models, captures SBSMI head_norm and calls
CompactViewNet.forward_features. CPU-only; no normalization/bandwidth fitting.
Seven synthetic tests pass. Real loader blocked by tuple/list config comparison
before first forward. CLI extract is a draft and must not be used until this
defect is reviewed and bounded integration passes; see g3c2_minimal_verification.

## G3-B preparation — TRAIN-only fitting API and metadata audit
`scripts/mmd_measurement_preparation.py` has no I/O/model dependencies. Reference
sampling accepts TRAIN-dated rows only; reference_pairs hashes unordered IDs.
`fit_train_normalizer` uses equal-family population moments and frozen std mask;
`TrainNormalizer.transform` applies them without refitting. `fit_train_bandwidth`
requires matching reference and TRAIN embedding-byte hashes, scores within-family
selected pairs and takes the lower weighted median with equal family pair mass.
These fitting functions have run on synthetic arrays ONLY. Actual embeddings are
not loaded by G3-B. Existing descriptive_mmd remains the independent statistic.
`cap_audit` retains original group IDs and reports both groups_after<5 and
groups_before-groups_after>=5; it does not resolve V1 ambiguity itself.
`python -B -m scripts.audit_mmd_measurement_preparation` reads a finite hashed
historical allowlist, reuses G2/Phase1 groups, checks config/class/source identity,
and writes canonical reference/pair/matched manifests and provenance in a new run.
No graph reconstruction, inference, observed MMD or future-stage transition.
Tests: test_mmd_measurement_preparation.py and test_g3b_historical_metadata.py.
Report: docs/g3b_measurement_validation_report.md; post-cap gate remains BLOCKED.

## G3-A numerical primitive — synthetic verification only
`scripts.descriptive_mmd.biased_rbf_mmd2(X,Y,bandwidth_h=...)` accepts already
transformed matrices and returns the biased float64 RBF MMD2 plus kernel means,
raw MMD2 and a negative-roundoff flag. It includes self diagonals and computes
direct distances in tiles; it neither fits normalization nor chooses bandwidth.
`class_conditional_mmd2` accepts family-to-matrix mappings and explicit ordered
support, returning per-family results and their equal-weight mean. Missing
requested families fail; outside-support keys are ignored. Historical support
thresholds and grouping belong to future callers, not this numerical module.
No CLI, I/O, model or dataset dependency is introduced. Only synthetic tests
call this module at this checkpoint. Evidence: tests/test_descriptive_mmd.py
and docs/synthetic_mmd_verification_report.md. G3-B is not implemented here.

## G2 historical episode audit — metadata only
Entry: `python -B -m scripts.audit_historical_episodes`.
Finite allowlist verifies frozen historical manifest/index/TLSH/config hashes;
no raw metadata/binary/image/checkpoint/prediction reads and no model imports.
`select_families` recomputes prefix >=50 unique SHA/>=3 months; final51 is only
a candidate envelope. `pair_edges` exhaustively compares saved TLSH digests on
the union of episode vertices, retaining exact-content or distance<=20 edges.
`retained_decisions` applies chronological exclusion with all earlier records,
including excluded references; train keeps nonidentical near copies.
`graph_members` constructs induced train, AB and whole-episode graphs separately.
Only AB graph10 plus primary retained counts select S; C cannot bridge signal
groups. `support_table` keeps failed families in K and records explicit reasons.
Each run exports immutable episode manifests/mappings, witnesses, memberships,
source/config hashes and reports. Final51 mapping and legacy exports are untouched.
`verify_historical_episode_audit` traverses adjacency lists independently of
UnionFind to verify18 graph partitions, then checks hashes/counts/support/order.
Evidence: docs/historical_episode_feasibility_report.md. Actual image-loader
integration, prefix checkpoints and statistical independence are NOT established.
Only G2 authorized/completed; these modules do not compute Q/D/U or train models.

## Approved March paired whole-group bootstrap
Purpose: estimate conditional uncertainty in differences between five fixed methods,
without fitting temperatures, weights or networks. Entry point:
`python -B -m scripts.run_march_bootstrap`. This writes a new immutable run directory.
The runner verifies the frozen February weight chain, March predictions and audited
grouping hashes; joins groups by sample ID; verifies labels and class order; and
reproduces original point metrics before drawing anything.

In paired_group_bootstrap.py, sample_metrics produces per-sample accuracy indicators,
true-class negative log probabilities and Brier scores summed over all 51 classes.
aggregate_groups sums these values within each of the 2,234 active components.
For group g, n_g is its retained sample count and A_gvm is its sum for method v and
metric m. A draw b supplies integer multiplicity k_bg for every group, shared by
ALL methods. resampled_metrics computes:

`theta_bvm = sum_g(k_bg * A_gvm) / sum_g(k_bg * n_g)`.

Thus selecting a group twice repeats every member twice, and the denominator is
the actual draw size. draw_multiplicities uses NumPy default_rng/PCG64, seed 42,
and Multinomial(G, [1/G]*G), equivalent to G whole-group selections with replacement.
paired_differences subtracts each reference from quality-only on the same draw.
percentile_interval takes the 2.5th/97.5th percentiles with linear interpolation.
These additive metrics remain defined when a class is absent; all draws are kept.
No Macro-F1, balanced-accuracy or ECE interval is calculated in this scope.

The runner checks group sums against explicitly repeated shared sample indices;
saves every multiplicity, sample/group mapping, actual draw size, metrics/differences,
configuration and hashes. influence_diagnostics computes descriptive leave-one-group-
out point differences and multiplicity associations, never modifies the main draws.
Tests: tests/test_paired_group_bootstrap.py (11); complete suite 120/120 PASS.
Evidence: docs/march_paired_bootstrap_results.md. Not a new training experiment or
final holdout evaluation; frozen-input conditional uncertainty only.

## March metrics and support-only pilot
evaluate_quality_march.py verifies the locked February weight chain, canonical
March dataset and inference temperatures, then compares five methods from saved
logits. It first validates three-view alignment, mixes probabilities using exact
frozen weights, and reproduces the existing four-method artifact. Predictions,
confusion matrices, ECE bins, all seven metrics and hashes are saved in a new run.
No temperatures, model parameters or mixture weights are fitted here.
march_group_feasibility.py uses only March inventory records for graph construction.
All original March records can connect components; only retained IDs contribute
to the group-by-class count matrix. Multinomial group multiplicities produce 500
support-only pilot draws. Each row preserves mixed-label membership. The saved
missing mask refers to the ORIGINAL 47 supported classes; it never changes the
metric denominator. No metric deltas, confidence intervals or p-values are computed
for these pilot draws. The frozen February implementation/artifacts remain unchanged.

## Quality-only cross-fitting and freeze
crossfit_february.py loads canonical February logits from the three pinned models,
verifies policy/checkpoint/class provenance, then calls quality_fusion.cross_fit.
aligned_folds joins the common manifest by ID. For score fold f, fit T on all
other folds with the existing standard-NLL solver; emit 51 log probabilities.
Every sample is scored once per view. No CNN or March input is required.
freeze_quality_weights.py verifies OOF hashes and reproduces probabilities from
saved temperatures without refitting. It recomputes support from the fold manifest,
averages sample NLL within each of S's classes, then averages those class means.
softmax(-L) yields three global weights. Full-February inference temperatures,
models, fold/support manifests and February provenance are bound in weights.json.
check_quality_freeze.py is the read-only resume gate: validates checksums, source
chain, copied manifests and exact numerical reproduction from saved OOF artifacts.
It detects changes to weights/folds and does not refit temperatures. A changed
hash requires investigation; do not simply replace the expected checksum.
Current research boundary: quality-only March evaluation and the approved additive-
metric bootstrap are complete. Broader reliability and future evaluation await approval.

## Conditional February grouping audit
scripts/february_group_audit.py reads pinned historical manifests and TLSH inventory,
filters February before graph construction, and compares every February pair.
Union-find forms content-equality/TLSH<=10 components, including excluded bridge
vertices. Only retained members enter five deterministic, label-stratified folds.
class_support.csv uses retained sample and component counts; fold_summary.csv
checks fit/score isolation. groups.csv, members.csv and edges.csv expose transitive
and mixed-label relationships. report.json records input/code/artifact hashes.
No logits, checkpoint deserialization, calibration or binary reading in this step.

## Implemented historical path
Metadata -> train-only family cohort -> train/validation manifest -> disarmed
binary bytes -> three representations -> separate models -> saved logits ->
February temperature fit -> March forward validation.

- inspect_metadata.py chooses eligible families; build_train_val_manifest.py
  applies dates and the frozen cohort. docs/class_mapping.json orders 51 classes.
- sbsmi_stream.py counts adjacent six-bit states, row-normalizes and floors to
  uint8 pixels (PNG). The final partial state is retained as an integer.
- raw_byte_stream.py performs masked area aggregation from width-256 byte layout;
  entropy_stream.py computes overlapping local entropy (window 256, stride 128),
  entropy_image_stream.py aggregates it. These views are float32 NPY arrays.
- build_*_dataset.py writes representations and provenance reports. Raw/entropy
  builders use prior view reports to verify identical source content.
- SBSMIDataset / FloatImageDataset decode images. HistoricalPolicyDataset applies
  the shared retained manifest. Train remains 18,061; primary validation is 8,131.
- MalSBSLCNet is the SBSMI branch; two independent CompactViewNet instances handle
  raw/entropy. All accept [B,1,64,64] and output [B,51] logits, not probabilities.
- train_sbsmi.py / train_float_view.py train and save per-epoch checkpoints.
  Dataset generation supports reuse; training does not implement full resume.
- evaluate_sbsmi_filtered.py / evaluate_float_view.py save predictions and reports.
- calibrate_filtered_view.py verifies prediction provenance and reuses
  fit_temperature/probability_metrics from calibrate_sbsmi.py. Positive scalar T
  rescales logits before softmax; it preserves predicted class ordering.

## BALANCED Phase 1: counts-only feasibility (2026-10-14)
Entry point: python -B -m scripts.run_balanced_support_audit. It reads saved
February/March inventory, primary retained manifests, frozen artifact checksums,
three checkpoint bytes and six prediction identity arrays. No torch/model import,
binary/image loading, MMD, uncertainty draws, inference or future-data processing.
Preprocessing identity is canonical sorted-JSON SHA, not raw config-file SHA;
the runner verifies the former against evaluations and records the latter too.

balanced_support_audit.build_joint_graph compares all historical unordered pairs
and uses UnionFind on exact content equality OR TLSH <=10. Excluded vertices can
connect components but never re-enter retained support. Three partitions record
joint, retained-only and month-local graphs, enabling comparison with legacy local
partitions without replacing them. class_support reports every original class and
candidate >=20 samples/>=5 joint groups in both months. Counts are based on retained
samples; labels report support/mixed groups but do not decide graph edges.

equal_side_capacity uses exact subset-sum bits to find the largest balanced
subsample budget available from two whole-group sides for one class/month.
allocation_witness builds ONE greedy shared assignment per month, respecting
multi-label groups. These inspect counts, not null probabilities or MMD. The two
month assignments are independent within-month witnesses, not a cross-month split.
Neither capacity nor a witness proves null exchangeability or adequate test power.

Output: balanced_phase1_audit/20261013_233825_921873 under data/derived; source
hashes, graph edges/members/groups, all-class support, witness, layer consistency,
scope snapshot and checksums. 25-family S* is a candidate SIGNAL support set, not a
new 25-class classifier. Existing 51-class mapping and primary selection unchanged.
Tests: test_balanced_support_audit.py (10); full suite 130 PASS. Scope/results:
docs/balanced_research_scope.md and docs/balanced_phase1_results.md. Next dependency
is methodology approval, especially null dependence and small per-side budgets.

## Not yet implemented as a complete research pipeline
Drift/uncertainty-weighted reliability fusion; multi-seed and future-test comparisons.
Quality-only fusion and its bounded March uncertainty are implemented above.
TLSH is duplicate control, not a drift
estimator. Temperature calibration is not reliability estimation.

## Evidence levels
Tests validate numerical/edge-case behavior; pilot reports validate integration;
checkpoints and metric artifacts document experiments. Research claims additionally
need valid comparisons and the frozen evaluation protocol. See PROJECT_PROGRESS.md.

## Artifact register
check_baseline_artifacts.py checks the three explicitly selected historical runs
without importing torch or deserializing checkpoints. Default mode only reads;
--write-register updates docs/baseline_artifacts.json after every check passes.
It checks hashes and bindings, not sample ordering inside NPZ or recalculated
metrics. Fusion must verify those independently before combining predictions.

## Uniform fusion
uniform_fusion.py validates three complete ID/class/label sequences before mixing.
For each view, compute log-softmax(logits / February temperature); logaddexp across
views minus log(3) gives the fixed probability mean without underflow in NLL.
classification_metrics computes accuracy, supported/fixed macro-F1, balanced
accuracy, NLL, multiclass Brier and 15-bin top-label ECE. Labels do not enter weights.

evaluate_uniform_fusion.py pins the baseline register and historical policy; it
reuses load_filtered_predictions to verify both months against canonical manifests,
then checks temperature/checkpoint bindings and compares the three March sequences.
It recomputes calibrated single-view scores and requires agreement with saved
calibration reports before writing four-method predictions and metrics in a new run.
No models are loaded and no optimizer is invoked. Scope is March development only.

Reliability design is kept separately in docs/reliability_protocol_proposal.md;
its formulas and candidate parameters are not executable configuration. The key
dependency is honest held-out historical windows, not an additional fusion layer.

## G3-C-I addition — Prefix training adapter and execution boundary (2026-10-15)

`episode_training_dataset.EpisodeTrainingDataset` reads hash-pinned G2 TRAIN
metadata before constructing any image path. A TrainingSample keeps original
Final51 storage index separate from the E1/E2 target. Only the target reaches
CrossEntropyLoss. All paths to images/reports derive from validated TRAIN IDs;
decoders preserve PNG /255 and already-normalized NPY semantics.

`episode_training_dataset.preflight` reads three views with no model and emits
ordered image/report receipts bound to code/config/mapping hashes. This has been
tested only with synthetic image fixtures; metadata-only real bindings verified.
`train_prefix_view.verify_preflight` rejects stale/missing/different-case receipts;
each training read checks exact receipt bytes. Training is a separate explicit CLI
path, never triggered by import, environment inspection, tests or preflight.

The new trainer reuses legacy model classes and fixed epoch100 policy; saves
source/provenance, initial state and per-epoch verified checkpoints. Tests run CPU
random-model eval only, zero optimizer steps; actual GPU loop remains unverified.
Report: docs/g3ci_prefix_training_implementation_report.md. D026 records the gate.
# D052 additive multi-seed path (2026-10-17)

`train_final_seed.py` wraps the existing primary TRAIN loader and unchanged CNNs.
Its narrow43–46 seed allowlist, hash-pinned metadata preflight and exclusive run
directories prevent warm starts or legacy overwrite. It validates installed Adam
defaults without a step during tests;100 fixed epochs write atomic checked
checkpoints and explicit COMPLETE/FAILED receipts. No resume/automatic latest.

`process_final_seed.py` only loads verified selected-seed epoch100. Existing
logit/feature endpoints and historical manifests feed A-only temperature, B/S
class-balanced Q and TRAIN-only normalization/bandwidth for descriptive D.
Shared frozen matched IDs retain seed42's sampling namespace, but statistics
are recomputed for each encoder. Receipt joins compare IDs/class/content hashes
before `finalize` saves Q and beta1 QD. No C/Future stage, no U penalty.
Real execution remains separate from synthetic integration evidence.

`learned_static_solver.py` is a pure proposed convex numerical baseline; no I/O
or empirical fit CLI. `multiseed_statistics.py` provides fixed-support Macro-F1,
per-seed mean/worst aggregation and shared whole-group/seed draw primitives.
Arithmetic tests do not establish resampling assumptions or coverage.

`automation/final_multiseed.ps1` separates preflight/train/verify/post and fails
before execution without the corresponding operator acknowledgement. A flag is
not owner authorization. `verify_pre_future_engineering.py` reproduces only new
targeted CPU/synthetic tests and an optional curated local source addendum.
