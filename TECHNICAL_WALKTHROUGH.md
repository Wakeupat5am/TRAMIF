# TRAMIF architecture map

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
