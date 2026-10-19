# Decision log

## D062 — Owner authorizes final scoring; preserve weak incremental-D result (2026-10-18)
Owner explicitly approves complete D062 under frozen D060 bundle/statistical protocol,
with no methodology/frozen artifact changes;STOP for serious errors requiring change.
Execute CPU consumer of D061 fullK51 raw logits;reuse frozen seven-arm evaluator,
all5seeds,retention0/10/20,TRAIN-size bins and exact Mean/Worst aggregation.
No fresh CNN/inference/fits/weight choices. Udiagnostic,Ddescriptive,beta1/gamma0.
37targeted tests PASS;independent raw-logit/probability/metric/aggregation and CSV
readback PASS. No serious mismatch,empirical retry or protocol decision needed.
Receipt:data/derived/final_future_scoring_v1/20261018_141835_165405/receipt.json;SHA:53c3c885851136549eacb8756ccd3c44debec3fea5e24bc722efb6a730840bc6.
Report:docs/FINAL_FUTURE_RESULTS.md. Primary QD-Q Mean+0.0000789850145024 and
Worst+0.000171204140653;Mean target .02 not met. Retain all weak/negative findings,
including seeds43/44 Mean deltas and historical E2. Do not retune to rescue D.
Approved reduced fixed-horizon experiment completed;no claim of full framework,
population significance,future-year generalization or independence. No follow-on
experiment/statistical procedure implicitly approved;no commit/push.

## D061 — Authorize protected Future preparation only; stop before scoring (2026-10-18)
Owner attachment explicitly opens April–September input discovery,retention0/10/20,
three-view preparation and registered15-model frozen inference under D060 bundle.
No scoring/comparison/statistical evaluation or parameter fitting authorized.
Execution complete:data/derived/protected_future_inputs_v1/20261018_082841_011123/;receipt SHA:116f994f5b3bf4372de248fe34b855e01c87f037955795a8156fa34d433ac771.
12 targeted tests PASS;real-data independent input/logit readback PASS:20,363primary
samples,61,089images,15K51arrays. All20,053 exclusion witnesses and36 bounded KEEP
queries checked;all six months nonempty. Current/source snapshots match.
Labels describe frozen cohort/mapping;similarity does not use labels. Sensitivity
sets nested;no new grouping or selection methodology introduced. Earlier failed
discovery attempts retained;no actual Future metadata repaired/excluded to pass.
Missing original-ZIP CRC authentication and off-device backup limitations disclosed.
Status READY_FOR_FINAL_SCORING is engineering readiness only;separate explicit owner
approval required for D062. No final metrics,retuning,training,commit or push.

## D060 — Adopt D059 with required file-size analysis; integration authorized (2026-10-18)
Owner approves O1/O2/O4 and O3 WITH MODIFICATION:retain retention10/20 sensitivity
and require TRAIN-quartile file-size-stratified descriptive reporting. Record
docs/D060_O3_SCOPE_AMENDMENT.md before changing the specification hash; preserve
immutable D059 snapshot. Same two outcomes,five seeds,seven arms,eight contrasts;
no CIs/p-values/significance,Udiagnostic,Ddescriptive,beta1/gamma0 unchanged.
Owner authorizes synthetic evaluator/retention/size tests,combined artifact freeze,
reproduction and backup/restore. No Future access,training/refit/retuning or Git writes.
Execution:44distinct targeted tests PASS;five-seed/seven-arm synthetic replay plus
independent630CM readback PASS. Combined freeze and reference addendum PASS.
Size cutoffs86414/254672/721840 bytes from18061TRAIN samples only. One adapter
summary-status mismatch fixed with regression,original failure retained.
Local1872-file/15checkpoint backup restored;replay matches;same-machine limitation
explicit. Reference inventories separately frozen/restored,not graphs recomputed.
See docs/D060_PRE_FUTURE_INTEGRATION_RESULTS.md;no new scientific choice pending
for reduced study. Real Future input/inference integration remains unperformed and
separately protected;no implicit Future authorization from engineering PASS.

## D059 — Review final statistics; proposed scope awaits owner adoption (2026-10-18)
Owner authorizes analysis,one authoritative specification and bounded synthetic
verification; this does NOT approve replacing V1 statistical methodology.
Recommendation in docs/FINAL_STATISTICAL_PROTOCOL.md:preserve both V1 estimands,
five seeds/fullK51/monthly support;seven available arms;family8 contrasts with
QD-NewQ primary for Mean and Worst. No QU/QDU claims or best-seed selection.
Do not run group-by-seed coverage experiment with an unjustified DGP merely to
obtain PASS. Percentile CI coverage NOT ESTABLISHED; recommend original-data
estimates and seed spread without CIs/p-values/significance. This requires explicit
amendment of framework/V1 interval and scope requirements. O1-O4 consolidate all
requested owner decisions,including practical target and deferred controls.
19/19 targeted synthetic tests PASS; no coverage/real-data validation inferred.
Original V1,framework,D057 solver,D058 adapter hashes unchanged; legacy artifacts
preserved. No Future/training/inference/new fit/retuning/commit/push.
After owner adoption only:combined package/integration/backup; Future remains a
separate approval. D059 is REVIEW_COMPLETE_AWAITING_OWNER,not PRE_FUTURE_READY.

## D058 — Adopt D057 arithmetic and execute five-seed learned-static batch (2026-10-18)
Owner explicitly approves exact solver_v2 for production and one sequential CPU
batch42-46. Keep objective/cohort/K51/own February T/uniform/projection/Armijo,
max10000 and residual1e-10;STOP on failure without retry or retuning.
Add production adapter with new source binding;preserve D052/D056/D057 bytes,
candidate metadata and historical receipts. Record present adoption separately.
38 targeted tests+all-five input/provenance preflight PASS before fitting.
5/5 convergence and independent Decimal60 KKT residual<=1e-10,objective,
simplex/source/input checks PASS before per-seed freeze. No errors/retries.
Entropy zero for45/46 is an allowed verified boundary optimum;all seeds retained.
Receipts and full precision:docs/LEARNED_STATIC_V2_RESULTS.md.
This completes the historical learned-static fit component,not its held-out
evaluation or full V1 research validation. No same-B superiority/significance claim.
No further methodology/statistical/Future execution implicitly approved;no CNN,
inference,T refit,Q/QD/U changes,commit or push. Next:existing statistics/scope
owner decisions before combined final freeze and separate Future permission.

## D057 — Diagnose numerical failure; isolated arithmetic correction, no new production fit (2026-10-18)
Owner authorizes saved-input diagnostic reproduction,independent mathematical
checks,defensible candidate correction and targeted tests. No registered empirical
fitting or freezing. D056 exact trajectory reproduced; Decimal70 establishes
normal-direction mass error dominates tangent descent at iteration26. Original
gradient,projection and Armijo formulas are mathematically valid; floating storage
and absolute objective comparison prevent correct constrained descent near optimum.
Implement separate D057-affine-simplex-armijo-v2-candidate: implicit largest simplex
coordinate,centered gradient,compensated sums and direct Armijo log1p remainder.
Keep uniform,float64,projection,step1,halving,c1e-4,60backtracks,10000iterations,
residual1e-10. No slack,new optimization algorithm,regularizer or near-solution freeze.
Mathematical protocol unchanged; changed arithmetic/source version is PROPOSED,
NOT APPROVED for registered fitting. Do not replace D052 source pin silently.
31/31 targeted tests PASS; six controlled objectives independently checked; one
corrected step at saved failure agrees with Decimal. No full corrected real solve.
All D056 files/original source hashes preserved. See numerical diagnosis report.
Next owner decision:adopt versioned correction and authorize integration plus new
sequential five-seed CPU fit with unchanged STOP rules. No automatic retry now.
No Future,CNN,inference,T refit,Q/D/U,performance selection,commit or push.

## D056 — Explicit learned-static execution approval; stop on Armijo failure (2026-10-18)
Owner explicitly approves numerical protocol and CPU B/S fitting for all five
Final seeds,with mandatory STOP on nonconvergence/provenance error and no solver,
tolerance or performance-based changes. This resolves D055's approval ambiguity.
Execute one unchanged-solver invocation. Seed42 stops at iteration26 with
Armijo failure,residual1.9771081438690885e-9 above1e-10;no weights registered.
Seeds43-46 not attempted. No retry/fallback,source/frozen artifact changes,
CNN/inference/T refit/MMD/Future or Q/QD retuning. Preserve diagnostic candidate
as unregistered only;do not count it as completed baseline. Report/evidence in
docs/LEARNED_STATIC_FIT_RESULTS.md. Next requires numerical failure review and
owner decision before another empirical attempt;approval is not permission to
silently relax convergence criteria.

## D055 — Learned-static runner ready; explicit fitting approval required (2026-10-18)
Owner said "ok tiep tuc" after the proposed learned-static next step. Implemented
the bounded CPU runner using unchanged D052 projected-gradient solver and D054
verified artifacts;11 targeted tests PASS. Objective/cohort/temperature/coefficients
unchanged. No inference,CNN training,T refit,MMD or Future access.
Automatic approval review rejected the empirical-fit command before execution:
generic continuation and an approval CLI flag do not establish the separate owner
numerical-protocol approval required by the existing boundary. Do not bypass or
retry by another route. No empirical fit/weights are registered. Await explicit
owner approval for the published uniform-start,Armijo/simplex,10000-iteration,
1e-10 projected-residual protocol on all five historical B/S sets. Convergence
failure must STOP without tolerance changes or approximate weight registration.
New files:scripts/fit_final_learned_static.py,tests/test_fit_final_learned_static.py.

## D054 — Verify five-seed artifacts without reopening experiments (2026-10-17)
Owner authorizes readback/verification and descriptive consolidation only. Accept
registered legacy seed42 contract without imposing new COMPLETE schema;new43-46
runs use their saved COMPLETE/source/config/payload contracts.15/15 checkpoints
and5/5 historical pipelines are ARTIFACT_VERIFIED,not RESEARCH_VALIDATED.
Use explicit canonical ID joins for legacy prediction ordering;never pair unchecked
positions. Recorded training/processing snapshots match current contracts;no source
or frozen hash repair. Recompute only verification arithmetic:Q,TRAIN moments/pair
distances,stored MMD-term aggregation and fixed softmax weights. No T/kernel refit.
Retain every seed42-46. Per-view descriptive mean/sampleSD(ddof1)/min/max across
five seeds do not establish independent temporal replication or significance.
beta1,gamma0,U excluded and E2 negative fusion evidence remain unchanged.
Three initial STOP receipts record verifier legacy-adaptation bugs,not failed
training/artifacts;regression tests and final independent readback PASS.
No training/inference,retry,learned-static fit,real bootstrap,Future or commit/push.
Missing12-seed checkpoint blocker is resolved. Other scientific/scope/Future gates
remain;do not mark full V1 or research hypothesis complete. See D054 reports.

## D053 — Final multi-seed execution approved for the operator (2026-10-17)
Owner approves12 Final CNN runs:seeds43,44,45,46 x SBSMI/raw-byte/entropy,
unchanged architecture,hyperparameters,frozen TRAIN cohort,K51 and fixed epoch100.
Owner alone starts sequential PowerShell training; Codex must not start training
or poll epochs. No parallel runs,automatic retry,resume or performance-based seed
exclusion. Preserve existing seed42 and all frozen contracts.
After ALL THREE checkpoints for a seed pass verification, owner may explicitly
start that seed's historical postprocessing:February A-only temperature,March B Q,
seed-specific TRAIN normalization/bandwidth,February-March descriptive MMD and
Q/QD weights with beta1,gamma0. Never reuse another seed's temperatures/encoder
statistics; U remains diagnostic-only. No best-seed selection or beta retuning.
Empirical learned-static fitting still needs separate numerical-protocol approval;
Future April-September remains closed. No commit/push.
D052 receipt and12 relevant source/test/wrapper/sheet hashes verified unchanged
before this documentation-only approval update; wrapper parser PASS. Reuse the
25-test synthetic receipt without rerunning tests. No real-data integration PASS
is claimed. Current state:READY_FOR_OPERATOR_EXECUTION; await operator outputs.

## D052 — Authorized pre-Future engineering; scientific boundaries retained (2026-10-17)
Owner says to incorporate the prompt refinements and execute independent work.
Use the latest continuation state, preserve completed E1/E2 and seed42; add
Final43–46 trainer and seed-specific historical processing, not12 GPU runs or
actual postprocessing. Each seed needs its own T/Q/TRAIN normalizer/bandwidth/D;
same frozen IDs/S,beta1,gamma0. No best-seed or outcome-driven retuning.
Uniform-start projected-gradient learned-static solver and V1 statistical consumer
are synthetic-tested PROPOSALS; no empirical fitting/resampling/CI authorization.
Consolidate real scientific decisions once in docs/PRE_FUTURE_OWNER_DECISIONS.md.
CPU synthetic benchmark is model-only; missing renderer does not justify installing
dependencies. Full V1 not ready; no silent reduction to seed42-only or dropped U
contrasts/multiplicity change. Separate owner resource, methodology and Future
approval remain required.25 focused tests are engineering evidence, not research
validation. New local addendum is not off-device/raw-input backup.

## D051 — Historical-only pre-Future package and controlled stop (2026-10-17)
Autonomous request authorizes independent synthesis, Final component freeze,
reproduction/paper preparation and dormant evaluator, not Future data. Freeze
seed42 Final QD with E1 beta1; actual Future execution requires separate approval.
Readiness is BLOCKED for full V1: missing four seeds, learned-static solver,
statistics/completeness and controls. No silent scope downgrade or claimed full
framework completion. U remains excluded. Local snapshot/restore is curated,
does not establish independent off-device backup. See readiness/results reports.

## D050 — E2 held-forward result, no retuning (2026-10-17)
Execute authorized CPU inference with correct E2 K42 checkpoint and February C
manifest N3563. Use E1 beta1 verified before reading E2 outcome; selector disabled
in E2 code/tests. Forecast MAE improves slightly .563763→.561424, but QD F1 drops
.683443→.683434 and NLL/Brier worsen versus Q. Keep beta and negative result.
Episodes overlap; no independent-replication, significance or Future claim.

## D049 — E1 selection/evaluation authorized by autonomous completion request (2026-10-17)
Owner explicitly opens historical E1-C scoring and E2 follow-on work. Frozen D048
scores choose beta1 (MAE .469168 vs .471385 for beta0); E1 selection saved before
E2. Evaluate single views/Uniform/NewQ/selectedQD on identical N3695/K32 cohort.
E1 selected-QD metrics are development selection, not independent test evidence.
No grid/preprocessing/T/support changes, no U penalty or new statistics. Store all
candidate residuals, probabilities, metrics and hashes. Verification PASS.

## D048 — Approved V1-derived descriptive QD selection lock (2026-10-17)
Owner accepted the proposal to lock D selection before opening E1 outcome scores.
Use V1 beta{0,.5,1} and explicit fallback d=primary MMD2/2; a1,gamma0 because U
has been excluded by owner. Three QD candidates replace the proposed nine QDU
grid only within this reduced scope. Selection uses equal-family/equal-view
forecast MAE on E1; no fusion-accuracy selection, no outcome-driven grid changes.
Freeze beta before E2; beta0 valid. This records later authorization relative to
Amendment01's earlier unapproved grid state; no standardized-D interpretation.
5 tests and saved forecasts PASS. No actual selection/QD weights/C metrics yet.
Evidence and next bounded E1 evaluation scope:docs/qd_selection_protocol_lock.md.

## D047 — E1-C export executed; outcome scoring still closed (2026-10-17)
Owner continued the explicit extraction step and requested teaching WHY/meaning/
consequences before each action. Run frozen CPU inference exactly once for3 E1
views on3695 retained C samples. K32 logits/export integrity PASS; no metrics read,
T refit or fusion weights changed. Existing V1§8.2 selection dependency remains;
do not infer coefficient/methodology approval from extraction authorization.
Evidence:docs/e1_outcome_preparation.md and032743_694605 receipt/readback.

## D046 — Prepare E1-C without opening outcome metrics (2026-10-17)
Owner continued the proposed E1 comparison preparation. Implement CPU raw-logit
export using frozen E1 models and synthetic-tested five-method comparison helper.
Use all frozen primary C3695/K32, not Q support subset. Metadata checks PASS;
5 tests PASS. No actual C inference/scoring performed. Preserve V1 §8.2 requirement
to lock coefficient selection before E1 outcome losses; no implicit authority to
choose D/U or rewrite previously observed outcomes. Exact owner-run extraction
command and existing dependency documented in docs/e1_outcome_preparation.md.

## D045 — New Quality-only fixed rule frozen (2026-10-17)
Owner said continue after D044's proposed New-Q-only baseline. Execute the stated
V1 New-Q rule softmax(-L), a=1, beta=gamma=0, from frozen B support losses for
E1/E2/Final. No coefficient selection, implicit U use or legacy February freeze
replacement.5 tests and saved-source/weight verification PASS; output paths and
full-precision weights documented in docs/new_quality_only_weights.md. This is
not approval to access C/Future or evidence of fusion improvement. U remains
diagnostic-only. No outcome predictions read and no inference/training/commit/push.

## D044 — Historical Q executed, weights not authorized (2026-10-17)
Owner approved one finite CPU batch computing actual Q for E1/E2/Final from saved
B logits with fixed A temperatures. Keep frozen S17/25/25, primary retention,
K32/42/51 softmax and sample-level means within families. Q=-L; L is equal-family
NLL. Also preserve Std-S and Std-all estimands. No capped Drift inputs or legacy
helper that implicitly creates weights.5 tests PASS;9 results and immutable
per-sample/per-family artifacts independently verified. Evidence in
docs/historical_quality_results.md. Lower observed SBSMI loss is not a future
reliability claim. U remains diagnostic-only. No weight/coefficients/C/Future use,
training/inference/refit, commit/push or methodology change.

## D043 — Owner-completed prefix logits/calibration verified (2026-10-17)
Owner ran E1 then E2 preparation. Both PASS receipts, output/source hashes,
K32/K42 mappings, checkpoint bindings and A-only fit identities verified. No
temperature is at bounds. Final existing calibration remains unchanged. No
inference repeated by this verification. Historical Q input preparation complete;
propose B-only Quality computation on frozen support as next bounded task. No
empirical U or fusion coefficients authorized. See historical preparation report.

## D042 — U diagnostic-only; historical prediction preparation (2026-10-17)
Owner accepts D041 negative evidence without estimator repair or synthetic search.
Authorize artifact inspection and missing logit-export implementation/tests, not
bulk inference. U remains excluded from fusion. Reuse Final registered filtered
logits/February temperatures. Prefix A/B must use its own K32/K42 checkpoints and
full retained manifests, not capped embeddings or Final outputs. Fit T on A only;
future Q uses full retained B in frozen S. New CPU preparation CLI passes6 tests;
all historical static bindings verified. E1/E2 execution remains with owner.
No new scientific gate introduced for file checks. See historical preparation report.

## D041 — Approved G5 simulation complete; U fusion blocked (2026-10-16)
Owner explicitly approved D040 plan implementation/execution in one finite batch.
Ran exactly2,100 primary fits and210 secondary fits on synthetic data; original
estimator/formula/seeds/layouts unchanged.9 targeted tests PASS, immutable outputs
verified. Statistical triggers did not abort execution or cause retries.
12/21 mean-U/empirical-SD ratios below.8, including independent unequal/dominant
group cases; none above1.25. All seed43 diagnostics below threshold. Engineering
PASS does not establish statistical validity. Record U_NOT_APPROVED_FOR_FUSION;
no weights/coefficients or empirical U authorized. See simulation results report.
Recommend bounded historical input readiness and owner review of diagnostic-only
U; do not add synthetic searches to rescue the method. No commit/push.

## D040 — Finite U simulation design, execution pending (2026-10-16)
Owner requested the proposed simulation design after D039. Record exact scenarios,
loss generator, streams, budgets, estimands and reporting rules in
docs/quality_uncertainty_simulation_plan.md. No scenario search or real loss access.
Keep sample-level class means and common operational-group draws. Compare U with
independent outer-dataset spread; do not infer real-data coverage or independence.
Proposed ratio review tolerances require approval; no automatic methodology gate
closure. Simulation implementation/execution and real historical adapters remain
separate from this documentation-only authorization.

## D039 — Synthetic Quality uncertainty estimator accepted for implementation (2026-10-16)
Owner approved the explicit pure synthetic step after D038. Implement fixed
sample-level equal-family loss, shared exponential group weights, default
2,000/PCG64/seed42 and ddof1 SD. Explicit supplied draws are for testing/replay,
not authorization to alter the historical protocol. Preserve operational group
IDs across classes/views; do not substitute equal-group sample mass from MMD.
10 tests and synthetic save/replay PASS; no real Q/U or weighting computed.
No statistical validity or coefficient gate is closed. Next proposed task is a
finite simulation design review. Evidence:docs/quality_uncertainty_synthetic_results.md.

## D038 — U design review, methodology acceptance pending (2026-10-16)
Owner requested continuation after the D037 batch and proposed U design gate.
Document V1-compatible estimator and input/approval boundaries in
docs/historical_quality_uncertainty_design.md; no automatic estimator approval.
Separate all-retained-B Quality target from capped Drift samples. Existing
February OOF fusion helper must not silently define new episode Q/U semantics.
Do not invent a substitute for V1's coefficient-dependent dominance diagnostic:
it remains pending until coefficient/grid and recomputation rules are approved.
Recommend pure synthetic implementation first; empirical U and negative-penalty
reliability claims remain unvalidated. No new losses, U or weights computed.

## D037 — Approved sampling diagnostics completed (2026-10-16)
Owner's “ok làm đi” accepts the concrete D036 plan: cap64 sampler review accepted
PASS_WITH_LIMITATIONS for E1/E2/Final, and same-primary-sample group-balanced
biased MMD. Keep seed42, frozen subset/support, TRAIN normalization and primary h.
Each selected sample receives1/(represented groups * members in its group),
separately per family/month; shared and multi-label group IDs remain unchanged.
Outer aggregation remains equal-family. This changes the diagnostic empirical
target, not the primary analysis, and supplies no independence/null guarantee.
All18 computations retained; no favorable arm selected.7 tests and saved-artifact
verification PASS. E2 descriptive view order changes under group balancing.
No fusion coefficients or weights approved by this decision. See
docs/drift_sampling_sensitivity_results.md. Stop after this bounded batch.

## D036 — Cap64 focused review recommendation, not gate approval (2026-10-16)
Owner authorized review after D035, not an unspecified changed estimator. Exact
metadata/subset integrity checked; concentration/shared/multi-label incidence
quantified with denominators explicit. Recommend cap64 PASS_WITH_LIMITATIONS for
E1/E2/Final; owner acceptance remains pending. Proposed group-balanced diagnostic
uses same cap128 samples with equal represented-group mass within family/month,
biased weighted MMD with diagonals and equal-family outer mean. No resampling,
new cutoff or primary replacement. Detailed formula/tests and finite18-computation
plan documented in docs/cap64_focused_review.md. No execution/automatic next gate.

## D035 — Fixed bandwidth sensitivity; cap64 review trigger (2026-10-16)
Owner requested continuation after descriptive Drift; execute the explicitly
predeclared h/2,2h diagnostic on unchanged primary data.18 scores and402 family
outputs saved; all results retained, no winner chosen. Metadata-only cap64 uses
same hash ranking/seed, nested subset; no below5 support failure, but lost>=5
triggers require review. Do not treat D023 primary128 acceptance as cap64 approval.
Group-balanced estimator is underspecified in V1; propose equal represented-group
mass within each family/month on same primary samples, not new sampling. This
changes the empirical target and needs owner approval before code/execution.
No coefficients, Q/U or Future access. Two targeted diagnostic tests PASS.

## D034 — Observed descriptive MMD; output-manifest limitation (2026-10-16)
Owner authorized next described step, computing historical A/B descriptive
class-conditional MMD only. All9 computations/201 family outputs completed with
locked support, kernel and TRAIN parameters. No weights or significance claims.
Readback found writer included its still-empty checksums.json self-hash. Preserve
original outputs, document invalid self-entry, verify report/CSV hashes and
aggregation/bounds separately, save supplemental receipt. Fix/test writer for
future use; no rerun to replace immutable artifacts.3 targeted tests PASS.
This is technical measurement completion with explicit manifest limitation, not
research validation, sensitivity completion or permission to choose coefficients.

## D033 — Owner-authorized TRAIN-only fit and A/B transform (2026-10-16)
Owner requested continuation after saved embedding verification; bounded scope
announced: fit TRAIN-only normalization/bandwidth and transform A/B, no observed
MMD. Reused existing G3-B numerical utilities unchanged. All9 case/view fits PASS;
no masked dimensions, full rank, finite positive bandwidths and finite transformed
arrays. No epsilon rescue, altered sampler, fit from A/B or coefficient changes.
One new serialization/no-refit test PASS and independent output integrity checks
PASS. These are technical checks, not evidence for null standardization or Drift
predicting future error. MMD/signals remain next separately bounded work.

## D032 — Owner-run extraction artifacts verified (2026-10-16)
Owner authorized checking nine existing runs only. Reused frozen G3-B plans and
G3-C2 checkpoint registry, no resampling or graph rebuild. Independent saved-array
checks verify27 NPY files and receipt hashes, IDs/order/mapping, finite float32,
checkpoint/source bindings and same encoder state across stages. 3 verifier tests
PASS. Image/report checks remain execution-receipt evidence; no independent
re-inference or full-image rescan. No scientific Drift claim follows from PASS.
Normalization/bandwidth fitting and observed MMD not performed or opened by this
verification. Next gate must use only TRAIN references for fitted parameters.

## D031 — Approved loader repair; bounded G3-C2 PASS (2026-10-16)
Owner said to begin the proposed serialization fix/regression/bounded rerun.
Compare trainer-canonical JSON bytes, accepting tuple/list encoding equivalence
without coercing values or dropping fields. Keep all existing hash and semantic
guards; no model/methodology change. 9 tests PASS and all nine encoders PASS CPU
integration on three frozen TRAIN-reference samples per case/view. No bulk
extraction or A/B/C inference; old failure retained. This closes the bounded
runtime blocker, not whole-dataset or MMD/research validation. Extraction commands
are supplied for the owner; no automatic transition to signals or Future.

## D030 — G3-C2 stopped at new loader defect (2026-10-16)
Owner authorized bounded TRAIN runtime for nine frozen encoders and preparation
of an extraction CLI only upon PASS. Seven synthetic tests pass; first runtime
load fails before forward because embedded Adam betas is a tuple while adjacent
JSON stores a list. Diagnostic-only read confirms canonical JSON equality and
the embedded config hash. No scientific-method change or checkpoint defect is
established. Honor explicit STOP: no loader repair/retry, no remaining inference
or bulk extraction. Proposed next action needs owner review: canonical config
comparison with regression test, followed by the same bounded runtime gate.

## D029 — Owner-authorized E1-SBSMI completed, fixed epoch100 retained (2026-10-16)
Owner explicitly authorized one official100-epoch E1-SBSMI run after quick readiness.
AC online/not discharging and remaining conditions PASS; prior power blocker cleared.
No settings changed. Run20261016_012854_728594 completed100 epochs/22000steps, exit0.
Retain epoch100 as preselected, not chosen by97.64% online training accuracy.
All checkpoint hashes and final config/mapping/cutoff verified. This is technical
training completion, not research validation or unseen evaluation. Actual45m04s
wall time is recorded rather than replacing it with the earlier timing estimate.
No retry/resume, additional model runs, A/B/C/Future, embeddings or commit/push.
Report: docs/e1_sbsmi_prefix_training_result.md. Stop for owner review.

## D028 — G3-C-R audit holds training recommendation on power readiness (2026-10-16)
Owner authorized readiness/resource audit only, not GPU optimization.
Existing preflight/test/source bindings remain valid; no rerun of full images/tests.
CUDA queries and current environment/Adam defaults PASS; output probe PASS.
RESOURCE_READY BLOCKED at observed ACLineStatus0/battery70%/Silent mode, despite
adequate snapshot memory/disk headroom. No power settings changed.
Recommend holding six-run approval until AC is connected and focused readiness
rechecked. Six fixed runs and methodology unchanged; 2GiB output allowance is
PROPOSED operational planning, not an enforced new threshold or measured bound.
121.15min remains a loop-only projection. Saved optimizer state and real GPU-loop
behavior UNKNOWN; no registered checkpoints deserialized. Report explicit limits
of interruption/disk-full handling; no trainer patch made without approval.
Evidence: docs/g3cr_pretraining_readiness_report.md and new g3cr_readiness artifacts.
No training, forward/backward, optimizer step, embeddings/MMD/Q/U/Future or
commit/push. STOP for owner review, no automatic training when audit status changes.

## D027 — Full prefix TRAIN image preflight PASS, training still gated (2026-10-16)
Owner approved only G3-C-F E1/E2 × three views with unchanged implementation.
E1 7034/7034 each; E2 9803/9803 each, zero errors/skips/repairs, both exit0.
Source/input bindings and receipts verified before/after; no rule, selection,
sampler, decoder, mapping or data changes. No failed run/retry occurred.
Accept as operational TRAIN image/report readiness, not scientific validity,
training success, independent binary rehash or frozen-encoder integration.
Evidence: docs/g3cf_full_train_image_preflight_report.md and new immutable
prefix_preflight_v1 receipts. Actual GPU loop/resources and saved baseline
optimizer-state parity remain separate issues. No GPU training authorization;
next proposed is pre-training readiness/resource review, not automatic training.
No CNN inference, optimizer steps, registered checkpoint load, embeddings,
MMD/Q/U, A/B/C/Future processing, commit/push. STOP for review.

## D026 — G3-C-I implementation-only completed; training remains gated (2026-10-15)
Owner approved new prefix adapter/trainer and CPU synthetic/metadata tests only.
Implemented frozen G2 TRAIN bindings, separate Final storage index and episode
target, exact existing decoding, fresh model families, seed42/batch32/Adam lr1e-4
decay1e-6/unweighted CE/no scheduler/fixed100 and epoch100-only registration.
No methodology changes. Existing SBSMI reports lack representation; representation
is bound through the pinned config, not a fabricated required report field.
20/20 targeted tests PASS. Environment and effective Adam defaults match baseline
code/config evidence; stored baseline optimizer state remains uninspected/UNKNOWN.
Full image/report preflight command implemented but NOT RUN on historical images.
Future training requires current all-view receipt, external owner approval and
explicit execution flag; flag alone is not an approval mechanism. No resume/retry.
No optimizer steps, training/GPU forward, registered checkpoint load, embeddings,
MMD/Q/U/Future or commit/push. Review report and next preflight approval pending:
docs/g3ci_prefix_training_implementation_report.md.

## D025 — G3-C-P prefix training design submitted, approval pending (2026-10-15)
Owner authorizes design only. Proposed six fresh seed42 E1/E2 runs, K32/K42 from
frozen G2 TRAIN/mapping, same architecture families/preprocessing and baseline
batch32/Adam lr1e-4 weight_decay1e-6/unweighted CE/no schedule. Proposed fixed100
epochs and epoch100-only selection before any run, not best A/B/C checkpoint.
SBSMI legacy lacked explicit pre-run selection field; do not rewrite history.
No warm-start, group-balanced training, cap128 training subset or use of Final51
targets; source report indices checked only for integrity and remapped by family.
New modules proposed, no legacy edits. Default Adam detail parity, image preflight,
tests and resource readiness remain prerequisites, not assumed verified.
No automatic resume/retry or configuration changes to rescue failures proposed.
Evidence/design: docs/g3cp_prefix_training_design.md. Timing projection uses only
recorded epoch durations and batch ratios, not measured prefix performance.
Implementation, GPU training and G3-C2 require separate approvals. No training-code
or model execution, inference/training, embeddings/MMD/Q/U/Future/commit/push. STOP.

## D024 — G3-C1 static readiness findings, no execution approval (2026-10-15)
Owner authorized static CNN/extractor audit only. Final registered checkpoint
hashes and source/config/mapping bindings verified; seed42/K51/18061 in configs.
E1/E2 prefix checkpoints with K32/K42 and appropriate cutoffs not found in scoped
run inventory. Final encoder reuse for prefix episodes would violate chronology;
early epochs of a full-window run are not a workaround. Prefix training requires
a separate approved gate, never automatic fallback.
Architecture feature endpoints match specification, but no manifest-bound Drift
extractor exists; legacy loaders/trainers are final-specific. Final static encoder
readiness is not complete extraction readiness or runtime validity. SBSMI config
lacks an explicit pre-run metric-independent selection field; do not invent one.
Report: docs/g3c1_frozen_encoder_readiness_report.md. No new methodology selected,
no G3-C2 execution, checkpoint deserialization/model load, inference/training,
embeddings/MMD/Q/U, Future access or commit/push. STOP for owner review.

## D023 — Owner closes G3-B PASS_WITH_LIMITATIONS (2026-10-15)
APPROVED by owner: G3-B post-cap sampler review is PASS_WITH_LIMITATIONS for
E1, E2 and Final. The mandatory loss>=5 group reviews have been completed and
accepted for all three cases. This supersedes the pending sampler-review status
in D021/D022; it does not retroactively change their evidence or decisions.
Keep seed42, cap128, frozen support, sampling algorithm and all existing matched
manifests unchanged. Post-cap groups<5/family/month is support failure; losing
>=5 groups is a mandatory sampler review trigger, NOT automatic family exclusion.
Preserve/report group loss, concentration, shared-month and multi-label dependence
limitations, including Final agent/February12/21 in one group. Target remains
sample-level descriptive MMD; no statistical-independence or null-standardized
Drift claim. No group-balanced resampling or sampler modification is authorized.
Sensitivity analysis is required before stronger stability claims; its explicit
design/interpretation must be fixed before observed Drift results are used to
choose methods. No sensitivity execution or alternative selection is authorized
here; primary manifests remain frozen. Any change of target needs separate review.
Only documentation updated. V1, code, checkpoints and immutable artifacts untouched.
Separate frozen-encoder/extractor integration gate remains NOT EXECUTED/NOT
AUTHORIZED by this decision. Minimum checks are recorded in the review report
section8; actual embeddings, TRAIN parameters and integration remain UNKNOWN.
No CNN training, inference, observed MMD, Future access, commit/push. STOP.

## D022 — Owner clarifies post-cap rule; sampler review recommendation pending (2026-10-15)
APPROVED interpretation: post-cap<5 groups/family/month is support failure;
loss>=5 groups is STOP for sampler review, not automatic support failure. No
seed42/cap128/frozen-support/algorithm changes authorized. V1 unchanged.
Focused G3-B-artifact-only review verified12 hashes and recounted selected
membership/support/concentration. Joint active group losses E1/E2/Final are
42.69%/42.74%/43.49%; max within-cell concentration43.48%/40.63%/57.14%.
Final agent/February retains all21 files: its12-file majority group predates cap.
No evidence establishes group independence or actual MMD influence. Full per-cell
pre-cap group-size comparison is unavailable within the allowed artifacts.
PROPOSED, NOT OWNER-APPROVED: accept each sampler review PASS_WITH_LIMITATIONS
for descriptive sample-level measurement; do not substitute group-balanced target.
Evidence: docs/g3b_post_cap_owner_review.md. Gate remains pending owner decision;
no real embeddings/inference or next stage authorized/executed. No sampler/code
or legacy artifact changes, Future access, training, MMD/Q/U, commit/push.

## D021 — G3-B preparation verified; post-cap decision blocked (2026-10-15)
Owner authorizes TRAIN-only preparation utilities, synthetic tests and historical
metadata cap audit only. V1/Amendment01/G3-A MMD implementation remain unchanged.
Engineering details frozen before selection: JSON SHA256 namespace TRAMIF-G3B-v1,
seed42, purpose/case/family/IDs, lexical digest ties; cap128; unordered TRAIN pairs;
lower weighted median with exact rational equal-family pair masses. No search over
these choices. Mean/population variance use equal-family TRAIN mixture; std<=1e-8
mask per V1; no epsilon rescue. Actual TRAIN embedding parameters remain UNKNOWN.
23/23 targeted tests PASS; immutable audit has26 output hashes and38 input/source
hashes verified. Evidence: docs/g3b_measurement_validation_report.md.
E1/E2/Final post-cap below5 group cells=0/0/0, but lost>=5 cells=15/24/27. Min groups
after cap=8/11/7. Both predicates reported, no favorable interpretation selected.
All three case group-support gates BLOCKED_OWNER_REVIEW: owner must resolve exact
stop predicate before dependent execution. Frozen S=17/25/25, primary policy and
group IDs preserved; no excluded bridge returned to scoring. Shared groups remain.
Canonical ID/class/config alignment PASS only; real embeddings alignment UNKNOWN.
No inference/training/actual MMD/Q/U/weights, Future access, graph reconstruction,
legacy changes or commit/push. G3-B audit COMPLETE and STOPPED, not scientific PASS.

## D020 — G3-A synthetic MMD implementation authorized and verified (2026-10-15)
Owner authorizes only the independent V1 biased RBF MMD2 primitive and synthetic
tests, advancing D019's proposed D1 only. Supplied h, float64, self diagonals,
explicit support and equal-family aggregation are preserved. Engineering choices:
direct-distance tiles of 32; fixed negative-roundoff guard 64*eps64 with raw-value
diagnostics; overflow fails instead of silently saturating. These are not tuned
with Accuracy and do not establish statistical standardization or validity.
16/16 targeted tests PASS; independent scalar reference has 48 comparisons.
Evidence/limitations: docs/synthetic_mmd_verification_report.md.
No legacy code/V1 changes, real embeddings, historical losses, Future access,
training, inference, null, fusion or new weights. No full-suite claim.
G3-A complete at synthetic numerical level; STOP before G3-B. D019's scientific
limits and unapproved beta/gamma remain in force.

## D019 — Conditional descriptive Drift scope approved; execution withheld (2026-10-15)
Owner accepts focused G4 findings: same-period group-disjoint null with exact
V1 budgets is INFEASIBLE for E1/E2/Final; other null designs remain NOT
ESTABLISHED, not universally impossible. Null-standardization is suspended
within current research scope. G4 null validity has NOT become PASS.
Owner conditionally approves V1 section6.4 descriptive class-conditional MMD as
exploratory Drift. D_cond/2 is a numerical bound only, not standardized drift,
significance, calibrated cross-view scale or evidence of future reliability.
Framework departure and reduced claims recorded separately in
docs/RESEARCH_PROTOCOL_V1_AMENDMENT_01_DESCRIPTIVE_DRIFT.md; V1 unchanged.
Frozen family support, chronology, grouping/retention, final mapping and Future
protection remain unchanged. Quality-only and learned static fusion remain
important controls. No beta/gamma, coefficient grid or Drift weighting approved.
Current authorization is documentation ONLY: no implementation/tests execution,
training/inference, embeddings, real MMD, legacy edits or artifact mutations.
Amendment proposes four bounded verification tasks: synthetic numerical tests,
train-only normalization/bandwidth contracts, post-cap support, integration
readiness. They are NOT executed or automatically authorized. The ambiguous
post-cap stop wording in V1 must be resolved before guard implementation;
no support threshold changed here. Next proposed task is approval of D1.
STOP after recording decision. No Future access, commit or push.

## D018 — G4 matched-null NOT ESTABLISHED; stop without fallback (2026-10-15)
Focused owner approval covered evidence/statistical review only. Immutable G2
and final February-March support/memberships reused; eight hashes and metadata
count checks PASS. No graph rebuilding, TLSH pairs, allocation search or MMD.
Distinguish design-specific impossibility from general unknown: same-period
group-disjoint two-side null cannot meet frozen n=min(N_A,N_B,128) across all S
in E1/E2/Final; simple per-class count bounds already prove this. A more general
dependence-matched nondegenerate null has UNKNOWN allocation and NOT ESTABLISHED
statistical validity. Equal conditional marginals do not imply exchangeable
whole blocks; shared-month/multi-label structures remain within signal support.
No threshold/budget/support change, replacement scheme, null estimator, epsilon,
coefficient, descriptive fallback or next-stage execution is approved here.
Report: docs/matched_null_feasibility_report.md. Propose owner scope/claim decision
at G4, including whether to approve V1's explicit descriptive departure; do not
activate it automatically. G2 computational findings preserved. Batch STOPPED.

## D017 — G2 counts feasible; no advancement authorized (2026-10-15)
Owner authorized historical episode feasibility only. Followed V1 E1/E2 dates,
prefix >=50 unique samples/>=3 months, primary retention0, sensitivity10/20,
signal support >=20 retained samples/>=5 joint AB groups per month.
Final51 is a candidate envelope justified by monotonic eligibility, not the
episode classifier size. Re-selected K_e=32/42 from prefixes; S_e=17/25.
All computations use frozen pre-April historical exports and saved digests;
no full raw metadata or Future records read. Source/config/ID provenance checked.
Separate induced AB graph prevents outcome C from bridging signal groups.
Whole-episode graph includes C for dependence audit only; no predictions/losses.
Both episodes PASS computational count/group criteria, NOT research validation.
Label-availability time remains UNKNOWN; image-content/episode-loader integration
not verified. Existing January-trained encoders cannot serve prefix episodes.
Counts/graphs show substantial remaining dependence; operational groups are not
proven independent. No null, estimator, coefficient, scope-reduction or label
assumption approved by this record. No thresholds/timelines/final mapping changed.
Evidence: docs/historical_episode_feasibility_report.md; run
historical_episode_feasibility/20261014_233013_930021; independent verification
historical_episode_feasibility_verification/20261014_233343_897002.
25/25 targeted tests; 50 artifact and29 input/source hashes;18 partitions PASS.
STOP after G2; propose matched-null allocation/dependence review per V1 step2,
subject to owner approval. Later adapter work also requires authorization.
No legacy code changes, CNN training/inference, Q/D/U/weights, commit or push.

## D016 — Research Protocol V1 documented, not method approval (2026-10-14)
Owner authorized creation of docs/RESEARCH_PROTOCOL_V1.md against source commit
d8f0c676f604a744f06800ab63c69dc6fe8beea3 and the original framework.
The document preserves BOTH mean monthly and worst-month Macro-F1 as primary
outcomes and records VERIFIED/PROPOSED/UNKNOWN separately. It specifies one
gated research path, explicit framework departures and a bounded first task:
historical E1/E2 feasibility audit before any prefix training.
Null-standardized D remains the framework primary specification, but its
feasibility/validity is not established by existing evidence; descriptive fallback is
separately named and requires approval. Positive group-weight U, coefficient
grid/selection, five-seed work and final statistics are proposals, not authorized
changes. No new weights or scientific validity claims follow from this record.
Existing source/code/legacy artifacts preserved; 24 artifact checksums rechecked.
Only protocol/checkpoint documentation was written; no tests/experiments,
inference/training, Future access, commit or push. Method/resource/future gates
remain with the owner; earlier approvals retain their original scope.

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
