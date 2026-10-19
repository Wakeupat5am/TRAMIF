# TRAMIF — historical Methods and Results draft

Status updated D062,2026-10-18: factual historical and final fixed-horizon results
are available. This remains a paper draft,not full original-framework validation.
No significance or universal future-generalization claim. Original historical
Methods/Results below remain evidence of earlier development.

## Final frozen Future results (D062)

Following D060 owner-adopted scope/statistical amendments and D061 input-only
verification,the owner authorized final April–September evaluation. We report all
five seeds,sevendefinedmethods,retention0/10/20 and frozen TRAIN-size strata.
Primary retention0 contains20,363samples;monthly true-supported families range37–47.
Mean of six monthly Macro-F1 scores is computed per seed,then averaged across five;
Worst is each seed's minimum monthly score,then averaged across seeds.

| Method | Mean-month Macro-F1 % | Worst-month Macro-F1 % |
|---|---:|---:|
| SBSMI |64.272882|57.456785|
| Raw-byte |54.674800|47.956444|
| Entropy |52.291846|45.263350|
| Uniform |64.377877|56.235290|
| New Q |64.749422|56.630088|
| Q+D |64.757320|56.647208|
| Learned-static |64.936615|57.518346|

The primary Q+D−Q gain is+0.0078985percentage points on Mean and+0.0171204on Worst.
The predeclared joint practical target (Mean>=2pp andWorst>0) is not met. Seeds43/44
have negative Mean deltas. Q-only shows modest fixed-cohort gains over Uniform;
D adds little numerical benefit,and Q+D is below learned-static on both primary
aggregates. These are descriptive comparisons,no validated CIs/p-values.
Sensitivity10/20 does not rescue the incremental-D practical target;per-size gains
vary in sign. Counts/class composition change with retention. Stratification does
not establish causal adjustment. No post-Future choice of models,coefficients or
seeds was made. Complete mean/SD/range,monthly,all8contrasts,secondary and size tables:
`FINAL_FUTURE_RESULTS.md`;fullprecision artifacts/hashes are linked there.

The paper must retain the negative U simulation and E2 development findings.
Claims concern descriptive view–encoder drift and the frozen Q/QD study;QU/QDU,
null-standardized Drift,capacity-matched controls and statistical coverage are not
validated. Broader controls/end-to-end cost and independent backup remain limited.

## Methods

We study closed known-family classification under temporal dataset change using
three representations of the same BODMAS disarmed binary: raw-byte intensity,
local entropy and SBSMI bit-state transition images. The inputs are disarmed
bytes, not restored binaries or reconstructed feature vectors. Input sample SHA
identifiers are distinguished from SHA256 hashes of the disarmed bytes. Canonical
timestamps/family labels, class mappings and common view sample identities are
preserved. Representation details follow the saved preprocessing specification;
no new input truncation or resizing was selected using outcomes.

The Final cohort has51 families selected only from Aug2019–Jan2020 train, requiring
at least50 unique samples in three train months. February/March are historical
development. April–September are protected Future Test. Prefix episode E1 trains
through October(K32), calibrates in November, estimates signals in December and
scores January. E2 trains through November(K42), calibrates in December, estimates
signals in January and scores February. Prefix models never use Final checkpoints.

SBSMI uses MalSBSLCNet; raw-byte and entropy use CompactViewNet. All nine registered
seed42 checkpoints use fixed epoch100 selection. Architectures differ, so results
concern representation/encoder pairs without a capacity-matched superiority claim.
Standard scalar temperature scaling is fit on retained A only; no B/C refit.
Probabilities retain all K classes even when signal support is smaller.

The primary near-duplicate policy is chronological content equality and standard
TLSH distance0 retention, with operational grouping10 and sensitivity20. Labels
do not define similarity. The policy was developed after initial SBSMI validation,
so that history is not presented as preregistered. Operational connected groups
do not establish statistical independence. Label timestamps are retrospective
metadata, not proof of real-time label availability at each historical date.

Historical Quality L is equal-family mean calibrated NLL on full retained B within
frozen support S. Each support family needs20 samples and five operational A/B
groups in both months. S sizes are17/25/25 for E1/E2/Final; classifier K remains
32/42/51. Q-only weights are softmax(-L). Uniform averages calibrated probabilities.

Drift is primary class-conditional biased RBF MMD² on normalized frozen embeddings,
with TRAIN-only reference normalization/bandwidth and predetermined matched cap128
samples. Equal-family aggregation is numerically bounded as d=MMD²/2. Null
standardization was not established; descriptive D and its cross-view scale are
exploratory. Fixed bandwidth/cap/group-mass sensitivities are disclosed rather
than choosing the best one.

The pre-outcome selector uses beta in{0,.5,1}, gamma0. Forecast family loss is
L_v,c,B+beta*d_v. E1 selects minimum equal-family/equal-view absolute forecast
error, with fixed1e-12/simple tie rule. Beta is then held fixed for E2. Fusion
weights are softmax(-L-beta*d); no C labels fit temperatures or weights. E1 labels
are used to select beta and score its development result, so post-selection E1
metrics are not independent validation. E2 is a retrospective held-forward check;
the two episodes overlap.

An attempted positive group-weight uncertainty estimator failed simulation review
in several layouts, including unequal independent groups. It is retained as
negative methodological evidence and excluded from fusion; QU/QDU are unavailable.

## Historical Results

E1 selects beta1: forecast MAE .471385→.469168. E2 holds beta1 and obtains
.563763→.561424. Reductions are approximately.47% and.41%, respectively; these
are point diagnostics without inferential intervals. The selected grid is small
and the maximum drift loss adjustment is small; no larger grid was tried after
viewing outcomes.

New Q exceeds Uniform in supported Macro-F1 by1.5032pp(E1) and1.3399pp(E2).
QD adds.015557pp on E1 but loses.000818pp on E2 versus New Q. QD accuracy is unchanged
in both episodes; NLL and Brier worsen slightly in both. E2's SBSMI alone has lower
NLL/Brier than either fusion. Thus the small forecast-MAE improvement does not
translate into demonstrated additional fusion utility beyond Q in E2.

## Limitations and remaining work

Only one seed/two overlapping retrospective episodes; no Future execution.
Descriptive distance is not null-standardized; support/cohort/classifiers differ
between episodes; group/collection/label dependence and size/composition controls
remain incomplete. Learned-static solver has not been approved. Broad neural/PE
baselines, capacity/cost controls, four extra Final seeds and confirmatory
uncertainty remain incomplete. No claim of SOTA, intrinsic representation
superiority, efficiency advantage or universal future reliability is justified.
The framework's mean/worst-month five-seed Future outcomes remain unanswered.
# D052 status note — engineering does not add experimental evidence

The extra four Final seeds remain untrained;12 selected checkpoints are missing.
Seed-specific calibration/Q/D code is synthetic-tested but not executed on real
data. Learned-static solver and final interval protocol are proposed, not empirical
baselines or approved inferential results. Existing E1/E2 results/beta1 unchanged.
CPU random-input model-only medians are recorded in PRE_FUTURE_ENGINEERING_RESULTS.md;
they do not establish end-to-end efficiency or eliminate capacity confounding.
No Future result, multiseed claim or full-V1 completion is asserted.
