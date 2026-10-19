# D058 — Five-seed learned-static weights frozen

2026-10-18. **5/5 production fits PASS; 38/38 targeted tests PASS.** Owner explicitly
adopted the D057 arithmetic implementation and authorized one fresh sequential
CPU batch. No failures, retries, tolerance changes or seed selection occurred.

## Protocol and source binding

Objective: class-balanced mixture NLL on4064 retained March B samples in frozen
S25, with full K51 probabilities and each seed's own fixed February temperature.
Uniform start; Euclidean projected gradient; Armijo c1e-4,step1,halving,max60
backtracks,max10000 iterations; projected residual<=1e-10. Q/QD unchanged.

Approved solver bytes reused unchanged:
`scripts/learned_static_solver_v2.py`, SHA256
`8cafeea83b83986111e9376813e5d23b72ac87b4ffdc8356b08096771a298452`.
New production adapter:`scripts/fit_final_learned_static_v2.py`, SHA256
`f2e99bff1640afa7428c728c3f115316ff1ac533359eaa97972d6ab4ea4e2d15`.

The preserved D057 numerical API/version still carries candidate metadata.
The D058 adapter records explicit production adoption separately, promotes its
result only after verification, and freezes `CONVERGED_VERIFIED` weights. This
does not rewrite D057's historical approval state or silently change source pins.
The original solver/runner and all five D056 failure files are unchanged.

## Preflight and verification

Before any fit:38 targeted tests, exact D057 solver/dependency hashes, D054 pinned
registry, immutable historical selection, all five calibrated-logit datasets,
canonical ID/class ordering, own-seed checkpoint/T provenance and complete S25
coverage PASS. All seeds use the same4064 canonical IDs, with no Drift cap.

Each result passes finite/nonnegative simplex checks and residual<=1e-10, followed
by independent Decimal60 raw-log gradient, exact family-balanced weighting and
Euclidean projected KKT verification at the same1e-10 threshold. Separate
log-sum-exp objective and Decimal objective agree within1e-12. This objective
tolerance is readback validation, not an optimization stopping rule. Source/input
hashes are rechecked before every freeze; each seed has its own COMPLETE receipt.

## Results

Weights in view order SBSMI,raw-byte,entropy. Full precision is stored in JSON;
rounding here is for display only.

| Seed | SBSMI | Raw-byte | Entropy | Fitted NLL | Iterations | Solver residual | Independent KKT residual | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
|42|0.912648168|0.026938909|0.060412922|0.749230423|28|6.38003e-11|6.38003e-11|PASS|
|43|0.923037359|0.022154675|0.054807966|0.723598731|52|6.89769e-11|6.89769e-11|PASS|
|44|0.840706987|0.030923928|0.128369085|0.771493748|32|5.44671e-11|5.44671e-11|PASS|
|45|0.891019827|0.108980173|0|0.707409759|34|5.32768e-11|5.32768e-11|PASS|
|46|0.947720418|0.052279582|0|0.704685290|19|5.21799e-11|5.21799e-11|PASS|

All stored weight sums are1.0 in compensated float64 readback; the solver's
implicit simplex-coordinate semantics are retained in verification metadata.
Zero entropy weights for45/46 are feasible boundary optima checked by KKT,
not manually clipped or grounds to omit those seeds.

Descriptive statistics across all five seeds (sample SD,ddof1):

| View | Mean weight | SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
|SBSMI|0.903026552|0.040368426|0.840706987|0.947720418|
|Raw-byte|0.048255454|0.035845895|0.022154675|0.108980173|
|Entropy|0.048717995|0.053068449|0|0.128369085|

## Immutable artifacts

Production root:
`data/derived/final_learned_static_v2/20261018_003544_337822/`.
For each seed42-46:`seed_<seed>/weights.json`, `verification.json`,
`fit_inputs.npz`, `COMPLETE.json`. Root contains source snapshots,config,
selection,38-test log/receipt,preflight,summary and final receipt.

Receipt SHA256:
`0dc546341da26602d83c237209fd0df5319a12ef6967af6a06ea186634b97bd8`.

Independent stdlib readback and descriptive CSV/JSON:
`data/derived/final_learned_static_v2_readback/20261018_003638_283122/`.
Readback receipt SHA256:
`34745461d851d4cbec3e4ec7ab3a99c9a7a32d72674da387e1052ccc4dddeab5`.
It verifies all output/source hashes,per-seed COMPLETE chains,simplex/residuals,
Decimal KKT results,config bindings,all-seed retention and D056 preservation.
No fitting,calibration or inference was repeated in readback.

## Interpretation and remaining work

**INTEGRATION VERIFIED:** the adopted solver converges on all five registered
historical datasets and valid learned-static weights are frozen. The NLLs above
are training objectives on the same March B/S25 used for fitting; they are not
held-out estimates and do not show superiority over Q/QD or Uniform. Five seeds
do not represent independent temporal experiments. No confidence interval or
significance claim is made. This does not resolve statistical/scope limitations.

No CNN training,inference,temperature refit,Q/D/U recomputation,Future access,
commit or push. All seeds retained,with no beta/Q/QD retuning.

One proposed next step: resolve the already-consolidated final statistical
coverage/contrast/multiplicity decisions in `PRE_FUTURE_OWNER_DECISIONS.md`,
before combining a final five-seed experimental freeze and separately requesting
Future authorization. No new statistical execution is authorized by D058.
