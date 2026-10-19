# G5 synthetic estimator verification

Date: 2026-10-16. Conclusion: **PASS — mathematical implementation and synthetic
serialization/replay only. Statistical validity NOT ESTABLISHED.**

## Implemented contract

`scripts/quality_uncertainty.py` has no filesystem, CNN, calibration or fusion
execution. `estimate_uncertainty` receives three aligned nonnegative loss arrays,
explicit sample IDs, families, group IDs and fixed support. It rejects missing
support, duplicate sample IDs, invalid dimensions and nonfinite values.

The point estimate is the equal-family mean of sample-level loss within each
family. Active group IDs are sorted lexicographically. `draw_group_weights` uses
PCG64 seed42, 2,000 exponential(rate1) draws by default, shared across all three
views and all classes in the same operational group. Positive draws multiply
both loss sums and sample counts; this does not equalize group mass. No family
or replicate is dropped. U is SD across replicate losses, ddof1, not SD/sqrt(B).

Returned evidence includes original/per-class loss, draws/order, replicate losses,
mean/bias, covariance, diagnostic quantiles and NumPy version. It does not produce
fusion weights. Explicit draw input supports exact replay and deterministic tests.
No code in this module validates historical chronology: a separate bound-input
adapter is still required before use on real data.

## Verification

Command: `.venv\Scripts\python.exe -B -m unittest tests.test_quality_uncertainty -v`

**10/10 tests PASS, 0.229s**:
- Independent sample-level reference versus group-total implementation; ddof1,
  covariance and all-one draws recovering original L.
- Constant loss within each family gives U approximately0 (absolute tolerance
  1e-14 for float64 arithmetic, not clipping estimator output).
- Paired views, affine loss relationships and a multi-label shared group.
- Invariance to common positive rescaling of a replicate's group weights.
- Exact default PCG64 replay, canonical group order and sample permutation.
- Outside-support observations do not enter balanced estimation or active groups.
- Invalid identities/support, zero/negative/nonfinite draws, invalid draw shapes,
  nonfinite/negative loss and arithmetic overflow fail closed.

One synthetic fixture additionally saves and reloads actual draws/results;
replicate losses and SD replay exactly with `allow_pickle=False`. Output checksums
verified. This is synthetic integration evidence, not historical integration.

Evidence directory:
`data/derived/quality_uncertainty_synthetic/20261016_164608_501043/`

Files: `synthetic_arrays.npz`, `receipt.json`, `checksums.json`.
Receipt SHA256:
`ea8e184467d3a20f4865c3b4aeced7d51ec428ff037bc9f74fe578a71c1962d9`.
Receipt records source/design hashes, synthetic identities, draw configuration,
environment version and tests. No real losses appear in these artifacts.

## Limits and next gate

Float64 intermediate overflow or nonpositive denominator stops execution;
no repair, replicate rejection or selective retry is implemented. Output SD near
zero is retained as computed. Draw storage scales with replicates × active groups;
large historical resource requirements have not been benchmarked in this batch.

Not verified: repeated-dataset statistical calibration, dependent-group behavior,
seed43 stability, rare-family/dominant-group influence, real logits/calibration
bindings, prefix input completeness or the coefficient-dependent dominance gate.
Operational groups are not proven independent. This estimator omits CNN-training,
calibration-fit and temporal uncertainty. No conclusion about future reliability.

Next proposed task: predeclare a finite synthetic simulation design with unequal
groups, dependence and rare families, including comparison with repeated-dataset
spread. Review the design before executing it. Do not use these tests to authorize
real Q/U, negative U penalties or QU/QDU weights. No C/Future, inference, training,
legacy changes, commit or push in this batch.
