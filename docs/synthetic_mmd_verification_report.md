# G3-A — Synthetic MMD verification

Date: 2026-10-15. Status: **PASS for the tested synthetic numerical contract**.
This is not validation of descriptive Drift on TRAMIF or research validation.
Authority: owner G3-A approval following D019; V1 section 6 and Amendment 01.

## Mathematical contract

For supplied bandwidth h > 0 and feature width p:

`k(x,y) = exp(-sum_j (x_j-y_j)^2 / (2*p*h))`.

Here h is the V1 bandwidth parameter, not a standard deviation to square again.
For n rows in X and m rows in Y, the biased estimator is

`MMD2 = sum_ij k(X_i,X_j)/n^2 + sum_ij k(Y_i,Y_j)/m^2
        - 2*sum_ij k(X_i,Y_j)/(n*m)`.

All ordered pairs, including self diagonals, are included. The RBF kernel is
positive semidefinite and lies in [0,1], so the exact biased estimator is in
[0,2]. Identical empirical measures give zero; a positive empirical value alone
is not evidence of statistical significance or predictive reliability.

`class_conditional_mmd2` averages per-family results with weight 1/|S| over an
explicit ordered support S. It never infers support, intersects missing families,
or substitutes sample-count weighting. Requested missing/empty families fail.
Keys outside S are deliberately ignored. Historical minimum sample/group counts,
caps, normalization, bandwidth fitting and matching are outside this primitive.

## Implementation and numerical decisions

- Module: `scripts/descriptive_mmd.py`; public functions `biased_rbf_mmd2` and
  `class_conditional_mmd2`. No file I/O, models, fitting, null or fusion imports.
- Inputs become float64; matrices must be nonempty, real, finite and dimensionally
  compatible. Invalid bandwidths/block sizes fail. Inputs are not mutated.
- Direct coordinate differences avoid cancellation from Gram-distance formulas
  at large common offsets. Kernel computation is vectorized within sample tiles.
- Default tile size is 32. The difference tile requires about `8*32*32*p` bytes:
  16 MiB for p=2048, 4 MiB for p=512, plus inputs, conversions and other arrays.
  These are allocation estimates, not measured whole-process RAM guarantees.
  All requested families may retain converted arrays; this is not out-of-core.
  Runtime remains quadratic in sample counts.
- Kernel block totals and the final three-term sum use `math.fsum`. Only negative
  final values in `[-64*eps64,0)` are clipped, retaining the raw value and a flag.
  `64*eps64 = 1.4210854715202004e-14` is a fixed engineering guard for cancellation
  of O(1) terms, not a universal error bound for arbitrary dimensions/conditioning.
  Larger negatives, invalid means and out-of-range results raise errors.
- Arithmetic overflow fails explicitly even when a more elaborate numerical
  algorithm could recover a finite kernel. Exponential underflow to zero is
  accepted; subnormal squared distances may also underflow. Float64 cannot recover
  distinctions already lost in the supplied numbers. No relative-accuracy
  guarantee is made for MMD extremely close to zero.
- Neither tolerance nor tile size is selected using classification Accuracy.
  The independent-reference absolute tolerance is 2e-14, on a precision scale.

## Verification evidence

Executed only:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_descriptive_mmd.py -v
```

Observed result: **16/16 tests PASS**, 0.358 seconds, exit code 0. This is a
targeted-suite result, not a claim that the full repository suite was rerun.

Coverage: identical empirical measures; analytically known singleton result;
constructed separated point masses; symmetry and row permutation; theoretical
bounds; duplicates; unequal counts; float32 conversion/input immutability;
large shared offsets; invalid shapes/types/non-finite inputs; invalid h/tiles;
explicit arithmetic overflow and injected final-stage roundoff; equal-family
aggregation and missing/extra support handling. The scalar-loop reference uses
independent `math.exp`/`math.fsum`, not production kernel helpers: 48 comparisons
across four shapes, three bandwidths and four tile sizes, including p=2048.
The separated-mass test does not assert monotonicity for every possible shift.

SHA-256 of the verified sources:

- module: `574b361ea9b28bdce5810dd3152858987e901f75663208af9afca670fcb9ff01`
- tests: `c7d3d97ddf8f2c8a7be8a8739e41714aedae8bd285dbbecca64ab8bbb74e62e4`
- unchanged V1: `e64664c4c499ef762b4a52c45889f0d05db6741ec5021b249f2072585f816546`

## Limits and stop

IMPLEMENTED and UNIT TESTED: the standalone formula and synthetic contracts.
UNKNOWN: behavior on actual embeddings, train-only normalization/bandwidth
stability, post-cap group support, cross-view statistical comparability and
whether descriptive drift predicts future error. Null validity remains NOT
ESTABLISHED. Dividing MMD2 by 2 would only bound a number, not standardize it.

No historical predictions/losses, real embeddings or Future Test were read;
no CNN, actual Drift, Q/D/U weighting, inference or training was run. Legacy code,
V1, checkpoints and artifacts were not changed. Stop at G3-A. G3-B needs separate
authorization; no automatic normalization, bandwidth fitting or support changes.
