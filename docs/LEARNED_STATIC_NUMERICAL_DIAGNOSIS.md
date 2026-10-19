# D057 — Learned-static numerical diagnosis and candidate correction

2026-10-18. **VERIFIED ROOT CAUSE; 31/31 targeted tests PASS.**
Candidate arithmetic correction is ready for owner review. **No registered
empirical fitting, no frozen weights, no full corrected seed42 solve.** Original
D056 failure, solver, runner, source hashes and frozen research artifacts remain
unchanged. This report supersedes the undiagnosed status, not the failed result.

## 1. Exact scope and evidence

Only the pinned D056 seed42 `fit_inputs.npz` (4064 March B/S25 rows, three
true-label log probabilities derived from calibrated K51 outputs), config and
failure/readback receipts were read. No new calibration, logits inference,
checkpoint loading, other-seed input processing, Q/D/U or Future access.

| Evidence | Unique artifact directory under `data/derived/` |
|---|---|
| Original failed run, preserved | `final_learned_static/20261018_000858_569331/` |
| Original hash readback | `final_learned_static_failure_verification/20261018_000959_177046/` |
| Exact original trajectory + Decimal oracle | `learned_static_diagnosis/20261018_001906_619077/` |
| Final targeted tests + controlled comparisons + one diagnostic step | `learned_static_numerical_verification/20261018_002401_874991/` |

Reproduction receipt SHA256:
`1ed6d5c9a6812104d21a73f5650252594c63fc00ade0c5ff6e25ae061a982765`.
Final verification receipt SHA256:
`1637bade61a3a89aea336bbeebd91ca54e39ab2aa0535f58f80ba1cb46f7d093`.

`trajectory.json` records all original iterations/trial steps. Failure at iteration
26, stored weights, objective and residual reproduce **exactly**, with an additional
call to the untouched original solver confirming the same exception. Source hash
checks precede execution. The receipt rechecks all five D056 files and all four
original source/test hashes. Source snapshots and final test logs are preserved.
The preliminary 29-test verification directory `20261018_002311_682907` is retained;
the final receipt above adds two anchor/stationarity regression tests, not a
different solver or selection of favorable numerical results.

## 2. Mathematical review

Let ell_iv be the fixed February-calibrated, full-K51 log probability of the true
class for sample i and view v. For family c(i) in frozen S25, set
alpha_i = 1 / (|S| n_c(i)). The objective on the simplex is

    F(w) = -sum_i alpha_i log(sum_v w_v exp(ell_iv)),
    w_v >= 0, sum_v w_v = 1.

Each family has equal total weight; no B label is used to refit temperature and
there is no renormalization across S. B labels are legitimately used to fit this
supervised static-mixture baseline, not to evaluate its generalization here.

Set m_i=max_v ell_iv and a_iv=exp(ell_iv-m_i). Then

    F(w) = -sum_i alpha_i m_i - sum_i alpha_i log(a_i dot w),
    g_v = -sum_i alpha_i a_iv / (a_i dot w),
    H_uv = sum_i alpha_i a_iu a_iv / (a_i dot w)^2.

H is positive semidefinite, so this is convex (not necessarily strictly convex).
The original objective and analytical gradient formulas are correct. Row-max
scaling cancels mathematically and avoids underflow of all three probabilities.
An independent scalar Decimal exp(log) oracle differs from the prepared-matrix
gradient by at most 1.028e-18 at the checked point. Finite-difference tangent
derivatives at uniform and failure points agree at multiple step sizes; at the
failure point h=1e-6,1e-7,1e-8 gives errors approximately
4.27e-12,4.27e-14,4.27e-16 for direction (1,-1,0), consistent with O(h^2).

Exact KKT conditions: active weights have a common gradient; inactive gradients
are no smaller. With sum alpha=1, w dot g=-1, so the common active gradient is -1.
At D056 the independent active-gradient spread is **3.19715e-9**. The candidate
has not met the approved stationarity condition and must not be accepted.

For t>0 and d=projection(w-tg)-w, exact feasible Euclidean projection gives
g dot d <= -||d||^2/t. Thus the original Armijo rule
F(w+d) <= F(w)+1e-4 g dot d is mathematically appropriate. The projected residual
||w-projection(w-g)||_infinity is also appropriate. Neither needs relaxing.

## 3. Verified failure mechanism

The implementation checks simplex mass only to absolute tolerance 1e-12, then
evaluates the **off-simplex extension** of F at rounded stored coordinates.
Near stationarity the gradient is almost the normal vector (-1,-1,-1). Tiny
projection/storage errors in total mass therefore dominate true tangent descent.

| At original iteration 26 | Measured value |
|---|---:|
| Stored sum(w)-1, Decimal on exact binary64 values | +2.220446049250313e-16 |
| Original projected residual | 1.9771081438690885e-9 |
| Original unit-step change in total mass | -5.551115123125783e-16 |
| Normal contribution to g dot d | +5.551115124267209e-16 |
| Tangent contribution to g dot d | -5.970607822500114e-18 |
| Net g dot d, Decimal | +5.491409046042208e-16 |
| Objective increase of that rounded candidate, Decimal | +5.508729578657234e-16 |
| Objective increase computed by original float64 subtraction | +6.661338147750939e-16 |
| One ULP of original absolute objective | 1.1102230246251565e-16 |

The original candidate is not a proper descent direction on the represented
constraint; Armijo rejects it even under 70-digit arithmetic. **Merely evaluating
the same off-simplex objectives more accurately would not fix this failure.**
All 60 original float64 trials reject; independently checked trials 0,1,5,20,59
also reject in Decimal. At very small steps projection noise persists while the
meaningful tangent step vanishes. More backtracking is not a justified remedy.

Represent the same point on the exact simplex (largest coordinate implicit),
then compute an exact Euclidean unit step with the independent Decimal gradient:

- g dot d = -5.970607189496e-18;
- F(new)-F(old) = -4.238554244371e-18;
- Armijo required increment = -5.970607189496e-22;
- exact Armijo **passes**; projection inequality holds to Decimal rounding.

This also exposes a second problem: legitimate decreases are below an absolute
float64 objective ULP. Comparing two O(1) losses can lose the sign or accept a
true increase when they round equal. Both problems are covered by the correction.
Gradient summation error (maximum1.3323e-15 here) is not the primary explanation.

## 4. Candidate correction and classification

New isolated module: `scripts/learned_static_solver_v2.py`, version
`D057-affine-simplex-armijo-v2-candidate`, SHA256
`8cafeea83b83986111e9376813e5d23b72ac87b4ffdc8356b08096771a298452`.
The production runner still imports the original module; its D052 pin is unchanged.

1. **Affine simplex representation.** Interpret the largest coordinate as
   1 minus the other two. Recover its tiny storage correction with `math.fsum`.
   Apply it to mixture evaluation and compute candidate differences in a common
   tangent chart, including when the largest coordinate changes. Eight machine
   eps is an internal storage-validation bound, not a convergence tolerance.
2. **Centered gradient.** Compute g_j-g_anchor directly from per-sample view
   contrasts with compensated summation. Subtracting a common scalar does not
   alter an exact Euclidean simplex projection or the stationarity criterion.
3. **Direct Armijo gap.** For z_i=(a_i dot d)/(a_i dot w), evaluate

       F(w+d)-F(w)-c g dot d
       = (1-c) g dot d + sum_i alpha_i [z_i-log1p(z_i)], c=1e-4.

   This is an algebraic identity. For |z|<=1e-3, compute the log remainder via
   terms k=2..12; omitted tail <=|z|^13/[13(1-|z|)] <=7.71e-41. Else use log1p.
   The threshold is justified by numerical truncation, not accuracy tuning.
   Require negative g dot d and gap<=0, with **no added acceptance slack**.
   A zero step above the residual threshold cannot masquerade as progress.

Solver arithmetic remains float64; Decimal70 is only an independent diagnostic
oracle. Uniform initialization, projected gradient, Euclidean projection, step1,
halving, Armijo1e-4,60 backtracks,10000 iterations and residual<=1e-10 remain.
Nonfinite comparison/gradient errors and nonconvergence fail closed. No Newton,
grid, restarts, alternate tie selection, clipping of failed weights or tolerance
relaxation was introduced.

**Classification:** an arithmetic bug correction preserving the mathematical
protocol, but a changed executable implementation and numerical trajectory.
It is **PROPOSED, NOT APPROVED FOR REGISTERED FITTING**. Owner adoption/version
registration is required before replacing the pinned implementation and starting
a new batch. No research objective/cohort/coefficient amendment is proposed.

## 5. Verification and limits

**31/31 targeted tests PASS**:20 new numerical tests +6 original solver +5 runner.
Tests cover gradient finite differences/raw-log oracle, Euclidean projection,
residual equivalence, switching anchors, KKT at known optima, boundary zeros,
flat/nearly identical views, ill-conditioning with explicit budget failure,
small target probabilities, mass-error invariance, actual ascent rejection,
stable remainder at its switch, determinism and nonfinite/Armijo failures.

| Controlled synthetic objective | Original | Candidate | Candidate residual |
|---|---|---|---:|
| Known interior optimum (5/7,3/14,1/14) | Armijo failure | Converges in24 iterations | 8.986e-11 |
| Dominant-view vertex | Converges | Converges in1 | 0 |
| Symmetric edge optimum | Converges | Converges in1 | 2.220e-16 |
| Exact flat tie | Uniform | Uniform at0 | 0 |
| Symmetric nearly identical views | Uniform | Uniform at0 | 0 |
| Full target log probabilities shifted by -1000 | Converges | Converges in34 | 5.415e-11 |

For the interior case the closed-form solution is available from frequencies
(0.60,0.25,0.15) and likelihoods0.8 on diagonal/0.1 off diagonal. Independent
Decimal KKT stationarity error8.98554e-11 and separate log-sum-exp objective
readback agree. Returning without exception is not the sole acceptance evidence.

Exactly **one** candidate unit step was tested at the saved D056 failure point:
its float64 Armijo gap is -4.237957322092e-18; Decimal gives
-4.237957310570e-18 (relative difference2.719e-9). Both accept. Residual decreases
from1.9771080e-9 to8.3005072e-10, **still above1e-10**. No full corrected empirical
solve was run, and no claim of seed42/five-seed convergence is made.

Limits: this is not a proof for all floating-point inputs. Very ill-conditioned
views can still exceed the fixed iteration budget. Relative scaled probabilities
can underflow if view log probabilities differ beyond float64 range; unsupported
or nonfinite arithmetic must STOP, not silently recover. Near-ties need not have
uniquely identified weights. The residual is stationarity, not a universal error
bound on weights/objective or research validity. No performance-based choice was made.

## 6. Files and next action

New: the candidate solver, `scripts/diagnose_learned_static.py`,
`scripts/verify_learned_static_numerics.py`, `tests/test_learned_static_numerics.py`,
and this report. Tracking: PROJECT_PROGRESS, DECISION_LOG, readiness note and continuation state.
No modification to original solver/runner/tests, V1, checkpoints or frozen artifacts.

**One next action:** owner approve adopting this versioned arithmetic correction,
then integrate it using a new explicit source binding (retain D052/D056 evidence)
and authorize a fresh sequential CPU batch for seeds42-46. Keep all current
hyperparameters and STOP-on-failure. This D057 batch stops before that integration
and production execution; it does not request reapproval of the unchanged objective.
