# D060 adoption notice — final scope approved

The single current statistical specification is `docs/FINAL_STATISTICAL_PROTOCOL.md`.
Owner D060 approved O1/O2/O4 and O3 WITH required retention10/20 AND TRAIN-file-size
stratification;see D060_O3_SCOPE_AMENDMENT.md. The older statistical proposals
below remain historical context,not a competing execution instruction. Numerical
solver adoption and five fits are already resolved by D058;do not request them again.

## Historical consolidated owner decisions — full V1 engineering, 2026-10-17

This document proposes decisions; it does not amend V1 or grant execution approval.
E1/E2 results, beta=1, existing Q/D and seed42 remain unchanged. U is diagnostic-only.

## 1. Learned-static numerical protocol — approval required

Keep V1's exact objective on retained B within frozen S, with full K-class
calibrated probabilities: minimize equal-family mean mixture NLL on the simplex.
Recommend the one implementation in `scripts/learned_static_solver.py`:

- float64, uniform initialization, projected gradient; Euclidean simplex projection
  permits exact zero boundary weights. Initial step1; Armijo coefficient1e-4,
  halve step, at most60 backtracks; at most10,000 iterations.
- Stop only if infinity norm of w−projection(w−gradient) ≤1e-10. This is a numerical
  stationarity criterion, not accuracy tuning. Convexity makes a sufficiently small
  residual meaningful, but it is not a universal objective-gap guarantee.
- Deterministic uniform-start trajectory selects ties; identical views return
  uniform. Do not claim unique identification or minimum-norm selection among all
  minimizers. Record weights/objective/residual/iterations/S/input hashes/solver source.
- Row-max log scaling avoids underflow of all three target probabilities. Zero
  candidate mixtures cause backtracking; nonfinite arithmetic, failed line search
  or iteration exhaustion STOP without registering approximate weights.
- No regularizer, grid, random starts, or C/Future access. Synthetic known optima,
  boundary solutions, analytical-vs-finite-difference gradients and imbalanced
  family replication PASS. Empirical fitting is NOT executed or authorized.

After approval, fit one learned-static baseline per Final seed using its own
February T/March B/S. Historical E1/E2 fits, if desired for a table, must be labeled
retrospective; they cannot alter already-completed selection or beta.

## 2. Statistics — retain V1 estimands; execution/coverage approval required

Primary outcomes: per-seed mean of SIX monthly supported Macro-F1 scores and
per-seed minimum of those six; average each outcome across all five registered
seeds. A minimum of seed-averaged monthly scores is a different quantity.

Recommend retaining V1's proposed paired group×seed percentile design, subject to
coverage validation and owner approval, rather than inventing a replacement:

1. Freeze original monthly true-support sets; keep all51 output classes. Missing
   classes in a draw retain their original denominator; F1=0 when its denominator
   is zero. No discarded replicates. Empty month/nonfinite metric STOP, no retries.
2. After separate Future approval, construct operational joint six-month groups
   from all original eligible Future samples, including excluded bridges, then
   score retained members only. No historical bridges. No such graph/input read now.
3. PCG64 seed42,20,000 paired draws: sample five seed indices with replacement and
   G whole active component indices with replacement. Freeze lexicographic group
   order. Same draws for all methods; preserve cross-month/multilabel membership.
   Save draw indices and per-month sample counts. Means/worsts are computed per
   selected seed before averaging, then subtract paired method outcomes.
4. Original-data point estimates remain primary. Pointwise95% percentile intervals
   are descriptive conditional intervals, not automatically simultaneous coverage.
   Retain the conservative V1 family-size10 quantiles0.0025/0.9975 for available
   contrasts until a formal scope amendment: do not silently shrink multiplicity
   because QU/QDU are unavailable. Unavailable contrasts stay NOT ESTIMABLE.

**Owner must freeze the contrast family.** Recommended active signal contrast is
QD−NewQ for both primary outcomes. Report QD−Uniform and QD−learned-static as
secondary; three singles remain point baselines. This is an explicit proposed
scope departure from full QDU's ten contrasts, not an already-approved change.
Keep conservative family10 intervals if conditional intervals are eventually
reported; do not assert family-wise significance absent supported marginal coverage.

**Coverage remains UNKNOWN.** Arithmetic tests are not coverage simulation.
Before confirmatory inference, approve one finite coverage experiment covering
unequal/giant components, rare-class disappearance, correlated seeds and tied
worst months, with known population confusion probabilities. Predeclare generator,
outer/inner budgets and precision before executing it. No empirical CIs or
simulation search were run in this batch. If assumptions/coverage fail, recommend
reporting original point estimates plus seed spread and explicitly conditional
uncertainty only; any estimator/claim change needs owner approval.

Components do not prove independence; seeds share data, and six fixed months do
not represent independent temporal populations. This cannot measure all training,
calibration, method-selection or future-domain uncertainty. No p-values are proposed.

## 3. Compute and scope decisions

- Authorize remaining12 Final CNN fits separately: seeds43–46×3 views, fixed100
  epochs. The current batch authorizes code/preflight only, not GPU execution.
- Authorize seed-specific real historical postprocessing separately. CLI/tests
  exist; real end-to-end integration remains NEEDS_CHECKPOINT. No seed42 T or
  embedding statistics may be reused for different encoders.
- Preserve negative E2 fusion evidence. QD's slightly lower E2 F1 and higher
  NLL/Brier are not reasons to retune beta1 or select a different seed.
- Decide scope of remaining framework controls: file-size-stratified RQ2,
  capacity-matched/same-view controls,1D/PE/gating baselines and end-to-end costs.
  Without them, do not claim intrinsic view superiority, causal drift effects,
  cost advantage, SOTA or completion of the full original framework. A bounded
  Q/QD study with explicit limits is a proposed scope choice, not silently adopted.
- Off-device backup remains unverified. A local curated restore is insufficient
  protection against loss of this machine. Do not delete original artifacts.
- Future input preparation/evaluation is still a separate protected approval.

The solver and arithmetic implementation are engineering-ready; statistical scope
and inferential validity merit High/Astra methodology review. No new model or
reasoning setting has been selected automatically.
