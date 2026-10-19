# Remaining sampling diagnostics — cap64 metadata reviewed; computation pending

Focused review complete: docs/cap64_focused_review.md and
data/derived/cap64_focused_review/20261016_134158_860751/. Proposed decisions
PASS_WITH_LIMITATIONS for all3 cases, owner acceptance still pending. This report
specifies the exact group-weighted estimator and a finite18-computation batch.

Metadata execution update: `data/derived/cap64_metadata_review/20261016_132758_657700/`.
Same ranking/seed retained, nested first64 verified from locked primary selection.
E1/E2/Final retain minimum8/11/7 groups; no below5 failures. Additional loss>=5
relative to primary occurs in11/17/17 family-month cells; total loss>=5 relative
to pre-cap occurs in19/30/30. All three cases therefore require sampler review
before cap64 MMD. Proposed per-side sample counts823/1234/1244. See counts.csv
and proposed_subset.csv. Primary unchanged; no cap64 MMD computed.

2026-10-16. Bandwidth h/2 and2h is a separate fixed diagnostic. The remaining
sampling controls must not replace the primary sample-level result.

## Cap64

Keep the existing seed42/hash namespace/ranking, frozen S, TRAIN normalization
and primary h. For each family and side choose the first min(n_primary,64) under
the existing matched-A/matched-B ranking. This is the nested cap64 subset of
already-extracted primary samples, not a new seed or a search. Save exact IDs and
counts/groups/concentration. Use equal-family aggregation unchanged.

Before computing: count post-cap groups; if any family/side has<5, STOP this arm
for review. Report every loss>=5 as a new sampler-review trigger (D023 accepted
the primary cap128 review, not every future cap). Do not silently drop families,
alter cap or choose another seed. Run metadata review first; computation follows
only once any triggered review is resolved. No new embeddings should be needed.

## Group-balanced diagnostic

V1 names this diagnostic but does not select an algorithm. Proposed concrete
choice: reweight the SAME primary matched samples, without resampling. Within
each family/side, give every represented operational group equal total mass and
divide that mass equally between its represented samples:

    a_i = 1 / (G_A,c * n_A,c,g(i))
    b_j = 1 / (G_B,c * n_B,c,g(j))
    MMD2_weighted,c = a^T K_AA a + b^T K_BB b - 2 a^T K_AB b

Keep diagonals, frozen h/normalization, frozen support and equal-family outer
average. Shared-month groups stay shared; multi-label components are handled
within each family, preserving original group IDs. No graph rebuild or deletion.

This changes the empirical target from equal samples to equal represented groups
within family/month. It does not restore independence, estimate all pre-cap groups
or create a null test. Comparing it to primary diagnoses concentration effects;
neither estimate is chosen as the winner. Approval needed for this specific
weighting definition before implementation or real-data execution.

Verification before execution: uniform sample weights reproduce existing MMD;
singleton groups reproduce sample-level result; duplicating identical members
within a group preserves its total mass; dense reference agreement, symmetry,
finite/nonnegative weights summing to1, bounds, shared/multi-label integrity.
Report all per-family values/differences, group counts, concentrations and source
hashes. No grid search, coefficients, Q/U, predictions or Future access.

## Finite next batch proposed

1. Cap64 metadata-only review completed; immutable proposed subset and STOP flags saved.
2. Owner reviews cap triggers and approves or rejects the specific group-weighted
   estimator above. No blanket reinterpretation of the primary policy.
3. If approved/eligible, run each diagnostic once with both favorable and adverse
   findings reported. Preserve primary results; close sensitivity scope.

Reason for this gate: Amendment01 requires separate approval for sensitivity that
changes support/aggregation; D023 prohibits silently introducing group-balanced
resampling or changing the primary sampler. This proposal does neither yet.
