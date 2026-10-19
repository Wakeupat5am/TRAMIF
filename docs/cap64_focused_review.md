# Cap64 focused review and final sampling-diagnostic proposal

2026-10-16. VERIFIED metadata only; no new MMD, embeddings, graphs or weights.
Owner requested focused cap64 review and concrete group-balanced design.

## Evidence and scope

Artifacts: `data/derived/cap64_focused_review/20261016_134158_860751/` contains
report.json, family_month_counts.csv and checksums.json. Input hashes bind G3-B
primary matched manifests and existing cap64 proposal. All134 family/month cells
recomputed and compared to stored cap counts. Each proposed row is an exact
primary row; same seed42/hash ranking, first min(n,64), no alternative seeds.
No samples added, labels/groups changed, missing family or below5-group cell.
Shared/multi-label incidence below is measured WITHIN each selected set, not the
complete graph. Original group IDs are never rebuilt or split. The receipt also
counts retained samples inheriting shared/multi-label flags from primary selection
so disappearing second-month/label observations do not imply actual independence.

## Findings

| Case | Unique groups primary→64 (union A/B) | Min groups/family/month | Max fractional group loss per cell vs primary | Max group share after64 | Largest increase in group share |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | 1348→1046 | 8 | 50.79% | 48.44% | 4.96 percentage points |
| E2 | 2170→1638 | 11 | 50.42% | 39.06% | 3.13 percentage points |
| Final | 2069→1572 | 7 | 50.00% | 57.14% | 5.47 percentage points |

Unique groups are union counts, not sums over families/months. Fractional losses
above are maxima for individual cells; they are not overall loss percentages.

| Case | Shared-month groups primary→64 | Multi-label groups primary→64 | Samples in shared groups after64 | Samples in multi-label groups after64 |
| --- | --- | --- | ---: | ---: |
| E1 | 87→66 | 21→10 | 366 | 45 |
| E2 | 134→106 | 28→21 | 550 | 137 |
| Final | 105→79 | 22→15 | 541 | 197 |

Using flags defined on PRIMARY instead, cap64 retains406/596/574 samples in
shared-month groups and60/153/204 samples in multi-label groups (E1/E2/Final).
Thus reduced within-subset incidence must not be described as removal of dependence.

Most concentrated cells: E1 dinwod/B48.44% (primary43.48%), qqpass/A35%,
dinwod/A31.25%; E2 dinwod/A39.06% (primary40.62%), sytro/A35.19%,
sytro/B31.48%; Final agent/A57.14% (unchanged), agent/B47.62% (unchanged),
benjamin/B42.19% (primary44.29%). These are not chosen exclusion criteria.

## Review recommendation — owner decision pending

| Case | Proposed decision | Reason and limit |
| --- | --- | --- |
| E1 | PASS_WITH_LIMITATIONS | Frozen support/IDs intact; no<5 groups; concentration persists and may increase |
| E2 | PASS_WITH_LIMITATIONS | Same; group loss is substantial, not a support failure by the owner-defined rule |
| Final | PASS_WITH_LIMITATIONS | Same; agent/A remains dominated by one represented group,57.14% |

This recommendation is INFERRED for a sample-level descriptive sensitivity, not
statistical independence or validation of random-sampling representativeness.
Hash ranking preserves the specified file-level target; multiple similar files
can therefore dominate a selected family. Group balancing changes that target,
so it is a separate diagnostic and cannot be silently substituted for primary.
No new concentration cutoff is invented after seeing counts. Review triggers
have been analyzed here; only owner acceptance closes the pending cap64 gate.

## Concrete group-balanced estimator to approve

Use SAME cap128 primary samples, h, normalizer, group IDs and frozen support.
For each family c, month t, G_ct is the number of represented groups and n_ctg
the number of selected samples in group g. Give sample i weight
`a_i=1/(G_Ac*n_Acg(i))` on A, and analogous `b_j` on B. Both vectors sum to1.

Compute biased weighted MMD2 with diagonals:

`a^T K_AA a + b^T K_BB b - 2*a^T K_AB b`.

Equal-family outer mean remains unchanged. A group appearing in both months
keeps the same ID; each side's weights normalize separately. A multi-label group
keeps its ID and contributes within each represented family; labels are not used
to redefine similarity. No resampling, new encoder, kernel fit or graph rebuild.

Interpretation: a distribution with equal mass per REPRESENTED group within
family/month. It cannot recover missing/pre-cap groups, remove intergroup
dependence, create a null test or measure uncertainty. No group-balanced scores
or fusion weights were computed during this review.

## Finite execution batch after approval

1. Implement weighted-kernel primitive with float64, diagonal terms, block32,
   explicit finite normalized nonnegative weights and existing numerical-roundoff
   contract. Reject meaningful out-of-range values; no score-driven clipping.
2. Synthetic tests: uniform weights match unweighted primitive; singleton groups
   match primary; exact duplicate members preserve group mass/result; independent
   dense-reference agreement, symmetry, bounds, invalid weights, shared/mixed-label
   group handling. Keep original MMD implementation unchanged.
3. Run cap64 at primary h and same-sample group-balanced at primary h once each
   for all9 case/views:18 aggregates plus every family score. Use fixed original
   TRAIN parameters, no h/2×cap64 factorial or search. Save all deltas, manifests,
   weights/provenance and errors. Preserve primary; no winner selection.
4. Close descriptive sensitivity scope with limitations. Q/U/coefficient design
   remains separate; no Future, C, model inference or training in this batch.

Approval requested: accept cap64 review for all3 cases as PASS_WITH_LIMITATIONS;
approve the explicit weighted estimator above and its finite execution batch.
Basis for approval boundary: D023 sampler-review rule and Amendment01 separate
approval for changed aggregation/empirical target, not an implementation preference.
