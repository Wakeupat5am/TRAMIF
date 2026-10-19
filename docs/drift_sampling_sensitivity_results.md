# Descriptive Drift sampling sensitivity — 2026-10-16

Status: IMPLEMENTED, UNIT TESTED and EXPERIMENTALLY VERIFIED for these bounded
numerical diagnostics. Predictive reliability is NOT RESEARCH VALIDATED.

Owner accepted D036's cap64 review with limitations and the explicit group-mass
estimator. Primary outputs, frozen support S17/25/25, normalization and bandwidth
remain unchanged. No selection among diagnostic arms is performed.

## Results

All values below are biased MMD², equal-family means (rounded for display).

| Case | View | Primary | Cap64 | Group-balanced |
|---|---|---:|---:|---:|
| E1 | SBSMI | .04214170 | .04576320 | .02959635 |
| E1 | Raw-byte | .03376942 | .03735057 | .02422601 |
| E1 | Entropy | .03646468 | .04013508 | .02648453 |
| E2 | SBSMI | .04751424 | .04952240 | .03180793 |
| E2 | Raw-byte | .03642355 | .03846931 | .02598159 |
| E2 | Entropy | .03792895 | .03991642 | .02440323 |
| Final | SBSMI | .04439401 | .04608704 | .02955484 |
| Final | Raw-byte | .05222656 | .05355233 | .03709054 |
| Final | Entropy | .05985462 | .06125048 | .04031864 |

Cap64 increases scores2.33–10.60%, preserving within-case view order. Group
balancing lowers scores27.37–35.66%; E2 entropy becomes lower than raw-byte.
These are descriptive observations, not evidence one model is more reliable.
Smaller empirical samples can change biased MMD through self/pair weighting;
the increase is not automatically stronger underlying distribution shift.

## Estimator and verification

Group-balanced uses identical primary samples, with sample mass
1/(G*n_g) within each family/month. G counts represented groups, n_g their selected
members. Shared-month/multi-label IDs are retained; sides normalized separately.
Weighted biased MMD includes all diagonals; the outer mean weights families equally.
This targets represented-group mass, not every group in the full pre-cap cohort.

7 targeted tests PASS (0.025s): dense independent reference, uniform/singleton
parity, repeated identical group-member invariance, symmetry/bounds, identical
sets, invalid inputs, ID-based alignment and metadata mismatch rejection.
Command: `.venv\Scripts\python.exe -B -m unittest tests.test_weighted_descriptive_mmd tests.test_drift_sampling_sensitivity -v`.

All18 aggregates and402 per-family scores saved. Independent readback verified
checksums, finite bounds, equal-family means and134 normalized group-weight cells.
Full precision, family-level contrasts, counts and source/input hashes are in:
`data/derived/drift_sampling_sensitivity/20261016_135024_796782/report.json`.
SHA256: `9136a8913403535b19e0f73e2075679294df8d308b75888d7b45563a4cf92e17`.
Config, sample_weights.json and checksums.json accompany it.
Separate immutable readback receipt:
`data/derived/drift_sampling_sensitivity/20261016_135024_796782_verification/receipt.json`.

## Limits and stopping point

Cap64 review retains group loss/concentration/shared-month/multi-label limitations
documented in cap64_focused_review.md. No below-five support cell is accepted.
Neither diagnostic proves independent groups, null-standardized Drift, comparable
statistical scales across encoders or predictive value beyond Quality.
Weighting changes the empirical target; a smaller MMD is not a better estimator
by itself. All scores, including the E2 order reversal, are retained.

Bandwidth and sampling sensitivity scope is now completed. No CNN inference,
training, Q/U, fusion weights, C/Future access, commit or push occurred. Next
proposed step: review the historical Quality uncertainty estimator contract and
available inputs before authorizing its implementation. No automatic next gate.
