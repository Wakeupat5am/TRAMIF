# Observed descriptive Drift — computation complete

2026-10-16. Reused G3-A biased RBF MMD2 float64 with diagonals, block32,
TRAIN-only parameters, frozen support and equal-family aggregation. No retuning.

| Case: A to B | Support | SBSMI MMD2 | Raw-byte MMD2 | Entropy MMD2 |
| --- | ---: | ---: | ---: | ---: |
| E1: Nov to Dec2019 | 17 | 0.0421416990 | 0.0337694234 | 0.0364646844 |
| E2: Dec2019 to Jan2020 | 25 | 0.0475142430 | 0.0364235521 | 0.0379289536 |
| Final: Feb to Mar2020 | 25 | 0.0443940142 | 0.0522265584 | 0.0598546191 |

Values are descriptive distances under specific encoders/kernels. They are not
p-values, null-standardized scores or demonstrated predictions of future errors.
Separate bandwidths and numeric scaling do not establish comparable statistical
scales between views. E1/E2/Final use different encoders/support, so rows do not
form a common-encoder temporal trend. No intrinsic view-stability ranking claimed.

201 family scores saved, with kernel means, raw/clipped MMD2, matched sample counts,
operational groups/loss/concentration and disarmed-file size quartiles per side.
All scores finite/in[0,2]; zero negative-roundoff clips. Saved aggregates verified
independently by equal-family means; matched counts and post-cap groups>=5 checked.
D023 accepted loss/concentration/dependence limitations remain. Sensitivities not
run, and TLSH groups are not proven independent.

Original artifacts:
`data/derived/observed_descriptive_drift/20261016_122644_350078/`.
report.json SHA256: `873d2b893d07fb41d74415299d116627c70317b3375abc6fb07d28b61019b393`.
Files: report.json (full precision/provenance), per_family.csv, checksums.json.

**Manifest limitation:** original writer accidentally included checksums.json's
own empty-file hash. That self-entry is invalid. Report/CSV entries match their
actual bytes. All originals preserved; supplemental independent verification at
`data/derived/observed_descriptive_drift_verification/20261016_122803_197377/receipt.json`
binds actual original files and documents defect, checks201 family results/kernel
means/bounds and9 aggregates. Status PASS_DATA_WITH_MANIFEST_LIMITATION. Use the
supplemental receipt with original outputs, not the invalid manifest self-entry.
No repeated MMD to replace results. Writer now excludes self and regression test
passes.3 targeted wrapper tests PASS (0.007s); original MMD primitive unchanged.

New files: scripts/run_frozen_descriptive_drift.py,
tests/test_frozen_descriptive_drift.py. No training/inference, C/Future, Q/U,
weights, coefficients, significance tests or sensitivity analysis executed.

Next proposed: review bounded predeclared sensitivity execution (bandwidth h/2,2h
and sampling controls under fixed primary; no selecting a winning variant).
U estimation/quality/fusion remains a separate approved-methodology gate.
