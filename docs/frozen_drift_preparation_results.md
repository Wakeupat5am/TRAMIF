# TRAIN-only normalization and bandwidth — PASS

2026-10-16. Owner requested continuation after verification of extracted arrays.
Scope: fit TRAIN parameters, transform saved A/B. No inference or observed MMD.

| Case | View | Kept/input | Rank | h | Condition number |
| --- | --- | --- | ---: | ---: | ---: |
| E1 | SBSMI | 2048/2048 | 2048 | 1.7327566555 | 191.4303 |
| E1 | Raw-byte | 512/512 | 512 | 1.5111827103 | 25.5165 |
| E1 | Entropy | 512/512 | 512 | 1.5059474101 | 57.3255 |
| E2 | SBSMI | 2048/2048 | 2048 | 1.7143463266 | 112.3194 |
| E2 | Raw-byte | 512/512 | 512 | 1.4923524833 | 21.8748 |
| E2 | Entropy | 512/512 | 512 | 1.5067792608 | 40.1502 |
| Final | SBSMI | 2048/2048 | 2048 | 1.7147100536 | 73.6846 |
| Final | Raw-byte | 512/512 | 512 | 1.4976050341 | 18.2584 |
| Final | Entropy | 512/512 | 512 | 1.5007592671 | 37.0197 |

Use full precision in artifacts, not rounded table values. All dimensions retain
TRAIN std>1e-8; no all-masked/nonfinite/h<=0 condition encountered. Full rank and
finite conditioning are diagnostics, not proof of statistical stability.

Existing G3-B utilities unchanged: population moments with equal total mass per
TRAIN family (all K32/42/51), TRAIN coordinate masking, fixed within-family pairs,
rho=distance squared/retained dimensions, lower weighted median with equal family
pair mass. Pair IDs match saved G3-B train_pairs.csv exactly. No A/B fit or rescue.
Bandwidths are kernel parameters, not observed Drift scores.

Parameters NPZ are restored and hash checked; restored TRAIN transforms exactly
match original objects. A/B transform uses those restored frozen parameters;
parameter payload hash unchanged after all transforms. Original source embeddings,
checkpoints, protocol and utilities not modified. No CNN/optimizer/C/Future read.
BLAS threads bounded to1 as CPU resource control, not a fitted hyperparameter.

One new targeted synthetic test PASS (0.010s): parameter save/load, altered-hash
rejection, and A/B transforms cannot modify fit. Existing verified G3-B utilities
were reused without rerunning broad audits/suites. Independent readback confirmed
64 output file hashes and27 float64 arrays' shape/finite/payload hashes PASS.

Artifacts: `data/derived/frozen_drift_preparation/20261016_122145_477857/`.
report.json SHA256: `78b31028bd48a9dcc4fc18d4191810197b295e007e1c7a4b2eb91c8de757b276`.
Each case/view has normalizer.npz, parameters.json (sample order per stage),
bandwidth.json (full precision/pairs/distances), three normalized NPY arrays and
checksums. Top-level checksums.json binds all64 output files. Use only PASS runs.
New code: scripts/prepare_frozen_drift_measurements.py;
test: tests/test_frozen_drift_measurements.py.

Next proposed: descriptive class-conditional MMD on frozen A/B, with approved
support/equal-family aggregation and claim limits. No observed MMD, sensitivity,
Q/U or weights computed here. Normalized embeddings and separate h values do not
establish cross-encoder statistical scale equivalence, significance or reliability.
