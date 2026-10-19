# Bandwidth sensitivity — technical PASS, sampling diagnostics pending

2026-10-16. Fixed factors0.5/2 from V1; no outcome-driven selection. Frozen TRAIN
normalizer, matched samples, support and h unchanged in primary artifacts.

| Case | View | h/2 | Primary h (saved) | 2h |
| --- | --- | ---: | ---: | ---: |
| E1 | SBSMI | 0.06590133 | 0.04214170 | 0.02405852 |
| E1 | Raw-byte | 0.05240940 | 0.03376942 | 0.01938599 |
| E1 | Entropy | 0.05592677 | 0.03646468 | 0.02115894 |
| E2 | SBSMI | 0.07476058 | 0.04751424 | 0.02701562 |
| E2 | Raw-byte | 0.05530415 | 0.03642355 | 0.02115892 |
| E2 | Entropy | 0.05757275 | 0.03792895 | 0.02213400 |
| Final | SBSMI | 0.06928184 | 0.04439401 | 0.02536305 |
| Final | Raw-byte | 0.08051632 | 0.05222656 | 0.03004375 |
| Final | Entropy | 0.08896635 | 0.05985462 | 0.03535940 |

MMD2 magnitude changes +48.64..57.34% at h/2 and -43.14..40.92% at2h.
Within-case ascending descriptive orders remain raw-byte/entropy/SBSMI forE1/E2
and SBSMI/raw-byte/entropy forFinal. This supports order stability only under
these two tested bandwidth perturbations. It does not calibrate scales between
encoders, establish statistical independence/significance, imply intrinsic view
superiority or validate future-error prediction. Absolute score dependence matters
for any later coefficient design, still unapproved. No winner/threshold chosen.

Artifacts: `data/derived/drift_bandwidth_sensitivity/20261016_132622_260352/`.
report.json SHA256: `47fb132e52939cd1460718a72b68d76d4500163c20bcc9f53e49ac13cde3e80f`.
18 aggregates/402 family scores, primary differences, source/input hashes and
config stored. Checksums exclude self; independent readback checks hashes, bounds,
family counts and equal-family aggregates PASS.2 targeted synthetic tests PASS.
Primary report pinned directly; old self-manifest defect not silently consumed.

## Cap64 metadata — REVIEW REQUIRED

`data/derived/cap64_metadata_review/20261016_132758_657700/` contains locked-source
checksums, proposed nested subset, all counts/concentration and STOP flags.

| Case | Samples/side | Min groups after | Cells below5 | Additional group loss>=5 | Total loss>=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | 823 | 8 | 0 | 11 | 19 |
| E2 | 1234 | 11 | 0 | 17 | 30 |
| Final | 1244 | 7 | 0 | 17 | 30 |

Cells mean family/month. Additional loss compares cap64 to primary128; total loss
compares to original pre-cap. No support failure, but D023 requires sampler review
for lost>=5. No cap64 MMD executed. Do not silently drop families or choose seeds.

Group-balanced is not yet defined sufficiently by V1. Concrete proposal in
docs/drift_sampling_sensitivity_plan.md gives equal represented-group mass within
family/month on unchanged primary samples, using a weighted kernel mean. It changes
the empirical target; owner approval required under Amendment01 before execution.
No group-balanced implementation, Q/U, fusion weights, model inference or Future.
