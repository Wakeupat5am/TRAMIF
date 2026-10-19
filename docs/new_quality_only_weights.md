# D045 — New Quality-only baseline weights

2026-10-17. Engineering PASS; weights frozen, outcome evaluation NOT EXECUTED.
Owner requested continuation after D044's explicit New-Q-only next-step proposal.

Use full-precision fixed historical class-balanced losses, a=1:
`w_v=exp(-L_v-max(-L))/sum_u exp(-L_u-max(-L))`.
Fusion recipe averages calibrated K-class probabilities with these static weights.
No weight tuning, Drift/uncertainty term, coefficient search or prediction metrics.

| Case | SBSMI | Raw-byte | Entropy |
|---|---:|---:|---:|
| E1 |.47334653674532445|.2446128520684008|.2820406111862747|
| E2 |.4225266637757557|.2805146225080068|.2969587137162376|
| Final |.4079024636580072|.2984266072023347|.2936709291396582|

SBSMI receives the largest weight in each case because its historical balanced
loss is lowest. This does not establish future reliability, improvement versus
Uniform or superiority of one representation. Legacy February OOF Quality weights
are unchanged; New Q uses B history and different support. A comparison must keep
information budgets explicit, and cannot evaluate these weights on B as an unseen
holdout. E1/E2 outcome work requires its own correct prefix predictions and manifest.

5 targeted tests PASS: uniform-loss answer, named-view alignment and independent
reference, monotonic ordering, additive-shift invariance, numerical extremes and
invalid input rejection. Stored losses/labels/masks and equal-family means checked
against D044, with explicit support K32/S17, K42/S25, K51/S25 and B counts3130,
3659,4064. Current Q source/input hashes rechecked; no model, fitting or new Q run.
Independent readback verifies all output/source hashes and math.exp weights/sums
to absolute1e-15. Rounded table values must not replace saved artifact precision.

Artifacts:`data/derived/new_quality_only_v1/20261017_020915_979476/`.
Contains config, E1/E2/Final_weights.json, completion.json and checksums.json.
Each freeze includes losses, ordered views, support/mapping, selection hash and
checkpoint/calibration/source prediction provenance. Completion SHA256:
`7c7158308955b5e8b54d098af12e221ae06f38685cbb7509fd03c2c778f5d852`.

Next proposed work: prepare the E1 outcome-C comparison of three calibrated
single views, Uniform and this frozen New Q, with identical retained samples and
mapping. Do not substitute E2/Final predictions for E1's prefix model. No C/Future
input opened here. U remains diagnostic-only and excluded from fusion. No training,
inference, new MMD/U, commit/push or legacy artifact changes.
