# Historical logits/calibration preparation

## Update: E2 AVAILABLE; preparation complete (2026-10-17)
Owner run170643_356292 completed: all6 A3664/B3929 bundles, K42 and3 A-only
temperatures available. Receipt/output/source hashes and fit IDs/mapping/checkpoint
bindings verified. T values1.9568116821086452 /1.6570949683076943 /
1.536167830064124; no boundary optimum. State unchanged per trainer receipt.
Receipt:data/derived/historical_logits_preparation/20261016_170643_356292/receipt.json.
SHA256:276aa30fc864ac2967454977b65beda6b0ee98c40a194cfb146cf7598ab62c6a.
E1/E2/Final now AVAILABLE. Proposed next step is fixed-A-temperature historical
Q estimation on B in frozen support, not new inference or U/fusion computation.

## Update: E1 AVAILABLE (2026-10-17)
Owner run170412_203009 completed. Receipt, all output/input hashes, A-only IDs,
mapping and checkpoint/temperature bindings verified; no inference rerun.
E1 now has all6 bundles (A2023/B3480,K32) and3 temperatures:
1.7981106221917444 /1.5522668783890357 /1.4417744357961038, none at bounds.
Receipt:data/derived/historical_logits_preparation/20261016_170412_203009/receipt.json.
SHA256:7821c6ccbc6bc1b29dccb46c050f250e0d4a32b2ffe9e7eb913f93d560068e19.
The initial MISSING inventory below is preserved as historical evidence. E2 is
the next owner-run preparation; Final remains available for reuse.

Date: 2026-10-17 Asia/Saigon. D042. U remains diagnostic-only and
U_NOT_APPROVED_FOR_FUSION; no estimator rescue or new synthetic search.

## Current evidence

| Case, all3 views | K | Complete retained A | Complete retained B | B in frozen S | Logits A/B | Temperature | Preparation code |
|---|---:|---:|---:|---:|---|---|---|
| E1 |32|2,023|3,480|3,130 (S17)|MISSING in inspected locations|MISSING|READY, synthetic verified|
| E2 |42|3,664|3,929|3,659 (S25)|MISSING in inspected locations|MISSING|READY, synthetic verified|
| Final |51|3,799|4,332|4,064 (S25)|AVAILABLE, verified|AVAILABLE, verified|Reuse; no new inference needed|

AVAILABLE means saved arrays/metadata/hash bindings checked, not newly evaluated
accuracy or research validity. READY does not mean real prefix inference executed.

Audited receipt:
`data/derived/historical_logits_preparation/20261016_170159_059085/receipt.json`
SHA256:`9cd6293aefb50d596043c0555d38a99b76c92aa27757d752876f290446a2807f`.
Readback verified all output checksums and73 input/source hashes.
Earlier receipt170056 retained; final receipt binds completed synthetic tests.

Nine selected checkpoint file hashes checked without torch.load or model forward.
Prefix COMPLETE/epoch100/config receipts, K32/K42, cutoff and episode mapping
match; Final selected K51 checkpoint hashes match the existing registry and
prediction/temperature provenance. Final training chronology remains the prior
verified baseline evidence; this batch does not rebuild its training audit.

## Artifact discovery and reuse

Actual directory/file inventory, not absence inferred from registry:
- prefix_training_v1 and frozen_encoder_v1 contain no filenames identifying
  logits/predictions/temperatures/calibration. The extractor code saves embeddings,
  not complete logit bundles. New historical_logits_preparation contains audit
  receipts only before real execution.
- sbsmi_training has6 prediction/temperature candidate files; float_view_training
  has6. Candidates correspond to Final runs, including older unfiltered SBSMI
  outputs. Do not reuse these as K32/K42 prefix predictions.
- Read only the six registered filtered February/March prediction files and
  three registered filtered February temperatures. No C outcome or Future file
  opened. Missing is scoped to the inspected project artifact locations, not a
  claim that no copy could exist elsewhere on the computer.

Final sources: `docs/baseline_artifacts.json` entries and paths bound into receipt.
Temperatures: SBSMI2.5202034447760657; raw-byte2.211872879462139;
entropy1.9354895383696529. Fit-source hashes point to the matching full retained
February predictions. No refit. All saved IDs, order, labels and class order agree
across views, and sample sets match retained joint-group memberships. T and logits
bind to the same view checkpoint and preprocessing config.

## Implementation and verification

New `scripts/prepare_historical_logits.py` reuses frozen checkpoint loader and
image decoder. Its selection uses full G2 A_keep/B_keep plus AB_10_members, not
MMD matched/capped manifests. Frozen support and class mapping are hash-bound;
timestamps are checked in UTC; excluded samples do not re-enter. Original Final51
image storage labels are validated separately from episode target labels.

`prepare --case E1/E2 --execute-inference` is explicit CPU inference, all three
views. For each view it exports raw classifier logits from complete A, fits scalar
T from A standard NLL with existing bounds[.05,20], then exports complete B.
No loss/Q/U or fusion computation on B. No temperature fit receives B. Model eval,
inference_mode, state/buffer hash and gradient checks protect frozen CNNs. Image
content hashes must match joint-group sources and agree across views. Inputs and
sources are rehashed before a PASS receipt. Failures preserve FAILED.json; no retry.

Outputs in a new unique run: selection.json; each view's A.npz, B.npz and
temperature.json; receipt/checksums. Bundles retain IDs, episode class order and
labels. Existing MMD embeddings are never used. Final inference is deliberately
not offered by this CLI because verified artifacts already exist.

6/6 targeted tests PASS (0.040s): label mapping, IDs/order, date/retention guards,
A-only fit, CPU random-model state/shapes K32/K42/K51, synthetic orchestration and
saved bundle readback. No registered CNN loaded in tests; no real-image inference.
Historical audit is metadata/saved-logit verification only, not a new research gate.

## Exact commands for owner execution after approval

Run from `C:\Users\hoang\Desktop\TRAMIF`. One command handles all3 views and both
months for its episode, including A-only calibration. No separate calibration
command or manual choice of latest directory is needed.

```powershell
.\.venv\Scripts\python.exe -B -m scripts.prepare_historical_logits prepare --case E1 --execute-inference
```

After E1 completes successfully and E2 execution is approved:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.prepare_historical_logits prepare --case E2 --execute-inference
```

Do not run E2 automatically if E1 fails. Final needs neither command. The only
immediate next execution proposed is E1 preparation above. This batch did not run
either command, compute Q/U, select coefficients, access Future, commit or push.
