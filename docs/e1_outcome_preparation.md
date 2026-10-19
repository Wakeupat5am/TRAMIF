# D046 — E1-C comparison preparation

## D047 execution update — 2026-10-17

Owner authorized continuation of the explicit export. Frozen CPU extraction
completed for all3 views, each(3695,32), using the existing tested implementation.
No new source change; prior5 targeted tests remain the relevant code evidence.
Saved arrays independently checked: finite logits, exact common IDs/order,
K32 class order/labels, original frozen-state receipts and output hashes PASS.
Non-image inputs were rehashed independently; image hashes were rechecked inside
the extractor before the completion receipt. No C losses/prediction metrics read.

Run:`data/derived/e1_outcome_preparation/20261017_032743_694605/`.
Receipt SHA256:`9cd2ceac5bc7c8c947c2461d339032a7f6d8e34c11a60c552446707b2cfb3a34`.
Readback:`data/derived/e1_outcome_preparation/20261017_032743_694605_verification/receipt.json`.
Existing readiness/execution-pending statements below preserve earlier history.
No training/refit/U/Drift/Future/commit/push. Next is the previously documented
owner selection-scope decision before opening outcome losses, not more extraction.

### Why these steps matter

1. Check existing receipts: avoid duplicate computation and wrong-run reuse.
2. Use E1 encoders for January: preserve training cutoff and episode meaning;
   E2/Final logits would introduce different training history/capacity/mapping.
3. Verify identities/class order: prevent mixing probabilities for different
   samples/classes; a successful array operation alone cannot detect this error.
4. Preserve receipt/provenance: make the exact data/model/result chain reusable.
5. Lock selection before outcome scores: protect the distinction between method
   selection and its evaluation. Changing the candidate rule after viewing E1
   losses must be recorded as adaptive development, not untouched evidence.

2026-10-17. **METADATA_READY; 5 targeted tests PASS.** No real C inference or
prediction metrics executed. Preparation follows owner's continuation of D045.

Frozen G2 C_keep hash confirms3,695 January2020 samples,32 observed E1 families.
All retained C samples are comparison inputs; S17 was used for Quality estimation,
not an outcome population restriction. T comes from November A; Q/weights come
from December B. Frozen E1 K32 checkpoints and mapping bind across views.
No A/B IDs occur in C. This checks identity separation, not statistical independence.
No grouping policy or signal graph is rebuilt using C.

Existing E1 preparation has A/B bundles only; no C bundle there. E2's January B
logits use E2 K42 encoders and cannot stand in for E1 outcome predictions.
No Final/E2 outcome prediction arrays were loaded during this preparation.

## Implementation

New `scripts/prepare_e1_outcome.py`:
- `inspect`: metadata plus frozen weights/T/checkpoint-hash checks, no model load.
- `extract --execute-inference`: explicit CPU export using unchanged E1 frozen
  model loader, decoder and classifier-logit function; 3 views, all3,695 samples.
  Records state/buffer invariance, content-hash agreement and full input hashes.
  Saves raw logits/labels/IDs/class order; no C losses or evaluation metrics.
- `compare_fixed`: pure synthetic-tested helper for the eventual five-method
  comparison; no real evaluation command is enabled in this batch.

Synthetic tests cover January-only retention guards, mappings, strict three-view
alignment, probability-mixture reference, label changes leaving mixed probabilities
unchanged, absent-class macro-F1 denominator and uniform-weight equivalence.
5/5 PASS (0.012s), no registered models run.

Metadata receipt:
`data/derived/e1_outcome_preparation/20261017_021201_397264/receipt.json`.
SHA256:`f9f4db845a3e33fa5c15e333cfad9af90a0852233392f726ec610cc1be2dd420`.
All recorded source/input and output hashes independently rechecked PASS.

## Next execution and scientific boundary

From repository root, owner can explicitly execute C-logit preparation:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.prepare_e1_outcome extract --execute-inference
```

One command exports all three views to a new immutable run; no training, fit,
weight tuning, U, D or Future access. The command was not executed in this batch.

V1 §8.2 requires the coefficient candidate set to be locked before E1 outcome
losses are examined for method selection. D/U weighting/selection choices remain
unapproved; U is diagnostic-only. Before actual comparison metrics are opened,
owner must either lock the relevant selection protocol or explicitly make this a
Q-only outcome evaluation and document that later D choices cannot claim untouched
E1 selection evidence. Do not silently treat an already-seen outcome as untouched.
This is an existing V1 dependency, not a newly invented audit gate.

When authorized, planned comparison is calibrated SBSMI/raw-byte/entropy, Uniform
and fixed New Q on identical C IDs with all32 outputs. Metrics: accuracy, supported
macro-F1, fixed32 macro-F1, balanced accuracy, NLL, multiclass Brier and ECE15;
confusion/per-class data saved. No significance or Future-generalization claim from
one retrospective episode. Code/metadata readiness does not establish research
validity. No commit/push.
