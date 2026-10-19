# G3-C2 — Minimal verification: bounded runtime PASS

## Approved repair and rerun (2026-10-16)

Canonical JSON comparison fixes tuple/list encoding without changing numerical
values, class mappings, cutoff checks or checksum guards. Regression tests reject
changed betas/K/cutoff/mapping, added fields, string coercion and NaN.
9/9 targeted tests PASS in0.259s; no broad suite/audits repeated.

| Case | K | SBSMI | Raw-byte | Entropy |
| --- | ---: | --- | --- | --- |
| E1 | 32 | PASS [3,2048] | PASS [3,512] | PASS [3,512] |
| E2 | 42 | PASS [3,2048] | PASS [3,512] | PASS [3,512] |
| Final | 51 | PASS [3,2048] | PASS [3,512] | PASS [3,512] |

All use CPU, eval(), requires_grad_(False), inference_mode(). Entire state_dict
hash including BatchNorm buffers is unchanged. Repeated outputs exactly equal;
single-versus-batch maximum absolute error0 for all nine (fixed tolerance1e-5).
Finite outputs, no gradients, three-view sample IDs/order/content hashes PASS.
Exactly first3 frozen TRAIN-reference rows per case; no A/B/C/Future forward.
No optimizer/training, MMD, normalization/bandwidth fitting or bulk extraction.

Receipt: `data/derived/frozen_encoder_v1/20261016_110812_281815/receipt.json`
SHA256: `0d995d66f717397c823cb3a0b412dcb785cc95126e1d76be4f5cd7bad3a67af1`.
Companion checksums.json binds receipt; receipt binds checkpoints, manifests,
source code and image/report bytes. Previous failure remains immutable.

## Owner-run extraction commands (not executed here)

Each command extracts all three views for ONE frozen selection into a new unique
directory under data/derived/frozen_encoder_v1. No fitting or re-selection.
TRAIN reference is needed for subsequent TRAIN-only normalization/bandwidth;
A/B embeddings alone do not supply those parameters. Commands run sequentially
and stop at first nonzero exit; no retry/resume or method change.

```powershell
Set-Location -LiteralPath 'C:\Users\hoang\Desktop\TRAMIF'
foreach ($caseName in @('E1', 'E2', 'Final')) {
    foreach ($stageName in @('train-reference', 'A', 'B')) {
        & .\.venv\Scripts\python.exe -B -m scripts.frozen_encoder_extraction extract --case $caseName --stage $stageName
        if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $caseName / $stageName. STOP." }
    }
}
```

Expected rows per view: E1 reference3266/A1137/B1137;
E2 reference4197/A1734/B1734; Final reference5354/A1768/B1768.
Outputs: sbsmi.npy, raw-byte.npy, entropy.npy, receipt.json with ordered samples
and mapping, checksums.json. Use only runs with PASS receipt and valid checksums;
FAILED runs or partial arrays must not be consumed. These complete selections,
actual normalization/mask/bandwidth and downstream MMD remain UNKNOWN until run.
No inference performance or scientific claim follows from the bounded PASS.

## Historical first attempt — BLOCKED (retained below)

2026-10-16. No legacy models/trainers/artifacts modified.

| Check | Result |
| --- | --- |
| Synthetic endpoints K32/K42/K51, dimensions, state/repeatability | PASS |
| Alignment, forbidden stages/chronology, nonfinite/eval/freeze guards | PASS |
| Targeted suite | 7/7 PASS, 0.317s |
| E1-SBSMI real checkpoint loader | FAIL: config comparison serialization |
| Other eight real checkpoint loaders | UNKNOWN; not attempted |
| Actual TRAIN embeddings/state invariance/alignment | UNKNOWN; stopped before forward |
| Bulk extraction readiness | BLOCKED; draft CLI must not be run |

Failure receipt: `data/derived/frozen_encoder_v1/20261016_105811_828183/FAILED.json`.
The hash-pinned checkpoint stores `config.adam_defaults.betas` as tuple
`(0.9, 0.999)`; adjacent JSON stores list `[0.9, 0.999]`. Raw Python equality
in the new loader therefore fails. Diagnostic-only CPU deserialization verified
canonical JSON bytes equal and canonical hash matches checkpoint config hash.
No checkpoint corruption is indicated. No forward, A/B/C/Future access,
optimizer steps, MMD or retraining. No retry or repair after explicit STOP.

Proposed fix (NOT APPLIED): compare canonical config bytes/hashes while keeping
all semantic mapping/cutoff/checkpoint checks; add tuple/list regression test.
Then rerun bounded verification, not bulk extraction:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_frozen_encoder_extraction.py" -v
.\.venv\Scripts\python.exe -B -m scripts.frozen_encoder_extraction verify
```

Do not run the second command until the fix is reviewed. Draft extract choices
are limited to E1/E2/Final and train-reference/A/B, using unchanged G3-B plans.
Normalization and bandwidth are not fitted by extraction; future fitting must
use only frozen TRAIN reference. These contracts are not yet real-data validated.
