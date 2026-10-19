# G3-C-F — Full TRAIN image/report preflight

Ngày: 2026-10-16 (Asia/Saigon). **E1 PASS; E2 PASS.**
Scope: operational verification of cached TRAIN images/reports, not CNN training,
inference, convergence validation or research validity. Owner approved only this gate.

## 1. Execution and counts

Executed once each, sequentially, with the existing implementation unchanged:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.episode_training_dataset --case E1
.\.venv\Scripts\python.exe -B -m scripts.episode_training_dataset --case E2
```

Both commands exited 0. E2 started only after E1 PASS. Repo venv Windows Store
Python ran outside the sandbox under approval review; no dependency installation.

| Episode | View | Expected samples | Verified image/report pairs | Classes | Result |
| --- | --- | ---: | ---: | ---: | --- |
| E1 | SBSMI | 7034 | 7034 | 32 | PASS |
| E1 | Raw-byte | 7034 | 7034 | 32 | PASS |
| E1 | Entropy | 7034 | 7034 | 32 | PASS |
| E2 | SBSMI | 9803 | 9803 | 42 | PASS |
| E2 | Raw-byte | 9803 | 9803 | 42 | PASS |
| E2 | Entropy | 9803 | 9803 | 42 | PASS |

Total 50,511 verified episode×view image/report pairs. This is not a count of
distinct binaries: episodes overlap and each sample has three representations.
Errors: 0. Missing/skipped/repaired samples: 0. No retries, regeneration, selection
changes or edits to images/reports. Every expected ID has one receipt per view.

## 2. Verified contracts

- Before image reads, source hashes matched G3-C-I verification evidence; frozen
  G2 TRAIN manifests, episode specs, memberships and mappings passed their pinned
  checksums. E1 cutoff is 2019-11-01 UTC; E2 cutoff is 2019-12-01 UTC, exclusive.
- Exact canonical IDs, order, timestamps, families, retention and episode targets
  matched G2. Three views have identical ordered TrainingSample records. Stored
  Final51 class index is checked separately; episode targets remain K32/K42.
- Every allowlisted image/report existed and passed file SHA-256 validation.
  Reports matched sample identity, timestamp, family, config, source mapping,
  PASS status and G2 disarmed-content SHA. Thus all three reports identify the
  same source content. Binary bytes were not reopened/rehashed in this gate.
- SBSMI: PNG L64×64, decoded pixel checksum, uint8 converted to float32 /255.
  Raw-byte/entropy: NPY float32 64×64, finite values in [0,1], no additional
  scaling. Returned tensors [1,64,64]. Preprocessing/config identity was verified;
  this does not regenerate images or independently re-prove the transform.
- Path construction used only frozen TRAIN IDs. No A/B/C manifests or images,
  outcome predictions/losses, raw metadata CSV, binaries or Future Test were read.
  Episode specs contain stage dates, not sample access to those stages.
- After both runs, all 12 metadata/config/mapping input-file hashes and source
  bindings matched the before snapshot. Receipt hashes, current bindings and all
  ordered IDs were checked again. This final verification read metadata/receipts,
  not a second scan of images. Image/report hashes describe their verified bytes
  at preflight time; the trainer must still enforce per-read receipt checks.

Common ordered-ID hashes across three views:

- E1: `8348b2143e4e38664b00b5dac0c3b3569d392db86dec60469f438ae25081e9e0`
- E2: `9e20b8a36b7fe384a4e6999405f4d12e4261b92795ea827ab6651ffce4d5aa6f`

No unit suite rerun: source remained identical to G3-C-I's 20/20 tested versions.
This gate adds full TRAIN image/report integration evidence, not a new unit-test
total or evidence that training/encoder extraction works.

## 3. Immutable receipts and checksums

Paths relative to `C:/Users/hoang/Desktop/TRAMIF/`; each preflight used a new run
directory and the receipts contain per-image/report hashes and provenance.

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| E1 PASS receipt | `data/derived/prefix_preflight_v1/E1/20261016_005713_190675/receipt.json` | `2942ea823bc521a7b4a249f42f977178ea6de9613dc206a9ecb951d2e0fcd76f` |
| E2 PASS receipt | `data/derived/prefix_preflight_v1/E2/20261016_005833_872347/receipt.json` | `45a6bdd174cf3c2caff907d1f85494bae3c2af5bf104d9b176e3aac86d4b9fd3` |
| Before source/input verification | `data/derived/g3cf_preflight_verification/20261016_005659_113736/before.json` | `741fadbdf47b9d5099866ce2e546cee746db24e040a4a12342f041483f9b75cd` |
| After verification and count summary | `data/derived/g3cf_preflight_verification/20261016_005659_113736/after.json` | `03ae0b59ea932071a7d04a5758ee0c00525a3b8a85d7887e9c37b58403faf343` |

Implementation source hashes retained:

- `scripts/episode_training_dataset.py`:
  `27b4422a49f5e45388594c26cec5876ed9dfaf7a057851dc2e2da67c637c3a9c`
- `scripts/train_prefix_view.py`:
  `2106df5388e2c33d65c8320be4c44cbc57cc15f9a9fc156fe85532bfff266f57`

## 4. Remaining gates before GPU training

**No preflight blocker found.** These remain separate, unverified or unapproved:

1. Explicit owner approval for six bounded GPU runs and compute budget. The CLI
   execution flag and PASS receipts do not constitute that approval.
2. Resource readiness: available RAM/VRAM, free disk/headroom, power/thermal state,
   execution/interruption policy. No resource benchmark was run here.
3. Owner review of G3-C-I parity evidence: current versions/Adam defaults match
   baseline source/config evidence, but saved baseline optimizer state has not
   been inspected. No registered checkpoint was loaded to resolve this.
4. Actual GPU loop, optimization, prefix checkpoints and extraction remain
   unverified. Fixed epoch100/seed42 and frozen data rules cannot be changed to
   rescue a failed run without review. Changed code/bindings invalidate receipts.

**One proposed next step:** a separate pre-training readiness/resource review to
decide whether to authorize the six prefix runs. Do not start training automatically.

Only this report, PROJECT_PROGRESS and DECISION_LOG were changed; new evidence
artifacts were saved under ignored data/derived. Source, legacy files, models,
G2 manifests, V1 and checkpoints unchanged. No GPU training, optimizer steps,
CNN inference, embeddings, MMD/Q/U, Future Test, commit or push. **STOP for review.**
