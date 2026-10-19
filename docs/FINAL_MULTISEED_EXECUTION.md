# Final seeds43–46 execution sheet

Purpose: measure training-seed variability without choosing a best seed. All five
seeds stay in the report, including poor outcomes/failures. Existing seed42 is reused.
This sheet prepares commands; it is not GPU/post-training authorization.

D053 update2026-10-17:owner has now approved operator-started execution of all12
training runs and the historical postprocessing below, after all three checkpoints
of the corresponding seed PASS verification. READY_FOR_OPERATOR_EXECUTION.
The D052 commands/wrapper are unchanged. Codex does not start runs or poll epochs.
Run sequentially; no automatic retry/resume or seed exclusion. Approval flags below
acknowledge D053; learned-static empirical fitting and Future remain unauthorized.

## Status and immutable contracts

| Component | Status |
|---|---|
| Three legacy seed42 epoch100 checkpoints | AVAILABLE, existing registered hashes |
| seeds43,44,45,46×SBSMI/raw-byte/entropy | MISSING:12 checkpoints |
| Trainer CLI/output binding | READY_FOR_OPERATOR_EXECUTION under D053 |
| Metadata/existence preflight | PASS3 views,18,061 IDs,K51; not a repeated full image audit |
| Seed-specific CPU postprocessing | D053 APPROVED; NEEDS3 VERIFIED CHECKPOINTS + operator start |
| Learned static | Synthetic-tested proposed solver; EMPIRICAL OWNER_APPROVAL_REQUIRED |
| Future/statistics | BLOCKED pending approvals/completeness/coverage |

All runs: TRAIN2019-08–2020-01,18,061 unique IDs,K51; seed-specific fresh CPU
initialization and loader RNG; batch32,565 batches/epoch,56,500 optimizer steps
per run,100 epochs, fixed epoch100. Adam lr1e-4,weight_decay1e-6,betas(.9,.999),
eps1e-8 and the installed verified defaults; unweighted CE; no scheduler, AMP,
augmentation, resume or warm start. Last batch13; no samples dropped.

Architecture/preprocessing match legacy: SBSMI MalSBSLCNet140,467 parameters;
raw-byte/entropy each CompactViewNet59,059. Images float32(1,64,64), range[0,1];
raw/entropy are already normalized, SBSMI divided by255. Original models unchanged.
Code/default parity is verified. Legacy saved optimizer-state parity is UNKNOWN;
this batch did not deserialize legacy optimizer states or assert bitwise GPU identity.

Output per run: `data/derived/final_seed_training/seed_<seed>/<view>/<unique-run>/`
with source snapshots,config.json,initial_state.pt,100 checkpoint/metadata pairs,
COMPLETE.json. A failure leaves FAILED.json; never auto-retry/resume or pick latest.
Verification demands exactly one COMPLETE candidate and checks config/mapping/
cutoff/source hashes/epoch100 metadata and checkpoint bytes. An ambiguous run stops.

## Commands

Run from `C:\Users\hoang\Desktop\TRAMIF`. First read-only metadata preflight:

```powershell
& .\automation\final_multiseed.ps1 -Stage preflight
```

After separate approval, check AC/non-discharging, adequate available RAM/VRAM and
≥2GiB free disk before EACH run. No concurrency. This batch observes AC online
and about78.8GiB disk free, but these are time-sensitive, not future guarantees.
Live resource verification must be repeated by the operator at execution time.

Exactly12 training invocations below. Each invokes the tested Python train CLI
with a freshly hash-pinned view preflight. Flags acknowledge approval; they do
not grant it. Execute individually; STOP after any nonzero exit/FAILED receipt.

```powershell
& .\automation\final_multiseed.ps1 -Stage train -Seed 43 -View sbsmi -OwnerApprovedTraining
& .\automation\final_multiseed.ps1 -Stage train -Seed 43 -View raw-byte -OwnerApprovedTraining
& .\automation\final_multiseed.ps1 -Stage train -Seed 43 -View entropy -OwnerApprovedTraining
& .\automation\final_multiseed.ps1 -Stage train -Seed 44 -View sbsmi -OwnerApprovedTraining
& .\automation\final_multiseed.ps1 -Stage train -Seed 44 -View raw-byte -OwnerApprovedTraining
& .\automation\final_multiseed.ps1 -Stage train -Seed 44 -View entropy -OwnerApprovedTraining
& .\automation\final_multiseed.ps1 -Stage train -Seed 45 -View sbsmi -OwnerApprovedTraining
& .\automation\final_multiseed.ps1 -Stage train -Seed 45 -View raw-byte -OwnerApprovedTraining
& .\automation\final_multiseed.ps1 -Stage train -Seed 45 -View entropy -OwnerApprovedTraining
& .\automation\final_multiseed.ps1 -Stage train -Seed 46 -View sbsmi -OwnerApprovedTraining
& .\automation\final_multiseed.ps1 -Stage train -Seed 46 -View raw-byte -OwnerApprovedTraining
& .\automation\final_multiseed.ps1 -Stage train -Seed 46 -View entropy -OwnerApprovedTraining
```

Verify a completed seed without model inference (change43 to44/45/46):

```powershell
& .\automation\final_multiseed.ps1 -Stage verify -Seed 43
```

After that seed's three checkpoints verify AND separate real postprocessing
approval, execute CPU historical processing:

```powershell
& .\automation\final_multiseed.ps1 -Stage post -Seed 43 -OwnerApprovedPostTraining
& .\automation\final_multiseed.ps1 -Stage post -Seed 44 -OwnerApprovedPostTraining
& .\automation\final_multiseed.ps1 -Stage post -Seed 45 -OwnerApprovedPostTraining
& .\automation\final_multiseed.ps1 -Stage post -Seed 46 -OwnerApprovedPostTraining
```

The wrapper first verifies all three selected-seed checkpoints. Each view then
exports full February3799/March4332 logits and raw features plus5354 TRAIN
reference embeddings; fits T using February only; computes March Q on4064
retained samples within fixed S25 (no Drift cap); fits TRAIN-only equal-family
normalizer/bandwidth; computes primary descriptive MMD on the SAME1768 A and1768 B
matched IDs. Finally compare complete three-view IDs/order/content hashes/mapping,
then freeze Q=softmax(-L) and QD=softmax(-L-d), beta1,gamma0. Each view receipt pins
its seed/checkpoint/T/logits/reference/source; no C/Future stage exists.

Postprocessing saves unique view directories under `data/derived/final_seed_processing/`
with logits/embeddings,T,Q,per-sample NLL,mask,normalizer,bandwidth,pairs,per-family
MMD,manifests/content hashes/source snapshots and receipts; `freeze/` has full
precision weights and receipt hashes. Real integration is NOT yet verified.
Source changes invalidate receipts; investigate rather than editing frozen hashes.

## Resource budget and failure rules

| View | Runs | Historical100-epoch loop time/run | Additional checkpoint storage |
|---|---:|---:|---:|
| SBSMI |4|44.03min|about0.688GiB total4×100 |
| Raw-byte |4|43.08min|about0.320GiB total4×100 |
| Entropy |4|42.77min|about0.320GiB total4×100 |

Historical estimate8.659GPU hours for12 loops; it excludes repeated per-epoch
metadata/hash checks, checkpoint round-trips, initialization, thermal effects,
shared GPU load and postprocessing. No finish-time guarantee. New payloads may be
larger than legacy:1.328GiB is a baseline-format projection, not a reservation.
Preserve≥2GiB temporary headroom in addition to planned remaining files. Total
RAM/VRAM need a fresh check; CPU process peaks are not GPU training requirements.

STOP for source/config/input mismatch, missing/corrupt images, changed environment
defaults, nonfinite loss/gradients/state, insufficient resources, interrupted I/O
or run ambiguity. No dropped samples, CPU-training fallback or checkpoint reselection.
Only epoch100 COMPLETE runs qualify; owner handles interrupted runs without
performance-driven seed exclusion. None of these12 training runs was started here.
