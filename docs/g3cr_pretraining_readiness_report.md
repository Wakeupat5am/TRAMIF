# G3-C-R — Pre-training readiness/resource audit

Ngày: 2026-10-16 (Asia/Saigon), observations khoảng 08:12–08:14.
**Overall: BLOCKED for starting the six runs now — AC power offline.**
No training authorization is implied by any PASS below.

## 1. Readiness decision

| Area | Decision | Evidence / boundary |
| --- | --- | --- |
| DATA_READY | PASS | E1/E2 receipt SHA-256, exact G2 bindings and ordered metadata validated; source/config hashes unchanged. Reuse G3-C-F image verification; no image rescan |
| CODE_READY | PASS | G3-C-I 20/20 targeted tests reused with matching source hashes; static trainer/checkpoint/failure review. Actual GPU training loop remains unexecuted, not proven by these tests |
| ENVIRONMENT_READY | PASS | Python/torch/NumPy/runtime versions match; CUDA availability and device memory query succeed; effective Adam defaults match. No kernel/forward benchmark |
| RESOURCE_READY | BLOCKED | RAM/VRAM/disk snapshot and output write probe are adequate for proposed planning allowance, but Windows reports ACLineStatus=0, battery 70%, Silent power scheme. Do not start this prolonged batch on this snapshot |
| SCIENTIFIC_PROTOCOL_READY | PASS | For the bounded prefix-training design only: frozen TRAIN, K32/K42, fixed seed/config/epoch100, fresh initialization and no outcome selection. Not validation of D/U, convergence or the research hypothesis |

Saved baseline optimizer-state parity remains **UNKNOWN**. This is not evidence
of a mismatch: current code/default parity is verified, but registered checkpoints
were not deserialized. Prefix training starts fresh and does not reuse those
optimizer states. Owner should acknowledge this limit at training approval rather
than claim checkpoint-state equality. No new scientific choice was made here.

## 2. Evidence freshness

Both receipts still match G3-C-F:

| Case | Receipt under data/derived/prefix_preflight_v1/ | SHA-256 |
| --- | --- | --- |
| E1 | `E1/20261016_005713_190675/receipt.json` | `2942ea823bc521a7b4a249f42f977178ea6de9613dc206a9ecb951d2e0fcd76f` |
| E2 | `E2/20261016_005833_872347/receipt.json` | `45a6bdd174cf3c2caff907d1f85494bae3c2af5bf104d9b176e3aac86d4b9fd3` |

`verify_preflight` accepted both, including all three views, frozen metadata,
mapping, ID order, content identity and source/config hashes. G3-C-F before-snapshot
source/input hashes also remain equal. Thus receipts are **not stale with respect
to checked source/metadata/config bindings**. Images were not rehashed in this audit;
the trainer's per-read image/report receipt checks remain necessary.

Reused test evidence:
`data/derived/g3ci_prefix_implementation/20261015_155521_441848/verification.json`,
SHA `9db35dbbe6033667b0cfea5abd8fd2d710d77662d9cde7e180e231f5d080ef96`.
No tests or full image preflight rerun. No registered checkpoint read except file
size stat for storage estimates; no checkpoint deserialization or inference.

New audit artifacts:
`data/derived/g3cr_readiness/20261016_011243_214340/`:

- `resources.json`, SHA `262a3d574a3d95c65c05dce137f7ceda36e1a51feab1f0a72de8121b715e150f`.
- `verification.json`, SHA `fec9f5dd83745a5a6619c7deef138528b8d355ceec20e370081b6c9c99978ff1`.

## 3. Resource snapshot — VERIFIED, not a load test

| Item | Observation |
| --- | --- |
| GPU / driver | NVIDIA GeForce RTX 4060 Laptop GPU / 617.42, WDDM |
| NVIDIA-SMI before CUDA context | total 8188 MiB, free 7956 MiB, used field 0 MiB; no running processes reported by full table, compute-app query empty |
| PyTorch device query | CUDA available=True; total 7.996 GiB, free 6.939 GiB after context initialization; no GPU model/tensor forward/backward |
| RAM | 15.245 GiB total; 6.448 GiB available |
| Output disk C | 81.657 GiB free |
| GPU thermal/power | 45–46°C, approximately 19 W, 0% utilization; initial table reported cap55 W, query power.limit was N/A |
| Windows power | Silent scheme; GetSystemPowerStatus ACLineStatus=0 (offline), battery70%; CIM battery status1 |
| Unknown | Sustained thermal throttling, CPU temperature, training peak RAM/VRAM, actual wall time; no stress test performed |

NVIDIA-SMI and PyTorch memory numbers are different snapshots/APIs; do not force
them to sum identically under WDDM or infer a hidden process from the difference.
Observed driver/runtime cooperation is established by successful CUDA device and
memory queries with torch CUDA12.6. NVIDIA-SMI's displayed CUDA UMD13.4 is not the
torch build runtime. This is API readiness, not proof all training kernels work.
Baseline driver-level reproducibility is not established by package-version parity.

Create/write/fsync/read/rename succeeded at the actual output root using a NEW
probe, retained at:
`data/derived/prefix_training_v1/_readiness_probes/20261016_011243_214340/write_probe.txt`.
It is explicitly not a training run; no immutable artifact modified. Disk quota,
future contention or later permission changes are not guaranteed by this probe.

## 4. Six-run budget and parity

All runs: seed42, batch32, 100 epochs, epoch100 selected, Adam lr1e-4 and
weight_decay1e-6, unweighted CE; no scheduler/augmentation/AMP/warm-start/resume.
Fresh existing architecture family per view, not Final checkpoint initialization.
E1 cutoff2019-11-01 UTC; E2 cutoff2019-12-01 UTC (exclusive); start2019-08-01.

| Run | N / K | Batches per epoch / tail | Total optimizer steps | Projected loop minutes | 100-checkpoint disk proxy GiB |
| --- | --- | --- | ---: | ---: | ---: |
| E1 SBSMI | 7034 / 32 | 220 / 26 | 22000 | 17.14 | 0.172 |
| E1 Raw-byte | 7034 / 32 | 220 / 26 | 22000 | 16.77 | 0.080 |
| E1 Entropy | 7034 / 32 | 220 / 26 | 22000 | 16.66 | 0.080 |
| E2 SBSMI | 9803 / 42 | 307 / 11 | 30700 | 23.92 | 0.172 |
| E2 Raw-byte | 9803 / 42 | 307 / 11 | 30700 | 23.41 | 0.080 |
| E2 Entropy | 9803 / 42 | 307 / 11 | 30700 | 23.24 | 0.080 |

Total: 600 epochs, 158100 steps, approximately121.15 minutes of projected loops,
approximately0.664 GiB of checkpoint-size proxies. **Neither is a guarantee.**
Timing uses the three hash-matched Final histories and batch-ratio scaling
220/565 or307/565. Historical epoch maxima were40.90s,73.09s,45.60s (SBSMI/raw/entropy).
Hashing metadata every epoch, image/report reads, initialization, save/load/fsync,
source copying, disk cache, power limits and temperature can increase wall time.
Silent/battery conditions are not comparable to an assumed sustained training setup.

Disk proxies use Final epoch100 sizes 1846468/858884/858884 bytes, not serialized
prefix measurements. Prefix heads are smaller, but new provenance/RNG fields,
six initial states, per-run preflight copies, source snapshots, epoch JSONs and
temporary/failed files add overhead. **PROPOSED planning allowance: reserve 2 GiB
for this batch's outputs including temporary-file headroom**, instead of treating
0.664 GiB as a cap. This is an operational budget for owner review, not a measured
upper bound or a new implemented threshold. Current81.657 GiB is ample relative
to it. Do not delete artifacts automatically if space later becomes insufficient.

RAM/VRAM: three metadata adapters/receipts are resident, images are loaded batchwise
with num_workers0; no full image tensor bank. Historical baseline peak allocated
GPU memory was approximately125.7–127.2 MiB (G3-C-P evidence), not total CUDA/WDDM
footprint and not a bound on prefix usage. Current headroom appears adequate for
one run, **INFERRED**; actual peak remains UNKNOWN. Six runs must be sequential.

Environment verified: Python3.11.9, torch2.13.0+cu126, NumPy2.4.6, CUDA runtime12.6.
CPU-only Adam constructor (one dummy parameter, no step) confirmed:
betas(0.9,0.999), eps1e-8, amsgrad/maximize/capturable/differentiable/
decoupled_weight_decay=False, foreach/fused=None, lr/decay as above.

## 5. Failure review and STOP conditions

Static evidence in `scripts/train_prefix_view.py`:

- `train` line176: explicit execution guard, receipt/environment checks before run;
  fresh initialization, no evaluation loader or outcome-driven epoch selection.
- Batch loop lines239–254: non-finite loss/gradient stops; optimizer settings not
  automatically changed. ID coverage and source/input binding checked per epoch.
- `save_checkpoint` line127: exclusive temp write, fsync, weights-only round-trip
  of newly generated state, finite/equality checks then rename; no overwrite.
- `COMPLETE` line284 only after100 epochs. Exception/KeyboardInterrupt is caught
  and FAILED written; no resume/retry. OOM must stop, not shrink batch or switch CPU.

Limits: power loss/forced process kill cannot guarantee FAILED or fsync completion;
a temp/orphan checkpoint is not a completed run. Disk-full/permission failures can
also prevent writing FAILED and mask the original error; preserve incomplete
directory/terminal error for review. STARTED creation is outside the try block.
No automatic free-disk, AC, RAM or thermal monitor exists in this trainer. These
limitations were reported, not silently patched or labeled runtime verified.

Before any separately approved run, STOP if AC remains offline, CUDA disappears,
resource headroom no longer supports the approved budget, hashes/receipt bindings
change, or input files fail verification. During a run STOP on OOM/non-finite
loss/gradients/state, image/report/hash mismatch, checkpoint round-trip/I/O error,
or interruption. Do not proceed to the next run after failure, auto-resume, choose
another seed/checkpoint, reduce epochs/batch size or repair data. Changing power
scheme, monitoring automation or adding resource guards requires its own action
authorization; this audit changed none of them.

## 6. One proposed decision

**Hold GPU training approval now. Connect AC power, then authorize a focused
resource/power recheck and owner review of the six-run budget/known limits.**
If that review passes, propose six sequential runs; do not start the first run
automatically. Silent mode is a throughput uncertainty, not proof the model is
invalid; no power-mode change is prescribed as a methodology requirement.

Only report/progress/decision docs and new audit/probe artifacts changed. No
source/config/seed/sampler/model/checkpoint changes, training smoke run, forward,
backward, optimizer step, registered checkpoint load, embeddings, MMD/Q/U,
A/B/C or Future access, dependencies changes, commit/push. STOP for owner review.
