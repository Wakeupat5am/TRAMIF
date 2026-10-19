# G3-C-P — Prefix training design gate

Ngày: 2026-10-15. **DESIGN ONLY — PROPOSED implementation/training plan.**
Owner chỉ phê duyệt viết thiết kế. Không triển khai adapter/trainer/tests, không
instantiate CNN, optimizer step, inference hoặc embeddings. Final K51 checkpoints
giữ nguyên. G3-B D023 vẫn PASS_WITH_LIMITATIONS; thiết kế này không sửa V1/G2.

## 1. Evidence và ma trận sáu planned runs

Nguồn: V1 §4/§10, `docs/g3c1_frozen_encoder_readiness_report.md`, frozen G2,
`scripts/train_sbsmi.py`, `scripts/train_float_view.py`, hai models và ba loaders
hiện có. Đọc adjacent config/history của đúng ba registered epoch100 baselines;
chỉ dùng timing/memory/checkpoint metadata, không dùng performance để chọn cấu hình.

**VERIFIED metadata:** G2 train manifests, mappings, specs và train00 memberships
(8 files) khớp saved checksums. Không tính lại cohort/retention/graph. Prefix
checkpoints chưa có evidence theo G3-C1; toàn bộ runs dưới đây **PLANNED**, chưa chạy.

| Planned run | TRAIN UTC, khoảng nửa mở | N / K | Architecture | Input → logits | Batches/epoch; last batch | Seed / epochs / selected |
| --- | --- | --- | --- | --- | --- | --- |
| E1-SBSMI | [2019-08-01,2019-11-01) | 7034 / 32 | MalSBSLCNet | [B,1,64,64] → [B,32] | 220;26 | 42 / 100 / epoch100 |
| E1-Raw | như E1 | 7034 / 32 | CompactViewNet | [B,1,64,64] → [B,32] | 220;26 | 42 / 100 / epoch100 |
| E1-Entropy | như E1 | 7034 / 32 | CompactViewNet | [B,1,64,64] → [B,32] | 220;26 | 42 / 100 / epoch100 |
| E2-SBSMI | [2019-08-01,2019-12-01) | 9803 / 42 | MalSBSLCNet | [B,1,64,64] → [B,42] | 307;11 | 42 / 100 / epoch100 |
| E2-Raw | như E2 | 9803 / 42 | CompactViewNet | [B,1,64,64] → [B,42] | 307;11 | 42 / 100 / epoch100 |
| E2-Entropy | như E2 | 9803 / 42 | CompactViewNet | [B,1,64,64] → [B,42] | 307;11 | 42 / 100 / epoch100 |

Actual timestamp min của cả hai là 2019-08-29; max E1=2019-10-31 12:11:02+00:00,
E2=2019-11-29 22:01:43+00:00. Không giả định dữ liệu phủ mỗi ngày của window.
Train dùng toàn K và toàn frozen TRAIN, **không** chỉ S17/S25, không dùng cap128
hay TRAIN reference subset của MMD. S chỉ phục vụ signals sau này.

Thứ tự đề xuất: E1-SBSMI, E1-Raw, E1-Entropy, E2-SBSMI, E2-Raw, E2-Entropy;
tuần tự một GPU. Fresh initialization mỗi run, không warm-start từ Final hoặc E1.
Tổng 600 epochs, 158.100 optimizer steps nếu hoàn tất; chỉ 6 epoch100 là outputs
được đăng ký cho downstream. Không tìm best run/seed/checkpoint theo A/B/C.

## 2. Frozen data/cutoff/mapping binding

G2 root: `data/derived/historical_episode_feasibility/20261014_233013_930021/`.
Các SHA-256 dưới đây là authority input, không tự thay expected hash khi mismatch:

| Case / file | SHA-256 |
| --- | --- |
| E1/train_keep.csv | `9743b0cc00278fc415707b61bdd0d060b333aec0b1a264f54ee8e53bdc82b706` |
| E1/class_mapping.json | `ef1019d8a936093c87ecf73b6bfa760cc0718e13aee8e2a885f089c3964dd887` |
| E1/episode_spec.json | `53193445036aa60fb89e2bace637e5287ffcaac36393fd065e7a5ad67cfbb500` |
| E1/train_00_members.csv | `16db61fd7b8a8c887e5e05b9c1e539ca6adef7679e7f507b5c761102d25854e7` |
| E2/train_keep.csv | `6c8b1f79449332a82438b13d03fd9bfa9ce0509b3e9017105eecb4e009810768` |
| E2/class_mapping.json | `c5f246b054dbfa9138b63175dc9fc01e257350050b1cab16884307111ddb20d8` |
| E2/episode_spec.json | `c1a060746a3a4bfe5c45de7abdc760a3e6fcb2c4e2aae76aef133bfe234bc326` |
| E2/train_00_members.csv | `fb93ebdf873e09f022d9025e7e856194e21cd21d5aac5edaab7e92cfc3200e56` |

Adapter consumes a case-specific pinned bundle, not a free-form date filter or
directory glob. For each row: valid unique SHA identifier, family in episode
mapping, UTC-aware timestamp in window, stage=train, threshold=0, decision=KEEP;
ID/order/metadata exactly match frozen manifest. Membership must be retained,
stage=train and match timestamp/family; saved content SHA supplies provenance.
Require exactly 7034/9803 rows and contiguous labels 0..31/0..41. Unexpected rows
are errors, **not silently filtered out**. Do not rebuild graph or reinterpret TLSH0.

Runtime read allowlist: pinned case TRAIN bundle; pinned three preprocessing
configs; source mapping only for storage-report integrity; image/report paths
derived exclusively from approved TRAIN IDs. No A_keep/B_keep/C_keep, metrics,
raw BODMAS CSV, binaries or Future data are needed by the training entry point.
Spec contains A/B/C dates only as boundaries; those stages cannot become samples.
Mock tests should prove rejection before any attempted image read.

## 3. Planned adapter architecture — new modules, legacy unchanged

**PROPOSED files**, not created in this gate:

1. `scripts/episode_training_dataset.py`: immutable manifest binding, image/report
   decoder and episode label adapter; no trainer or GPU selection side effects.
2. `scripts/train_prefix_view.py`: one new trainer with enumerated case/view and
   frozen approved run specification. It imports existing model classes unchanged.
3. Targeted tests for adapter, bindings and trainer configuration; no generic
   refactor or extraction pipeline in this implementation batch.

Do **not** instantiate legacy final dataset and then filter it: those constructors
read the full historical manifest, enforce 51 classes and final periods. New
decoder reproduces the relevant hash/shape/scaling contract, tested for parity,
without changing legacy modules or trusting positional alignment.

### Storage identity versus training target

All E1/E2 TRAIN images are in existing final **train storage**:
`data/derived/{sbsmi,raw_byte,entropy}_dataset_v1/train/<sha>.{png,npy}` with JSON
report. Episode role, original storage split, original report class index and
episode target label must be separate fields.

- Verify report sample_identifier, timestamp, family, split=train, representation,
  preprocessing config ID, PASS/round-trip status and disarmed content SHA against
  the frozen TRAIN membership record; require three views identify the same bytes.
- Stored report `class_index` is the original Final51 index. It may be checked
  against the hash-bound source mapping **only as storage integrity evidence**.
  Training target is always `episode_mapping[family]`; never return/use original
  class_index for loss or determine classifier width from it. Test a family whose
  episode and source indices differ. No edits to existing reports/mappings.
- Sample identifier is not the disarmed-file content hash or image-file hash.
  Verify image bytes against PNG/NPY hash; SBSMI also decoded pixel hash. Do not
  compare PNG hash to original sample SHA or execute/re-hash malware binaries.
- SBSMI PNG L64×64 uint8 → float32/255 → [1,64,64]. Raw NPY float32[64,64] already
  divided255; entropy NPY already divided8, window256/stride128, mask-aware area
  resize. No second division, per-image min-max scaling, augmentation, resize
  change or z-score. MMD TRAIN normalization is outside CNN training.
- Read/hash/decode the same bytes, NPY allow_pickle=False; validate finite[0,1],
  dimensions/channels/dtype. Missing file, malformed report, mismatch or extra
  sample: STOP and record failure; no auto repair, zero-fill or silent skip.
- Dataset canonical order is exact G2 train_keep order, same for three views.
  Return ID and metadata with tensor/episode label so batching remains auditable.
  Epoch shuffle permutes canonical indices with a dedicated seeded generator;
  no label-based sampler. Record permutation hash and verify each ID exactly once
  per epoch. File mutations after preflight must still fail per-read hash checks.

## 4. Hyperparameter and checkpoint parity assessment

Source configs: `data/derived/sbsmi_training/20261005_105325_352508/config.json`,
`data/derived/float_view_training/raw-byte/20261009_145102_408184/config.json`,
`data/derived/float_view_training/entropy/20261011_001259_203865/config.json`.

| Item | VERIFIED baseline evidence | PROPOSED prefix parity / limits |
| --- | --- | --- |
| Architecture | MalSBSLCNet SBSMI; CompactViewNet raw/entropy; BN, dropout0.4 | Same classes/source hashes; only classifier output K32/K42; head dimensions2048/512 unchanged |
| Initialization | Fresh model constructors; no checkpoint initialization | Fresh random initialization each case/view, no transfer from later-trained model |
| Loss | Unweighted CrossEntropyLoss on logits | Same; no class weights, label smoothing or sampled-class objective |
| Optimizer | Adam, lr1e-4, weight_decay1e-6 | Same; no AdamW substitution or learning-rate scaling by sample count |
| Adam details | Trainer leaves betas/eps/amsgrad/foreach etc. to installed defaults; configs do not enumerate all | Resolve/record effective defaults under recorded torch2.13.0+cu126 before freeze; validate against baseline optimizer metadata where needed. Exact effective parity UNKNOWN until this check; no silent version change |
| Schedule/duration | scheduler=None; registered runs100 epochs | Fixed100, constant LR, no early stopping/validation-driven scheduling; fewer prefix updates are intentional, not matched-step training |
| Batching | batch32, shuffle=True, drop_last=False, num_workers0, pin_memory=True | Same; final batches26/11 support BN1d; do not drop rows to fix an issue |
| Numerics | float32, mixed_precision=False; no finite gradient clipping limit | Same; finite loss/gradient/state checks. clip_grad_norm(max_norm=inf,error_if_nonfinite=True) is validation, not a finite clipping threshold |
| Seeds/backend | Python/NumPy/Torch/CUDA seed42; independent loader generator42; cuDNN deterministic=True, benchmark=False | Same; log all effective flags/environment; no cross-hardware bitwise guarantee |
| Augmentation | raw/entropy config None; SBSMI loader source has none | None for all; no fitted preprocessing from A/B/C |
| Save policy | Every completed epoch, atomic temp rename, CPU save/load equality checks | Same; save all100 epochs, SHA/config/state verification; epoch100 only downstream |
| Resume | raw/entropy explicitly unsupported; SBSMI entry has no resume path | No automatic resume. Failed run preserved and marked failed/incomplete; retry from seed42 requires explicit authorization, never overwrite or select best restart |

Current Compact trainer hard-codes 59059 parameters for K51; new trainer must not
copy that assertion unchanged. Expected classifier shape [K,512] or [K,2048]
and model-derived/source-validated parameter count must be recorded per run.

**Checkpoint selection before training:** freeze epoch100 for all six; no best
loss/Accuracy on TRAIN/A/B/C, no selection from outcome months or Future. Train
loss is logged for diagnosis/finite checks only. SBSMI legacy config saved every
epoch and had100 planned epochs, but lacked an explicit preregistered selection
field; do not rewrite that history. Prefix rule will be explicit prospectively.
Uniform duration does not guarantee convergence or identical capacity/cost across
architectures/episodes. One seed is prefix screening, not multi-seed validation.

## 5. Tests and pre-GPU acceptance — NOT EXECUTED

All tests below need an implementation-only approval first. CPU synthetic CNN
forward checks, if approved then, use fresh random models; never load trained
checkpoints for inference or run optimizer steps. Tests do not authorize GPU work.

| Test group | Required PASS evidence before GPU training |
| --- | --- |
| Immutable binding | Hashes/counts/unique IDs/order/mappings match frozen E1/E2 exactly; altered manifest/checksum/mapping/config rejected |
| Chronology/leakage | UTC boundary tests, wrong stage/retention/family, A/B/C/Future and unknown ID injections fail before file reads; access spy shows only case TRAIN allowlist read |
| Labels | Label space exactly K_e, correct family→episode index across views; deliberately different Final index cannot reach training target; wrong label fails |
| Images | Synthetic PNG/NPY hand answers for zero,255, fractional values, normalized entropy; exact shapes/scaling; corrupt/truncated/missing/hash/dtype/range failures |
| Decoder parity | Known synthetic bytes match legacy decoder mathematics; separately authorized read-only preflight of every selected historical image/report verifies availability/content provenance across views, no model |
| Deterministic data order | Repeated fresh generators42 yield identical per-epoch permutations; each epoch covers N IDs once; batch lengths220/307 and tails26/11; canonical order unchanged |
| Architecture | Instantiate fresh models with K32/K42; classifier shapes, finite synthetic logits [B,K], correct loss accepts episode labels; static no warm-start and correct architecture hashes |
| Config | Freeze lr/decay/loss/scheduler/seed/epochs/defaults; validate Adam param-group configuration without step; environment/default parity unresolved→STOP |
| Checkpoint I/O | Synthetic/random-state save/load equality, metadata binding, atomic writes, failure/incomplete-run handling and no overwrite; no trained forward required |
| Guards | Import/test cannot start training; explicit main/run config; no evaluation loader, no epoch choice from metrics; no hidden resume, sampler mutation or auto batch reduction |

Historical image preflight is proposed future work, not a claim that this design
opened images. It should reuse cached images and report hashes, not regenerate
datasets. Lightweight metadata tests can reuse G2 evidence without re-auditing it.
First actual GPU batch/epoch is already training and belongs to training approval;
do not disguise a GPU optimization smoke run as design/preflight.

## 6. Compute estimate from existing evidence

Read-only summaries from the three registered run `history.csv` files,100 rows
each. No new benchmark. Device in configs: RTX4060 Laptop GPU; Python3.11.9,
torch2.13.0+cu126, NumPy2.4.6, CUDA runtime12.6. Final has565 batches/epoch.

| View | Logged epoch median [min,max], seconds | Sum100 epochs | Max allocated GPU MiB | INFERRED E1 / E2 loop time |
| --- | --- | --- | ---: | --- |
| SBSMI | 26.46 [17.00,40.90] | 44.03 min | 127.17 | 17.14 / 23.92 min |
| Raw-byte | 25.73 [14.97,73.09] | 43.08 min | 125.73 | 16.77 / 23.41 min |
| Entropy | 25.47 [19.88,45.60] | 42.77 min | 125.73 | 16.66 / 23.24 min |

Projection uses `historical_sum_seconds * prefix_batches/565`,100 epochs. Sum
six projected loops≈**121.15 minutes**. This assumes similar throughput and is
not a measured prefix runtime, reservation limit or confidence interval. Timing
in trainer stops before checkpoint save/load/hash, and excludes initialization
and full preflight. Extra integrity I/O, thermal/power/cache variation and smaller
heads affect timing. Total end-to-end wall time remains **UNKNOWN**; do not promise
a two-hour completion. GPU allocated memory is not total VRAM/reserved/context or
process RAM. Actual prefix peak RAM/VRAM, disk free space and power state UNKNOWN.

A storage planning proxy, not a cap:100 checkpoints×2 episodes×sum three Final
checkpoint byte sizes≈0.664GiB; new heads are smaller but metadata/serialization,
temporary files, logs and failed runs need headroom. No new image copies planned;
actual storage must be checked before execution, not auto-delete artifacts.

Timing-source SHA256 (same order SBSMI/raw/entropy):
`238315fd122bf4dba4db6a191ae8481585b59be84ff9e03cc7c09c3b035c9cdc`,
`e6114ad2d98f24d6f24006b4176ddb7c233ba998cc24d63e905bab587500b442`,
`36f06d06adff239421ba2665fb590935cc9c19d0703cda748870d7d5248e9a20`.

Owner must approve six sequential runs,100 epochs each and resource budget after
preflight; no auto-parallelization/extra seeds, hidden retries, batch-size changes
or extended epochs. OOM/non-finite/I/O failures stop; changing configuration needs
review. Training completion does not authorize calibration/extraction/MMD.

## 7. Provenance and run lifecycle plan

New immutable output root proposed:
`data/derived/prefix_training_v1/<case>/<view>/<UTC-run-id>/`.
Never write in Final runs or G2/G3-B artifacts. Save before first optimizer step:

- Full run spec and config hash, case/view/cutoff/K/class mapping, canonical TRAIN
  IDs/order/labels hash, retained membership/content bindings, per-view image/report
  file hashes and preprocessing IDs. Source Final mapping is labeled source-only.
- Model/adapter/trainer/test source hashes, Git HEAD plus dirty diff/untracked
  source snapshot hashes, V1 and relevant decision IDs, dependency versions,
  effective Adam options/backend flags/device and resource approval scope.
- Explicit fresh initialization and seed42; initial state hash, RNG states,
  independent loader generator state, planned epoch policy and checkpoint selector.
- Targeted test and full image-preflight receipts; reject stale receipts if any
  bound code/input/config changes. Do not merely trust an old test-count summary.

Per epoch: actual ID coverage/permutation hash, steps, timing, peak allocated and
reserved GPU memory (with metric definitions), training diagnostics/failures;
model/optimizer state, CPU/CUDA/Python/NumPy/loader RNG states, epoch/config in
checkpoint. RNG recording supports provenance; it does not imply resume support.
Checkpoint save→safe metadata/state round-trip verification→hash registry, then
completion marker. Failed temp files/run status must not be registered as complete.
Final run receipt requires epochs1..100 complete, final epoch hash verified and
all bindings unchanged; freeze only epoch100. No output predictions/embeddings
or access to A/B/C occurs in this trainer.

## 8. Risk register and approval blockers

| Risk | Consequence | Required response / acceptance |
| --- | --- | --- |
| Final labels leak into prefix labels | Wrong supervised target/class count | Separate source/episode namespaces; remapping tests and exact frozen mappings PASS |
| Train-ID membership expands by path/date fallback | Chronological leakage | Pinned explicit ID allowlist; unknown/non-TRAIN record fails, no filtering-to-success |
| Later-trained initialization | Prefix encoder leaks later history | Fresh initialization only; no Final/E1 warm-start |
| Preprocessing/report mismatch | Noncomparable or corrupted input | Per-read hashes, finite/shape/scaling and cross-view content identity; fail closed |
| BN tail or class-dependent sampling | Missing samples/objective drift | Current tails26/11 safe; full-coverage permutations, no drop_last/balancing |
| Hyperparameter/default/version drift | Uncontrolled comparison | Config parity/defaults receipt and owner review if mismatch; no automatic upgrades |
| Fixed100 epochs insufficient | Weak prefix fit | Report honestly; no A/B/C-driven extension or best-epoch replacement |
| Sample dependence and retrospective labels | Overstated generalization | Preserve G2/G3-B caveats; train near-duplicates stay per frozen policy; no independence claim |
| Resource/interruption/retry selection | Incomplete provenance/unbounded cost | Sequential bounded authorization, preserve failures, explicit retry approval; no best restart |

**Blockers before any GPU training:** (1) owner approval of this design and a
bounded implementation-only batch; (2) implemented adapter/trainer + targeted
tests PASS; (3) complete allowlisted image/report preflight PASS; (4) effective
optimizer/environment parity resolved; (5) frozen run/config/checkpoint/failure
policy and storage/resource readiness; (6) separate explicit training approval.
No threshold/seed/cutoff/retention change is delegated to implementation.

**One next step proposed:** owner review this design and approve implementation
of the new episode-aware adapter/trainer/tests, with training still disabled.
No files in that planned implementation have been created here. Runtime extractor
integration G3-C2 remains a separate gate; Final is not retrained. **STOP.**
