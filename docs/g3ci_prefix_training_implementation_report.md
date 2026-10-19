# G3-C-I — Prefix training implementation only

Ngày: 2026-10-15. **IMPLEMENTED; targeted tests 20/20 PASS.**
Không phải training approval hoặc bằng chứng prefix encoders đã tồn tại.
Không optimizer step, backward, GPU forward, registered checkpoint load,
historical-image inference, embeddings, MMD/Q/U/fusion hoặc Future Test.

## 1. Implementation và phạm vi

Hai module mới, không sửa legacy:

- `scripts/episode_training_dataset.py`: frozen G2 trust anchors, TRAIN-only
  adapter, immutable sample records, hash-bound source mapping/configs, decoder,
  và lệnh image/report preflight độc lập. Không dùng legacy constructors vốn đọc
  full historical manifest và hard-code K51.
- `scripts/train_prefix_view.py`: fresh model factories, optimizer/default guard,
  seeded DataLoader, fixed 100-epoch trainer, atomic checkpoint verification,
  explicit failure records và hash-bound preflight guard. Training loop đã viết
  nhưng **chưa chạy**. CLI không có lựa chọn best epoch, resume, warm-start,
  scheduler, seed search hoặc tự đổi batch size.

E1 dùng đúng N=7034/K=32; E2 N=9803/K=42. TRAIN UTC lần lượt
`[2019-08-01,2019-11-01)` và `[2019-08-01,2019-12-01)`.
Manifest hashes/order/cutoff/retention/mappings được kiểm tra trước đọc ảnh;
không filter để loại hàng sai cho đủ điều kiện. Chỉ pinned TRAIN metadata,
source mapping và preprocessing configs được đọc trong metadata verification.
Không mở A/B/C manifests, predictions, losses hoặc binaries.

`TrainingSample.target` lấy từ episode mapping; `source_class_index` chỉ kiểm tra
report lưu theo Final51. Tensor trả về cùng target và SHA để kiểm tra coverage.
SBSMI PNG chia 255 đúng một lần; raw/entropy NPY giữ nguyên float32 đã normalized.
Các hash được kiểm tra trên chính bytes đem decode. Mỗi report phải khớp content
SHA trong G2 membership. Pinned SBSMI config xác nhận representation: legacy
SBSMI report không có trường representation, adapter không yêu cầu trường giả.

Preflight phải đủ ba views của cùng case, cùng ID/order/targets/content identity;
receipt lưu từng image/report hash và binding metadata/source-code hash. Trainer
kiểm tra receipt và so lại hash từng ảnh/report khi đọc. Không tái tạo ảnh.

## 2. Acceptance checks và evidence

| Check | Kết luận | Bằng chứng và giới hạn |
| --- | --- | --- |
| Frozen G2 IDs/hash/mapping/chronology | **PASS — metadata verified** | Constructors cho E1/E2 × 3 views chỉ đọc metadata; N/K và ordered TrainingSample tuples bằng nhau. Không audit lại graph/retention |
| Leakage/retention guards | **PASS — unit tested** | Synthetic wrong stage, UTC offset/naive date, cutoff, April timestamp, unknown family/ID, duplicate, exclusion/reference/threshold bị reject; không mở Future thật |
| Episode versus Final labels | **PASS — unit tested** | Synthetic source index 3 chuyển thành episode target 0, không dùng index 3 cho loss; corrupt source index bị reject |
| Decoder/scaling parity | **PASS — synthetic integration** | Tensor so sánh trực tiếp với legacy `__getitem__` trên temp synthetic files; không gọi legacy full-dataset constructor |
| Missing/corrupt/hash/finite/range/shape | **PASS — unit tested** | File thiếu, report sai, image checksum/pixel checksum sai, NPY sai dtype/shape, NaN/Inf và ngoài [0,1] bị reject |
| Deterministic shuffle/coverage | **PASS — unit tested** | Hai independent seeded loaders, hai epochs, synthetic IDs với N=7034/9803; 220/307 batches, tails 26/11, mỗi ID đúng một lần |
| Six prefix architecture shapes | **PASS — CPU synthetic** | Random fresh models, K32/K42 × 3 views, eval/no-grad; logits [2,K] và [1,K], finite loss, state dict bất biến, không gradients/optimization |
| Effective Adam/environment | **PASS — current runtime** | Versions/defaults bên dưới; constructor only, optimizer state rỗng và parameters không đổi |
| Stored baseline optimizer state parity | **UNKNOWN** | Không deserialize registered checkpoints. Không biến current default equivalence thành xác nhận saved optimizer tensors/options |
| Synthetic checkpoint I/O | **PASS — unit tested** | Save/load mới bằng weights_only=True, exact tensor/metadata equality; không overwrite; corrupt/non-finite payload không thành registered checkpoint |
| Failure/no auto execution | **PASS — unit tested** | Import không đọc inputs/load checkpoints/probe CUDA/step; train default guard từ chối. Injected constructor failure dưới mock readiness ghi STARTED/FAILED, không checkpoint/COMPLETE/retry |
| Full historical image preflight | **UNKNOWN — NOT RUN** | Preflight logic chỉ chạy trên synthetic fixtures; metadata PASS không chứng minh ảnh thật còn đầy đủ |
| GPU loop/100-epoch completion | **UNKNOWN — NOT RUN** | Training loop chỉ static reviewed; không gọi optimizer.step trong verification |
| Prefix RAM/VRAM/wall time/storage | **UNKNOWN** | Chưa resource preflight hoặc measurement; không dùng estimated timing làm actual runtime |
| Scientific validity/convergence | **UNKNOWN** | Unit tests không kiểm chứng research hypothesis hoặc chất lượng model |

Targeted invocations (CUDA_VISIBLE_DEVICES rỗng, Python `-B`):

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_episode_training_dataset.py" -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_train_prefix_view.py" -v
```

Final execution: dataset 10 tests / 1.321s / OK; trainer 10 tests / 1.392s / OK.
Không chạy full repository suite; không cộng số này vào checkpoint count cũ để
tuyên bố toàn repo PASS. Windows Store Python không khởi động được trong sandbox;
hai lệnh được chạy bằng cùng repo venv ngoài sandbox sau approval review, CPU-only.

Durable verification receipt, code/config hashes và restrictions:
`data/derived/g3ci_prefix_implementation/20261015_155521_441848/verification.json`.
Receipt ghi lại kết quả invocation đã hoàn thành; writer không chạy tests lại.
Tám unique legacy source bindings từ ba baseline configs vẫn khớp. V1 giữ hash
`e64664c4c499ef762b4a52c45889f0d05db6741ec5021b249f2072585f816546`.

## 3. Effective configuration và parity

**VERIFIED hiện tại:** Python 3.11.9; torch 2.13.0+cu126; NumPy 2.4.6;
CUDA build runtime 12.6. Đây là version inspection, không xác nhận GPU execution.
Baseline config/source evidence và current runtime khớp về versions và cách gọi
`Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)`.

Effective defaults đo bằng CPU optimizer constructor, không step:

```text
lr=0.0001, weight_decay=0.000001, betas=(0.9,0.999), eps=1e-8
amsgrad=False, maximize=False, foreach=None, capturable=False
differentiable=False, fused=None, decoupled_weight_decay=False
```

`make_optimizer` từ chối defaults khác. `environment` từ chối version mismatch
trước GPU training. Đây không phải permission tự nâng cấp/downgrade dependencies.
Saved optimizer state vẫn UNKNOWN do không được mở registered checkpoints.
Code-level/default parity có evidence; khi duyệt training, owner cần chấp nhận
giới hạn này hoặc cho một metadata-only checkpoint inspection riêng nếu cần.

Fixed policy: seed42, batch32, unweighted CrossEntropyLoss, no scheduler/no AMP,
no augmentation, fresh init, 100 epochs, chỉ epoch100 cho downstream. Same model
families và feature widths; chỉ K32/K42 khác Final. `max_norm=inf` là finite-gradient
guard, không đưa finite clipping hyperparameter mới vào phương pháp.

## 4. Preflight command đã chuẩn bị — CHƯA ĐƯỢC CHẠY

Sau một approval riêng cho full TRAIN image/report preflight:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.episode_training_dataset --case E1
.\.venv\Scripts\python.exe -B -m scripts.episode_training_dataset --case E2
```

Mỗi lệnh đọc hữu hạn đúng frozen TRAIN IDs × 3 views, không CNN/optimizer/binaries,
không A/B/C/Future, không scan các thư mục dữ liệu ngoài allowlist. Inputs read-only;
outputs mới tại `data/derived/prefix_preflight_v1/<case>/<unique UTC run>/`:
`receipt.json` nếu PASS hoặc `FAILED.json` nếu lỗi. Chỉ in summary/path/hash, không
in toàn memberships. Không ghi đè artifacts cũ hoặc tự repair ảnh.

Trước preflight: owner review code/tests và cấp quyền đọc toàn selected image/report
set; giữ source hashes, G2 anchors, configs/mapping/model/V1 hashes hiện tại.
Thiếu file/hash mismatch phải dừng, không mở raw binary để cứu lỗi. Hai receipts
PASS là prerequisite cho sáu runs; thay code/binding sau đó làm receipts stale.

Training CLI yêu cầu `train`, case/view, preflight path/hash và `--execute-training`.
Flag là acknowledgement của operator, **không** chứng minh owner đã phê duyệt.
Batch này không cung cấp lệnh training để chạy tiếp tự động. Implementation không
có cơ chế tự chuyển từ preflight sang training.

## 5. Provenance, failures và resource blockers

Trainer ghi mới dưới `data/derived/prefix_training_v1/<case>/<view>/<unique run>/`:
config/hash, full TRAIN sample identities/episode targets, preflight receipt,
source snapshots/hashes (kể cả dirty/untracked new code), Git HEAD/status/diff hash,
environment/defaults, initial state, epoch checkpoints/receipts và coverage hashes.
Checkpoint lưu model/optimizer/RNG states, verify temp trước rename, hash registry;
COMPLETE chỉ sau đủ 100 epochs, selected=epoch100. Exception/interrupt ghi FAILED,
không automatic retry/resume/skip/config fallback. Metadata/source mutation phát
hiện trước đăng ký epoch; ảnh/report pinned per-read với preflight receipt.

Implementation hiện thực các chính sách đó nhưng actual GPU run/checkpoint lifecycle
chưa integration verified với optimization thật. Failure test dùng mock readiness và
lỗi synthetic trước model construction; không phải một training smoke run.

Resource readiness còn thiếu: available VRAM/RAM, disk headroom, power/thermal state,
permission cho sáu runs tuần tự, interruption/retry decision. G3-C-P 121.15 phút chỉ
là projected loop time, không budget hoặc runtime bảo đảm. Storage proxy 0.664 GiB
cho epoch checkpoints chưa bao gồm source/initial state/receipts/temp/failed runs.
Không tự đo VRAM bằng forward hoặc tự xóa dữ liệu để đủ chỗ.

## 6. Files thay đổi và một bước tiếp theo

Tạo mới hai modules, `tests/test_episode_training_dataset.py`,
`tests/test_train_prefix_view.py` và báo cáo này. Cập nhật PROJECT_PROGRESS,
DECISION_LOG (D026), TECHNICAL_WALKTHROUGH; tạo verification receipt riêng ở ignored
data/derived. Không sửa legacy trainers/loaders/models, V1, frozen manifests hoặc
checkpoints; không commit/push.

**Đề xuất duy nhất:** owner review implementation này, sau đó phê duyệt một batch
full TRAIN image/report preflight cho E1/E2 (ba views), vẫn không GPU training.
Chưa tự chạy batch đó. G3-C-I dừng ở implementation/tests; chờ review.
