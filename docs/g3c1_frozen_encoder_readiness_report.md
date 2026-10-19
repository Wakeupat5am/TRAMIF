# G3-C1 — Frozen encoder readiness, static audit

Ngày: 2026-10-15. **Static audit COMPLETE; chưa có end-to-end extractor ready.**
Không import torch/project modules, deserialize checkpoints, instantiate model,
đọc images, chạy forward hoặc tạo embeddings. Checkpoint chỉ được đọc bytes để
hash; metadata lấy từ adjacent training configs và frozen registry. Không dùng
outcome metrics để chọn checkpoint; giữ đúng ba epoch100 đã đăng ký.

## 1. Case × view inventory

READY dưới cột encoder nghĩa là đủ static candidate evidence cho bước integration
riêng, không phải runtime PASS. Overall readiness còn yêu cầu extractor nối đúng
accepted manifests. BLOCKED không có nghĩa kiến trúc hiện tại bị chứng minh sai.

| Case | View | Checkpoint đúng case | Architecture/output dự kiến | Encoder static | Overall extraction readiness |
| --- | --- | --- | --- | --- | --- |
| E1 | SBSMI | Không tìm thấy prefix checkpoint có provenance; path/hash UNKNOWN | MalSBSLCNet/K32 | BLOCKED | BLOCKED: thiếu checkpoint, prefix loader/extractor |
| E1 | Raw-byte | Không tìm thấy prefix checkpoint có provenance; path/hash UNKNOWN | CompactViewNet/K32 | BLOCKED | BLOCKED: như trên |
| E1 | Entropy | Không tìm thấy prefix checkpoint có provenance; path/hash UNKNOWN | CompactViewNet/K32 | BLOCKED | BLOCKED: như trên |
| E2 | SBSMI | Không tìm thấy prefix checkpoint có provenance; path/hash UNKNOWN | MalSBSLCNet/K42 | BLOCKED | BLOCKED: thiếu checkpoint, prefix loader/extractor |
| E2 | Raw-byte | Không tìm thấy prefix checkpoint có provenance; path/hash UNKNOWN | CompactViewNet/K42 | BLOCKED | BLOCKED: như trên |
| E2 | Entropy | Không tìm thấy prefix checkpoint có provenance; path/hash UNKNOWN | CompactViewNet/K42 | BLOCKED | BLOCKED: như trên |
| Final | SBSMI | F-S, tồn tại/hash khớp | MalSBSLCNet/K51 theo config | READY (static) | BLOCKED: chưa có manifest-bound feature extractor |
| Final | Raw-byte | F-R, tồn tại/hash khớp | CompactViewNet/K51 theo config | READY (static) | BLOCKED: chưa có manifest-bound feature extractor |
| Final | Entropy | F-E, tồn tại/hash khớp | CompactViewNet/K51 theo config | READY (static) | BLOCKED: chưa có manifest-bound feature extractor |

**Phạm vi inventory VERIFIED:** shallow run/config enumeration ở
`data/derived/sbsmi_training`, `data/derived/float_view_training/{raw-byte,entropy}`
và `tiny_training`; inventory tên directory ngay dưới `data/derived`; search tên
file/extractor symbols trong scripts/models. Có10 training configs:2 SBSMI,3
raw-byte,5 entropy. Tất cả khai báo K51/train_samples18061/seed42, gồm one-epoch
smoke runs và các run chỉ có5/27/32/64 checkpoints. Không dùng các run này thay
registered baseline. Không suy lý do dừng hoặc selection từ số file. Không có
prefix checkpoint được đăng ký/tìm thấy trong scope này; không khẳng định đã
kiểm kê mọi ổ đĩa hoặc checkpoint không đăng ký bên ngoài repository.

## 2. Final checkpoint identity và provenance

Các paths dưới `C:/Users/hoang/Desktop/TRAMIF/`:

| ID | Checkpoint path | Bytes | SHA-256 VERIFIED |
| --- | --- | ---: | --- |
| F-S | data/derived/sbsmi_training/20261005_105325_352508/epoch_100.pt | 1846468 | `0c7d76ec3e077e5f88dd6d455b14a041ee2d701a2dc170d5718928695bcf2c4d` |
| F-R | data/derived/float_view_training/raw-byte/20261009_145102_408184/epoch_100.pt | 858884 | `893eb01789858abab523a3909d40a403328f27d17901dc76a9c94bfbd551341f` |
| F-E | data/derived/float_view_training/entropy/20261011_001259_203865/epoch_100.pt | 858884 | `1fc07d49f8dbc0f83cb8e88e1e987cbb070de707713f1f8d941f9e1626771cfa` |

Cả ba khớp `docs/baseline_artifacts.json`. Adjacent `config.json` hashes lần lượt:

- F-S: `7397ae51a3f728a0861378097335c35d0366f75fd3db824796bc166f2a245011`.
- F-R: `b1cbc65d82d2854bd1563464dea6fbdfe9ce031f877482d2fbc687306d56f5b3`.
- F-E: `960650411b1a5e926c841c311e05d387fd08c22b01b7e2fdd7d57b2c946428d8`.

**VERIFIED:** source hashes trong ba configs khớp current training/model/dataset
files (3/5/5 bindings). Canonical JSON dataset config IDs khớp cả ba training
configs. Class mappings trong configs khớp `docs/class_mapping.json`,51 outputs;
mapping file hash `a38b074ce5b977db8e7139b7539fc8f290cb970b7b4b7279edaad150eef007be`.
Ba preprocessing configs bind cùng historical manifest hash
`757251a8986c8808c304d980ff534c4cb8c557f54fcfe3ff168947b5808e362d`; file bytes khớp.

Raw/entropy bind `train_selection` threshold0, selected_samples18061; canonical
selection hashes khớp và retained train manifest bytes khớp
`ffc1f4663fb9cdcd23bb6b84c62318686b9199167e9dac90f489d5323784516a` tại
`data/derived/historical_near_duplicates_v1/20261009_143301_921802/threshold_00/train_keep.csv`.
SBSMI predates policy: trainer dùng original split train; config không có policy
selection hash. Reuse evidence G2/G3-B về unchanged18061 train rows, không audit
lại retention. Evaluator có explicit unchanged-training-selection guard.

**Cutoff INFERRED từ hash-matched code/config:** trainers dùng train split18061;
`SBSMIDataset`/`FloatImageDataset` quy định [2019-08-01,2020-02-01), nên Final
training phù hợp Aug–Jan; không phải prefix E1/E2. Hash chain không tự chứng minh
toàn bộ lịch sử ngoài logs. **UNKNOWN ở audit này:** metadata/tensor shapes bên
trong serialized checkpoint và strict-load runtime; không deserialize để kiểm tra.

**Seed/selection VERIFIED là config statements:** seed42 cả ba; raw/entropy ghi
epoch100 fixed before run/no metric-based selection và không evaluate validation
trong training. SBSMI ghi epochs100, save every completed epoch, validation_used=false,
future_test_used=false; không có explicit preregistered checkpoint-selection field.
Registry/evaluator pin epoch100; không biến điều đó thành bằng chứng preregistration.
Không chọn checkpoint mới từ outcome scores; chronology/selection honesty giữ nguyên.

## 3. Architecture và feature endpoint

- `models/malsbslcnet.py:117` (`MalSBSLCNet`), lines151–175: pool128×4×4,
  flatten2048, `head_norm=BatchNorm1d(2048)`, sau đó dropout0.4/classifier.
  **VERIFIED code** có đúng endpoint trước dropout; **INFERRED shape** [N,2048].
  Không có `forward_features` ở SBSMI. Cần adapter/hook có kiểm soát để lấy output
  head_norm; hiện forward chỉ trả logits. Không cần tự đổi architecture.
- `models/compact_view_net.py:47` (`forward_features`): pool128×2×2, flatten512,
  trả `head_norm` output. **VERIFIED code**, **INFERRED shape** [N,512]; forward
  thêm dropout/classifier. Raw/entropy dùng hai checkpoints riêng, không share weights.
- Cả hai constructors nhận num_classes>=2 nên architectural K32/K42 khả thi;
  điều này không tạo prefix-trained models. Expected endpoint không lệch V1.
- Search scripts/models chỉ thấy feature smoke checks và đề xuất extraction,
  chưa có production extractor bind G3-B reference/matched manifests. Temperature
  calibration không phải phép biến đổi các pre-classifier embeddings này.

## 4. Preprocessing, mappings và sample alignment

**VERIFIED static code/config:** tất cả input [N,1,64,64], float32.
`SBSMIDataset.__getitem__` (line146): PNG L/uint8, bytes/pixel hashes, chia255 một
lần. `FloatImageDataset.__getitem__` (line198): NPY float32 [64,64], finite và[0,1],
hash check; **không chia255 lần nữa**. Raw preprocessing đã chia255; entropy đã
chia8 (window256/stride128), mask-aware resize. Không random augmentation hoặc
fit dataset-level normalization trong các __getitem__ đã đọc. TRAIN embedding
z-score G3-B là bước riêng, không được trộn với image scaling.

Loaders giữ source-manifest order và kiểm tra report IDs/family/class index;
`HistoricalPolicyDataset` kiểm tra retention/order/hash rồi delegate base loader.
G3-B accepted reference/pair/matched manifests và S17/25/25 được reuse bằng
evidence D023, không chọn lại hoặc audit lại. K32/K42 là episode output mappings;
S không thay output mapping. Final giữ51 outputs, không thu về25.

**Blocker:** base loaders hard-code final51 và final split windows (SBSMI lines86,
100; Float lines108,120). Policy wrapper vẫn kế thừa final mapping. Trainers còn
require18061 rows. Không thể chỉ đổi num_classes hoặc month flag để có prefix
training đúng: phải có adapter dùng đúng G2 IDs/episode mapping/chronology và
storage split của images. A/B của E1/E2 nằm trong final train storage; không
được hiểu role A/B thành đường dẫn validation hoặc gán final class index cho K_e.
Hash-ranked G3-B order cũng khác source chronological order: cần explicit ID-based
index map, không zip arrays theo vị trí. Dataloader hiện trả tensor/label, không
trả SHA, nên extractor phải preserve/verify IDs bên ngoài. Actual alignment UNKNOWN.

## 5. Eval/state mutation và leakage review

`evaluate_sbsmi_filtered.py:124–160` và `evaluate_float_view.py:136–173` có strict
load, model.eval(), shuffle=False/drop_last=False và torch.inference_mode(); loops
chỉ lấy logits, không optimizer/backward. Đây là **VERIFIED static evaluation
patterns**, không phải extractor đã kiểm chứng. Không chạy các evaluator để lấy
embeddings vì chúng thực hiện evaluation và ghi outputs ngoài scope.

Trong eval, BatchNorm dùng saved running statistics và dropout tắt (**INFERRED từ
module semantics**). Bản thân forward_features/forward không tự bật eval/freeze;
caller quên eval có thể cập nhật BN buffers và sample-dependent behavior. No-grad
một mình không bảo vệ BN. Không thấy requires_grad_(False) trong evaluator;
inference_mode chặn graph trong loop nhưng persistent freeze/state invariance
vẫn cần explicit integration contract. Same encoder instance/state phải dùng cho
TRAIN reference/A/B; không fit BN trên A/B, không train-mode extraction.

Leakage blocker rõ nhất: Final CNN train qua Oct/Nov cutoff, không dùng cho E1/E2,
kể cả bỏ classifier hoặc lấy early epoch. Early epoch của full-window run vẫn
thấy full train split. Không truy cập outcome để sửa selection. Không thấy random
augmentation/stateful fitting trong loader paths đã đọc; actual behavior của
adapter/extractor chưa tồn tại vẫn UNKNOWN.

## 6. G3-C2 cần kiểm chứng riêng (chưa chạy)

1. Safe strict checkpoint load, embedded metadata/state-dict shape/buffer checks
   với correct case/view/config/mapping; fail closed nếu thiếu provenance.
2. Manifest-bound adapter và extractor: exact IDs/order/family/labels, correct
   storage paths/scaling, hooks/endpoints2048/512, no silent missing-sample skips.
3. Eval/frozen/no-grad execution, state hashes trước–sau, BN counters/running
   statistics bất biến; repeated input/single-vs-batch/chunk outputs trong tolerance
   được ghi trước; dtype/finite values và hook cleanup.
4. TRAIN-only parameters bind đúng encoder/reference; A/B không refit. Actual
   mask/rank/conditioning/h còn UNKNOWN, không cứu degeneracy bằng A/B/epsilon.
5. Bounded integration input/resource allowlist và immutable provenance logs.
   Runtime PASS không mở full extraction, observed MMD hoặc Future tự động.

## 7. Một bước tiếp theo đề xuất

**Xin owner phê duyệt một training gate riêng cho prefix E1/E2, bắt đầu bằng
thiết kế kế hoạch hữu hạn tạo sáu checkpoints đúng cutoff.** Gate phải chốt
G2 train manifests/K32/K42 mappings, prefix-capable loader, seed/config, checkpoint
selection không dùng outcomes và compute budget trước khi train. Không tự train
hoặc sửa legacy loaders trong audit này. Extractor integration sau đó cần approval
G3-C2 riêng; không dùng Final checkpoints để bỏ qua blocker prefix.

Chỉ tạo báo cáo/cập nhật checkpoint docs. Không chạy tests/runtime; hash và static
inspection không được ghi là inference PASS. Không sửa V1/models/manifests/
checkpoints/artifacts, không training/embeddings/MMD/Q/U/Future, commit/push.
**Dừng sau G3-C1 và chờ review.**
