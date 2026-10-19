# G3-B — Measurement preparation validation

Ngày: 2026-10-15. Scope: synthetic numerical tests và historical metadata only.
Không chạy embeddings, MMD thực tế, CNN, Q/U hoặc fusion.

## Contract ghi trước khi chạy audit

V1 §6 giữ nguyên. Những chi tiết serialization/tie-breaking chưa có giá trị cụ
thể trong V1 được cố định ở đây để tái lập implementation, không chọn theo
support, embeddings hay performance:

- SHA-256 ranking: hash UTF-8 của JSON compact array
  `["TRAMIF-G3B-v1",42,purpose,case,family,...ids]`, `ensure_ascii=True`.
  `purpose` là `train-reference`, `train-pair`, `matched-A`, `matched-B`.
  SHA phải là 64 ký tự hex lowercase. Digest tie được phá bằng ID lexical.
- TRAIN reference dùng toàn K theo mapping order; mỗi family lấy tối đa128.
  Pair được chọn từ reference này, unordered IDs lexical, không self-pair,
  tối đa128/family, cùng pair IDs cho ba views. Seed42/cap128 cố định.
- Weighted median là **lower weighted median**: giá trị nhỏ nhất mà cumulative
  mass đạt ít nhất 1/2. Khi mass đúng 1/2, không lấy trung bình với giá trị sau.
  Mỗi family có tổng pair mass bằng nhau; giữ zero distances. Rational weights
  dùng để quyết định median, tránh lỗi rounding tại tie; distances float64.
- Normalization dùng population variance của equal-family TRAIN mixture (ddof0),
  không lấy trung bình các within-family variances bỏ qua between-family means.
  Mask TRAIN std <=1e-8 đúng V1; không thêm epsilon. Overflow/non-finite,
  all-masked hoặc h<=0 phải fail.
- Cap A/B: n=min(N_A,N_B,128) mỗi family thuộc frozen S. Chỉ retained records
  được chọn; giữ nguyên group IDs từ graph đã có, kể cả excluded bridge effects.
- Báo riêng post-cap groups<5 và lost groups>=5. Không tự chọn stop predicate
  thay owner nếu hai cách đọc V1 dẫn đến quyết định khác nhau.

Đây là engineering contract cho G3-B, không xác nhận statistical independence,
bandwidth adequacy hay approval inference. Namespace và median tie convention là
chi tiết được công khai trong batch, không được mô tả là giá trị đã có sẵn trong V1.
Không thử nhiều namespaces, seeds hoặc conventions để tìm support thuận lợi.

## 1. Kết luận

**G3-B implementation/audit COMPLETE; gate cho real measurement BLOCKED.**

| Hạng mục | Kết luận có bằng chứng | Giới hạn |
| --- | --- | --- |
| Deterministic sampling | PASS: train reference, within-family pairs, matching/cap; source order reversal không đổi selection | Không chứng minh sample độc lập hoặc representative |
| TRAIN normalization | PASS trên synthetic known answers, masking, chronology và leakage guards | Actual mean/std/mask/rank/conditioning trên malware embeddings UNKNOWN |
| TRAIN bandwidth | PASS trên synthetic weighted median và provenance tests | Actual h/degeneracy trên ba encoders UNKNOWN |
| Frozen pre-cap support | PASS: E1=17, E2=25, Final=25; toàn bộ counts khớp saved support | Operational groups không phải independent groups đã được chứng minh |
| Post-cap support | Không trường hợp nào dưới5 groups; có15/24/27 family-month mất≥5 groups | **BLOCKED_OWNER_REVIEW** cho E1/E2/Final vì ambiguity ảnh hưởng quyết định |
| Alignment | PASS ở canonical manifest/config: cùng IDs/order/class mapping cho ba view plans | Actual image/embedding alignment UNKNOWN; chưa inference |

Không đổi Protocol V1, Amendment01, descriptive_mmd.py hoặc G3-A tests. Không
đánh dấu gate PASS chỉ vì tất cả targeted tests PASS.

## 2. TRAIN reference và preparation

Reference dùng K của từng case, không chỉ S. A/B matched statistic dùng frozen S.

| Case | TRAIN cutoff (exclusive) | K | TRAIN samples | Selected reference | Selected unordered pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | 2019-11-01 | 32 | 7.034 | 3.266 | 4.096 |
| E2 | 2019-12-01 | 42 | 9.803 | 4.197 | 5.376 |
| Final | 2020-02-01 | 51 | 18.061 | 5.354 | 6.528 |

**Chỉ chọn pair IDs, không tính distances của malware thật.** Mỗi family có128
pairs trong run này. Utility cho phép ít hơn nếu có ít unordered pairs và giữ
tổng mass mỗi family bằng nhau; family dưới2 reference samples phải fail.

Với reference family c có n_c samples và K families:

`mu_j = (1/K) sum_c [(1/n_c) sum_i x_cij]`

`var_j = (1/K) sum_c [(1/n_c) sum_i (x_cij - mu_j)^2]`

Mỗi sample mass `1/(K*n_c)`. Variance là population variance của mixture, có
between-family variation. Mask std<=1e-8; A/B chỉ transform bằng parameters này.
Không có fit(A/B) hoặc bandwidth fallback từ A/B. Các APIs kiểm tra timestamp UTC,
stage train, family/ID coverage và hash reference+embedding bytes khi nối bandwidth
với normalizer. Chúng không thể tự phát hiện caller giả mạo metadata/embedding
identity; loader/extractor provenance vẫn cần xác minh trước real inference.

Normalized TRAIN within-family pair distance là `rho=||z_i-z_j||²/p`, p là số
coordinates còn lại. Pair c có mass `1/(K*m_c)`. Lower weighted median rho là h;
không bỏ zero-distance pairs, không cộng epsilon khi median bằng0. Utility không
gọi MMD; synthetic integration test truyền h sang G3-A unchanged để kiểm tra API.

Numerics: float64, centered variance, lỗi overflow/non-finite dừng xử lý;
arrays parameters read-only. Weighted median dùng rational cumulative mass để
không đảo quyết định tại50% do rounding. Rank/condition diagnostic dùng SVD của
centered normalized TRAIN reference nhân căn row weights; rank tolerance là
`max(shape)*eps64*largest_singular_value`. Rank thiếu thì condition_number=null,
không bịa một số hữu hạn. Rank diagnostic không thay mask hoặc tạo threshold mới.
Chưa đo resource use của fitting trên actual2048/512-dimensional reference;
reference matrix và SVD đang in-memory. Pair ranking tối đa8.128 candidates/family
từ128 reference samples; đây không phải tính lại TLSH graph/pairs.

## 3. Historical cap findings

E1 A/B=Nov–Dec2019; E2=Dec2019–Jan2020; Final=Feb–Mar2020.

| Case | S | Samples A/B trước cap trong S | Samples mỗi phía sau cap | Min groups/family/month sau cap | Số family-month còn<5 | Số family-month mất≥5 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| E1 | 17 | 1.789 / 3.130 | 1.137 | 8 | 0 | 15 |
| E2 | 25 | 3.503 / 3.659 | 1.734 | 11 | 0 | 24 |
| Final | 25 | 3.380 / 4.064 | 1.768 | 7 | 0 | 27 |

Ví dụ cho thấy hai predicates khác nhau: E1 mira/B từ360 còn126 groups (mất234),
E2 upatre/B từ365 còn123 (mất242), Final upatre/A từ370 còn122 (mất248).
Chúng đều vẫn trên5 groups. Theo cách đọc literal “mất≥5” phải STOP review;
theo cách đọc “còn<5” không trigger. Không âm thầm chọn cách đọc thuận lợi.
Chi tiết mọi family/month và cả hai boolean nằm trong `cap_counts.csv`.

Giữ nguyên joint component IDs khi cap; không dựng lại groups hoặc đưa excluded
bridge samples vào statistic. Counts dưới đây chỉ đếm active contributions thuộc
S, không thay toàn cohort graph và không phải kiểm định independence:

| Case | Active groups trước→sau | Cross-month groups trước→sau | Multi-label groups trước→sau | Largest active group trước→sau |
| --- | --- | --- | --- | --- |
| E1 | 2.352→1.348 | 114→87 | 35→21 | 127→68 |
| E2 | 3.790→2.170 | 203→134 | 47→28 | 187→74 |
| Final | 3.661→2.069 | 161→105 | 36→22 | 278→93 |

Post-cap còn shared/multi-label components. Cap không giải quyết null validity;
null-standardization vẫn suspended/NOT ESTABLISHED theo D019.

## 4. Provenance, artifacts và kiểm chứng

Audit run:

`data/derived/g3b_measurement_validation/20261015_073946_804374/`

Verification độc lập bằng stdlib, không import audit utilities:

`data/derived/g3b_measurement_verification/20261015_074109_956562/report.json`

- **26/26 output hashes và38/38 input/source hashes PASS**; persisted manifest IDs
  unique, frozen support giữ nguyên và equal A/B counts được kiểm tra lại.
- Audit `checksums.json` SHA256:
  `1b581899db556e2c2076eae1f281323922a80c54e99bf247e16a3566ea1985b1`.
- `report.json`: finite allowlist, input/source hashes, HEAD và case summaries.
- `config.json`: seed/cap/namespace/mask/median contract.
- Mỗi case: `train_reference.csv`, `train_pairs.csv`, `matched_samples.csv`,
  `cap_counts.csv`, `class_mapping.json`, `support.json`, `view_plan.json`,
  `summary.json`. Một matched manifest dùng chung cho ba views, có ordered hash.
- Chỉ đọc G2/Phase1 checksums, metadata reports, mappings, support, AB memberships,
  train/A/B keeps và alignment records; frozen historical manifest và ba dataset
  configs. Không mở edges, binaries/images, checkpoint, predictions/losses hoặc
  raw metadata chứa Future. Saved graph10 memberships được reuse, không rebuild.
- G2 E1/E2 manifests và final dataset configs chứng minh source identity ở mức
  metadata; không giả vờ kiểm tra ba embedding arrays chưa tồn tại.

Targeted commands đã chạy:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_mmd_measurement_preparation.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_g3b_historical_metadata.py -v
.\.venv\Scripts\python.exe -B -m scripts.audit_mmd_measurement_preparation
```

**17 synthetic tests PASS (0,091s); 6 historical metadata tests PASS (6,637s).**
Tổng23/23 targeted tests; không rerun toàn repository hoặc cộng vào count cũ để
claim full suite. Hash verification sau run PASS. Tests bao gồm hand-calculated
moments/median, row-order invariance, finite checks/masking, rejected chronology,
no A/B refit, reference provenance mismatch, pair uniqueness/caps, mẫu synthetic
phân biệt hai stop predicates, alignment corruption và finite-allowlist rejection.
Test results receipt được ghi từ tool output thực tế, không giả thành raw log.

## 5. Blockers trước real embeddings

1. **Owner phải xác nhận exact post-cap stop predicate**, hoặc phê duyệt amendment
   mới nếu đổi ý nghĩa; V1 hiện giữ nguyên. Không đổi seed/cap/S để vượt guard.
2. Reference/pair/cap contract và manifests của batch này cần review; chúng là
   preparation artifacts, không tự mở authorization real measurements.
3. E1/E2 cần đúng prefix-trained encoders; final January-trained encoders không
   thay thế được. Availability/cutoff và extractor layer/provenance cần audit riêng.
4. Actual train moments, mask, rank/conditioning và bandwidth chưa biết. Có thể
   fail all-masked, non-finite hoặc h<=0; không cứu bằng A/B hoặc epsilon.
5. Actual extractor sample/class order, eval mode/no state mutation và feature
   dimensions chưa integration verified trong G3-B. Không suy từ metadata PASS.

Không scientific threshold/coefficients mới, không observed MMD/Q/U, không
Future, training/inference, legacy changes, commit/push. **Dừng tại G3-B để owner
review; không tự chạy bước tiếp theo.**
