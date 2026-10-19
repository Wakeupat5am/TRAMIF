# G3-B — Focused post-cap owner review

**Current owner decision (2026-10-15, D023): G3-B CLOSED
PASS_WITH_LIMITATIONS for E1/E2/Final.** Sections1–6 retain the original review
and proposal; their pending-approval wording is historical. See sections7–8 for
the accepted decision and separate, unexecuted next-gate checklist.

Ngày: 2026-10-15. **Kết luận đề xuất: E1/E2/Final PASS_WITH_LIMITATIONS cho sampler
metadata review; chưa được owner phê duyệt và không tự đóng gate.**

## 1. Quy tắc owner đã xác nhận và evidence

- Còn <5 groups/family/month sau cap: support failure.
- Mất >=5 groups/family/month: STOP để review sampler, không tự động support failure.
- Giữ seed42, cap128, frozen S, hash-ranking algorithm và mọi selected IDs.

Nguồn duy nhất cho số liệu: G3-B run
`data/derived/g3b_measurement_validation/20261015_073946_804374/` và
`docs/g3b_measurement_validation_report.md`. Đã kiểm lại checksum manifest
`1b581899db556e2c2076eae1f281323922a80c54e99bf247e16a3566ea1985b1`, rồi
**12/12 hashes PASS** cho `{E1,E2,Final}/{summary.json,cap_counts.csv,
matched_samples.csv,support.json}`. Từ matched manifests, đếm độc lập unique IDs,
support, counts/groups/largest group từng family-month và joint incidence sau cap;
tất cả khớp artifacts. Không chạy lại sampler hoặc mở nguồn G2/Phase1.

**VERIFIED** trước cap là số liệu đã lưu trong G3-B, không tái dựng graph hoặc
đọc lại pre-cap memberships. Actual pre-cap size của từng group trong từng cell
không có đầy đủ ở bộ artifacts này; tác động tăng/giảm concentration theo từng
group vì vậy **UNKNOWN**, ngoại trừ cell giữ toàn bộ samples.

## 2. Mất groups và concentration tổng thể

“Group” trong bảng này là distinct original component có ít nhất một contribution
thuộc S trong A hoặc B; không cộng đúp shared-month/multi-label groups.
Loss rate = (G_before − G_after)/G_before. Các groups không bị xóa khỏi graph;
chỉ không còn sample được chọn trong phạm vi đang đếm.

| Case | Joint groups trước→sau | Groups mất | Tỷ lệ mất | Samples sau cap A+B | Largest joint group sau cap / samples | Min groups/cell | Cells <5 / mất>=5 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| E1 | 2.352→1.348 | 1.004 | 42,69% | 2.274 | 68/2.274 = 2,99% | 8 | 0 / 15 |
| E2 | 3.790→2.170 | 1.620 | 42,74% | 3.468 | 74/3.468 = 2,13% | 11 | 0 / 24 |
| Final | 3.661→2.069 | 1.592 | 43,49% | 3.536 | 93/3.536 = 2,63% | 7 | 0 / 27 |

Theo từng tháng, largest active group sau cap chiếm: E1 A=34/1.137 (2,99%),
B=40/1.137 (3,52%); E2 A=53/1.734 (3,06%), B=47/1.734 (2,71%);
Final A=53/1.768 (3,00%), B=55/1.768 (3,11%). Các global fractions nhỏ **không**
chứng minh từng family ít concentration: MMD chính được tính theo family.

## 3. Các family/month tập trung cao nhất

Top3 mỗi case theo largest-group sample fraction trong cell; đây là descriptive
ranking của một selection cố định, không phải threshold mới để pass/fail.

| Case / family / month | Samples trước→sau | Groups trước→sau | Groups mất (%) | Largest group sau cap |
| --- | --- | --- | --- | --- |
| E1 / dinwod / Dec2019 | 128→92 | 55→43 | 12 (21,82%) | 40/92 = 43,48% |
| E1 / qqpass / Nov2019 | 26→20 | 9→8 | 1 (11,11%) | 7/20 = 35,00% |
| E1 / dinwod / Nov2019 | 92→92 | 40→40 | 0 (0%) | 28/92 = 30,43% |
| E2 / dinwod / Dec2019 | 128→128 | 54→54 | 0 (0%) | 52/128 = 40,63% |
| E2 / sytro / Dec2019 | 57→54 | 22→22 | 0 (0%) | 19/54 = 35,19% |
| E2 / small / Jan2020 | 118→118 | 31→31 | 0 (0%) | 41/118 = 34,75% |
| Final / agent / Feb2020 | 21→21 | 9→9 | 0 (0%) | 12/21 = 57,14% |
| Final / agent / Mar2020 | 47→21 | 9→7 | 2 (22,22%) | 10/21 = 47,62% |
| Final / benjamin / Mar2020 | 70→70 | 24→24 | 0 (0%) | 31/70 = 44,29% |

Chỉ Final/agent/February có một group chiếm >50% trong cell; đây là mô tả,
không phải acceptance cutoff được đặt sau khi nhìn số liệu. Cell này không bị
subsample nên concentration được giữ nguyên từ input. E2/dinwod/December cũng
giữ toàn bộ128 samples. Ngược lại, không đủ pre-cap group-size detail để kết luận
hash selection đã khuếch đại hay giảm concentration của E1/dinwod/December hoặc
Final/agent/March. Không đọc thêm dữ liệu hoặc thử seed để lấp khoảng trống này.

## 4. Shared-month và multi-label incidence

Incidence chỉ trong contributions thuộc frozen S, giữ nguyên IDs của joint graph.
Một group có thể đồng thời shared-month và multi-label; không cộng hai cột.

| Case | Shared-month groups trước→sau | Multi-label groups trước→sau | Samples sau cap thuộc shared groups | Samples sau cap thuộc multi-label groups |
| --- | --- | --- | --- | --- |
| E1 | 114→87 | 35→21 | 600/2.274 | 119/2.274 |
| E2 | 203→134 | 47→28 | 910/3.468 | 279/3.468 |
| Final | 161→105 | 36→22 | 872/3.536 | 347/3.536 |

Selection không loại bỏ dependence giữa tháng hoặc giữa families. Những counts
này không đo covariance hoặc xác nhận group independence.

## 5. Đánh giá khoa học và đề xuất theo case

**INFERRED từ cơ chế:** sample-level ranking không dùng group size để cân bằng.
Nó cho mỗi selected file cùng mass trong family; group có nhiều files có thể có
mass lớn. Nếu xem hashes như pseudo-random ordering thì đây tương ứng sampling
theo files, không uniform sampling theo groups. Một deterministic selection không
tự chứng minh tính ngẫu nhiên, representativeness hoặc absence of sampling bias.

Trong biased sample-level MMD, group có m/n files tạo (m/n)^2 phần ordered
within-month kernel terms, và tương tác với mọi group khác/cross-month terms.
Ví dụ agent/February là (12/21)^2≈32,65% terms chỉ từ group đó. Đây là tỷ lệ
terms, **không phải** tỷ lệ MMD score: chưa có embeddings/kernel values, và MMD
là hiệu của ba kernel means. Không suy ảnh hưởng thực tế hoặc hướng drift từ
metadata. Equal-family aggregation giới hạn mỗi family ở1/|S| nhưng không làm
các files trong family độc lập hay cân bằng groups.

Group-balanced sampling/weighting sẽ đổi empirical target từ file mixture sang
group-weighted mixture. Không mặc định nó tốt hơn; không thay primary sampler.
Descriptive empirical MMD vẫn tính được với repeated/dependent samples, nhưng
i.i.d. inference, significance, effective sample size và future predictive value
không được bảo đảm. Null-standardization vẫn chưa được thiết lập.

| Case | Đề xuất | Lý do và giới hạn bắt buộc |
| --- | --- | --- |
| E1 | PASS_WITH_LIMITATIONS | Mọi cell>=8 groups; coverage sau cap đúng; concentration cao nhất43,48%. Mất42,69% active groups và87 shared groups còn lại phải được công bố. Không có evidence metadata buộc đổi sample-level estimand, cũng không có chứng nhận representativeness. |
| E2 | PASS_WITH_LIMITATIONS | Mọi cell>=11 groups; cao nhất40,63% đã có trong cell không subsample. Mất42,74% groups;134 shared groups và28 multi-label groups còn lại giới hạn inference. |
| Final | PASS_WITH_LIMITATIONS | Mọi cell>=7 groups; agent/February57,14% là concentration có sẵn, không phải cap-induced support failure. Agent/March47,62% và loss43,49% toàn case là limitations trọng yếu; không gọi25 families là25 independent replications. |

Đề xuất không dựa trên threshold concentration mới hoặc tiêu chí “không majority
là an toàn”. PASS_WITH_LIMITATIONS chỉ nói có thể giữ selection cho **descriptive
sample-level target** với hạn chế nêu rõ; không phải gate inference/embeddings PASS.

## 6. Một quyết định cuối cùng để owner phê duyệt

**Đề nghị owner chấp nhận sampler review E1/E2/Final là PASS_WITH_LIMITATIONS,
giữ nguyên manifests/seed42/cap128/S/algorithm, và ghi nhận các review triggers
mất>=5 groups đã được xem xét, không phải support failure.** Công bố per-family
concentration/shared incidence cùng mọi descriptive result sau này. Không thêm
điều kiện phải tìm seed hoặc sampler khác để giảm concentration.

Đây là đề xuất, **chưa đóng gate**. Real embeddings vẫn cần authorization riêng
và các checks prefix encoder, train-only parameters, numerical degeneracy và
actual alignment còn UNKNOWN từ G3-B. Không mở implementation, sensitivity hoặc
thí nghiệm mới trong review này. Không sửa V1/legacy artifacts, không training,
inference, embeddings, observed MMD/Q/U, Future hoặc commit/push. **DỪNG.**

## 7. Owner acceptance — D023 (2026-10-15)

Owner đã chấp nhận **PASS_WITH_LIMITATIONS** cho sampler review E1/E2/Final.
Các triggers mất>=5 groups đã được review và accepted; không còn pending ở
G3-B. Còn<5 groups/family/month là support failure; mất>=5 là bắt buộc review,
không tự động loại family. Giữ nguyên seed42, cap128, frozen S, sampling algorithm
và toàn bộ matched manifests. Không chỉnh immutable artifacts để đổi status cũ.

Tiếp tục báo group loss, concentration, shared-month/multi-label dependence như
sections2–4. Target là **sample-level descriptive MMD**, không statistical
independence hoặc null-standardized Drift. Không đưa group-balanced resampling
vào pipeline. Acceptance này không khẳng định actual embedding/MMD validity.

**Predeclared requirement:** cần sensitivity analysis trước khi đưa claim mạnh
về stability. Thiết kế, giới hạn diễn giải và cách báo mọi kết quả phải được
chốt trước; không chọn alternative methods bằng observed Drift để thay primary.
Chưa chạy sensitivity, chưa phê duyệt sampler thay thế, không đổi primary
seed/cap/support/manifests. Mọi sensitivity đổi đối tượng đo cần approval riêng.

## 8. Minimum checks cho frozen-encoder/extractor integration gate riêng

**Chỉ checklist đề xuất, chưa thực hiện.** Gate phải có authorization riêng;
không tự mở training nếu thiếu checkpoint hoặc tự dùng encoder sai chronology.

1. **Checkpoint và chronology:** xác minh hash, architecture, class mapping và
   training provenance. E1 train Aug–Oct, E2 Aug–Nov; Final Aug–Jan. Cùng một
   frozen encoder/view cho TRAIN reference và A/B của case đó. Nếu prefix
   checkpoints thiếu hoặc cutoff không xác minh được: BLOCKED, không thay bằng
   Final encoder và không tự retrain.
2. **Input identity:** bind hashes của accepted reference/pair/matched manifests,
   preprocessing config và checkpoint. Chỉ allowlist historical IDs; check exact
   IDs/order/families/class order ở cả ba views; không silently sort/intersect hoặc
   bỏ sample lỗi. Check image/content provenance khi được phép đọc input thật.
3. **Feature endpoint:** xác minh SBSMI `head_norm` sau flatten, trước dropout và
   classifier (2048); CompactViewNet `forward_features` (512). Check correct
   input scaling, batch/sample axes và dimension; không lấy logits làm embeddings.
4. **Frozen execution:** `eval`, no-gradient extraction, không augmentation hoặc
   optimizer step; BatchNorm/dropout đúng mode. Check parameters/buffers trước–sau
   không đổi; repeated input và single/batched extraction khớp trong numerical
   tolerance ghi trước. Không mặc định bitwise equality trên mọi hardware.
5. **TRAIN-only preparation binding:** selected TRAIN IDs khớp reference; mean,
   variance/mask và h chỉ fit từ TRAIN của đúng case/view. A/B chỉ transform.
   Kiểm tra finite values, retained dimensions, rank/conditioning và h>0; all-
   masked/non-finite/h<=0 phải STOP, không epsilon rescue hoặc refit từ A/B.
   Không tính observed MMD trong integration gate trừ approval riêng.
6. **Bounded verification và evidence:** sau khi được phép, chạy integration
   check nhỏ theo resource budget/allowlist đã chốt, không tự full extraction.
   Lưu source/config/checkpoint/input hashes, layer, IDs/order, shapes/dtype,
   environment, tolerance và failures trong run mới. Mock tests không thay bằng
   claim real integration verified; báo từng case PASS/FAIL/UNKNOWN và dừng nếu
   provenance/alignment sai. Việc full extraction/MMD phải có phạm vi được duyệt.

Batch closure này chỉ sửa decision/progress/report; không chạy bất kỳ check
integration nào ở trên, không sửa V1/code/manifests/checkpoints/artifacts, không
training/inference, real embeddings, observed MMD, Future, commit/push. **DỪNG.**
