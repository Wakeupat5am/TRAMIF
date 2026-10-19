# Amendment 01 — Descriptive Drift exploratory

Ngày **15/10/2026, Asia/Saigon**. Quyết định owner: **D019**.
Áp dụng cùng [Research Protocol V1](C:/Users/hoang/Desktop/TRAMIF/docs/RESEARCH_PROTOCOL_V1.md), không sửa hoặc thay thế bản V1.
V1 SHA-256: `e64664c4c499ef762b4a52c45889f0d05db6741ec5021b249f2072585f816546`.

## 1. Phạm vi được phê duyệt

Owner chấp nhận kết luận [G4 audit](C:/Users/hoang/Desktop/TRAMIF/docs/matched_null_feasibility_report.md): same-period group-disjoint matched null với exact V1 budgets là **INFEASIBLE** cho E1/E2/Final; statistical validity của các null designs khác **NOT ESTABLISHED**, không phải chứng minh mọi null đều bất khả thi.

**APPROVED SCOPE CHANGE:** tạm dừng null-standardization trong phạm vi nghiên cứu hiện tại; phê duyệt có điều kiện descriptive class-conditional MMD theo V1 §6.4 làm phép đo Drift **exploratory**. G4 được xử lý bằng một approved departure, **không chuyển kết luận null validity thành PASS**.

**AUTHORIZATION HIỆN TẠI CHỈ LÀ DOCUMENTATION.** Chưa cho phép viết implementation/tests, chạy synthetic experiments, chọn sample cap manifest, tạo embeddings, tính MMD thực tế, training hoặc inference. Các bước kỹ thuật ở §4 là đề xuất cho batch được duyệt sau.

Các kết luận và stop decisions trong báo cáo G4/D018 được giữ nguyên như lịch sử. Amendment này chỉ cập nhật quyết định scope sau khi owner review.

## 2. Thay đổi so với framework và giới hạn claims

| Nội dung | Framework/V1 primary | Scope được phê duyệt có điều kiện |
| --- | --- | --- |
| Drift | Null-standardized class-conditional MMD | Descriptive class-conditional MMD; null-standardization tạm dừng |
| Phép biến đổi | Trừ null mean, chia null variability | `d_desc = D_cond / 2`, chỉ là phép đưa vào khoảng số học |
| Diễn giải | Phụ thuộc valid matched null | Exploratory description dưới encoder, kernel, normalization và sampling đã công khai |
| RQ1 | View stability với các controls phù hợp | Chỉ mô tả/sensitivity của view–encoder pairs; không chứng minh intrinsic representation superiority |
| RQ2/RQ3 | Predictive value ngoài Q / frozen reliability fusion | Chưa được trả lời bằng descriptive distance; cần protocol/approval và evidence riêng |

Theo định nghĩa V1, `D_cond` là trung bình đều theo frozen families của **biased MMD²**, có diagonal terms, dùng RBF kernel. `D_cond/2` nằm trong [0,1] về mặt toán học. Đây **không phải** standardized drift, significance score, p-value, probability of drift hoặc một phép hiệu chỉnh thống kê.

Không coi cùng numerical range là bằng chứng ba encoders có cùng thang đo thống kê. Z-score, chia dimension và bandwidth riêng không tự giải quyết comparability. Một positive MMD² trên hai finite samples cũng không tự chứng minh population shift hoặc dự báo future model error. Penalty “D lớn ⇒ reliability thấp” chưa được xác nhận và chưa được phê duyệt để weighting.

**Giữ nguyên:** frozen support E1 S=17, E2 S=25, Final S*=25; episode/final chronology; prefix eligibility và final51 mapping; primary retention0, sensitivity10/20, operational grouping10; excluded bridge policy; Future protection. Không hạ support hoặc bỏ family khi gặp kết quả khó. Không gộp outcome C vào signal graph A/B.

**Giữ đối chứng quan trọng:** Quality-only với matched information budget và learned static fusion. Không thay chúng bằng chỉ Uniform để làm claim thuận lợi hơn. Amendment không xác nhận learned static fusion đã được triển khai hoặc chạy.

**CHƯA PHÊ DUYỆT:** beta/gamma, grid/scale cho coefficients, D trong fusion weights, weights mới, U methodology hoặc các gates khác. Không chuyển coefficients của standardized D sang `d_desc`; không dùng March/outcome/Future performance để chọn bandwidth, cap hoặc coefficients.

## 3. Trạng thái bằng chứng

- **VERIFIED qua báo cáo đã có:** G4 có count blockers và unresolved dependence/exchangeability; support/memberships của G2 và Phase1 có provenance. Không chạy lại audit trong batch tài liệu này.
- **APPROVED:** scope departure và claim limits nêu trên, có điều kiện kiểm chứng kỹ thuật trước khi dùng phép đo.
- **PROPOSED, NOT EXECUTED:** kế hoạch §4; chi tiết numerical tolerances/sampling namespace chưa tự trở thành approved implementation.
- **UNKNOWN:** actual embedding alignment/degeneracy, bandwidth, post-cap group support, descriptive scores và cross-view stability. Không tạo giá trị thay thế hoặc suy kết quả từ tests cũ.

## 4. Kế hoạch kiểm chứng kỹ thuật D — chưa thực hiện

Thứ tự bốn task hữu hạn; mỗi task cần scope approval tương ứng. Không mở inference chỉ vì tests synthetic PASS.

| Task | Mục đích và nội dung đề xuất | Acceptance / stop | Dependencies và outputs dự kiến |
| --- | --- | --- | --- |
| **D1 — Synthetic numerical contract** | Kiểm tra RBF biased MMD²: identical empirical sets cho0; symmetry; within-set order invariance; known-answer toy kernels; dense/chunked parity; bounds; NaN/Inf và invalid shapes. Distinct finite samples dưới cùng distribution không bị ép trả0. Thêm constructed mean-shift cases nhưng không tuyên bố mọi shift đều monotonic với một bandwidth. | Full-precision results khớp reference độc lập trong float64 tolerance được chốt trước; không clip lỗi lớn. Test PASS chỉ xác nhận phép tính, không xác nhận significance/null calibration. | Approval viết code/tests. Outputs: specification, targeted tests, test log; không dữ liệu thực hoặc embeddings. |
| **D2 — Train-reference normalization và bandwidth contract** | Theo V1 §6.1: deterministic train-only reference, equal-family mass, coordinate mean/variance, mask low-variance dimensions; squared distance chia số dimensions còn lại; weighted median within-family train pair distances cho bandwidth. Giữ zero-distance pairs. Kiểm tra hand-calculated weighted moments/median, pair sampling không self/duplicate, order invariance và bindings với cutoff. | A/B/C không thể thay đổi fitted train parameters nếu train cố định; không fit lại từ evaluation. All-masked, non-finite hoặc median≤0 phải STOP; không thêm epsilon cứu bandwidth. Layer/dimension, tolerance và weighted-median tie convention phải được chốt trước implementation. | D1; synthetic/mock data đủ kiểm tra logic. Outputs: reference/pair selection contract và tests. Actual normalization/bandwidth cần embeddings và approval riêng, chưa có trong batch này. |
| **D3 — Historical post-cap support/alignment audit** | Dùng memberships đã lưu, không rebuild graph. Với từng c trong frozen S: `n_c=min(N_Ac,N_Bc,128)` ở cả hai tháng; deterministic SHA ranking, cùng source IDs giữa views. Chốt namespace/seed42/tie rules trước chọn; lưu manifests/hashes. Đếm groups trước/sau cap, shared/multi-label incidence và concentration, giữ original component IDs. | Counts đúng, IDs unique và shared giữa views, không dùng losses/features để chọn, không đưa excluded samples vào scoring. Kiểm tra group-support rule được xác nhận trước; fail thì STOP, không đổi seed/cap/S hoặc bỏ shared groups để đạt. | Approval historical metadata-only audit; không cần inference. Outputs: cap manifests, per-family support report và provenance mới, không ghi đè legacy. |
| **D4 — Integration readiness và bounded diagnostic plan** | Kiểm tra cùng encoder frozen cho A/B, correct prefix cutoff, layer SBSMI head_norm2048 và CompactViewNet features512 theo V1; train-only parameter provenance, input/output class/sample alignment, eval-mode/no-state-mutation contract. Thiết kế diagnostics bandwidth h/2,2h, cap64 và group-balanced đối với primary128 theo V1. | Mock tests/metadata checks không được ghi là actual extraction verified. Actual extraction/MMD chỉ được chạy sau explicit approval; prefix không dùng January-trained encoder. Sensitivities phải báo cả kết quả bất lợi, không chọn cấu hình thắng để thay primary. | D1–D3, sampling/numerical contract resolved; approval riêng trước real-data inference/MMD và trước sensitivities. Outputs trước execution: readiness checklist và bounded run specification, không scores/weights. |

**Điểm cần làm rõ trước D3, không tự sửa V1:** câu §6.2 “Nếu cap làm mất ≥5 groups/class/month” có thể bị hiểu là mất ít nhất5 groups so với trước cap hoặc còn dưới5 groups. Kế hoạch phải ghi cả counts trước/sau và số groups mất; cần owner xác nhận exact stop predicate trước khi viết guard. Ngưỡng support gốc ≥20 samples/≥5 groups mỗi tháng vẫn giữ nguyên. Không coi amendment này đã âm thầm giải quyết ambiguity đó.

Normalization/bandwidth của §6.1 và cap/sensitivities của §6.2/6.4 là cơ sở lập kế hoạch, không phải bằng chứng chúng đã chạy hoặc blanket authorization cho mọi chi tiết còn PROPOSED trong V1. Numerical bound/known-answer tests không tạo statistical calibration giữa views. Bất kỳ sensitivity nào cần thay support hoặc aggregation phải được duyệt riêng; không sửa estimand để cứu kết quả.

## 5. Điều kiện dừng và bước đề xuất

STOP khi provenance/cutoff/alignment không khớp, cap support không đạt, normalization/bandwidth suy biến, non-finite hoặc numerical discrepancy ngoài tolerance. Báo lỗi/negative findings; không tự đổi threshold, sampler, layer, kernel, support hoặc weights.

**Bước tiếp theo đề xuất:** xin phê duyệt **D1 synthetic numerical contract/tests** với scope chỉ synthetic, cùng các numerical acceptance details được chốt trước khi chạy. D2–D4 vẫn là roadmap chưa thực hiện. Sau khi technical verification hoàn tất cũng chưa được dùng D trong fusion: coefficient/weighting approval là gate riêng.

Batch này chỉ ghi nhận quyết định D019 và kế hoạch. Không sửa legacy code/models/artifacts hoặc V1; không train/inference, embeddings, MMD/Q/D/U, Future, commit/push. **DỪNG sau documentation.**
