# TRAMIF — Research Protocol V1

Ngày: **14/10/2026, Asia/Saigon**.
Source checkpoint: **d8f0c676f604a744f06800ab63c69dc6fe8beea3**.
Trạng thái: **ĐẶC TẢ CÓ DECISION GATES — không phải authorization chạy QDU/Future Test**.

## 0. Phạm vi quyết định và evidence labels

- **VERIFIED:** xác nhận từ source, Git hoặc artifacts được dẫn; không đồng nghĩa research hypothesis đúng.
- **PROPOSED:** thiết kế chính của V1, cần phê duyệt ở gate tương ứng trước triển khai.
- **UNKNOWN:** chưa đủ bằng chứng; không thay bằng giả định thuận lợi.

Yêu cầu viết tài liệu không tự phê duyệt retraining, inference, estimators, coefficients hoặc Future Test. Legacy methodology, weights, manifests và code không đổi.

**Primary design PROPOSED:** frozen known-family classification; phát triển trên chronological prefix episodes; đo Q và kiểm chứng D/U; chọn một rule nhỏ bằng lịch sử; freeze; so sánh cùng information budget trên sáu tháng tương lai. D chính theo framework là **null-standardized class-conditional MMD**, chỉ dùng nếu null gate đạt. Hiện gate này CHƯA đạt. V1 cho một audit hữu hạn và fallback cần phê duyệt, không cho tìm null/coefficient vô hạn đến khi có kết quả đẹp.

V1 chốt cấu trúc ra quyết định. Không thể xác nhận validity của estimator chỉ bằng cách viết protocol.

## 1. Nguồn chuẩn và trạng thái kiểm chứng

### 1.1 Nguồn

1. [Framework gốc](C:/Users/hoang/Desktop/Capstone/Temporal_Reliability_Malware_Research_Framework.docx), 16/09/2026; SHA-256 **829f50703c12c0b1d1f876ec13cdd8320bafe429266e8a07f9c81f75352c3400**.
2. Git checkpoint trên; [AGENTS.md](C:/Users/hoang/Desktop/TRAMIF/AGENTS.md).
3. [PROJECT_PROGRESS.md](C:/Users/hoang/Desktop/TRAMIF/PROJECT_PROGRESS.md), [DECISION_LOG.md](C:/Users/hoang/Desktop/TRAMIF/DECISION_LOG.md), [TECHNICAL_WALKTHROUGH.md](C:/Users/hoang/Desktop/TRAMIF/TECHNICAL_WALKTHROUGH.md).
4. Code và immutable artifacts. Statement cũ trong README không ghi đè bằng chứng mới.

**VERIFIED khi soạn:** HEAD đúng checkpoint yêu cầu; workspace ban đầu sạch; đọc framework gồm tables/equations. Không chạy tests, training, inference, MMD, bootstrap mới hoặc xử lý Future Test. Đã kiểm tra lại checksum manifests:

| Directory dưới C:/Users/hoang/Desktop/TRAMIF/data/derived/ | Kết quả |
| --- | --- |
| quality_weight_freeze/20261013_124031_275438/ | 3/3 khớp |
| quality_march_evaluation/20261013_125046_074164/ | 3/3 khớp |
| march_paired_bootstrap/20261013_131843_476287/ | 9/9 khớp |
| balanced_phase1_audit/20261013_233825_921873/ | 9/9 khớp |

Checksum xác minh integrity, không thay thế statistical review hoặc tái chạy experiment.

### 1.2 Những gì đã có

**VERIFIED từ code/reports:**

- Final cohort 51 families; train 18.061 samples; primary retained February 3.799 và March 4.332.
- Ba views có preprocessing/model/training artifacts, chung class order. SBSMI dùng MalSBSLCNet; raw-byte/entropy dùng hai CompactViewNet riêng.
- Full-February calibration, Uniform và legacy February OOF Quality-only đã có. Legacy weights: SBSMI 0.4017215600259309; raw-byte 0.2866850774030112; entropy 0.3115933625710579.
- March Q−Uniform: Accuracy +0,507849 điểm phần trăm; NLL −0,012938074; Brier −0,006207650. Pointwise conditional intervals có lợi; Q vẫn kém SBSMI về Brier. Không suy ra thắng mọi baseline/metric.
- Joint historical audit: 4.048 active components; 178 components xuyên tháng chứa 2.197 retained samples; 41 multi-label components; lớn nhất 278.
- Candidate S*: 25 families, chứa 3.380 February/4.064 March samples; classifier vẫn 51 outputs.
- 130/130 tests PASS là checkpoint Phase 1, không phải tests chạy lại khi viết V1.

Nguồn: [Phase 1 results](C:/Users/hoang/Desktop/TRAMIF/docs/balanced_phase1_results.md), [March results](C:/Users/hoang/Desktop/TRAMIF/docs/march_quality_only_results.md), [bootstrap results](C:/Users/hoang/Desktop/TRAMIF/docs/march_paired_bootstrap_results.md).

**UNKNOWN/chưa thực hiện:** E1/E2 feasibility; actual embeddings; valid matched null; D/U usefulness; new coefficients/weights; multi-seed stability; Future outcomes. README còn ghi chưa có training; base commit cũ trong checkpoint text là lịch sử, không phải HEAD hiện tại.

## 2. Research questions và falsifiable claims

| RQ theo framework | Đại lượng/đối chứng | Kiểm chứng và kết quả âm hợp lệ |
| --- | --- | --- |
| **RQ1 — View stability** (§1.5, §12.1) | Within-family drift của ba fixed view–encoder pairs; chung sample matching/family weighting | Kiểm tra khác biệt qua bandwidth/support/group controls. Không thấy khác biệt ổn định cũng là kết quả; architectures khác nhau nên không quy toàn bộ khác biệt cho representation |
| **RQ2 — Predictive value của D ngoài Q** (§12.2) | Next-window per-view class-balanced NLL forecast: Q so với Q+D; Q+U/QDU là mở rộng | Đánh giá cửa sổ SAU thời điểm tính signals; đo predictive error, không chỉ correlation/fusion accuracy. D không giảm predictive error là kết quả âm |
| **RQ3 — Frozen reliability fusion** (§12.3) | Mean monthly và worst-month Macro-F1; QDU so với New Q/strong matched controls | Cùng samples, class mapping, temperatures, history, checkpoint budget. Chỉ thắng Uniform không xác nhận D/U có giá trị ngoài Q |

Không claim benign detection, naming unseen families, online adaptation hoặc hiệu quả trên malware năm 2026. Unknown/excluded-family prevalence vẫn phải báo cáo sau khi được phép xử lý Future. Contrastive learning là optional theo framework.

## 3. Timeline và information budget

| Giai đoạn | Khoảng ngày nửa mở, UTC | Vai trò |
| --- | --- | --- |
| Final train | [2019-08-01, 2020-02-01) | Family eligibility; CNN; normalization/bandwidth reference |
| February | [2020-02-01, 2020-03-01) | A: calibration và drift reference |
| March | [2020-03-01, 2020-04-01) | B: New Q/U và drift A→B; historical development đã quan sát |
| Future Test | [2020-04-01, 2020-10-01) | Sáu tháng evaluation sau freeze/authorization riêng |

**Đã chốt cho final:** 51-class mapping; eligibility ≥50 unique samples/≥3 train months; chronology, preprocessing và primary near-duplicate policy hiện tại.

**PROPOSED:** New Q/QD/QU/QDU dùng cùng March S*, cùng full-February temperatures và cùng February–March access. Legacy February Q là reference riêng, không làm đối chứng duy nhất của QDU. Kết quả new March-fitted weights trên March chỉ là fit/development diagnostic.

Không dùng Future labels, embeddings, distributions, unlabeled batches hoặc summaries để thiết kế rule, coefficients hay fallback. Future chronological filtering theo policy đã khóa không được cập nhật weights. Post-test drift diagnostics là phân tích riêng.

**UNKNOWN:** thời điểm nhãn family có sẵn. Observation timestamp không phải label-availability time hay PE compilation time. Không xác minh được thì chỉ claim retrospective temporal classification với giả định lịch sử có nhãn, không gọi prospective deployment.

## 4. Historical Forward Episodes: audit trước training

### 4.1 Lịch đề xuất cố định

| Episode | Encoder train | A: calibration/reference | B: Q/U, drift A→B | C: outcome |
| --- | --- | --- | --- | --- |
| E1 | Aug–Oct 2019 | Nov 2019 | Dec 2019 | Jan 2020 |
| E2 | Aug–Nov 2019 | Dec 2019 | Jan 2020 | Feb 2020 |
| Final | Aug 2019–Jan 2020 | Feb 2020 | Mar 2020 | Apr–Sep, frozen weights |

**PROPOSED:** E1 chọn coefficients; E2 kiểm tra held-forward, không retune. Prefix screening dùng seed42, tối đa hai episodes. Chúng overlap và retrospective, không phải hai independent temporal replications. Encoder hiện tại train đến January không được thay prefix encoder.

### 4.2 Audit bắt buộc, hiện UNKNOWN

1. Chọn K_e từ prefix train theo cùng ≥50 unique samples/≥3 months; chưa biết K_e. Không ép 51 classes và không đổi final mapping. Mỗi episode dùng một mapping K_e chung cho ba views.
2. Chứng minh input universe đầy đủ. Manifest hiện tại đã lọc theo final cohort: chỉ reuse nếu chứng minh prefix-eligible families là subset của final-eligible theo cùng rule và đủ mọi sample cần thiết. Nếu không, yêu cầu historical-only export có provenance; không đọc full raw CSV có Future chỉ để tiện.
3. Tái tạo retention theo chronology episode; không reuse nguyên final validation retention cho Nov/Dec/Jan. Train cutoff khác. Later-excluded samples vẫn là reference/bridge theo policy; C không đi vào train.
4. Audit saved historical SHA/TLSH inventories: coverage/TNULL, group sizes, mixed labels, shared train–A/B/C, A–B/B–C, per-family concentration. Labels không quyết định similarity.
5. Signals chỉ dùng graph A/B đã định trước; C không làm bridge để đổi groups/weights trước scoring C. Sau freeze episode có thể audit dependence với C, không refit.
6. Candidate S_e: ≥20 retained samples và ≥5 operational joint A/B groups trong MỖI tháng. Report tất cả families/coverage/failure reasons; không hạ ngưỡng hoặc loại family do loss/drift xấu.
7. Kiểm tra sample/content overlaps bị retention policy cấm và remaining near-duplicate dependence ở 10/20. Shared components không tự chứng minh label leakage, nhưng hạn chế independence.
8. RQ2 outcome dùng common support S_eval = S_e giao các classes có true support ở C. Rule khóa trước outcome; chỉ để scoring các per-class forecasts đã lưu, không refit/select signals. Report lost support và full K_e metrics; nếu S_eval rỗng thì RQ2 NOT ESTIMABLE. Không dùng support C để chọn S_e/grid.
9. Kiểm tra model/data APIs có hard-code 51/final dates không; kiểm tra label-availability evidence. Chưa sửa code ở bước tài liệu.

**PASS kỹ thuật đề xuất:** input/provenance/chronology đúng; K_e≥2, S_e≥2; A/B/C không rỗng; support requirements giữ nguyên; cùng IDs/order cho ba views. Đây chỉ là computational feasibility, không đủ power cho RQ2. Group concentration và label limits cần review riêng trước training.

**UNKNOWN tại V1:** counts/groups E1/E2; audit chưa chạy. Nếu fail, STOP: không tự đổi cutoff, thêm episode, hạ support hoặc reuse January encoder. Owner quyết định redesign hoặc giảm claim.

## 5. Q: historical Quality

**PROPOSED, phù hợp framework §6–7:**

    ell_iv = -log p_v(y_i | x_i)
    L_v = (1/|S|) sum_c [(1/n_c) sum_{i in B_c} ell_iv]
    Q_v = -L_v

S chung cho các arms/views. Probabilities vẫn trên K_e (episode) hoặc 51 (final), không renormalize trên S. Natural logs, đơn vị nats. Report standard NLL toàn B, NLL trên S và per-class losses.

Temperature fit trên A bằng standard NLL, bounds hiện có [0.05,20]; B không fit temperature. Final seed42 giữ nguyên temperatures; seeds mới fit February T riêng trong artifacts mới.

**VERIFIED:** [quality_fusion.py](C:/Users/hoang/Desktop/TRAMIF/scripts/quality_fusion.py) có February OOF class-balanced loss/softmax(-L); [calibrate_sbsmi.py](C:/Users/hoang/Desktop/TRAMIF/scripts/calibrate_sbsmi.py) có solver/bounds. New March/episode Q chưa tính.

## 6. D: class-conditional MMD và null feasibility

### 6.1 Frozen features, normalization, kernel — PROPOSED

- SBSMI: head_norm sau flatten, trước dropout/classifier, dimension 2048. Raw-byte/entropy: CompactViewNet.forward_features, dimension 512.
- Eval mode, frozen BatchNorm, không augmentation/running-stat update. Một encoder duy nhất cho A/B mỗi episode; không so latent coordinates của hai mạng retrain độc lập.
- TRAIN reference: tối đa 128 samples/family K_e, deterministic SHA-256 ranking với namespace/seed42; lấy tất cả nếu ít hơn; cùng source IDs giữa views, lưu manifest.
- Fit coordinate mean/variance bằng equal-family mixture: mỗi family tổng mass bằng nhau, mỗi file trong family cùng mass. Z-score A/B bằng TRAIN parameters.
- Mask coordinates có TRAIN standard deviation ≤1e-8. Đây là numerical tolerance đề xuất, không phải scientific threshold. Stop nếu không còn dimension/non-finite; report masked dimensions, rank và conditioning.
- Với p_v dimensions còn lại: rho(z,z') = ||z-z'||²/p_v; RBF k = exp[-rho/(2 h_v)].
- h_v là weighted median rho trên TRAIN within-family unordered non-self pairs, tối đa 128 pairs/family chọn deterministic; equal total pair mass mỗi family. Không lọc zero-distance pairs để làm median đẹp. h_v≤0/non-finite: STOP, không fit từ March.

Z-score/dimension scaling/bandwidth riêng không bảo đảm raw MMD comparable giữa encoders.

### 6.2 Sample matching và statistic — PROPOSED

Final S* candidate là 25 families; đây là SIGNAL support, evaluation vẫn full cohort. Mỗi class c chọn n_c = min(n_Ac,n_Bc,128) samples mỗi tháng bằng SHA ranking, cùng IDs mọi view, không chọn theo embeddings/loss/confidence.

Support ≥20/≥5 groups xét trước cap; report groups sau cap. Nếu cap làm mất ≥5 groups/class/month, STOP review sampler, không thử seed khác đến khi đạt.

    D_vc = mean_{i,j in A_c} k(z_i,z_j)
         + mean_{i,j in B_c} k(z_i,z_j)
         - 2 mean_{i in A_c,j in B_c} k(z_i,z_j)
    D_v,cond = mean_{c in S} D_vc

Đây là biased MMD² gồm diagonal terms, không âm theo lý thuyết. Chỉ clip negative roundoff trong tolerance float64 đã kiểm chứng, không che lỗi lớn. Marginal MMD là mixture diagnostic, không thay conditional primary.

Report per-family scores, support/groups, file-size distributions. Không coi entropy là ground-truth packing label; packing covariates chỉ dùng khi provenance hợp lệ.

### 6.3 Null-standardization: scientific gate chưa đạt

Framework yêu cầu:

    Dtilde_v = max(0, (D_v,cond - mu_v,null)/(sigma_v,null + epsilon))

Null hypothesis: không có thay đổi conditional embedding distribution giữa windows, với support/sample size/dependence được mô hình hóa đúng. Standardization không tự tạo p-value.

Matched null phải chứng minh:

1. Sampling unit, joint null distribution và exchangeability assumptions rõ ràng.
2. Whole-group/multi-label integrity, matched class counts/cap và aggregation đúng observed statistic.
3. Group xuất hiện ở hai tháng được xử lý đúng: không hoán đổi month labels từng file hoặc ép multi-month group thành một tháng.
4. Group-disjoint same-period null không được mặc định đại diện observed statistic có shared groups.
5. Đủ allocation cho toàn S; không âm thầm giảm support; null variance finite/stable.
6. Synthetic controls cho no-shift/known-shift, unequal groups, shared incidence, scale/dimension changes; không chọn estimator bằng outcome loss.

**VERIFIED:** Phase 1 chưa xác định null hợp lệ. Với agent February, hai group-disjoint sides chỉ có tối đa 9 samples mỗi phía; observed February–March có shared components. Match số file mà sai dependence không giải quyết validity.

**UNKNOWN:** null generator phù hợp, variance, draws cần đủ precision và epsilon theo scale. Không dùng epsilon tùy tiện cứu near-zero variance.

**Giới hạn hữu hạn:** một null-feasibility review có allocation/dependence evidence trước D implementation. Nếu không có null hợp lệ cho toàn S, ghi FAIL/UNAVAILABLE, dừng standardized D. Không tự chạy permutation/wild bootstrap chỉ vì thư viện có sẵn.

### 6.4 Fallback duy nhất để xin phê duyệt

**PROPOSED fallback, chưa kích hoạt:** descriptive exploratory score d_desc = D_cond/2. RBF biased MMD² thuộc [0,2], nên d_desc thuộc [0,1]. Đây chỉ là **numerical bound, KHÔNG phải null calibration hoặc standardized drift**.

Hệ quả: RQ1 mô tả dưới kernel/encoder cụ thể; cross-view ranking cần sensitivity; matched-null requirement chưa hoàn thành. D-weighting là hypothesis thiết kế mới. Owner phải duyệt scale/grid và giảm claim TRƯỚC E1 selection; không chuyển beta từ Dtilde sang d_desc im lặng.

Nếu fallback không được duyệt, Q/D/U scope giữ ở mức được phép, QD/QDU ghi NOT ESTIMABLE; không điền D=0 và gọi full reliability thành công.

Predeclared sensitivities sau approval: h/2 và 2h; cap64 so với primary128; file-weighted so với group-balanced diagnostic. Không chọn cấu hình thắng trên Future. Sensitivity lớn phải được công bố.

## 7. U: uncertainty của historical Quality

### 7.1 Estimator — PROPOSED

U là uncertainty của L_v trên B conditional vào model, T, labels, support và operational groups. Không phải predictive entropy, disagreement, epistemic neural uncertainty, training-seed uncertainty hoặc temporal uncertainty.

Positive group-weight bootstrap: 2.000 replicates, PCG64 seed42. Group = joint A/B content-SHA equality hoặc standard TLSH distance≤10 component; chỉ retained B samples tham gia loss. Mỗi replicate b sinh a_bg ~ Exponential(rate=1) cho active B group g; CÙNG weight cho mọi view/class/sample thuộc group.

    L_v^(b) = (1/|S|) sum_c [
        sum_{i in B_c} a_b,g(i) ell_iv / sum_{i in B_c} a_b,g(i)
    ]
    U_v = sample_SD_b(L_v^(b))

Không chia SD cho sqrt(2000). Point L_v vẫn là loss gốc, không thay bằng bootstrap mean. Positive weights giữ observed support, không tạo support mới/independence.

### 7.2 Verification trước weighting

- Known-answer tests: equal losses→U=0; whole-group/shared weights; invariance khi nhân tất cả weights cùng hằng số; reproducibility; finite denominators.
- Simulations có independent clusters và dependent/unequal-size groups; không coi finite SD là bằng chứng frequentist coverage.
- Report original/replicate mean, SD, quantiles, group concentration, per-family influence.
- Seed43 diagnostic bổ sung, không chọn seed đẹp nhất. Review nếu SD khác seed42 vượt max(0.005 nat, 10% SD lớn hơn). Đây là engineering trigger đề xuất, không phải validity theorem.
- Leave-one-group-out diagnostic, không loại group khỏi main estimate. Nếu mất class trong S thì ghi undefined, không đổi denominator. Nếu mất support hoặc induced weights thay đổi >0,05 tuyệt đối ở bất kỳ view trong grid đã khóa, STOP review dominance. Ngưỡng 0,05 là operational tolerance cần duyệt, không chứng minh independence.
- Non-finite replicate: lưu failure và STOP; không discard/retry chọn lọc.

Chỉ dùng U vào weights khi estimator/stability checks đạt và owner chấp nhận conditional interpretation/influence report. Nếu fail: U diagnostic only, QU/QDU dừng; không tự thay bằng entropy hoặc zero.

Bootstrap U khác bootstrap uncertainty của method differences ở §11.

## 8. Fusion và historical coefficient selection

### 8.1 Công thức — PROPOSED

    r_v = -a L_v - beta d_v - gamma U_v
    w_v = exp(r_v-max(r)) / sum_u exp(r_u-max(r))
    p_fusion(y|x) = sum_v w_v p_v(y|x)

Primary d là Dtilde nếu null gate đạt; descriptive fallback có version/approval riêng.

| Arm | Score |
| --- | --- |
| Uniform | 0 |
| New Q | −L_v |
| Q+D | −L_v − beta* d_v |
| Q+U | −L_v − gamma* U_v |
| QDU | −L_v − beta* d_v − gamma* U_v |

a=1 giữ quality scale và giảm hyperparameters; beta,gamma≥0. QD/QU dùng cùng beta*/gamma* của QDU để isolate signal removal. Tắt D/U phải tái tạo New Q; không phải Legacy Q vì history/support khác.

Negative penalty là hypothesis, không chứng minh D/U cao đồng nghĩa reliability thấp. U có đơn vị nats; beta chuyển dimensionless D vào log-score scale. Không tự clip scores/ép minimum weights. Non-finite normalization là lỗi phải dừng; finite nhưng extreme weights là diagnostic/possible negative result, không lý do retune trên Future.

### 8.2 Một cách chọn coefficients hữu hạn — PROPOSED

Không chốt ln(2)/1 chỉ vì tiện. Grid đề xuất:

    a=1; beta in {0,0.5,1}; gamma in {0,0.5,1}

9 candidates; zero cho phép kết luận signal không được chọn. Penalty 0,5/1 tương ứng nhân relative weight odds bởi exp(−0,5)/exp(−1) khi signal tăng một đơn vị. Đây là bounded prior grid, không optimal scale đã chứng minh. Duyệt grid SAU measurement gate và TRƯỚC đọc E1 outcome losses để selection; không nới grid sau outcome.

RQ2 forecaster đề xuất:

    predicted_next_loss_v,c = L_v,c,B + beta d_v,A→B + gamma U_v,B

L_v,c,B là mean NLL của class c trong B; d_v và U_v vẫn là GLOBAL signals đã tính trên S_e trước C. Lưu per-class forecasts cho mọi c trong S_e trước evaluation. Q-only là persistence forecast L_v,c,B. Khi C có labels, tính absolute residual với mean NLL của cùng class trong C; average đều trên S_eval rồi trên ba views. Không recompute d/U, Q fusion weights hoặc support từ C.

Chọn beta*,gamma* bằng minimum MAE này trên E1. Không fit thêm intercept/slope/neural predictor; family/view residuals phụ thuộc nhau, không coi chúng là nhiều independent temporal observations. Tie trong 1e-12: ít nonzero terms hơn, rồi beta+gamma nhỏ hơn, rồi beta nhỏ hơn.

Freeze trước E2. Report E2 prediction MAE/residuals của Q/QD/QU/QDU, fusion metrics và coverage. Không retune theo E2 hoặc March. Forecaster có nonnegative penalties nên chỉ dự báo loss tăng; phải báo cáo thất bại khi quality cải thiện.

**Giới hạn:** ba correlated views ở E1 và hai overlapping episodes không đủ fit/validate general error-prediction mechanism. Đây là screening định lượng held-forward, yếu hơn một fitted predictor qua nhiều independent episodes theo §12.2. Muốn claim mạnh cần thêm episodes/corpus qua approval mới.

File-size/composition control cho RQ2: primary residual average giữ equal-family weights; báo cáo sensitivity theo file-size quartile cutoffs fit từ prefix TRAIN, cùng cutoffs cho B/C. Khác biệt do thiếu class/bucket phải được công bố, không loại để tăng kết quả. Đây là stratified diagnostic, chưa phải multivariable adjustment cho packing/collection confounding; nếu thiếu packing provenance, ghi UNKNOWN và giới hạn claim theo §12.2.

Nếu winner có beta/gamma=0, giữ nguyên và ghi arm degeneracy/negative evidence; không ép coefficients dương để “có QDU”. Nếu E2 không cải thiện, báo cáo âm, không chọn runner-up. Việc estimator UNAVAILABLE khác với selection hợp lệ chọn coefficient zero.

## 9. Baselines và controls theo claim

Primary comparison set dùng cùng calibrated per-seed predictions, IDs/order, retention và historical budget:

1. SBSMI calibrated; 2. raw-byte calibrated; 3. entropy calibrated;
4. Uniform; 5. New Q; 6. QD; 7. QU; 8. QDU; 9. learned static fusion.

Legacy February Q chỉ là historical reference seed42, không đánh tráo với New Q.

**Learned static fusion PROPOSED:** minimize class-balanced mixture NLL trên B/S:

    min_{w>=0,sum w=1} mean_c mean_{i in B_c}
        -log(sum_v w_v p_v(y_i|x_i))

Hai degrees of freedom, không CNN training; cùng access với QDU. Numerical solver/tie-breaking deterministic phải được test/freeze trước evaluation. Không thêm regularization search hoặc dùng C/Future fit weights.

| Claim | Control cần thiết |
| --- | --- |
| D/U bổ sung ngoài Q | QD−NewQ, QU−NewQ, QDU−NewQ; chung history/capacity |
| Tốt hơn học weights đơn giản | QDU−learned static fusion |
| Signal semantics hữu ích | Historical score-shuffle giữ Q, permutations D/U giữa views; diagnostic, không permutation p-value vì views không exchangeable |
| Multi-view hơn capacity/cost | Capacity-matched single-view hoặc same-view ensemble và measured-latency control; approve architecture/budget trước training, nếu thiếu thì không claim efficiency advantage |
| Image geometry/SBSMI adjacency quan trọng | MLP trên cùng SBSMI, state-permutation/quantization controls trước khi claim |

Framework §11.3 còn có 1D-byte model, PE-feature GBT, content gating, drift-only. V1 đề xuất drift-only r=−beta*d là diagnostic nếu D hợp lệ. MLP/1D/GBT/gating chưa thuộc batch chính: **scope reduction cần duyệt**, không tuyên bố hoàn thành tất cả framework baselines hoặc SOTA. Strong cost-matched claim cần capacity controls, không được bỏ âm thầm.

## 10. Seeds, data validity, costs, reproducibility

**PROPOSED final seeds:** 42,43,44,45,46. Legacy seed42 không đổi; seeds mới cùng architecture/training config/duration, February T và March signals riêng; beta*/gamma* chung, không optimize mỗi seed. Prefix screening một seed không trở thành multi-seed mechanism validation.

**VERIFIED:** train scripts hiện hard-code42, cần authorized code change trước multi-seed. Không chọn best seed hoặc loại seed performance thấp. Freeze failure/resume policy trước run.

Primary retention vẫn exact content hoặc TLSH distance0, chronological later-copy exclusion; train giữ non-identical near-duplicates theo existing policy. Retention sensitivity10/20 khác operational grouping10. TLSH0 không đồng nghĩa identical bytes. Labels không quyết định similarity.

Sensitivity performance dùng frozen primary models/T/weights trên alternative retained sets để isolate evaluation-selection effect; không âm thầm refit. Đây chưa phải end-to-end retraining sensitivity. Report coverage, exclusions, unknown proportions và TNULL.

Leakage checks: prefix eligibility, content/sample IDs, class order, checkpoint cutoff, temperature fit scope, group construction cutoff, normalization/bandwidth reference và hashes. Ghi trung thực policy đã được phát triển sau initial SBSMI validation.

Costs: total/branch parameters; CPU/GPU settings; read/hash/transforms/inference/fusion và end-to-end latency; peak RAM/VRAM; file-size buckets. Không suy total latency từ CNN FLOPs hoặc 64×64 input.

Artifacts: immutable runs; source commit/dirty diff hash; environment; full-precision configs/weights; manifests; generator/version/seeds; predictions; failures; draws và hashes. Git không backup data/. Cần backup/restore check trước final experiment. Chỉ release permitted manifests/code/results, không tự phân phối malware binaries.

## 11. Metrics, uncertainty và multiplicity

### 11.1 Giữ HAI primary outcomes theo framework §11.4

Với seed s, month m: F_sm là macro-F1 trên eligible classes có true support trong ORIGINAL retained month; publish monthly support sets. Classifier vẫn 51 outputs; predictions vào absent labels vẫn tạo false negatives cho true families.

    Mean_s = (1/6) sum_m F_sm
    Worst_s = min_m F_sm

Point outcomes chính: average_s Mean_s và average_s Worst_s trên năm seeds. Report từng month/seed; min_m average_s F_sm là diagnostic khác, không hoán đổi định nghĩa worst-month.

Secondary: pooled fixed-51 F1, Accuracy, balanced accuracy, per-class recall (undefined khi không true support), NLL, multiclass Brier sum trên51 classes, top-label ECE15 theo implementation hiện có; 1/3/6-month cumulative horizons và coverage.

Practical target **PROPOSED:** mean-monthly gain ≥2 percentage points và worst-month gain >0 so với New Q để gọi cải thiện thực tiễn trên cả hai outcomes. 2pp là mục tiêu dự án từ ví dụ framework, không expected result/theorem. Mean tăng nhưng worst giảm là trade-off, không dominance.

### 11.2 Confirmatory contrasts — PROPOSED

New Q−Uniform; QD−New Q; QU−New Q; QDU−New Q; QDU−learned static fusion.

Hai outcomes × năm contrasts = family 10 comparisons. Không chuyển contrast thắng thành primary. Single-view comparisons, extra controls và secondary metrics vẫn báo cáo, nhưng không thêm simultaneous superiority claim nếu chưa bổ sung correction trước test.

### 11.3 Statistical specification — PROPOSED, cần simulation gate

- 20.000 paired replicates, PCG64 seed42; predictions/T/weights từng model frozen.
- Joint operational groups xuyên sáu test months theo content equality/TLSH≤10. PROPOSED graph vertices là toàn bộ original future known-cohort records của sáu tháng, gồm cả primary-excluded bridge records; chỉ retained samples được scoring. Không thêm train/February/March bridge vertices trong primary resampling graph. Điều này có thể bỏ sót dependence qua lịch sử; phải nêu limitation, không coi các components là độc lập đã chứng minh. Đây là resampling sau authorization, không đổi retention/weights.
- Mỗi replicate chọn 5 seed indices và G whole groups có hoàn lại. Cùng seed/group draw cho MỌI method; group multiplicities chung xuyên months/seeds. Không tách group theo label/month; không coi seed×sample là independent rows.
- Tái tính monthly confusion matrices; giữ ORIGINAL monthly support denominator. Missing class trong replicate không bị loại; F1=0 nếu denominator zero theo convention đã ghi. Recall không true support đánh dấu undefined. Không discard missing-class replicates.
- Tính Mean/Worst từng selected seed rồi average; paired subtract. Point estimates lấy dữ liệu gốc, không bootstrap means.
- Lưu sample counts từng month/draw. Empty month hoặc non-finite metric: lưu failure và STOP review, không retry/drop hoặc đổi denominator sáu tháng.
- Report pointwise95% percentile CIs. Family10 primary comparisons: thêm Bonferroni99,5% percentile intervals (quantiles0,0025/0,9975) từng comparison. Không gọi pointwise95% simultaneous.
- Bonferroni chỉ hợp lệ nếu marginal intervals có coverage phù hợp. Percentile bootstrap là xấp xỉ: simulation phải xét unequal groups, rare classes, min-of-months và correlated seeds. Nếu coverage/assumptions không đủ cơ sở, không đưa confirmatory significance claim; report conditional uncertainty và xin điều chỉnh estimator.

Chỉ năm seeds/sáu months; TLSH groups không chứng minh independence, seeds dùng chung train. Đây không bao phủ mọi training/calibration/selection uncertainty. Report seed spread và fixed-model group intervals riêng, không gọi là guarantee tổng quát.

Family/temporal-block sensitivity theo framework, **PROPOSED diagnostics:** leave-one-family-out contrasts với thay đổi support công khai; resampling ba contiguous two-month blocks từ frozen prediction summaries. Đây là sensitivity với aggregation/temporal composition, không estimator chính hoặc bằng chứng generalize sang năm khác. Không dùng chúng chọn lại method.

## 12. Decision Gates

“Đã chốt” không đồng nghĩa cho phép training/Future.

| Gate | Trạng thái | Điều kiện đóng gate | Khi không đạt |
| --- | --- | --- | --- |
| G0 Source/legacy protection | VERIFIED | HEAD đúng;24 checksums khớp;legacy preserved | Mismatch→STOP, không thay expected hash |
| G1 RQ/timeline/mapping/policy | Đã chốt cho final | Framework/AGENTS; giữ chronology lịch sử | Không tự đổi cohort/split/threshold |
| G2 Episode feasibility | UNKNOWN | §4 audit input/K_e/support/groups/labels | Owner chọn redesign/giảm claim, chưa train |
| G3 Embedding/MMD | PROPOSED | Approve §6; alignment/train-only parameters/synthetic checks | Không tune từ outcome để chữa estimator |
| G4 Matched null | CHƯA ĐẠT | Dependence/exchangeability/allocation argument và simulation plan | Standardized D unavailable; xin fallback riêng |
| G5 U estimator | PROPOSED, validity UNKNOWN | Known answers, common draws, MC/influence review | U diagnostic; QU/QDU dừng |
| G6 Grid/RQ2 scope | PROPOSED | Scale rõ; duyệt grid/selection/tie-break TRƯỚC E1 selection | Không tự cố định ln2/1 hoặc search March/Future |
| G7 Seeds/baselines/cost | PROPOSED, cần resource approval | Five seeds, matched access, solver/controls/budget frozen | Thu hẹp efficiency/SOTA claim, không giấu missing baseline |
| G8 Statistics | Hai primary outcomes giữ nguyên; estimator PROPOSED | Simulation/denominator/correction/sensitivity đã duyệt | Old March CI không thay Future protocol |
| G9 Final freeze | CHƯA ĐƯỢC PHÉP | Gates resolved hoặc approved departures; manifests/config/weights/backup | Không truy cập Future |
| G10 Future execution | Approval riêng | Frozen package, six-month reporting đầy đủ | Không retune; breaches/failures công khai |

## 13. Thay đổi so với framework và approval cần có

| Nội dung | V1 và lý do | Hệ quả/approval |
| --- | --- | --- |
| Hai mean/worst outcomes (§11.4) | Giữ cả hai; không hạ worst xuống secondary như đề xuất hội thoại trước | Giữ requirement; averaging/correction details cần duyệt |
| Standardized D (§5–6) | Primary spec giữ, hiện blocked; descriptive fallback tách tên/version | Fallback cần owner approval/giảm claim; null requirement vẫn chưa hoàn tất |
| Quality SE (§6) | Positive group-weight SD thay ordinary resampling để giữ observed support | Estimator extension cần simulation; không mặc định coverage đúng |
| Coefficient grid (§7) | a=1, tiny beta/gamma grid; E1 chọn E2 kiểm tra | Giảm flexibility; grid và scale cần duyệt, không optimality claim |
| RQ2 fitted predictor/adjustment (§12.2) | Fixed-form per-class forecast với global D/U, hai episodes; equal-family/size-stratified diagnostics | Screening yếu hơn general fitted predictor và full covariate adjustment; claim mạnh cần thêm episodes/packing evidence |
| March selection (§7) | March dùng final historical signals, không tune coefficients mới | Giảm reuse; E1/E2 cũng retrospective. Timeline selection cần duyệt |
| Five seeds (§11.4) | Giữ final five, prefix một seed | Chưa đạt cho đến khi chạy; mechanism seed variability chưa xác nhận |
| Broad baselines (§11.3)/controls (§13) | Learned fusion chính; baselines rộng/capacity cần scope gate | Bỏ là approved scope reduction, không claim hoàn thành toàn framework/SOTA |
| Family/temporal statistics (§11.4) | Group×seed bootstrap chính; family/block diagnostics tách estimands | Cần duyệt assumptions; không coi groups/seeds/months independent mặc định |
| Training-specified duplicates (§11.2) | Giữ policy thực tế, khai báo đã phát triển sau initial validation | Không phục hồi preregistration bằng tài liệu mới; legacy vẫn development |
| Contrastive/adaptation/rejection | Ngoài V1 | Optional theo framework; không claim các khả năng này |

## 14. Implementation hữu hạn và acceptance

Tài liệu này không chạy các bước dưới đây.

| Bước | Dependencies và outputs | Acceptance / stop |
| --- | --- | --- |
| **1. Historical episode feasibility audit** | Owner duyệt §4 + historical allowlist; report/count/group/support manifests/hashes; không CNN | K_e/S_e/chronology/provenance rõ E1/E2; UNKNOWN công khai; fail không tự bước2 |
| **2. Measurement/null/U gates** | Bước1; feature specification, null decision, U tests/simulation plan | G3–G6 duyệt trước signals/selection; null fail thì quyết định fallback một lần |
| **3. Prefix experiments** | Resource approval; tối đa6 CNN fits seed42; E1 selection/E2 forward report | C không vào signals; giữ negative results; không nới grid sau E2 |
| **4. Final historical multi-seed package** | G7; thêm12 CNN fits, per-seed T/signals/arms, learned fusion/cost controls | Five seeds hoặc reported failure; không best-seed selection; legacy preserved |
| **5. Statistics validation + freeze** | G8; simulation/tests, immutable plan/provenance/backup | Hai outcomes, denominator/uncertainty executable; xin G10 |
| **6. Authorized Future evaluation/report** | Explicit G10; ngoài authorization hiện tại | Sáu tháng/cùng sets/all arms/seeds/coverage/negative findings; không retune |

Planning estimate: logs cũ ghi khoảng130 phút epochs cho một bộ ba CNN; thêm bốn final seeds có thể ~8,7 GPU giờ riêng epochs nếu tốc độ tương tự. Prefix/extraction/controls cần budget riêng sau audit. Estimate không là authorization.

**Definition of Done experiment:** scope/provenance đúng; tests/integration evidence phù hợp; đầy đủ outputs/failures/negative findings. PASS không yêu cầu method thắng.

**Definition of Done nghiên cứu:** RQ1–RQ3 có measured evidence hoặc explicit NOT ESTIMABLE/unsupported finding; mọi approved arm/metric được báo cáo; final protocol không test-driven tuning; reproducibility/backup đầy đủ; departures minh bạch. QDU không hơn Q vẫn là kết quả hoàn thành hợp lệ, không tự mở phase để cứu hypothesis.

## 15. Nguồn phương pháp

- [Gretton et al., JMLR 2012 — A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html): cơ sở MMD, không chứng minh drift dự báo classification error.
- [Chwialkowski et al., NeurIPS 2014 — A Wild Bootstrap for Degenerate Kernel Tests](https://proceedings.neurips.cc/paper_files/paper/2014/hash/4e382cb49370f64415df2672b19fb1f2-Abstract.html): dependence cần assumptions/resampling thích hợp; không tự áp dụng cho TLSH components.
- [Rubin, Annals of Statistics 1981 — The Bayesian Bootstrap](https://people.eecs.berkeley.edu/~jordan/sail/readings/rubin.pdf): cơ sở random positive weights; group/class-balanced adaptation ở V1 cần verification riêng.

**Kết luận:** V1 có một primary path và các stop gates hữu hạn. Các scientific choices chưa đủ bằng chứng vẫn chưa được phê duyệt; việc tạo tài liệu không biến chúng thành methodology đã validated.
