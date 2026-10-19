# G4 — Focused matched-null feasibility audit

Ngày **15/10/2026, Asia/Saigon**. Chỉ G4 được phê duyệt. Source HEAD kế thừa G2: `d8f0c676f604a744f06800ab63c69dc6fe8beea3`.

## 1. Kết luận và stop decision

“Allocation” phải chỉ rõ đang phân bổ cho thiết kế nào. Có đủ samples để tạo hai tập observed A/B không đồng nghĩa tạo được một **non-degenerate matched-null distribution**.

| Trường hợp | Allocation cho matched null đầy đủ theo V1 | Same-period, hai phía group-disjoint với counts V1 | Statistical validity |
| --- | --- | --- | --- |
| E1 — Nov–Dec2019, S=17 | **UNKNOWN** | **INFEASIBLE**, dù lấy November hay December | **NOT ESTABLISHED** |
| E2 — Dec2019–Jan2020, S=25 | **UNKNOWN** | **INFEASIBLE**, dù lấy December hay January | **NOT ESTABLISHED** |
| Final — Feb–Mar2020, S*=25 | **UNKNOWN** | **INFEASIBLE**, dù lấy February hay March | **NOT ESTABLISHED** |

UNKNOWN ở cột đầu: chưa có một null generator bảo toàn dependence/class budgets và có randomization law được biện minh. Không phải kết luận mọi null toán học đều bất khả thi. INFEASIBLE chỉ áp dụng cho thiết kế cụ thể ở cột tiếp theo, có chứng minh đếm tại §4. Không dùng identity allocation làm bằng chứng FEASIBLE cho một phân phối null có variance hữu ích.

**G4 chưa đạt; standardized D hiện chưa sẵn sàng. DỪNG.** Không chọn null mới, không mở implementation/simulation/embeddings, không thay support hoặc chuyển sang descriptive MMD. G2 count feasibility vẫn giữ nguyên; nó không đóng statistical gate này.

## 2. Evidence inspection và allowlist tối thiểu

Đọc V1 §4/6/12, báo cáo G2 §3/5/6/8, BALANCED Phase1 phần joint graph/support/null; progress/decision headers để giữ checkpoint. Không khảo sát lại repository.

Hai immutable roots:

- **H:** `C:/Users/hoang/Desktop/TRAMIF/data/derived/historical_episode_feasibility/20261014_233013_930021/`.
- **F:** `C:/Users/hoang/Desktop/TRAMIF/data/derived/balanced_phase1_audit/20261013_233825_921873/`.

Data allowlist của phép xác minh lần này: `H/checksums.json`, `H/report.json`, `H/{E1,E2}/{support.csv,AB_10_members.csv}`, `F/checksums.json`, `F/report.json`, `F/class_support.csv`, `F/members.csv`; thêm G2 verification report đã có để kiểm tra provenance chain. Không mở edges/pair vertices, C memberships, inventory/raw metadata, binary, image, checkpoint, predictions hoặc losses.

**VERIFIED:** 8/8 checksums của hai reports và sáu support/membership files khớp checksum manifests. Hai report hashes còn đối chiếu với checkpoint đã biết:

- [H/report.json](C:/Users/hoang/Desktop/TRAMIF/data/derived/historical_episode_feasibility/20261014_233013_930021/report.json): `8f232f09d7ddcec2e5b71ba68d0627c1a96c2359631c858e67f329c8832f4f1e`.
- [F/report.json](C:/Users/hoang/Desktop/TRAMIF/data/derived/balanced_phase1_audit/20261013_233825_921873/report.json): `c86639b6eeae08e80cbe83ae858d6ae2e099c7c5f0f3d6a0e7fcbb8033a6668f`.

G2 đã xác minh18 graph partitions; **không chạy lại**. Lần này chỉ đếm membership IDs đã lưu, kiểm tra unique IDs/dates và đối chiếu lại mọi per-family sample/group count với support CSV. Không tái dựng graph, không tính TLSH distances. Four targeted assertions kiểm tra công thức bound tại §4 (empty, all-in-one-group, uneven và balanced cases) đều PASS. Đây là assertions của phép kiểm tra metadata ngắn qua stdin, không phải một suite unit tests mới; không tạo production utility hoặc artifacts sao chép. Python3.12 có sẵn, stdlib, không cài dependencies.

Provenance giới hạn như G2: tin vào frozen historical exports và saved group memberships đã verified; không re-hash binaries. Không có April–September records trong allowlist.

## 3. Null hypothesis, unit và dependence cần giữ

Với view v, family c thuộc frozen S, encoder và train-reference transformation cố định, đặt Z_v là embedding đã chuẩn hóa. Null mục tiêu:

`H0: Law(Z_v | Y=c, month=A) = Law(Z_v | Y=c, month=B), cho mọi c trong S.`

E1: A=November/B=December; E2: A=December/B=January; Final: A=February/B=March. Không cần H0 về class prevalence vì statistic là class-conditional, nhưng vẫn cần mô hình đúng về cách quan sát/giữ mẫu. **H0 bằng nhau về phân phối marginal không kéo theo toàn bộ dependent observations exchangeable.** Exchangeability nghĩa là phép hoán đổi được chọn giữ nguyên joint probability law dưới H0, không chỉ giữ số lượng.

Sampling unit có thể xem xét là **joint A/B connected component** đã lưu, bao gồm vector contributions theo month × class. Nếu một component chứa nhiều family hoặc cả hai tháng, mọi thao tác phải áp dụng nhất quán cho toàn block; không chia theo class/month để dễ phân bổ. Excluded vertices giữ vai trò xác định component nhưng không trở lại statistic. Chọn samples trong S để đếm không tách/rebuild group IDs.

Nếu cùng resampling dùng cho ba views, giữ chung sample/block identities và cross-view association. Không suy tính bất biến của joint law từ ba marginal H0 riêng lẻ. Groups TLSH là operational units, chưa phải independent/exchangeable sampling units; chỉ biết graph không biết covariance của embeddings.

**VERIFIED từ toàn retained cohort, tái sử dụng reports:** E1/E2/Final có124/218/178 cross-month groups và42/57/41 multi-label groups. Đếm riêng contributions thuộc signal support nhưng giữ nguyên joint IDs cho thấy blocker không chỉ do classes ngoài S:

| Case | S | A/B samples trong S | Active groups có S members | Cross-month groups trong S | Samples trong các shared groups | Multi-label groups trong S |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| E1 | 17 | 1.789 / 3.130 | 2.352 | 114 | 1.290 | 35 |
| E2 | 25 | 3.503 / 3.659 | 3.790 | 203 | 2.049 | 47 |
| Final | 25 | 3.380 / 4.064 | 3.661 | 161 | 2.085 | 36 |

Đây là **pre-cap** counts. V1 định n_c=min(N_Ac,N_Bc,128), nhưng chưa có frozen capped selection manifest trong evidence đã kiểm tra. Không tự chọn namespace/seed/ranking để làm cap thuận lợi. Group incidence và ≥5 groups sau cap vẫn UNKNOWN. Không mặc định mọi shared group trước cap chắc chắn còn shared sau cap, cũng không mặc định cap xóa dependence. Các counterexamples đếm dưới đây không phụ thuộc vào cách chọn cap.

## 4. Minimal verification: một thiết kế đã có đủ bằng chứng để loại

Thiết kế xem xét: lấy một historical month làm nguồn null, chia thành hai phía **không lặp samples, group-disjoint**, mỗi phía có đúng n_c samples của mọi class c như observed V1. Không đổi statistic budget hoặc S.

Điều kiện cần: `N_tc >= 2*n_c`. Nếu group lớn nhất của family c trong tháng t có M_tc samples, sức chứa đều hai phía còn bị chặn bởi:

`capacity_tc <= min(floor(N_tc/2), N_tc - M_tc)`.

Lý do: hai phía dùng samples khác nhau; một group không thể bị chẻ giữa hai phía. Đây là **upper bound**, không phải bằng chứng tồn tại partition đạt bound. Multi-label constraints chỉ có thể làm việc phân bổ đồng thời khó hơn. Không chạy partition search vì counterexamples đã đủ:

| Case / family thuộc frozen S | N_A | N_B | n_c mỗi phía | Upper bound A | Upper bound B | Kết luận |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E1 / qqpass | 26 | 20 | 20 | 13 | 10 | Cả hai nguồn đều không đủ |
| E2 / delf | 28 | 20 | 20 | 14 | 10 | Cả hai nguồn đều không đủ |
| Final / ceeinject | 20 | 20 | 20 | 10 | 10 | Cả hai nguồn đều không đủ |
| Final / agent, diagnostic bổ sung | 21 | 47 | 21 | 9 | 22 | February không đủ; B bound22 không tự chứng minh joint feasibility |

Với agent February, M=12 nên phía đối diện còn tối đa9; đây cũng khớp capacity9 đã có trong Phase1. Không cần chạy lại thuật toán capacity. Nguồn chính xác: `H/E1/support.csv`, `H/E2/support.csv`, `F/class_support.csv`; largest-group counts đối chiếu từ immutable memberships.

**INFERRED bằng điều kiện cần:** same-period disjoint allocation toàn S, exact V1 budgets là INFEASIBLE cho cả ba cases. Kết luận không yêu cầu MMD, embeddings hoặc performance metrics. Giảm n_c/loại các family này, thêm replacement, gộp tháng khác hoặc cắt group là đổi thiết kế, không phải cách “sửa” audit được phép. Phase1 greedy witness chỉ bảo đảm một số counts nhỏ hơn, không cứu được exact budgets V1.

Ngay cả khi đủ counts, null có disjoint groups cũng không tự khớp dependence của observed A/B có shared components. Đây là blocker thống kê riêng, không được giải quyết bằng tăng số draws.

## 5. Statistical reasoning: vì sao chưa có null thay thế được biện minh

Các mục sau là kiểm tra assumptions, **không phải danh sách thuật toán đã chạy hoặc đề xuất thử tuần tự**.

| Construction | Assumptions / integrity | Kết luận hiện tại |
| --- | --- | --- |
| Random split / permutation từng file trong class | Cần file exchangeability; phá component và cross-class block dependence | Không có cơ sở áp dụng |
| Gán mỗi pooled component vào một tháng giả | Preserves block nhưng ép shared-month component thành one-month; thay incidence/dependence | Không khớp observed design V1 |
| Swap toàn vector A/B của từng component | Cần block-law symmetry dưới A↔B conditional vào class/incidence design; unequal class counts khiến swaps bị ràng buộc đồng thời | Allocation không suy ra từ group count; symmetry không suy ra từ H0; **UNKNOWN / NOT ESTABLISHED**, dừng trước search |
| Giữ shared blocks cố định, chỉ randomize exclusive blocks | Cần conditional exchangeability của phần còn lại; phải xác định conditional target và xử lý đóng góp shared blocks | Chưa có cơ sở joint law; không xem là tự động valid fallback |
| Bootstrap nguyên paired A/B components theo empirical law | Giữ incidence và pairing có thể được, nhưng bản thân resampling không áp đặt H0; có thể giữ cả observed drift | Sampling uncertainty không tự là null distribution |
| Wild bootstrap | Cần một theorem/estimator đúng với dependency process và statistic; không suy từ tên phương pháp | Không có evidence về dependence decay/stationarity hoặc cách áp dụng cho TLSH blocks; chưa triển khai |

Global A/B swap của toàn tập matched có thể giữ counts và cho cùng symmetric MMD; identity/global swap vì thế không cung cấp non-degenerate null calibration. Một phân bổ duy nhất không xác định sampling probabilities, mu_null hoặc sigma_null. Không chọn epsilon để cứu variance chưa biết.

**Cơ sở phương pháp:** bài MMD gốc đặt sampling problem với i.i.d. observations; không thể mượn nguyên calibration đó để chứng minh validity cho TLSH-dependent data. [Gretton et al., 2012, §2.1](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf).
Wild-bootstrap theory được viện dẫn có assumptions về stationary/weakly dependent processes và kernel statistics; metadata TLSH không xác nhận các assumptions đó. Đây là giới hạn khi áp dụng sang TRAMIF, không phải bác bỏ wild bootstrap nói chung. [Chwialkowski et al., 2014, §2–3](https://proceedings.neurips.cc/paper_files/paper/2014/file/4e382cb49370f64415df2672b19fb1f2-Paper.pdf).

**UNKNOWN trong cả ba cases:** covariance/dependence dưới H0; between-component independence; month-label exchangeability conditional vào graph được tạo từ nội dung; informative component sizes và chronological retention có ảnh hưởng gì tới null law; actual post-cap incidence; kernel/null variance/stability. Không thể xác minh các điều này chỉ bằng hashes và counts. Các months liên tiếp không được random assignment; prospective label availability cũng chưa được biết. Hai prefix episodes còn cần encoder được train đúng cutoff, nhưng training thêm tự nó không giải quyết null assumptions.

**INFERRED:** conditioning trên graph/class counts không tự hợp thức hóa permutation. Phải chứng minh phép biến đổi giữ joint law có điều kiện; grouping được xây từ nội dung nên không mặc định là covariate ngoại sinh. Multi-label groups khiến per-class allocations độc lập không hợp lệ cho aggregated statistic. Không kết luận các assumptions chắc chắn sai chỉ vì chưa kiểm chứng; kết luận đúng là **NOT ESTABLISHED**.

## 6. Blockers, candidate và bước đề xuất

Blockers chung: (1) same-period design thiếu counts ở frozen support; (2) shared/multi-label components đòi dependence-matched joint law chưa có; (3) không có exchangeability argument đủ mạnh; (4) post-cap membership chưa frozen/verified; (5) null variance và finite-sample calibration chưa xác định. Không thiếu sức tính toán hay thiếu số thuật toán đã thử.

**Không có candidate null được khuyến nghị triển khai trong scope hiện tại**, nên không mở synthetic validation để “tìm estimator thắng”. Nếu một candidate sau này được biện minh và phê duyệt, điều kiện validation cần chốt trước gồm: known no-shift và known-shift synthetic controls, shared/multi-label/unequal blocks, đúng frozen class budgets/cap và retained-incidence rule; kiểm tra null mean/variance, degeneracy và calibration dưới assumptions cũng như khi assumptions vi phạm. Simulation có thể bác bỏ candidate hoặc kiểm tra tính đúng dưới mô hình giả lập; không chứng minh real TLSH groups exchangeable. **Chưa chạy hoặc phê duyệt bất kỳ simulation nào.**

**Một bước tiếp theo đề xuất:** owner review quyết định scientific scope tại G4: ghi nhận standardized D chưa có cơ sở, rồi quyết định có phê duyệt departure/fallback mô tả theo V1 §6.4 với claim giảm tương ứng hay không. Đây là yêu cầu quyết định, không phải kích hoạt fallback. Không tự chọn D=0, thay support, sửa protocol hoặc chạy giai đoạn kế tiếp.

Research Protocol V1 và legacy artifacts giữ nguyên. Không có graph/pair recomputation, outcome predictions/losses, training/inference, embeddings, MMD/Q/D/U, Future access, commit/push. **Focused G4 audit hoàn thành; statistical gate NOT ESTABLISHED và batch đã dừng.**
