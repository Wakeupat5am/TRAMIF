# G2 — Historical Episode Feasibility Audit

Ngày: **15/10/2026, Asia/Saigon**. Source HEAD: `d8f0c676f604a744f06800ab63c69dc6fe8beea3`.
Phạm vi được owner phê duyệt: **G2 duy nhất**, theo §4 của Research Protocol V1.

## 1. Kết luận và phạm vi của PASS

| Episode | Kết luận G2 về metadata/count/group feasibility | Families từ prefix K_e | Signal support S_e | S_eval tại C | Training-ready / scientific validity |
| --- | --- | ---: | ---: | ---: | --- |
| E1 | **PASS** | 32 | 17 | 17 | **Chưa training-ready; scientific validity UNKNOWN** |
| E2 | **PASS** | 42 | 25 | 25 | **Chưa training-ready; scientific validity UNKNOWN** |

**VERIFIED:** đủ input lịch sử trong phạm vi cohort, hashes/provenance khớp, chronological boundaries đúng, K_e≥2, S_e≥2, train/A/B/C không rỗng; cùng canonical sample IDs/order và mapping cho ba views ở mức manifest/config. Không hạ bất kỳ support threshold nào.

PASS này là **computational feasibility của G2**, không phải chứng nhận independence, null validity, statistical power, label availability, chất lượng mô hình hoặc image-loader integration. Chưa mở/re-hash ảnh và chưa chạy loader/inference cho episode mới. Không đánh dấu các phần này là đã verified.

**Batch dừng ở đây.** Không train/inference, không tính Q/D/U, không đọc predictions/loss của outcome months, không tạo weights, không truy cập April–September 2020 và không commit/push. Protocol, final mapping, legacy code/checkpoints/artifacts không đổi.

## 2. Evidence và cách tái kiểm chứng

Run audit, tên directory theo UTC:

`C:/Users/hoang/Desktop/TRAMIF/data/derived/historical_episode_feasibility/20261014_233013_930021/`

- [Báo cáo máy đọc và source hashes](C:/Users/hoang/Desktop/TRAMIF/data/derived/historical_episode_feasibility/20261014_233013_930021/report.json).
- [Allowlist đầy đủ](C:/Users/hoang/Desktop/TRAMIF/data/derived/historical_episode_feasibility/20261014_233013_930021/allowlist.json).
- [Checksums cho 50 artifacts](C:/Users/hoang/Desktop/TRAMIF/data/derived/historical_episode_feasibility/20261014_233013_930021/checksums.json).
- [Independent verification](C:/Users/hoang/Desktop/TRAMIF/data/derived/historical_episode_feasibility_verification/20261014_233343_897002/report.json): **50/50 artifact hashes, 29/29 source/input hashes, 18/18 graph partitions PASS**.

SHA-256 của audit `report.json`: `8f232f09d7ddcec2e5b71ba68d0627c1a96c2359631c858e67f329c8832f4f1e`.
Protocol V1 giữ nguyên SHA-256: `e64664c4c499ef762b4a52c45889f0d05db6741ec5021b249f2072585f816546`.

Các file dưới mỗi `E1/`, `E2/`:

| Artifact | Vai trò |
| --- | --- |
| `eligibility.csv` | Counts prefix của tất cả 51 candidate families, kể cả không đạt K_e |
| `class_mapping.json`, `episode_spec.json` | Mapping RIÊNG của episode và timeline; không ghi đè final mapping |
| `decisions.csv` | KEEP/EXCLUDE tại 0/10/20, timestamp, witness SHA, witness stage và distance |
| `train_keep.csv`, `A_keep.csv`, `B_keep.csv`, `C_keep.csv` | Primary retained IDs, chung cho ba views |
| `*_view_alignment.json` | Canonical order và source hash; **không phải bằng chứng ảnh đã được mở** |
| `support.csv` | Per-family A/B samples/groups/concentration, reasons khi fail, C support |
| `train_*_members.csv`, `AB_*_members.csv`, `whole_episode_*_members.csv` | Membership ở graph thresholds 00/10/20, gồm bridge nodes và retained flags |
| `report.json` | Counts, group-size histograms, criteria và limitations của episode |

Run root còn lưu `pair_vertices.csv` và `edges_le20.csv`. Đã so sánh toàn bộ **221.246.130 unordered pairs** trên **21.036** vertices thuộc union E1/E2, lưu **301.981** edges ≤20 hoặc exact-content. Không dùng approximate nearest-neighbor search. Thời gian audit thực tế **171,67 giây**, CPU; đây không phải training benchmark.

## 3. Input allowlist và completeness

Chỉ đọc các dữ liệu đã xuất riêng cho lịch sử:

| Input dưới repo | Phạm vi và mục đích |
| --- | --- |
| `data/manifests/train_validation_manifest.csv` | 26.324 IDs, timestamp, family; Aug2019–Mar2020 |
| `data/manifests/train_validation_zip_index.csv` | Cùng IDs; size/CRC provenance; không mở ZIP |
| `data/derived/train_tlsh_v1/{train_tlsh.csv,config.json,summary.json}` | 18.061 train samples; content SHA và TLSH đã lưu |
| `data/derived/validation_tlsh_v1/{validation_tlsh.csv,config.json,summary.json}` | 8.263 Feb/Mar samples; content SHA và TLSH đã lưu |
| `data/derived/historical_near_duplicates_v1/20261009_143301_921802/{config.json,policy_snapshot.md}` | Binding hashes, primary0/sensitivity10/20, chronology |
| `data/derived/{sbsmi,raw_byte,entropy}_dataset_v1/config.json` | Cùng manifest/index/mapping hashes |
| `docs/train_family_cohort.csv`, `docs/class_mapping.json` | Frozen candidate envelope và final mapping |
| Protocol/policy và source files trong `allowlist.json` | Kiểm tra rule, API và source provenance; không execute builders cũ |

Các đường dẫn raw metadata, binaries, images, checkpoints và predictions không nằm trong data allowlist. `Inputs.read()` chặn đường dẫn ngoài allowlist và path escape, kiểm tra expected hashes. Các inventories được kiểm tra timezone, dates, IDs, labels, config IDs, counts, size, CRC và digest format. `py-tlsh==5.0.0`, standard distance có length component. Re-hash toàn bộ inputs sau run xác nhận không thay đổi trong khi audit.

Historical envelope theo tháng: Aug **576**, Sep **3.397**, Oct **3.869**, Nov **2.274**, Dec **3.814**, Jan **4.131**, Feb **3.825**, Mar **4.438**. Đây là counts của final-51 envelope, **không phải toàn BODMAS**. March được xác minh vì cùng historical export, nhưng không là vertex của E1/E2 và không dùng chọn cohort/support.

**Completeness argument:** với cùng rule ≥50 unique SHA và ≥3 train months, counts trong prefix không thể vượt counts Aug–Jan. Vì vậy mọi prefix-eligible family phải nằm trong final-eligible families. Có thể dùng final51 làm *candidate envelope*, sau đó chọn lại K_e hoàn toàn từ prefix. Audit đối chiếu counts mỗi family trong frozen cohort với toàn train inventory; source `inspect_metadata.py` và `build_train_val_manifest.py` xác nhận rule và xuất mọi matching historical row.

**Provenance limit:** đây là bằng chứng nhất quán với frozen historical export và generator đã có, không phải một lần tái trích xuất độc lập từ raw metadata. Không mở raw CSV chứa Future, không đọc lại bytes để xác minh content SHA/TLSH. Nếu provenance của export gốc bị bác bỏ, kết luận completeness phải mở lại thành UNKNOWN và yêu cầu historical-only export có nguồn xác thực; không tự mở raw file để sửa.

## 4. Timeline và retention thực tế

Mọi mốc là khoảng ngày nửa mở UTC; timestamp tie dùng SHA. Prefix families được chọn trước khi xét A/B/C. Số lượng primary:

| Episode | Stage | Khoảng UTC | Original | Retained | Excluded | Families retained |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| E1 | train | [2019-08-01,2019-11-01) | 7.034 | 7.034 | 0 | 32 |
| E1 | A | [2019-11-01,2019-12-01) | 2.033 | 2.023 | 10 | 32 |
| E1 | B | [2019-12-01,2020-01-01) | 3.496 | 3.480 | 16 | 32 |
| E1 | C | [2020-01-01,2020-02-01) | 3.727 | 3.695 | 32 | 32 |
| E2 | train | [2019-08-01,2019-12-01) | 9.803 | 9.803 | 0 | 42 |
| E2 | A | [2019-12-01,2020-01-01) | 3.680 | 3.664 | 16 | 42 |
| E2 | B | [2020-01-01,2020-02-01) | 3.964 | 3.929 | 35 | 41 |
| E2 | C | [2020-02-01,2020-03-01) | 3.589 | 3.563 | 26 | 40 |

Trong MỖI stage trên, retained count = unique sample identifiers = unique disarmed-content SHA. Không có duplicate-content groups trong original episode cohort; không có TNULL. Không có sample ID xuất hiện ở hai stages trong cùng episode. Metadata lịch sử bắt đầu Aug29, không giả định có sample từ Aug1; min/max timestamps thực tế nằm trong các `report.json`.

Retention giữ nguyên semantics hiện hành: train chỉ loại exact-content copies; train khác bytes vẫn giữ dù TLSH gần. A/B/C loại nếu exact-content hoặc distance≤threshold với **bất kỳ record xuất hiện trước**, kể cả record trước đã bị loại. Không dùng label để quyết định similarity; giữ witness cho mọi exclusion. Không reuse final validation retention cho Nov/Dec/Jan.

Sensitivity dưới đây chỉ là counts, **không đổi primary dataset hoặc signal support**:

| Episode | Retention threshold | Train kept | A kept | B kept | C kept |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | 0 (primary) | 7.034 | 2.023 | 3.480 | 3.695 |
| E1 | 10 | 7.034 | 892 | 1.561 | 1.824 |
| E1 | 20 | 7.034 | 717 | 1.360 | 1.566 |
| E2 | 0 (primary) | 9.803 | 3.664 | 3.929 | 3.563 |
| E2 | 10 | 9.803 | 1.660 | 1.947 | 1.540 |
| E2 | 20 | 9.803 | 1.443 | 1.677 | 1.279 |

Chênh lệch coverage lớn cho thấy primary0 còn nhiều dependence ở ngưỡng10/20. Không suy performance từ số lượng này; không đổi threshold sau khi thấy counts.

## 5. Groups, support và concentration

**Signal graph**: induced graph chỉ trên original A/B records thuộc K_e; content equality hoặc TLSH≤10; excluded A/B records có thể làm bridges. Train và C không làm bridges cho S_e. Whole-episode graph có C chỉ dùng audit dependence, không quay lại chọn groups/support cho signals.

S_e yêu cầu **≥20 primary-retained samples VÀ ≥5 operational joint A/B groups trong MỖI tháng**. Group count không phải independent observations. Không có fold assignment, null resampling hoặc bootstrap trong batch này.

| Đại lượng | E1 | E2 |
| --- | ---: | ---: |
| Eligible classifier families K_e | 32 | 42 |
| Families đạt S_e | 17 | 25 |
| Families không đạt S_e, vẫn giữ trong classifier | 15 | 17 |
| A samples thuộc S_e / toàn A retained | 1.789 / 2.023 | 3.503 / 3.664 |
| B samples thuộc S_e / toàn B retained | 3.130 / 3.480 | 3.659 / 3.929 |
| S_eval = S_e có true support tại C | 17 | 25 |
| A/B groups, gồm cả không có retained member | 2.790 | 4.009 |
| Active A/B groups (≥1 retained member) | 2.789 | 4.009 |
| Largest A/B group, retained members | 127 | 187 |
| Multi-label active A/B groups | 42 | 57 |
| Cross-month active A/B groups | 124 | 218 |
| Retained samples trong cross-month A/B groups | 1.351 | 2.188 |
| Singleton active A/B groups | 2.396 | 3.461 |

Full size histograms có trong `graphs.AB_10.retained_size_histogram`. Largest groups chiếm 127/5.503 và 187/7.593 A/B retained samples; không có predeclared cutoff để gọi một group là “giant” hay tự động fail G2. Concentration theo family có thể nghiêm trọng hơn tỷ lệ toàn bộ: E1 B `dinwod` có **52/128=40,625%** samples trong một group dù có55 groups; E2 B `small` có **41/118≈34,75%** dù có31 groups. Support counts đủ không loại bỏ dominance risk.

Ví dụ support failure không được “sửa cho đạt”: E2 `fasong` có26 A samples nhưng chỉ4 A groups; E1 `qqpass` vừa đủ20 B samples và11 B groups nên đạt. E2 B thiếu `allaple`; C thiếu `allaple` và `tescrypt`. Không thu nhỏ classifier để giấu các lớp này. Tất cả reasons và counts từng family ở [E1 support](C:/Users/hoang/Desktop/TRAMIF/data/derived/historical_episode_feasibility/20261014_233013_930021/E1/support.csv) và [E2 support](C:/Users/hoang/Desktop/TRAMIF/data/derived/historical_episode_feasibility/20261014_233013_930021/E2/support.csv).

C chỉ được dùng để đếm support và audit dependence. Không family nào trong S_e bị chọn bằng C labels, prediction metrics hay loss. C support không làm thay đổi S_e; trong dữ liệu này S_eval không mất family nào.

Các graph bổ sung dưới đây đều đếm **primary-retained members**, chỉ thay graph threshold:

| Episode/scope | Threshold | Active groups | Largest retained group | Multi-label groups | Cross-stage groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 train | 0 | 7.016 | 2 | 1 | 0 |
| E1 train | 10 | 4.060 | 230 | 108 | 0 |
| E1 train | 20 | 3.382 | 230 | 138 | 0 |
| E1 A/B | 20 | 2.495 | 127 | 47 | 159 |
| E1 whole episode | 0 | 16.214 | 2 | 1 | 0 |
| E1 whole episode | 10 | 8.226 | 341 | 173 | 535 |
| E1 whole episode | 20 | 6.899 | 341 | 207 | 660 |
| E2 train | 0 | 9.768 | 3 | 1 | 0 |
| E2 train | 10 | 5.347 | 264 | 132 | 0 |
| E2 train | 20 | 4.449 | 264 | 169 | 0 |
| E2 A/B | 20 | 3.642 | 229 | 60 | 268 |
| E2 whole episode | 0 | 20.924 | 3 | 1 | 0 |
| E2 whole episode | 10 | 10.351 | 372 | 209 | 634 |
| E2 whole episode | 20 | 8.702 | 372 | 250 | 769 |

Các concentration indices trong JSON là `(sum n_g)^2 / sum n_g^2`, chỉ mô tả mất cân bằng group sizes; **không gọi là số groups độc lập/effective statistical sample size**.

## 6. Chronology và leakage review

**VERIFIED, trong allowlisted cohort:** không có exact-content overlap; không còn prohibited earlier-match cho bất kỳ retained A/B/C row ở từng retention threshold tương ứng; không có C vertices trong signal graph. Mọi exclusion witness đi trước theo UTC timestamp/SHA. Retained sets tại0/10/20 lồng nhau đúng. Labels không ảnh hưởng edges.

Primary-retained records vẫn có các direct near-duplicate edges tại10 (số pairs, không phải số samples):

| Stage pair | E1 | E2 |
| --- | ---: | ---: |
| train → A | 13.418 | 22.711 |
| train → B | 17.556 | 23.437 |
| train → C | 17.076 | 15.228 |
| A → B | 4.411 | 10.562 |
| A → C | 5.212 | 5.399 |
| B → C | 10.374 | 11.968 |

Primary0 có **0 cross-stage retained edges** tại0; không đồng nghĩa “không còn leakage/dependence”. Whole-episode graph10 có6.161 E1 và7.695 E2 retained samples thuộc cross-stage components. TLSH0 có thể khác bytes; distance>10/20 không chứng minh independence. Graph chỉ nằm trong eligible cohort; benign/ngoài cohort không làm bridges và có thể còn dependence chưa phát hiện. Connected-component transitivity cũng có thể nối các endpoints vốn không trực tiếp gần nhau.

**UNKNOWN:** label-availability time. Metadata chỉ có observation timestamp và nhãn family hiện có; không có bằng chứng nhãn đã biết tại từng historical cutoff. Chỉ được diễn giải retrospective temporal classification nếu owner chấp nhận giả định nhãn lịch sử; chưa đủ cơ sở gọi prospective deployment. Không suy label availability từ timestamp.

**VERIFIED từ source/protocol:** checkpoints hiện tại đã train đến January, không được reuse làm prefix encoders. E1 C January và E2 C February đã thuộc lịch sử phát triển từng được xem trong các thí nghiệm legacy; đây không phải untouched holdouts. Hai episodes overlap, không là independent temporal replications. Các bước scientific review/selection sau này phải công khai tính retrospective này.

Near-duplicate policy đã được phát triển sau initial SBSMI validation. Audit này không viết lại lịch sử đó hoặc biến protocol thành preregistration trước mọi quan sát.

## 7. Implementation readiness và tests

Code mới:

- [audit_historical_episodes.py](C:/Users/hoang/Desktop/TRAMIF/scripts/audit_historical_episodes.py): allowlist/provenance → prefix eligibility → exhaustive saved-digest distances → chronological decisions → scoped graphs → support/artifacts. Không import model/torch hoặc execute legacy metadata builders.
- [test_historical_episode_audit.py](C:/Users/hoang/Desktop/TRAMIF/tests/test_historical_episode_audit.py): **17/17 PASS**, gồm thresholds, unique IDs, UTC boundaries, excluded bridges/reference, C isolation, label-blind similarity, TNULL, checksum/path guards, legacy-policy differential check, native TLSH API và synthetic episode integration.
- Existing [test_near_duplicate_core.py](C:/Users/hoang/Desktop/TRAMIF/tests/test_near_duplicate_core.py): **8/8 PASS**. Tổng **25/25 targeted tests PASS**; không chạy lại toàn bộ130 tests, không cộng thành kết quả full-suite mới.
- [verify_historical_episode_audit.py](C:/Users/hoang/Desktop/TRAMIF/scripts/verify_historical_episode_audit.py): đối chiếu hashes/counts/order/support; dựng lại18 graph partitions bằng graph traversal, khác UnionFind của audit. Integration verification PASS trên artifacts thực tế.

Lệnh đã thực hiện:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_historical_episode_audit.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_near_duplicate_core.py -v
.\.venv\Scripts\python.exe -B -m scripts.audit_historical_episodes
.\.venv\Scripts\python.exe -B -m scripts.verify_historical_episode_audit --run-dir 'data/derived/historical_episode_feasibility/20261014_233013_930021'
```

Sandbox ban đầu không khởi động được Python Microsoft Store của `.venv`; cùng interpreter đã chạy được ngoài sandbox sau approval của execution layer. Không cài package hoặc sửa môi trường. Không có test failure của implementation bị bỏ qua.

**API blockers trước prefix training:** model constructors nhận `num_classes`, nhưng `sbsmi_dataset.py` và `float_image_dataset.py` cố định final51 mapping và final dates; `train_sbsmi.py`/`train_float_view.py` yêu cầu18.061 train samples, seed42; float trainer còn kiểm tra59.059 parameters. `historical_policy_dataset.py` bọc final dataset. Chưa có episode loader/trainer adapter với K=32/42 và cutoff tương ứng. Audit chỉ ghi nhận, không sửa legacy API. Generated canonical manifests không thay thế một lần kiểm tra loader trên ảnh thật sau khi được phê duyệt.

## 8. Blockers, unknowns và bước đề xuất

| Hạng mục | Trạng thái sau G2 | Hệ quả |
| --- | --- | --- |
| Historical input/provenance, prefix cohort, chronology, counts/support | **VERIFIED trong scope đã nêu** | Không cần đổi timeline/threshold để vượt count gate |
| Manifest/config alignment ba views | **VERIFIED** | Actual image-content/loader integration chưa được xác minh trong batch này |
| Group dependence/concentration | **VERIFIED có phụ thuộc; statistical validity UNKNOWN** | Không mặc định group-independent null/bootstrap; review trước training/signals |
| Historical label availability | **UNKNOWN** | Xin quyết định retrospective assumption hoặc cung cấp provenance nhãn; không tự claim prospective |
| Prefix model/dataset adapters, prefix checkpoints | **Chưa có** | Không được dùng final January checkpoints thay thế |
| D matched-null, U estimator, coefficient selection | **Chưa được kiểm chứng/phê duyệt ở batch này** | G2 PASS không đóng G3–G6 |
| QDU improvement / RQ1–RQ3 | **Chưa được đánh giá bởi G2** | Không suy research validity từ25 tests hoặc đủ support |

**Bước kỹ thuật tiếp theo đề xuất, chưa thực hiện:** theo bước2 của V1, xin owner duyệt một **matched-null allocation/dependence feasibility audit** dùng memberships A/B và S_e vừa lưu. Đầu ra cần là: đơn vị resampling đề xuất, khả năng giữ whole-group/multi-label/shared-month integrity và matched class counts, các allocation không thể thực hiện, cùng exchangeability assumptions. Acceptance là kết luận có/không có cơ sở cho null design và lý do kiểm chứng được; không cần tạo embeddings, tính MMD hoặc thử outcome losses để chọn null. Đủ allocation chưa đồng nghĩa valid null. Nếu không đủ cơ sở thì giữ standardized D ở UNKNOWN/UNAVAILABLE và xin quyết định fallback riêng; không tự chuyển sang descriptive D.

Sau review các scientific gates, công việc engineering cần phê duyệt riêng là episode dataset/config adapter dựa trên immutable manifests G2. Acceptance: đúng K_e/canonical IDs giữa ba views, cutoff isolation, không ghi đè legacy51 mapping, input mismatch phải fail. Sau đó mới có thể kiểm tra loader trên ảnh và xin phép prefix training; chưa thực hiện các bước này.

Trước khi chạy prefix experiments còn phải giải quyết scientific gates theo thứ tự V1: label interpretation, review dependence/concentration, approve feature/null/U methodology và coefficient selection, rồi resource approval. Có data support không thay thế các gates này. Nếu provenance/label assumptions không thể chấp nhận hoặc null/estimator fail, dừng phần liên quan và xin redesign/giảm claim, không tự hạ ngưỡng, đổi tháng hay thêm episodes.

**Đã hoàn thành và dừng G2.** Artifacts trong `data/` bị Git ignore, cần backup riêng; không có thay đổi dữ liệu/model hay mở Future Test.
