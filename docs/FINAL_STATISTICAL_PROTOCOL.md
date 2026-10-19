# D059 — Final pre-Future statistical specification

2026-10-18. **OWNER APPROVED D060 — O3 MODIFIED; FUTURE ACCESS STILL CLOSED.**
Owner duyệt O1/O2/O4 và O3 với file-size analysis bắt buộc. Amendment được ghi
trước khi đổi hash tại `docs/D060_O3_SCOPE_AMENDMENT.md`; snapshot D059 giữ nguyên.
Đây là statistical specification có thẩm quyền cho combined freeze; không cấp quyền
mở Future. V1 và Amendment 01 gốc không đổi. Thay đổi tiếp cần decision/version mới.

## 1. Evidence và thứ tự thẩm quyền

Nguồn: framework gốc `Temporal_Reliability_Malware_Research_Framework.docx`
§11.3–11.4,12.2–12.3; V1 §8–13; Amendment 01; D041/D048/D050/D054/D058.
Framework giữ mean-monthly và worst-month Macro-F1, five seeds, bootstrap intervals
và broad controls. Bỏ intervals/controls là **scope amendment cần duyệt**, không
phải cách tuyên bố đã hoàn tất toàn bộ framework.

| Hạng mục | Evidence/trạng thái hiện tại | Xử lý D059 |
|---|---|---|
| Hai primary estimands, K51, chronology | V1 §11.1; framework §11.4 | Giữ nguyên; §2 làm rõ executable details |
| Five seeds42–46, mỗi seed ba checkpoints/T/Q/D | D054 `FINAL_MULTISEED_VERIFICATION.md`, `FINAL_MULTISEED_RESULTS.md`: 15/15 và5/5 VERIFIED | Tái sử dụng; không audit/train lại |
| beta1,gamma0 | D048 lock; D049 E1 selection; D050 E2 không retune | Giữ nguyên |
| Descriptive d=MMD²/2 | Amendment01,D019,D048; bounded score không phải statistical standardization | Không gọi standardized/significance/predictor đã validated |
| U | D041 simulation negative; diagnostic-only | QU/QDU UNAVAILABLE, không thế bằng zero rồi đặt tên QDU |
| Learned-static | D058 `LEARNED_STATIC_V2_RESULTS.md`: 5/5 converged/frozen | Có baseline; fitted March NLL không phải held-out performance |
| Group×seed percentile CIs, family10 | V1 §11.2–11.3 / D052 owner-decisions: PROPOSED | D060 owner approves point-only/family8; coverage NOT ESTABLISHED |
| Scope/cost/composition controls | V1 §8–10, D050,D052: chưa đầy đủ | Đề xuất scope hẹp và departures rõ ở §7 |
| Future | Chưa truy cập | Mọi support counts/metrics Future vẫn UNKNOWN |

D058 receipt: `data/derived/final_learned_static_v2/20261018_003544_337822/receipt.json`.
D054 receipt: `data/derived/final_multiseed_verification/20261017_164739_728045/receipt.json`.
D050 evidence: `docs/historical_research_results.md`.
Đây là tái sử dụng evidence đã kiểm chứng; D059 không chạy lại training, solver,
calibration, Q/D hay historical evaluation.

## 2. Exact primary estimands — không thay V1

Index: seed s thuộc (42,43,44,45,46), method a, month m thuộc April–September2020.
Chỉ sau authorization, áp dụng frozen primary retention policy và mapping K51.
I_m là toàn bộ **original primary-retained known-family** samples của tháng m.
S_m = các class K51 có ít nhất một true sample trong I_m. Publish sorted S_m,
class counts, IDs/order và hashes một lần; dùng chung cho mọi seed/method.
Không dùng S25 historical quality support, cap128, ≥20 samples hay ≥5 groups để
lọc evaluation families. Không chọn S_m theo predicted labels/performance.
Các tập S_m thực tế chưa biết; chỉ quy tắc định nghĩa đã cố định trong V1.

Full K51 calibrated probabilities được giữ nguyên. Với seed s:
- Single-view: softmax(logits_v/T_sv), T từ February riêng seed/view.
- Uniform: trung bình ba probability vectors.
- New Q: frozen w=softmax(-L) từ March B/S25.
- Q+D: frozen w=softmax(-L-d), beta1,gamma0.
- Learned-static: frozen D058 mixture weights; zero weights được phép.
Các fusion đều là sum_v w_sv p_sv. Không average logits, refit temperature,
renormalize trên S25/S_m hoặc average probabilities giữa seeds.

Prediction là argmax trên51 columns theo mapping gốc. Exact probability ties:
chọn canonical class index nhỏ nhất, như NumPy argmax hiện có. Không epsilon-tie
để đổi prediction. Float64 cho metrics; stable log-softmax/log-mixture cho NLL;
mixture với zero weight bỏ term đó trong log-sum, không thay weight bằng epsilon.
Check normalized probabilities với absolute tolerance1e-12,rtol0 như existing
`uniform_fusion.classification_metrics`; không silently repair/clamp/renormalize
inputs sai. Underflow của exp cho một xác suất rất nhỏ không được dùng để thay
log-probability hữu hạn đã tính ổn định khi tính NLL.

Với confusion matrix C (row=true,column=prediction),

    F1_samc = 2 TP_c / (2 TP_c + FP_c + FN_c)
    F_sam = (1 / |S_m|) sum_{c in S_m} F1_samc
    Mean_sa = (1/6) sum_m F_sam
    Worst_sa = min_m F_sam
    Mean_a = (1/5) sum_s Mean_sa
    Worst_a = (1/5) sum_s Worst_sa

Mỗi family trong S_m có trọng số bằng nhau, rồi mỗi tháng bằng nhau, rồi mỗi seed
bằng nhau. Không pooled-sample F1 thay mean-monthly; không min_m mean_s F_sam thay
mean_s min_m F_sam. Ví dụ: năm seeds có tháng tệ khác nhau, F1=0 ở một tháng và1
ở năm tháng còn lại: Mean=5/6, Worst=0; min của seed-mean theo tháng là0.8 và
**không phải** primary Worst.

Missing/edge rules:
- Class absent khỏi original I_m không vào supported macro denominator nhưng vẫn
  có output column. Predict vào absent class vẫn tạo FN cho true class.
- Per-class F1 nếu denominator=0 được ghi0; ở original supported class denominator
  luôn dương. Fixed51 secondary F1 dùng đủ51 classes với convention này.
- Per-class recall nếu không true support ghi null/undefined, không biến thành0.
  Balanced accuracy average recall trên true-supported classes.
- Empty/invalid month, thiếu seed/method/cell, duplicate/missing IDs, provenance
  mismatch hoặc nonfinite metric: STOP; không bỏ tháng, sample hay seed để hoàn tất.
  Có thể lưu partial diagnostics nhưng primary aggregate ghi NOT_ESTIMABLE.
- Exact tied worst months: lưu tất cả tied month IDs, scalar minimum chỉ tính một
  lần. Không đổi month/seed theo tie. Deltas giữ full precision; làm tròn chỉ lúc
  hiển thị. F1 và deltas lưu proportions; percentage points =100×delta.
- Schema dùng exact seed/method/month order; không tự suy alignment từ array shape.
  Loader tương lai phải verify canonical IDs, labels, timestamps, mapping và hashes
  trước khi gọi pure-array aggregation. D059 không chứng minh loader integration.

## 3. Reporting và một fixed contrast family

Seven methods theo thứ tự:
`sbsmi, raw-byte, entropy, uniform, new-quality-only, quality-plus-drift, learned-static`.
Report đủ 5×7×6 monthly scores, từng seed Mean/Worst và từng method mean, sample
SD(ddof=1),min,max trên5 seeds. Seed SD/range chỉ là spread của repeated fits,
không SE,confidence interval hay independent temporal replication.

Đề xuất **một family gồm8 paired outcome contrasts**:

| Priority | Left − Right | Hai outcomes | Lý do |
|---|---|---|---|
| Primary | Q+D − New Q | Mean và Worst | Giá trị bổ sung của D ngoài Q; cùng CNN/access/capacity |
| Secondary | New Q − Uniform | Mean và Worst | Giá trị của historical quality weighting |
| Secondary | Q+D − Uniform | Mean và Worst | Full available rule so với fusion không học weights |
| Secondary | Q+D − Learned-static | Mean và Worst | So với baseline học static weights cùng historical budget |

Mỗi delta_s=outcome_s,left−outcome_s,right; báo mean,SD(ddof1),min,max của đúng năm
paired deltas. Không subtract SDs hoặc đối chiếu hai range để suy significance.
Primary improvement direction là delta>0 ở cả hai outcomes. Không chọn primary
sau outcomes. Single-view comparisons được giữ trong complete method/month tables
và descriptive deltas QD−từng single-view; không chọn best single-view từ Future
làm primary, không claim simultaneous superiority.

Multiplicity: family8 là reporting family **không phải tám hypothesis tests đã
calibrate**. Công bố cả8, cả dấu âm; không p-values, không significance stars,
không 95%/99.5% intervals, không FWER/FDR claim. Không cần/không giả vờ đã áp dụng
Bonferroni cho point estimates. Owner D060/O2 đã thay proposed V1 family10 chứa
unavailable U arms bằng family8. Không bỏ unavailable arms rồi vẫn claim full V1.
Nếu muốn inferential testing sau này, phải duyệt một study/protocol trước một
holdout mới; không chọn test sau khi nhìn sáu tháng này rồi gọi confirmatory.

Practical target đề xuất giữ V1: QD−Q Mean≥0.02 và Worst>0 trên aggregate gốc.
Chỉ gọi “đạt practical target trên horizon quan sát”, không significance.
Mean tăng/Worst giảm là trade-off. Small positive dưới target được báo numerical
gain; negative/no gain là kết quả hợp lệ. Target không cho phép retune/loại seed.

## 4. Secondary metrics — mọi method, cùng data

Giữ `scripts/uniform_fusion.py::classification_metrics`:
- Accuracy; supported và fixed51 Macro-F1; supported balanced accuracy; per-class
  support/recall/F1; confusion matrix; sample-mean NLL dùng natural logarithms.
- Multiclass Brier =mean_i sum_{k=0}^{50}(p_ik−1[y_i=k])², không chia thêm51.
- Top-label ECE15 với equal-width bins: floor(15×maxprob), capped14. Bins
  [j/15,(j+1)/15), cuối chứa1; empty bins count0,accuracy/confidence null, contribution0.
- Report mỗi month/seed. Pooled April,April–June,April–September horizons: concatenate
  samples, tính lại confusion/probability metrics; đặc biệt pooled fixed51 F1.
  Không average monthly F1 để giả pooled F1. Average các summary qua5 seeds sau
  khi tính từng seed. Không average probabilities thành seed ensemble.
- Pooled/secondary scores không thay hai primary outcomes hoặc quyết định coefficient.
  Report cohort coverage, absent/unknown/excluded families và preprocessing failures.
  Unknown families ngoài K51 không âm thầm đưa vào known-family accuracy; ghi coverage
  riêng, không claim open-set/benign-vs-malware detection.

## 5. Bootstrap review và coverage decision

**Recommendation: chưa dùng paired group×seed percentile intervals cho final report.**
NOT ESTABLISHED không đồng nghĩa chứng minh mọi bootstrap bất khả thi. D059 không
thực hiện real resampling hoặc một coverage simulation với DGP tùy tiện.

| Vấn đề | Assumption cần có / giới hạn evidence | Hệ quả |
|---|---|---|
| Operational TLSH components | Within-group dependence được giữ, nhưng between-group independence/exchangeability chưa chứng minh; relation qua history, collection/packing có thể còn | Whole-group integrity là engineering property, không coverage theorem |
| Cross-month/multi-label groups | Nếu bootstrap, phải resample nguyên group và giữ month/class labels; không tách theo tháng/lớp | Shared draw giữ pairing nhưng rare support/unequal size vẫn có vấn đề |
| Rare classes/giant groups | Draw có thể mất class hoặc cả tháng; original denominator giữ lại,F1zero là convention | Không tạo thông tin mới; zero-mixture distribution và random sample totals có thể méo percentile; empty month STOP |
| Five seeds/shared data | Independent random streams có thể cho independent fits CONDITIONALLY trên fixed data; sharing data không tự chứng minh dependent RNG | Chỉ5 fits và shared data không đại diện uncertainty do training sample,calibration,selection/temporal population; không coi seed×sample IID |
| Six fixed months | Đây là fixed deployment horizon, không random sample của mọi future months | Không bootstrap tháng tùy tiện để claim future-year/general drift robustness |
| Worst=min six estimates | Nếu unique well-separated population minimum có thể smooth locally; ties/near-ties làm minimum nonregular | Standard percentile coverage không tự kế thừa coverage của từng monthly F1 |
| Group×seed product resampling | Cần một defensible crossed-effects sampling model và estimator-specific validity | Kết quả cho mean/variance ở crossed arrays không tự áp dụng nonlinear F1/min/5seeds |
| 20,000 draws,Bonferroni | Nhiều draws giảm Monte Carlo error CHỈ của chosen resampling distribution; correction cần valid marginal inference | Không chữa sai dependence model,bias hoặc nonsmoothness |

Đề xuất inference target hẹp: **descriptive performance của fixed fitted systems
trên fixed six-month known-family cohort**, cộng variation của năm registered fits.
Không đưa 95%CI cho một population estimand chưa có defensible sampling model.
Không dùng t-test/Wilcoxon/sign test trên5 seeds hoặc6 months để thay bootstrap.

Vì sao không thêm coverage simulation: finite synthetic experiment có thể kiểm tra
coverage CHỈ dưới DGP/parameter regimes đã nêu; không xác nhận TLSH groups độc lập
hoặc timing/collection process thực tế. Hiện không có justified bounded DGP family
để bảo vệ coverage claim cho cả Mean và Worst. Tạo simulation cho ra PASS sẽ không
giải quyết assumption còn thiếu. Các synthetic checks §8 kiểm tra arithmetic,
không gọi coverage validation. Không có simulation result bị bỏ hoặc lựa chọn.

Nguồn primary (lập luận áp dụng cho TRAMIF ở bảng trên là methodological inference):
- [Owen & Eckles (2012)](https://arxiv.org/abs/1106.2125): product reweighting có
  kết quả variance cho mean trong crossed random-effects settings dưới conditions;
  không phải coverage guarantee cho mọi metric/data array.
- [Fang & Santos (2019)](https://academic.oup.com/restud/article/86/1/377/5094886):
  standard bootstrap có thể fail cho directionally differentiable nhưng không
  fully differentiable functionals. Đây là cảnh báo cho ties ở minimum, không
  khẳng định mọi worst-month interval đều sai.

## 6. Research interpretation — kết quả nào trả lời câu hỏi nào

- RQ1: khác biệt descriptive class-conditional MMD của **view–encoder pairs** với
  frozen normalization/kernel/support. Không intrinsic representation stability,
  standardized drift hoặc calibrated significance. Bandwidth/dimension/cap/group
  concentration vẫn ảnh hưởng; giữ sensitivity findings D050.
- RQ2: limited historical next-window forecast screening, không chứng minh D dự báo
  future error tổng quát. E1 là selection, E2 là overlapping held-forward check.
  D050 E2 forecast MAE Q .563763497→QD .561424341, nhưng fusion F1 giảm
  .683443→.683434, NLL/Brier xấu hơn; không retune beta1.
- RQ3: đánh giá frozen Q/QD/learned-static trên final horizon sau freeze/approval;
  QD−Q isolate weighting difference trên cùng models. D có ích hay không phải chờ
  actual results; thắng metric không chứng minh mechanism/causal link.
- Historical March fit là development. D058 learned-static March NLL không phải
  bằng chứng generalization hoặc fair held-out superiority.
- U negative evidence được báo cáo; không full Q+D+U validation, không extrapolate
  từ engineering PASS hoặc prior March bootstrap sang final coverage.

## 7. Minimum defensible scope và controls

| Category | Requirement / decision | Claim khi thiếu |
|---|---|---|
| A essential | Seven arms,five seeds,same IDs/K51,T/access; Q/QD/learned weights frozen; no best seed; both primary outcomes; all months and secondary tables | Thiếu thì chưa hoàn tất final comparison |
| A essential | Frozen primary retention0,chronology/label/mapping/provenance,coverage; disclose policy developed after initial validation | Không claim duplicate-free independent population |
| A retained V1 diagnostic | Retention10/20 evaluation-selection sensitivity với same primary models/T/weights,không retrain | Báo all thresholds,coverage; không chọn threshold sau performance; NOT YET RUN, cần Future authorization |
| A essential disclosure | Parameter counts per branch,total fusion cost; documented compute/environment và available CPU synthetic benchmark | Không gọi synthetic CNN latency là end-to-end speed |
| A required D060/O3 | TRAIN-fitted file-size quartile descriptive reporting; frozen cutoffs,counts/support and all methods/seeds | Chưa chạy Future; stratification không phải causal/packing adjustment; không retune |
| B optional under narrow scope | Family-composition/packing multivariable adjustment | Unadjusted RQ2 screening; không claim D useful beyond size/packing confounding |
| B optional under narrow scope | Capacity-matched single view,same-view ensemble,MLP/state-permutation | Không intrinsic view superiority,geometry claim hoặc multiview capacity/cost advantage |
| B optional under narrow scope | Measured end-to-end transforms/I/O/hash/CNN/fusion latency,RAM/VRAM across file sizes | Không efficiency superiority/deployment throughput guarantee |
| B optional under narrow scope | Family leave-out/temporal-block sensitivity,score-shuffle,drift-only | Không independence/significance claim; không gọi omitted sensitivity requirements completed |
| C unsupported/out of reduced study | Null-standardized D,U fusion,QU/QDU,full adjusted predictor; broad1D-byte/PE-GBT/gating,contrastive/adaptation/rejection controls | Không full-original-framework validation,SOTA,causal/general robustness claim |

V1 §8 yêu cầu file-size stratification cho RQ2; framework §11.4 yêu cầu group
intervals/family/block sensitivities và §11.3 rộng hơn seven methods. D060 owner
giữ file-size stratification và retention10/20; chấp nhận defer các controls còn
lại với claim reductions ở trên, không xóa requirement lịch sử. Giữ cả hai primary outcomes/five seeds
và retention sensitivity10/20; không hạ về seed42-only hoặc lấy accuracy làm primary.
Size/capacity evidence không cần thiết để mô tả chênh lệch QD−Q với same encoders,
nhưng cần thiết nếu muốn giải thích nguyên nhân hoặc claim vượt confounding.

File-size convention D060: pooled frozen TRAIN sizes, float64 linear quantiles
.25/.5/.75; bins (-inf,q25],(q25,q50],(q50,q75],(q75,inf). Ties không bị chia;
empty bins ghi NOT_ESTIMABLE/count0, không đổi cutoffs. Report monthly/bin support,
fixed51 và supported metrics,thêm F1 trên original whole-month support. Không dùng
bucket summaries thay primary Mean/Worst; không chọn phương pháp từ bucket tốt.

## 8. Executable contract và bounded verification

Additive pure-array module `scripts/final_point_statistics.py::aggregate_monthly`:
input(5,7,6), explicit ordered seeds/methods/months; no filesystem/data readers,
no model, no resampling or fitting. Reuses existing metric definitions, with
`multiseed_statistics.outcomes` used as independent aggregation cross-check in tests.
Returns per-seed primary outcomes,monthly scores,seed spread,all worst-month ties
and8paired deltas. D060 records owner adoption; updated source is separately pinned.
This is not a Future loader; old seed42/six-arm evaluator is not a five-seed
seven-arm integration. That bounded integration remains before execution.

**19/19 targeted tests PASS**:13new plus6existing. Command:
`python -B -m unittest tests.test_final_point_statistics tests.test_multiseed_statistics -v`.
Observed runtime0.215s. Checks: averaging order,unequal month counts,pooled mismatch,
class imbalance,absent labels,undefined recall/F1,exact argmax ties,all-five-seed
order,invalid/missing/nonfinite cells,ddof1,8paired differences,determinism,
nonmutation; legacy synthetic whole-group/percentile arithmetic remains tested.
Legacy bootstrap functions unchanged and NOT authorized for real execution.
No claim of statistical coverage, real-data alignment or Future integration PASS.

Freeze package must bind this spec/code/tests hashes; all15checkpoints,per-seed
temperatures/Q/D/learned weights,source snapshots,mapping,selection/retention policy;
future input receipts only after separate approval. Require immutable output
directories,complete errors,canonical IDs,class/month counts,all predictions/logp,
monthly confusion/per-class/pooled/seed/contrast tables and checksums. Readback must
recompute aggregation from saved predictions; do not substitute receipt status for
actual numerical verification. No new optimizer/inference has run in D059.

## 9. Owner decisions adopted in D060 and finite next step

| ID | Owner decision required | Recommendation |
|---|---|---|
| O1 | APPROVED | Original-data points+all-five seed SD/range; no final bootstrap,CIs,p-values or significance claims; coverage NOT ESTABLISHED |
| O2 | APPROVED | Exactly8 outcome contrasts §3; QD−Q primaryboth; secondary Q−Uniform,QD−Uniform,QD−learnedboth; singleviews descriptive |
| O3 | APPROVED WITH MODIFICATION | Seven-arm Q/QD study; descriptive D,Udiagnostic-only; retain5seeds,2outcomes,retention10/20 AND required TRAIN-quartile file-size analysis; disclose deferred controls |
| O4 | APPROVED | Edgecases,secondary metrics and practical target §2–4; no post-Future coefficient/method/seed/support/metric changes |

Sau khi O1–O4 được duyệt: **một batch bounded combined freeze** — wire/test seven-arm
five-seed consumer bằng synthetic inputs; package existing verified artifacts and
this spec; verify readback/backup coverage và báo phần chưa off-device. Không refit
hay audit training lần nữa. Legacy seed42 package không đủ thay final combined one.
Sau freeze+reproduction/backup và explicit owner Future authorization mới được
prepare protected inputs/infer/evaluate primary+predeclared sensitivities. D059
approval không mặc nhiên cấp quyền Future. Invalid/empty data hoặc provenance
mismatch STOP; pipeline repair phải có incident record, không selection thuận lợi.

Forbidden after opening Future: coefficient/weight/T/checkpoint/support/threshold
retuning; choose best seed/month/baseline; replace estimands; tune statistical test
from observed outcomes; drop failures/negative results; train using Future;
promote QDU unavailable into evaluated arm. New exploratory questions phải tách
nhãn và cần new untouched data cho confirmatory claims.

**Completion:** D059 review and owner decisions are complete via D060. Engineering
integration/freeze evidence is recorded separately; final effects remain unmeasured.
Research may finish with weak/negative QD results under the approved reduced scope;
không cần mở lại search để cứu hypothesis. Chưa PRE_FUTURE_READY/PROJECT_COMPLETE.

D059 verification receipt: `data/derived/d059_statistical_specification/20261018_052308_445235/receipt.json`.
Contains immutable specification/source snapshots, fixed candidate config,
observed targeted-test receipt and input/source hashes; no Future inputs.
