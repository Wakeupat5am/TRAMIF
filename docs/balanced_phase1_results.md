# BALANCED Phase 1 — scope and feasibility results

Date: 2026-10-14 Asia/Saigon. Only Tasks 1 and 2 completed. STOP before D/U.
Scope: `docs/balanced_research_scope.md`; owner approval D014.
Run: `data/derived/balanced_phase1_audit/20261013_233825_921873/` (UTC directory).
Report SHA-256: `c86639b6eeae08e80cbe83ae858d6ae2e099c7c5f0f3d6a0e7fcbb8033a6668f`.

## Evidence and boundaries

**VERIFIED:** saved inventory/retention hashes, 51-class mapping, three checkpoint
file hashes, February temperature bindings, six historical prediction files' IDs,
labels and class order; within-view preprocessing identities agree across months.
Existing February and March graph partitions were reproduced, not just counts.
The old freeze, March evaluation and bootstrap checksum manifests were checked.
No checkpoint deserialization, binary/image loading or CNN execution occurred.
Dataset IDs use canonical sorted-JSON hashes, separately from config file hashes.
An initial new-runner check confused these identities, failed before output, and
was corrected against both existing dataset loaders; a regression test covers it.

**VERIFIED:** exhaustive graph/count audit on saved historical TLSH/SHA inventories.
Binary content was NOT rehashed; provenance rests on those verified inventories.
**INFERRED:** bounded feature extraction is practical based on layer sizes, but
actual throughput, feature degeneracy and MMD behavior remain **UNKNOWN**.
**PLANNED:** March Q, D, U and new weights. No estimator was executed or approved.
The counts passing unit/integration checks do not establish statistical validity.

## Task 1 — nine-arm information budget

All arms retain the three August-January checkpoints, 51 outputs, primary retention
threshold 0 and existing full-February temperatures. No March temperature refit.

| Arm | Fusion information | Role |
| --- | --- | --- |
| SBSMI calibrated | February temperature for its fixed CNN | Single-view control |
| Raw-byte calibrated | Same period, its own fixed CNN/temperature | Single-view control |
| Entropy calibrated | Same period, its own fixed CNN/temperature | Single-view control |
| Uniform | Three calibrated probabilities, fixed thirds | Fixed fusion control |
| Legacy February Q | Existing February OOF quality/support/weights | Immutable historical reference |
| New March Q | Shared v2 support; March calibrated quality | Matched-history control, planned |
| Q+D | Exactly new Q plus historical February-March drift | Planned ablation |
| Q+U | Exactly new Q plus group uncertainty of quality | Planned ablation |
| Q+D+U | Same Q/D/U as these ablations | Planned full rule |

All four new arms have the same historical information access, S*, calibration and
checkpoints. Disabling a signal is the intended ablation. Comparing full v2 ONLY
with legacy February Q would confound newer history with additional mechanisms.
March becomes historical development for v2: evaluating its newly fitted weights
on March cannot provide independent validation. Future data remain unopened.
Full framework crosswalk and permitted claims are in the scope document.

## Task 2A — joint graph

Graph vertices: original 8,263 historical cohort samples (3,825 February and 4,438
March). Primary-retained 8,131 samples remain 3,799/4,332. The other 132 records
can connect graph paths but do not re-enter scoring or support counts.
Edges: content SHA equality OR standard TLSH distance <=10 (length included).
All 34,134,453 unordered pairs checked; no approximate neighbor search.

| Finding | Count |
| --- | ---: |
| Exact content equality edges, within/across months | 0 / 0 |
| TLSH <=10 within-month edges | 48,946 |
| TLSH <=10 cross-month edges | 12,280 |
| Cross-month edges with both endpoints retained | 10,586 |
| Joint components, all / active retained | 4,048 / 4,048 |
| Components touching both months, all / retained | 179 / 178 |
| Retained samples in cross-month components | 2,197 (27.020%) |
| Multi-label components, all / retained | 41 / 41 |
| Retained-only graph active groups | 4,048 |
| Extra retained-component merges caused by excluded bridges | 0 |

Retained group size: 3,454 singletons, 310 pairs; median 1, p90 2, p95 3,
p99 23.53, maximum 278 (3.419% of retained samples). Full frequency table is in
report.json. Largest component J000001 is small: 180 February + 98 March retained
samples, 287 vertices including 9 excluded. Other large groups include 212 mixed
sillyp2p/small and 204 small. No component dominates the entire cohort, but that
does not imply weak dependence or adequate class-specific effective information.

Size concentration `N^2 / sum(group_size^2)` is 221.836, NOT a count of independent
statistical units. Graph groups only encode this particular similarity rule.
Transitive paths may join endpoints with distance >10; different components can
still be correlated or similar by a criterion TLSH misses. No outside-month bridge
nodes were used. Threshold 10 is an audited proposal, not an independence guarantee.

Month-local active groups were 2,032 February / 2,234 March; joint-component IDs
represented within each month give 2,005 / 2,221. Cross-month paths merge local
groups: e.g. sillyp2p 24 -> 16 February and 34 -> 28 March; small 49 -> 35 and
42 -> 37. Do not count 2,005+2,221 as distinct joint groups: 178 are shared.
The observed absence of extra excluded-bridge merges is not assumed in the code;
a synthetic regression test specifically exercises such a merge.

## Task 2B — candidate common support S*

Rule unchanged: >=20 retained samples AND >=5 JOINT operational groups in EACH
month. **25 families qualify**, containing 3,380 February (88.971%) and 4,064 March
(93.813%) samples. These are signal-estimation candidates, NOT a replacement
evaluation cohort. The classifier/evaluation retain all 51 output classes.
Cross-month merges reduce group counts but do not change the qualifying set versus
month-local counts in this particular audit. Every failed class has insufficient
samples in at least one month; some also fail group support.

Table entries are retained samples / joint groups. Complete diagnostics and failure
reasons are in class_support.csv (including local groups and group concentration).

| Family | February n/g | March n/g | S* |
| --- | ---: | ---: | :---: |
| agent | 21/9 | 47/9 | yes |
| allaple | 0/0 | 0/0 | no |
| ausiv | 1/1 | 0/0 | no |
| autoit | 66/32 | 54/36 | yes |
| autorun | 58/47 | 41/9 | yes |
| benjamin | 233/53 | 70/24 | yes |
| berbew | 23/18 | 131/85 | yes |
| bladabindi | 16/15 | 16/14 | no |
| blocker | 5/5 | 3/2 | no |
| ceeinject | 20/12 | 20/20 | yes |
| coinminer | 43/43 | 20/20 | yes |
| cryptinject | 4/4 | 6/3 | no |
| delf | 18/11 | 18/14 | no |
| dinwod | 37/29 | 187/29 | yes |
| ditertag | 8/7 | 3/3 | no |
| dorv | 15/13 | 15/15 | no |
| fasong | 15/7 | 7/3 | no |
| fuerboos | 1/1 | 2/2 | no |
| gandcrab | 106/49 | 137/49 | yes |
| ganelp | 319/90 | 601/106 | yes |
| gepys | 140/113 | 69/33 | yes |
| gupboot | 7/5 | 4/2 | no |
| lunam | 89/87 | 39/38 | yes |
| mira | 54/52 | 260/251 | yes |
| mocrt | 8/8 | 3/3 | no |
| musecador | 312/43 | 44/20 | yes |
| mydoom | 6/6 | 29/23 | no |
| nitol | 14/14 | 11/11 | no |
| occamy | 9/8 | 43/42 | no |
| padodor | 51/51 | 65/63 | yes |
| picsys | 36/32 | 69/53 | yes |
| pluto | 22/6 | 11/3 | no |
| pykspa | 58/57 | 94/92 | yes |
| qqpass | 17/16 | 24/13 | no |
| sfone | 247/241 | 258/258 | yes |
| sillyp2p | 118/16 | 299/28 | yes |
| small | 397/35 | 432/37 | yes |
| socks | 1/1 | 0/0 | no |
| soltern | 87/14 | 17/12 | no |
| stormser | 24/5 | 9/3 | no |
| sytro | 75/19 | 7/7 | no |
| tescrypt | 0/0 | 0/0 | no |
| unruy | 97/35 | 224/64 | yes |
| upatre | 413/370 | 328/272 | yes |
| urelas | 71/46 | 42/31 | yes |
| vb | 4/4 | 2/2 | no |
| vflooder | 27/8 | 19/7 | no |
| vobfus | 4/3 | 6/6 | no |
| wabot | 178/164 | 334/272 | yes |
| wacatac | 193/95 | 199/148 | yes |
| zbot | 31/27 | 13/12 | no |

## Task 2C — drift feasibility and a design gate

**VERIFIED from source:** MalSBSLCNet.head_norm supplies 2,048 features before
dropout; CompactViewNet.forward_features supplies 512 features for each other view.
Extraction would require eval mode (fixed BatchNorm statistics, dropout disabled),
identical per-view checkpoint/preprocessing on both months and ID-joined outputs.
SBSMI currently has no forward_features helper, but a read-only head_norm hook is
possible. Layers were inspected, not executed. Saved evaluation NPZs have logits,
not these embeddings. No new features were extracted.

**INFERRED resource scale:** 8,131 * (2048+512+512) float32 values = 95.285 MiB of
feature payload; all 18,061 train references would add 211.652 MiB. Metadata,
activations, framework runtime and temporary distance matrices are additional.
A full 8,131-square float64 kernel alone is about 504.4 MiB; it is unnecessary
for bounded per-class processing. Future approval should specify batching/chunking
and a train-only bandwidth reference cap before extraction; these are not selected.
Runtime/peak memory UNKNOWN until an authorized bounded extraction smoke check.
CPU extraction is possible; GPU inference requires separate approval. No retraining
is necessary for comparable February/March coordinates using these fixed encoders.

A train-only reference can fit normalization and bandwidth without future leakage.
Choose its sampling manifest from the existing train split before historical signal
inspection. Its embeddings are in-sample: using it as a scaling reference is not
evidence of honest historical held-out performance. Zero variance/bandwidth and
feature quality remain untested. Final encoders cannot create honest pre-January
forecast episodes. Both February/March are after CNN fitting; February was used
for temperature calibration (not for changing this frozen encoder).

Counts permit computing a class-conditional distance on 25 families, but **do not
establish a valid normalized drift score or useful predictive signal**. No MMD,
drift/error correlation, null distribution or null variance has been measured.

Counts-only null feasibility:
- Exact per-family binary group allocation capacity is saved. A separate single
  shared greedy allocation witness respects mixed-label groups per month. It is
  NOT a random null distribution, repeated split search or approved estimator.
- For agent February, the largest group has 12/21 samples; the other groups total
  9. Therefore the maximum equal subsample budget from disjoint group sides is 9.
  No different partition can supply 10 on both sides without splitting a group.
- In the shared witness, all 25 families have nonempty sides; February has 24
  families with >=10 per side, 20 with >=20, 9 with >=64 and 4 with >=128. March
  has respectively 25, 22, 12 and 7. ceeinject allows only 10 per side each month;
  coinminer March 10; berbew February 11. A cap of 128 is an upper bound, not a
  requirement that all families achieve 128. Variable smaller budgets still need
  statistical justification, equal observed/null budgets and approval.
- Single-family exact capacities need not be simultaneously attainable because
  mixed-label groups couple allocations. The shared witness gives one attainable
  count configuration, not an optimal/unique partition. Its February and March
  allocations are independent witnesses for within-month splitting, not a common
  cross-month sample-separation scheme.
- Real February/March share 178 retained components; a within-month null made
  from disjoint groups has a different dependence structure from the observed
  two-month comparison. It cannot automatically be called a calibrated null or
  yield valid significance/standardized drift evidence. Deleting overlapping
  groups would change the signal target/support and is NOT an approved fallback.

**Unresolved design gate:** preserve S* and honestly accept bounded, descriptive
MMD with explicit uncertainty limitations, OR design a dependence-matched null
before adopting a standardized score. Neither is chosen in this batch. A simple
disjoint same-period split cannot be silently promoted to the latter. Any
restriction of S*, variable-budget rule, normalization/degeneracy handling or
fallback needs owner approval. Until then drift weighting is blocked by design,
not by unavailable files or a failed numeric implementation.

## Task 2D — positive group-weight U feasibility (analysis only)

Unit: one joint operational component, restricted to its retained March members
when estimating March quality. A multi-label component must receive one shared
weight for all its samples/classes; use that exact weight vector for all views.
Do not generate independent per-view or per-class group draws.

Candidate, NOT executed/approved: draw e_g > 0 (e.g. Exp(1)) for each active group;
for view v and class c compute weighted loss

`L_vc^(b) = sum_{i:y_i=c} e_{g(i)}^(b) * [-log p_v(y_i|x_i)] / sum_{i:y_i=c} e_{g(i)}^(b)`

and `L_v^(b) = mean_{c in S*} L_vc^(b)`. Probabilities would use the unchanged
February temperatures. Normalize inside each class to preserve equal class weight.
The same positive weights preserve every class already in S*, but invent neither
new groups nor missing-class evidence. Numerically nonpositive/nonfinite draws
must trigger an explicit check; positivity in theory alone is insufficient.

Candidate U is the spread (standard deviation) of these quality replicates, not
the Monte Carlo standard error of their mean (do NOT divide it by sqrt(B)). Number
of replicates, prior/weight distribution, seed, diagnostic stability tolerances and
interpretation remain unapproved. No draws, quality scores or U were computed.

Unlike ordinary whole-group resampling, no group gets zero multiplicity, avoiding
its missing-class replicates. This is a DIFFERENT weighting scheme/assumption,
not the previously approved March metric-CI estimator. Do not describe the result
automatically as a frequentist confidence interval, Bayesian model posterior,
predictive entropy or epistemic uncertainty. It conditions on fixed CNNs,
temperatures, labels, support and this grouping; no training-seed/temporal coverage.

**Count-feasible, statistically limited:** candidate March classes have >=9 groups,
but concentration can be high. agent has 47 samples/9 groups; its largest group
contains 25 (53.19%) and size concentration count is 2.737. Other concentration
counts: dinwod 3.694, benjamin 4.487, autorun 4.631, sillyp2p 5.098, small 5.653.
These are descriptions, not independent ESS. Shared positive weights may expose
dominance but cannot cure it or prove low-U models generalize better. U may penalize
a view for difficult/support-poor families rather than genuinely lower reliability.
Require diagnostics and bounded claims; gamma is not selected here.

## Tests, artifacts and conclusions

Full CPU suite: **130/130 PASS**, 1.343s unittest runtime, including 10 new tests.
New tests cover graph edges/bridges/label invariance, retained scope, exact TNULL
handling, support boundaries, exact capacity against brute force, whole mixed-group
allocation, ID/class matching, semantic config hashes and CSV round-trip.
Real audit validates source hashes, six alignments, exhaustive graph, prior local
partitions and unchanged retained coverage. No old experiment was rerun or replaced.
The original failed byte-hash check produced no audit directory.

Artifacts: report.json; class_support.csv (51 families); groups.csv; members.csv;
edges.csv; source_consistency.csv; null_allocation_witness.csv;
null_witness_support.csv; scope_snapshot.md; checksums.json. All under the run above.
Source/input hashes are in report.json. Git does not back up ignored data/.

Scientific limits: one training seed, no honest historical prefix episodes;
operational similarity is not independence; S* covers only 25 classes for signals;
March was repeatedly observed and is now historical development; view architectures
differ; label availability before April is retrospective/UNKNOWN in deployment.
March is later than temperature fitting, but cross-month near-duplicate relationships
mean it is not an independent sample from the February calibration population. This
limits quality-transfer evidence even though no future label entered fitting.
No result here supports QDU superiority, causal drift effects or future robustness.

## Task 3–5 roadmap — requires new approval

| Task | Work and acceptance criteria | Approval gate |
| --- | --- | --- |
| 3: exact methodology | Decide S*, fixed-encoder layer/scaling/train reference, observed MMD budgets, cross-month dependence/null treatment, positive-weight U interpretation/diagnostics, coefficients and unavailable-signal rules. All four new arms share rules; no March score-based grid search disguised as validation. Specify finite resource budget. | Owner approves a written executable protocol; unresolved null design must be explicit, not hidden by epsilon or dropped classes. |
| 4: bounded historical implementation | Only after approval: batch feature extraction, known-answer signal tests, finite/support/provenance checks; compare deterministic saved embeddings across batch sizes; calculate historical Q/D/U and freeze all four new weights with hashes. No Future Test. | Separate inference/resource approval; stop on failed support or degenerate normalization; no automatic fallback. |
| 5: research readiness/freeze | Audit identical historical budget and final arms, costs, predeclared primary/secondary metrics, multiplicity/uncertainty, primary/sensitivity duplicate policy and limitations. Freeze protocol and weights before any future outcomes. | Explicit owner authorization to open Future Test, separate from approval to implement historical D/U. |

Recommended next action is Task 3 methodology design only, resolving dependence and
small null budgets first. Phase 1 is complete; D/U implementation remains stopped.
