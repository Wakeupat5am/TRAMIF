# Historical research checkpoint — D049/D050

2026-10-17. **Engineering PASS; scientific evidence is limited historical development.**
Frozen D048 candidates0/.5/1, gamma0, primary descriptive MMD²/2 and A-only
temperatures were used unchanged. E1 selected beta1 by minimum forecast MAE;
the selection artifact was saved before E2 metadata/inference/evaluation.
No E2 selection was performed. No p-values, intervals, extra seeds or Future data.

## RQ2: next-window loss prediction

| Episode | K | C retained | S_eval | Q MAE | Selected QD MAE | Relative reduction |
|---|---:|---:|---:|---:|---:|---:|
| E1: January2020 |32|3,695|17|.471385198|.469167689|.4704%|
| E2: February2020 |42|3,563|25|.563763497|.561424341|.4149%|

E1 candidate MAEs beta0/.5/1: .471385198/.469933904/.469167689. No tie or missing
signal-support families. Equal-family mean absolute error averaged over three
views, not pooled-sample prediction accuracy. E1 is selection evidence; E2 is a
held-forward historical check, with overlapping training/months. These two points
cannot establish a general predictive mechanism or statistical superiority.
The penalty only predicts loss increases. Size/composition-adjusted evidence is
incomplete; do not attribute residual differences causally to representation drift.

## RQ3: six-method outcome point comparisons

| Episode | Method | Accuracy % | Supported Macro-F1 | NLL | Brier |
|---|---|---:|---:|---:|---:|
| E1 | SBSMI |73.3965|.686237|.815196|.379140|
| E1 | Raw-byte |63.3559|.552131|1.163046|.464961|
| E1 | Entropy |67.1177|.590746|1.084235|.458185|
| E1 | Uniform |72.1516|.679846|.785493|.365722|
| E1 | New Q |73.8566|.694878|.760230|.354726|
| E1 | QD beta1 |73.8566|.695034|.760346|.354771|
| E2 | SBSMI |77.7154|.670809|1.231740|.340308|
| E2 | Raw-byte |63.0368|.540895|1.654942|.468590|
| E2 | Entropy |63.5420|.557511|1.614566|.476761|
| E2 | Uniform |77.2383|.670044|1.278282|.369560|
| E2 | New Q |77.9961|.683443|1.255607|.356922|
| E2 | QD beta1 |77.9961|.683434|1.255891|.357082|

New Q exceeds Uniform by1.5032/1.3399 percentage points of supported F1 in E1/E2.
QD versus Q: E1 F1 +.015557pp; E2 F1 -.000818pp. Accuracy is unchanged in both;
NLL/Brier worsen slightly in both. E2 SBSMI has lower NLL/Brier than both fusions.
Do not summarize this as winning every baseline. E2 has40 true-supported classes
out of42 outputs; fixed-K F1, balanced accuracy, ECE15, confusion matrices and
per-class metrics are retained in report.json/CSV. No denominator changes.

## RQ1 and U: existing evidence, no new signal search

Primary descriptive MMD² (SBSMI/raw/entropy): E1 .042142/.033769/.036465;
E2 .047514/.036424/.037929; Final .044394/.052227/.059855.
Fixed bandwidth h/2 and2h preserve within-case order but change magnitudes by
roughly+49–57%/-41–43%. Cap64 changes scores+2.33–10.60%; equal represented-group
mass lowers them27.37–35.66% and reverses E2 raw/entropy order. This concerns
encoder/kernel/sample targets, not intrinsic view stability or standardized drift.
No sensitivity winner replaces the primary. Group dependence/concentration remains.

U simulation is negative methodological evidence:12/21 mean-U/empirical-SD ratios
below.8, including unequal independent groups; maximum mean underestimation48.5%.
U remains diagnostic-only; QU/QDU unavailable. No additional synthetic search.

## Evidence and reproducibility

- E1:`data/derived/historical_episode_evaluation/E1/20261017_033742_471099/`.
  Report SHA256:`06c988e1ea17ae2214dc711732ea2c9ed008f91a53a93f3dec7c2af4c71e1a98`.
  beta_selection.json SHA256:`a36ec42417923921739f455814d8a41e41a9937fb8789ce014794236802820c2`.
- E2:`data/derived/historical_episode_evaluation/E2/20261017_033806_789235/`.
  Report SHA256:`e9bb1a06628f95b207505eb7859d57c40c44accaabf703422f80d5ec5c10e9c7`.
- Independent verification receipts under `data/derived/historical_episode_verification/`.
  Seven metrics independently recomputed for all six methods, both cases;
  all arrays/hashes/forecast arithmetic PASS (absolute tolerance1e-12 metrics).
- Focused20-test suite PASS,0.024s (evaluation6, selection5, mixture5, dormant
  Future point/guard4). No full-suite count inferred.
- Tables:`data/derived/historical_research_summary/20261017_034235_417680/`:
  method_metrics.csv(12 rows), forecast_mae.csv(4), paired_point_differences.csv(6).
  historical_metrics.vl.json is a standard Vega-Lite figure specification only;
  not rendered because plotting libraries are unavailable. No package installed.

Outputs include full probabilities/log probabilities, raw E2 logits, residuals,
frozen selections, hashes and failures. Current results do not answer full V1's
five-seed/six-Future-month primary outcomes. Learned-static solver/tie-break and
broader capacity/cost baselines remain unapproved/incomplete. No SOTA, efficiency,
causal or significance claim follows. Future processing remains closed.
