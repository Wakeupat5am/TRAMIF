# D054 — Final five-seed historical Q/D and frozen weights

All15 checkpoint artifacts and5 historical seed pipelines are ARTIFACT_VERIFIED.
Numbers below describe fixed historical artifacts; no future performance,
significance,independent temporal replications or QD superiority is inferred.

Order of views:SBSMI,raw-byte,entropy. T is fitted on February3799 only.
L is March retained B/S25 class-balanced NLL on4064 samples,with all51 outputs;
smaller L is better. d=MMD2/2 is descriptive and not null-standardized.
Q weights=softmax(-L);QD weights=softmax(-L-d);beta1,gamma0,U excluded.
All five seeds included,no best-seed selection. Full precision is retained in
`data/derived/final_multiseed_verification/20261017_164739_728045/summary.json`.

## Per-seed/view results

| Seed | View | T | L | MMD2 | d | Q weight | QD weight | Checkpoint/processing |
|---|---|---:|---:|---:|---:|---:|---:|---|
|42|sbsmi|2.520203|0.755550|0.044394|0.022197|0.407902|0.409306|VERIFIED/VERIFIED|
|42|raw-byte|2.211873|1.068054|0.052227|0.026113|0.298427|0.298283|VERIFIED/VERIFIED|
|42|entropy|1.935490|1.084118|0.059855|0.029927|0.293671|0.292412|VERIFIED/VERIFIED|
|43|sbsmi|2.485598|0.730279|0.044291|0.022145|0.421055|0.422467|VERIFIED/VERIFIED|
|43|raw-byte|2.126724|1.106119|0.054884|0.027442|0.289144|0.288581|VERIFIED/VERIFIED|
|43|entropy|1.896896|1.103847|0.056860|0.028430|0.289801|0.288952|VERIFIED/VERIFIED|
|44|sbsmi|2.451075|0.792358|0.043322|0.021661|0.408639|0.409908|VERIFIED/VERIFIED|
|44|raw-byte|2.151911|1.088823|0.053330|0.026665|0.303799|0.303222|VERIFIED/VERIFIED|
|44|entropy|1.920150|1.143752|0.054342|0.027171|0.287562|0.286870|VERIFIED/VERIFIED|
|45|sbsmi|2.444445|0.717882|0.042762|0.021381|0.411772|0.413291|VERIFIED/VERIFIED|
|45|raw-byte|2.180539|0.980110|0.053580|0.026790|0.316791|0.316244|VERIFIED/VERIFIED|
|45|entropy|1.902391|1.134623|0.057302|0.028651|0.271437|0.270465|VERIFIED/VERIFIED|
|46|sbsmi|2.418638|0.708261|0.043179|0.021590|0.421601|0.423156|VERIFIED/VERIFIED|
|46|raw-byte|2.205135|1.049071|0.055783|0.027891|0.299841|0.299056|VERIFIED/VERIFIED|
|46|entropy|1.895933|1.122695|0.056074|0.028037|0.278558|0.277788|VERIFIED/VERIFIED|

## Descriptive statistics across seeds

Computed separately per view over all five seeds,using full precision;SD is sample
SD(ddof=1). No grouping of views as independent observations,no confidence intervals.
These summarize historical estimates/weights,not future prediction accuracy.

| View | Metric | N | Mean | SD | Min | Max |
|---|---|---:|---:|---:|---:|---:|
|sbsmi|temperature|5|2.463992|0.039473|2.418638|2.520203|
|sbsmi|L|5|0.740866|0.033802|0.708261|0.792358|
|sbsmi|mmd2|5|0.043590|0.000718|0.042762|0.044394|
|sbsmi|d|5|0.021795|0.000359|0.021381|0.022197|
|sbsmi|Q_weight|5|0.414194|0.006676|0.407902|0.421601|
|sbsmi|QD_weight|5|0.415626|0.006738|0.409306|0.423156|
|raw-byte|temperature|5|2.175237|0.035923|2.126724|2.211873|
|raw-byte|L|5|1.058435|0.048765|0.980110|1.106119|
|raw-byte|mmd2|5|0.053961|0.001390|0.052227|0.055783|
|raw-byte|d|5|0.026980|0.000695|0.026113|0.027891|
|raw-byte|Q_weight|5|0.301600|0.010049|0.289144|0.316791|
|raw-byte|QD_weight|5|0.301077|0.010034|0.288581|0.316244|
|entropy|temperature|5|1.910172|0.017176|1.895933|1.935490|
|entropy|L|5|1.117807|0.024029|1.084118|1.143752|
|entropy|mmd2|5|0.056887|0.002007|0.054342|0.059855|
|entropy|d|5|0.028443|0.001004|0.027171|0.029927|
|entropy|Q_weight|5|0.284206|0.009043|0.271437|0.293671|
|entropy|QD_weight|5|0.283297|0.008984|0.270465|0.292412|

## Interpretation

The fixed Q/QD formulas consistently allocate the largest weight to SBSMI in
these five historical runs. That does not prove intrinsic representation
superiority:architectures differ in capacity,and neither weights nor historical
losses establish future fusion performance. QD only modestly changes Q weights.
Do not average these weights across seeds and replace the registered per-seed
methods;each seed retains its own checkpoint,T,normalization,bandwidth and signals.

Mean/SD/min/max are descriptive training-seed variation on the same data. No seed
was excluded. E2 negative fusion evidence and two-overlapping-episode limitations
remain. U is still diagnostic-only;descriptive drift is not statistically
standardized. Full V1 remains blocked by the missing learned-static baseline,
statistical approval/coverage,scope controls and protected Future work.

Verification details/checkpoint paths/hashes:docs/FINAL_MULTISEED_VERIFICATION.md.
Independent stdlib readback confirmed all5 output hashes,15 rows,18 descriptive
rows,sample SD and Python-math softmax weights. No experiment was rerun.
