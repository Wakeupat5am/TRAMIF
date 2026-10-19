# D044 — Historical Quality results

2026-10-17 Asia/Saigon. **Engineering PASS**: implemented, unit tested and verified
on saved historical logits. Research usefulness for future prediction remains
unvalidated. No inference, temperature fitting, U, new Drift or fusion weights.

Frozen A→B: E1 November→December2019 (K32/S17), E2 December2019→January2020
(K42/S25), Final February→March2020 (K51/S25). Full primary-retained B used;
support selects samples only, not softmax outputs. No Drift cap.

Loss ell=-logsoftmax(logits/T)[label], float64, natural logs (nats).
L=equal-family mean of per-family mean NLL over fixed S; lower is better.
Q=-L, so higher Q is better. Std-S weights samples equally within S; Std-all
uses every retained B sample, including families outside S. These are different
estimands. T uses full-precision saved A-only calibration, never refitted on B.

| Episode/view | K/S | B in S / all | T | L balanced | Std-S | Std-all | Family mean min–max | Status |
|---|---|---|---:|---:|---:|---:|---|---|
| E1 SBSMI |32/17|3130/3480|1.798111|.450909|.790719|.799414|.028968–1.521931|PASS|
| E1 raw-byte |32/17|3130/3480|1.552267|1.111060|1.038725|1.047737|.207934–3.453693|PASS|
| E1 entropy |32/17|3130/3480|1.441774|.968686|.938057|.938691|.193367–2.256435|PASS|
| E2 SBSMI |42/25|3659/3929|1.956812|.752457|.728627|.749916|.016953–4.117790|PASS|
| E2 raw-byte |42/25|3659/3929|1.657095|1.162084|1.040138|1.075689|.140631–3.804750|PASS|
| E2 entropy |42/25|3659/3929|1.536168|1.105116|.978008|1.003791|.099849–4.901443|PASS|
| Final SBSMI |51/25|4064/4332|2.520203|.755550|1.046715|1.070787|.024273–3.954421|PASS|
| Final raw-byte |51/25|4064/4332|2.211873|1.068054|1.276609|1.302756|.045299–4.407652|PASS|
| Final entropy |51/25|4064/4332|1.935490|1.084118|1.278638|1.308594|.038815–4.079199|PASS|

SBSMI has the lowest observed L in each case. Entropy is below raw-byte for E1/E2;
their order reverses for Final. Differences between L and Std-S demonstrate the
effect of family weighting; neither metric can replace the other silently. Cases
have different checkpoints, K and support/cohorts: these rows alone do not isolate
temporal degradation, establish statistical superiority or justify fusion weights.

## Verification and reproducibility

New module reuses stable calibrated_log_probabilities and existing hash-bound
selection/checkpoint/decoder provenance helpers; no model loader or inference
function is called. Does not call legacy estimate_quality because that function
also creates fusion weights. Exact prefix receipt and checksum manifests are
pinned; saved selections match current immutable G2 A/B bindings. Final registry
is bound to the previously verified receipt. Current checkpoint, T, mapping,
retention, dates, group IDs and support bindings checked for all views. Explicit
ID join records source row indices before any NLL computation; no intersection,
positional assumption, support refit or output renormalization.

5/5 targeted tests PASS (0.018s): stable shifted NLL/T, full K versus S,
imbalanced aggregation, explicit ID alignment, missing/duplicate IDs, wrong
classes/labels, nonfinite inputs, absent support and invalid T fail closed.
Execution: `.venv\Scripts\python.exe -B -m scripts.compute_historical_quality`.

Artifacts: `data/derived/historical_quality_v1/20261017_005343_157189/`.
Completion SHA256:
`17e887d981568bdc79a2af11488590cdac3e4abc0ab6584e30ae9c10a9c9e270`.
Contains config, each case's canonical B manifest/mapping/support, each view's
sample_losses.npz (all B NLL, S mask, IDs/labels/source row indices), quality.json
(full-precision summaries, per-family counts/means and source paths), completion
receipt with97 source/input hashes, and checksums for23 output files.

Independent saved-output verification: all hashes, nine summaries and per-sample
losses checked against shifted-exponential reference, rtol/atol1e-12; aggregate
means checked to absolute1e-14. Receipt:
`data/derived/historical_quality_v1/20261017_005343_157189_verification/receipt.json`.

Next proposed: construct the New-Q-only historical baseline from these fixed
losses under the approved weight rule, preserving paired samples and information
budget. Separate approval is needed before creating weights or evaluating C.
U remains diagnostic-only and excluded from fusion. No C/Future access, commit
or push in this batch. All prior artifacts preserved.
