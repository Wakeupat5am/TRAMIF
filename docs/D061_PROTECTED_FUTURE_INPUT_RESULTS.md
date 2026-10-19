# D061 — Protected Future input preparation

Status: **READY_FOR_FINAL_SCORING — separate owner authorization required.**
Real-data input/inference integration PASS; Future prediction performance UNKNOWN.
No final scoring, method comparison, training, fitting, retuning, commit or push.

## Contract and completeness

Owner explicitly authorized April–September preparation/inference using D060.
D060 bundle, references, 15 checkpoint hashes, dependencies and canonical K51
mapping verified before execution and frozen contract rechecked at completion.
All known-family binaries exist and are nonempty. All selected timestamps are UTC;
IDs are unique canonical 64hex. Metadata includes benign and out-of-K records:

| Month | K51 malware | Other-family malware | Benign |
|---|---:|---:|---:|
| 2020-04 | 4066 | 574 | 5201 |
| 2020-05 | 4352 | 1097 | 6121 |
| 2020-06 | 3514 | 703 | 8182 |
| 2020-07 | 3090 | 1905 | 6392 |
| 2020-08 | 2822 | 1001 | 2514 |
| 2020-09 | 2989 | 1588 | 4198 |

Totals: 20,833 K51 malware, 6,868 other-family malware, 32,608 benign.
Original BODMAS README defines empty family as benign; the disarmed archive only
provides malware binaries. Benign is not a missing known-family input. Other-family
malware availability was checked, but content hashing/similarity is restricted to
the frozen K51 study cohort. This is not benign detection or open-set evaluation.
All 20,833 known-family TLSH hashes are VALID; TNULL=0. Exact-content duplicate
or conflicting-family groups touching Future within the frozen K51 scope: 0.

## Retention

| Month | retention0 kept/excluded | retention10 kept/excluded | retention20 kept/excluded |
|---|---:|---:|---:|
| 2020-04 | 4032/34 | 1962/2104 | 1597/2469 |
| 2020-05 | 4311/41 | 2318/2034 | 1909/2443 |
| 2020-06 | 3483/31 | 2414/1100 | 2221/1293 |
| 2020-07 | 3058/32 | 1990/1100 | 1807/1283 |
| 2020-08 | 2686/136 | 1493/1329 | 1324/1498 |
| 2020-09 | 2793/196 | 1625/1364 | 1423/1566 |
| Total | 20363/470 | 11802/9031 | 10281/10552 |

Uses frozen direct-earlier content-equality/TLSH policy, including excluded earlier
records as references; no family label decides similarity. Sensitivity sets are
nested primary subsets. Witness links are not independent statistical components.
All 20,053 exclusion decisions across thresholds independently checked, plus
36 deterministic KEEP queries against every earlier reference (12/threshold).
All six primary months nonempty. No source binary/image was deleted.

## Image/logit integration and evidence

- 20,363 primary samples; 61,089 images verified, exactly 64x64, one channel.
- Frozen SBSMI PNG L; raw-byte/entropy float32 NPY with original [0,1] scaling.
- 15/15 full K51 raw-logit arrays, each (20363,51), finite float32.
- Exact canonical IDs, labels, class order and three-view/five-seed alignment PASS.
- Own-seed checkpoint and February-temperature bindings PASS. Temperatures were
  neither applied to the saved raw logits nor refit; fusion is not evaluated here.
- Frozen inference: eval/inference_mode, gradients disabled; repeated first batch
  exact; model parameters and BatchNorm buffers unchanged.
- TRAIN-only size cutoffs preserved: 86414/254672/721840 bytes; not fit on Future.
- 12/12 targeted synthetic tests PASS. Their source hashes match the run snapshot
  and current code. They are not substitutes for real-data integration evidence.
- Independent real readback PASS; receipt indexes 122,235 files with SHA-256.
- Elapsed preparation/inference/readback: 18773.816 seconds (~5h13m).

Run: `data/derived/protected_future_inputs_v1/20261018_082841_011123/`.
Completion: `receipt.json`, SHA-256 `116f994f5b3bf4372de248fe34b855e01c87f037955795a8156fa34d433ac771`.
Independent verification: `independent_verification.json`, SHA-256
`d07b2c3e0e4a47f92c68668e0ae68f5f5ddfa92aae09d46d6ebc32f07d874844`.
Also: `config.json`, `all_future_metadata.json`, `known_inventory.json`,
`primary_manifest.json`, `retention_{0,10,20}.json`, `retention_counts.json`,
`images_manifest.json`, `IMAGES_VERIFIED.json`, `logits_manifest.json`, source
snapshots, `targeted_tests.json`, `source_scope_notes.json` and
`content_identity_quality.json`. Final terminal readback confirmed summary/manifest
hashes and current/source-snapshot equality without repeating inference.

## Preserved failures and limitations

Two initial discovery failures are retained at sibling runs
`20261018_082625_209316` and `20261018_082736_022585`. Adapter validation initially
interpreted out-of-period benign metadata (naive timestamps/noncanonical IDs).
Filtering the authorized period before strict validation fixed that code path;
regression test added. No actual Future row was repaired or dropped to pass.

The original binary ZIP was deleted earlier. Current bytes are fingerprinted and
cross-view checked, but authoritative original-ZIP CRC authentication is unavailable.
Bounded KEEP readback is not an independent exhaustive recomputation of all pairs.
TLSH does not prove statistical independence. Off-device backup of this new input
package is not verified; D060 local backup does not include these new outputs.
No metric, CI, p-value, method contrast or performance superiority is reported.

## Stop decision

No remaining input/inference blocker was observed. Separate owner authorization is
required for frozen final scoring (seven arms/five seeds, primary and sensitivity
sets, TRAIN-size strata, exact D060 statistical specification). Do not launch D062
automatically. Preparation PASS is not RESEARCH VALIDATED or PROJECT_COMPLETE.
