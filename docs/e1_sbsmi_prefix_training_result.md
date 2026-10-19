# E1-SBSMI — First official prefix training result

2026-10-16 (Asia/Saigon). **Technical completion VERIFIED; 100/100 epochs.**
Owner authorized exactly one E1-SBSMI run, not the remaining five CNNs.

Readiness before launch: ACLineStatus1; PowerOnline=True, Discharging=False,
Charging=True; CUDA available, RAM6.17GiB, VRAM6.94GiB, disk81.66GiB available.
Receipt/source/G2 bindings PASS. Read-only mid-run snapshot also confirmed AC
online, not discharging, disk above2GiB; no continuous power-monitor guarantee.

Run: `data/derived/prefix_training_v1/E1/sbsmi/20261016_012854_728594/`.
Frozen E1 TRAIN Aug–Oct2019: 7034 samples, K32, exclusive cutoff2019-11-01 UTC.
Fresh initialization, seed42, batch32, unweighted CE, Adam lr1e-4/decay1e-6,
no scheduler/augmentation/warm-start/resume. Fixed100 epochs; epoch100 selected
prospectively, not from accuracy or any A/B/C outcome.

| Result | Value |
| --- | --- |
| Process exit | 0 |
| Epochs / steps | 100 / 22000 |
| Samples per epoch / batches | 7034 / 220 |
| Sample presentations | 703400; not unique samples |
| Actual process wall time | 2703.6048804s = 45min03.60s |
| Summed epoch loop time | 2631.6848673s = 43min51.68s |
| Epoch100 online training loss | 0.07478166174780146 |
| Epoch100 online training accuracy | 0.9764003411998863 = 97.6400% |
| Max logged GPU allocated / reserved | 126.57470703125 / 148 MiB |
| Logged warnings/errors | None found in captured console |
| FAILED.json / COMPLETE.json | absent / present and validated |

GPU memory metrics are PyTorch allocator measurements, not total WDDM/process
VRAM. Online training accuracy combines predictions made while weights evolve;
it is not fixed-checkpoint evaluation or evidence of future generalization.
Actual time exceeds the prior17.14min E1-SBSMI loop projection. No hyperparameter
was changed to meet that estimate; no unsupported attribution of the slowdown.

Selected checkpoint: `epoch_100.pt`.
SHA-256: `2aec068ba4c55dbd25e56b9a415fc103bce1288ec3be8b6ccfaf6bff60baef4a`.
Config canonical SHA: `85390300c67e480d279342830bfccb1edf6a4c6509ae6095fb5f1237c40c9da1`.
COMPLETE SHA: `9708d235c5a5a552fdc1d39628f45dcbb67f1e633a6c75f7a236086dd93f96e6`.

Post-run verification: all100 checkpoint hashes match epoch records/COMPLETE,
consecutive epochs1..100, every record7034 samples/220steps, finite diagnostics;
selected record exactly epoch100. CPU weights-only load of this NEW prefix
checkpoint verified embedded config/hash, classifier [32,2048], finite model
state and order hash. Mapping/cutoff/current source hashes match frozen bindings.
No registered Final checkpoint loaded; no forward/inference during verification.

Evidence directory:
`data/derived/e1_sbsmi_training_readiness/20261016_012823_662282/`:
quick_check.json, execution_start.json, training_console.log, execution_end.json,
monitor_01.json and completion_verification.json. No retry/resume or second run.

No A/B/C evaluation, embeddings, MMD/Q/U, Future Test, code/model/config change,
commit/push. Only E1-SBSMI is now technically complete; five prefix CNNs and
extractor integration remain outside this authorization. STOP for owner review.
