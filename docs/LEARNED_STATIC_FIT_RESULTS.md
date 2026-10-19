# D056 — Learned-static fitting stopped under approved protocol

2026-10-18. Owner explicitly approved the published uniform-start projected-gradient
solver,Armijo backtracking,max10000 iterations and residual<=1e-10 for five Final
seeds on March B/S25. The previous automatic-approval rejection is resolved by that
explicit authorization. One CPU invocation was executed;no fallback or retry.

## Actual result

| Seed | Status | Iteration | Projected residual | Required residual | Weights frozen |
|---|---|---:|---:|---:|---|
|42|STOP:Armijo line-search failed|26|1.9771081438690885e-9|<=1e-10|No|
|43|NOT ATTEMPTED|—|—|<=1e-10|No|
|44|NOT ATTEMPTED|—|—|<=1e-10|No|
|45|NOT ATTEMPTED|—|—|<=1e-10|No|
|46|NOT ATTEMPTED|—|—|<=1e-10|No|

Exception:`RuntimeError: Armijo failed; no empirical fallback` at
`scripts/learned_static_solver.py`,solve,line62. The failure occurs before the
10000-iteration ceiling:the line search cannot accept a step under its fixed
backtracking budget. Recorded residual does not satisfy the stopping contract.
No approximate candidate is accepted as a learned-static baseline.

The diagnostic objective at termination is0.7492304226946944. This is a same-data
March B/S fit objective,not independent evaluation or a significance result.
Diagnostic candidate weights in NOT_CONVERGED.json are explicitly unregistered;
they must not be used in inference,fusion,seed selection or a frozen package.
The cause of line-search failure is not diagnosed by the stored diagnostic alone.
Do not infer data corruption or alter numerical tolerances from these values.

## Evidence

Run:`data/derived/final_learned_static/20261018_000858_569331/`.
Files:config.json,selection.json,seed_42/fit_inputs.npz,
seed_42/NOT_CONVERGED.json,FAILED.json. No weights.json or successful receipt exists.
Failure readback receipt is recorded in automation/continuation_state.json.
It hashes all five run files and confirms unchanged source hashes,no registered
weights,no retry and no further-seed execution. Existing Q/QD artifacts unchanged.

Inputs use D054's pinned verification receipt and the D052 solver source hash.
Seed42 B predictions are explicitly joined by canonical sample ID. February T is
fixed,Softmax retains K51,March fit subset is4064 samples across frozen S25.
No T fit,inference,MMD computation,CNN training,Future access or learned-static
coefficient search occurred.11 targeted synthetic tests passed before the attempt;
they did not guarantee empirical numerical convergence.

## Stop decision

Learned-static baseline is INCOMPLETE. Preserve this negative engineering outcome
and all failure artifacts. Do not continue seeds43-46 or retry seed42,change
solver/Armijo/tolerance,or register the near-converged candidate without owner review.
Next proposed action:a focused numerical diagnosis/design review of the recorded
failure and unchanged solver,before requesting approval for any revised numerical
implementation or new fitting attempt. The objective/cohort/Q/QD need no change.
Full V1 is not ready;other statistical/scope/backup/Future restrictions remain.
