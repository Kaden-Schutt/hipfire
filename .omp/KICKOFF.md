# DS4 MI300X/gfx942 overnight campaign — kickoff

Read `.omp/GOAL.md`, `.omp/DS4-INVARIANTS.md`, `.omp/SYSTEM.md`, `CLAUDE.md`,
`AGENTS.md`, `docs/VALIDATION.md`, `docs/QUANTIZATION.md`, `docs/QUANTIZE.md`, and
`docs/REDLINE.md` before changing code or launching remote work. `GOAL.md` is
authoritative. This kickoff does not weaken any gate.

## First action: create the persistent goal

Call the OMP `goal` tool with `op=create`, no token budget, and this exact objective:

> On MI300X/gfx942, first maximize coherent product-path prefill and batch-1 ordinary
> AR decode of the existing DeepSeek V4 Flash preview MQ4R artifact without changing
> its weights or serving semantics; second, only after A is frozen, quantize
> deepseek-ai/DeepSeek-V4-Flash-0731 into the same MQ4R format and quality-max it with
> streamed Hessians/GPTQ plus same-model KLD/PPL and coherence proof; third, only
> after A and B are frozen, wire and validate DS4-scoped long-context Q8 and
> FWHT3-K/Q8-V paths on gfx942.

If an unfinished goal already exists with the same objective, resume it instead of
creating a duplicate. Build a todo list mirroring G0, A, B, and C. At most one gate is
`in_progress`; G0 precedes A, A precedes B compute, and B precedes C.

## Immediate execution order

1. Confirm `pwd`, branch, commit, dirty diff, `.omp` profile, and that the protected
   source worktree will remain read-only.
2. Create the local ledger/evidence index. Have `ds4-mi300x-operator` perform the
   remote identity, `gfx942`, HBM, process, and 5 TB scratch preflight before any GPU
   work.
3. Dispatch the first read-only batch concurrently, naming agents explicitly:
   - `ds4-state-census`
   - `ds4-forward-map`
   - `ds4-quant-rotation-contract`
   - `ds4-shape-census`
   - `ds4-attn-recurrent`
   - `ds4-dispatch-build-firewall`
   - `ds4-validation`
   - `ds4-model-delta-0731`
4. Once hardware/model identities are returned, let the operator resume exactly one
   background 0731 download/verification job if needed. This is logistics only; B
   computation remains gated. Use the idle time for source/ISA analysis.
5. Dispatch the A design batch concurrently:
   - `ds4-cdna-isa`
   - `ds4-compiled-isa`
   - `ds4-clang-builtins`
   - `ds4-mq4r-mfma`
   - `ds4-mq4r-packing`
   - `ds4-e8-mfma` and `ds4-e8-packing` only for tensor classes proven present
   - `ds4-rotate`
   - `ds4-prefill`
   - `ds4-ar-decode`
   - `ds4-occupancy`
   - `ds4-roofline`
6. Do not dispatch `ds4-mq2-mfma` or `ds4-mq2-packing` unless the artifact census
   proves the current model is not MQ4R. Do not dispatch B quant agents before A exit
   or `ds4-kv-fwht3` before B exit.
7. Reconcile the read-only reports with `ds4-first-divergence` and
   `ds4-adjudicator`. Freeze the first A implementation slice, run GitNexus impact,
   then assign exactly one of `ds4-compose-hip`, `ds4-compose-rust`,
   `ds4-compose-integration`, or `ds4-compose-tests`.
8. Give all remote build/test/profile/bench commands to `ds4-mi300x-operator` only.
   Preserve every artifact under the selected scratch evidence root and ledger the
   verdict before the next lever.
9. Repeat the measured A loop until the A exit criteria are satisfied. Then freeze A
   and dispatch `ds4-quant-calibration` plus `ds4-kld-hessian` for B. After B exits,
   dispatch `ds4-kv-fwht3` for C.

## Continuation behavior

Do not stop after writing a plan or after the first agent wave. Continue implementing,
testing, measuring, rejecting, and promoting against `GOAL.md`. Use async/background
jobs for long downloads and builds, poll them, and fill waits with eligible read-only
work. Never duplicate an expensive job simply because it has not emitted output.

Do not ask the user for routine technical choices. Pause only for a genuine authority,
credential, destructive-action, missing-source, or hardware blocker. In that case,
write the exact blocker to the ledger and leave the goal active. Otherwise, either the
campaign is still working when the user returns or all three gates are genuinely
complete.

