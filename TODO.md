# TODO

## Evaluation Branch

### Active

- Extend startup device inventory into daemon-owned placement and scheduler
  routing:
  - `hipfire-model` now owns a shared accelerator inventory/device contract,
    `/health.runtime_workers` reports the shared inventory payload, and the
    daemon exposes a typed JSONL `inventory` command;
  - server health polls daemon inventory when a daemon is already running and
    otherwise reports `source=not_probed`, while worker identity carries
    accelerator kind/device id;
  - priority prefill scheduler admission can now consume accelerator inventory
    and reject sessions targeting missing or unavailable worker devices;
  - daemon inventory now includes explicitly configured XDNA1/NPU rows from the
    NPU module runtime/artifact contract;
  - Rust server now has a daemon-inventory-backed `PriorityPrefillScheduler`
    construction seam, and `/health` shares the same inventory probe helper;
  - replace the XDNA1 env/artifact sentinel with a real XRT hardware probe when
    NPU runtime dispatch owns that boundary;
  - wire live server request scheduling to pass daemon inventory into scheduler
    admission at the request queue site instead of using a single
    server-selected HIP placement.
- Regenerate quality KLD references as first-party HFQM `.kldref.hfq`
  packages. Do not trust previously downloaded raw `.kldref.bin` files for
  baseline claims. Regeneration must use Hipfire reference execution with
  `--kv-mode fp32` and FP32 DeltaNet state, and metadata must record the source
  model hash, slice hash, KV mode, state precision, `top_k`, context length,
  and producer command.
- Unblock full KLD regeneration throughput. The current first-party producer
  emits correct metadata and can produce smoke `.kldref.hfq` packages, but the
  0.8B BF16 `top_k=256` path still runs at roughly 20 scored tokens/s on the
  full 2048-token slice shape. Add a GPU-side `top_k=256` reducer or a faster
  BF16/F32 `lm_head` path before replacing the legacy raw refs.
- Keep model-backed profile collection in the eval harness. The `profile`
  battery should run a real Hipfire model-backed anchor and ingest runtime
  evidence artifacts, especially `moe_router_histogram.json` for MoE/A3B
  models.
  - The profile row now declares the expected runtime evidence kinds
    (`performance`, `memory`, `launch_counts`, `moe_router_histogram`) so
    artifact collection regressions stay visible in row metadata.
  - Generic runtime oneshot evidence writers and sparse router-histogram
    artifact rendering now live in `hipfire-evidence`; model-specific gatherers
    adapt their native counters into the shared histogram contract.
  - `GenerateTextRequest.evidence_dir` lets daemon-backed AR text generation
    write standard runtime oneshot artifacts directly via `hipfire-evidence`;
    DFlash/MTP/VL evidence emission remains a path-specific follow-up.
- Run the full no-GPU handoff gate before committing the branch.
- Extend Qwen35 paged expert execution coverage beyond the current indexed MQ
  routed path: grouped Path 2, Paro, full-precision, and CPU-fallback paged
  paths still need explicit tests before broad admission.
- Extend routed-only Qwen3 MoE forward validation beyond the current one-token
  modular/lazy expert smoke: multi-token prefill, decode continuation, eager vs
  paged parity on a smaller artifact, and grouped routed batches still need
  explicit coverage before removing the guarded execution flag.
- Convert eval execution from “examples sub-process fan-out” toward daemon-backed,
  reusable model processes before the next modularization step:
  - The user-visible symptom is repeated model loading in `--tier fast` runs:
    same model path is reloaded in separate subprocesses for smoke/coherence/
    speed/pflash/dflash rows because examples executor launches one process per row.
  - Evidence from latest run:
    - command line `hipfire-eval --model <hfq> --tier fast` emitted multiple
      `loading weights: ... HFQ payload` blocks in logs for the same model.
    - per-row commands in results showed duplicated `examples/run` plus speed cases.
  - Root cause hypothesis:
    - evaluator still routes key batteries to `examples` binaries (new subprocesses)
      not a resident daemon/service boundary.
    - no shared model lifecycle exists across battery rows; every row independently
      executes `run`/`run --session-reset-smoke`/`bench_qwen35_speed` etc.
  - What to finish to align with daemon/lib+IPC direction:
    - finish `EvalClient` transport in eval harness with request batching for same
      model across batteries where semantics allow;
    - daemon-backed smoke model-load, finite greedy decode, repeated greedy
      reset/recall, and explicit `--executor daemon --battery speed` timing
      anchors now use the shared JSONL daemon adapter; examples-backed pp32/pp128
      speed-gate rows remain the benchmark-grade default;
    - default `--executor auto` now prefers daemon-backed smoke/speed rows and
      shares one daemon load across those rows when they run together;
    - add process and handle ownership so one model-load maps to one resident daemon
      process per loaded quant/placement;
    - make battery rows consume a shared model cache key and avoid repeated
      `hfq` opens.
  - Success criteria:
      - `--tier fast` on a single model produces a single model-load event per
        daemon-backed model per run.
      - row metrics still preserve current throughput correctness and hard-fail
        semantics.
      - no functional regression in `--battery speed`, while retaining the
        command-level `--prefill-list` optimization already in `bench_qwen35_speed`.

- [Documentation debt] Refresh docs drift where active behavior has changed:
  - `docs/CHAT.md` is missing the `/set <key> <val>` command that `cli/chat.ts`
    supports for live session parameter updates.
  - TriAttention sidecar examples and naming in docs should consistently use
    `.triattn.hfq` as canonical; `.triattn.bin` is allowed only for explicit
    legacy compatibility.
  - `docs/QUANTIZATION.md` still claims MQ3 prefill is non-WMMA in places; the
    runtime now contains WMMA prefill paths for MQ3 on supported RDNA3/4 targets.
  - `docs/env-vars.md` is behind the current source surface and should be
    regenerated from `./scripts/regen-env-vars-doc.sh` after this cycle.

### Deferred

- `crates/hipfire-runtime/src/kv_adaptive.rs` — adaptive KV precision downshift
  module, written but not yet wired in. Missing pieces before use:
  - `KvCache::transcode_v_step` and `transcode_k_step` methods (called by
    `maybe_downshift`) are not yet implemented.
  - Module is not declared in `crates/hipfire-runtime/src/lib.rs`.
  - `Conservative` and `Aggressive` presets share the same floor values
    (both Fwht2/Lloyd2) — likely a placeholder; floors should differ.
  - No integration site in the decode loop (`maybe_downshift` must be called
    after each committed token write alongside `maybe_evict`).

- Adapt the build system to autodiscover GPUs the same way NPU detection now
  works: query HIP/ROCm at `cargo build --features npu-kernels` time (or a
  new `gpu-kernels` feature) to identify installed GPU arch(s) and select
  kernel tuning parameters automatically, rather than requiring the user to
  pass an explicit arch flag. The NPU precedent is `detect_npu()` in
  `tools/npu/build_qwen35_swiglu.py` + `HIPFIRE_NPU_TARGETS` env override in
  `crates/hipfire-arch-qwen35/build.rs`.
- Finish full daemon-backed `hipfire bench` replacement after eval-backed speed
  rows match the current public output shape.
- Promote long-context, vision, CASK/TriAttention, DFlash resident, cold-process
  distribution, and Kernel Atlas artifact ingestion from explicit skipped or
  external-evidence rows into native model-backed eval batteries.
- Extend host capability profiling beyond the current GPU/storage/memory report
  to measure NPU bandwidth paths when an NPU is present, and store measured
  bandwidth alongside static hardware metadata in eval reports.
- Migrate imatrix, CASK/TriAttention, DFlash sidecars, and other non-weight
  analysis packages into metadata-rich HFQM containers after the KLD reference
  package format is settled.

## PFLASH Review Debt (migrated from MANUAL_REVIEW)

- Investigate and close the remaining long-context pflash score kernel regression:
  - `qwen3.5-4b` + drafter paths currently exhibit `ScoringDegenerate { non-finite
    scores: 337 NaN, 0 inf }` at 32K NIAH source (`21551` tokens) despite 8K/16K pass.
  - Work to isolate root cause (drafter forward numerical blow-up vs score kernel)
    and add targeted diagnostics in `pflash::compute_scores_batched_gpu`.
- Keep the historical full-coherence/speed hang trace in the historical docs index
  as an execution quirk note, and re-run the gate from a clean environment when
  the next session reset is available.

## FWHT Residual QJL Transform

Status: deferred.

- Implement a Johnson-Lindenstrauss / QJL transformation on the residual in the FWHT path. The current FWHT path applies a signed-FWHT rotation to Q/K for attention and leaves the residual stream without a separate QJL transform.



## Check all hot paths for graph safety
>>> One issue surfaced before verification: gemm_f16_x_f32_wmma currently launches with raw stack kernargs rather than the graph-safe blob helper. I’m tightening the env gate so this experimental route only runs outside hipGraph capture; captured paths will keep using the scalar default until the dispatcher wrapper is made graph-safe.


## further investigate using packed 4 bit operations on gfx1151/RDNA3/RDNA3.5

## DeltaNet state precision (follow-ups to Phase A gate, 2026-06-15)

Phase A made the DeltaNet recurrent state default to **FP32** for all
current models, gated on redundancy (`head_dim × n_heads`) via
`qwen35::default_state_quant` (env: `HIPFIRE_DN_STATE_QUANT`,
`HIPFIRE_DN_STATE_FP32_BELOW`). Two kernels are missing to generalize it:

1. **Real FP16 DeltaNet state kernel.** `StateQuant` is only
   `{FP32, Q8, Q4}`; there is no `gated_delta_net_f16`. The plan wants
   FP16 for high-redundancy models (cheaper than FP32, safer than Q8),
   but it doesn't exist, so the gate currently selects FP32 for the whole
   upper tier (threshold `usize::MAX`). To finish:
   - add `StateQuant::FP16` + alloc path in `DeltaNetState::new_with_quant`;
   - add `gpu.gated_delta_net_f16(...)` + `_batch_seq` / `_routed_batch_seq`
     dispatch (mirror the FP32 kernels);
   - coherence-gate the f16 state numerics, then lower the default
     `HIPFIRE_DN_STATE_FP32_BELOW` so big models use FP16.

2. **Higher-precision tree DeltaNet replay (FP32 *and* FP16).** Tree-mode
   spec-decode hard-errors on FP32 state (`qwen35.rs`: *"FP32-state batched
   prefill does not support tree DeltaNet replay yet"*); only
   `gated_delta_net_q8_tree_batch_seq` exists. Consequence: with the FP32
   default, tree-based spec-decode (DDTree, DFlash-tree,
   `spec_step_dflash_mtp_tree`) cannot run — Phase B MTP drafting must use
   the **non-tree** `spec_step_mtp` path. Future work, both precisions:
   - `gated_delta_net_f32_tree_batch_seq` — FP32 S-tape replay, unblocks
     tree drafting at full anchor precision (serves the FP32-forced
     low-redundancy models like 0.8B directly).
   - `gated_delta_net_f16_tree_batch_seq` — FP16 S-tape replay, pairs with
     the FP16 state kernel (item 1) so high-redundancy models can run tree
     drafting at the cheaper FP16 tier once FP16 coherence is established.
   Until either lands, tree drafting needs `state_quant=q8` (only safe on
   high-redundancy models, never 0.8B/2B).

## Native Rust/GPU Hessian collector (finish the collect_hessian.rs scaffold)

The QTIP LDLQ path (Phase C1e) needs per-layer input Hessians. The native
`crates/hipfire-runtime/src/bin/collect_hessian.rs` is currently a SCAFFOLD
that panics ("not implemented"); we fall back to the Tier-2 Python collector
`scripts/collect_hessian.py` (HF transformers + ROCm/CPU torch). The Python
path WORKS (validated 2026-06-16: 0.8B → 2.21 GB HFHS file, read by
`hessian_io.rs`), so the HFHS-v1 format + approach are proven.

Implement the native Tier-1 collector (the scaffold's own TODO list) so we
get GPU speed and drop the torch dependency:
- BF16 safetensors → device tensors loader.
- On-GPU K×K outer-product Hessian rank-1-update kernel (accumulate
  H += xᵀx over calibration tokens per GPTQ-target linear).
- `ActivationCapture` wiring at the GPTQ-target dispatch sites in the
  qwen35/generic forward (capture each linear's input activations).
- HFHS-v1 binary writer matching `scripts/collect_hessian.py` (so
  `hessian_io.rs` reads it unchanged).

Benefit: CPU torch is slow (0.8B calibration is minutes; bigger models
hours); the GPU forward + on-GPU accumulation would be far faster, and it
removes the Python/torch tooling dependency from the quant pipeline.
Reference: the validated Tier-2 `scripts/collect_hessian.py` + the existing
scaffold's documented deliverables.
