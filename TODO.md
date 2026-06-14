# TODO

## Evaluation Branch

### Active

- Extend startup device inventory into daemon-owned placement and scheduler
  routing:
  - server startup now exposes basic accelerator inventory and worker identity
    carries accelerator kind/device id;
  - enumerate all HIP GPUs at daemon startup, not just selected device 0;
  - report per-device id, arch, VRAM, integrated/discrete class, and HIP
    runtime from the daemon as well as the server;
  - probe available NPUs/XDNA devices separately when present;
  - make the priority microbatching scheduler route by worker/device capability
    instead of using a single server-selected HIP placement.
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
- Run the full no-GPU handoff gate before committing the branch.
- Extend Qwen35 paged expert execution coverage beyond the current indexed MQ
  routed path: grouped Path 2, Paro, full-precision, and CPU-fallback paged
  paths still need explicit tests before broad admission.
- Extend routed-only Qwen3 MoE forward validation beyond the current one-token
  modular/lazy expert smoke: multi-token prefill, decode continuation, eager vs
  paged parity on a smaller artifact, and grouped routed batches still need
  explicit coverage before removing the guarded execution flag.

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


## hipfire-eval is still loading model multiple times rather than reseting state between tests.

## further investigate using packed 4 bit operations on gfx1151/RDNA3/RDNA3.5
