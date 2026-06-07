# TODO

## Evaluation Branch

### Active

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

## FWHT Residual QJL Transform

Status: deferred.

- Implement a Johnson-Lindenstrauss / QJL transformation on the residual in the FWHT path. The current FWHT path applies a signed-FWHT rotation to Q/K for attention and leaves the residual stream without a separate QJL transform.



## Check all hot paths for graph safety
>>> One issue surfaced before verification: gemm_f16_x_f32_wmma currently launches with raw stack kernargs rather than the graph-safe blob helper. I’m tightening the env gate so this experimental route only runs outside hipGraph capture; captured paths will keep using the scalar default until the dispatcher wrapper is made graph-safe.
