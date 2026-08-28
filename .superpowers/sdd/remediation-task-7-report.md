# Task 7 Report — Lowered Gemma4 Ownership and Transactional Construction

**Status: IMPLEMENTED — focused ownership, rollback, telemetry, and accessor tests are green.**

## Scope

The lowered Gemma4 route now owns every completed GPU allocation until publication. The
changes preserve generation/cache behavior and retain the existing pool-backed expert
view and tied-LM-head alias contracts.

## Red/green evidence

- RED: `cargo test -p rdna-compute cached_bytes_reports_actual_free_list_capacity --lib`
  initially failed because `GpuPool::cached_bytes` did not exist.
- GREEN: the same accessor test passed (`1 passed`, 192 filtered).
- RED: `cargo test -p hipfire-arch-gemma4 lowered_constructor --lib` initially failed
  because the construction-stage matrix API did not exist.
- GREEN: `cargo test -p hipfire-arch-gemma4 lowered_constructor --lib` passed (`2 passed`,
  24 filtered).
- `cargo test -p hipfire-arch-gemma4 lowered_ --lib` passed (`4 passed`, 24 filtered).
- `cargo test -p hipfire-arch-gemma4 lowered_allocation_telemetry --lib` passed (`2 passed`,
  26 filtered).
- `cargo test -p hipfire-loader gemma4 --lib` passed (`6 passed`, 55 filtered).
- `cargo test -p hipfire-loader --features lowered-fault-inject gemma4 --lib` passed
  (`6 passed`, 55 filtered).

The focused runs emitted only pre-existing compiler warnings. No formatter, linter,
project-wide suite, broad product gate, or GPU model-quality gate was run.

## Owner/free table

| Resource | Owning representation | Teardown | Alias/view rule |
|---|---|---|---|
| Embedding | `Gemma4Weights::embed_tokens` | `gpu.free_tensor` | `lm_head` is a borrowed tied alias and is not freed |
| Dense layer weights | `WeightTensor` values | `WeightTensor::free_all` | Includes primary, AWQ, and owning Paro sidecars |
| Dense layer norms/scalars | `GpuTensor` values | `gpu.free_tensor` | Each descriptor is freed once |
| MoE router | Owning `WeightTensor` | `free_all` | Sidecars are included |
| MoE expert pools | `experts_gate_up_pool`, `experts_down_pool` | `gpu.free_tensor` once per pool | Expert `WeightTensor` subviews are borrowed and never independently freed |
| MoE pointer tables/scales/norms | Owning `GpuTensor` values | `gpu.free_tensor` | Pointer tables are released before pools |
| Lowered scratch | All scratch `GpuTensor` fields | Reverse-order `gpu.free_tensor` | No scratch field is omitted |
| Position scalar | `DeviceBuffer` from `hip.malloc(4)` | Explicit `gpu.hip.free` | No `Drop` assumption remains |
| Sliding/full KV | `KvCache` owning vectors | `KvCache::free_gpu` | K/V/scales/Givens owners are released by the cache |
| Reusable pool | `GpuPool` free lists | `Gpu::drain_pool` at normal unload | `Gpu::pool_cached_bytes` reports actual cached capacities |

## Fault stages

`Gemma4ConstructionStage` is ordered as:

1. `Weights`
2. `Scratch`
3. `SlidingKv`
4. `FullKv`
5. `Session`

With the `lowered-fault-inject` feature, setting
`HIPFIRE_GEMMA4_FAIL_STAGE` to `weights`, `scratch`, `sliding_kv`, `full_kv`, or
`session` returns an injected error immediately after that completed boundary. The
carrier staging owner drops in reverse order: full KV, sliding KV, scratch, weights.
The lower weight and scratch constructors have their own reverse-order owner
transactions, so partial failures before a carrier boundary are covered as well.
Publication takes each staging owner exactly once and only occurs after the session
boundary.

## Telemetry fields

One opt-in developer flag, `HIPFIRE_GEMMA4_ALLOC_TELEMETRY=1`, emits a compact
`[gemma4 alloc]` line at lowered publication and unload. Fields are:

- `phase` and operator-provided `HIPFIRE_GEMMA4_ALLOC_CYCLE` (`cycle`);
- actual lowered `owner_bytes`;
- actual reusable `pool_bytes` from `Gpu::pool_cached_bytes`;
- `free_device_bytes` from `hipMemGetInfo`, or `unknown` when unavailable;
- `graph_resident`, `graph_blob_count`, and `module_count` observability;
- comma-separated `freed_owner_labels`.

The production call sites use the gated `emit_from_gpu` helper, so disabled telemetry
performs no VRAM query and emits no product log noise. The rdna-compute additions are
generic read-only accessors only: `GpuPool::cached_bytes`, `Gpu::pool_cached_bytes`, and
`Gpu::loaded_module_count`; no allocator mutation or architecture-specific telemetry
was exposed.

## Commits

- `44711a71b` — `fix(gemma4): make lowered ownership transactional`

## Concerns / self-review

- The focused suite proves construction API coverage and compile/runtime contracts but
  does not run the full 12B/26B GPU fixture fault matrix in this worker pass; those
  fixtures and broad lifecycle gates remain controller-owned.
- Telemetry intentionally reports pool-cached bytes before the normal external pool
  drain, making owner-versus-cache variance observable rather than hiding it.
- Existing generation, cache, cursor, and forward code paths were not changed.
