# Ship 4.1 dev log

**Branch:** `feature/ship-4.1-moe-family-resolution` off `integration/dispatch-unification`
**Tracking:** [#397](https://github.com/warpfront/hipfire/issues/397) Step 4.1
**Verification GPU:** gfx1151 (gfx11-family; gfx12 pending per SF-4.1.1)

---

## 2026-06-06 — W0+W1 landed (family owns resolution + ctx threading + runtime guard)

**Commits:** single commit (W0+W1 co-landed to keep workspace green)

### W0 — dispatch crate (GPU-free)

| Change | Files |
|---|---|
| `MoeParams.res: MoeResolution` → `MoeParams.dtypes: MoeDtypes` + `batch_size: usize` | `families/moe.rs` |
| `MoeFamily::run` forwards `ctx` (was `_ctx`) to `run_moe_decode` | `families/moe.rs` |
| `run_moe_decode(ctx, gpu, p)` — computes `MoeResolution::resolve(&p.dtypes, p.k)` internally | `pipeline/mod.rs` |
| Runtime `p.batch_size != 1` guard (matching bias-aware precedent) | `pipeline/mod.rs` |
| 5 internal `DispatchCtx::new` deleted; `ctx` threaded to all `gemv.run_auto` | `pipeline/mod.rs` |
| `run_moe_decode_cpu_fallback(ctx, …)` — 2 per-expert `DispatchCtx::new` removed | `pipeline/mod.rs` |
| `FIXME(Step 8)` at 4 hardcoded `1` batch_size literals | `pipeline/mod.rs` |
| `execute_pipeline` + `dispatch_fused` forward `ctx`; `dispatch_fused` caller in `gemv.rs` updated | `pipeline/mod.rs`, `gemv.rs` |

### W1 — qwen35 decode site

| Change | Files |
|---|---|
| Deleted `MoeResolution::resolve` call — model no longer resolves | `qwen35.rs:4600` |
| `MoeParams { dtypes: moe_dtypes, batch_size: 1, ... }` | `qwen35.rs:4616-4617` |
| Builds one `DispatchCtx::new(gpu)` → `moe_family().run(&ctx, gpu, &moe_params)` | `qwen35.rs:4663-4665` |

### W0 tests (GPU-free, 8 new)

`MoeResolution::resolve` unit cells in `crates/hipfire-dispatch-tests/src/qwen35.rs`:
- k=8 MQ4 indexable → `use_gpu_topk` ✅
- k≠8 (1,2,4,6,7,9,16) → CPU fallback ✅
- Non-indexable routed (Q8) → fallback even with k=8 ✅
- MQ6 indexable ✅
- Paro indexable (with sidecar) ✅
- Paro without sidecar → fallback ✅
- `needs_x_rot_local` when gate-side MQ4 ✅
- All-F32 → no rotation ✅

Test results: **190/190 pass** (123 dispatch + 66 dispatch-tests + 1 golden).

### Verification (gfx1151)

| Fixture | Model | Prompt | Result |
|---|---|---|---|
| A3B k=8 (GPU fast path) | `qwen3.6-35b-a3b.mq4` | "What is 2+2?" | **Coherent** — output "4" |
| A3B k=8 (GPU fast path) | `qwen3.6-35b-a3b.mq4` | sheep reasoning (60 tok) | **Coherent** — correct answer "9" with reasoning |
| coherence-gate (dense) | 0.8B/4B/9B/27B MQ4/MQ3 | standard battery | **All coherent** (cap/code/reason/tool-call) |

**Note on coherence-gate A3B:** The first two coherence-gate runs produced garbled A3B
output due to **stale incremental compilation artifacts**. The coherence-gate.sh
rebuild trigger did not include dispatch-crate files (`moe.rs`, `pipeline/mod.rs`,
etc.), so `cargo build --release` sometimes skipped recompilation after `MoeParams`
struct layout changes. A `cargo clean -p hipfire-dispatch -p hipfire-arch-qwen35
-p hipfire-runtime` + rebuild resolved the issue. The trigger list has been updated
(SF-4.1.4).

### Gaps documented (dispatch_todo.md)

| Gap | Severity | Assignee |
|---|---|---|
| SF-4.1.1 — gfx1201 cross-arch verification | HIGH | Kaden |
| SF-4.1.2 — k≠8 CPU-top-K fallback fixture | MED | TBD |
| SF-4.1.3 — A3B DFlash draft model missing | LOW | TBD |
| SF-4.1.4 — coherence-gate rebuild trigger (FIXED) | LOW | ✅ |
| SF-4.1.5 — batch_size guard unit test (GPU-gated) | LOW | TBD |

### Grep audit (W1)

- Zero `pipeline::run_moe_decode(` in `qwen35.rs` ✅
- Zero `MoeResolution::resolve(` in `qwen35.rs` ✅
- `moe_family().run(` call site verified ✅
- Single `DispatchCtx::new(gpu)` at MoE decode site ✅

### Out of scope (unchanged)

- `moe_grouped` (grouped prefill, qwen35.rs:7280+)
- ds4 MoE (PR #428, separate)
- Multi-GPU MoE decode
- `routed_experts` per-call `Vec` alloc (pre-existing)
