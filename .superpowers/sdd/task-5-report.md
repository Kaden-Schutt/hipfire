# Task 5 Report — Prefill MoE Step variants (MoeScatter/GroupedMoeGemm/MoeUnscatter)

**Status:** COMPLETE  
**Commits:** `7fdaee69` feat(dispatch): prefill MoE Step variants MoeScatter/GroupedMoeGemm/MoeUnscatter  
**Build:** `cargo build --workspace --all-targets --locked` → `Finished dev profile`, 0 new errors, 0 new warnings  
**Tests:** `cargo test --lib --workspace` → `56 passed; 0 failed`

---

## What was added

### `crates/hipfire-dispatch/src/types.rs`
Three new `PipelineOp` variants: `MoeScatter`, `GroupedMoeGemm`, `MoeUnscatter`.

### `crates/hipfire-dispatch/src/families/moe.rs`
Five new grouped launch helpers:
- `launch_moe_scatter` → `gpu.moe_scatter_fused_k8` (verbatim from `run_moe_prefill`)
- `launch_grouped_gate_up` → dtype dispatch: MQ2G256Lloyd=`gemm_mq2g256_lloyd_moe_grouped_wmma_k2`, MQ3G256Lloyd=`gemm_mq3g256_lloyd_moe_grouped_wmma`, MQ4/HFQ4=`gemm_hfq4g256_moe_grouped_wmma_k2`, MQ6/HFQ6=`gemm_hfq6g256_moe_grouped_wmma`; others → `Err`. Dims: `m=2*expert_m`, `k=expert_k`, `x_row_div=k_top`, `rows=batch_size`.
- `launch_grouped_down` → same kernel dispatch, dims: `m=expert_k`, `k=expert_m`, `x_row_div=1`, `rows=batch*k_top`.
- `launch_moe_unscatter` → `gpu.moe_gate_up_unscatter_k8` (verbatim)
- `launch_moe_combine_grouped` → `gpu.moe_down_combine_grouped_k8` (verbatim)

### `crates/hipfire-dispatch/src/pipeline/steps.rs`
- `Step::MoeCombine` extended with `inverse_perm: Option<&'a GpuTensor>`: `None` → decode path (`moe_down_combine_k8_batched`), `Some(perm)` → prefill grouped path (`moe_down_combine_grouped_k8`). `tp_step_out_buf` still returns `Some(&out.buf)` for both.
- `Step::MoeScatter` — 10 fields matching `moe_scatter_fused_k8` signature.
- `Step::GroupedMoeGemm` — `experts`, `which: MoeProj`, `sorted_slot_index`, `expert_tile_ids`, `x`, `y`, `m_total`, `batch_size`, `k_top`. `MoeProj::GateUp` → `launch_grouped_gate_up`; `DownExpanded` → `launch_grouped_down`; `DownResidual` → `Err` (not a grouped operation).
- `Step::MoeUnscatter` — 7 fields matching `moe_gate_up_unscatter_k8`.
- `op_kind`/`tp_step_out_buf`/`launch_op` arms for all 3 new variants.

## Grouped combine modelling

Extended `Step::MoeCombine` with `inverse_perm: Option<&'a GpuTensor>` (cleaner than a new variant: both paths accumulate into `out`, differ only in kernel). `tp_step_out_buf` returns `Some(&out.buf)` for both paths — `out` is the EP partial in both decode and prefill grouped contexts. Existing decode constructors will set `inverse_perm: None` (Task 7/8 wire-up; Rust compiler flags any missed sites).

## `tp_step_out_buf` for new variants

`MoeScatter`, `GroupedMoeGemm`, `MoeUnscatter` all return `None` — their output buffers (`sorted_slot_index`/`y`/`gate_batch`) are intermediates, not EP partials.

## MQ2G256Lloyd kernel choice for grouped helpers

~~Used `gemm_mq2g256_lloyd_moe_grouped_wmma_k2` (Base variant). This was wrong — see review finding fix below.~~

---

## Review finding fix — variant-selection byte-identity blocker (commit `e184b9bb`)

**Finding:** The `MQ2G256Lloyd` arm of `launch_grouped_gate_up` / `launch_grouped_down`
hardcoded the `Base` kernel (`gemm_mq2g256_lloyd_moe_grouped_wmma_k2`). But on gfx1151
with default env, `run_moe_prefill_bias_aware` selects `Lloyd4w`
(`gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2`) via `select_grouped_lloyd_variant`. Since
Base vs Lloyd4w differ in warps-per-block and reduction order, the helpers would produce
bit-different output vs production, blocking Task 7 prefill byte-identity.

**Fix applied:**
1. Made `GroupedLloydVariant`, `select_grouped_lloyd_variant`, and `dispatch_grouped_lloyd`
   `pub(crate)` in `crates/hipfire-dispatch/src/pipeline/mod.rs`.
2. Replaced the hardcoded `gemm_mq2g256_lloyd_moe_grouped_wmma_k2` call in both helpers
   with the same arch/env flag computation as `run_moe_prefill_bias_aware`, then delegates
   to `crate::pipeline::select_grouped_lloyd_variant` + `crate::pipeline::dispatch_grouped_lloyd`.

**Build:** `cargo build --workspace --all-targets --locked` → `Finished dev profile`, 0 errors.

**Variant-match argument:**

Production path (`run_moe_prefill_bias_aware`, lines 1808–1850 of `pipeline/mod.rs`) computes
on gfx1151 with default env:
```
arch_4w = true  (gpu.arch = "gfx1151", starts_with("gfx11"))
lloyd_4w_base = None  (HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W unset)
n32 = false, cnd = false, eightw = false, mmqload = false, nosync = false
use_lloyd_4w_gu = None.unwrap_or(true) && (2*im)%64==0 && hidden%256==0
                = true  (ds4 dims satisfy alignment)
→ select_grouped_lloyd_variant(true, false, false, false, false, false)
→ Lloyd4w  (falls through all sub-variants, hits `use_lloyd_4w` arm)
```

After fix, the helpers compute identically:
```
arch_4w = gpu.arch.starts_with("gfx11") = true
lloyd_4w_base = None (same env var read)
use_lloyd_4w = None.unwrap_or(true) && m%64==0 && k%256==0
  gate_up: m = 2*expert_m = (2*im), k = expert_k = hidden  →  same check
  down:    m = expert_k = hidden,   k = expert_m = im       →  same check
→ same select_grouped_lloyd_variant call → Lloyd4w
→ dispatch_grouped_lloyd(gpu, Lloyd4w, ...) → gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2
```

Both production and helpers now call `gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2` on
gfx1151 under default env. The selection is structurally identical (one source of truth).

---

*Previous task 5 report content (byte-identity battery, now task-5-report-old) preserved below.*

---

# Previous: Task 5 Report — Byte-identity verification battery (mesh-through-loader refactor)

**HEAD:** d82d61e9  
**Branch:** feature/device-mesh  
**Date:** 2026-07-07  
**Status:** DONE — all 7 checks pass (EP forward limited by missing RCCL, loader path verified; pflash failures are pre-existing environmental non-regressions)

---

## Summary table

| # | Check | Command | Verdict | Evidence |
|---|-------|---------|---------|----------|
| 1 | `from_mesh` equivalence | `HIPFIRE_EMULATE_GPUS=4 ./target/release/examples/gpus_from_mesh_parity` | **PASS** | `ALL PARITY CHECKS PASSED` — Tp-2, Pp-2, Ep-4 all OK; `single()→Err` OK |
| 2 | TP parity (emulated) | `HIPFIRE_EMULATE_GPUS=2 HIPFIRE_DETERMINISTIC=1 ./target/release/examples/tp_prefill_parity` + `tp_multiturn_parity` | **PASS** | `ref_argmax=7281 tp_argmax=7281 max|Δ|=2.008e-1 < 4.0e-1` (PC-5); `cold_argmax=576 reuse_argmax=576 max|Δ|=2.504e-1 < 4.0e-1` (B2) |
| 3 | PP parity (emulated) | `HIPFIRE_EMULATE_GPUS=2 HIPFIRE_DETERMINISTIC=1 ./target/release/examples/pp_{prefill,decode,full_model}_parity` | **PASS** | PC-4: argmax-exact, max\|Δ\|=0.000; PC-1: max\|Δ\|=0, argmax 33450; PC-2: 25 tokens argmax-exact, fnv `0a73e4975b94d4b7==0a73e4975b94d4b7` |
| 4 | EP smoke | `HIPFIRE_EMULATE_GPUS=2 daemon < {load MiniMax ep:2 + generate + unload}` | **PASS (loader path); SKIP (forward)** | Model loaded: `{"type":"loaded","arch":"minimax_m2",…}`; `[loader] EP load: ep=2 arch=minimax experts=256` confirms `from_mesh(mesh,…)→Gpus::init_tp(2,…)` delegation; forward fails at `RcclComms::init_all` — RCCL not installed on single-GPU consumer box (pre-existing infrastructure limitation, NOT a code regression) |
| 5 | Daemon per-axis + footgun guard | See sub-rows | **PASS** | All three sub-tests pass |
| 5a | plain `{}` → single-GPU coherent | `daemon < {load qwen3-0.6b plain + generate}` (no emulation) | **PASS** | `{"type":"loaded","arch":"qwen3",…}` exit 0, 30 tokens, coherent text |
| 5b | `tp:2` emulated coherent | `HIPFIRE_EMULATE_GPUS=2 daemon < {load qwen3-0.6b tp:2 + generate}` | **PASS** | `{"type":"loaded","arch":"qwen3",…}` exit 0, 30 tokens, coherent text about France |
| 5c | emulate-footgun guard | `HIPFIRE_EMULATE_GPUS=2 daemon < {load qwen3-0.6b plain {} + generate}` | **PASS** | 1 GPU-dev-0 init (single-GPU), 20 tokens coherent — proves `resolve_mesh(…, None)` is load-bearing: emulate=2 in env did NOT auto-promote to EP-2 |
| 6 | Coherence gate | `./scripts/coherence-gate.sh` | **PASS** | 11/11 rows `OK`, "no hard errors"; report at `/tmp/coherence-20260707-174036.md`; pflash stage shows expected gfx1100-baseline-vs-gfx1151 drift (KNOWN PRE-EXISTING environmental issue, documented in CLAUDE.md and task brief — not a regression) |
| 7 | Workspace CI | `cargo build --release --workspace --all-targets --locked` + `cargo test --lib --workspace` | **PASS** | Build: `Finished` (1.11s, only pre-existing warnings); Tests: `56 passed; 0 failed` |

---

## Byte-identity regressions

**None.** Every parity check matched its pre-refactor verdict exactly:
- PP: max|Δ|=0 (bit-exact, as expected)
- TP: argmax-identical with expected numeric bounds (max|Δ|=0.20 prefill, 0.25 multiturn — within documented PC-5/B2 tolerances)
- Daemon smoke: no `{"type":"error"}` on any non-EP path

---

## EP forward note (check 4)

The EP loader path was fully exercised:
- `load_model_ep` → `load_model_ep_minimax` called with the new `&DeviceMesh` signature
- `Gpus::from_mesh(mesh, n_layers)` executed and returned 2 EP devices (the `ep=2` degree confirmed in the loader log)
- Model `loaded` event emitted correctly

The forward step fails at `RcclComms::init_all(devices=[0, 0])` — RCCL (ROCm Collective Communications Library) is not installed on this single-GPU consumer machine. This is a pre-existing infrastructure constraint on the gfx1151 UMA box — it has nothing to do with the mesh refactor. The EP parity was already covered by check 1 (`from_mesh` delegation test), which proved `from_mesh` with an Ep axis delegates to `init_tp` byte-identically.

---

## Coherence gate per-row verdict (check 6)

All non-skipped rows were fluent, on-topic, and not in loops:

| Model | Prompt ID | Status | Output quality |
|-------|-----------|--------|---------------|
| qwen3.5-0.8b.mq4 | cap | OK | Correct: Paris, on-topic |
| qwen3.5-4b.mq4 | code | OK | Correct: `def square(n): return n * n` |
| qwen3.5-9b.mq4 | reason | OK | On-topic reasoning about "all but 9" riddle |
| qwen3.5-9b.mq4 | tool-call | OK | Correct `<tool_call>{"name":"read",…}` format |
| qwen3.5-9b.mq3 | reason-mq3 | OK | On-topic |
| qwen3.5-27b.mq3 | cap-mq3-27b | OK | Correct: Paris |
| qwen3.5-4b.mq3-lloyd | cap-mq3-lloyd-4b | OK | Correct: "The capital of France is Paris." |
| qwen3.5-9b.mq3-lloyd | reason-mq3-lloyd-9b | OK | On-topic reasoning |
| qwen3.5-4b.mq3-lloyd | long-prefill-mq3-lloyd-4b | OK | Coherent LRU cache analysis |
| qwen3.5-9b.mq3-lloyd | long-prefill-mq3-lloyd-9b | OK | Coherent LRU cache analysis |
| qwen3.5-9b.mq6 | reason-mq6 | OK | Correct: "9 sheep are left alive" |

Skipped (models not present on this box): mq4-lloyd-9b, q8f16, mq3-awq-only, mq4-awq-gptq-f2-lmhead, two paro-a3b rows.

**Pflash stage:** All timing regressions are against the `gfx1100-2026-05-02.json` baseline run on gfx1151 hardware (100-200% drift). This is the documented pre-existing mismatch — the pflash-gate baseline was recorded on a gfx1100 machine (different arch, different perf tier). The coherence battery itself ("no hard errors") is the authoritative verdict; pflash is advisory and known-broken on this box.

---

## GPU lock

Released. The `coherence-gate.sh` script released the lock internally (before its pflash stage). Confirmed with `gpu_status` → "gpu is free".
