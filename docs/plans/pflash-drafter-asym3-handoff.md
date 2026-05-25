# PFlash drafter asym3-KV migration + Q8 batched flash LDS escape — agent handoff

## Context

`fix/long-decode-nan` (PR-pending) closes the original NaN bug in `attention_flash_q8_0_tile.hip` (partials stride + head_dim=256 coverage, both pre-existing) and wires `--pflash` into `dflash_spec_demo` so PFlash compression can feed DFlash speculative decode.

Validation grid on 7900 XTX, qwen3.5-27b target + qwen35-27b-dflash draft + qwen3.5-0.8b PFlash drafter, asym3 KV, NIAH fixtures `niah_{16k,32k,64k,128k}.jsonl` × `keep_ratio {5%,3%,1%}`: **needle recovered in every cell**. But the perf profile surfaces an architectural cliff that **wasn't** in the original bug class:

| source tok | drafter prefill tok/s | path |
|---|---|---|
| 10890 (16K)  | 4823 | `attention_q8_0_kv_batched_masked` (batched flash) |
| 21560 (32K)  | 1547 | per-position fallback ← LDS cap hit |
| 43305 (64K)  |  791 | per-position fallback |
| 86468 (128K) |  398 | per-position fallback |

At 128K source the PFlash compress takes **217s** — 97% of TTFT. The original kernel fix is what *makes this fallback path produce correct output*; without it, every PFlash compression past 15K source would have NaN-corrupted importance scores. So the fix is load-bearing for any long-source PFlash use, but the compress cost is now the dominant bottleneck.

## Why this is happening

`crates/hipfire-arch-qwen35/src/qwen35.rs` line 5021:

```rust
} else if max_ctx_len > LDS_CTX_LIMIT {  // = 15000
    // Per-position flash Q8 attention for long-context prefill.
    for b in 0..n {
        gpu.attention_flash_q8_0(...)  // single-token flash kernel, called N times
    }
} else {
    gpu.attention_q8_0_kv_batched_masked(...)  // proper batched flash
}
```

Why the 15K cap: `attention_q8_0_kv_batched_masked` is a single-launch batched flash that puts `tile_scores + ws + out_run` in LDS, all sized to ctx_len. At ctx_len > 15000 the layout overflows the 56KB usable LDS on gfx1100 (8KB margin under 64KB hard cap). The fallback loop trades correctness for throughput.

The drafter is *forced* to Q8 KV by `pflash.rs:604`:

```rust
assert!(kv.quant_q8, "drafter_prefill: drafter KV must be Q8_0");
```

…because `compute_scores_batched_gpu` (line 779) dispatches to the GPU kernel `pflash_score_q8_kv_blocks` which reads Q8 K cache layout directly (2-byte f16 scale + 32 i8 values per 32-element block). asym3 K is 3-bit Givens-rotated with 4-byte f32 cnorm + packed bits — different layout, no existing score kernel.

## Two independent levers (do both, in this order)

### Lever 1 — asym3 KV on the PFlash drafter (high leverage, smaller delta)

asym3 batched flash uses tiled partials-buffer reduction (`attention_flash_asym3_batched_masked`, `attention_flash_asym3_tile` + Q8 reduce) and has **no LDS cap**. Moving the drafter to asym3 KV sidesteps the 15K cliff entirely.

**Changes required:**

1. **New HIP kernel `kernels/src/pflash_score_asym3_kv.hip`** (~120 LOC; port the asym3 tile kernel's dequant pattern into the Q8 score kernel's reduction structure):
   - Same `[block_idx]` grid and `[256]` thread layout as `pflash_score_q8_kv_blocks`
   - Per-thread dequant of K[d] for `d in tid..kv_dim step 256`:
     - Compute kv_head `h`, in-head dim `dim_in_head`, and which Givens block (`b = dim_in_head / 2`)
     - Load `cnorm` (f32 at start of head's K bytes)
     - Decode the 3-bit code at the right offset using `TURBO_C3_256[code]` (from `turbo_common.h`)
     - Apply inverse Givens rotation with `cos_theta[b]`, `sin_theta[b]` (passed as kernel params from `kv.givens_cos.as_ref().unwrap()` / `givens_sin`)
   - Same dot/nb/nl accumulators + shared-memory reduction + cosine math as Q8 version
   - Stride: `4 + (head_dim * 3) / 8` bytes per head (not `blocks_per_head * 34`)
   - Test against `compute_scores_batched` CPU reference once you write the matching CPU dequant in step 4
   - Use `attention_flash_asym3_tile.hip` lines 53-99 as your reference for the dequant math

2. **`kernels/src/lib.rs` / `crates/rdna-compute/src/kernels.rs`** — export new constant `PFLASH_SCORE_ASYM3_KV_SRC` via the existing `include_str!` pattern.

3. **`crates/rdna-compute/src/dispatch.rs`** — add `pflash_score_asym3_kv(...)` wrapper next to `pflash_score_q8_kv`. Takes `k_cache, cos, sin, scores, n, n_kv_heads, head_dim, block_size, n_blocks, last_pos`. Use `ensure_givens4_kernel` for include-prepend of `turbo_common.h` + `givens_common.h`.

4. **`crates/hipfire-arch-qwen35/src/pflash.rs`:**
   - Change `load_drafter` signature to take a `kv_mode: KvMode` enum (or just a `bool use_asym3`); allocate `KvCache::new_gpu_asym3` instead of `new_gpu_q8` when asym3. `KvCache::new_gpu_asym3` already exists.
   - Drop the `assert!(kv.quant_q8)` in `drafter_prefill` (line 604); the underlying `qwen35::forward_prefill_batch` already supports asym3 — it routes to `attention_flash_asym3_batched_masked` at line 5001 of qwen35.rs.
   - In `compute_scores_batched_gpu` (line 779): dispatch on `kv.quant_q8` vs `kv.quant_asym3` to call the right scoring kernel.
   - In `compute_scores_batched` (CPU path, line 692-697): add an asym3 dequant variant (mirror `dequant_q8_kv_position` for the 3-bit-rotated case) — needed as the reference for cross-checking the new GPU kernel.

5. **`crates/hipfire-runtime/examples/dflash_spec_demo.rs`** — add `--pflash-kv-mode {q8,asym3}` with default q8 for back-compat. Pass through to `pflash::load_drafter`.

6. **Validation:** re-run the 12-cell grid (4 fixtures × 3 ratios) with `--pflash-kv-mode asym3`. Targets:
   - Drafter prefill stays at ~4800 tok/s across all source sizes (no fallback)
   - 128K compress drops from ~218s to ~18s
   - Needle recovered in all cells; importance scores within ~1% MSE of the Q8 path (asym3 introduces quantization noise on K but the cosine score is robust)

**Acceptance criteria:**
- `dflash_spec_demo --pflash <drafter> --pflash-kv-mode asym3` works at niah_128k.jsonl with sub-30s TTFT at any keep_ratio
- Existing `--pflash-kv-mode q8` (default) is byte-exact vs master on the niah_16k grid
- `cargo test -p hipfire-arch-qwen35 pflash` (if any) passes
- `scripts/coherence-gate-dflash.sh` passes (canonical asym3 path, unaffected)

### Lever 2 — Tiled Q8 batched flash to escape the LDS cap (larger but broader payoff)

Q8 KV at long ctx isn't only a PFlash problem. Target prefill with Q8 KV at long ctx, batched verify on Q8, anything that goes through `forward_prefill_chunk` with `max_ctx_len > 15000` falls into the same per-position loop. A tiled Q8 batched flash would help all of these.

**Changes required:**

1. **New `kernels/src/attention_flash_q8_0_batched_tile.hip`** — analog of `attention_flash_q8_0_tile.hip` but the Q dim is `[N × n_heads × head_dim]` instead of `[n_heads × head_dim]`. Block becomes `[n_heads, n_tiles, N]` or `[N, n_heads × n_tiles, 1]` depending on which dim you parallelize. Recommend the `[n_heads × n_tiles, N, 1]` grid with one tile per workgroup and N being the batch (matches asym3 batched tile in `attention_flash_asym3_tile_batched.hip`).

2. **New `kernels/src/attention_flash_q8_0_batched_reduce.hip`** — same 2-pass reduce as asym3's batched reduce. Partials buffer carries the extra N dim.

3. **`crates/rdna-compute/src/dispatch.rs`** — `attention_flash_q8_0_batched_masked(...)` wrapper. Takes `tree_bias`, `block_start`, `block_cols` to mirror the existing masked-batched signature for asym variants.

4. **`crates/hipfire-arch-qwen35/src/qwen35.rs`** — at line 5021 replace the per-position fallback with the new batched call. Keep the existing in-LDS-budget path (`attention_q8_0_kv_batched_masked`) for `ctx ≤ 15000` so we don't pay the tile-overhead at short ctx.

5. **Validation:** profile target prefill at 16K and 32K with `--kv-mode q8`. Currently dominated by the fallback at 32K. Target a 3-6× prefill speedup at 32K, 8-12× at 64K.

**Acceptance criteria:**
- niah_{32k,64k} `pflash_niah_bench --q8kv` prefill speed within 30% of asym3's prefill speed at the same fixture (proves the batched flash is doing real work)
- Canonical merge_sort bench unchanged on the q8 path (because at short ctx the existing path is used)
- New kernel passes a parity test vs running the single-token kernel position-by-position on the same input

## What NOT to touch

- The fix on this branch in `attention_flash_q8_0_tile.hip` — the stride and head_dim coverage fixes are correct and validated; both Levers above depend on the kernel being correct for the fallback path that's still in production.
- `pflash::compute_scores_cpu` (line 818) — Phase 1.2 reference path, predates batched. Leave it as the per-token reference.
- The score-layer auto-pick logic (`score_layer_idx`, line 217 and 768) — works correctly for both Q8 and asym3 since it's based on layer structure, not KV layout.
- `wrap_chatml` in dflash_spec_demo — orthogonal.

## Where to start

```bash
git fetch origin
git checkout -b feat/pflash-asym3-drafter origin/master
# Cherry-pick or wait for the long-decode-nan PR to merge if it hasn't yet.
# The Lever 1 work depends on the kernel fix being on master.
```

Read in order:
1. `kernels/src/pflash_score_q8_kv.hip` (115 LOC, full kernel)
2. `kernels/src/attention_flash_asym3_tile.hip` lines 22-101 (the asym3 dequant + Givens pattern you'll port)
3. `crates/hipfire-arch-qwen35/src/pflash.rs` lines 478-540 (`load_drafter`) and 743-793 (`compute_scores_batched_gpu`)
4. `crates/rdna-compute/src/dispatch.rs` around the `pflash_score_q8_kv` wrapper (grep `pflash_score_q8_kv`)

## Repro for the bottleneck this fixes

```bash
cargo build --release --features deltanet --example dflash_spec_demo
./target/release/examples/dflash_spec_demo \
  --target ~/.hipfire/models/qwen3.5-27b.mq4 \
  --draft ~/.hipfire/models/qwen35-27b-dflash.mq4 \
  --prompt-file benchmarks/longctx/niah/niah_128k_prompt.txt \
  --pflash ~/.hipfire/models/qwen3.5-0.8b.mq4 \
  --keep-ratio 0.03 --max 64 --kv-mode asym3 --no-chatml --ctx 8192
```

(generate the prompt file via the small Python in the PR description, or just inline the JSONL `filler_text + needle + question`)

Expected after Lever 1: `pflash_compress_ms: ~18000` (down from `217723`).
Expected after Lever 2: same drafter prefill speedup carries over to target prefill on long-ctx Q8 workloads.
