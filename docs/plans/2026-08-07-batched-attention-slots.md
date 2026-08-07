# Ragged Multi-Slot Batched Attention (SP1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give hipfire's Q8_0 and asym3 attention kernels a *slot* dimension, so a single launch serves several independent sequences — each at its own position, each attending only to its own KV — as the kernel foundation for running 3–4 coding agents concurrently on a 32 GB R9700.

**Architecture:** The batched kernels already take a device `positions[]` array and launch one workgroup per (head, query-row-tile). The only thing binding them to one sequence is the scalar `k_cache`/`v_cache` base pointer. We add two device arrays — a per-slot `KvSlotDesc` table and a per-tile `tile_slot[]` map — and route every KV address through one device helper, `kv_offset_for(slot, pos)`. Contiguous per-slot slabs today; a block table later changes that one function, not the kernels.

**Tech Stack:** Rust (`rdna-compute` crate), HIP C++ kernels compiled at runtime via `ensure_kernel` with `#define` injection, `cargo run --release --example` harnesses.

**Spec:** `docs/specs/2026-08-07-batched-attention-slots.md`. Read it before starting; this plan implements it and does not restate its reasoning.

## Global Constraints

- **Implementation branch:** `feat/batched-attn-impl`, worktree `~/repos/hipfire-batchattn-impl`. **All code work happens here.** It was cut from `feat/batched-attention-slots` (worktree `~/repos/hipfire-batchattn`), which carries the spec and this plan and is kept pristine as the reference — do not commit code to it. Both descend from `origin/beta` @ `e2f7dd1a`.
- **Toolchain:** ROCm 7.2.2, `hipcc` at `/opt/rocm-7.2.2/bin/hipcc` (already on PATH). Target arch for local builds is `gfx1151`.
- **Dev hardware is gfx1151 (Strix Halo); target is gfx1201 (R9700).** No tuned constant may be baked into a Rust `const`. Every tuning value must be env-overridable.
- **KV modes in scope: Q8_0 and asym3 only.** Do not touch asym2, asym4, fwht{2,3,4}, or lloyd kernels.
- **Backward compatibility is mandatory.** When the descriptor pointer is null, every kernel must behave *bitwise identically* to its current single-sequence behaviour. All existing call sites pass null initially.
- **Q8_0 block layout:** 34 bytes per 32 values (`f16` scale at bytes 0–1, then 32 `int8` codes). Per-position stride is `n_kv_heads * (head_dim/32) * 34` bytes.
- **Existing routing constant:** `LDS_CTX_LIMIT = 15000` at `crates/hipfire-runtime/src/llama.rs:2637`. Do not change its value.
- **Tile constants:** `TILE_SIZE = 128`, `stride = 2 + head_dim` (partials are `m`, `l`, then `head_dim` accumulators), `max_tiles = ceil(max_ctx_len / TILE_SIZE)` — `crates/rdna-compute/src/attention.rs:3473`.
- **Licence header** on every new file, matching existing files:
  ```
  // SPDX-License-Identifier: Apache-2.0
  // Copyright (c) 2026 Nick Woolmer
  // hipfire — see LICENSE and NOTICE in the project root.
  ```
- **Always pass `--features deltanet`.** Every cargo command in this plan carries it. Both target models are DeltaNet hybrids, and more immediately: `cargo test -p rdna-compute` **cannot build without it** on `beta`. The example `rope_compact_offset_check` calls `rope_partial_interleaved_f32{,_batched}`, which are `#[cfg(feature = "deltanet")]` (`crates/rdna-compute/src/norm.rs:711` and `:966`), but the crate declares no `required-features` for it.
- **That example breakage is PRE-EXISTING on `origin/beta` and is NOT ours to fix.** Do not "repair" it, do not add `required-features`, do not edit `rope_compact_offset_check`. If you see four `E0599: no method named rope_partial_interleaved_f32...` errors, you dropped `--features deltanet`. Verified baseline **with** the flag: `118 passed, 0 failed, 1 ignored`.
- **Build check after every Rust change:** `cargo build --release -p rdna-compute --features deltanet`.
- **Commit after every task.** No task may be left half-committed.

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `kernels/src/kv_slot_desc.h` | `KvSlotDesc` struct + `kv_offset_for()` device helper. The single paged seam. | create |
| `crates/rdna-compute/src/kv_slots.rs` | Host-side `KvSlotDesc` mirror, tile-list builder, and (from Task 7) the shared arena builder used by both the correctness gate and the benchmark. | create |
| `crates/rdna-compute/src/lib.rs` | Register the new module. | modify |
| `kernels/src/attention_q8_0_kv_batched.hip` | LDS decode/verify kernel — accept descriptor args. | modify |
| `kernels/src/attention_flash_q8_0_tile_batched.hip` | Q8 tile kernel — accept descriptor args. | modify |
| `kernels/src/attention_flash_asym3_tile_batched.hip` | asym3 tile kernel — accept descriptor args. | modify |
| `kernels/src/attention_q8_0_flash_prefill.hip` | FA-2 prefill kernel — accept descriptor args. | modify |
| `crates/rdna-compute/src/attention.rs` | Launcher plumbing for all four kernels. | modify |
| `crates/rdna-compute/examples/test_batched_attn_slots.rs` | Correctness harness: golden, isolation, adversarial shapes. | create |
| `crates/rdna-compute/examples/q8_batched_attn_microbench.rs` | Extend with a multi-slot sweep. | modify |
| `crates/rdna-compute/examples/probe_batching_ceiling.rs` | Task 0 latency-vs-context slope probe. | create |
| `docs/perf-checkpoints/2026-08-07-batching-ceiling-probe.md` | Task 0 results. | create |
| `docs/perf-checkpoints/2026-08-07-asym3-quality-gate.md` | Quality gate results. | create |

**Task order rationale:** Task 1 (Task 0 in spec numbering) runs first because its result can cancel or re-aim the rest. Tasks 2–3 build the ABI. Tasks 4–7 port one kernel each — each independently reviewable and revertible. Task 8 benches. Task 9 gates asym3 quality.

---

### Task 1: Batching-ceiling probe (spec §12, "Task 0")

Validates or replaces the roofline estimates *before* kernel work. Measures whole-step decode latency versus context length at batch 1 and fits a line; the intercept is the context-independent term (weights), the slope is the KV/attention term.

**Files:**
- Create: `crates/rdna-compute/examples/probe_batching_ceiling.rs`
- Create: `docs/perf-checkpoints/2026-08-07-batching-ceiling-probe.md`

**Interfaces:**
- Consumes: nothing.
- Produces: measured `a` (ms, context-independent) and `b` (ms per 1K context) per model; a corrected version of spec §8. No code consumed by later tasks.

**Why no per-op timing:** per-operation `device_synchronize` fabricates GPU speedups. The slope fit needs only whole-step wall time, which is why it is the chosen method.

- [ ] **Step 1: Write the probe harness**

Create `crates/rdna-compute/examples/probe_batching_ceiling.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Task 0 of the SP1 batched-attention plan: measure the batching ceiling
// empirically instead of trusting roofline arithmetic.
//
// Method: whole-step decode latency at batch 1 across context lengths, then a
// least-squares fit of t(ctx) = a + b*ctx. `a` is the context-independent term
// (weights, DeltaNet, dense projections); `b*ctx` is the KV/attention term.
// Predicted batched step time at N slots is a_amortised + N*b*ctx.
//
// There is deliberately NO per-operation device_synchronize here: per-op syncs
// fabricate GPU speedups and would corrupt the fit. Only whole-step wall time
// is measured.
//
// Env: CTXS (comma-separated context lengths), ITERS, WARMUPS.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Least-squares fit of y = a + b*x. Returns (a, b).
fn linfit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    assert!(denom.abs() > 1e-9, "context lengths must not all be equal");
    let b = (n * sxy - sx * sy) / denom;
    let a = (sy - b * sx) / n;
    (a, b)
}

fn main() {
    let ctxs: Vec<usize> = std::env::var("CTXS")
        .unwrap_or_else(|_| "4096,16384,32768,65536".into())
        .split(',')
        .map(|s| s.trim().parse().expect("CTXS must be integers"))
        .collect();
    let iters = env_usize("ITERS", 9);
    let warmups = env_usize("WARMUPS", 3);

    // Attention-only proxy for a decode step: one FA layer's worth of work at
    // batch 1, repeated `layers` times. Shapes default to qwen3.6-35b-a3b's
    // full-attention layers (nh=16, nkv=2, hd=256, 10 FA layers).
    let nh = env_usize("NH", 16);
    let nkv = env_usize("NKV", 2);
    let hd = env_usize("HD", 256);
    let layers = env_usize("LAYERS", 10);

    let mut gpu = Gpu::init().expect("gpu init");
    let blocks_per_head = hd / 32;
    let bytes_per_pos = nkv * blocks_per_head * 34;

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for &ctx in &ctxs {
        let cache_bytes = ctx * bytes_per_pos;
        let mut kv = vec![0u8; cache_bytes];
        for blk in kv.chunks_mut(34) {
            blk[0] = 0x00;
            blk[1] = 0x3C; // fp16 1.0
            for (j, b) in blk[2..].iter_mut().enumerate() {
                *b = ((j as i32 % 7) - 3) as i8 as u8;
            }
        }
        let k_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("k upload");
        let v_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("v upload");

        let q_data: Vec<f32> = (0..nh * hd).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let q = gpu.upload_f32(&q_data, &[nh * hd]).expect("q upload");
        let out = gpu.zeros(&[nh * hd], DType::F32).expect("out");

        // positions are i32 bits uploaded through upload_raw — there is no
        // upload_i32 on Gpu. This matches q8_batched_attn_microbench.rs.
        let pos_data: Vec<i32> = vec![(ctx - 1) as i32];
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, 4) };
        let positions = gpu.upload_raw(pos_bytes, &[1]).expect("pos upload");

        let stride = 2 + hd;
        let max_tiles = ctx.div_ceil(128);
        let partials = gpu
            .zeros(&[nh * max_tiles * stride], DType::F32)
            .expect("partials");

        let mut run = |g: &mut Gpu| {
            for _ in 0..layers {
                g.attention_flash_q8_0_batched_masked(
                    &q, &k_cache, &v_cache, &out, &positions,
                    nh, nkv, hd, ctx, ctx, 1, &partials, None, 0, 0,
                )
                .expect("attn");
            }
        };

        for _ in 0..warmups {
            run(&mut gpu);
        }
        gpu.hip.device_synchronize().unwrap();

        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            run(&mut gpu);
            // One sync per WHOLE measured block, never per kernel. Per-op syncs
            // fabricate GPU speedups and would corrupt the slope fit.
            gpu.hip.device_synchronize().unwrap();
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples[samples.len() / 2];

        println!("ctx={ctx:>7}  median_ms={median:8.3}");
        xs.push(ctx as f64);
        ys.push(median);
    }

    let (a, b) = linfit(&xs, &ys);
    println!();
    println!("fit: t(ctx) = {a:.4} ms + {:.6} ms per 1K ctx", b * 1000.0);
    println!("  a (context-independent, does amortise across slots): {a:.4} ms");
    println!("  b (KV term, does NOT amortise across slots):         {:.6} ms/1K", b * 1000.0);
    for n in [2usize, 4, 8] {
        for &ctx in &ctxs {
            let seq = (a + b * ctx as f64) * n as f64;
            let bat = a + b * ctx as f64 * n as f64;
            println!("  N={n} ctx={ctx:>7}: seq={seq:8.3}ms batched={bat:8.3}ms speedup={:.2}x", seq / bat);
        }
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build --release -p rdna-compute --features deltanet --example probe_batching_ceiling`
Expected: compiles clean. The `Gpu` API used here is `upload_f32`, `upload_raw`, `zeros`, and `gpu.hip.device_synchronize()` — there is no `alloc_tensor` for zeroed buffers, no `upload_i32`, and no `Gpu::synchronize`. `crates/rdna-compute/examples/q8_batched_attn_microbench.rs` is the reference idiom.

- [ ] **Step 3: Run the probe**

Run: `cargo run --release -p rdna-compute --features deltanet --example probe_batching_ceiling`
Expected: four `ctx=... median_ms=...` lines with monotonically increasing times, then a fit line with positive `a` and positive `b`.

If `b` comes out near zero or negative, the measurement is broken — most likely the kernel is not actually reading the full context. Do not proceed; investigate first.

- [ ] **Step 4: Run for the 27B shape**

Run: `NH=24 NKV=4 HD=256 LAYERS=16 cargo run --release -p rdna-compute --features deltanet --example probe_batching_ceiling`
Expected: a noticeably steeper `b` than the 35B shape — the 27B moves 32 KB/token against the 35B's 10 KB.

- [ ] **Step 5: Write the results note**

Create `docs/perf-checkpoints/2026-08-07-batching-ceiling-probe.md` containing: the raw `ctx`/`median_ms` tables for both shapes, the fitted `a` and `b`, the predicted speedups at N=2/4/8, and an explicit statement of whether spec §8's ~3.3× (27B) and ~1.8× (35B) survive. State plainly that this measures the attention term only — the weights term is inferred, not measured, because SP1 does not run a full model forward.

- [ ] **Step 6: Commit**

```bash
git add crates/rdna-compute/examples/probe_batching_ceiling.rs docs/perf-checkpoints/2026-08-07-batching-ceiling-probe.md
git commit -m "perf(attn): empirical batching-ceiling probe

Latency-vs-context slope fit at batch 1, separating the term that
amortises across slots from the KV term that does not. No per-op
device_synchronize — those fabricate GPU speedups."
```

---

### Task 2: `KvSlotDesc` device header and `kv_offset_for()`

The paged seam. Every KV address in every ported kernel goes through this one helper.

**Files:**
- Create: `kernels/src/kv_slot_desc.h`

**Interfaces:**
- Consumes: nothing.
- Produces: `struct KvSlotDesc { unsigned long long k_base; unsigned long long v_base; int seq_len; int cap; }` and `__device__ unsigned long long kv_offset_for(const KvSlotDesc&, int pos, int per_pos_bytes)`. Tasks 4–7 include this header. Task 3 mirrors the struct layout in Rust.

- [ ] **Step 1: Write the header**

Create `kernels/src/kv_slot_desc.h`:

```c
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Multi-slot KV descriptor — the single point of KV address translation for
// every batched attention kernel.
//
// A "slot" is one independent sequence in a batch. Today each slot owns a
// contiguous slab of `cap` tokens inside a per-layer arena, so translation is
// base + pos*stride. When this moves to paged block tables, ONLY
// kv_offset_for() changes — no kernel is touched. That is the entire reason
// this indirection exists; do not inline it away.
//
// Layout must stay byte-identical to the Rust mirror in
// crates/rdna-compute/src/kv_slots.rs. 24 bytes, 8-byte aligned.

#ifndef HIPFIRE_KV_SLOT_DESC_H
#define HIPFIRE_KV_SLOT_DESC_H

struct KvSlotDesc {
    unsigned long long k_base;  // byte offset of this slot's K slab in the arena
    unsigned long long v_base;  // byte offset of this slot's V slab in the arena
    int seq_len;                // logical KV length; kernel reads [0, seq_len)
    int cap;                    // physical slab capacity in tokens; seq_len <= cap
};

// Byte offset of position `pos` within slot `s`'s K slab.
// `per_pos_bytes` is the per-position stride in bytes, uniform across slots
// (n_kv_heads * (head_dim/32) * 34 for Q8_0).
__device__ __forceinline__ unsigned long long kv_offset_for_k(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    return s.k_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

__device__ __forceinline__ unsigned long long kv_offset_for_v(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    return s.v_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

// Single-slot fallback used when the descriptor pointer is null. Keeps the
// ported kernels on ONE code path: callers build this on the stack from the
// legacy scalar args, so there is no `if (descs) ... else ...` around every
// KV read. Behaviour is then bitwise identical to the pre-SP1 kernel.
__device__ __forceinline__ KvSlotDesc kv_slot_legacy(int seq_len, int max_seq)
{
    KvSlotDesc s;
    s.k_base = 0ULL;
    s.v_base = 0ULL;
    s.seq_len = seq_len;
    s.cap = max_seq;
    return s;
}

#endif  // HIPFIRE_KV_SLOT_DESC_H
```

- [ ] **Step 2: Verify the header compiles standalone**

Run:
```bash
cd ~/repos/hipfire-batchattn && echo '#include "kernels/src/kv_slot_desc.h"
__global__ void probe(const KvSlotDesc* d, unsigned long long* o){ *o = kv_offset_for_k(d[0], 5, 1088); }' > /tmp/kvdesc_probe.hip && hipcc -c /tmp/kvdesc_probe.hip -o /tmp/kvdesc_probe.o --offload-arch=gfx1151 && echo COMPILE_OK
```
Expected: `COMPILE_OK`.

If `hipcc` is not on PATH, source the ROCm environment first — a missing `hipcc` is an environment problem, not a code problem.

- [ ] **Step 3: Verify the struct is 24 bytes**

Run:
```bash
cd ~/repos/hipfire-batchattn && echo '#include <cstdio>
#include "kernels/src/kv_slot_desc.h"
int main(){ printf("%zu %zu\n", sizeof(KvSlotDesc), alignof(KvSlotDesc)); }' > /tmp/kvdesc_size.cpp && g++ /tmp/kvdesc_size.cpp -o /tmp/kvdesc_size && /tmp/kvdesc_size
```
Expected: `24 8`

If it prints anything else, the Rust mirror in Task 3 must match whatever it prints — but 24/8 is what the layout above should give, and a different answer means a field was changed.

- [ ] **Step 4: Commit**

```bash
git add kernels/src/kv_slot_desc.h
git commit -m "feat(attn): KvSlotDesc device header and kv_offset_for helper

The single point of KV address translation for batched attention. Moving
to paged block tables later changes only this file."
```

---

### Task 3: Host-side slot table and tile-list builder

**Files:**
- Create: `crates/rdna-compute/src/kv_slots.rs`
- Modify: `crates/rdna-compute/src/lib.rs`

**Interfaces:**
- Consumes: the 24-byte layout from Task 2.
- Produces:
  - `pub struct KvSlotDesc { pub k_base: u64, pub v_base: u64, pub seq_len: i32, pub cap: i32 }` (`#[repr(C)]`)
  - `pub fn build_tiles(slot_query_counts: &[usize], br: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>)` returning `(tile_slot, tile_row0, tile_qbase)`
  - `pub fn total_rows(slot_query_counts: &[usize]) -> usize`

`tile_qbase` is included from the start because the prefill kernel in Task 6 needs it: `tile_row0` is slot-relative (it restarts at 0 for each slot) but `q` and `out` are indexed by the global flat row.

- [ ] **Step 1: Write the failing test**

Create `crates/rdna-compute/src/kv_slots.rs` with the struct, a stub `build_tiles` that returns empty vectors, and these tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desc_is_24_bytes() {
        assert_eq!(std::mem::size_of::<KvSlotDesc>(), 24);
        assert_eq!(std::mem::align_of::<KvSlotDesc>(), 8);
    }

    #[test]
    fn tiles_never_span_a_slot() {
        // 3 slots with 1, 3 and 8 query rows; BR = 4.
        // Slot 0 -> 1 tile, slot 1 -> 1 tile, slot 2 -> 2 tiles. Total 4.
        let (tile_slot, tile_row0, _) = build_tiles(&[1, 3, 8], 4);
        assert_eq!(tile_slot, vec![0, 1, 2, 2]);
        assert_eq!(tile_row0, vec![0, 0, 0, 4]);
    }

    #[test]
    fn tile_qbase_is_the_global_flat_row() {
        // Same shape: global flat rows are 0 | 1,2,3 | 4..11, so the four
        // tiles start at global rows 0, 1, 4 and 8.
        let (_, _, tile_qbase) = build_tiles(&[1, 3, 8], 4);
        assert_eq!(tile_qbase, vec![0, 1, 4, 8]);
    }

    #[test]
    fn br_one_gives_one_tile_per_row() {
        let (tile_slot, tile_row0, tile_qbase) = build_tiles(&[1, 1, 1, 1], 1);
        assert_eq!(tile_slot, vec![0, 1, 2, 3]);
        assert_eq!(tile_row0, vec![0, 0, 0, 0]);
        assert_eq!(tile_qbase, vec![0, 1, 2, 3]);
    }

    #[test]
    fn zero_query_slots_produce_no_tiles() {
        // A slot with nothing to do this step must not get a tile — an empty
        // tile would read uninitialised Q and write garbage to out.
        // Slot 2's rows still start at global row 2, after slot 0's two rows.
        let (tile_slot, tile_row0, tile_qbase) = build_tiles(&[2, 0, 3], 4);
        assert_eq!(tile_slot, vec![0, 2]);
        assert_eq!(tile_row0, vec![0, 0]);
        assert_eq!(tile_qbase, vec![0, 2]);
    }

    #[test]
    fn total_rows_sums_query_counts() {
        assert_eq!(total_rows(&[1, 3, 8]), 12);
        assert_eq!(total_rows(&[]), 0);
    }

    #[test]
    fn mixed_prefill_and_decode_batch() {
        // The shape SP1 exists for: slot 0 verifies 8 draft tokens, slot 1
        // chunk-prefills 256, slots 2-3 decode 1 each. BR = 8.
        let (tile_slot, _, _) = build_tiles(&[8, 256, 1, 1], 8);
        assert_eq!(tile_slot.iter().filter(|&&s| s == 0).count(), 1);
        assert_eq!(tile_slot.iter().filter(|&&s| s == 1).count(), 32);
        assert_eq!(tile_slot.iter().filter(|&&s| s == 2).count(), 1);
        assert_eq!(tile_slot.iter().filter(|&&s| s == 3).count(), 1);
        assert_eq!(tile_slot.len(), 35);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --release -p rdna-compute --features deltanet kv_slots`
Expected: FAIL — `tiles_never_span_a_slot` and the others fail on empty vectors returned by the stub.

- [ ] **Step 3: Write the implementation**

Replace the stub in `crates/rdna-compute/src/kv_slots.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Host-side mirror of the multi-slot KV descriptor and the flat row-tile list
// that drives batched attention launches.
//
// A "row tile" is up to BR consecutive query rows belonging to ONE slot. No
// tile may span a slot boundary — a workgroup owns one tile and reads one
// slot's KV, so a straddling tile would read the wrong sequence's cache.

/// Byte-identical mirror of `struct KvSlotDesc` in `kernels/src/kv_slot_desc.h`.
/// 24 bytes, 8-byte aligned. Changing either side without the other silently
/// corrupts every KV address.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSlotDesc {
    /// Byte offset of this slot's K slab within the layer's K arena.
    pub k_base: u64,
    /// Byte offset of this slot's V slab within the layer's V arena.
    pub v_base: u64,
    /// Logical KV length. The kernel reads positions `[0, seq_len)`.
    pub seq_len: i32,
    /// Physical slab capacity in tokens. Invariant: `seq_len <= cap`.
    pub cap: i32,
}

/// Total query rows across all slots.
pub fn total_rows(slot_query_counts: &[usize]) -> usize {
    slot_query_counts.iter().sum()
}

/// Build the flat tile list. Returns `(tile_slot, tile_row0, tile_qbase)`:
///
/// - `tile_slot[t]`  — slot index owning tile `t`
/// - `tile_row0[t]`  — first query row of tile `t` *within its slot*
/// - `tile_qbase[t]` — first query row of tile `t` in the *global* flat row
///   space, which is how `q` and `out` are indexed
///
/// Both row indices are needed: KV addressing is slot-relative (via the
/// descriptor's `seq_len`) while Q/out addressing is global. Conflating them
/// makes slot 0 correct and every later slot read the wrong query.
///
/// Slots with zero query rows produce no tiles — an empty tile would read
/// uninitialised Q and write garbage into `out`.
pub fn build_tiles(
    slot_query_counts: &[usize],
    br: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    assert!(br > 0, "br must be positive");
    let mut tile_slot = Vec::new();
    let mut tile_row0 = Vec::new();
    let mut tile_qbase = Vec::new();
    let mut global = 0usize;
    for (slot, &m) in slot_query_counts.iter().enumerate() {
        let mut row0 = 0usize;
        while row0 < m {
            tile_slot.push(slot as i32);
            tile_row0.push(row0 as i32);
            tile_qbase.push((global + row0) as i32);
            row0 += br;
        }
        global += m;
    }
    (tile_slot, tile_row0, tile_qbase)
}
```

Then add to `crates/rdna-compute/src/lib.rs`, alongside the other `pub mod` declarations:

```rust
pub mod kv_slots;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --release -p rdna-compute --features deltanet kv_slots`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/rdna-compute/src/kv_slots.rs crates/rdna-compute/src/lib.rs
git commit -m "feat(attn): host-side KV slot table and row-tile builder

Tiles never span a slot boundary; zero-query slots produce no tiles."
```

---

### Task 4: Port `attention_q8_0_kv_batched` (LDS decode/verify path)

**Files:**
- Modify: `kernels/src/attention_q8_0_kv_batched.hip`
- Modify: `crates/rdna-compute/src/attention.rs:1584-1680` (`attention_q8_0_kv_batched_masked`)

**Interfaces:**
- Consumes: `KvSlotDesc` / `kv_offset_for_k` / `kv_offset_for_v` / `kv_slot_legacy` (Task 2); `KvSlotDesc` Rust mirror (Task 3).
- Produces: `Gpu::attention_q8_0_kv_batched_masked_slots(..., slot_descs: Option<&GpuTensor>, row_slot: Option<&GpuTensor>)`. The existing `attention_q8_0_kv_batched_masked` stays, delegating with `None, None`.

**Critical:** with `slot_descs == nullptr` the kernel must produce bitwise-identical output to today. That is what Step 5 checks.

- [ ] **Step 1: Add the descriptor args to the kernel**

In `kernels/src/attention_q8_0_kv_batched.hip`, add `#include "kv_slot_desc.h"` after the `hip_runtime.h` include, then append two parameters to the kernel signature (append only — never reorder, the kernarg blob is positional):

```c
    int block_cols,                         // ignored when tree_bias == nullptr
    const KvSlotDesc* __restrict__ slot_descs,  // [n_slots] or nullptr = legacy
    const int* __restrict__ row_slot            // [batch_size] or nullptr = legacy
) {
```

- [ ] **Step 2: Route KV reads through the descriptor**

Immediately after `const int seq_len = ...` in the kernel body, insert:

```c
    // One code path for both modes: in legacy mode we synthesise a descriptor
    // with zero bases, so the address arithmetic below is unchanged and the
    // output is bitwise identical to the pre-SP1 kernel.
    const int slot = (row_slot != nullptr) ? row_slot[b] : 0;
    const KvSlotDesc desc = (slot_descs != nullptr)
        ? slot_descs[slot]
        : kv_slot_legacy(seq_len, max_seq);
    const int per_pos_bytes = n_kv_heads * (head_dim / 32) * 34;
```

Then replace the K read at the `Phase 1` loop:

```c
            const unsigned char* blk = k_cache
                + kv_offset_for_k(desc, t, per_pos_bytes)
                + (kv_head_block_start + bi) * 34;
```

Apply the identical substitution to the V read later in the kernel, using `kv_offset_for_v` and `v_cache`.

Also override `seq_len` when a descriptor is present, since the slot's own length is authoritative:

```c
    const int eff_seq_len = (slot_descs != nullptr) ? desc.seq_len : seq_len;
```

and use `eff_seq_len` in place of `seq_len` in every loop bound *except* the shared-memory pointer arithmetic (`workspace = sdata + seq_len`), which must keep using the launch-wide value so per-row LDS slices stay within the allocation.

- [ ] **Step 3: Add the launcher variant**

In `crates/rdna-compute/src/attention.rs`, rename the existing body to `attention_q8_0_kv_batched_masked_slots` with two extra trailing parameters, and push them onto both the `params` vector and the `KernargBlob`:

```rust
        let mut desc_ptr: *mut std::ffi::c_void = match slot_descs {
            Some(t) => t.buf.as_ptr(),
            None => std::ptr::null_mut(),
        };
        let mut rs_ptr: *mut std::ffi::c_void = match row_slot {
            Some(t) => t.buf.as_ptr(),
            None => std::ptr::null_mut(),
        };
```

appended to `params` after `bc`, and in the blob closure after `b.push_i32(bc)`:

```rust
                b.push_ptr(desc_raw);
                b.push_ptr(rs_raw);
```

Then keep the original entry point as a thin delegate:

```rust
    /// Legacy single-sequence entry point. Preserved so existing call sites
    /// are untouched; passes null descriptors, which the kernel treats as
    /// legacy mode with bitwise-identical output.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_q8_0_kv_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.attention_q8_0_kv_batched_masked_slots(
            q, k_cache, v_cache, out, positions, n_heads, n_kv_heads, head_dim,
            max_seq, max_ctx_len, batch_size, tree_bias, block_start, block_cols,
            None, None,
        )
    }
```

- [ ] **Step 4: Build**

Run: `cargo build --release -p rdna-compute --features deltanet`
Expected: compiles clean, no changes needed at any existing call site.

- [ ] **Step 5: Verify legacy mode is bitwise identical**

Run: `cargo run --release -p rdna-compute --features deltanet --example test_q8_flash_prefill`
Expected: same pass output as on a clean `origin/beta` checkout. Capture both and compare:

```bash
git stash && cargo run --release -p rdna-compute --features deltanet --example test_q8_flash_prefill > /tmp/before.txt 2>&1; git stash pop
cargo run --release -p rdna-compute --features deltanet --example test_q8_flash_prefill > /tmp/after.txt 2>&1
diff /tmp/before.txt /tmp/after.txt && echo BITWISE_IDENTICAL
```
Expected: `BITWISE_IDENTICAL`.

If they differ, stop. A legacy-mode change means the descriptor path leaked into the null case, and every later task builds on this being clean.

- [ ] **Step 6: Commit**

```bash
git add kernels/src/attention_q8_0_kv_batched.hip crates/rdna-compute/src/attention.rs
git commit -m "feat(attn): slot descriptors for the Q8 LDS batched kernel

Null descriptor = legacy mode, verified bitwise identical."
```

---

### Task 5: Port `attention_flash_q8_0_tile_batched` and the shared launcher

This is the long-context path — the one that actually runs at agent context lengths, since `LDS_CTX_LIMIT = 15000`.

**Files:**
- Modify: `kernels/src/attention_flash_q8_0_tile_batched.hip`
- Modify: `crates/rdna-compute/src/attention.rs:3437+` (`launch_asym_flash_batched`)

**Interfaces:**
- Consumes: Task 2's header; Task 3's Rust mirror.
- Produces: `launch_asym_flash_batched` gains trailing `slot_descs: Option<&GpuTensor>, row_slot: Option<&GpuTensor>` parameters; `Gpu::attention_flash_q8_0_batched_masked_slots(...)` exposes them. All six existing asym/fwht wrappers pass `None, None` unchanged.

**Note:** the tile kernel's grid is `[n_heads, max_tiles, chunk]` and the launcher already loops over `batch_offset` when `partials` capacity forces sub-batching. The `row_slot` lookup must therefore use the **global** row index `batch_offset + blockIdx.z`, not `blockIdx.z` alone. Getting this wrong produces correct results at small batch and silent cross-slot corruption at large batch — precisely the bug Task 7's isolation test exists to catch.

- [ ] **Step 1: Add the descriptor args to the tile kernel**

In `kernels/src/attention_flash_q8_0_tile_batched.hip`, add `#include "kv_slot_desc.h"`, then append to the signature after `int window`:

```c
    int window,                                // sliding-window span; <= 0 = full causal
    const KvSlotDesc* __restrict__ slot_descs, // [n_slots] or nullptr = legacy
    const int* __restrict__ row_slot           // [total_batch] or nullptr = legacy
) {
```

In the body, after the existing row index is computed, add:

```c
    // batch_offset is the sub-batch base the launcher is currently emitting.
    // row_slot is indexed by GLOBAL row, so it must include that offset —
    // using the local index silently reads the wrong slot once the partials
    // buffer forces sub-batching.
    const int global_row = batch_offset + (int)blockIdx.z;
    const int slot = (row_slot != nullptr) ? row_slot[global_row] : 0;
    const KvSlotDesc desc = (slot_descs != nullptr)
        ? slot_descs[slot]
        : kv_slot_legacy(positions[global_row] + 1, max_seq);
    const int per_pos_bytes = n_kv_heads * (head_dim / 32) * 34;
```

Replace every `k_cache + (size_t)<pos> * ...` with `k_cache + kv_offset_for_k(desc, <pos>, per_pos_bytes) + <head-block offset>`, and the same for `v_cache` with `kv_offset_for_v`.

- [ ] **Step 2: Thread the parameters through the shared launcher**

In `launch_asym_flash_batched`, add two trailing parameters:

```rust
        slot_descs: Option<&GpuTensor>,
        row_slot: Option<&GpuTensor>,
```

Push them as the last two kernargs in the non-WMMA branch only. The WMMA branch omits `v_mode_bits` and has its own kernarg layout — leave it untouched and assert it is not reached with descriptors:

```rust
        assert!(
            !(use_wmma_grid && slot_descs.is_some()),
            "multi-slot descriptors are not supported on the WMMA tile grid; \
             the WMMA kernarg layout differs and asym4-WMMA is out of SP1 scope"
        );
```

- [ ] **Step 3: Update all existing callers of the shared launcher**

Every current caller — `attention_flash_q8_0_batched_masked`, `attention_flash_asym3_batched`, `attention_flash_asym3_batched_masked`, `attention_flash_asym2_batched`, `attention_flash_asym4_batched`, `attention_flash_asym4_batched_masked`, `attention_flash_fwht{2,3,4}_batched*`, and the windowed variant — gains `None, None` at the end of its call.

Run this to enumerate them so none is missed:
```bash
grep -n "launch_asym_flash_batched(" crates/rdna-compute/src/attention.rs
```

- [ ] **Step 4: Add the slots entry point**

```rust
    /// Multi-slot Q8_0 tiled flash attention. `slot_descs` is `[n_slots]`
    /// `KvSlotDesc`; `row_slot` is `[batch_size]` slot indices per query row.
    /// Passing `None` for both is exactly the legacy single-sequence path.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_q8_0_batched_masked_slots(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
        slot_descs: Option<&GpuTensor>,
        row_slot: Option<&GpuTensor>,
    ) -> HipResult<()> {
        self.launch_asym_flash_batched(
            "attention_flash_q8_0_tile_batched",
            kernels::ATTENTION_FLASH_Q8_0_TILE_BATCHED_SRC,
            "attention_flash_q8_0_tile_batched",
            q, k_cache, v_cache, out, positions, q, q,
            n_heads, n_kv_heads, head_dim, max_seq, max_ctx_len, batch_size,
            partials, tree_bias, block_start, block_cols, 0, -1, false,
            slot_descs, row_slot,
        )
    }
```

Match the exact argument order of the existing `attention_flash_q8_0_batched_masked` body — copy it and add the two trailing arguments rather than retyping from this plan, since the `cos_theta`/`sin_theta` dummies and `v_mode_bits` value must stay as they are.

- [ ] **Step 5: Build and verify legacy is unchanged**

Run:
```bash
cargo build --release -p rdna-compute --features deltanet
cargo run --release -p rdna-compute --features deltanet --example q8_batched_attn_microbench > /tmp/mb_after.txt 2>&1
```
Expected: builds clean; the microbench still runs and reports both arms.

- [ ] **Step 6: Commit**

```bash
git add kernels/src/attention_flash_q8_0_tile_batched.hip crates/rdna-compute/src/attention.rs
git commit -m "feat(attn): slot descriptors on the shared asym/q8 tile launcher

row_slot is indexed by GLOBAL row so sub-batching cannot desync it.
WMMA grid asserted out of scope — different kernarg layout."
```

---

### Task 6: Port `attention_flash_asym3_tile_batched` and `attention_q8_0_flash_prefill`

Two kernels, one task: asym3 rides the launcher already modified in Task 5, so its change is confined to the `.hip` file, and the prefill kernel is the same mechanical edit against a different grid.

**Files:**
- Modify: `kernels/src/attention_flash_asym3_tile_batched.hip`
- Modify: `kernels/src/attention_q8_0_flash_prefill.hip`
- Modify: `crates/rdna-compute/src/attention.rs` (prefill launcher at `:1822`; asym3 wrapper at `:3788`)

**Interfaces:**
- Consumes: Tasks 2, 3, 5.
- Produces: `Gpu::attention_flash_asym3_batched_masked_slots(...)` and `Gpu::attention_q8_0_flash_prefill_slots(..., slot_descs, tile_slot, tile_row0)`.

**asym3 stride note:** asym3 K is *not* 34-byte Q8_0 blocks — it is 3-bit Givens-rotated. Its per-position K stride differs from V's, which stays Q8_0. Find both before editing:

```bash
grep -n "k_cache +\|v_cache +\|per_pos\|stride" kernels/src/attention_flash_asym3_tile_batched.hip
```

Reuse the kernel's own existing stride expressions verbatim as the `per_pos_bytes` arguments — pass the K stride to `kv_offset_for_k` and the V stride to `kv_offset_for_v`. Do **not** copy the Q8 value of `n_kv_heads * (head_dim/32) * 34`; using it for K would compute addresses roughly 2.7× too large and read past the arena.

**Prefill grid note:** the prefill kernel's grid is `[batch.div_ceil(br), n_heads]`, so `blockIdx.x` is the tile index. Use `tile_slot[blockIdx.x]` and `tile_row0[blockIdx.x]` — this is the kernel that genuinely needs the tile arrays from Task 3, because BR > 1 there.

- [ ] **Step 1: Port the asym3 tile kernel**

Add `#include "kv_slot_desc.h"` and the same two trailing parameters as Task 5 Step 1. Compute `slot`/`desc` from `batch_offset + blockIdx.z` exactly as in Task 5. Replace the K and V base addressing with `kv_offset_for_k` / `kv_offset_for_v`, using the kernel's own existing per-position stride expression for `per_pos_bytes`.

- [ ] **Step 2: Add the asym3 slots wrapper**

Copy the body of the existing `attention_flash_asym3_batched_masked` into a `_slots` variant with trailing `slot_descs` / `row_slot`, forwarding them to `launch_asym_flash_batched`; make the original delegate with `None, None`.

- [ ] **Step 3: Port the prefill kernel**

In `kernels/src/attention_q8_0_flash_prefill.hip`, add `#include "kv_slot_desc.h"` and three trailing parameters:

```c
    const KvSlotDesc* __restrict__ slot_descs,  // [n_slots] or nullptr
    const int* __restrict__ tile_slot,          // [n_tiles] or nullptr
    const int* __restrict__ tile_row0           // [n_tiles] or nullptr
) {
```

In the body, where the tile's first query row is currently derived from `blockIdx.x * BR`:

```c
    const int tile = blockIdx.x;
    const int slot = (tile_slot != nullptr) ? tile_slot[tile] : 0;
    // Legacy: row0 = tile * BR. Multi-slot: the builder supplies it, because
    // tiles restart at 0 for each slot.
    const int row0 = (tile_row0 != nullptr) ? tile_row0[tile] : (tile * BR);
    const KvSlotDesc desc = (slot_descs != nullptr)
        ? slot_descs[slot]
        : kv_slot_legacy(0, max_ctx_len);
    const int per_pos_bytes = n_kv_heads * (head_dim / 32) * 34;
```

and route the K/V tile reads through `kv_offset_for_k` / `kv_offset_for_v`.

**Careful — the two row indices are different.** KV addressing is slot-relative, but `q` and `out` are indexed by the *global* flat row. `tile_row0[t]` is slot-relative; `tile_qbase[t]` (already produced by Task 3) is the global one. Add a fourth kernel parameter for it:

```c
    const int* __restrict__ tile_qbase          // [n_tiles] or nullptr
) {
```

and in the body:

```c
    // Slot-relative row, for causal masking against this slot's own history.
    const int row0 = (tile_row0 != nullptr) ? tile_row0[tile] : (tile * BR);
    // Global flat row, for indexing q and out. In legacy mode these coincide.
    const int qbase = (tile_qbase != nullptr) ? tile_qbase[tile] : (tile * BR);
```

Use `qbase` for every `q`/`out` offset and `row0` only where the query's position within its own sequence matters. Conflating them leaves slot 0 correct and every later slot reading the wrong query — a bug that Task 7's golden test catches only because its slots hold *distinct* data.

- [ ] **Step 4: Build the slots entry point for prefill**

Add `Gpu::attention_q8_0_flash_prefill_slots(...)` taking trailing `slot_descs`, `tile_slot`, `tile_row0`, `tile_qbase` as `Option<&GpuTensor>`, with the existing `attention_q8_0_flash_prefill` delegating with four `None`s. The grid becomes `[n_tiles, n_heads]` where `n_tiles = tile_slot.len()` in multi-slot mode and `batch_size.div_ceil(br)` in legacy mode.

Run: `cargo test --release -p rdna-compute --features deltanet kv_slots`
Expected: PASS, 7 tests (unchanged from Task 3 — this task adds no new host-side tests).

- [ ] **Step 5: Build and check legacy prefill is unchanged**

Run:
```bash
cargo build --release -p rdna-compute --features deltanet
cargo run --release -p rdna-compute --features deltanet --example test_q8_flash_prefill > /tmp/after6.txt 2>&1
diff /tmp/before.txt /tmp/after6.txt && echo LEGACY_UNCHANGED
```
Expected: `LEGACY_UNCHANGED` against the Task 4 Step 5 baseline.

- [ ] **Step 6: Commit**

```bash
git add kernels/src/attention_flash_asym3_tile_batched.hip kernels/src/attention_q8_0_flash_prefill.hip crates/rdna-compute/src/attention.rs
git commit -m "feat(attn): slot descriptors for asym3 tile and Q8 flash prefill

Prefill takes both tile_row0 (slot-relative, for causal masking) and
tile_qbase (global flat row, for q/out indexing)."
```

---

### Task 7: Correctness harness — golden, isolation, adversarial shapes

The gate for the whole sub-project. Spec §9.

**Files:**
- Create: `crates/rdna-compute/examples/test_batched_attn_slots.rs`

**Interfaces:**
- Consumes: every `_slots` entry point from Tasks 4–6; `build_tiles` from Task 3.
- Produces: a pass/fail harness. No code consumed downstream.

- [ ] **Step 1: Write the golden-equivalence test**

First add the arena builder to `crates/rdna-compute/src/kv_slots.rs` so Task 8's benchmark uses the *same* layout as this correctness gate — two harnesses that disagree about the arena would measure and verify different things:

```rust
/// Build one KV arena holding `seq_lens.len()` contiguous slabs and the
/// matching descriptor table. Each slab is `cap` tokens; `cap` is rounded up
/// so a future page size divides it (spec §6.4).
///
/// `poison_except`: when `Some(target)`, every slab other than `target` is
/// filled with NaN-producing bytes. Used by the isolation test.
///
/// Slab contents vary by slot index — identical slabs would let a cross-slot
/// addressing bug pass by symmetry.
pub fn build_arena(
    seq_lens: &[usize],
    per_pos_bytes: usize,
    poison_except: Option<usize>,
) -> (Vec<u8>, Vec<KvSlotDesc>) {
    const PAGE_TOKENS: usize = 128; // == TILE_SIZE, so pages divide slabs later
    let mut arena = Vec::new();
    let mut descs = Vec::with_capacity(seq_lens.len());
    for (slot, &sl) in seq_lens.iter().enumerate() {
        let cap = sl.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;
        let base = arena.len() as u64;
        let poisoned = poison_except.is_some_and(|t| t != slot);
        for blk_idx in 0..(cap * per_pos_bytes / 34) {
            // f16 scale: 0x7E00 is NaN; otherwise a per-slot varying value.
            let (lo, hi) = if poisoned {
                (0x00u8, 0x7Eu8)
            } else {
                let h = half_from_f32(0.02 + (((blk_idx + slot * 7) % 13) as f32) * 0.005);
                ((h & 0xFF) as u8, (h >> 8) as u8)
            };
            arena.push(lo);
            arena.push(hi);
            for j in 0..32 {
                arena.push((((blk_idx * 31 + j * 17 + slot * 101) % 251) as i32 - 125) as i8 as u8);
            }
        }
        descs.push(KvSlotDesc {
            k_base: base,
            v_base: base, // K and V arenas are separate buffers, same offsets
            seq_len: sl as i32,
            cap: cap as i32,
        });
    }
    (arena, descs)
}
```

`half_from_f32` already exists in `crates/rdna-compute/examples/test_q8_flash_prefill.rs`; move it into `kv_slots.rs` as a `pub fn` and have the example use it from there rather than keeping two copies.

Then create `crates/rdna-compute/examples/test_batched_attn_slots.rs`. For a given `(n_slots, per-slot seq_len, per-slot M)` shape it must:

1. Call `build_arena` for K and V.
2. For each slot, call the legacy single-sequence kernel against that slot's slab in isolation, collecting reference outputs.
3. Build `descs` / `row_slot` / `positions` and make one batched multi-slot call.
4. Compare, requiring `max |batched - reference| <= 1e-3 * max(1.0, |reference|)`.

```rust
fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-3 * w.abs().max(1.0);
        let err = (g - w).abs() / tol;
        if err > worst {
            worst = err;
            worst_i = i;
        }
    }
    assert!(
        worst <= 1.0,
        "{label}: worst element {worst_i} at {worst:.2}x tolerance \
         (got {}, want {})",
        got[worst_i], want[worst_i]
    );
    println!("  {label}: OK (worst {worst:.3}x tolerance)");
}
```

- [ ] **Step 2: Write the cross-slot isolation test**

The sharpest instrument in the plan — it catches descriptor and stride bugs directly.

```rust
/// Fill every slot except `target` with NaN, then confirm `target`'s output is
/// unchanged. A wrong k_base, a wrong stride, or an off-by-one seq_len will
/// pull a NaN in and the comparison explodes immediately.
///
/// NaN is used rather than a large finite value on purpose: NaN propagates
/// through the softmax instead of being suppressed by the running max, so a
/// single leaked element is unmissable.
fn test_cross_slot_isolation(gpu: &mut Gpu, shape: &Shape) {
    let clean = run_batched(gpu, shape, /*poison=*/ None);
    for target in 0..shape.n_slots {
        let poisoned = run_batched(gpu, shape, Some(target));
        let a = slot_output(&clean, shape, target);
        let b = slot_output(&poisoned, shape, target);
        assert!(
            b.iter().all(|v| v.is_finite()),
            "slot {target}: NaN leaked in from a neighbouring slot"
        );
        assert_close(&format!("isolation slot {target}"), &b, &a);
    }
}
```

- [ ] **Step 3: Write the adversarial-shape sweep**

```rust
struct Shape {
    n_slots: usize,
    seq_lens: Vec<usize>,
    m_per_slot: Vec<usize>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

fn shapes() -> Vec<Shape> {
    let mut v = Vec::new();
    // GQA 8:1 — qwen3.6-35b-a3b full-attention layers.
    // GQA 6:1 — qwen3.6-27b full-attention layers.
    for &(nh, nkv) in &[(16usize, 2usize), (24, 4)] {
        // Wildly unequal context in one batch.
        v.push(Shape { n_slots: 4, seq_lens: vec![1, 512, 8192, 100_000],
                       m_per_slot: vec![1, 1, 1, 1], n_heads: nh, n_kv_heads: nkv, head_dim: 256 });
        // Mixed M: a zero-query slot, a decode, a small verify, a big verify.
        v.push(Shape { n_slots: 4, seq_lens: vec![4096, 4096, 4096, 4096],
                       m_per_slot: vec![0, 1, 3, 8], n_heads: nh, n_kv_heads: nkv, head_dim: 256 });
        // Mixed prefill + decode — the batch shape SP1 exists for.
        v.push(Shape { n_slots: 4, seq_lens: vec![32_768, 1024, 512, 512],
                       m_per_slot: vec![8, 256, 1, 1], n_heads: nh, n_kv_heads: nkv, head_dim: 256 });
        // seq_len below TILE_SIZE, and non-multiples of BR/BC.
        v.push(Shape { n_slots: 3, seq_lens: vec![7, 129, 131],
                       m_per_slot: vec![1, 5, 1], n_heads: nh, n_kv_heads: nkv, head_dim: 256 });
        // Slot-count sweep 1..=8 at a fixed modest context.
        for n in 1..=8usize {
            v.push(Shape { n_slots: n, seq_lens: vec![2048; n], m_per_slot: vec![1; n],
                           n_heads: nh, n_kv_heads: nkv, head_dim: 256 });
        }
    }
    v
}
```

Each shape runs golden equivalence and isolation, for **both** KV modes.

- [ ] **Step 4: Assert the asym3 arm is really asym3**

Guards the §4.4 trap — a silent fall back to q8 would make an unimplemented asym3 path pass everything.

```rust
// The asym3 arm allocates a 3-bit-K arena. If the code under test were
// silently running the Q8 path it would read 34-byte blocks out of a
// smaller buffer and produce garbage, not a clean pass — but assert the
// byte budget explicitly so the intent is recorded and a future refactor
// cannot quietly reintroduce the fallback.
assert!(
    asym3_bytes_per_pos < q8_bytes_per_pos,
    "asym3 arena is not smaller than Q8 — the asym3 path is not active \
     (see spec §4.4: QWEN35_PARO_POLICY silently downgrades asym3 to q8)"
);
```

- [ ] **Step 5: Run the harness**

Run: `cargo run --release -p rdna-compute --features deltanet --example test_batched_attn_slots`
Expected: every shape prints `OK` for golden and isolation on both KV modes, then a final `ALL SHAPES PASS` line.

The most likely failure is the Task 5 global-row bug: correct at small batch, wrong once `partials` capacity forces sub-batching. If isolation fails only at larger `n_slots`, look there first.

- [ ] **Step 6: Commit**

```bash
git add crates/rdna-compute/examples/test_batched_attn_slots.rs crates/rdna-compute/src/kv_slots.rs crates/rdna-compute/examples/test_q8_flash_prefill.rs
git commit -m "test(attn): golden, cross-slot isolation and adversarial shapes

NaN-poison isolation is the sharp instrument for descriptor and stride
bugs; golden equivalence is tolerance-based because tiling reorders
accumulation."
```

---

### Task 8: Multi-slot benchmark and `TILE_SIZE` sweep

Spec §10.

**Files:**
- Modify: `crates/rdna-compute/examples/q8_batched_attn_microbench.rs`
- Create: `docs/perf-checkpoints/2026-08-07-multislot-attention-bench.md`

**Interfaces:**
- Consumes: the `_slots` entry points.
- Produces: measured `TILE_SIZE` default and the batched-vs-sequential curve. No code consumed downstream.

- [ ] **Step 1: Make `TILE_SIZE` overridable**

In `crates/rdna-compute/src/attention.rs`, replace `const TILE_SIZE: usize = 128;` inside `launch_asym_flash_batched` with:

```rust
        // gfx1151 is the dev box; gfx1201 is the target. Never bake a tuned
        // constant into a `const` — see spec §11.
        let tile_size: usize = std::env::var("HIPFIRE_ATTN_TILE_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&t: &usize| t > 0 && t % 32 == 0)
            .unwrap_or(128);
```

and replace every use of `TILE_SIZE` in that function with `tile_size`. The LDS request `(TILE_SIZE * 4) as u32` becomes `(tile_size * 4) as u32`.

- [ ] **Step 2: Add the multi-slot sweep arm**

Append to `crates/rdna-compute/examples/q8_batched_attn_microbench.rs`, reusing the existing `time` closure defined at line 88:

```rust
    // ── Multi-slot sweep: batched vs sequential ─────────────────────────────
    // (A) one batched launch over n_slots, versus
    // (B) n_slots sequential single-slot launches.
    // Spec §2 criterion 2: batched must beat sequential at every n_slots >= 2.
    // A regression here is a failure, not a tuning outcome.
    for &n_slots in &[1usize, 2, 4, 8] {
        let per_slot_ctx = env_usize("SLOT_CTX", 32768);
        let shape = vec![per_slot_ctx; n_slots];
        let (batched_ms, seq_ms) = bench_slots(&mut gpu, &shape, &vec![1usize; n_slots]);
        println!(
            "n_slots={n_slots:2} ctx={per_slot_ctx:6} : batched {batched_ms:8.3} ms  \
             sequential {seq_ms:8.3} ms  speedup {:.2}x",
            seq_ms / batched_ms
        );
        if n_slots >= 2 {
            assert!(
                batched_ms < seq_ms,
                "batched ({batched_ms:.3} ms) must beat sequential \
                 ({seq_ms:.3} ms) at n_slots={n_slots} — spec §2 criterion 2"
            );
        }
    }

    // Ragged batch: max_tiles is derived from the batch MAXIMUM context, so
    // short slots launch tiles that immediately early-exit. Measure that waste
    // rather than assuming it is negligible (spec §7).
    {
        let ragged = vec![1024usize, 4096, 32768, 100_000];
        let uniform = vec![100_000usize; 4];
        let (ragged_ms, _) = bench_slots(&mut gpu, &ragged, &vec![1usize; 4]);
        let (uniform_ms, _) = bench_slots(&mut gpu, &uniform, &vec![1usize; 4]);
        let useful: usize = ragged.iter().sum();
        let launched = 100_000 * 4;
        println!(
            "ragged {ragged_ms:8.3} ms vs uniform-max {uniform_ms:8.3} ms  \
             (useful KV {useful}, tiles sized for {launched}, \
              waste {:.1}%)",
            100.0 * (1.0 - useful as f64 / launched as f64)
        );
    }
```

with this helper above `main`, built on `kv_slots::build_arena` and `kv_slots::build_tiles` from Task 7 — the bench and the correctness gate must share one arena layout or they are measuring and verifying different things:

```rust
/// Time one batched multi-slot launch against n_slots sequential single-slot
/// launches over the same arena. Returns (batched_ms, sequential_ms).
fn bench_slots(gpu: &mut Gpu, seq_lens: &[usize], m_per_slot: &[usize]) -> (f64, f64) {
    use rdna_compute::kv_slots::{build_arena, build_tiles, KvSlotDesc};

    let nh = env_usize("NH", 16);
    let nkv = env_usize("NKV", 2);
    let hd = env_usize("HD", 256);
    let per_pos_bytes = nkv * (hd / 32) * 34;

    let (arena, descs) = build_arena(seq_lens, per_pos_bytes, None);
    let k_cache = gpu.upload_raw(&arena, &[arena.len()]).expect("k arena");
    let v_cache = gpu.upload_raw(&arena, &[arena.len()]).expect("v arena");

    let rows: usize = m_per_slot.iter().sum();
    let (tile_slot, _, _) = build_tiles(m_per_slot, 1);
    let desc_bytes = unsafe {
        std::slice::from_raw_parts(
            descs.as_ptr() as *const u8,
            descs.len() * std::mem::size_of::<KvSlotDesc>(),
        )
    };
    let d_descs = gpu.upload_raw(desc_bytes, &[descs.len() * 3]).expect("descs");
    let ts_bytes = unsafe {
        std::slice::from_raw_parts(tile_slot.as_ptr() as *const u8, tile_slot.len() * 4)
    };
    let d_row_slot = gpu.upload_raw(ts_bytes, &[tile_slot.len()]).expect("row_slot");

    // positions[r] = that row's own slot's seq_len - 1
    let mut pos_data: Vec<i32> = Vec::with_capacity(rows);
    for (slot, &m) in m_per_slot.iter().enumerate() {
        for _ in 0..m {
            pos_data.push(seq_lens[slot] as i32 - 1);
        }
    }
    let pos_bytes =
        unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, rows * 4) };
    let positions = gpu.upload_raw(pos_bytes, &[rows]).expect("positions");

    let q_data: Vec<f32> = (0..rows * nh * hd)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let q = gpu.upload_f32(&q_data, &[rows * nh * hd]).expect("q");
    let out = gpu.zeros(&[rows * nh * hd], DType::F32).expect("out");

    let max_ctx = *seq_lens.iter().max().unwrap();
    let max_tiles = max_ctx.div_ceil(128);
    let partials = gpu
        .zeros(&[rows * nh * max_tiles * (2 + hd)], DType::F32)
        .expect("partials");

    let batched = time(gpu, &|g: &mut Gpu| {
        g.attention_flash_q8_0_batched_masked_slots(
            &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, max_ctx, max_ctx,
            rows, &partials, None, 0, 0, Some(&d_descs), Some(&d_row_slot),
        )
        .expect("batched slots");
    });

    // Sequential arm: one legacy launch per slot, against that slot's slab.
    // Sub-slicing the arena per slot is what the descriptor does on the GPU;
    // here we re-upload each slab so the legacy kernel sees a base of 0.
    let sequential = time(gpu, &|g: &mut Gpu| {
        for (slot, &sl) in seq_lens.iter().enumerate() {
            let off = descs[slot].k_base as usize;
            let len = descs[slot].cap as usize * per_pos_bytes;
            let slab = g.upload_raw(&arena[off..off + len], &[len]).expect("slab");
            let p = g.upload_raw(
                unsafe { std::slice::from_raw_parts(&(sl as i32 - 1) as *const i32 as *const u8, 4) },
                &[1],
            ).expect("pos");
            g.attention_flash_q8_0_batched_masked(
                &q, &slab, &slab, &out, &p, nh, nkv, hd, sl, sl, 1, &partials, None, 0, 0,
            )
            .expect("sequential");
        }
    });

    (batched, sequential)
}
```

Note the sequential arm re-uploads each slab inside the timed region, which charges it an upload cost the batched arm does not pay. Hoist those uploads out of the closure before timing — otherwise the comparison flatters the batched path and the spec §2 criterion-2 assertion becomes meaningless.

- [ ] **Step 2b: Record the LDS-vs-tile crossover for multi-slot batches**

Required by spec §2 criterion 3 and spec §7. The LDS path's grid is `[n_heads, batch_size]` — only 64 workgroups at `n_heads=16, N=4, M=1` — while the tile path is already massively parallel. Below `LDS_CTX_LIMIT` the router picks LDS, and that may be the wrong choice once several slots are batched.

```rust
    // Which path wins below the crossover, at multi-slot batch? The router
    // sends ctx < LDS_CTX_LIMIT (15000) to the LDS kernel, whose grid is
    // [n_heads, batch] — thin. Measure both paths at the same shape rather
    // than assuming the existing single-sequence crossover still holds.
    for &ctx in &[2048usize, 8192, 14000] {
        for &n_slots in &[1usize, 4, 8] {
            let lds_ms = bench_lds_path(&mut gpu, ctx, n_slots);
            let tile_ms = bench_tile_path(&mut gpu, ctx, n_slots);
            println!(
                "ctx={ctx:6} n_slots={n_slots:2} : LDS {lds_ms:8.3} ms  \
                 tile {tile_ms:8.3} ms  winner={}",
                if lds_ms < tile_ms { "LDS" } else { "TILE" }
            );
        }
    }
```

Do **not** change `LDS_CTX_LIMIT` in this task. Record the finding; changing the router for multi-slot batches is an SP3 scheduler decision and needs its own review.

- [ ] **Step 3: Add the 32 GB budget assertion**

Mandatory per spec §10 — this box has ~125 GiB of shared memory and would otherwise pass an over-budget design that OOMs on the R9700.

```rust
// This box has ~125 GiB shared; the target R9700 has 32 GB. Without this
// assertion an over-budget configuration passes here and OOMs on target.
const R9700_VRAM_BYTES: u64 = 32 * 1024 * 1024 * 1024;
let budget = std::env::var("HIPFIRE_VRAM_BUDGET_BYTES")
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(R9700_VRAM_BYTES);
assert!(
    total_alloc_bytes <= budget,
    "configuration needs {:.2} GiB but the R9700 target has {:.2} GiB",
    total_alloc_bytes as f64 / 1073741824.0,
    budget as f64 / 1073741824.0
);
```

- [ ] **Step 4: Run the sweep**

Run:
```bash
for ts in 64 128 256; do
  echo "=== TILE_SIZE=$ts ==="
  HIPFIRE_ATTN_TILE_SIZE=$ts cargo run --release -p rdna-compute --features deltanet --example q8_batched_attn_microbench
done
```
Expected: batched beats sequential at every `n_slots >= 2`. Per spec §2 criterion 2, a regression against the sequential baseline is a failure, not a tuning outcome.

- [ ] **Step 5: Write the results note**

Create `docs/perf-checkpoints/2026-08-07-multislot-attention-bench.md` with: the batched-vs-sequential table per `n_slots` and KV mode; the `TILE_SIZE` sweep and the chosen default with its justification; the LDS-vs-tile crossover table from Step 2b, with a recommendation for SP3 on whether the multi-slot router should differ from the single-sequence one; the ragged-batch waste measurement; and an explicit statement that these are **gfx1151** numbers and may not transfer to gfx1201.

- [ ] **Step 6: Commit**

```bash
git add crates/rdna-compute/src/attention.rs crates/rdna-compute/examples/q8_batched_attn_microbench.rs docs/perf-checkpoints/2026-08-07-multislot-attention-bench.md
git commit -m "perf(attn): multi-slot bench, TILE_SIZE sweep, 32GB budget gate"
```

---

### Task 9: asym3 quality gate

Spec §13. asym3 is not what these models ship with; this decides whether it may become a batched default.

**Files:**
- Create: `docs/perf-checkpoints/2026-08-07-asym3-quality-gate.md`

**Interfaces:**
- Consumes: nothing from earlier tasks — this measures model quality, not kernel correctness.
- Produces: a recorded verdict. Blocks spec §2 criterion 5.

- [ ] **Step 1: Confirm the resolved KV mode on both arms**

Run each model once per arm and capture the carrier's `KV cache:` log line. Per spec §4.4, `HIPFIRE_KV_MODE=asym3` silently yields q8 on the PaRo loader path, which would make the whole gate meaningless.

```bash
for m in qwen3.6:27b qwen3.6:35b-a3b; do
  for kv in q8 asym3; do
    echo "--- $m / $kv ---"
    HIPFIRE_LOCAL=1 HIPFIRE_KV_MODE=$kv hipfire run "$m" "hi" 2>&1 | grep -i "KV cache"
  done
done
```
Expected: for each model the two lines differ, and the `asym3` line names asym3. If they match, **stop** — `QWEN35_PARO_POLICY` has silently downgraded asym3 to q8 (spec §4.4) and the gate would be comparing q8 against q8.

- [ ] **Step 2: Run the KLD comparison**

Use the existing tooling: `benchmarks/quality-baselines/harness/kld_reduce.py` and `scripts/reap/kld_compare.py`. Report **mean and median** — they diverge sharply on long-tail distributions and the mean alone has misled before.

Score on the model's **own generated output**, not a reference completion; scoring against reference output flatters the quantised model.

Do **not** use synthetic filler prompts. Long prompts built from a small random vocabulary are pathologically out-of-distribution and degenerate on both arms, which reads as a quantisation failure and is not one.

- [ ] **Step 3: Run the KV identicality dashboard**

Run `scripts/kv_quality_dashboard.py` over the quantisation-gate output for both arms.

- [ ] **Step 4: Run coherence at agent-realistic context**

Use the `coherence-gate-qwen35-*` scripts on real prose and code prompts at the long contexts agents actually run at — not short prompts, since rotated-K error accumulates with context.

- [ ] **Step 5: Record the verdict**

Create `docs/perf-checkpoints/2026-08-07-asym3-quality-gate.md` with the resolved-mode evidence from Step 1, the mean and median KLD per model, the dashboard summary, the coherence results, and one of three verdicts per model: asym3 becomes the batched default; asym3 stays opt-in via `HIPFIRE_KV_MODE`; or asym3 is rejected with numbers.

All three are acceptable outcomes. If asym3 is rejected for the 27B, record that spec §4.2's capacity argument fails with it and 4 agents × 128K on the 27B goes back to not fitting — the fallback is fewer agents or shorter contexts, not another compression trick.

- [ ] **Step 6: Commit**

```bash
git add docs/perf-checkpoints/2026-08-07-asym3-quality-gate.md
git commit -m "docs(attn): asym3 quality gate results

Resolved-mode evidence included per spec §4.4 — a silent q8 fallback
would make the whole comparison meaningless."
```

---

## Completion

SP1 is done when spec §2's five criteria hold. At that point the descriptor ABI is frozen for SP2 (multi-slot KV allocator, batched DeltaNet, batched sampling), which should get its own spec before implementation.

**What SP1 deliberately does not deliver:** an end-to-end throughput number. Three quarters of both models' layers are DeltaNet and still single-sequence, so every result here must be labelled an attention-kernel result until SP2/SP3 land.
