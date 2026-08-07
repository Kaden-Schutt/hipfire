# Ragged multi-slot batched attention (SP1)

- **Date:** 2026-08-07
- **Base:** `origin/beta` @ `e2f7dd1a`
- **Branch:** `feat/batched-attention-slots`
- **Worktree:** `~/repos/hipfire-batchattn`
- **Status:** design approved, plan pending

## 1. Goal

Run **3–4 coding agents concurrently on a single R9700 (32 GB)** as fast as the
hardware allows, on `qwen3.6:27b` and `qwen3.6:35b-a3b`.

That end goal is a program, not one change. This spec covers **SP1**: the
attention kernels gain a *slot* dimension, so one launch serves several
independent sequences, each at its own position, each attending only to its own
KV. Everything else is deliberately out of scope and listed in §3.

Development and testing happen on the **Strix Halo box (gfx1151)**. The target
is **R9700 (gfx1201)**. That gap is a first-class risk — see §11.

## 2. Success criteria

SP1 is done when all of the following hold:

1. Golden equivalence, cross-slot isolation, and adversarial-shape tests pass
   for both models' head configurations (§9).
2. At fixed per-slot context and `N` in 2–8, one batched launch over `N` slots
   is faster than `N` sequential single-slot launches, measured by the §10
   microbenchmark. (The size of the win is an output of the work, not a
   precondition; regression against the sequential baseline is a failure.)
3. The split-K occupancy default is chosen from a measured sweep, not guessed.
4. Task 0's measured batching ceiling is recorded, and the §8 roofline estimates
   are either confirmed or replaced by the measurement.

## 3. Decomposition and non-goals

| | Sub-project | Status |
|---|---|---|
| **SP1** | **Ragged multi-slot batched attention kernels + descriptor ABI** | **this spec** |
| SP2 | Multi-slot KV allocator lifecycle, batched KV write/RoPE, batched DeltaNet state update, batched sampling | later |
| SP3 | Ragged batched forward through the qwen35 layer driver; scheduler mixing chunked prefill with decode; per-slot spec decode with independent draft lengths and acceptance | later |
| SP4 | Daemon concurrency: multiple in-flight requests, per-slot session + prefix cache, admission control, 32 GB budget enforcement | later |

**SP1 explicitly excludes** the KV allocator lifecycle, DeltaNet batching, the
scheduler, and any daemon change. SP1 *does* own the descriptor ABI and the
arena layout contract, so SP2 has nothing to renegotiate.

Note the shape of the models: both are **hybrid**, `full_attention_interval: 4`.
Only 16 of the 27B's 64 layers and 10 of the 35B's 40 layers are full attention.
The other three quarters are DeltaNet linear attention with recurrent state, and
batching *those* is SP2. SP1 therefore cannot on its own produce an end-to-end
throughput number — Task 0 exists to bound what it will be worth.

## 4. Scope decision: Q8_0 only

Both models declare `default_kv_mode: "q8"` in `registry/v1.json`. The
asym{2,3,4} / fwht{2,3,4} / lloyd KV families are opt-in quality experiments;
covering them would roughly triple the kernel surface for no benefit to the
stated goal. They keep working unchanged on the single-sequence path.

If a rotated-K mode later becomes default, it gets its own spec. The descriptor
ABI in §6 is dtype-agnostic, so that port is mechanical.

## 5. Prior art

### 5.1 Internal — the batched kernels already have the right shape

Everything below is present on `origin/beta`.

| File / symbol | What it already gives us |
|---|---|
| `attention_q8_0_kv_batched` (`crates/rdna-compute/src/attention.rs:1584`) | Grid `[n_heads, batch_size]`; takes a **device `positions[]` array**, so every query row already carries its own context length. Optional `tree_bias` per-row mask. |
| `attention_flash_q8_0_tile_batched` + `attention_flash_q8_0_reduce` | Tile + reduce two-kernel split with partial `(m, l, acc)` combining. LDS is `O(tile)`, not `O(ctx)`. |
| `attention_q8_0_flash_prefill` (+ `_wmma`) (`attention.rs:1822`) | FlashAttention-2 shaped; launches `grid = [batch.div_ceil(br), n_heads]` — i.e. **already tiled into BR-sized row tiles**. |
| `attention_decode_batched_history` | Same "one workgroup per (head, query row)" idiom in F32. |
| DDTree tree-attention bias (`PRIOR-ART.md` §7) | Per-row masking already overlays a verify mask onto these kernels. |
| ds4 expert paging (`8d002eec`, `ca31ebcf`) | Precedent for device-side pointer tables, including "skip the upload when nothing moved". |

**The single thing binding these kernels to one sequence is the scalar
`k_cache`/`v_cache` base pointer and the scalar `max_seq` stride.** The ragged
query dimension already exists. This is why SP1 is a tractable change rather
than a rewrite.

Related prior work that informs the tuning, not the structure:
`docs/perf-checkpoints/2026-07-29-*` (flash prefill, BR=8/BC=16 winner, tile
choice non-monotonic in LDS) and the `Q8_BATCHED_LDS_CROSSOVER` finding (the
real LDS bound is ~16000, not the shipped 8192).

### 5.2 External

- **FlashAttention `varlen` / `cu_seqlens`** — the flat-rows-plus-descriptors
  idiom this design adopts.
- **vLLM PagedAttention** — block tables; the destination of the §6.3 seam.
- **Flash-decoding split-K** — partitioning the KV dimension to fill the machine
  at low batch; §7.

## 6. Architecture

### 6.1 The unit of work is a row tile

A **row tile** is up to `BR` consecutive query rows belonging to **one** slot.
`BR = 1` for the decode kernels; `BR = 8` (current tuned default) for the flash
prefill kernel. No tile may span a slot boundary.

The launcher builds a flat tile list from the per-slot query counts, so a batch
where slot 0 verifies 8 draft tokens, slot 1 is chunk-prefilling 256 tokens, and
slots 2–3 each decode 1 token is just a flat list of tiles with slot tags. This
is what makes "N sequences × M query tokens each, M ragged" fall out rather than
being special-cased.

### 6.2 Descriptor ABI

Two new device arrays. `positions[]` keeps its current meaning and layout.

```c
struct KvSlotDesc {        // one per active slot
    uint64_t k_base;       // byte offset into the layer's K arena
    uint64_t v_base;       // byte offset into the layer's V arena
    int32_t  seq_len;      // logical KV length for this slot
    int32_t  cap;          // physical slab capacity, in tokens
};

// tile_slot[t] -> slot index for flat row tile t
// tile_row0[t] -> first query row of tile t, within its slot
```

Grid becomes `[n_heads, n_tiles]` for the decode kernels and
`[n_tiles, n_heads]` for the prefill kernel (preserving each kernel's existing
axis order, so the launch-config code stays recognisable).

The kernel-side change is mechanical: every `k_cache + pos*kv_stride` becomes
`k_cache + kv_offset_for(slot, pos)`.

Descriptor tables are uploaded once per step, not per layer. Following the ds4
precedent, the upload is skipped when the table is unchanged.

### 6.3 The paged seam

All KV addressing goes through **one** device-side helper:

```c
__device__ inline uint64_t kv_offset_for(const KvSlotDesc& s, int pos);
// today:  s.k_base + (uint64_t)pos * kv_stride       (contiguous slabs)
// later:  block_table[s.block_ofs + pos/PAGE] * PAGE_BYTES + (pos%PAGE)*kv_stride
```

`kv_stride` is the existing per-position stride scalar
(`n_kv_heads * head_dim`, in the kernel's element units) already passed to these
kernels — it is not part of the descriptor, because it is uniform across slots.

Swapping to a block table changes this function and the descriptor struct — not
the kernels. Because the flash path already walks KV in tiles, choosing
`PAGE` as a multiple of the tile size makes the paged upgrade nearly free.

This is the concrete form of "design for small, don't foreclose large": SP1
ships contiguous per-slot slabs sized for 2–8 agents, and the indirection that
scales past that is in place from day one.

### 6.4 Arena layout contract (owned by SP1, consumed by SP2)

Per layer, one K arena and one V arena. A slot occupies a contiguous slab of
`cap` tokens at `k_base`. Slabs are `cap`-aligned so a future page size divides
them. `seq_len <= cap` always; the kernel reads `[0, seq_len)` and never touches
bytes beyond it, which is what makes the isolation test in §9 meaningful.

## 7. Occupancy and split-K

At `N = 4`, `M = 1`, `n_heads = 16` the decode grid is 64 workgroups. The R9700
has 64 CUs: exactly one workgroup per CU, no oversubscription, no latency
hiding, each workgroup serially streaming a long KV. Batching already beats four
sequential launches of 16 workgroups each, but it leaves throughput on the floor.

The fix is split-K (flash-decoding): partition each tile's KV range into `S`
chunks, extend the grid by an `S` axis, and combine partial `(m, l, acc)`
triples.

**We do not build that machinery.** `attention_flash_q8_0_tile_batched` +
`attention_flash_q8_0_reduce` already are it. Today `S` is chosen to keep LDS
under 64 KB. SP1 reframes it as an *occupancy* decision:

```
S = clamp(target_wgs / (n_heads * n_tiles), 1, seq_len / MIN_CHUNK)
```

floored by `MIN_CHUNK` so the reduce pass does not dominate at short context.
`target_wgs` and `MIN_CHUNK` are env-overridable and defaulted from the measured
sweep in §10. This is a launcher change to an existing two-kernel path.

## 8. Roofline estimates (to be validated by Task 0)

Per decode step at 32K context, batch 4. **These are bandwidth-roofline
arithmetic, not measurements** — Task 0 exists to confirm or replace them. They
ignore embedding-table gathers and assume ~0.55 B/param at mq4.

| | 27B dense | 35B-A3B MoE |
|---|---|---|
| Layers / full-attention | 64 / 16 | 40 / 10 |
| Heads (q/kv), head_dim | 24 / 4, 256 | 16 / 2, 256 |
| KV per token | **32 KB** | **10 KB** |
| Weights per step, batched | ~15 GB (all amortise) | ~1.2 GB dense + ~2.2 GB experts |
| KV per step, 4 slots @32K | 4.3 GB | 1.3 GB |
| Batched total | 19.3 GB | 4.7 GB |
| 4× sequential | 64.3 GB | 8.4 GB |
| **Aggregate speedup** | **~3.3×** | **~1.8×** |

Two structural facts drive this:

- **Attention KV reads never amortise across slots.** Unlike weights, they scale
  linearly with batch. The longer the agents' contexts, the less batching buys.
- **MoE expert reads barely amortise at small batch.** Four sequences drawing
  top-8 of 256 experts collide rarely, so expert traffic scales ~4×. Only the
  dense half of the 35B amortises.

The memory ceiling inverts the ranking:

| | 4 agents @32K | 4 agents @128K |
|---|---|---|
| 27B (15 GB weights) | 19.3 GB — fits | 32.2 GB — **does not fit** |
| 35B-A3B (~20 GB weights + draft + MTP) | 21.4 GB — fits | 25.5 GB — fits |

**27B batches better; 35B-A3B scales to longer contexts.** That is a real
product choice for "3–4 agents on one R9700", and the bench should decide it
rather than assume it.

## 9. Correctness

Three layers, each cheap to run.

1. **Golden equivalence.** For each slot, run the existing single-sequence
   kernel; compare against one batched multi-slot launch covering all slots.
   Tolerance-based, not bitwise — split-K reorders accumulation. Reuse the
   tolerance framework from the flash-prefill 14/14 shape test.
2. **Cross-slot isolation.** Fill every *other* slot's KV slab with NaN and
   confirm the target slot's output is unchanged. Descriptor and stride bugs are
   the entire new failure mode, and this catches them directly.
3. **Adversarial shapes.** Wildly unequal `seq_len` in one batch (1 vs 100K);
   per-slot `M` mixed across 0/1/3/8; slot counts 1–8; `seq_len` below tile
   size; non-multiples of BR/BC; GQA groups 6:1 (27B) and 8:1 (35B); a mixed
   batch of one prefill tile plus three decode tiles.

Harness: extend `crates/rdna-compute/examples/test_q8_flash_prefill.rs`, which
already avoids the ~10 s model load.

## 10. Benchmark

Extend `q8_batched_attn_microbench.rs` to sweep `(n_slots, M_s, ctx_s)` and
report attention-kernel milliseconds plus aggregate effective tok/s against the
`n_slots ×` sequential baseline. Sweep `target_wgs` and `MIN_CHUNK` to pick the
split-K defaults.

**A 32 GB budget assertion in the harness is mandatory.** This box has ~125 GiB
of shared memory; an over-budget design would otherwise pass here and OOM on the
target.

## 11. Risks

- **We cannot validate RDNA4 perf on gfx1151.** Tile, BR/BC, and split-K
  constants tuned here may not transfer to gfx1201. Every one of them must be
  env-overridable, as `HIPFIRE_FLASH_PREFILL_BR/_BC` already are, and the spec
  should not bake a tuned constant into a `const`.
- **The 35B's ~1.8× may not justify batching it at all** versus keeping spec
  decode at batch 1. Task 0 settles this before the kernel work is finished.
- **Attention is only part of a decode step.** SP1 alone cannot move end-to-end
  throughput while DeltaNet (three quarters of the layers) is still
  single-sequence. Reported wins must be labelled as attention-kernel wins until
  SP2/SP3 land.
- **Split-K changes accumulation order**, so golden tests are tolerance-based;
  a regression that shifts numerics slightly could hide inside tolerance. The
  isolation test is the sharper instrument and should gate merges.

## 12. Task 0 — empirical batching-ceiling probe

Runs **before** the kernel work and validates §8.

Measure whole-step decode latency at batch 1 on both models across context
lengths (4K, 16K, 32K, 64K), then fit `t(ctx) = a + b·ctx`. The intercept `a` is
the context-independent term (weights, DeltaNet, dense projections); the slope
`b` is the KV/attention term. Predicted batched step time at `N` slots is then
`a_amortised + N·b·ctx`, where `a_amortised = a` for the dense 27B and
`a_dense + N·a_expert` for the 35B.

**Method constraint: no per-operation `device_synchronize`.** Per-op syncs
fabricate false GPU speedups and would corrupt this measurement. The slope fit
is chosen precisely because it needs only whole-step wall time.

For the 35B, also measure the **expert-overlap factor**: instrument router
top-k selections and count distinct experts touched per layer when `N` sequences
are decoded, rather than assuming disjointness. That number is what turns the
`a_expert` term from an estimate into a measurement.

Deliverable: a short results note under `docs/perf-checkpoints/`, and either
confirmation of the §8 table or a corrected one.
