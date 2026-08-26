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
3. The `TILE_SIZE` default is confirmed or replaced by a measured sweep, and the
   LDS-vs-tile crossover behaviour for multi-slot batches is recorded.
   **Status: partially met.** The sweep ran and the crossover is recorded (§10,
   and the bench checkpoint) — TILE won 62 of 63 multi-slot trials, contradicting
   the shipped `LDS_CTX_LIMIT = 15000`, which was deliberately left unchanged as
   an SP3 routing decision. The default remains 128 and is **not** confirmed:
   `TILE_SIZE=256` measured fastest at every slot count once the benchmark's
   sequential arm was corrected, but that pass ran at load average 9 where even
   the shipped default failed its criterion in 1 trial of 8 on noise alone.
   Promoting 256 requires a confirmation run on an idle box.
4. Task 1's measured batching ceiling is recorded. **AMENDED 2026-08-07:** an
   attention-only probe cannot measure the weights term, so §8's *aggregate*
   3.3× / 1.8× figures are explicitly out of SP1's reach and are deferred to
   SP2/SP3, when a full batched forward exists. What SP1 must record instead is
   the measured attention-term behaviour: the KV slope per shape and the
   achieved-bandwidth headroom (done — see
   `docs/perf-checkpoints/2026-08-07-batching-ceiling-probe.md`), plus the
   batched-vs-sequential attention numbers from §10.
5. The asym3 quality gate (§13) has run on both models and its result is
   recorded. asym3 does not become any model's batched default until it passes.

## 3. Decomposition and non-goals

| | Sub-project | Status |
|---|---|---|
| **SP1** | **Ragged multi-slot batched attention kernels + descriptor ABI** | **this spec** |
| SP2 | Multi-slot KV allocator lifecycle, batched KV write/RoPE, batched DeltaNet state update, batched sampling | later |
| SP3 | Ragged batched forward through the qwen35 layer driver; scheduler mixing chunked prefill with decode; per-slot spec decode with independent draft lengths and acceptance | later |
| SP4 | Daemon concurrency: multiple in-flight requests, per-slot session + prefix cache, admission control, 32 GB budget enforcement, **KV swap-on-idle** (§15) | later |

**SP1 explicitly excludes** the KV allocator lifecycle, DeltaNet batching, the
scheduler, and any daemon change. SP1 *does* own the descriptor ABI and the
arena layout contract, so SP2 has nothing to renegotiate.

Note the shape of the models: both are **hybrid**, `full_attention_interval: 4`.
Only 16 of the 27B's 64 layers and 10 of the 35B's 40 layers are full attention.
The other three quarters are DeltaNet linear attention with recurrent state, and
batching *those* is SP2. SP1 therefore cannot on its own produce an end-to-end
throughput number — Task 0 exists to bound what it will be worth.

## 4. Scope decision: Q8_0 and asym3

SP1 covers **two** KV modes: `Q8_0` and `asym3` (3-bit Givens-rotated Lloyd-Max
K, Q8_0 V). The remaining modes — asym2, asym4, fwht{2,3,4}, lloyd — stay on the
single-sequence path.

### 4.1 Why both

**Q8_0 is what these two models ship with today.** `registry/v1.json` sets
`default_kv_mode: "q8"` for `qwen3.6:27b` and `qwen3.6:35b-a3b`, and
`hipfire-cli/src/main.rs:5731` passes that registry value down as the KV-mode
override. Leaving it on the single-sequence path would strand the production
default.

**asym3 is what `auto` means, and it is the capacity lever.** hipfire's own
resolution logic is explicit:

- `normalize_full()` maps `"auto"` / `"turbo"` / `"turbo3"` → `Asym3`
  (`crates/hipfire-runtime/src/kv_mode.rs:62`)
- `QWEN35_HFQ_POLICY.default = Asym3` — the qwen35 carrier's own default
- the `Asym3Auto` sentinel collapses to `Asym3` **iff `head_dim == 256`**, and
  both target models are 256
- there is a test named `qwen35_default_kv_mode_is_asym3`
- `PRIOR-ART.md` §6 describes it as the best-quality rotated K per RotorQuant

An earlier draft of this spec called the asym family "opt-in quality
experiments" on the strength of the registry value alone. That was wrong: it is
true of production today, but the library default and everything `auto` resolves
to is asym3.

### 4.2 What asym3 buys

Nominal K 3-bit + V 8-bit against 8+8 gives 0.6875× the KV bytes (block scales
and rotation tables are secondary and ignored here). At 4 agents × 128K context:

| | Q8_0 | asym3 |
|---|---|---|
| 27B KV | 17.2 GB → **32.2 GB total, does not fit** | 11.8 GB → **26.8 GB total, fits** |
| 35B-A3B KV | 5.4 GB → 25.5 GB, fits | 3.7 GB → 23.8 GB, fits |

Capacity is the lesser half. KV traffic is the term that **never amortises
across slots** (§8), so cutting it by 31% raises the batching ceiling itself,
not merely the memory headroom.

**OUTCOME (Task 9, measured 2026-08-07): asym3 was REJECTED as a default for
both models and this capacity argument therefore FAILS.** asym3 changes ~30% of
top-1 token choices against q8 (27B 68.8% agreement, 35B-A3B 70.3%; mean KLD
0.297 / 0.552 nats), diverging within the first two generated tokens. It remains
fully supported and opt-in via `HIPFIRE_KV_MODE=asym3`, but it does not ship as
a default, so **4 agents × 128K on the 27B goes back to not fitting in 32 GB**.
The fallback is fewer agents or shorter contexts on the 27B — not another
compression trick. The 35B-A3B is unaffected: its 10 KB/token KV already fits 4
agents × 128K in ~25.5 GB at q8. See
`docs/perf-checkpoints/2026-08-07-asym3-quality-gate.md`.

### 4.3 Why this is cheap

asym{2,3,4} and fwht{2,3,4} all route through a single shared
`launch_asym_flash_batched` helper whose `positions[]` / `k_cache` / `max_seq` /
`batch_size` structure matches the Q8 path, and the asym family shares
`attention_flash_asym_reduce_batched`. So asym3 costs roughly two more `.hip`
files plus one shared-launcher edit — and doing the descriptor work *in that
shared helper* puts asym2/asym4/fwht* within trivial reach later without
widening SP1 now.

### 4.4 Trap to remember

`QWEN35_PARO_POLICY` deliberately omits `Asym3` from its `accepted` set, so
`HIPFIRE_KV_MODE=asym3` **silently yields q8** on that loader path
(`kv_mode.rs:84`). That is easy to misread as a broken kernel or a no-op
optimisation. Any asym3 measurement must first confirm the resolved mode from
the carrier's own `KV cache:` log line.

## 5. Prior art

### 5.1 Internal — the batched kernels already have the right shape

Everything below is present on `origin/beta`.

| File / symbol | What it already gives us |
|---|---|
| `attention_q8_0_kv_batched` (`crates/rdna-compute/src/attention.rs:1584`) | Grid `[n_heads, batch_size]`; takes a **device `positions[]` array**, so every query row already carries its own context length. Optional `tree_bias` per-row mask. |
| `attention_flash_q8_0_tile_batched` + `attention_flash_q8_0_reduce` | Tile + reduce two-kernel split with partial `(m, l, acc)` combining. LDS is `O(tile)`, not `O(ctx)`. |
| `attention_q8_0_flash_prefill` (+ `_wmma`) (`attention.rs:1822`) | FlashAttention-2 shaped; launches `grid = [batch.div_ceil(br), n_heads]` — i.e. **already tiled into BR-sized row tiles**. |
| `attention_decode_batched_history` | Same "one workgroup per (head, query row)" idiom in F32. |
| `launch_asym_flash_batched` (`attention.rs:3443`) | **One shared launcher** behind asym{2,3,4} and fwht{2,3,4}, same `positions[]` / `k_cache` / `max_seq` / `batch_size` structure as the Q8 path. Adding the descriptor here covers asym3 now and the rest almost free later. |
| `attention_flash_asym_reduce_batched` | Shared partial-combiner for the whole asym family — the asym counterpart of `attention_flash_q8_0_reduce`. |
| DDTree tree-attention bias (`PRIOR-ART.md` §7) | Per-row masking already overlays a verify mask onto these kernels. |
| ds4 expert paging (`8d002eec`, `ca31ebcf`) | Precedent for device-side pointer tables, including "skip the upload when nothing moved". |

**The single thing binding these kernels to one sequence is the scalar
`k_cache`/`v_cache` base pointer and the scalar `max_seq` stride.** The ragged
query dimension already exists. This is why SP1 is a tractable change rather
than a rewrite.

Related prior work that informs the tuning, not the structure:
`docs/perf-checkpoints/2026-07-29-*` (flash prefill, BR=8/BC=16 winner, tile
choice non-monotonic in LDS). The LDS-crossover investigation that found the
real bound to be ~16000 rather than 8192 has already landed on beta —
`LDS_CTX_LIMIT` is `15000` — so SP1 inherits the raised value and must not
re-litigate it.

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

// row_slot[r]   -> slot index for flat query row r   (BR == 1 kernels)
// tile_slot[t]  -> slot index for flat row tile t    (prefill, BR > 1)
// tile_row0[t]  -> first query row of tile t WITHIN ITS SLOT
// tile_qbase[t] -> first query row of tile t in the GLOBAL flat row space
```

**AS BUILT — this section was corrected after implementation. Three things
differ from the original design and SP2 must read the corrected version.**

**Three tile arrays, not two.** `tile_row0` is slot-relative; `tile_qbase` is
global. They coincide only with a single slot. `q` and `out` are packed across
all slots, so they are indexed by `tile_qbase`, while causal masking is
slot-relative. **`tile_qbase` is the load-bearing one.** `tile_row0` is
currently **ABI-reserved and unread by every kernel** — it stays in the
positional kernarg layout between `tile_slot` and `tile_qbase`, because
`positions[]` is already indexed by global row and therefore carries each row's
own absolute position, making a slot-relative lookup redundant. Do not remove
it without re-cutting the kernarg order.

**The decode kernels did NOT get a tile grid.** The original text said "grid
becomes `[n_heads, n_tiles]` for the decode kernels". They keep
`[n_heads, batch_size]` and index `row_slot[row]` directly, because at `BR == 1`
a tile *is* a row and a tile list would be pure indirection. Only the prefill
kernel (`BR > 1`) uses the tile arrays, on `[n_tiles, n_heads]`.

**`positions[]` is authoritative for the causal bound — never `desc.seq_len`.**
`positions[row] + 1` is the per-row causal window; `desc.seq_len` is the slot's
logical length. They differ whenever a slot has more than one query row. The
descriptor supplies the slab **base address only**. Violating this caused SP1's
only Critical defect: a tile kernel bounded by `desc.seq_len` while the shared
reduce kernel still bounded by `positions[]`, so the reduce folded in stale
partials the tile kernel never wrote.

Descriptor tables are uploaded once per step, not per layer. Following the ds4
precedent, the upload is skipped when the table is unchanged.

### 6.3 The paged seam

All KV addressing goes through device-side helpers in
`kernels/src/kv_slot_desc.h`:

```c
__device__ inline unsigned long long kv_offset_for_k(const KvSlotDesc& s, int pos, int per_pos_bytes);
__device__ inline unsigned long long kv_offset_for_v(const KvSlotDesc& s, int pos, int per_pos_bytes);
// today:  s.{k,v}_base + (unsigned long long)pos * per_pos_bytes   (contiguous slabs)
// later:  block_table[...] * PAGE_BYTES + (pos % PAGE) * per_pos_bytes
```

**AS BUILT — two helpers, not one, and the stride is a parameter.** asym3 forced
this: its K is 3-bit Givens-rotated while its V is Q8_0, so K and V have
*different* per-position strides and cannot share one helper or one stride
scalar. The stride is therefore passed per call rather than living in the
descriptor.

**ABI constraint that fell out of it:** the Q8_0 flash-prefill kernel requires
`v_base == k_base` and uses a single shared slab offset. Keeping both bases live
across its K/V staging loop — which stages both in one pass — costs 23 VGPRs and
25% of its occupancy (16 → 12 waves/SIMD), measured on gfx1151 *and* gfx1201, on
the null-descriptor legacy path. Q8_0 K and V share a stride, so a slot sits at
the same offset in both arenas and the constraint is free to honour. **asym3 is
exempt and must keep both bases.** SP2's allocator must satisfy this for Q8
arenas.

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

### 6.5 Kernels in scope

| Kernel | Mode | Role |
|---|---|---|
| `attention_q8_0_kv_batched` | Q8_0 | LDS `scores[ctx]`, short context |
| `attention_flash_q8_0_tile_batched` + `attention_flash_q8_0_reduce` | Q8_0 | tile + reduce, long context |
| `attention_q8_0_flash_prefill` (+ `_wmma`) | Q8_0 | FA-2 prefill, large M |
| `attention_flash_asym3_tile_batched` + `attention_flash_asym_reduce_batched`, via `launch_asym_flash_batched` | asym3 | tile + reduce, long context |

Routing between the LDS and tile paths stays on the existing `LDS_CTX_LIMIT`
logic (`crates/hipfire-runtime/src/llama.rs:2637`, currently `15000`, inside
`forward_prefill_chunk`). SP1 does not change that value.

The prefill kernel is in scope so a batch can mix chunked prefill with decode
from the start — agents arrive with long prompts mid-flight, and deferring this
would force SP3 to re-cut the tile list.

## 7. Occupancy

**Split-K already exists and the long-context path is not occupancy-starved.**
An earlier draft of this spec claimed the decode grid was 64 workgroups and
proposed adding a split-K axis. That was wrong, and the correction matters
because it removes what looked like the main kernel work.

The tile path launches `grid = [n_heads, max_tiles, chunk]` with 32-thread
blocks, where `max_tiles = ceil(max_ctx_len / TILE_SIZE)` and
`TILE_SIZE = 128`, now resolved once by `Gpu::attn_tile_size()`. The KV-split axis is already there. At
`n_heads = 16`, 32K context and 4 slots that is `16 × 256 × 4 ≈ 16k`
workgroups — ample. And because routing crosses to this path above
`LDS_CTX_LIMIT = 15000`, the contexts agents actually run at are served by the
path that is already parallel.

So SP1 does **not** add a split-K axis. It does two smaller things:

1. **Make `TILE_SIZE` overridable** rather than a hard `const`, so the split
   count is a tunable if the sweep in §10 shows a better value on gfx1201. This
   is a one-line change plus plumbing, not a redesign.
2. **Check the LDS path's occupancy**, which is the one that genuinely is thin:
   `grid = [n_heads, batch_size]` gives 64 workgroups at `n_heads = 16`,
   `N = 4`, `M = 1`. That path only runs below `LDS_CTX_LIMIT`, so it matters for
   short-context agents and for the low end of the bench sweep. If the sweep
   shows it losing to the tile path at small `N`, the fix is to lower the
   crossover for multi-slot batches — a routing change, not a kernel change.

**Ragged cost note.** `max_tiles` is derived from the batch's *maximum*
`max_ctx_len`, so a batch mixing a 1K slot with a 100K slot launches ~780 tiles
for every slot, and the short slot's tiles early-exit on `positions[b]`. Those
are cheap but not free. Whether ragged batches should be split by context
magnitude is a scheduler question for SP3; SP1 only needs to *measure* the waste
and record it, which §10's unequal-`seq_len` sweep does.

## 8. Roofline estimates (to be validated by Task 0)

Per decode step at 32K context, batch 4. **These are bandwidth-roofline
arithmetic, not measurements** — Task 0 exists to confirm or replace them. They
ignore embedding-table gathers and assume ~0.55 B/param at mq4.

| | 27B dense | 35B-A3B MoE |
|---|---|---|
| Layers / full-attention | 64 / 16 | 40 / 10 |
| Heads (q/kv), head_dim | 24 / 4, 256 | 16 / 2, 256 |
| KV per token, Q8_0 | **32 KB** | **10 KB** |
| KV per token, asym3 | **22 KB** | **6.9 KB** |
| Weights per step, batched | ~15 GB (all amortise) | ~1.2 GB dense + ~2.2 GB experts |
| KV per step, 4 slots @32K | 4.3 GB | 1.3 GB |
| Batched total | 19.3 GB | 4.7 GB |
| 4× sequential | 64.3 GB | 8.4 GB |
| **Aggregate speedup** | **~3.3×** | **~1.8×** |

Two structural facts drive this:

- **Attention KV reads never amortise across slots — in bytes.** The bytes scale
  linearly with batch. This is why asym3 matters for throughput and not only for
  capacity: at 0.6875× the KV bytes it shrinks the term batching cannot remove.
  The table above is the Q8_0 case; asym3 raises both aggregate speedups.

  **CORRECTION from Task 1, measured 2026-08-07.** True in bytes, **false in
  time** — the original framing here was pessimistic. The batch-1 attention
  kernel sustains only **~22-28%** (35B shape) and **~33-38%** (27B shape) of
  achievable bandwidth, and the figure is *flat* across a 16× context range, so
  it is a sustained efficiency ceiling rather than launch overhead. The
  denominator is measured, not theoretical: BabelStream on this device gives
  **223.9 GB/s Triad / 239.3 Copy** (87-93% of the 256 GB/s LPDDR5X figure).
  Four slots must read 4× the KV bytes but need not spend 4× the time: there is
  **~2.6-3.1× headroom on the 27B shape and ~3.6-4.6× on the 35B-A3B shape**.

  **MEASURED, third and current statement (2026-08-07).** Two earlier attempts
  at this paragraph were wrong in opposite directions and are recorded in git
  history: the original was too pessimistic about *whether* batching helps, and a
  correction was too optimistic about *how much* — the latter drawn from a
  benchmark whose sequential arm aliased K to V, touching half the distinct bytes
  and running artificially fast. That defect was found in review, fixed, and
  re-measured.

  Corrected, 35B-A3B shape, `TILE_SIZE=128`, median of 8 trials:

  | n_slots | 1 | 2 | 4 | 8 |
  |---|---|---|---|---|
  | speedup | 1.00× | 1.16× | 1.21× | **1.36×** |

  - **Batching is a real win that grows with slot count**, reaching ~1.36× on the
    attention term at 8 slots.
  - **Descriptor indirection is free.** The n=1 case is 1.00×; the previously
    reported 0.85× "indirection cost" was the aliasing artefact.
  - Achieved bandwidth at n=8, now counting `partials` round-trip as well as KV
    (~24% of real DRAM traffic, previously uncounted): **72.2 GB/s** (35B-A3B) and
    **95.5 GB/s** (27B) against the 223.9 GB/s Triad ceiling.

  **Substantial headroom therefore remains** — roughly 2.3-3× at 8 slots — and
  batching does not close it. Whether kernel-efficiency work (the 32-thread
  workgroup, unaligned 34-byte Q8_0 reads, scalar dequant) is a larger prize than
  further batching is a live and worthwhile question, but it is **not settled by
  this data** and the earlier confident claim to that effect is withdrawn.

  **Caveat on precision, not direction.** This pass ran on a box under load
  average 8-9 from unrelated work. Even the shipped default failed the
  batched-beats-sequential criterion in 1 of 8 trials on noise alone, and the 27B
  n=2 margin was noisier still. The *direction and growth* of the win are solid;
  individual margins are ±0.1. `TILE_SIZE=256` measured fastest at every slot
  count (~7-13% over 128 by n=8) and now passes the criterion once the arm is
  fair — but **the shipped default stays 128**, and flipping it should wait for a
  confirmation run on an idle box. That question was answered — see the measured statement below. See `docs/perf-checkpoints/2026-08-07-batching-ceiling-probe.md`.
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

All three run for **both** KV modes. The asym3 arm must assert its resolved mode
first — see the §4.4 trap, where a silent fall back to q8 would make an untouched
asym3 path look correct.

Harness: extend `crates/rdna-compute/examples/test_q8_flash_prefill.rs`, which
already avoids the ~10 s model load.

## 10. Benchmark

Extend `q8_batched_attn_microbench.rs` to sweep `(kv_mode, n_slots, M_s, ctx_s)`
and report attention-kernel milliseconds plus aggregate effective tok/s against
the `n_slots ×` sequential baseline. Sweep `TILE_SIZE` to confirm or replace the
current `128`; asym3 and Q8_0 may want different values, since asym3 moves fewer
bytes per KV element and so shifts the compute/bandwidth balance. Include at
least one batch with strongly unequal `seq_len` so the ragged-tile waste noted
in §7 is measured rather than assumed.

**A 32 GB budget assertion in the harness is mandatory.** This box has ~125 GiB
of shared memory; an over-budget design would otherwise pass here and OOM on the
target.

## 11. Risks

- **We cannot validate RDNA4 perf on gfx1151.** `TILE_SIZE` and BR/BC values
  tuned here may not transfer to gfx1201. Every one of them must be
  env-overridable, as `HIPFIRE_FLASH_PREFILL_BR/_BC` already are, and none should
  be baked into a `const`.
- **The 35B's ~1.8× may not justify batching it at all** versus keeping spec
  decode at batch 1. Task 0 settles this before the kernel work is finished.
- **Attention is only part of a decode step.** SP1 alone cannot move end-to-end
  throughput while DeltaNet (three quarters of the layers) is still
  single-sequence. Reported wins must be labelled as attention-kernel wins until
  SP2/SP3 land.
- **Split-K changes accumulation order**, so golden tests are tolerance-based;
  a regression that shifts numerics slightly could hide inside tolerance. The
  isolation test is the sharper instrument and should gate merges.
- **asym3 could silently resolve to q8** on the PaRo loader path (§4.4), making
  a completely unimplemented asym3 arm pass every test and show no speedup. Every
  asym3 result must carry the resolved-mode log line as evidence.
- **asym3 is not the shipped default for these models**, so widening scope to it
  imports a quality question SP1 would not otherwise have. §13 contains it, but
  if the gate fails, the capacity argument in §4.2 fails with it and the 27B's
  4-agent × 128K configuration goes back to not fitting.

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

## 13. asym3 quality gate

asym3 is **not** what `qwen3.6:27b` and `qwen3.6:35b-a3b` ship with — they ship
q8 (§4.1). SP1 makes asym3 *available* batched; this gate decides whether it may
become the batched default for either model.

Measure asym3 against a q8 reference on **both** models:

- **KLD of next-token distributions** over a fixed prompt set, using the existing
  KLD tooling (`benchmarks/quality-baselines/harness/kld_reduce.py`,
  `scripts/reap/kld_compare.py`). Report mean *and* median — the two diverge
  sharply on long-tail distributions and the mean alone has misled before.
- **KV-mode identicality** via `scripts/kv_quality_dashboard.py`, which already
  exists for exactly this comparison.
- **Coherence** via the `coherence-gate-qwen35-*` scripts on real prose and code
  prompts at the long contexts agents actually run at — not short prompts, since
  rotated-K error accumulates with context.

**Method constraint:** score on the model's *own* generated output, not on a
reference completion. Scoring against reference output flatters the quantised
model and understates divergence.

**Do not use synthetic filler prompts.** Long prompts built from a small random
vocabulary are pathologically out-of-distribution and degenerate into gibberish
on *both* arms, which reads as a quantisation failure and is not one.

Outcome is one of: asym3 becomes the batched default for a model; asym3 stays
opt-in via `HIPFIRE_KV_MODE`; or asym3 is rejected for a model with the numbers
recorded. Any of the three is an acceptable result for SP1 — the gate exists so
the choice is made on measurement.

## 14. Considered and rejected: spilling experts to host RAM

Spilling the 35B's MoE experts to host memory to free VRAM was considered. The
machinery exists — `hipfire_runtime::weight_pager::WeightPager` and
`Qwen35Config::paged_experts`, with device-side expert pointer tables — so it
would be cheap to attempt. It is rejected for this program on four grounds:

1. **The prior art already reached this verdict.** The ds4 expert-streaming
   design states it as a non-goal: *"Non-goal: interactive serving speed.
   Streaming trades throughput hard."* That work was NVMe-backed (3.13 GB/s
   measured, ~1.8 tok/s floor at total miss), so host RAM is a materially better
   regime — but its other warning transfers intact: **expert routing skew is
   uncharacterised**, so hit rate at a given resident budget is unknown.
2. **Bandwidth.** The R9700 is roughly 645 GB/s local against ~50 GB/s practical
   for PCIe 5.0 x16 — about 13× slower. Expert traffic is ~0.53 GB per decode
   step at batch 1; resident that is ~0.8 ms, spilled ~10.6 ms, which exceeds the
   entire rest of the step.
3. **Spill and batching are antagonistic.** Four sequences drawing top-8 from 256
   experts collide rarely, so the expert working set grows nearly linearly with
   batch. The PCIe term therefore scales with `N` — and it is precisely the term
   batching is meant to amortise. At batch 4 that is ~2.1 GB/step over PCIe,
   ~42 ms.
4. **The 35B is not the model with the fit problem.** Per §8, 4 agents at 128K is
   25.5 GB for the 35B (fits, with headroom) and 32.2 GB for the 27B (does not).
   The 27B is dense — it has no experts to spill. asym3 (§4.2) solves the actual
   binding constraint, and reduces bandwidth rather than relocating it to a
   13×-slower bus.

**Hazard if this is ever revisited:** the Strix Halo development box has unified
memory, so a spilled-expert design would show **no** penalty in testing here and
fall off a cliff on the R9700. Any such experiment needs R9700 access to mean
anything. This is the same class of hazard as the 32 GB budget assertion in §10.

## 15. KV swap-on-idle (SP4)

Added 2026-08-07 after the asym3 rejection removed the capacity lever that
§4.2 relied on. **Scope: SP4, as an admission-control feature — not kernel work.**

### The problem it solves

The 27B needs 33.25 GB for 4 agents × 128K at q8 (15 GB weights + 18.25 GB KV
at 34 KB/token) against 32 GB of card, less whatever is reserved. asym3 would
have closed the gap but was rejected on quality (§13). 4 agents × ~96-112K fits
today with no new machinery, so this is only worth building if 128K × 4 is a
hard requirement on that specific model — the 35B-A3B already does 4 × 128K in
25.8 GB.

### What is viable, and what is not

**Per-step streaming is not viable and must not be built.** During decode every
slot reads its *entire* KV every step — there is no cold portion to leave
behind. For the 27B at 128K that is 4.56 GB per slot per step: 7.1 ms from
R9700 VRAM at ~640 GB/s versus **91.3 ms** over PCIe 5.0 x16 at ~50 GB/s.
Layer-wise prefetch does not rescue it either — per layer that is 285 MB, 5.7 ms
of transfer against 0.44 ms of compute, so there is nothing to hide it behind.
The result would be ~11 tok/s aggregate, worse than simply admitting fewer
agents.

**Swap-on-idle is viable.** Page a slot's *whole* KV to host when its agent is
not generating, page it back on resume. The 4.56 GB is then paid **once per
activation** (~91 ms) and amortised over an entire generation, rather than per
step. Coding agents are bursty — they generate, then block on tool calls, test
runs or the human — so typically 1-2 of 4 are decoding at any instant. Two
resident plus two swapped is 15 + 2×4.56 = **24.1 GB**, comfortable.

This is OS virtual memory, and it is what vLLM does for preemption.

### Why it belongs in SP4

Deciding *who is resident* is admission control. SP4 already owns per-slot
sessions, the prefix cache and 32 GB budget enforcement; residency is the same
decision surface. It needs nothing from the kernels beyond what SP1 already
shipped: `kv_offset_for()` is the seam a swappable or paged KV plugs into, and
`WeightPager` is precedent for the paging machinery itself.

### Hazard, and it is the same one that sank expert spill

**Strix Halo has unified memory, so a swap prototype will show near-zero
transfer cost on the dev box and fall off a cliff on the R9700, where the
transfer crosses PCIe.** Any latency claim about this feature requires R9700
access to mean anything. See §14, where this exact hazard invalidated the
expert-spill idea, and `docs/perf-checkpoints/2026-08-07-batching-ceiling-probe.md`.

### Prerequisite before building

Confirm that 4 agents × ~100K is genuinely insufficient. A swap subsystem is
real work — eviction policy, transfer scheduling, and the correctness risk of a
half-swapped slot being read — and 96K × 4 fits today at 28.69 GB.

## 16. SCOPE GAP found by SP3 (2026-08-08): the WMMA flash-prefill kernel

§6.5 listed four kernels for the slot port and noted `attention_q8_0_flash_prefill`
"(+ `_wmma`)" in the prior-art table — but **only the scalar variant was ported.
`attention_q8_0_flash_prefill_wmma` has no `_slots` version anywhere.**

That omission was invisible until SP3's golden gate ran, because the legacy
fingerprint exercises the *scalar* path (every one of its 11 rows prints
`kernel=scalar`). On **gfx11xx — including this dev box and the gfx1201 target —
the `AttnQ8_0KvBatchedMasked` dispatch defaults ON into the WMMA
f16-accumulate kernel for *any* batched prefill, before it ever reaches the
LDS/tiled crossover that SP1 ported.** So the kernel the reference actually uses
for batched prefill on the target hardware is the one kernel SP1 did not port.

**Consequence, measured:** `forward_batch_slots` is bit-identical to the
reference at `n_slots == 1` (0.000× tolerance, after SP3 special-cases the
single-active-slot step onto the WMMA kernel directly), and diverges at
`n_slots >= 2` at 20.49× tolerance — because a genuinely multi-slot step has no
single active slot and falls back to the ported kernels, which are not what the
reference runs.

**This is new kernel work in `rdna-compute`, not a `forward_slots.rs` fix.**
Porting `attention_q8_0_flash_prefill_wmma` to take `KvSlotDesc` + `row_slot` is
the same mechanical change SP1 applied four times, with one added constraint
already recorded in §6.3: the WMMA grid has a **different kernarg layout** (it
omits `v_mode_bits`), which is why SP1's `launch_asym_flash_batched` asserts
descriptors can never reach it. That assertion is now load-bearing and must be
revisited as part of the port.

Until then, **the multi-slot forward is correct for one slot and incorrect for
more than one** — which is precisely the case the whole programme exists to
serve, so this is the top-priority follow-up.
