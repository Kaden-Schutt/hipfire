# DeepSeek V4 — routed-expert paging from NVMe

Date: 2026-08-02
Status: design approved, not yet implemented

## Goal

Run a DeepSeek V4 trunk whose routed experts do not fit in memory, by keeping a
bounded in-memory cache of experts and reading the rest from the HFQ file on
demand. Two motivations, in priority order:

1. **Unblock quant measurement.** A clean quant-error KLD needs a
   higher-precision reference of the SAME checkpoint. An MQ3 trunk is ~121 GB
   (112.6 GB experts + 8.2 GB dense) against ~102 GB available on STARLING, so
   today it cannot be loaded at all and `KLD(MQ3 ‖ MQ2)` cannot be measured.
2. **Make higher-precision quants usable** where quality justifies the
   throughput cost, rather than being limited to what fits.

Non-goal: interactive serving speed. Streaming trades throughput hard (see
Prior art) and this design does not pretend otherwise.

## Prior art

| system | mechanism | note |
|---|---|---|
| [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4) | non-routed weights resident; routed experts in an in-memory cache, read from the GGUF on miss | The DS4 reference implementation. Runs 2-bit Flash on a 64 GB MacBook. Deliberately uses ordinary read/write I/O, **not mmap**, "so restoring cache entries does not add more VM mappings to a process that already maps the model". |
| [llama.cpp / ik_llama.cpp](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide) | mmap + OS page cache; `-ot` keeps `ffn_*_exps` off-GPU | ~2 tok/s for 671B on 96 GB + NVMe. Simpler, but delegates residency to the kernel — no hard bound. |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | experts *computed* on CPU (AMX), attention + KV on GPU | Not streaming: needs 382 GB DRAM. Solves a different problem. |
| [SSD Offloading … Considered Harmful](https://arxiv.org/pdf/2508.06978) | — | I/O energy exceeds compute savings; latency rises by orders of magnitude. Viable for "extremely memory-constrained systems" and offline/latency-tolerant work. That is exactly this design's remit. |

We follow DwarfStar: resident non-routed weights, bounded expert cache, explicit
reads. hipfire's dormant `weight_pager` (MAD-93 v0.1) already has this shape —
residency map, LRU, byte-range catalog, `PreadH2DTransport`, byte budget — so
this is wiring and hardening, not a redesign.

### Why not a second GPU or another machine's RAM

Streaming moves **weights** per token; expert parallelism moves **activations**:

| | per-token traffic |
|---|---|
| stream experts from NVMe | 1.74 GB (43 layers × 6 experts × 3 projections × 2.25 MB) |
| expert parallelism across devices | ~688 KB of activations |

A second GPU is therefore architecturally better, and hipfire already has EP for
ds4 (`ep_serve_ds4`). It needs hardware this box does not have (one GPU,
`card1`). Remote RAM used as a *paging store* is dominated by local NVMe — at a
measured 3.13 GB/s you would need 25 GbE just to match it — so it is rejected.
Remote *compute* (distributed EP) is viable in principle but needs a network
transport hipfire lacks; it is out of scope here and not precluded by this work.

## Measured hardware baseline

STARLING, WD_BLACK SN7100, `/data`:

- sequential read, O_DIRECT: **3.3 GB/s**
- random 2.25 MB reads, O_DIRECT, QD1: **3.13 GB/s, 0.75 ms/read**

Random reads match sequential at expert granularity, which is what makes
per-expert paging viable and fixes the cache granularity below.

## Architecture

### The seam: the device pointer table

ds4 already avoids 33K allocations by uploading, per (layer, projection), one
contiguous blob plus a device-side pointer table:

```
expert_w{1,2,3}_blob   : [n_routed_experts × bytes_per_expert] raw bytes
expert_w{1,2,3}_ptrs   : F32 tensor, 2 slots per u64 pointer
expert_w{1,2,3}_stride : bytes per expert
```

The indexed MoE GEMV kernels dereference the pointer table. **Only the 6 routed
entries are dereferenced per token**, so only those must be valid. That is the
whole basis of this design: no tensor layout changes, no kernel changes.

Change: allocate blobs with **K slots** (K ≪ n_routed_experts) instead of 256,
and repoint table entries at slots as experts become resident.

### New component

`Ds4ExpertPager`, owning:

- **slot pool** — the per-(layer, projection) cache blobs, allocated once
- **residency map** `(layer, expert, proj) → slot`
- **LRU** recency queue for eviction
- **catalog** `(layer, expert, proj) → (file_offset, byte_len)`, built from the
  HFQ tensor index at load
- **transport** — `PreadH2DTransport`, reused from `weight_pager`

It reuses `weight_pager`'s `Transport` trait, LRU and budget logic. The
ds4-specific part is the slot pool and pointer-table patching.

### Per-token data flow, per MoE layer

1. compute router scores (unchanged)
2. **D2H the top-k expert indices** (6 × u32 = 24 B) — new sync point
3. `ensure_resident` for 18 entries (6 experts × 3 projections); on a miss,
   evict LRU and `pread` 2.25 MB into the freed slot
4. patch the 18 pointer-table entries and H2D the tables (~6 KB/layer)
5. GEMV (unchanged)

**The sync point is inherent, not incidental.** Layer L+1's routing depends on
layer L's output, so cross-layer prefetch is impossible without *predicting*
routing (what MoE-Infinity does). We accept the stall: 43 syncs/token at
~50–200 µs ≈ 2–9 ms against a 72 ms token, i.e. 3–12%.

Layers 0–2 are hash-routed (`num_hash_layers = 3`) and take the existing
fast path; they page identically since they also produce top-k indices.

## Memory contract

**Never OOM, by construction — not by checking.**

"Budget" throughout means the **expert slot pool** specifically, in bytes. The
always-resident non-routed weights, KV/SWA caches and per-step scratch are
accounted separately and are NOT drawn from it; the pager owns only the slot
pool. Sizing at load is therefore
`slot_pool = min(configured, MemAvailable − non_routed − kv_and_scratch − headroom)`.

- Budget is fixed at load and enforced hard for the process lifetime.
  Auto-size for convenience, hard-enforce for determinism.
- Slot count follows from it:
  `K = slot_pool / (n_layers × 3 × bytes_per_expert)`, floored. For reference,
  `K = 256` is the fully-resident case and reproduces today's ~74 GB of expert
  blobs, so the pager degenerates to current behaviour at a large enough budget
  — which is what test 1 below asserts.
- **The slot pool is allocated once at load and never grows.** Steady-state
  paging performs zero allocation: a miss evicts into an existing slot and
  `pread`s into it. There is no code path from a cache miss to an allocator.
- Non-routed weights (8.2 GB) are always resident, as in DwarfStar.
- If non-routed weights plus the minimum viable slot pool exceed budget, the
  load **fails cleanly with a sizing error** rather than partially allocating.

Minimum viable K is `num_experts_per_tok = 6` per (layer, projection) — enough
for one token's working set. Anything less cannot make progress.

## Error handling

- **Short/failed read** → error out of the forward pass with layer/expert/offset
  context. Do not silently produce a zero or stale expert; a wrong expert is a
  silent quality regression, which is the failure mode this whole session has
  been chasing.
- **Budget too small at load** → clean error naming required vs available bytes.
- **Catalog miss** (expert absent from the HFQ index) → error at load during
  catalog construction, not at first use.
- **File truncated/replaced under us** → detected by byte-length mismatch
  against the catalog; error rather than reading garbage.

Failing closed matters here because the 4-bit drafter experiment produced 0%
acceptance *silently* — the DSpark MoE path accepted an unsupported format
without complaint. Paging must not repeat that.

## Testing

Paging is pure memory management over read-only weights, so it must be
**output-neutral**. Greedy decode on the plain ds4 path is now deterministic,
which makes that directly assertable:

1. **Bit-identical, cache large enough to hold everything** — pager on with
   budget ≥ full model vs pager off: identical committed token IDs.
2. **Bit-identical under heavy eviction** — shrink the budget so most tokens
   miss: still identical token IDs. This is the test that would catch a
   pointer-table patching bug or an off-by-one in slot reuse.
3. **Clean failure below the floor** — budget under the minimum: load errors,
   nothing allocated.
4. **Unit tests** on residency/LRU/eviction with a fake `Transport` (extends the
   6 existing `weight_pager` tests).
5. **End-to-end**: PPL on a streamed MQ3 trunk, then `KLD(MQ3 ‖ MQ2)` — the
   measurement that motivated the work.

## Scope

In: ds4 only; per-expert paging; auto-sized hard budget; pointer-table patching;
synchronous `pread`.

Out (YAGNI): `CpuRouter` (unnecessary — top-k comes from the device more cheaply
than recomputing the router on CPU); io_uring/P2P DMA; prefetch; async transfer
overlap; qwen35 refactor. The `Transport` trait seams the first two in later
without touching the forward path.

## Risks

- **Throughput floor.** ~1.8 tok/s if every expert misses. Acceptable for the
  stated remit; would not be acceptable for serving.
- **Sync-point cost is an estimate.** The 50–200 µs per D2H is not yet measured
  on this box; if it lands high, the 43-per-token count makes it material.
  Measure before optimising.
- **Routing skew is unknown.** Hit rate at a given budget depends on how skewed
  expert selection is, which we have not characterised. Worth logging hit rate
  from day one so the budget/throughput curve is empirical rather than assumed.
