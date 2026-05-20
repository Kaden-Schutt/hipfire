# DFlash high-ctx VRAM bloat — two surgical fixes

**Status:** Investigation 2026-05-15, fixes pending implementation.

## Background

On 24 GB 7900 XTX, hipfire DFlash with 27B Qwen3.5 + DFlash drafter OOMs at
ctx=17408 under `--kv-mode asym3`, despite KV-cache math saying ctx ≥64K
should fit:

| ctx | VRAM used | Status |
|---|---|---|
| 4096 | 17.6 GB | comfortable |
| 16384 | 24.16 GB | barely fits |
| 17408 | OOM | crash |

Investigation found two non-KV buffers scaling with `max_seq` that account
for the ~8 GB excess. Both are surgical fixes recovering ~3 GB at ctx=17K.

## Bug 1: `DflashScratch.mq_x_rot` over-sized

**File:** `crates/hipfire-runtime/src/dflash.rs:424-433`

```rust
let widest = std::cmp::max(l * ne * h, b * std::cmp::max(inter, qd));
Some(gpu.alloc_tensor(&[widest], DType::F32)?)
```

At ctx=17440, num_extract=5, hidden=5120: `l × ne × h × 4 = 1.74 GB`.
The other two uses of the buffer (`w_down`, `wo`) need only ~1.6 MB.

The `l × ne × h` term covers the **first-call** `fc` (target_hidden)
rotation, where the entire prefix gets rotated in one GEMM. Steady-state
cycles only rotate `delta ≤ block_size + accept ≈ 24` rows.

### Fix sketch

1. Shrink allocation:
   ```rust
   let widest = b * std::cmp::max(inter, std::cmp::max(qd, ne * h));
   ```
   Approximately `24 × max(17408, 17408, 25600) × 4 = ~2.5 MB`. Saves 1.74 GB.

2. **Critical:** the consumer that does the first-call `fc` rotation must
   chunk into batches of size `chunk = widest / (ne × h)`. Find the
   consumer in `dflash.rs` and/or `crates/rdna-compute/src/dispatch.rs`
   (look for the `fc` rotation call — uses `mq_x_rot` and rotates
   `target_hidden`). Chunked rotation = `ceil(l / chunk)` GEMMs instead
   of one. Expected cost: ~50 ms added to TTFT at ctx=17K (negligible vs
   seconds-scale prefill).

3. Verify chunked rotation is mathematically identical to the unchunked
   GEMM (FWHT is linear so this should hold).

## Bug 2: KV cache allocated for all 64 layers, only 16 carry KV

**Files:**
- `crates/hipfire-arch-qwen35/src/speculative.rs:208-246`
- `crates/hipfire-runtime/src/llama.rs:3071-3076`

`speculative.rs:209` already computes `n_kv_layers` correctly:
```rust
let n_kv_layers = config
    .layer_types
    .iter()
    .filter(|t| **t == qwen35::LayerType::FullAttention)
    .count();
```

But the four `KvMode` arms in the match below all pass `config.n_layers`
(64) instead of `n_kv_layers` (16) to the KvCache constructor. `llama.rs`
loops `for _ in 0..n_layers` allocating K+V buffers for every layer.

At ctx=17K with asym3: K+V per layer ≈ 25.9 MB. **48 of 64 (75%) are
LinearAttention layers that never write to a KV cache** — pure waste.

### Fix sketch

Choose implementation strategy:

**(a) Sparse representation** — allocate 16 K/V buffers, add a
`Vec<Option<usize>>` mapping `layer_idx → kv_idx`. Indexer becomes
`kv_cache.k_gpu[map[layer_idx].unwrap()]`. Cleaner but touches every
KV read/write site.

**(b) Sparse-by-None** — keep 64 entries, leave 48 as `None`. Allocator
just skips for LinearAttention layers. Indexer becomes
`kv_cache.k_gpu[layer_idx].as_ref().unwrap()`. Smaller change. Requires
`Vec<Option<GpuTensor>>` instead of `Vec<GpuTensor>` though.

**(c) Plumb layer_types into constructor** — pass `Option<&[LayerType]>`
to `KvCache::new_gpu_*` and check inside the alloc loop. Indexer stays
the same; underlying storage changes to sparse-by-None.

Recommendation: **(b)** — smallest blast radius. Allocator change is one
spot; indexer changes are one extra `.as_ref().unwrap()` per access.

All 4 KV modes need the same fix (Q8, Asym4, Asym3, Asym2). Best to land
as a single PR.

## Tertiary savings (out of scope for this fix)

To exceed 64K ctx, the next bottleneck is the duplicate hidden-state
representation:

- `DflashScratch.target_hidden`: `l × ne × h × 4` (~1.78 GB at 17K)
- `HiddenStateRingBuffer.layer_bufs`: `ne × max_pos × h × 4` (~1.78 GB at 17K)

These hold the same payload in different layouts. Collapsing into one
buffer + a permutation function saves another ~100 KB/token. At ctx=64K
that's ~6.4 GB. More invasive — touches both the `target_hidden`
populator and the `HiddenStateRingBuffer` consumer.

Deferred until the surgical fixes ship.

## Expected outcomes

Post fix-1 + fix-2 at 24 GB:

```
24 GB total
- ~14 GB weights + 1 GB drafter + 1 GB const = 16 GB
- ~64K × 94 KB legit ctx-linear buffers = 6 GB
= 2 GB headroom → ctx 64K achievable
```

Post fix-1 + fix-2 + target_hidden/hidden_rb collapse:

```
- ~128K × 64 KB legit ctx-linear buffers = 8 GB
= ~0 GB headroom → ctx 128K achievable on 24 GB
```

## Risk

Both fixes are correctness-preserving in the abstract but require
verification:

- Fix 1: chunked GEMM result must equal unchunked GEMM (likely automatic
  for FWHT but worth a smoke comparison).
- Fix 2: every site that reads `kv_cache.k_gpu[layer_idx]` must handle
  LinearAttention layers (which are now `None`) cleanly. Most such reads
  are inside FullAttention dispatch paths that won't fire on
  LinearAttention layers anyway — but worth grep'ing for indexing sites
  to confirm.

## Verification

After implementation, re-run the ctx bisect:
```
for ctx in 16384 32768 49152 65536 98304; do
    dflash_spec_demo --target ... --draft ... --max 30 --ctx $ctx \
        --kv-mode asym3 --no-chatml
done
```
Expected: ctx=65536 passes, ctx=98304 OOMs (until tertiary fix).
