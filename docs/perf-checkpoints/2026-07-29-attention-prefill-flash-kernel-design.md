# Query-tiled Q8 flash prefill attention for gfx1151

Date: 2026-07-29
Branch: `perf/flash-prefill-attention` (from `e2169c2d`, PR #534 head)
Status: design approved, not yet implemented

## Problem

Prefill throughput on gfx1151 / Qwen3.6-35B-A3B MQ4R halves by ~10K context
(1015 tok/s @1K → 316 @14.4K). Per-kernel rocprofv3 traces attribute this
entirely to attention: every other kernel scales with a fitted exponent of
1.00–1.05 (perfectly linear in N), while attention scales N^2.19 and grows from
12.3% of prefill GPU time at N=2048 to 64.3% at N=12288.

Three stacked effects, all traceable to one design choice:

1. **No K/V reuse.** Both existing kernels launch one workgroup per *individual
   query*, so each of the 256 queries in a prefill chunk re-streams the whole
   prefix. GQA makes it worse: `kv_group = n_heads/n_kv_heads = 8/2 = 4`, so
   four query heads redundantly read the same KV. Measured attention throughput
   is ~490 GFLOP/s, low single-digit percent of this GPU's FP32 peak.
2. **LDS scales with context.** `attention_q8_0_kv_batched` sizes
   `scores[]` by the full context: `shared_mem = (max_ctx_len + block_size +
   head_dim) * 4`. Resident workgroups per CU therefore fall as context grows,
   costing +31% efficiency per unit of attention work from N=2048 to N=8192
   (3.86 → 5.05 ×10⁻⁴ ms per unit).
3. **A forced crossover to a worse kernel.** Because of (2), dispatch switches
   at `Q8_BATCHED_LDS_CROSSOVER = 8192`
   (`crates/hipfire-dispatch/src/families/attention.rs:1240`) to
   `attention_flash_q8_0_tile_batched`, which measures **2.3× less efficient per
   unit of attention work** (11.60 vs 5.05 ×10⁻⁴). That kernel uses 32-lane
   blocks, a 5-deep `__shfl_xor` reduction *per key* to produce one score, and a
   partials buffer plus a second reduce pass.

Effect (3) is separately fixable by raising the constant — validated
bitwise-identical (160/160 committed tokens) at 12288 and worth 1.26–1.72× in
the 8.2K–11.5K window. This spec addresses the root cause instead.

## Non-goals

- Changing the attention *asymptote*. Exact softmax attention is O(N²) in FLOPs
  and that is a property of the math, not the implementation. FlashAttention
  does not reduce FLOPs; it reduces memory traffic. This model already applies
  the architectural mitigation: 30 of 40 layers are DeltaNet linear attention,
  only 10 carry KV. Sliding-window/sparse variants would change model outputs
  and are not available (`Qwen35Config` has no window field).
- Tree-verify (`tree_bias != nullptr`) and sliding-window (`window > 0`) paths.
- Decode (`batch_size == 1`). Untouched.

## Correctness bar

**Numerically equivalent, not bit-identical.** Online softmax necessarily
reorders reductions. Validation is by tolerance, coherence and KLD rather than
bit-equality; golden committed-token fixtures are re-baselined if they move.
Decided explicitly with the user.

## Design

### Kernel

New kernel `attention_q8_0_flash_prefill`, FlashAttention-2 shaped:

```
grid  = [ceil(N/Br), n_heads]
block = 256 threads (8 waves of 32)

per workgroup:
  stage Q tile (Br × head_dim) into LDS once
  O_acc[Br][head_dim] in registers; running m[Br], l[Br]
  for each K/V tile of Bc keys:
      stage K,V tile into LDS in native Q8_0 block form
      S = Q_tile · K_tile^T, causal mask per (q,k) from positions[]
      online softmax: new m, rescale l and O_acc, accumulate
      O_acc += P · V_tile
  write O = O_acc / l
```

K/V tiles are staged **once per Br queries** instead of once per query. K/V stay
in native Q8_0 blocks (34 B per 32 dims: fp16 scale + 32 int8) and are
dequantised on read; expanding to fp32 in LDS would not fit.

### Tile sizes and LDS budget

Starting point Br=16, Bc=32, head_dim=256:

| item | bytes |
|---|---:|
| Q tile 16×256 fp32 | 16,384 |
| K tile 32 × 8 blocks × 34 B | 8,704 |
| V tile | 8,704 |
| S tile 16×32 fp32 | 2,048 |
| total | ~35.8 KB (≈1 WG/CU) |

Br=8 gives ~26 KB (2 WGs/CU) and 256 workgroups for a 256-query chunk vs 128 at
Br=16. Q as fp16 saves a further 8 KB. Reuse factor equals Br, so even Br=8 is
an 8× cut in K/V traffic against today's 1×.

**LDS is a function of Br/Bc only, never of context.** This is the load-bearing
property: it removes the occupancy decay (effect 2) and makes the crossover
unnecessary (effect 3), because one kernel then serves every context length.

Br and Bc are compile-time parameters, swept by the microbench.

### Dispatch integration

In `AttnQ8_0KvBatchedMasked`, route to the new kernel only when
`tree_bias == nullptr && window <= 0`. All other cases keep today's behaviour.
Blast radius is one branch. Once the new kernel covers all context lengths,
`Q8_BATCHED_LDS_CROSSOVER` is deleted rather than retuned, and
`qwen35::PREFILL_MAX_BATCH` (currently 256, `qwen35.rs:6056`; its mirror
`llama.rs:1754` documents the sizing rationale as keeping flash_partials within
2 GB) becomes free to rise — which feeds back as more query tiles and more
parallelism. Raising it is a separate follow-up, not part of this spec.

### Known risk

Grid-level parallelism *drops*: 2048 workgroups today (8 heads × 256 queries) →
128–256. On small prefill chunks this is the one way the rewrite could
underperform. Br is therefore a swept parameter, and the kernel-level microbench
gates any end-to-end claim.

## Validation

Three tiers, cheapest first. Every stage must pass all three.

1. **Kernel correctness** — new test example under
   `crates/rdna-compute/examples/`: random Q/KV, compare against the existing
   kernel. Concrete tolerance (fp32 re-association over ≤16K accumulation
   steps): **max relative error ≤ 1e-4** on any output element with
   `|ref| > 1e-3`, **max absolute error ≤ 1e-5** otherwise, and **cosine
   similarity ≥ 1 - 1e-6** per (query, head) output vector.

   **Superseded during Task 1.** That split criterion is discontinuous at the
   `1e-3` boundary — an identical absolute error passes or fails depending on
   which side `|ref|` lands — and it produced a spurious failure on the first
   *correct* kernel (max_abs 1.2e-7, cosine exactly 1.0, yet "max_rel 2.7e-4").
   The implemented criterion is the numpy-`allclose` combined form:
   **|ref − new| ≤ ATOL + RTOL·|ref|** with ATOL=1e-5, RTOL=1e-4, plus cosine
   ≥ 1 − 1e-6. It remains strict: the passing single-tile kernel consumes 6.3%
   of the tolerance budget at its worst element. A failure outside these bounds
   is a bug, not re-association. Must cover: `seq_len` not a multiple of Bc,
   `seq_len < Bc`, ragged `positions[]`, and `n_kv_heads < n_heads` (GQA).
2. **Kernel perf** — `crates/rdna-compute/examples/q8_batched_attn_microbench.rs`
   (env-parameterised `NH/NKV/HD/N/CTX/ITERS`), swept over CTX, plus a Br/Bc
   sweep. No model load.
3. **End-to-end** — `coherence_probe --temperature 0.0 --emit-committed-jsonl`
   against the pristine baseline; realistic-prose `serve_harness` runs on both
   `redline` and `hip` backends; the chain per-position curve to confirm the
   8192 step is gone; rocprofv3 to confirm attention's fitted exponent drops
   from 2.19.

Benchmark inputs must use real in-distribution prose. Random-word filler makes
the model degenerate and fabricates false "corruption" findings — established
earlier in this investigation, reproduced identically on both backends.

## Staging

- **A** — query-tiled flash, scalar VALU math. Target: beat the current LDS
  kernel at ≤8192 *and* replace the tiled fallback above it.
- **C** — swap inner QK^T / P·V to WMMA, reusing the repo's existing WMMA GEMM
  patterns. Requires A's tiling as the foundation.
- **B** — remaining micro-optimisations (vectorised Q8 loads, coalescing) where
  A/C leave headroom.

Follow-up lever, deliberately out of v1: GQA head-grouping (one workgroup
serving all 4 query heads of a kv_group, giving 4×Br reuse). It multiplies
Q/O/S storage by 4, so it is a Stage-A tuning experiment, not a v1 requirement.

## Baseline measurements to beat

gfx1151, Qwen3.6-35B-A3B MQ4R, Q8 KV, MTP off, prefill chunk 256:

| N | prefill tok/s | attention share | ms per unit attention work (×10⁻⁴) |
|---:|---:|---:|---:|
| 2048 | 916.8 | 12.3% | 3.86 |
| 4096 | 772.1 | 23.4% | 4.42 |
| 8192 | 601.6 | 41.5% | 5.05 |
| 12288 | 358.6 | 64.3% | 11.60 (tiled fallback) |

Projected ceiling if the >8192 slice merely ran at the LDS kernel's 8192
efficiency: 12288 prefill 358.6 → ~490 tok/s. A query-tiled kernel should beat
that, since it also removes effects (1) and (2).
