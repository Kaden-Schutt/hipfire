# QTIP GPU beam-trellis encoder

Status: planned (not started)
Owner: unassigned
Motivation: the offline QTIP encode is the practical bottleneck. Measured on
gfx1151: **Llama-3.2-1B → qtip4 ran >1h at ~25 CPU cores** (beam=128, all
linears) and hadn't finished. The *decode* path is done and fast; the *encoder*
does not scale. `--beam N` (commit adding `qtip_beam_width`) is the cheap stopgap
(beam=32 ≈ 4× faster, decode err 0.074 vs 0.073 at 128 — negligible). The real
fix is to move the beam search to the GPU.

## Why it's a strong GPU fit

- **Across groups: embarrassingly parallel.** Each 256-weight group encodes
  independently (the trellis resets to state 0 at every group boundary). A 1B
  model is ~4M groups → one block/warp per group, thousands in flight vs 25 CPU
  cores.
- **Within a group: only the 256-position sweep is sequential.** Per position:
  expand `beam × 2^bits` candidates (128×16 = 2048 for qtip4), score each (one
  squared diff vs the computed codebook), prune to top-`beam`. Expansion +
  scoring are parallel; the codebook is the computed 1MAD hash the decode kernel
  already evaluates on-device (zero LDS, no table).

## Reference: the CPU encoder to port

`hipfire_quant_codecs::qtip::beam_encode_group_bits` (moved into the shared crate
this session). Per position:
1. expand each beam `(state, cost)` over `2^bits` symbols → `(s_new, cost+diff²,
   prev_idx, sym)`;
2. `sort_unstable_by (state, cost)` then `dedup_by_key(state)` (keep min-cost
   predecessor per state);
3. `select_nth_unstable_by(beam)` → top-`beam` by cost;
4. record backpointers; after 256 positions, pick min-cost final beam slot and
   backtrack to recover symbols. Then `optimal_scale_bits` (closed form).

## GPU mapping (proposed)

- **One block per group**, block size ~= `beam` (e.g. 128 threads); grid =
  n_groups. Optionally batch a few groups per block if occupancy allows.
- LDS: the 256 rotated weights, the current beam `(state,cost)` (128 entries),
  and per-position backpointers (256×128 `(prev_idx,sym)` — 256*128*3B ≈ 96 KB,
  too big for LDS; keep backpointers in **global** scratch [n_groups × 256 ×
  beam], or recompute via a second forward pass — decide by measuring).
- Per position: each thread owns a slice of the `beam×2^bits` candidates,
  computes cost, then a **block-level top-`beam` selection** over 2048 entries.
  This is the one non-trivial kernel primitive — options:
  - bitonic sort of 2048 in LDS then take first `beam` after a per-state dedup;
  - or a hash-into-LDS keyed by state keeping min cost (dedup for free), then a
    partial top-`beam` selection. State space is 2^12 = 4096, so a per-state
    min-cost table in LDS (4096 floats = 16 KB) makes dedup O(candidates) with
    atomics, then top-`beam` over the touched states.
- Backtrack (sequential, 256 steps, one thread) → symbols; compute optimal
  scale; write the packed group (`pack_qtip{3,4}_group` layout) directly.
- The forward **FWHT-256 rotation** of each group can run in the same kernel
  (the decode path already has FWHT), keeping rotate→encode→pack GPU-resident;
  weights stream in, packed groups stream out.

## Key simplification

The GPU encoder **need not match the CPU encoder bit-for-bit**. Any valid symbol
stream decodes correctly — the `.hfq` is self-consistent (its symbols + the
decode kernel *define* the stored weights). So the GPU beam may use its own
tie-breaking / a simpler top-k and still be correct. The only loss is
cross-encoder byte-identity of artifacts (a reproducibility nicety). ⇒ no need to
replicate Rust's `sort_unstable`/`dedup`/`select_nth` semantics.

## Integration

- New kernel(s) `qtip_beam_encode_g256` (+ bits param) in `kernels/` +
  `rdna-compute` dispatch method taking rotated weights → packed groups.
- Offline driver (`hipfire-quantize::pack_qtip_real_tensors`) gains a GPU path:
  upload the staged BF16 tensor, encode on GPU, download packed. Gate behind a
  flag (`--gpu-encode`) until proven; keep the CPU path as the reference/oracle.
- **Parity**: GPU-encoded symbols decoded (CPU `decode_group_bits`) must have
  MSE ≤ CPU-encoder MSE (not equal — better-or-equal). Reuse the
  `parity_gemv_qtip4g256` oracle scaffolding.

## Expected payoff / effort

- Payoff: ~10–30× over 25-core CPU (thousands of lanes) → ~1h/1B down to minutes;
  makes qtip3/qtip4 practical at ≥7B.
- Effort: a real kernel project (the LDS top-k is the crux), on the order of the
  W3A4 GEMM tuning arc. Not a quick win — but the decode side is already done,
  so this is the last piece for QTIP to be deployment-practical at scale.

## Sequencing

1. `--beam` stopgap [DONE].
2. Prototype + validate [DONE — commit adding qtip_viterbi_encode.hip]. Chose
   **full Viterbi** over beam: STATE_BITS=12 → only 4096 states, enumerable, so no
   top-k primitive needed and the result is OPTIMAL. kernels/src/qtip_viterbi_encode.hip
   (one block/group, dp[4096] ping-pong in 32KB LDS, per-state min over 2^bits
   predecessors, predecessor-selector backptr to global scratch, sequential
   backtrack). Gpu::qtip_viterbi_encode + parity_qtip_viterbi_encode example.
   VALIDATED gfx1151 vs CPU beam-128: mean MSE ratio 0.968 (GPU BETTER — optimal
   beats beam), scale match 1.2e-7, **253× faster than 1-core CPU** (325ms vs 82s
   @8192 groups) ≈ 20× vs the 25-core production path → ~1h/1B down to ~3min.
   worst_group_ratio ~1.05 is the expected optimal-scale-refit artifact (encode
   optimizes at the RMS seed; MSE measured at the refit scale), mean is better.
3. Productionize [NEXT]: hipfire-quantize is CPU-only today; add `rdna-compute` as
   a dep + a `--gpu-encode` path in pack_qtip_real_tensors (CPU FWHT-rotate → upload
   rotated weights → qtip_viterbi_encode → download symbols → CPU optimal_scale +
   pack). Default stays CPU beam (no GPU requirement unless --gpu-encode). Keep the
   CPU path as the reference. NOTE: this is the first OFFLINE tool to link rdna-compute.
4. Perf: the backptr global-scratch traffic (256×4096 B/group) likely dominates at
   scale — could shrink to `bits`-bit selectors or tile groups/block. LDS 32KB caps
   occupancy at 2 blocks/CU; measure before optimizing.
