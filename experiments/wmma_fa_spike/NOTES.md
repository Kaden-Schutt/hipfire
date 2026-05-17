# Phase 1.0 spike — design notes

## Status
- Scalar fp16 baseline kernel: written (`fa_scalar_fp16.hip`). 99 lines. Strips asym dequant + Givens from `attention_flash_asym2_tile_batched.hip`. Partials layout identical to production — consumable by `attention_flash_asym_reduce_batched.hip`.
- WMMA kernel: pending. First draft had two layout bugs (V B-fragment, header-write half-wave partition); deleted rather than commit broken. Awaiting PMC bandwidth-bound result before rewriting.
- Spike harness (Rust): not started.

## WMMA fragment layout — cheat sheet

WMMA: `D = A @ B^T + C`. `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a16, b16, c8) -> d8`.

For each wave32 invocation:
- `a16` is M-row fragment: lane `(tid & 15)` provides A-row `(tid & 15)`'s 16 K-dim values. Lanes 16..31 hold redundant copies of rows 0..15.
- `b16` is N-row fragment: lane `(tid & 15)` provides B-row `(tid & 15)`'s 16 K-dim values. Same redundancy.
- `c8` and `d8` are M×N output in C-layout: lane's 8 cells map to `(row, col) = (2*j + (tid >> 4), tid & 15)` for `j = 0..7`.

## FA loop nest (one workgroup at (h, m_tile, kv_tile))

```
acc_o[ND_FRAGS=8] = 0       // 8 × float8_t per lane → 64 VGPRs (hd=128)
m_state[8] = -inf
l_state[8] = 0

for kvb in [0, BLOCK_N=64, 2*BLOCK_N) :    // 2× per TILE_SIZE=128
  for sub in [0, WMMA_N=16, ..., BLOCK_N) : // 4× per BLOCK_N
    // ───── QK^T WMMA ─────
    // A=Q, B=K (both natural-coalesced loads from global)
    acc_qk = 0
    for k0 in [0, BLOCK_K=16, ..., head_dim) :  // 8× for hd=128
      a_reg[0..15] = Q[m_start + (tid&15)][h][k0..k0+15]   // coalesced
      b_reg[0..15] = K[kv_sub_start + (tid&15)][kv_h][k0..k0+15]  // coalesced
      acc_qk = wmma(a_reg, b_reg, acc_qk)

    // ───── Scale + causal mask ─────
    for j in 0..8 :
      cell_row = 2*j + (tid>>4)
      col_kv_pos = kv_sub_start + (tid&15)
      acc_qk[j] = (col_kv_pos <= lds_q_pos[cell_row]) ? acc_qk[j]*scale : -inf

    // ───── FA-2 online softmax (per row, in C-layout) ─────
    for j in 0..8 :
      row_max = __shfl_xor reduce over off ∈ {1,2,4,8}   // half-wave only, NEVER off=16
      m_new[j] = max(m_state[j], row_max)
      alpha[j] = exp(m_state[j] - m_new[j])
      acc_qk[j] = exp(acc_qk[j] - m_new[j])               // overwrite as P (still fp32)
      row_sum = __shfl_xor reduce
      l_new[j] = alpha[j]*l_state[j] + row_sum

    // ───── LDS round-trip: P from C-layout to A-layout ─────
    // Write: lane writes its 8 cells to lds_p[row][col].
    // Each cell (R, C) is written by exactly one lane (tid = C + (R%2)*16, j = R/2).
    // Bank-conflict-free if 16-wide stride: write lds_p[R*17 + C] to skew, or use fp32 LDS.
    for j in 0..8 :
      lds_p[2*j + (tid>>4)][tid&15] = acc_qk[j]            // fp32 LDS, narrow on load
    __syncthreads

    // ───── V LDS staging — KEY CORRECTNESS POINT ─────
    // Natural V load is A-layout (one KV pos × 16 V-dims per lane = coalesced).
    // WMMA P@V needs V in B-layout (one V-dim × 16 KV pos per lane = strided).
    // Solution: load V into LDS in natural order, read back transposed.
    //
    // V LDS: lds_v[BLOCK_N_sub=16][head_dim=128] = 4 KB (fp16). Per sub-block.
    //
    // Load (coalesced):
    //   each lane loads 16 KV-positions × 4 head-dims = 64 fp16 = 128 B
    //   (32 lanes × 128 B = 4 KB, covers all 16×128 cells)
    //   Specifically: thread (tid) writes lds_v[tid % 16][(tid/16)*64 + j] for j in 0..63
    //   (Or any other coalesced pattern — there's freedom here.)
    //
    // Read (B-layout for WMMA):
    //   v_reg[i] = lds_v[i][nd*16 + (tid&15)]  for i in 0..15
    //   Strided LDS read but fast (banked, hot in cache).

    for nd in 0..ND_FRAGS :   // 8 V-dim chunks for hd=128
      v_dim_start = nd * 16
      v_reg[0..15] = lds_v[0..15][v_dim_start + (tid&15)]
      p_reg[0..15] = lds_p[(tid&15)][0..15]              // A-layout reload, narrow to fp16
      pv = wmma(p_reg, v_reg, 0)
      for j in 0..8 :
        acc_o[nd][j] = alpha[j] * acc_o[nd][j] + pv[j]

    m_state = m_new
    l_state = l_new
    __syncthreads                                          // before reusing lds_p / lds_v

// ───── Write partials ─────
// acc_o[nd][j] is at (row = 2*j + (tid>>4), col = nd*16 + (tid&15))
// Partials layout: [batch × n_heads × max_tiles × (2 + head_dim)]
// Per-row stride is contiguous; per lane writes 8 row-slots × 8 V-dim-slots = 64 elements.
//
// m_state[j], l_state[j] are broadcast across half-wave cols via __shfl reductions.
// Header write: lane with (tid&15) == 0 in each half-wave writes its 8 rows' headers.
```

## Open issues to resolve in implementation

1. **LDS bank conflicts on the fp32 P-LDS**. With 32 banks of 4 B each, accessing column `c` from lane `c` is conflict-free for fp32, but reading a full row of 16 fp32 (64 B / lane via vector load) may conflict. Verify with `gfx-kernel-metadata` or rocprof.
2. **Strided V LDS reads**. Lane `(tid & 15)` reads 16 entries at stride `head_dim` in LDS bank space. Stride 128 fp16 entries = 256 B = 64 banks of stride. Per access, the 32 lanes hit 16 banks each (every other bank). Should be conflict-free but verify.
3. **Header redundancy**: m_state[j] / l_state[j] are valid in 16 lanes per half-wave after `__shfl_xor` reductions. Pick `(tid & 15) == 0` to write — that lane in each half-wave is responsible for its own redundant_bit's 8 rows.
4. **OOB seq_len handling**: cell-level mask via lds_q_pos was correct in v1. Per-row "skip if seq_len ≤ kv_tile_start" guard before writing partials.

## Why we don't pre-rotate Q (rev-2 plan change)

Original plan had a separate Givens pre-pass kernel. Consolidated review A12 said inline. For this spike Q comes in pre-rotated (synthetic test inputs), so the question doesn't arise — but for the production kernel: do Givens inline at the Q load (a_reg fill), since it's just 4 fmadd-pairs per dim-quadruple per lane.

## What this spike does NOT test

- Asym dequant overhead (intentional — spike isolates ALU)
- Givens rotation overhead (irrelevant for fp16-K spike)
- gfx12 builtin variant (Phase 3 concern)
- Tree-bias path (Phase 3)
- head_dim != 128 (deferred; spike pinned to 128)

## What this spike DOES test (the only valid claim from a positive result)

"WMMA-FA on pre-dequantized fp16 K/V is N% faster/slower than scalar wave32 FA on the same inputs, at this arch, this batch shape, this seq length."

A positive result here is necessary-but-not-sufficient to claim the production asym4 WMMA-FA will win — the production kernel adds inline dequant overhead which may erode the spike's margin.
