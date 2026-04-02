//! Built-in HIP kernel sources for inference operations.

/// GEMV F32: y = alpha * A * x + beta * y
/// Uses shared memory reduction across wavefronts.
pub const GEMV_SRC: &str = include_str!("../../../kernels/src/gemv.hip");


/// GEMV Q4_K: matrix-vector multiply with on-the-fly Q4_K dequantization.
/// A is stored as Q4_K blocks (144 bytes per 256 elements).
/// x is F32, y is F32. y = A_dequant * x.
///
/// Q4_K block layout (144 bytes for 256 elements):
///   [0:2]   f16 d (super-block scale)
///   [2:4]   f16 dmin (super-block min)
///   [4:16]  scales[12] (packed 6-bit scales/mins for 8 sub-blocks)
///   [16:144] qs[128] (4-bit quantized values, paired sub-blocks share 32 bytes)
///
/// Data layout: 4 groups of 64 elements. Each group has 2 sub-blocks sharing 32 bytes.
///   Group g (elements g*64..g*64+63):
///     sub-block 2g:   lower nibbles of qs[g*32..g*32+32] → elements g*64+0..g*64+31
///     sub-block 2g+1: upper nibbles of qs[g*32..g*32+32] → elements g*64+32..g*64+63
pub const GEMV_Q4K_SRC: &str = include_str!("../../../kernels/src/gemv_q4k.hip");


/// HFQ4-G128: flat 4-bit with 128-weight groups.
/// Block: [f32 scale][f32 zero][64B nibbles] = 72 bytes per 128 weights.
/// Minimal metadata → minimal VGPRs. Hypothesis: ≤32 VGPRs → max occupancy.
pub const GEMV_HFQ4G128_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g128.hip");


/// HFQ4-G128 batched GEMM: same tiled approach as G256 but 72 bytes/group, 4 weights/thread.
pub const GEMM_HFQ4G128_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g128.hip");


/// HFQ2-G256: flat 2-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][64B data] = 72 bytes per 256 weights (0.28 B/w).
pub const GEMV_HFQ2G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq2g256.hip");


/// HFQ8-G256: flat 8-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][256B data] = 264 bytes per 256 weights (1.03 B/w).
pub const GEMV_HFQ8G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq8g256.hip");


/// HFQ6-G256: flat 6-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][192B data] = 200 bytes per 256 weights (0.78 B/w).
/// Packing: 4 weights per 3 bytes (24 bits = 4×6 bits).
pub const GEMV_HFQ6G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq6g256.hip");


/// HFQ3-G256: flat 3-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][96B data] = 104 bytes per 256 weights (0.41 B/w).
/// Packing: 8 weights per 3 bytes (24 bits = 8×3 bits).
pub const GEMV_HFQ3G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq3g256.hip");


/// HFQ4-G512: flat 4-bit with 512-weight groups.
/// Block: [f32 scale][f32 zero][256B nibbles] = 264 bytes per 512 weights (0.516 B/w).
/// 264B ≈ 1 PCIe TLP, 2 L2 cache lines.
pub const GEMV_HFQ4G512_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g512.hip");


/// HFQ4-G1024: flat 4-bit with 1024-weight groups.
/// Block: [f32 scale][f32 zero][512B nibbles] = 520 bytes per 1024 weights (0.508 B/w).
pub const GEMV_HFQ4G1024_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g1024.hip");


/// HFQ4-G256: flat 4-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 weights.
/// Same coalesced width as Q4_K, 14 VGPRs instead of 39.
pub const GEMV_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256.hip");



/// HFQ2-G128: flat 2-bit with 128-weight groups. Finer granularity than G256.
/// [f32 scale (4B)][f32 zero (4B)][2-bit × 128 (32B)] = 40 bytes per 128 weights (0.3125 B/w).
/// 32 threads × 4 elements = 128 per group. Each thread reads 1 byte.
pub const GEMV_HFQ2G128_SRC: &str = include_str!("../../../kernels/src/gemv_hfq2g128.hip");


/// HFQ4-G256 wide GEMV: 2 rows per block (64 threads = 2 warps).
/// Each warp processes one row independently. Halves grid size.
pub const GEMV_HFQ4G256_WIDE_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256_wide.hip");


/// HFQ4-G256 batched GEMM: y[batch][row] = sum_k(A[row][k] * x[batch][k])
/// Loads weight data ONCE per group, multiplies against BATCH_TILE input vectors.
/// Grid: [M, ceil(batch_size/BATCH_TILE), 1]. Each block handles one row × BATCH_TILE batch elements.
/// x layout: [batch_size × K] row-major. y layout: [batch_size × M] row-major.
/// BATCH_TILE=8 keeps register pressure at ~26 VGPRs for good occupancy on RDNA.
pub const GEMM_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256.hip");


/// Fused QKV Q4_K: three GEMVs in one kernel launch.
/// Grid = (q_m + k_m + v_m) blocks. Each block determines which matrix by blockIdx range.
/// All three projections read the same input x (cached). Saves 2 kernel launches per layer.
pub const FUSED_QKV_Q4K_SRC: &str = include_str!("../../../kernels/src/fused_qkv_q4k.hip");


/// Fused Gate+Up Q4_K: two GEMVs in one kernel launch for FFN gate and up projections.
/// Grid = (gate_m + up_m) blocks. Saves 1 kernel launch per layer.
pub const FUSED_GATE_UP_Q4K_SRC: &str = include_str!("../../../kernels/src/fused_gate_up_q4k.hip");


/// GEMV Q8_0: matrix-vector multiply with on-the-fly Q8_0 dequantization.
/// Q8_0 block: 2 bytes f16 scale + 32 bytes int8 = 34 bytes per 32 elements.
/// v3: Processes 8 blocks (256 elements) per outer iteration to match Q4_K's loop count.
/// Byte loads → no nibble extraction → 16 VGPRs → F32-class occupancy.
/// Q8_0 GEMV wide: 256 threads with shared memory reduction for small matrices.
/// Each thread processes K/256 elements strided, then tree-reduce via shared memory.
/// Better for dim=1024 where 32-thread kernel underutilizes the GPU.
pub const GEMV_Q8_0_WIDE_SRC: &str = include_str!("../../../kernels/src/gemv_q8_0_wide.hip");


pub const GEMV_Q8_0_SRC: &str = include_str!("../../../kernels/src/gemv_q8_0.hip");


/// GEMV Q6_K: matrix-vector multiply with on-the-fly Q6_K dequantization.
/// Q6_K block: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes per 256 elements.
pub const GEMV_Q6K_SRC: &str = include_str!("../../../kernels/src/gemv_q6k.hip");


/// RMSNorm: y[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
pub const RMSNORM_SRC: &str = include_str!("../../../kernels/src/rmsnorm.hip");


/// Element-wise add
pub const ADD_SRC: &str = include_str!("../../../kernels/src/add.hip");


/// Element-wise in-place add: a[i] += b[i]
pub const ADD_INPLACE_SRC: &str = include_str!("../../../kernels/src/add_inplace.hip");


/// Element-wise multiply
pub const MUL_SRC: &str = include_str!("../../../kernels/src/mul.hip");


/// SiLU (Sigmoid Linear Unit): silu(x) = x * sigmoid(x)
pub const SILU_SRC: &str = include_str!("../../../kernels/src/silu.hip");


/// Fused SiLU(gate) * up: out[i] = silu(gate[i]) * up[i]
/// Saves one kernel launch + one intermediate buffer.
pub const SILU_MUL_SRC: &str = include_str!("../../../kernels/src/silu_mul.hip");


/// Softmax over last dimension (one block per row)
pub const SOFTMAX_SRC: &str = include_str!("../../../kernels/src/softmax.hip");


/// RoPE (Rotary Positional Embedding)
pub const ROPE_SRC: &str = include_str!("../../../kernels/src/rope.hip");


/// Batched RoPE: apply RoPE to [batch_size] positions at once.
/// q: [batch_size × n_heads_q × head_dim], k: [batch_size × n_heads_k × head_dim]
/// positions: [batch_size] int array of position indices.
/// Grid: [half, batch_size, 1]. Each thread handles one (position, freq_index) pair.
pub const ROPE_BATCHED_SRC: &str = include_str!("../../../kernels/src/rope_batched.hip");


/// Single-head causal attention on GPU.
/// One thread block per query head. Handles GQA (kv_group heads share same KV).
/// q: [n_heads * head_dim], k_cache: [seq_len * n_kv_heads * head_dim],
/// v_cache: same layout, out: [n_heads * head_dim].
pub const ATTENTION_SRC: &str = include_str!("../../../kernels/src/attention.hip");


/// Flash-Decoding attention: split KV scan across multiple blocks per head.
/// Phase 1: each block processes a chunk of KV positions, writes partial (max, sum, output).
/// Phase 2: reduction across chunks using online softmax correction.
/// Grid: [n_heads, n_chunks, 1]. Each block handles one (head, chunk) pair.
/// Partial results stored in partials buffer: [n_heads × n_chunks × (1 + 1 + head_dim)] floats.
pub const ATTENTION_FLASH_SRC: &str = include_str!("../../../kernels/src/attention_flash.hip");


/// Fused Gate+Up HFQ4-G256: two GEMVs in one launch (saves 1 launch per layer).
/// Grid: [gate_m + up_m, 1, 1]. Each block processes one row from gate or up weight.
pub const FUSED_GATE_UP_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/fused_gate_up_hfq4g256.hip");


/// INT8 co-located KV v2: [f16 scale (2B)][padding (2B)][int8 × head_dim] = 132 bytes per head.
/// f16 scale matches Q8_0 but with one block per head. Padding for 4-byte alignment.
pub const KV_CACHE_WRITE_INT8C_F16_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_int8c_f16.hip");


/// Attention with INT8 co-located f16 scale KV.
pub const ATTENTION_INT8C_F16_KV_SRC: &str = include_str!("../../../kernels/src/attention_int8c_f16_kv.hip");


/// INT8 co-located KV: [f32 scale][int8 × head_dim] = 132 bytes per head.
/// Symmetric quantization, no zero point. Dequant: scale * (float)val.
/// Minimized VGPRs: no zero register, no nibble math.
pub const KV_CACHE_WRITE_INT8C_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_int8c.hip");


/// Attention with INT8 co-located KV. Deferred scale multiply, 4×32 unrolled inner loop.
/// Q preloaded into shared memory. Scale applied ONCE per position, not per element.
pub const ATTENTION_INT8C_KV_SRC: &str = include_str!("../../../kernels/src/attention_int8c_kv.hip");


/// HFQ8 KV: FP32 scale+zero per head, contiguous uint8 data. Asymmetric quantization.
/// Scales: [max_seq × n_kv_heads × 2] f32 (scale, zero pairs).
/// Data: [max_seq × n_kv_heads × head_dim] uint8.
pub const KV_CACHE_WRITE_HFQ8_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_hfq8.hip");


/// Attention with HFQ8 KV cache. Flat layout, FP32 scale+zero, contiguous uint8 data.
pub const ATTENTION_HFQ8_KV_SRC: &str = include_str!("../../../kernels/src/attention_hfq8_kv.hip");


/// INT8 KV with separate scale array. Contiguous int8 values, one f32 scale per head.
/// Keys: [max_seq × n_kv_heads × head_dim] int8, Scales: [max_seq × n_kv_heads] f32.
/// Write: one warp per head, find amax via shuffle, quantize 4 elements per thread.
pub const KV_CACHE_WRITE_INT8_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_int8.hip");


/// Attention with INT8 KV (separate scale array). Clean indexed access, no block math.
pub const ATTENTION_INT8_KV_SRC: &str = include_str!("../../../kernels/src/attention_int8_kv.hip");


/// Batched causal attention: all query positions attend to their causal context.
/// Grid: [n_heads, seq_len, 1]. Each block handles one (head, query_position) pair.
/// Q/K/V are FP32: [seq_len × n_heads × head_dim] or [seq_len × n_kv_heads × head_dim].
/// Output: [seq_len × n_heads × head_dim].
/// For prefill: Q/K/V come from batched projections. KV also written to cache.
pub const ATTENTION_CAUSAL_BATCHED_SRC: &str = include_str!("../../../kernels/src/attention_causal_batched.hip");


/// Batched Q8_0 KV cache write: quantize multiple positions at once.
/// src: [batch_size × kv_dim] FP32. positions: [batch_size] int32.
/// Grid: [total_blocks × batch_size]. Each block handles one Q8_0 group for one position.
pub const KV_CACHE_WRITE_Q8_0_BATCHED_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_q8_0_batched.hip");


/// Quantize KV vector to Q8_0 format (same as GGML Q8_0 / existing GEMV kernels).
/// Block: [f16 scale (2B)][int8 × 32 (32B)] = 34 bytes per 32 elements.
/// head_dim=128 → 4 blocks × 34 = 136 bytes per head.
/// Layout: [max_seq × n_kv_heads × blocks_per_head × 34].
pub const KV_CACHE_WRITE_Q8_0_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_q8_0.hip");


/// Attention with Q8_0 quantized KV cache — same format as GGML Q8_0.
/// K and V caches stored as [max_seq × n_kv_heads × blocks_per_head × 34].
pub const ATTENTION_Q8_0_KV_SRC: &str = include_str!("../../../kernels/src/attention_q8_0_kv.hip");


/// Quantize KV vector to Q8 (int8 symmetric) and write to quantized KV cache.
/// Per head: [4B f32 scale][head_dim × int8 values] = head_dim + 4 bytes.
/// For head_dim=128: 132 bytes vs 512 bytes FP32 = 3.88x compression.
pub const KV_CACHE_WRITE_Q8_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_q8.hip");


/// Attention with Q8 quantized KV cache — symmetric int8, dequant on read.
pub const ATTENTION_Q8KV_SRC: &str = include_str!("../../../kernels/src/attention_q8kv.hip");


/// HFQ4 KV block: co-located FP32 scale+zero + packed nibbles. 72 bytes per head.
/// Layout per position: [n_kv_heads × 72] bytes. One cache line per head.
pub const KV_CACHE_WRITE_HFQ4_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_hfq4.hip");


/// HFQ4 KV with sign-flip decorrelation. Same format as HFQ4 (72B/head),
/// but values are multiplied by TURBO_SIGNS1 before quantization.
/// On read: Q is sign-flipped, V output is sign-flipped.
pub const KV_CACHE_WRITE_HFQ4S_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_hfq4s.hip");


/// Attention with HFQ4+sign-flip KV: Q sign-flipped before dot, V output sign-flipped.
/// Same HFQ4 block format (72B/head). Uses TURBO_SIGNS1 from constant memory.
pub const ATTENTION_HFQ4S_KV_SRC: &str = include_str!("../../../kernels/src/attention_hfq4s_kv.hip");


/// Attention with HFQ4 KV blocks v2. Tight single-block pattern.
/// 72 bytes per head = one HFQ4-G128 block (scale+zero+64 nibble bytes).
/// Q preloaded into shared memory. One scale+zero load per position.
pub const ATTENTION_HFQ4_KV_SRC: &str = include_str!("../../../kernels/src/attention_hfq4_kv.hip");


/// Quantize KV vector to HFQ4-G128 and write to quantized KV cache.
/// Input: kv_dim floats at kv_src. Output: packed HFQ4 at dst[pos * bytes_per_pos].
/// Each group of 128 floats → 72 bytes (4B scale + 4B zero + 64B nibbles).
/// For head_dim=128, one head = exactly one group = 72 bytes.
pub const KV_CACHE_WRITE_Q4_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_q4.hip");


/// Attention with quantized HFQ4 KV cache.
/// Same structure as attention_f32 but dequantizes K and V on the fly.
pub const ATTENTION_Q4KV_SRC: &str = include_str!("../../../kernels/src/attention_q4kv.hip");


// ═══════════════════════════════════════════════════════════════════════
// DeltaNet ops (Qwen3.5 linear attention)
// ═══════════════════════════════════════════════════════════════════════

/// Sigmoid: σ(x) = 1 / (1 + exp(-x)). Element-wise, in-place.
#[cfg(feature = "deltanet")]
pub const SIGMOID_SRC: &str = include_str!("../../../kernels/src/sigmoid.hip");


/// Softplus: log(1 + exp(x)), numerically stable. Element-wise, in-place.
#[cfg(feature = "deltanet")]
pub const SOFTPLUS_SRC: &str = include_str!("../../../kernels/src/softplus.hip");


/// L2 normalization per head: out[i] = x[i] / sqrt(sum(x²) + eps).
/// Grid: [n_heads]. Block: [32]. Each warp normalizes one head of head_dim elements.
#[cfg(feature = "deltanet")]
pub const L2_NORM_SRC: &str = include_str!("../../../kernels/src/l2_norm.hip");


/// Partial interleaved RoPE: rotate only first n_rot dims, pairs are adjacent (d0,d1),(d2,d3),...
/// Dims >= n_rot pass through unchanged.
/// Grid: [n_rot/2]. Block: [1]. Each thread handles one rotation pair.
#[cfg(feature = "deltanet")]
pub const ROPE_PARTIAL_INTERLEAVED_SRC: &str = include_str!("../../../kernels/src/rope_partial_interleaved.hip");


/// 1D causal depthwise convolution (kernel_size=4) with persistent ring buffer state.
/// For decode: one token at a time. conv_state: [n_channels × 3] ring buffer.
/// out[c] = w[0]*x[c] + w[1]*state[c][0] + w[2]*state[c][1] + w[3]*state[c][2]
/// Then shift state: state = [x, state[0], state[1]].
#[cfg(feature = "deltanet")]
pub const CONV1D_DECODE_SRC: &str = include_str!("../../../kernels/src/conv1d_decode.hip");


/// Gated output norm: rmsnorm(x) * silu(z). Fused single kernel.
/// x and z are [n_heads × head_dim]. weight is [head_dim] (shared across heads).
#[cfg(feature = "deltanet")]
pub const GATED_NORM_SRC: &str = include_str!("../../../kernels/src/gated_norm.hip");


/// Gated Delta Net — tiled LDS + warp-shuffle.
/// S[128×128] tiled into TILE_ROWS=8 row chunks. Each tile = 8×128×4 = 4KB LDS.
/// 64KB/4KB = 16 blocks/CU → 4 waves/SIMD. Rows are independent → perfect tiling.
/// 32 threads per block (one warp), each handles 4 columns.
/// Grid: [n_heads, HD/TILE_ROWS]. Block: [32].
#[cfg(feature = "deltanet")]
pub const GATED_DELTA_NET_SRC: &str = include_str!("../../../kernels/src/gated_delta_net.hip");


/// GDN Q8 — tiled LDS + warp-shuffle. Dequant tile into LDS, recurrence, requant back.
/// Tile = TILE_ROWS × 128 × 4B = 4KB. Same tiling as FP32 variant.
/// Grid: [n_heads, HD/TILE_ROWS]. Block: [32].
#[cfg(feature = "deltanet")]
pub const GATED_DELTA_NET_Q8_SRC: &str = include_str!("../../../kernels/src/gated_delta_net_q8.hip");


/// GDN recurrence with Q4-quantized S state in VRAM.
/// State layout: unsigned char s_q4[n_heads][HD*HD/2] (nibble-packed) + float s_scales[n_heads*HD].
/// Symmetric 4-bit: values -8..+7, scale = absmax/7. Per-row scale.
/// 8x compression vs FP32 (8KB + 512B scales per head vs 64KB).
#[cfg(feature = "deltanet")]
pub const GATED_DELTA_NET_Q4_SRC: &str = include_str!("../../../kernels/src/gated_delta_net_q4.hip");


/// Alpha gate compute on GPU: out[i] = softplus(alpha[i] + dt_bias[i]) * (-exp(a_log[i])).
/// Eliminates 85µs CPU roundtrip per DeltaNet layer.
#[cfg(feature = "deltanet")]
pub const ALPHA_GATE_SRC: &str = include_str!("../../../kernels/src/alpha_gate.hip");


/// Scale vector by constant: x[i] *= scale. Eliminates 48µs CPU roundtrip.
#[cfg(feature = "deltanet")]
pub const SCALE_F32_SRC: &str = include_str!("../../../kernels/src/scale_f32.hip");


/// Fused conv1d (kernel_size=4) + SiLU. Eliminates one kernel launch.
#[cfg(feature = "deltanet")]
pub const CONV1D_SILU_SRC: &str = include_str!("../../../kernels/src/conv1d_silu.hip");


/// GPU-side KV cache write using pos from a GPU buffer.
/// Copies kv_dim floats from src to dst at offset pos_buf[0] * kv_dim.
pub const KV_CACHE_WRITE_SRC: &str = include_str!("../../../kernels/src/kv_cache_write.hip");


/// GPU-side top-K + top-P sampling. Eliminates 600KB logits download per token.
/// Single block, 256 threads. Returns token ID + RNG state (8 bytes vs 600KB).
///
/// Phase 1: Parallel max reduction over vocab_size logits.
/// Phase 2: Threshold filter — collect candidates within 30*temp of max (atomic shared counter).
/// Phase 3: Thread 0 softmax + sort + top-p + sample on the small candidate set.
pub const SAMPLE_TOP_P_SRC: &str = include_str!("../../../kernels/src/sample_top_p.hip");


/// GEMV Q4_F16_G64: matrix-vector multiply with on-the-fly Q4_F16 dequantization.
/// Block layout: f16 scale (2B) + f16 min (2B) + uint8 quants[32] (32B) = 36 bytes per 64 elements.
/// Dequant: weight = (_Float16)(nibble) * scale + min — single FP16 FMA on RDNA.
/// Thread tid reads quants[tid], processes both nibbles (elements tid and tid+32).
pub const GEMV_Q4F16_G64_SRC: &str = include_str!("../../../kernels/src/gemv_q4f16_g64.hip");


/// GEMV Q4_F16_G64 wide: 256 threads, element-strided access, shared memory reduction.
/// Matches F32 GEMV's occupancy pattern to test whether occupancy explains the 40% vs 48% gap.
/// Each thread processes elements tid, tid+256, tid+512, ... across the row.
pub const GEMV_Q4F16_G64_WIDE_SRC: &str = include_str!("../../../kernels/src/gemv_q4f16_g64_wide.hip");


/// GEMV Q4_F16_G32: matrix-vector multiply with Q4_F16 group-32 dequantization.
/// Block layout: f16 scale (2B) + f16 min (2B) + uint8 quants[16] (16B) = 20 bytes per 32 elements.
/// Thread tid reads quants[tid&15], extracts its nibble based on tid < 16 or >= 16.
pub const GEMV_Q4F16_G32_SRC: &str = include_str!("../../../kernels/src/gemv_q4f16_g32.hip");


/// Q8_0 embedding lookup: dequantize one row from a Q8_0 table to F32.
/// Block: 2 bytes f16 scale + 32 bytes int8 = 34 bytes per 32 elements.
pub const EMBEDDING_Q8_SRC: &str = include_str!("../../../kernels/src/embedding_q8.hip");


/// Q4_K embedding lookup: dequantize one row from a Q4_K table to F32.
/// Avoids dequanting entire embedding to F32 (saves ~2GB for 150K+ vocabs).
/// 256 threads, one block, strided across the row's Q4_K blocks.
pub const EMBEDDING_Q4K_SRC: &str = include_str!("../../../kernels/src/embedding_q4k.hip");


/// HFQ4-G256 embedding lookup: dequantize one row from HFQ4-G256 table to F32.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 elements.
pub const EMBEDDING_HFQ4G256_SRC: &str = include_str!("../../../kernels/src/embedding_hfq4g256.hip");


/// HFQ4-G128 embedding lookup: dequantize one row from HFQ4-G128 table to F32.
pub const EMBEDDING_HFQ4G128_SRC: &str = include_str!("../../../kernels/src/embedding_hfq4g128.hip");


/// Q4_LUT GEMV: 4-bit with LDS codebook lookup.
/// Block: f16 codebook[16] (32 bytes) + u8 quants[16] (16 bytes) = 48 bytes per 32 elements.
/// Dequant: nibble → LDS[nibble] → f16 → FMA. No scale arithmetic per element.
/// 32 threads (single warp). Processes 8 blocks (256 elems) per outer iteration like Q8.
pub const GEMV_Q4LUT_SRC: &str = include_str!("../../../kernels/src/gemv_q4lut.hip");


/// Wave-cooperative Q4: use warp shuffle to distribute nibbles.
/// Same Q4_F16_G32 format (20 bytes/32 elem = 0.625 B/w).
/// 16 threads load 16 bytes, shuffle to give all 32 threads one nibble each.
/// Avoids the tid<16 conditional branch in the inner loop.
pub const GEMV_Q4WAVE_SRC: &str = include_str!("../../../kernels/src/gemv_q4wave.hip");


/// Q4 stored as Q8: 4-bit precision quantized but stored in int8 (1 byte per weight).
/// Same as Q8_0 format (34 bytes per 32 elements) but values clamped to [-8,7].
/// Gets Q8 occupancy (16 VGPRs, 84% peak BW) at 4-bit quality.
/// 1.0625 bytes/weight — only useful when VRAM is not the constraint.
pub const GEMV_Q4AS8_SRC: &str = include_str!("../../../kernels/src/gemv_q4as8.hip");


/// GEMV Q8_HFQ: split-metadata row layout — scales contiguous, then values contiguous.
/// Row layout: [f16 scales × n_groups | int8 values × K | padding to 128B]
/// Pure sequential value stream with no metadata gaps every 34 bytes.
/// Narrow variant: 32 threads (1 warp), 8x unrolled, warp shuffle reduction.
pub const GEMV_Q8HFQ_SRC: &str = include_str!("../../../kernels/src/gemv_q8hfq.hip");


/// GEMV Q8_HFQ wide: 2 warps per block, each processes one row independently.
/// Same split-metadata layout. 8x unrolled. Grid = ceil(M/2).
pub const GEMV_Q8HFQ_WIDE_SRC: &str = include_str!("../../../kernels/src/gemv_q8hfq_wide.hip");


/// Cross-entropy loss: -log(softmax[target]) computed entirely on GPU.
/// Input: logits[vocab_size], target_id (int). Output: loss (float).
/// Single block, 256 threads: parallel log-sum-exp reduction.
pub const CROSS_ENTROPY_LOSS_SRC: &str = include_str!("../../../kernels/src/cross_entropy_loss.hip");


/// GPU max-probability: compute max(softmax(logits)) entirely on GPU.
/// Output: single float = probability of the most likely token.
/// Used for early-exit confidence check (downloads 4 bytes instead of 600KB).
pub const MAX_PROB_SRC: &str = include_str!("../../../kernels/src/max_prob.hip");


/// GPU argmax: find index of maximum value.
pub const ARGMAX_SRC: &str = include_str!("../../../kernels/src/argmax.hip");


// ═══════════════════════════════════════════════════════════════════════════════
// TurboQuant KV cache: FWHT rotation + norm-corrected quantization
// Supports turbo_hfq2 (2-bit), turbo_hfq3 (3-bit), turbo_hfq4 (4-bit)
// ═══════════════════════════════════════════════════════════════════════════════

/// Shared FWHT + centroid tables used by all turbo kernels.
/// Included as a preamble in each kernel source string.
pub const TURBO_COMMON_SRC: &str = include_str!("../../../kernels/src/turbo_common.hip");


/// KV cache write kernel for turbo_hfq4 (4-bit).
/// Layout per head: [f32 norm (4B)][nibbles × head_dim/2 (64B)] = 68 bytes for head_dim=128.
/// One block per kv_head, thread 0 does all work (head_dim=128 is serial per head).
pub const KV_CACHE_WRITE_TURBO4_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_turbo4.hip");


/// KV cache write kernel for turbo_hfq3 (3-bit, split 2+1).
/// Layout per head: [f32 norm (4B)][2-bit low × hd/4 (32B)][1-bit high × hd/8 (16B)] = 52 bytes.
/// Fused KV write for turbo_hfq3 (3-bit). K+V in one kernel, 32-thread parallel FWHT.
/// Grid: n_kv_heads blocks, 32 threads each. Two passes: K then V.
pub const KV_CACHE_WRITE_TURBO3_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_turbo3.hip");


/// KV cache write kernel for turbo_hfq2 (2-bit).
/// Layout per head: [f32 norm (4B)][2-bit × hd/4 (32B)] = 36 bytes for head_dim=128.
pub const KV_CACHE_WRITE_TURBO2_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_turbo2.hip");


/// Attention kernel with turbo_hfq4 KV cache (4-bit, nibble-packed).
/// One wavefront (32 threads) per attention head.
/// Q is pre-rotated in shared memory, V output is inverse-rotated.
pub const ATTENTION_TURBO4_KV_SRC: &str = include_str!("../../../kernels/src/attention_turbo4_kv.hip");


/// Optimized attention with turbo_hfq3 KV cache.
/// Threads own dimensions (4 each × 32 = 128): zero V reduction, warp-shuffle K dots.
pub const ATTENTION_TURBO3_KV_SRC: &str = include_str!("../../../kernels/src/attention_turbo3_kv.hip");


/// Attention kernel with turbo_hfq2 KV cache (2-bit).
pub const ATTENTION_TURBO2_KV_SRC: &str = include_str!("../../../kernels/src/attention_turbo2_kv.hip");


// ═══════════════════════════════════════════════════════════════════════════
// Vision encoder kernels (ViT: GEMM, LayerNorm, GELU, bias-add)
// ═══════════════════════════════════════════════════════════════════════════

/// Batched GEMV (= GEMM) for F16 weights, F32 activations.
/// Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T
/// Grid=[M,N], Block=[32]. Each warp computes one dot product via shuffle reduce.
pub const GEMM_F16_SRC: &str = include_str!("../../../kernels/src/gemm_f16.hip");


/// Batched GEMM for F32: Y[M,N] = A[M,K] @ B[N,K]^T
pub const GEMM_F32_SRC: &str = include_str!("../../../kernels/src/gemm_f32.hip");


/// LayerNorm with bias: out = gamma * (x - mean) / sqrt(var + eps) + beta
/// Grid=[batch], Block=[min(256, n)].
pub const LAYERNORM_SRC: &str = include_str!("../../../kernels/src/layernorm.hip");


/// GELU activation (tanh approximation, matches gelu_pytorch_tanh).
pub const GELU_TANH_SRC: &str = include_str!("../../../kernels/src/gelu_tanh.hip");


/// Transpose: out[c, r] = in[r, c]. Converts [rows, cols] → [cols, rows].
pub const TRANSPOSE_SRC: &str = include_str!("../../../kernels/src/transpose.hip");


/// Fused ViT self-attention: Q@K^T → softmax → @V, reading QKV from [N, 3*hidden].
/// Grid=[n_heads, N]. Each block computes one (head, query_pos) output row.
pub const VIT_ATTENTION_SRC: &str = include_str!("../../../kernels/src/vit_attention.hip");


/// Bias-add: X[batch, n] += bias[n] (broadcast over batch dim)
pub const BIAS_ADD_SRC: &str = include_str!("../../../kernels/src/bias_add.hip");




// Asymmetric turbo: Q8 K + turbo4 V for head_dim=256
pub const KV_CACHE_WRITE_TURBO4_V256_SRC: &str = include_str!("../../../kernels/src/kv_cache_write_turbo4_v256.hip");
pub const ATTENTION_Q8K_TURBO4V_256_SRC: &str = include_str!("../../../kernels/src/attention_q8k_turbo4v_256.hip");
