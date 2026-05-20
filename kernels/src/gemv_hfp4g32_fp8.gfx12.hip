#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFP4G32 decode-path GEMV using gfx12 (RDNA4) FP8 dot4 instruction
// (__builtin_amdgcn_dot4_f32_fp8_fp8). One warp per output row M; each
// thread accumulates a slice of K, warp-reduces at the end.
//
// MOTIVATION: decode GEMV on gfx1201 for small-M attention shapes (e.g.
// k_proj, v_proj at M=512 K=2048) is heavily ALU-bound — measured 16-30%
// peak BW for M ≤ 2048 K=2048 on R9700 (~150-240 GB/s effective vs
// 800 GB/s peak). The fallback `gemv_hfp4g32.hip` does ~6-7 F32 ops per
// K-element in the dequant + multiply-accumulate chain; replacing those
// with one `dot4_f32_fp8_fp8` per 4 elements cuts inner-loop ALU by
// ~2-2.4× and lifts decode tok/s on the ALU-bound shapes. BW-bound FFN
// shapes (gate_up, w_down at ~78% peak) see only ~10% lift.
//
// LAYOUT MATCH WITH FALLBACK: same per-row 16B header + 8 × 17B blocks
// per 256-K group. Same thread-to-block mapping (block_in_iter = tid >>
// 2, nibble byte-offset = (tid & 3) * 4). NOT byte-exact with the
// fallback's add ordering — FP8 path uses a different accumulator
// structure and a different intermediate precision; tolerance ~5%
// relative against FP32 reference.
//
// ACTIVATION FORMAT (v3 / A3): x is consumed as FP8 (E4M3) bytes,
// produced by the FUSED `mq_rotate_x_dual_fp8` kernel that writes
// both F32 and FP8 outputs in one launch (gfx12 only). Earlier
// attempts:
//   v1 (pre-pass + ensure_fp8_x cache by src_ptr): correct in
//     isolation but lost 0.93× on 9B Qwen 3.5 MFP4G32 in production
//     — `mq_x_rot` scratch has stable pointer + per-call data
//     turnover, forcing cache invalidation per weight_gemv call and
//     putting pack launch on critical path.
//   v2 / A2 (inline F32→FP8 cvt per K-block): 0.79-0.97× synthetic
//     bench — 4× activation memory pressure from per-warp F32 reads
//     overwhelmed any ALU savings.
//   v3 / A3 (this — fused FP8 pack at rotation time): rotation
//     already reads K F32 + writes K F32; adding K FP8 byte writes
//     is ~25% more output BW (8KB → 10KB for K=2048, negligible) +
//     4 cvt VALU ops per K-block per thread. The FP8 GEMV then
//     reads FP8 directly with no separate pack launch.
//
// SCALE APPLICATION: per-block UE8M0 × row_scale is applied to the F32
// dot4 partial AFTER the 2 dot4 calls covering that block (one per 4
// K-elements; 8 K-elements per thread per block). No cross-lane work —
// each thread owns its block's scale natively under the HFP4G32 layout.
//
// Grid: [M, 1, 1]. Block: [32]. LDS: 32B (16-entry E2M1->E4M3 byte LUT).

// E2M1 (FP4) values pre-converted to E4M3 (FP8) bytes; same table as
// the prefill FP8 kernel. E2M1 max=6 fits cleanly in E4M3 max=448 with
// no scale baked, so the LUT is a fixed 16-byte lookup.
__device__ static constexpr unsigned char E2M1_AS_E4M3[16] = {
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC,
};

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfp4g32_fp8_gfx12(
    const char*          __restrict__ A,        // [M] rows of HFP4G32
    const unsigned char* __restrict__ X_fp8,    // [K] E4M3 FP8, produced by mq_rotate_x_dual_fp8 (or pack_f32_to_fp8)
    float*               __restrict__ y,        // [M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    __shared__ unsigned char lut[16];
    if (tid < 16) lut[tid] = E2M1_AS_E4M3[tid];
    __syncthreads();

    // Same row addressing as the fallback kernel.
    const int blocks_per_row = K / 32;
    const long long row_stride = 16 + (long long)blocks_per_row * 17;
    const char* hdr = A + (long long)row * row_stride;
    const char* row_ptr = hdr + 16;

    const unsigned short rs_bits = *(const unsigned short*)hdr;
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, rs_bits);
    const float row_scale_f = (float)row_scale_h;

    // Thread layout within an 8-block group (256 K-elements):
    //   block_in_iter = tid >> 2   (0..7)   — which of the 8 blocks
    //   sub          = tid & 3     (0..3)   — which 8-element slice of the block
    // So each thread covers 8 K-elements per group at base = group*256 + tid*8.
    const int block_in_iter = tid >> 2;
    const int sub           = tid & 3;
    const int nib_off       = sub * 4;
    const int thread_block_off = block_in_iter * 17;

    const int groups_per_row = K / 256;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + (long long)g * 136 + thread_block_off;

        // UE8M0 -> F32 via direct exponent bit construction.
        const unsigned int e = *(const unsigned char*)gp;
        const float sc = row_scale_f * __builtin_bit_cast(float, e << 23);

        // 8 nibbles for this thread's 8 K-elements, packed in 32 bits.
        const unsigned int pk = *(const unsigned int*)(gp + 1 + nib_off);

        // Build weight FP8 bytes (8 bytes = 2 uint32) from nibble pack.
        // Each nibble -> 1 byte E4M3 via LDS LUT. Lane-local lookups.
        unsigned int w_lo, w_hi;
        w_lo  = (unsigned int)lut[(pk      ) & 0xFu];
        w_lo |= (unsigned int)lut[(pk >>  4) & 0xFu] << 8;
        w_lo |= (unsigned int)lut[(pk >>  8) & 0xFu] << 16;
        w_lo |= (unsigned int)lut[(pk >> 12) & 0xFu] << 24;
        w_hi  = (unsigned int)lut[(pk >> 16) & 0xFu];
        w_hi |= (unsigned int)lut[(pk >> 20) & 0xFu] << 8;
        w_hi |= (unsigned int)lut[(pk >> 24) & 0xFu] << 16;
        w_hi |= (unsigned int)lut[(pk >> 28) & 0xFu] << 24;

        // Read 8 FP8 activation bytes for this thread's K-range. The
        // FP8 stream was produced upstream by either:
        //   (a) mq_rotate_x_dual_fp8 — fused rotation + FP8 pack
        //       (preferred — eliminates the per-call pack launch)
        //   (b) pack_f32_to_fp8_gfx12 — standalone pre-pass (used when
        //       caller doesn't go through the MFP4G32 rotation path,
        //       e.g. test paths or HFP4G32 direct callers).
        const int x_base = g * 256 + tid * 8;
        const unsigned int x_lo = *(const unsigned int*)(X_fp8 + x_base);
        const unsigned int x_hi = *(const unsigned int*)(X_fp8 + x_base + 4);

        // Two dot4 calls cover 8 elements. F32 partial accumulator
        // chained between them (single dep chain, fine because the per-
        // block scale FMA at the end resets the dependency).
        float partial = 0.0f;
        partial = __builtin_amdgcn_dot4_f32_fp8_fp8(w_lo, x_lo, partial);
        partial = __builtin_amdgcn_dot4_f32_fp8_fp8(w_hi, x_hi, partial);

        // Apply per-block scale (row_scale × UE8M0) and accumulate.
        acc = __builtin_fmaf(partial, sc, acc);
    }

    // Warp reduce + write.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) y[row] = acc;
}
