#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFP4G32 decode-path GEMV using gfx11 (RDNA3) v_dot2_f32_f16 instruction
// via __builtin_amdgcn_fdot2. One warp per output row M; each thread
// accumulates a slice of K, warp-reduces at the end.
//
// MOTIVATION: gfx1100 decode GEMV on HFP4G32 was severely ALU-bound —
// measured 8-40% peak BW across decode shapes on 7900 XTX (vs ~78% on
// gfx1201 for FFN). The fallback `gemv_hfp4g32.hip` reads x as F32,
// dequants weight to F16, then multiplies F32×F32 — ~8 single-VALU
// ops per 4 K-elements. Replacing the multiply chain with
// `__builtin_amdgcn_fdot2` (one v_dot2_f32_f16 = 2 FP16 muls + 1
// FP32 add per VALU) cuts the inner-loop multiply count ~4×.
//
// Activation x is consumed as F32 and converted to FP16 INLINE per
// K-block. The pre-pass `ensure_fp16_x` approach (v1) added a kernel
// launch per gemv call that exceeded the dot2 savings in production
// decode (0.89× on 0.8B HFP4G32, 2026-05-11) — same trap as the
// gfx12 FP8 v1 ensure_fp8_x pre-pass. F32→FP16 cvt is one VALU per
// element (essentially free), unlike FP8 cvt_pk_fp8_f32 which carried
// real per-call cost.
//
// LAYOUT MATCH WITH FALLBACK: same per-row 16B header + 8 × 17B blocks
// per 256-K group. Same thread-to-block mapping (block_in_iter = tid >> 2,
// nibble byte-offset = (tid & 3) * 4). NOT byte-exact with the
// fallback's add ordering — fdot2 uses F32 accumulator + per-block
// scale FMA, different reduction structure; tolerance ~2-3% relative
// against F32 reference.
//
// SCALE APPLICATION: per-block UE8M0 × row_scale (F32) is applied to
// the F32 partial AFTER each block's 4 fdot2 calls covering 8
// K-elements. No cross-lane work — each thread owns its block's scale
// natively under the HFP4G32 layout.
//
// Grid: [M, 1, 1]. Block: [32]. LDS: 32B (16-entry E2M1 FP16 LUT).

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfp4g32_dot2_gfx11(
    const char*  __restrict__ A,    // [M] rows of HFP4G32
    const float* __restrict__ x,    // [K] F32, converted to FP16 inline
    float*       __restrict__ y,    // [M]
    int M, int K
) {
    typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;

    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    // 16-entry signed E2M1 FP16 LUT in LDS.
    __shared__ _Float16 lut[16];
    if (tid < 16) {
        const float E2M1[16] = {
            +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        lut[tid] = (_Float16)E2M1[tid];
    }
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
    //   block_in_iter = tid >> 2  (0..7)
    //   sub           = tid & 3   (0..3)
    // Each thread covers 8 K-elements per group at base = group*256 + tid*8.
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

        // Load 8 F32 activation values for this thread's K-range and
        // convert to FP16 inline. F32→FP16 is one VALU per element
        // (truncate mantissa). Inlining the cvt avoids the pre-pass
        // kernel launch that ensure_fp16_x would add (~3-5 µs per
        // call) — that overhead exceeded the dot2 savings in
        // production decode (measured 0.89× on 0.8B HFP4G32 on
        // gfx1100 with ensure_fp16_x pre-pass, 2026-05-11).
        const int x_base = g * 256 + tid * 8;
        const float* xp = x + x_base;
        const half2_t x01 = {(_Float16)xp[0], (_Float16)xp[1]};
        const half2_t x23 = {(_Float16)xp[2], (_Float16)xp[3]};
        const half2_t x45 = {(_Float16)xp[4], (_Float16)xp[5]};
        const half2_t x67 = {(_Float16)xp[6], (_Float16)xp[7]};

        // Dequant nibbles → 8 FP16 weight values (unscaled), packed
        // into half2 pairs. Per-block scale `sc` is applied to the F32
        // partial after fdot2 accumulation.
        const half2_t w01 = {lut[(pk      ) & 0xFu], lut[(pk >>  4) & 0xFu]};
        const half2_t w23 = {lut[(pk >>  8) & 0xFu], lut[(pk >> 12) & 0xFu]};
        const half2_t w45 = {lut[(pk >> 16) & 0xFu], lut[(pk >> 20) & 0xFu]};
        const half2_t w67 = {lut[(pk >> 24) & 0xFu], lut[(pk >> 28) & 0xFu]};

        // 4 fdot2 with INDEPENDENT partial accumulators — each
        // `fdot2(a, b, 0.0)` has no carry dependency on the others, so
        // the compiler can dual-issue them. The single-carry version
        // (partial = fdot2(..., partial) ×4) emits clean disassembly
        // but its 4-deep dependency chain (fdot2 has ~4-cycle latency)
        // measured 0.90× in production decode despite 2.4× fewer
        // total instructions vs fallback — ILP, not instruction count,
        // is the binding constraint here.
        float p0 = __builtin_amdgcn_fdot2(w01, x01, 0.0f, false);
        float p1 = __builtin_amdgcn_fdot2(w23, x23, 0.0f, false);
        float p2 = __builtin_amdgcn_fdot2(w45, x45, 0.0f, false);
        float p3 = __builtin_amdgcn_fdot2(w67, x67, 0.0f, false);

        // Apply per-block scale (row_scale × UE8M0) and accumulate.
        // Reduce 4 partials in a balanced tree to preserve numerical
        // determinism across re-runs and keep the dep chain shallow.
        const float partial = (p0 + p1) + (p2 + p3);
        acc = __builtin_fmaf(partial, sc, acc);
    }

    // Warp reduce + write.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) y[row] = acc;
}
