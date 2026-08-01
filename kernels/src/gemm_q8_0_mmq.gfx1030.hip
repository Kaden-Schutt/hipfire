// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
// Dense Q8_0 × Q8_1 MMQ GEMM for exact gfx1030 via sdot4 — v7.
//
// Direct channel-test probe only; no production or A3B dispatch admits it.
// Host contract matches gemm_q8_0_mmq_4w_gfx1151:
//   W  : Q8_0 packed [M, K/32*34]  — per 32-K: [2B fp16 scale | 32B int8]
//   X  : block_q8_1_mmq [K/128 × N] via ensure_q8_1_mmq_x
//   Y  : FP32 [N, M] row-major
//
// v7: 32-row × 128-col tile, 256 threads (8 waves), BOTH operands LDS-staged.
// v3 widened the column tile but left every lane loading its own row pair,
// so each weight block was fetched 8× per workgroup (once per slot_oct
// group) — weight traffic unchanged vs v2. v4 stages the 32 rows' Q8_0
// blocks (34 B each → 9 dwords + f16 scale) in LDS once per (kb, sb) via
// lanes 0..31, then all 256 lanes read the pre-extracted int8 packs as LDS
// broadcasts. Global weight traffic is now exactly 1× per workgroup.
//
// Thread map:
//   row_local  = tid & 15        → rows row_start + row_local (+16 second row)
//   slot_oct   = tid >> 4 (0..15) → cols slot_oct*8 .. +7 of the 128-col tile
//
// Grid: [ceil(M/32), ceil(N/128)]. Block: [256].
// LDS: B/W rows use a 9-dword stride to avoid gfx10 bank aliasing.
#if defined(__gfx1030__)
#define HIPFIRE_GFX10_Q8_MMQ 1
#else
#define HIPFIRE_GFX10_Q8_MMQ 0
#endif
#define QK8_1 32
struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");
#if !HIPFIRE_GFX10_Q8_MMQ
extern "C" __global__ void gemm_q8_0_mmq_gfx1030(
    const unsigned char*, const void*, float*, int, int, int
) {}
#else
__launch_bounds__(256, 1)
extern "C" __global__ void gemm_q8_0_mmq_gfx1030(
    const unsigned char* __restrict__ A_q8, // [M, K/32 * 34]
    const block_q8_1_mmq* __restrict__ Xq,  // [K/128, N]
    float*               __restrict__ Y,    // [N, M]
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int row_start = (int)blockIdx.x * 32;
    const int col_start = (int)blockIdx.y * 128;
    if (row_start >= M || col_start >= N) return;
    const int row_local = tid & 15;
    const int slot_oct = tid >> 4; // 0..15 → 8-col group
    const int blocks_per_row = K / 32;
    const int out_row0 = row_start + row_local;
    const int out_row1 = row_start + row_local + 16;
    __shared__ int   lds_b[128 * 9];  // 128 cols × 8 dwords, padded to 9 banks
    __shared__ float lds_dx[128];
    __shared__ int   lds_w[32 * 9];   // 32 rows × 8 dwords, padded to 9 banks
    __shared__ float lds_ws[32];
    float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int blocks128 = K / 128;
    for (int kb = 0; kb < blocks128; kb++) {
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            const int bi = kb * 4 + sb;
            // ── Stage W: lanes 0..31 each extract one row's 32 int8 qs ────
            if (tid < 32) {
                const int row = row_start + tid;
                const int safe_row = (row < M) ? row : (M - 1);
                const unsigned char* bp =
                    A_q8 + ((size_t)safe_row * (size_t)blocks_per_row + (size_t)bi) * 34u;
                const unsigned char* qs = bp + 2;
                const uintptr_t base = (uintptr_t)qs;
                const uint32_t* dw = (const uint32_t*)(base & ~(uintptr_t)3u);
                const uint32_t sh = (uint32_t)(base & 3u); // byte offset 0..3
                uint32_t d0 = dw[0], d1 = dw[1], d2 = dw[2], d3 = dw[3];
                uint32_t d4 = dw[4], d5 = dw[5], d6 = dw[6], d7 = dw[7];
                // d8 holds qs[30..31] in its low half when sh==2 and is unused
                // when sh==0. A full dword load would read 2-4 B past the 34-B
                // block (past the buffer on the final row); use a 2-B load.
                uint32_t d8 = 0;
                if (sh == 2) {
                    d8 = (uint32_t)*(const uint16_t*)((const unsigned char*)dw + 32);
                }
                int* a = &lds_w[tid * 9];
                if (sh == 0) {
                    a[0] = (int)d0; a[1] = (int)d1; a[2] = (int)d2; a[3] = (int)d3;
                    a[4] = (int)d4; a[5] = (int)d5; a[6] = (int)d6; a[7] = (int)d7;
                } else {
                    // bytes [sh..sh+4) of each {hi,lo} pair
                    const uint32_t sel = 0x3210u + 0x1111u * sh;
                    a[0] = (int)__byte_perm(d0, d1, sel);
                    a[1] = (int)__byte_perm(d1, d2, sel);
                    a[2] = (int)__byte_perm(d2, d3, sel);
                    a[3] = (int)__byte_perm(d3, d4, sel);
                    a[4] = (int)__byte_perm(d4, d5, sel);
                    a[5] = (int)__byte_perm(d5, d6, sel);
                    a[6] = (int)__byte_perm(d6, d7, sel);
                    a[7] = (int)__byte_perm(d7, d8, sel);
                }
                lds_ws[tid] = (float)(*((const _Float16*)bp));
            }
            // ── Stage B: 128 cols × 32 int8 + 128 scales with 256 lanes ──
            {
                const int c = tid & 127;
                const int quat = tid >> 7; // 0 or 1
                const int col = col_start + c;
                int v0 = 0, v1 = 0, v2 = 0, v3 = 0;
                float dx = 0.f;
                if (col < N) {
                    const block_q8_1_mmq* xb =
                        Xq + (size_t)kb * (size_t)N + (size_t)col;
                    const int* xqs = (const int*)(xb->qs + sb * 32 + quat * 16);
                    v0 = xqs[0]; v1 = xqs[1]; v2 = xqs[2]; v3 = xqs[3];
                    if (quat == 0) {
                        float2 ds = __half22float2(xb->ds4[sb]);
                        dx = ds.x;
                    }
                }
                int* dst = &lds_b[c * 9 + quat * 4];
                dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
                if (quat == 0) lds_dx[c] = dx;
            }
            __syncthreads();
            // ── Compute: 8 cols × 2 rows; W read is an LDS broadcast ─────
            const int* a0 = &lds_w[row_local * 9];
            const int* a1 = &lds_w[(row_local + 16) * 9];
            const float wsc0 = lds_ws[row_local];
            const float wsc1 = lds_ws[row_local + 16];
            #pragma unroll 1
            for (int s = 0; s < 8; s++) {
                const int c = slot_oct * 8 + s;
                const int b0 = lds_b[c * 9 + 0];
                const int b1 = lds_b[c * 9 + 1];
                const int b2 = lds_b[c * 9 + 2];
                const int b3 = lds_b[c * 9 + 3];
                const int b4 = lds_b[c * 9 + 4];
                const int b5 = lds_b[c * 9 + 5];
                const int b6 = lds_b[c * 9 + 6];
                const int b7 = lds_b[c * 9 + 7];
                const float d_x = lds_dx[c];
                int sumi = 0;
                sumi = HIPFIRE_SDOT4_CALL(a0[0], b0, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a0[1], b1, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a0[2], b2, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a0[3], b3, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a0[4], b4, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a0[5], b5, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a0[6], b6, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a0[7], b7, sumi, false);
                acc0[s] += wsc0 * d_x * (float)sumi;
                sumi = 0;
                sumi = HIPFIRE_SDOT4_CALL(a1[0], b0, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a1[1], b1, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a1[2], b2, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a1[3], b3, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a1[4], b4, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a1[5], b5, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a1[6], b6, sumi, false);
                sumi = HIPFIRE_SDOT4_CALL(a1[7], b7, sumi, false);
                acc1[s] += wsc1 * d_x * (float)sumi;
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int s = 0; s < 8; s++) {
        const int out_col = col_start + slot_oct * 8 + s;
        if (out_col < N) {
            if (out_row0 < M)
                Y[(size_t)out_col * (size_t)M + (size_t)out_row0] = acc0[s];
            if (out_row1 < M)
                Y[(size_t)out_col * (size_t)M + (size_t)out_row1] = acc1[s];
        }
    }
}
#endif
