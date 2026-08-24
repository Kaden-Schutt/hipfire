// MagnumQuant 4-bit GEMV with per-block 16-entry fp16 Lloyd-Max codebook.
//
// Block layout (160 B/group, +17.6% bandwidth over uniform MQ4's 136 B):
//   bytes [0..32)   : 16 × fp16 codebook entries (sorted ascending at quant-time)
//   bytes [32..160) : 128 bytes packed 4-bit indices (low nibble = idx[2i],
//                     high nibble = idx[2i+1]), same nibble layout as uniform MQ4.
//
// Per-thread work: 8 weights / thread × 32 threads = 256 weights / block.
// Reconstruction is `cb[q]` via LDS-indexed lookup (branchless).
// This is the slow generic chip variant; LDS-codebook fast path mirrors
// gemv_mq3g256_lloyd.gfx1100.hip.
//
// Branchless codebook lookup (gfx1201 lesson, 2026-08):
// The old 15-deep `?:` chain (CB_LOOKUP4) does NOT lower to cmov/jump-table on
// gfx1201 — hipcc emits divergent control flow (measured 114 s_cbranch_execz +
// 105 v_cmpx_lt_i32, ~133 us at m=4096 k=5120). Stage each group's 16 fp16
// entries into a 16-slot __shared__ float table and index with ds_load_b32.
// Codebook is PER-GROUP data (unlike MQ4-GL's tensor-global CB), so staging
// sits inside the K loop. Block is one wave32; one __syncthreads after the
// cooperative write is required for cross-lane LDS visibility before indexed
// reads (same pattern as the gfx1100 sibling's TAIL_LOAD_AND_DOT). Measured
// ~48 us after the fix; double-barrier and no-barrier variants were not faster.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256_lloyd(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 160;
    const int boff = tid * 4;

    // Fixed 16-slot LDS table. Block is [32,1,1] = one wave; 64 B is trivial.
    // Prefer fixed over extern __shared__ so the launch wrapper stays sharedMem=0.
    __shared__ float cbt[16];

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 160;

        // Stage this group's 16 fp16 codebook entries into LDS (f32).
        // Lanes 0..15 each write one entry; all 32 lanes index afterward.
        if (tid < 16) {
            cbt[tid] = __half2float(((const __half*)gptr)[tid]);
        }
        // Cross-lane LDS RAW: writers are a subset of the wave; readers are all
        // lanes with data-dependent addresses. Wave lockstep does NOT make the
        // ds_store visible to other lanes — need a barrier (or at least an
        // lgkmcnt wait + wave sync). __syncthreads is the portable form; on a
        // single-wave block it is cheap (no multi-wave rendezvous). The next
        // iteration's write is also gated by this barrier at the top of that
        // iteration, so a trailing barrier is unnecessary.
        __syncthreads();

        const unsigned char* data = (const unsigned char*)(gptr + 32);

        // 8 × 4-bit nibble unpack (identical to uniform MQ4):
        //   data[boff + i] = idx[2i] (low) | idx[2i+1] (high)
        unsigned int b0 = (unsigned int)data[boff];
        unsigned int b1 = (unsigned int)data[boff + 1];
        unsigned int b2 = (unsigned int)data[boff + 2];
        unsigned int b3 = (unsigned int)data[boff + 3];

        unsigned int q0 =  b0       & 0xFu;
        unsigned int q1 = (b0 >> 4) & 0xFu;
        unsigned int q2 =  b1       & 0xFu;
        unsigned int q3 = (b1 >> 4) & 0xFu;
        unsigned int q4 =  b2       & 0xFu;
        unsigned int q5 = (b2 >> 4) & 0xFu;
        unsigned int q6 =  b3       & 0xFu;
        unsigned int q7 = (b3 >> 4) & 0xFu;

        int base_idx = g * 256 + tid * 8;

        // BRANCHLESS 16-entry lookup via LDS (ds_load_b32). Same reconstruction
        // as before: w = cb[q]. Semantics unchanged; only the select mechanism.
        acc += cbt[q0] * x[base_idx]
             + cbt[q1] * x[base_idx + 1]
             + cbt[q2] * x[base_idx + 2]
             + cbt[q3] * x[base_idx + 3]
             + cbt[q4] * x[base_idx + 4]
             + cbt[q5] * x[base_idx + 5]
             + cbt[q6] * x[base_idx + 6]
             + cbt[q7] * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
