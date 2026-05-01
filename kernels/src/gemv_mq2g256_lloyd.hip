// MagnumQuant 2-bit GEMV with per-block 4-entry fp16 Lloyd-Max codebook.
//
// Block layout (72 B/group, bandwidth-identical to uniform MQ2):
//   bytes [0..8)   : 4 × fp16 codebook entries (sorted ascending at quant-time)
//   bytes [8..72)  : 64 bytes packed 2-bit indices (4 per byte, little-endian within byte)
//
// Per-thread work: 8 weights / thread × 32 threads = 256 weights / block — same
// schedule as gemv_hfq2g256.hip. Reconstruction is `cb[q]` — direct lookup into
// the per-block fp16 codebook, NOT `scale*q + zero`. This rescues sub-9B
// quality where uniform 4-level codebooks collapse on heavy-tailed weight
// distributions.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;

        // Load 4 fp16 codebook entries → 4 floats (kept in registers).
        const __half* cb_h = reinterpret_cast<const __half*>(gptr);
        float cb0 = __half2float(cb_h[0]);
        float cb1 = __half2float(cb_h[1]);
        float cb2 = __half2float(cb_h[2]);
        float cb3 = __half2float(cb_h[3]);

        const unsigned char* data = (const unsigned char*)(gptr + 8);

        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 2;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];

        // Branchless 2-bit codebook lookup via inline switch on q ∈ {0,1,2,3}.
        // Compiler emits a register-resident jump table; no global memory hit.
        #define CB_LOOKUP(q) ((q) == 0 ? cb0 : ((q) == 1 ? cb1 : ((q) == 2 ? cb2 : cb3)))

        acc += CB_LOOKUP(b0 & 3)        * x[base_idx]
             + CB_LOOKUP((b0 >> 2) & 3) * x[base_idx + 1]
             + CB_LOOKUP((b0 >> 4) & 3) * x[base_idx + 2]
             + CB_LOOKUP(b0 >> 6)       * x[base_idx + 3]
             + CB_LOOKUP(b1 & 3)        * x[base_idx + 4]
             + CB_LOOKUP((b1 >> 2) & 3) * x[base_idx + 5]
             + CB_LOOKUP((b1 >> 4) & 3) * x[base_idx + 6]
             + CB_LOOKUP(b1 >> 6)       * x[base_idx + 7];

        #undef CB_LOOKUP
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
