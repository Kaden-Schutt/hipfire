#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Dequant TQ2G128 packed weights (PrismML Q2_0-layout ternary) to row-major FP16.
// Input:  A_packed [M × groups_per_row × 34 bytes] (2-bit packed, FP16 scale at block start)
// Output: W_f16 [M × K] FP16 (row-major)
//
// Grid: [M, groups_per_row]. Block: [32].
// Each block dequants one 128-element group for one row.
// Thread t dequants 4 elements (tid*4..tid*4+3) — 4 × 2-bit codes = 1 byte/lane.
//
// code = (qs[j/4] >> ((j%4)*2)) & 0x3;  out[j] = (code - 1) * d
// Codes {0,1,2,3} -> {-d, 0, +d, +2d}.

extern "C" __launch_bounds__(32, 16)
__global__ void dequant_tq2g128_to_f16(
    const char* __restrict__ A_packed,
    _Float16* __restrict__ W_f16,
    int M, int K
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 128;

    const char* gp = A_packed + (long long)row * groups_per_row * 34 + group * 34;
    _Float16 d_h = *(const _Float16*)gp;
    float d = (float)d_h;

    unsigned char b = (unsigned char)gp[2 + tid];

    _Float16* out = W_f16 + (long long)row * K + group * 128 + tid * 4;
    out[0] = (_Float16)(((float)((int)((b >> 0) & 0x3) - 1)) * d);
    out[1] = (_Float16)(((float)((int)((b >> 2) & 0x3) - 1)) * d);
    out[2] = (_Float16)(((float)((int)((b >> 4) & 0x3) - 1)) * d);
    out[3] = (_Float16)(((float)((int)((b >> 6) & 0x3) - 1)) * d);
}
