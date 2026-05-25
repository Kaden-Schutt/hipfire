/// ParoQuant Givens rotation kernel (F32 activations).
///
/// Applies K sequential rounds of learned pairwise Givens rotations to
/// activations in-place, preceded by per-channel scaling. This is the
/// online rotation step in the ParoQuant inference pipeline — applied
/// before each quantized GEMV.
///
/// Port of z-lab/paroquant rotation.cu (ICLR 2026), adapted for F32
/// activation tensors (hipfire convention).
///
/// Grid:  (ceil(seq_len / CTA_M), groups_per_row)
/// Block: (GROUP_SIZE / 2) = (64) for GROUP_SIZE=128
/// Smem:  CTA_M * GROUP_SIZE * sizeof(float)
///
/// Each thread owns one pair within a 128-element group.
/// Per rotation round: apply 2x2 Givens rotation in shared memory,
/// __syncthreads between rounds.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <assert.h>

#define CTA_M 4       // rows per block (seq_len dimension)
#define GROUP_SIZE 128 // channels per quantization group
#define HALF_GROUP 64  // threads per group = GROUP_SIZE / 2

extern "C" __global__ void givens_rotate_f32(
    float* __restrict__ x,                   // [seq_len, hidden_dim] — modified in place
    const short* __restrict__ pairs,         // [krot, hidden_dim] — pair indices
    const half* __restrict__ theta,          // [krot, hidden_dim/2] — rotation angles
    const half* __restrict__ channel_scales, // [hidden_dim] — per-channel scaling
    const int seq_len,
    const int hidden_dim,
    const int krot
) {
    const int row_block = blockIdx.x;
    const int group_idx = blockIdx.y;
    const int tid = threadIdx.x;  // 0..63

    const int ch_base = group_idx * GROUP_SIZE;
    if (ch_base >= hidden_dim) return;

    // Shared memory: CTA_M rows x GROUP_SIZE channels (F32)
    extern __shared__ float smem[];  // [CTA_M][GROUP_SIZE]

    // Load channel scales (FP16 -> F32)
    const float s0 = __half2float(channel_scales[ch_base + tid * 2]);
    const float s1 = __half2float(channel_scales[ch_base + tid * 2 + 1]);

    // Load activations into smem + apply channel_scales
    for (int r = 0; r < CTA_M; r++) {
        const int row = row_block * CTA_M + r;
        if (row >= seq_len) {
            smem[r * GROUP_SIZE + tid * 2]     = 0.0f;
            smem[r * GROUP_SIZE + tid * 2 + 1] = 0.0f;
            continue;
        }
        const int gx = row * hidden_dim + ch_base;
        smem[r * GROUP_SIZE + tid * 2]     = x[gx + tid * 2]     * s0;
        smem[r * GROUP_SIZE + tid * 2 + 1] = x[gx + tid * 2 + 1] * s1;
    }
    __syncthreads();

    // Apply krot sequential Givens rotation rounds
    for (int rot = 0; rot < krot; rot++) {
        // Each thread handles one pair per rotation round.
        // pairs layout: [krot, hidden_dim] — consecutive I16 pairs: (i, j)
        // Thread tid reads pairs[rot, ch_base + 2*tid] and pairs[rot, ch_base + 2*tid+1]
        // as the two channel indices within the group (0..127). Independent pairs guaranteed.
        const int i = (int)pairs[rot * hidden_dim + ch_base + tid * 2];      // channel i (0..127)
        const int j = (int)pairs[rot * hidden_dim + ch_base + tid * 2 + 1];  // channel j (0..127)

        // Pair indices must be in-group: ParoQuant's offline rotation construction
        // guarantees i,j < GROUP_SIZE. Hint to clang for codegen, hard-fail debug builds
        // if calibration data is malformed.
#ifdef NDEBUG
        __builtin_assume(i >= 0 && i < GROUP_SIZE);
        __builtin_assume(j >= 0 && j < GROUP_SIZE);
#else
        assert(i >= 0 && i < GROUP_SIZE);
        assert(j >= 0 && j < GROUP_SIZE);
#endif

        // theta layout: [krot, hidden_dim/2]
        const float angle = __half2float(theta[rot * (hidden_dim / 2) + group_idx * HALF_GROUP + tid]);
        const float c = cosf(angle);
        const float s = sinf(angle);

        for (int r = 0; r < CTA_M; r++) {
            const int row = row_block * CTA_M + r;
            if (row >= seq_len) continue;

            float* row_smem = &smem[r * GROUP_SIZE];
            const float xi = row_smem[i];
            const float xj = row_smem[j];

            //  Forward Givens: x[i] = xi*c + xj*s, x[j] = xj*c - xi*s
            row_smem[i] = xi * c + xj * s;
            row_smem[j] = xj * c - xi * s;
        }
        __syncthreads();
    }

    // Store rotated activations back to global memory
    for (int r = 0; r < CTA_M; r++) {
        const int row = row_block * CTA_M + r;
        if (row >= seq_len) continue;
        const int gx = row * hidden_dim + ch_base;
        x[gx + tid * 2]     = smem[r * GROUP_SIZE + tid * 2];
        x[gx + tid * 2 + 1] = smem[r * GROUP_SIZE + tid * 2 + 1];
    }
}
