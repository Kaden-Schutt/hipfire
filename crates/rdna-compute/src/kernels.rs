//! Built-in HIP kernel sources for inference operations.

/// GEMV F32: y = alpha * A * x + beta * y
/// Uses shared memory reduction across wavefronts.
pub const GEMV_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_f32(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    float alpha, float beta
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += A[row * K + k] * x[k];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = alpha * sdata[0] + beta * y[row];
    }
}

extern "C" __global__ void gemv_f16(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    float alpha, float beta
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += (float)A[row * K + k] * (float)x[k];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = alpha * sdata[0] + beta * y[row];
    }
}
"#;

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
pub const GEMV_Q4K_SRC: &str = r#"
#include <hip/hip_runtime.h>

__device__ __forceinline__ float half_to_float(unsigned short h) {
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp  = (h >> 10) & 0x1F;
    unsigned int frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) return __int_as_float(sign << 31);
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
        exp = 127 - 15 + 1 + exp;
        return __int_as_float((sign << 31) | (exp << 23) | (frac << 13));
    }
    if (exp == 31) return __int_as_float((sign << 31) | (0xFF << 23) | (frac ? (frac << 13 | 1) : 0));
    return __int_as_float((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13));
}

// Q4_K GEMV v6: 32 threads (single warp), 8 elements per thread per Q4K block.
// No shared memory needed — warp shuffle only. Higher occupancy.
// launch_bounds: 32 threads, target 20 blocks/CU (10 waves/SIMD) for max occupancy.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4k(
    const unsigned char* __restrict__ A_q4k,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A_q4k + (size_t)row * blocks_per_row * 144;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 144;
        float d = (float)*((const _Float16*)block);
        float dmin = (float)*((const _Float16*)(block + 2));
        const unsigned char* sc = block + 4;
        const float* xb = x + bi * 256;

        // Group 0: elements 0-63
        {
            unsigned char qbyte = block[16 + tid];
            float s0 = d * (float)(sc[0] & 63);
            float m0 = dmin * (float)(sc[4] & 63);
            sum += (s0 * (float)(qbyte & 0xF) - m0) * xb[tid];

            float s1 = d * (float)(sc[1] & 63);
            float m1 = dmin * (float)(sc[5] & 63);
            sum += (s1 * (float)(qbyte >> 4) - m1) * xb[32 + tid];
        }

        // Group 1: elements 64-127
        {
            unsigned char qbyte = block[48 + tid];
            float s2 = d * (float)(sc[2] & 63);
            float m2 = dmin * (float)(sc[6] & 63);
            sum += (s2 * (float)(qbyte & 0xF) - m2) * xb[64 + tid];

            float s3 = d * (float)(sc[3] & 63);
            float m3 = dmin * (float)(sc[7] & 63);
            sum += (s3 * (float)(qbyte >> 4) - m3) * xb[96 + tid];
        }

        // Group 2: elements 128-191 (different scale encoding)
        {
            unsigned char qbyte = block[80 + tid];
            float s4 = d * (float)((sc[8] & 0xF) | ((sc[0] >> 6) << 4));
            float m4 = dmin * (float)((sc[8] >> 4) | ((sc[4] >> 6) << 4));
            sum += (s4 * (float)(qbyte & 0xF) - m4) * xb[128 + tid];

            float s5 = d * (float)((sc[9] & 0xF) | ((sc[1] >> 6) << 4));
            float m5 = dmin * (float)((sc[9] >> 4) | ((sc[5] >> 6) << 4));
            sum += (s5 * (float)(qbyte >> 4) - m5) * xb[160 + tid];
        }

        // Group 3: elements 192-255
        {
            unsigned char qbyte = block[112 + tid];
            float s6 = d * (float)((sc[10] & 0xF) | ((sc[2] >> 6) << 4));
            float m6 = dmin * (float)((sc[10] >> 4) | ((sc[6] >> 6) << 4));
            sum += (s6 * (float)(qbyte & 0xF) - m6) * xb[192 + tid];

            float s7 = d * (float)((sc[11] & 0xF) | ((sc[3] >> 6) << 4));
            float m7 = dmin * (float)((sc[11] >> 4) | ((sc[7] >> 6) << 4));
            sum += (s7 * (float)(qbyte >> 4) - m7) * xb[224 + tid];
        }
    }

    // Warp shuffle reduction (5 steps, no shared memory)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
"#;

/// HFQ4-G128: flat 4-bit with 128-weight groups.
/// Block: [f32 scale][f32 zero][64B nibbles] = 72 bytes per 128 weights.
/// Minimal metadata → minimal VGPRs. Hypothesis: ≤32 VGPRs → max occupancy.
pub const GEMV_HFQ4G128_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq4g128(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        // 128 weights / 32 threads = 4 weights per thread
        int base_idx = g * 128 + tid * 4;
        int byte_off = tid * 2;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];

        float v0 = scale * (float)(b0 & 0xF) + zero;
        float v1 = scale * (float)(b0 >> 4)  + zero;
        float v2 = scale * (float)(b1 & 0xF) + zero;
        float v3 = scale * (float)(b1 >> 4)  + zero;

        acc += v0 * x[base_idx]     + v1 * x[base_idx + 1]
             + v2 * x[base_idx + 2] + v3 * x[base_idx + 3];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
"#;

/// HFQ4-G128 batched GEMM: same tiled approach as G256 but 72 bytes/group, 4 weights/thread.
pub const GEMM_HFQ4G128_SRC: &str = r#"
#include <hip/hip_runtime.h>

#define BATCH_TILE 8

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_hfq4g128(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    float acc[BATCH_TILE];
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        int base_k = g * 128 + tid * 4;
        int byte_off = tid * 2;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];

        float w0 = scale * (float)(b0 & 0xF) + zero;
        float w1 = scale * (float)(b0 >> 4)  + zero;
        float w2 = scale * (float)(b1 & 0xF) + zero;
        float w3 = scale * (float)(b1 >> 4)  + zero;

        for (int b = 0; b < local_bs; b++) {
            const float* xb = x + (batch_start + b) * K;
            acc[b] += w0 * xb[base_k]     + w1 * xb[base_k + 1]
                    + w2 * xb[base_k + 2] + w3 * xb[base_k + 3];
        }
    }

    for (int b = 0; b < local_bs; b++) {
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) y[(batch_start + b) * M + row] = val;
    }
}
"#;

/// HFQ2-G256: flat 2-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][64B data] = 72 bytes per 256 weights (0.28 B/w).
pub const GEMV_HFQ2G256_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq2g256(
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
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        // 256 weights / 32 threads = 8 weights per thread = 2 bytes (4 weights/byte at 2-bit)
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 2;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];

        acc += (scale * (float)(b0 & 3)       + zero) * x[base_idx]
             + (scale * (float)((b0 >> 2) & 3) + zero) * x[base_idx + 1]
             + (scale * (float)((b0 >> 4) & 3) + zero) * x[base_idx + 2]
             + (scale * (float)(b0 >> 6)       + zero) * x[base_idx + 3]
             + (scale * (float)(b1 & 3)        + zero) * x[base_idx + 4]
             + (scale * (float)((b1 >> 2) & 3) + zero) * x[base_idx + 5]
             + (scale * (float)((b1 >> 4) & 3) + zero) * x[base_idx + 6]
             + (scale * (float)(b1 >> 6)       + zero) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
"#;

/// HFQ8-G256: flat 8-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][256B data] = 264 bytes per 256 weights (1.03 B/w).
pub const GEMV_HFQ8G256_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq8g256(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 264;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 264;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        // 256 weights / 32 threads = 8 weights per thread = 8 bytes
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 8;

        acc += (scale * (float)data[byte_off]     + zero) * x[base_idx]
             + (scale * (float)data[byte_off + 1] + zero) * x[base_idx + 1]
             + (scale * (float)data[byte_off + 2] + zero) * x[base_idx + 2]
             + (scale * (float)data[byte_off + 3] + zero) * x[base_idx + 3]
             + (scale * (float)data[byte_off + 4] + zero) * x[base_idx + 4]
             + (scale * (float)data[byte_off + 5] + zero) * x[base_idx + 5]
             + (scale * (float)data[byte_off + 6] + zero) * x[base_idx + 6]
             + (scale * (float)data[byte_off + 7] + zero) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
"#;

/// HFQ6-G256: flat 6-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][192B data] = 200 bytes per 256 weights (0.78 B/w).
/// Packing: 4 weights per 3 bytes (24 bits = 4×6 bits).
pub const GEMV_HFQ6G256_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq6g256(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 200;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 200;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        // 256 weights / 32 threads = 8 weights per thread
        // 4 weights per 3 bytes → 8 weights = 6 bytes per thread
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 6;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];
        unsigned char b2 = data[byte_off + 2];
        unsigned char b3 = data[byte_off + 3];
        unsigned char b4 = data[byte_off + 4];
        unsigned char b5 = data[byte_off + 5];

        // 4 weights from first 3 bytes
        int q0 = b0 & 63;
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
        int q2 = (b1 >> 4) | ((b2 & 3) << 4);
        int q3 = b2 >> 2;
        // 4 weights from next 3 bytes
        int q4 = b3 & 63;
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
        int q6 = (b4 >> 4) | ((b5 & 3) << 4);
        int q7 = b5 >> 2;

        acc += (scale * (float)q0 + zero) * x[base_idx]
             + (scale * (float)q1 + zero) * x[base_idx + 1]
             + (scale * (float)q2 + zero) * x[base_idx + 2]
             + (scale * (float)q3 + zero) * x[base_idx + 3]
             + (scale * (float)q4 + zero) * x[base_idx + 4]
             + (scale * (float)q5 + zero) * x[base_idx + 5]
             + (scale * (float)q6 + zero) * x[base_idx + 6]
             + (scale * (float)q7 + zero) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
"#;

/// HFQ3-G256: flat 3-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][96B data] = 104 bytes per 256 weights (0.41 B/w).
/// Packing: 8 weights per 3 bytes (24 bits = 8×3 bits).
pub const GEMV_HFQ3G256_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq3g256(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 104;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 104;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        // 256 weights / 32 threads = 8 weights per thread
        // 8 weights × 3 bits = 24 bits = 3 bytes per thread
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 3;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];
        unsigned char b2 = data[byte_off + 2];

        // Unpack 8 × 3-bit from 24 bits
        int q0 = b0 & 7;
        int q1 = (b0 >> 3) & 7;
        int q2 = ((b0 >> 6) | (b1 << 2)) & 7;
        int q3 = (b1 >> 1) & 7;
        int q4 = (b1 >> 4) & 7;
        int q5 = ((b1 >> 7) | (b2 << 1)) & 7;
        int q6 = (b2 >> 2) & 7;
        int q7 = (b2 >> 5) & 7;

        acc += (scale * (float)q0 + zero) * x[base_idx]
             + (scale * (float)q1 + zero) * x[base_idx + 1]
             + (scale * (float)q2 + zero) * x[base_idx + 2]
             + (scale * (float)q3 + zero) * x[base_idx + 3]
             + (scale * (float)q4 + zero) * x[base_idx + 4]
             + (scale * (float)q5 + zero) * x[base_idx + 5]
             + (scale * (float)q6 + zero) * x[base_idx + 6]
             + (scale * (float)q7 + zero) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
"#;

/// HFQ4-G512: flat 4-bit with 512-weight groups.
/// Block: [f32 scale][f32 zero][256B nibbles] = 264 bytes per 512 weights (0.516 B/w).
/// 264B ≈ 1 PCIe TLP, 2 L2 cache lines.
pub const GEMV_HFQ4G512_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq4g512(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 512;
    const int row_bytes = groups_per_row * 264;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 264;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        // 512 weights / 32 threads = 16 weights per thread = 8 bytes
        int base_idx = g * 512 + tid * 16;
        int byte_off = tid * 8;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];
        unsigned char b2 = nibbles[byte_off + 2];
        unsigned char b3 = nibbles[byte_off + 3];
        unsigned char b4 = nibbles[byte_off + 4];
        unsigned char b5 = nibbles[byte_off + 5];
        unsigned char b6 = nibbles[byte_off + 6];
        unsigned char b7 = nibbles[byte_off + 7];

        acc += (scale * (float)(b0 & 0xF) + zero) * x[base_idx]
             + (scale * (float)(b0 >> 4)  + zero) * x[base_idx + 1]
             + (scale * (float)(b1 & 0xF) + zero) * x[base_idx + 2]
             + (scale * (float)(b1 >> 4)  + zero) * x[base_idx + 3]
             + (scale * (float)(b2 & 0xF) + zero) * x[base_idx + 4]
             + (scale * (float)(b2 >> 4)  + zero) * x[base_idx + 5]
             + (scale * (float)(b3 & 0xF) + zero) * x[base_idx + 6]
             + (scale * (float)(b3 >> 4)  + zero) * x[base_idx + 7]
             + (scale * (float)(b4 & 0xF) + zero) * x[base_idx + 8]
             + (scale * (float)(b4 >> 4)  + zero) * x[base_idx + 9]
             + (scale * (float)(b5 & 0xF) + zero) * x[base_idx + 10]
             + (scale * (float)(b5 >> 4)  + zero) * x[base_idx + 11]
             + (scale * (float)(b6 & 0xF) + zero) * x[base_idx + 12]
             + (scale * (float)(b6 >> 4)  + zero) * x[base_idx + 13]
             + (scale * (float)(b7 & 0xF) + zero) * x[base_idx + 14]
             + (scale * (float)(b7 >> 4)  + zero) * x[base_idx + 15];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
"#;

/// HFQ4-G1024: flat 4-bit with 1024-weight groups.
/// Block: [f32 scale][f32 zero][512B nibbles] = 520 bytes per 1024 weights (0.508 B/w).
pub const GEMV_HFQ4G1024_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq4g1024(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 1024;
    const int row_bytes = groups_per_row * 520;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 520;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        // 1024 weights / 32 threads = 32 weights per thread = 16 bytes
        // Process in 2 halves to limit register pressure
        int base_idx = g * 1024 + tid * 32;
        int byte_off = tid * 16;

        for (int h = 0; h < 2; h++) {
            unsigned char b0 = nibbles[byte_off + h * 8];
            unsigned char b1 = nibbles[byte_off + h * 8 + 1];
            unsigned char b2 = nibbles[byte_off + h * 8 + 2];
            unsigned char b3 = nibbles[byte_off + h * 8 + 3];
            unsigned char b4 = nibbles[byte_off + h * 8 + 4];
            unsigned char b5 = nibbles[byte_off + h * 8 + 5];
            unsigned char b6 = nibbles[byte_off + h * 8 + 6];
            unsigned char b7 = nibbles[byte_off + h * 8 + 7];
            int bi = base_idx + h * 16;

            acc += (scale * (float)(b0 & 0xF) + zero) * x[bi]
                 + (scale * (float)(b0 >> 4)  + zero) * x[bi + 1]
                 + (scale * (float)(b1 & 0xF) + zero) * x[bi + 2]
                 + (scale * (float)(b1 >> 4)  + zero) * x[bi + 3]
                 + (scale * (float)(b2 & 0xF) + zero) * x[bi + 4]
                 + (scale * (float)(b2 >> 4)  + zero) * x[bi + 5]
                 + (scale * (float)(b3 & 0xF) + zero) * x[bi + 6]
                 + (scale * (float)(b3 >> 4)  + zero) * x[bi + 7]
                 + (scale * (float)(b4 & 0xF) + zero) * x[bi + 8]
                 + (scale * (float)(b4 >> 4)  + zero) * x[bi + 9]
                 + (scale * (float)(b5 & 0xF) + zero) * x[bi + 10]
                 + (scale * (float)(b5 >> 4)  + zero) * x[bi + 11]
                 + (scale * (float)(b6 & 0xF) + zero) * x[bi + 12]
                 + (scale * (float)(b6 >> 4)  + zero) * x[bi + 13]
                 + (scale * (float)(b7 & 0xF) + zero) * x[bi + 14]
                 + (scale * (float)(b7 >> 4)  + zero) * x[bi + 15];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
"#;

/// HFQ4-G256: flat 4-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 weights.
/// Same coalesced width as Q4_K, 14 VGPRs instead of 39.
pub const GEMV_HFQ4G256_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq4g256(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        // 256 weights / 32 threads = 8 weights per thread = 4 bytes
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 4;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];
        unsigned char b2 = nibbles[byte_off + 2];
        unsigned char b3 = nibbles[byte_off + 3];

        acc += (scale * (float)(b0 & 0xF) + zero) * x[base_idx]
             + (scale * (float)(b0 >> 4)  + zero) * x[base_idx + 1]
             + (scale * (float)(b1 & 0xF) + zero) * x[base_idx + 2]
             + (scale * (float)(b1 >> 4)  + zero) * x[base_idx + 3]
             + (scale * (float)(b2 & 0xF) + zero) * x[base_idx + 4]
             + (scale * (float)(b2 >> 4)  + zero) * x[base_idx + 5]
             + (scale * (float)(b3 & 0xF) + zero) * x[base_idx + 6]
             + (scale * (float)(b3 >> 4)  + zero) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
"#;

/// HFQ4-G256 wide GEMV: 2 rows per block (64 threads = 2 warps).
/// Each warp processes one row independently. Halves grid size.
pub const GEMV_HFQ4G256_WIDE_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_hfq4g256_wide(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        int base_idx = g * 256 + lane * 8;
        int byte_off = lane * 4;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];
        unsigned char b2 = nibbles[byte_off + 2];
        unsigned char b3 = nibbles[byte_off + 3];

        acc += (scale * (float)(b0 & 0xF) + zero) * x[base_idx]
             + (scale * (float)(b0 >> 4)  + zero) * x[base_idx + 1]
             + (scale * (float)(b1 & 0xF) + zero) * x[base_idx + 2]
             + (scale * (float)(b1 >> 4)  + zero) * x[base_idx + 3]
             + (scale * (float)(b2 & 0xF) + zero) * x[base_idx + 4]
             + (scale * (float)(b2 >> 4)  + zero) * x[base_idx + 5]
             + (scale * (float)(b3 & 0xF) + zero) * x[base_idx + 6]
             + (scale * (float)(b3 >> 4)  + zero) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) y[row] = acc;
}
"#;

/// HFQ4-G256 batched GEMM: y[batch][row] = sum_k(A[row][k] * x[batch][k])
/// Loads weight data ONCE per group, multiplies against BATCH_TILE input vectors.
/// Grid: [M, ceil(batch_size/BATCH_TILE), 1]. Each block handles one row × BATCH_TILE batch elements.
/// x layout: [batch_size × K] row-major. y layout: [batch_size × M] row-major.
/// BATCH_TILE=8 keeps register pressure at ~26 VGPRs for good occupancy on RDNA.
pub const GEMM_HFQ4G256_SRC: &str = r#"
#include <hip/hip_runtime.h>

#define BATCH_TILE 8

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_hfq4g256(
    const char* __restrict__ A,
    const float* __restrict__ x,    // [batch_size, K]
    float* __restrict__ y,          // [batch_size, M]
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;

    // How many batch elements this block handles
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    float acc[BATCH_TILE];
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        int base_k = g * 256 + tid * 8;
        int byte_off = tid * 4;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];
        unsigned char b2 = nibbles[byte_off + 2];
        unsigned char b3 = nibbles[byte_off + 3];

        float w0 = scale * (float)(b0 & 0xF) + zero;
        float w1 = scale * (float)(b0 >> 4)  + zero;
        float w2 = scale * (float)(b1 & 0xF) + zero;
        float w3 = scale * (float)(b1 >> 4)  + zero;
        float w4 = scale * (float)(b2 & 0xF) + zero;
        float w5 = scale * (float)(b2 >> 4)  + zero;
        float w6 = scale * (float)(b3 & 0xF) + zero;
        float w7 = scale * (float)(b3 >> 4)  + zero;

        for (int b = 0; b < local_bs; b++) {
            const float* xb = x + (batch_start + b) * K;
            acc[b] += w0 * xb[base_k]     + w1 * xb[base_k + 1]
                    + w2 * xb[base_k + 2] + w3 * xb[base_k + 3]
                    + w4 * xb[base_k + 4] + w5 * xb[base_k + 5]
                    + w6 * xb[base_k + 6] + w7 * xb[base_k + 7];
        }
    }

    for (int b = 0; b < local_bs; b++) {
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) y[(batch_start + b) * M + row] = val;
    }
}
"#;

/// Fused QKV Q4_K: three GEMVs in one kernel launch.
/// Grid = (q_m + k_m + v_m) blocks. Each block determines which matrix by blockIdx range.
/// All three projections read the same input x (cached). Saves 2 kernel launches per layer.
pub const FUSED_QKV_Q4K_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void fused_qkv_q4k(
    const unsigned char* __restrict__ A_q,
    const unsigned char* __restrict__ A_k,
    const unsigned char* __restrict__ A_v,
    const float* __restrict__ x,
    float* __restrict__ y_q,
    float* __restrict__ y_k,
    float* __restrict__ y_v,
    int q_m, int k_m, int v_m, int K
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;

    // Determine which matrix this block processes
    const unsigned char* A;
    float* y;
    int local_row;
    if (gid < q_m) {
        A = A_q; y = y_q; local_row = gid;
    } else if (gid < q_m + k_m) {
        A = A_k; y = y_k; local_row = gid - q_m;
    } else {
        A = A_v; y = y_v; local_row = gid - q_m - k_m;
    }

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A + (size_t)local_row * blocks_per_row * 144;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 144;
        float d = (float)*((const _Float16*)block);
        float dmin = (float)*((const _Float16*)(block + 2));
        const unsigned char* sc = block + 4;
        const float* xb = x + bi * 256;

        {
            unsigned char qbyte = block[16 + tid];
            float s0 = d * (float)(sc[0] & 63);
            float m0 = dmin * (float)(sc[4] & 63);
            sum += (s0 * (float)(qbyte & 0xF) - m0) * xb[tid];
            float s1 = d * (float)(sc[1] & 63);
            float m1 = dmin * (float)(sc[5] & 63);
            sum += (s1 * (float)(qbyte >> 4) - m1) * xb[32 + tid];
        }
        {
            unsigned char qbyte = block[48 + tid];
            float s2 = d * (float)(sc[2] & 63);
            float m2 = dmin * (float)(sc[6] & 63);
            sum += (s2 * (float)(qbyte & 0xF) - m2) * xb[64 + tid];
            float s3 = d * (float)(sc[3] & 63);
            float m3 = dmin * (float)(sc[7] & 63);
            sum += (s3 * (float)(qbyte >> 4) - m3) * xb[96 + tid];
        }
        {
            unsigned char qbyte = block[80 + tid];
            float s4 = d * (float)((sc[8] & 0xF) | ((sc[0] >> 6) << 4));
            float m4 = dmin * (float)((sc[8] >> 4) | ((sc[4] >> 6) << 4));
            sum += (s4 * (float)(qbyte & 0xF) - m4) * xb[128 + tid];
            float s5 = d * (float)((sc[9] & 0xF) | ((sc[1] >> 6) << 4));
            float m5 = dmin * (float)((sc[9] >> 4) | ((sc[5] >> 6) << 4));
            sum += (s5 * (float)(qbyte >> 4) - m5) * xb[160 + tid];
        }
        {
            unsigned char qbyte = block[112 + tid];
            float s6 = d * (float)((sc[10] & 0xF) | ((sc[2] >> 6) << 4));
            float m6 = dmin * (float)((sc[10] >> 4) | ((sc[6] >> 6) << 4));
            sum += (s6 * (float)(qbyte & 0xF) - m6) * xb[192 + tid];
            float s7 = d * (float)((sc[11] & 0xF) | ((sc[3] >> 6) << 4));
            float m7 = dmin * (float)((sc[11] >> 4) | ((sc[7] >> 6) << 4));
            sum += (s7 * (float)(qbyte >> 4) - m7) * xb[224 + tid];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[local_row] = sum;
}
"#;

/// Fused Gate+Up Q4_K: two GEMVs in one kernel launch for FFN gate and up projections.
/// Grid = (gate_m + up_m) blocks. Saves 1 kernel launch per layer.
pub const FUSED_GATE_UP_Q4K_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void fused_gate_up_q4k(
    const unsigned char* __restrict__ A_gate,
    const unsigned char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;

    const unsigned char* A;
    float* y;
    int local_row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; local_row = gid;
    } else {
        A = A_up; y = y_up; local_row = gid - gate_m;
    }

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A + (size_t)local_row * blocks_per_row * 144;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 144;
        float d = (float)*((const _Float16*)block);
        float dmin = (float)*((const _Float16*)(block + 2));
        const unsigned char* sc = block + 4;
        const float* xb = x + bi * 256;

        {
            unsigned char qbyte = block[16 + tid];
            sum += (d * (float)(sc[0] & 63) * (float)(qbyte & 0xF) - dmin * (float)(sc[4] & 63)) * xb[tid];
            sum += (d * (float)(sc[1] & 63) * (float)(qbyte >> 4) - dmin * (float)(sc[5] & 63)) * xb[32 + tid];
        }
        {
            unsigned char qbyte = block[48 + tid];
            sum += (d * (float)(sc[2] & 63) * (float)(qbyte & 0xF) - dmin * (float)(sc[6] & 63)) * xb[64 + tid];
            sum += (d * (float)(sc[3] & 63) * (float)(qbyte >> 4) - dmin * (float)(sc[7] & 63)) * xb[96 + tid];
        }
        {
            unsigned char qbyte = block[80 + tid];
            sum += (d * (float)((sc[8] & 0xF) | ((sc[0] >> 6) << 4)) * (float)(qbyte & 0xF) - dmin * (float)((sc[8] >> 4) | ((sc[4] >> 6) << 4))) * xb[128 + tid];
            sum += (d * (float)((sc[9] & 0xF) | ((sc[1] >> 6) << 4)) * (float)(qbyte >> 4) - dmin * (float)((sc[9] >> 4) | ((sc[5] >> 6) << 4))) * xb[160 + tid];
        }
        {
            unsigned char qbyte = block[112 + tid];
            sum += (d * (float)((sc[10] & 0xF) | ((sc[2] >> 6) << 4)) * (float)(qbyte & 0xF) - dmin * (float)((sc[10] >> 4) | ((sc[6] >> 6) << 4))) * xb[192 + tid];
            sum += (d * (float)((sc[11] & 0xF) | ((sc[3] >> 6) << 4)) * (float)(qbyte >> 4) - dmin * (float)((sc[11] >> 4) | ((sc[7] >> 6) << 4))) * xb[224 + tid];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[local_row] = sum;
}
"#;

/// GEMV Q8_0: matrix-vector multiply with on-the-fly Q8_0 dequantization.
/// Q8_0 block: 2 bytes f16 scale + 32 bytes int8 = 34 bytes per 32 elements.
/// v3: Processes 8 blocks (256 elements) per outer iteration to match Q4_K's loop count.
/// Byte loads → no nibble extraction → 16 VGPRs → F32-class occupancy.
/// Q8_0 GEMV wide: 256 threads with shared memory reduction for small matrices.
/// Each thread processes K/256 elements strided, then tree-reduce via shared memory.
/// Better for dim=1024 where 32-thread kernel underutilizes the GPU.
pub const GEMV_Q8_0_WIDE_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Multi-row: 2 warps per block, each warp processes one row independently.
// Grid = ceil(M/2). No shared memory or cross-warp sync.
// 8x unrolled inner loop for ILP.
extern "C" __global__ void gemv_q8_0_wide(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float sum = 0.0f;

    int bi = 0;
    for (; bi + 7 < blocks_per_row; bi += 8) {
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            const unsigned char* block = row_data + (bi + u) * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + lane];
            sum += d * (float)qval * x[(bi + u) * 32 + lane];
        }
    }
    for (; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + lane];
        sum += d * (float)qval * x[bi * 32 + lane];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) y[row] = sum;
}
"#;

pub const GEMV_Q8_0_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Q8_0 GEMV v3: 32 threads, 8 blocks/iteration (256 elements), warp shuffle.
// Unrolled 8x to amortize loop overhead. Each thread processes 8 elements per iteration.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float sum = 0.0f;

    // Process 8 Q8_0 blocks (256 elements) per outer iteration
    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;

        // Unrolled: 8 blocks, each: load scale (f16→f32), load byte, FMA
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }

    // Handle remaining blocks (if K is not multiple of 256)
    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (tid == 0) y[row] = sum;
}
"#;

/// GEMV Q6_K: matrix-vector multiply with on-the-fly Q6_K dequantization.
/// Q6_K block: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes per 256 elements.
pub const GEMV_Q6K_SRC: &str = r#"
#include <hip/hip_runtime.h>

__device__ __forceinline__ float half_to_float_q6(unsigned short h) {
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp  = (h >> 10) & 0x1F;
    unsigned int frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) return __int_as_float(sign << 31);
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
        exp = 127 - 15 + 1 + exp;
        return __int_as_float((sign << 31) | (exp << 23) | (frac << 13));
    }
    if (exp == 31) return __int_as_float((sign << 31) | (0xFF << 23) | (frac ? (frac << 13 | 1) : 0));
    return __int_as_float((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13));
}

extern "C" __global__ void gemv_q6k(
    const unsigned char* __restrict__ A_q6k,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    if (row >= M) return;

    const int blocks_per_row = K / 256;
    const int bytes_per_block = 210;
    const unsigned char* row_data = A_q6k + (size_t)row * blocks_per_row * bytes_per_block;

    float sum = 0.0f;

    for (int elem = threadIdx.x; elem < K; elem += blockDim.x) {
        int block_idx = elem / 256;
        int within_block = elem % 256;

        const unsigned char* block = row_data + block_idx * bytes_per_block;
        const unsigned char* ql = block;           // 128 bytes
        const unsigned char* qh = block + 128;     // 64 bytes
        const signed char* sc = (const signed char*)(block + 192); // 16 bytes
        unsigned short d_bits = block[208] | (block[209] << 8);
        float d = half_to_float_q6(d_bits);

        // Which group (0 or 1) of 128 elements
        int group = within_block / 128;
        int in_group = within_block % 128;

        // Advance pointers for group
        const unsigned char* g_ql = ql + group * 64;
        const unsigned char* g_qh = qh + group * 32;
        const signed char* g_sc = sc + group * 8;

        // Which of 4 sub-positions within group
        int sub = in_group / 32;  // 0,1,2,3
        int l = in_group % 32;
        int is = l / 16;  // scale sub-index

        int q;
        switch (sub) {
            case 0: q = ((g_ql[l] & 0xF) | (((g_qh[l] >> 0) & 3) << 4)) - 32; break;
            case 1: q = ((g_ql[l + 32] & 0xF) | (((g_qh[l] >> 2) & 3) << 4)) - 32; break;
            case 2: q = ((g_ql[l] >> 4) | (((g_qh[l] >> 4) & 3) << 4)) - 32; break;
            case 3: q = ((g_ql[l + 32] >> 4) | (((g_qh[l] >> 6) & 3) << 4)) - 32; break;
        }

        float scale = (float)g_sc[is + sub * 2];
        float w = d * scale * (float)q;
        sum += w * x[elem];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = sdata[0];
    }
}
"#;

/// RMSNorm: y[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
pub const RMSNORM_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n, float eps
) {
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[blockIdx.x * n + i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int idx = blockIdx.x * n + i;
        out[idx] = x[idx] * weight[i] * rms;
    }
}
"#;

/// Element-wise add
pub const ADD_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;

/// Element-wise in-place add: a[i] += b[i]
pub const ADD_INPLACE_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void add_inplace_f32(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
"#;

/// Element-wise multiply
pub const MUL_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}
"#;

/// SiLU (Sigmoid Linear Unit): silu(x) = x * sigmoid(x)
pub const SILU_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void silu_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}
"#;

/// Fused SiLU(gate) * up: out[i] = silu(gate[i]) * up[i]
/// Saves one kernel launch + one intermediate buffer.
pub const SILU_MUL_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void silu_mul_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        out[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}
"#;

/// Softmax over last dimension (one block per row)
pub const SOFTMAX_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void softmax_f32(
    float* __restrict__ x,
    int n
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    float* row_data = x + row * n;

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        max_val = fmaxf(max_val, row_data[i]);
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_data[i] = expf(row_data[i] - max_val);
        sum += row_data[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_data[i] /= sum;
    }
}
"#;

/// RoPE (Rotary Positional Embedding)
pub const ROPE_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void rope_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    float freq_base
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    if (i >= half) return;

    int pos = pos_buf[0];
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float val = (float)pos * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    for (int h = 0; h < n_heads_q; h++) {
        int base = h * head_dim;
        float q0 = q[base + i];
        float q1 = q[base + i + half];
        q[base + i]        = q0 * cos_val - q1 * sin_val;
        q[base + i + half] = q0 * sin_val + q1 * cos_val;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = h * head_dim;
        float k0 = k[base + i];
        float k1 = k[base + i + half];
        k[base + i]        = k0 * cos_val - k1 * sin_val;
        k[base + i + half] = k0 * sin_val + k1 * cos_val;
    }
}
"#;

/// Batched RoPE: apply RoPE to [batch_size] positions at once.
/// q: [batch_size × n_heads_q × head_dim], k: [batch_size × n_heads_k × head_dim]
/// positions: [batch_size] int array of position indices.
/// Grid: [half, batch_size, 1]. Each thread handles one (position, freq_index) pair.
pub const ROPE_BATCHED_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void rope_batched_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,  // [batch_size]
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    float freq_base,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // freq index
    int b = blockIdx.y;  // batch index
    int half = head_dim / 2;
    if (i >= half || b >= batch_size) return;

    int pos = positions[b];
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float val = (float)pos * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    int q_stride = n_heads_q * head_dim;
    int k_stride = n_heads_k * head_dim;

    for (int h = 0; h < n_heads_q; h++) {
        int base = b * q_stride + h * head_dim;
        float q0 = q[base + i];
        float q1 = q[base + i + half];
        q[base + i]        = q0 * cos_val - q1 * sin_val;
        q[base + i + half] = q0 * sin_val + q1 * cos_val;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = b * k_stride + h * head_dim;
        float k0 = k[base + i];
        float k1 = k[base + i + half];
        k[base + i]        = k0 * cos_val - k1 * sin_val;
        k[base + i + half] = k0 * sin_val + k1 * cos_val;
    }
}
"#;

/// Single-head causal attention on GPU.
/// One thread block per query head. Handles GQA (kv_group heads share same KV).
/// q: [n_heads * head_dim], k_cache: [seq_len * n_kv_heads * head_dim],
/// v_cache: same layout, out: [n_heads * head_dim].
pub const ATTENTION_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void attention_f32(
    const float* __restrict__ q,          // [n_heads * head_dim]
    const float* __restrict__ k_cache,    // [max_seq * n_kv_heads * head_dim]
    const float* __restrict__ v_cache,    // [max_seq * n_kv_heads * head_dim]
    float* __restrict__ out,              // [n_heads * head_dim]
    const int* __restrict__ pos_buf,      // GPU buffer: seq_len = pos_buf[0] + 1
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,        // max sequence length (stride for cache indexing)
    float scale         // 1/sqrt(head_dim)
) {
    int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];

    const int h = blockIdx.x;  // query head index
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* q_head = q + h * head_dim;

    // Phase 1: compute attention scores (Q @ K^T) for all positions
    // Each thread handles a subset of positions
    float* scores = sdata;  // [seq_len] — reused for softmax

    // Find max for numerical stability
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        const float* k_t = k_cache + t * n_kv_heads * head_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_t[d];
        }
        float s = dot * scale;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    // Reduce max across threads
    // Workspace starts at sdata[seq_len] to avoid overlapping scores[0..seq_len-1]
    float* workspace = sdata + seq_len;
    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    float max_val = workspace[0];
    __syncthreads();

    // Exp and sum
    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }

    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    float sum_val = workspace[0];
    __syncthreads();

    // Normalize
    for (int t = tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // Phase 2: weighted sum of values
    // Each thread computes a subset of output dimensions
    float* out_head = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            val += scores[t] * v_cache[t * n_kv_heads * head_dim + kv_h * head_dim + d];
        }
        out_head[d] = val;
    }
}
"#;

/// Flash-Decoding attention: split KV scan across multiple blocks per head.
/// Phase 1: each block processes a chunk of KV positions, writes partial (max, sum, output).
/// Phase 2: reduction across chunks using online softmax correction.
/// Grid: [n_heads, n_chunks, 1]. Each block handles one (head, chunk) pair.
/// Partial results stored in partials buffer: [n_heads × n_chunks × (1 + 1 + head_dim)] floats.
pub const ATTENTION_FLASH_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void attention_flash_partial(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ partials,       // [n_heads, n_chunks, 2 + head_dim]
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale,
    int chunk_size
) {
    const int h = blockIdx.x;
    const int chunk_id = blockIdx.y;
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int chunk_start = chunk_id * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    if (chunk_end > seq_len) chunk_end = seq_len;
    if (chunk_start >= seq_len) return;
    const int chunk_len = chunk_end - chunk_start;

    const float* q_head = q + h * head_dim;

    // Compute scores for this chunk
    extern __shared__ float sdata[];
    float* scores = sdata;  // [chunk_size]

    float local_max = -1e30f;
    for (int t = tid; t < chunk_len; t += nthreads) {
        int pos = chunk_start + t;
        const float* k_t = k_cache + pos * n_kv_heads * head_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_t[d];
        }
        float s = dot * scale;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    // Reduce max
    float* ws = sdata + chunk_size;
    ws[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid + s]);
        __syncthreads();
    }
    float max_val = ws[0];
    __syncthreads();

    // Exp + sum
    float local_sum = 0.0f;
    for (int t = tid; t < chunk_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    ws[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) ws[tid] += ws[tid + s];
        __syncthreads();
    }
    float sum_val = ws[0];
    __syncthreads();

    // Weighted sum of values for this chunk
    int n_chunks = gridDim.y;
    int partial_stride = 2 + head_dim;  // [max, sum, output[head_dim]]
    float* partial = partials + (h * n_chunks + chunk_id) * partial_stride;

    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int t = 0; t < chunk_len; t++) {
            val += scores[t] * v_cache[(chunk_start + t) * n_kv_heads * head_dim + kv_h * head_dim + d];
        }
        partial[2 + d] = val;  // unnormalized weighted sum
    }
    if (tid == 0) {
        partial[0] = max_val;
        partial[1] = sum_val;
    }
}

// Phase 2: reduce partials across chunks using online softmax correction
extern "C" __global__ void attention_flash_reduce(
    const float* __restrict__ partials,
    float* __restrict__ out,
    int n_heads,
    int n_chunks,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    int partial_stride = 2 + head_dim;

    // Online softmax: combine chunks
    float global_max = -1e30f;
    float global_sum = 0.0f;

    // First pass: find global max
    for (int c = 0; c < n_chunks; c++) {
        const float* p = partials + (h * n_chunks + c) * partial_stride;
        float m = p[0];
        if (m > global_max) global_max = m;
    }

    // Second pass: compute corrected sum and weighted output
    float* out_head = out + h * head_dim;
    // Initialize output to zero
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_head[d] = 0.0f;
    }
    __syncthreads();

    for (int c = 0; c < n_chunks; c++) {
        const float* p = partials + (h * n_chunks + c) * partial_stride;
        float chunk_max = p[0];
        float chunk_sum = p[1];
        float correction = expf(chunk_max - global_max);
        float corrected_sum = chunk_sum * correction;
        global_sum += corrected_sum;

        for (int d = tid; d < head_dim; d += blockDim.x) {
            out_head[d] += p[2 + d] * correction;
        }
        __syncthreads();
    }

    // Final normalize
    float inv_sum = 1.0f / global_sum;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_head[d] *= inv_sum;
    }
}
"#;

/// Fused Gate+Up HFQ4-G256: two GEMVs in one launch (saves 1 launch per layer).
/// Grid: [gate_m + up_m, 1, 1]. Each block processes one row from gate or up weight.
pub const FUSED_GATE_UP_HFQ4G256_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void fused_gate_up_hfq4g256(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;

    const char* A;
    float* y;
    int local_row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; local_row = gid;
    } else {
        A = A_up; y = y_up; local_row = gid - gate_m;
    }

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)local_row * row_bytes;

    float acc = 0.0f;
    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 4;
        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];
        unsigned char b2 = nibbles[byte_off + 2];
        unsigned char b3 = nibbles[byte_off + 3];
        acc += (scale * (float)(b0 & 0xF) + zero) * x[base_idx]
             + (scale * (float)(b0 >> 4)  + zero) * x[base_idx + 1]
             + (scale * (float)(b1 & 0xF) + zero) * x[base_idx + 2]
             + (scale * (float)(b1 >> 4)  + zero) * x[base_idx + 3]
             + (scale * (float)(b2 & 0xF) + zero) * x[base_idx + 4]
             + (scale * (float)(b2 >> 4)  + zero) * x[base_idx + 5]
             + (scale * (float)(b3 & 0xF) + zero) * x[base_idx + 6]
             + (scale * (float)(b3 >> 4)  + zero) * x[base_idx + 7];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    if (tid == 0) y[local_row] = acc;
}
"#;

/// Quantize KV vector to Q8_0 format (same as GGML Q8_0 / existing GEMV kernels).
/// Block: [f16 scale (2B)][int8 × 32 (32B)] = 34 bytes per 32 elements.
/// head_dim=128 → 4 blocks × 34 = 136 bytes per head.
/// Layout: [max_seq × n_kv_heads × blocks_per_head × 34].
pub const KV_CACHE_WRITE_Q8_0_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write_q8_0(
    unsigned char* __restrict__ dst,     // quantized cache
    const float* __restrict__ src,       // [kv_dim] FP32 KV vector
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    // Grid: [n_kv_heads × blocks_per_head, 1, 1]. Block: [32, 1, 1].
    // Each threadblock quantizes one Q8_0 block (32 elements).
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;  // 0..31
    const int pos = pos_buf[0];

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (gid >= total_blocks) return;

    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    const int elem_offset = head_idx * head_dim + block_idx * 32 + tid;

    float val = src[elem_offset];

    // Warp reduction for max absolute value
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    // Output: pos * total_blocks * 34 + gid * 34
    unsigned char* out = dst + (size_t)pos * total_blocks * 34 + gid * 34;

    // Thread 0 writes the f16 scale
    if (tid == 0) {
        *((_Float16*)(out)) = (_Float16)scale;
    }
    // All threads write their int8 value
    out[2 + tid] = (unsigned char)(signed char)q;
}
"#;

/// Attention with Q8_0 quantized KV cache — same format as GGML Q8_0.
/// K and V caches stored as [max_seq × n_kv_heads × blocks_per_head × 34].
pub const ATTENTION_Q8_0_KV_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void attention_q8_0_kv(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn
) {
    int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];

    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* q_head = q + h * head_dim;
    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;

    float* scores = sdata;
    float* workspace = sdata + seq_len;

    // Phase 1: Q @ K^T with Q8_0 dequant
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float dot = 0.0f;
        for (int bi = 0; bi < blocks_per_head; bi++) {
            const unsigned char* block = k_cache + (size_t)t * total_blocks_per_pos * 34
                                        + (kv_head_block_start + bi) * 34;
            float d = (float)*((const _Float16*)block);
            for (int j = 0; j < 32; j++) {
                signed char qval = (signed char)block[2 + j];
                dot += q_head[bi * 32 + j] * (d * (float)qval);
            }
        }
        float s = dot * scale_attn;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    float max_val = workspace[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    float sum_val = workspace[0];
    __syncthreads();

    for (int t = tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // Phase 2: weighted sum of V with Q8_0 dequant
    float* out_head = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        int bi = d / 32;
        int bj = d % 32;
        for (int t = 0; t < seq_len; t++) {
            const unsigned char* block = v_cache + (size_t)t * total_blocks_per_pos * 34
                                        + (kv_head_block_start + bi) * 34;
            float vd = (float)*((const _Float16*)block) * (float)(signed char)block[2 + bj];
            val += scores[t] * vd;
        }
        out_head[d] = val;
    }
}
"#;

/// Quantize KV vector to Q8 (int8 symmetric) and write to quantized KV cache.
/// Per head: [4B f32 scale][head_dim × int8 values] = head_dim + 4 bytes.
/// For head_dim=128: 132 bytes vs 512 bytes FP32 = 3.88x compression.
pub const KV_CACHE_WRITE_Q8_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write_q8(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];

    const float* head_src = src + h * head_dim;

    // Find max absolute value for symmetric quantization
    extern __shared__ float smem[];
    float local_max = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(head_src[i]));
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float amax = smem[0];
    float scale = amax / 127.0f;
    float inv_scale = (scale > 0.0f) ? (127.0f / amax) : 0.0f;

    int bytes_per_head = 4 + head_dim;  // f32 scale + int8 values
    int bytes_per_pos = n_kv_heads * bytes_per_head;
    unsigned char* out = dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    if (tid == 0) *(float*)(out) = scale;
    __syncthreads();

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = head_src[i];
        int q = __float2int_rn(v * inv_scale);
        q = max(-127, min(127, q));
        out[4 + i] = (unsigned char)(signed char)q;
    }
}
"#;

/// Attention with Q8 quantized KV cache — symmetric int8, dequant on read.
pub const ATTENTION_Q8KV_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void attention_q8kv(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache_q8,
    const unsigned char* __restrict__ v_cache_q8,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn
) {
    int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];

    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* q_head = q + h * head_dim;
    int bytes_per_head = 4 + head_dim;
    int bytes_per_pos = n_kv_heads * bytes_per_head;

    float* scores = sdata;
    float* workspace = sdata + seq_len;

    // Phase 1: Q @ K^T with int8 dequant
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        const unsigned char* k_packed = k_cache_q8 + (size_t)t * bytes_per_pos + kv_h * bytes_per_head;
        float k_scale = *(const float*)(k_packed);
        const signed char* k_vals = (const signed char*)(k_packed + 4);

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * (k_scale * (float)k_vals[d]);
        }
        float s = dot * scale_attn;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    float max_val = workspace[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    float sum_val = workspace[0];
    __syncthreads();

    for (int t = tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // Phase 2: weighted sum of V with int8 dequant
    float* out_head = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            const unsigned char* v_packed = v_cache_q8 + (size_t)t * bytes_per_pos + kv_h * bytes_per_head;
            float v_scale = *(const float*)(v_packed);
            signed char v_q = (signed char)v_packed[4 + d];
            val += scores[t] * (v_scale * (float)v_q);
        }
        out_head[d] = val;
    }
}
"#;

/// Quantize KV vector to HFQ4-G128 and write to quantized KV cache.
/// Input: kv_dim floats at kv_src. Output: packed HFQ4 at dst[pos * bytes_per_pos].
/// Each group of 128 floats → 72 bytes (4B scale + 4B zero + 64B nibbles).
/// For head_dim=128, one head = exactly one group = 72 bytes.
pub const KV_CACHE_WRITE_Q4_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write_q4(
    unsigned char* __restrict__ dst,     // quantized cache
    const float* __restrict__ src,       // [kv_dim] FP32 KV vector
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int h = blockIdx.x;  // one block per KV head
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];

    const float* head_src = src + h * head_dim;

    // Shared memory for min/max reduction
    extern __shared__ float smem[];
    float* s_min = smem;
    float* s_max = smem + blockDim.x;

    // Find min/max across head_dim elements
    float local_min = 1e30f, local_max = -1e30f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = head_src[i];
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    float vmin = s_min[0];
    float vmax = s_max[0];

    float scale = (vmax - vmin) / 15.0f;
    float zero = vmin;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // Output layout per head: [4B scale][4B zero][head_dim/2 nibbles]
    int bytes_per_head = 8 + head_dim / 2;
    int bytes_per_pos = n_kv_heads * bytes_per_head;
    unsigned char* out = dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    // Write scale and zero (thread 0 only)
    if (tid == 0) {
        *(float*)(out) = scale;
        *(float*)(out + 4) = zero;
    }
    __syncthreads();

    // Quantize and pack nibbles (2 elements per byte)
    for (int i = tid; i < head_dim / 2; i += blockDim.x) {
        int e0 = i * 2;
        int e1 = i * 2 + 1;
        float v0 = head_src[e0];
        float v1 = head_src[e1];
        int q0 = __float2int_rn((v0 - zero) * inv_scale);
        int q1 = __float2int_rn((v1 - zero) * inv_scale);
        q0 = max(0, min(15, q0));
        q1 = max(0, min(15, q1));
        out[8 + i] = (unsigned char)((q1 << 4) | q0);
    }
}
"#;

/// Attention with quantized HFQ4 KV cache.
/// Same structure as attention_f32 but dequantizes K and V on the fly.
pub const ATTENTION_Q4KV_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void attention_q4kv(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache_q4,
    const unsigned char* __restrict__ v_cache_q4,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn
) {
    int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];

    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* q_head = q + h * head_dim;
    int bytes_per_head = 8 + head_dim / 2;
    int bytes_per_pos = n_kv_heads * bytes_per_head;

    float* scores = sdata;
    float* workspace = sdata + seq_len;

    // Phase 1: Q @ K^T with on-the-fly dequant
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        const unsigned char* k_packed = k_cache_q4 + (size_t)t * bytes_per_pos + kv_h * bytes_per_head;
        float k_scale = *(const float*)(k_packed);
        float k_zero  = *(const float*)(k_packed + 4);
        const unsigned char* k_nib = k_packed + 8;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d += 2) {
            unsigned char byte_val = k_nib[d / 2];
            float k0 = k_scale * (float)(byte_val & 0xF) + k_zero;
            float k1 = k_scale * (float)(byte_val >> 4) + k_zero;
            dot += q_head[d] * k0 + q_head[d + 1] * k1;
        }
        float s = dot * scale_attn;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    workspace[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
        __syncthreads();
    }
    float max_val = workspace[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    workspace[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) workspace[tid] += workspace[tid + s];
        __syncthreads();
    }
    float sum_val = workspace[0];
    __syncthreads();

    for (int t = tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // Phase 2: weighted sum of V with on-the-fly dequant
    float* out_head = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        int d_byte = d / 2;
        int d_nib_shift = (d & 1) * 4;
        for (int t = 0; t < seq_len; t++) {
            const unsigned char* v_packed = v_cache_q4 + (size_t)t * bytes_per_pos + kv_h * bytes_per_head;
            float v_scale = *(const float*)(v_packed);
            float v_zero  = *(const float*)(v_packed + 4);
            unsigned char byte_val = v_packed[8 + d_byte];
            float v_val = v_scale * (float)((byte_val >> d_nib_shift) & 0xF) + v_zero;
            val += scores[t] * v_val;
        }
        out_head[d] = val;
    }
}
"#;

/// GPU-side KV cache write using pos from a GPU buffer.
/// Copies kv_dim floats from src to dst at offset pos_buf[0] * kv_dim.
pub const KV_CACHE_WRITE_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write(
    float* __restrict__ dst,          // cache: [max_seq * kv_dim]
    const float* __restrict__ src,    // current KV: [kv_dim]
    const int* __restrict__ pos_buf,  // position buffer
    int kv_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    int pos = pos_buf[0];
    dst[pos * kv_dim + i] = src[i];
}
"#;

/// GPU-side top-K + top-P sampling. Eliminates 600KB logits download per token.
/// Single block, 256 threads. Returns token ID + RNG state (8 bytes vs 600KB).
///
/// Phase 1: Parallel max reduction over vocab_size logits.
/// Phase 2: Threshold filter — collect candidates within 30*temp of max (atomic shared counter).
/// Phase 3: Thread 0 softmax + sort + top-p + sample on the small candidate set.
pub const SAMPLE_TOP_P_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void sample_top_p(
    float* __restrict__ logits,          // [vocab_size] (modified in-place for repeat penalty)
    unsigned int* __restrict__ result,   // [2]: token_id, rng_state
    const unsigned int* __restrict__ repeat_tokens, // [repeat_window] recent token IDs
    int vocab_size,
    float temperature,
    float top_p,
    unsigned int rand_seed,
    int repeat_window,
    float repeat_penalty
) {
    extern __shared__ char smem[];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Shared memory layout:
    // [0..nthreads-1]: reduction workspace (floats)
    // [nthreads..nthreads+MAX_CAND-1]: candidate scores (floats)
    // [nthreads+MAX_CAND..nthreads+2*MAX_CAND-1]: candidate indices (ints)
    // [nthreads+2*MAX_CAND]: candidate count (int)
    const int MAX_CAND = 512;
    float* reduce = (float*)smem;
    float* cand_scores = reduce + nthreads;
    int* cand_indices = (int*)(cand_scores + MAX_CAND);
    int* cand_count = cand_indices + MAX_CAND;

    // Phase 0: Apply repetition penalty — only touches repeat_window positions
    for (int j = tid; j < repeat_window; j += nthreads) {
        int tok_id = repeat_tokens[j];
        if (tok_id < vocab_size) {
            float v = logits[tok_id];
            logits[tok_id] = (v > 0.0f) ? (v / repeat_penalty) : (v * repeat_penalty);
        }
    }
    __syncthreads();

    // Phase 1: Find max logit
    float local_max = -1e30f;
    for (int i = tid; i < vocab_size; i += nthreads) {
        local_max = fmaxf(local_max, logits[i]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float max_val = reduce[0];

    // Phase 2: Collect candidates above threshold
    float inv_temp = 1.0f / temperature;
    float thresh = max_val - 30.0f * temperature;

    if (tid == 0) *cand_count = 0;
    __syncthreads();

    for (int i = tid; i < vocab_size; i += nthreads) {
        if (logits[i] >= thresh) {
            int pos = atomicAdd(cand_count, 1);
            if (pos < MAX_CAND) {
                cand_scores[pos] = (logits[i] - max_val) * inv_temp;
                cand_indices[pos] = i;
            }
        }
    }
    __syncthreads();

    // Phase 3: Thread 0 — softmax, sort top-20, top-p filter, sample
    if (tid == 0) {
        int n = *cand_count;
        if (n > MAX_CAND) n = MAX_CAND;
        if (n == 0) { result[0] = 0; result[1] = rand_seed; return; }

        // Exp + sum
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float p = expf(cand_scores[i]);
            cand_scores[i] = p;
            sum += p;
        }

        // Partial selection sort — find top-20 only
        const int TOP_K = 20 < n ? 20 : n;
        for (int i = 0; i < TOP_K; i++) {
            int best = i;
            for (int j = i + 1; j < n; j++) {
                if (cand_scores[j] > cand_scores[best]) best = j;
            }
            if (best != i) {
                float ts = cand_scores[i]; cand_scores[i] = cand_scores[best]; cand_scores[best] = ts;
                int ti = cand_indices[i]; cand_indices[i] = cand_indices[best]; cand_indices[best] = ti;
            }
        }

        // RNG: xorshift32
        unsigned int rng = rand_seed;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float r = (float)rng / 4294967295.0f * sum;

        // Top-p + sample in one pass
        float cum = 0.0f;
        float p_thresh = top_p * sum;
        for (int i = 0; i < TOP_K; i++) {
            cum += cand_scores[i];
            if (cum >= r) {
                result[0] = (unsigned int)cand_indices[i];
                result[1] = rng;
                return;
            }
            if (cum >= p_thresh) break;
        }
        // Fallback: top-1
        result[0] = (unsigned int)cand_indices[0];
        result[1] = rng;
    }
}
"#;

/// GEMV Q4_F16_G64: matrix-vector multiply with on-the-fly Q4_F16 dequantization.
/// Block layout: f16 scale (2B) + f16 min (2B) + uint8 quants[32] (32B) = 36 bytes per 64 elements.
/// Dequant: weight = (_Float16)(nibble) * scale + min — single FP16 FMA on RDNA.
/// Thread tid reads quants[tid], processes both nibbles (elements tid and tid+32).
pub const GEMV_Q4F16_G64_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Q4_F16 group-64 GEMV: y = A * x
// 32 threads (single warp), warp shuffle reduction, no shared memory.
// FP16 dequant via native _Float16 FMA, FP32 accumulate for precision.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4f16_g64(
    const unsigned char* __restrict__ A_q4f16,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 64;
    const unsigned char* row_data = A_q4f16 + (size_t)row * blocks_per_row * 36;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 36;

        // Load scale and min — broadcast across warp via L0 cache
        _Float16 scale = *(const _Float16*)(block);
        _Float16 mn    = *(const _Float16*)(block + 2);

        // Each thread reads one byte, extracts both nibbles
        unsigned char qbyte = block[4 + tid];

        // Lower nibble -> element tid within block
        unsigned int nib_lo = qbyte & 0xF;
        _Float16 w0 = (_Float16)((unsigned short)nib_lo) * scale + mn;
        sum += (float)w0 * x[bi * 64 + tid];

        // Upper nibble -> element tid+32 within block
        unsigned int nib_hi = qbyte >> 4;
        _Float16 w1 = (_Float16)((unsigned short)nib_hi) * scale + mn;
        sum += (float)w1 * x[bi * 64 + 32 + tid];
    }

    // Warp shuffle reduction (5 steps for 32 threads)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
"#;

/// GEMV Q4_F16_G64 wide: 256 threads, element-strided access, shared memory reduction.
/// Matches F32 GEMV's occupancy pattern to test whether occupancy explains the 40% vs 48% gap.
/// Each thread processes elements tid, tid+256, tid+512, ... across the row.
pub const GEMV_Q4F16_G64_WIDE_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_q4f16_g64_wide(
    const unsigned char* __restrict__ A_q4f16,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    if (row >= M) return;

    const int blocks_per_row = K / 64;
    const unsigned char* row_data = A_q4f16 + (size_t)row * blocks_per_row * 36;

    float sum = 0.0f;

    // Element-strided: thread tid handles elements tid, tid+blockDim.x, ...
    for (int elem = threadIdx.x; elem < K; elem += blockDim.x) {
        int block_idx = elem >> 6;           // elem / 64 (shift since 64 = 2^6)
        int within = elem & 63;              // elem % 64

        const unsigned char* block = row_data + block_idx * 36;
        _Float16 scale = *(const _Float16*)(block);
        _Float16 mn    = *(const _Float16*)(block + 2);

        // Nibble extraction: elements 0-31 use lower nibble, 32-63 use upper
        int byte_idx = within & 31;          // within < 32 ? within : within - 32
        unsigned char qbyte = block[4 + byte_idx];
        unsigned int nibble = (within < 32) ? (qbyte & 0xF) : (qbyte >> 4);

        _Float16 w = (_Float16)((unsigned short)nibble) * scale + mn;
        sum += (float)w * x[elem];
    }

    // Shared memory tree reduction (same as F32 GEMV)
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sdata[0];
}
"#;

/// GEMV Q4_F16_G32: matrix-vector multiply with Q4_F16 group-32 dequantization.
/// Block layout: f16 scale (2B) + f16 min (2B) + uint8 quants[16] (16B) = 20 bytes per 32 elements.
/// Thread tid reads quants[tid&15], extracts its nibble based on tid < 16 or >= 16.
pub const GEMV_Q4F16_G32_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Q4_F16 group-32 GEMV: y = A * x
// 32 threads (single warp), 1 element per thread per block.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4f16_g32(
    const unsigned char* __restrict__ A_q4f16,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q4f16 + (size_t)row * blocks_per_row * 20;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 20;

        _Float16 scale = *(const _Float16*)(block);
        _Float16 mn    = *(const _Float16*)(block + 2);

        // Threads 0-15 read lower nibble, 16-31 read upper nibble of same byte
        unsigned char qbyte = block[4 + (tid & 15)];
        unsigned int nibble = (tid < 16) ? (qbyte & 0xF) : (qbyte >> 4);

        _Float16 w = (_Float16)((unsigned short)nibble) * scale + mn;
        sum += (float)w * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
"#;

/// Q8_0 embedding lookup: dequantize one row from a Q8_0 table to F32.
/// Block: 2 bytes f16 scale + 32 bytes int8 = 34 bytes per 32 elements.
pub const EMBEDDING_Q8_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_q8(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    int token_id, int dim
) {
    const int tid = threadIdx.x;
    const int blocks_per_row = dim / 32;
    const int bytes_per_row = blocks_per_row * 34;
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int block_idx = elem / 32;
        int within = elem % 32;
        const unsigned char* block = row + block_idx * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + within];
        output[elem] = d * (float)qval;
    }
}
"#;

/// Q4_K embedding lookup: dequantize one row from a Q4_K table to F32.
/// Avoids dequanting entire embedding to F32 (saves ~2GB for 150K+ vocabs).
/// 256 threads, one block, strided across the row's Q4_K blocks.
pub const EMBEDDING_Q4K_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_q4k(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    int token_id, int dim
) {
    const int tid = threadIdx.x;
    const int blocks_per_row = dim / 256;
    const int bytes_per_row = blocks_per_row * 144;
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int block_idx = elem / 256;
        int within = elem % 256;
        const unsigned char* block = row + block_idx * 144;

        float d = (float)*((const _Float16*)block);
        float dmin = (float)*((const _Float16*)(block + 2));
        const unsigned char* sc = block + 4;

        int group = within / 64;
        int in_group = within % 64;
        int sub = in_group / 32;
        int l = in_group % 32;
        int sb_idx = group * 2 + sub;

        float scale, mn;
        if (sb_idx < 4) {
            scale = d * (float)(sc[sb_idx] & 63);
            mn = dmin * (float)(sc[4 + sb_idx] & 63);
        } else {
            int i = sb_idx - 4;
            scale = d * (float)((sc[8 + i] & 0xF) | ((sc[i] >> 6) << 4));
            mn = dmin * (float)((sc[8 + i] >> 4) | ((sc[4 + i] >> 6) << 4));
        }

        unsigned char qbyte = block[16 + group * 32 + l];
        float nibble = (sub == 0) ? (float)(qbyte & 0xF) : (float)(qbyte >> 4);
        output[elem] = scale * nibble - mn;
    }
}
"#;

/// HFQ4-G256 embedding lookup: dequantize one row from HFQ4-G256 table to F32.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 elements.
pub const EMBEDDING_HFQ4G256_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_hfq4g256(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    int token_id, int dim
) {
    const int tid = threadIdx.x;
    const int groups_per_row = dim / 256;
    const int bytes_per_row = groups_per_row * 136;
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int group = elem / 256;
        int within = elem % 256;
        const unsigned char* gptr = row + group * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        int byte_idx = within / 2;
        unsigned char byte_val = gptr[8 + byte_idx];
        int nibble = (within % 2 == 0) ? (byte_val & 0xF) : (byte_val >> 4);
        output[elem] = scale * (float)nibble + zero;
    }
}
"#;

/// HFQ4-G128 embedding lookup: dequantize one row from HFQ4-G128 table to F32.
pub const EMBEDDING_HFQ4G128_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_hfq4g128(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    int token_id, int dim
) {
    const int tid = threadIdx.x;
    const int groups_per_row = dim / 128;
    const int bytes_per_row = groups_per_row * 72;
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int group = elem / 128;
        int within = elem % 128;
        const unsigned char* gptr = row + group * 72;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        int byte_idx = within / 2;
        unsigned char byte_val = gptr[8 + byte_idx];
        int nibble = (within % 2 == 0) ? (byte_val & 0xF) : (byte_val >> 4);
        output[elem] = scale * (float)nibble + zero;
    }
}
"#;

/// Q4_LUT GEMV: 4-bit with LDS codebook lookup.
/// Block: f16 codebook[16] (32 bytes) + u8 quants[16] (16 bytes) = 48 bytes per 32 elements.
/// Dequant: nibble → LDS[nibble] → f16 → FMA. No scale arithmetic per element.
/// 32 threads (single warp). Processes 8 blocks (256 elems) per outer iteration like Q8.
pub const GEMV_Q4LUT_SRC: &str = r#"
#include <hip/hip_runtime.h>

// LDS: 16 f16 codebook entries per block = 32 bytes. For 8 blocks = 256 bytes.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4lut(
    const unsigned char* __restrict__ A_q4lut,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    // 256 bytes of LDS for 8 codebooks (one per sub-block in the unrolled iteration)
    __shared__ _Float16 codebook[8][16];

    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q4lut + (size_t)row * blocks_per_row * 48;

    float sum = 0.0f;

    // Process 8 blocks (256 elements) per outer iteration
    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 48;
        const float* xb = x + oi * 256;

        // Phase 1: Load 8 codebooks into LDS. 32 threads, 128 entries = 4 per thread.
        // Each codebook is 16 entries of f16 = 32 bytes, at offset block*48.
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int flat = tid * 4 + i;        // 0..127
            int blk = flat / 16;           // which of 8 blocks
            int entry = flat % 16;         // which codebook entry
            const _Float16* cb = (const _Float16*)(base + blk * 48);
            codebook[blk][entry] = cb[entry];
        }
        // No __syncthreads needed — single warp, lockstep execution on RDNA

        // Phase 2: Each thread processes 8 elements (one per block).
        // Packing: byte[i] = elem[i](lo nibble) | elem[i+16](hi nibble)
        // Thread tid: tid<16 reads low nibble of byte[tid], tid>=16 reads high nibble of byte[tid-16]
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* quants = base + sub * 48 + 32;
            unsigned char qbyte = quants[tid & 15];
            unsigned int nibble = (tid < 16) ? (qbyte & 0xF) : (qbyte >> 4);
            float w = (float)codebook[sub][nibble];
            sum += w * xb[sub * 32 + tid];
        }
    }

    // Handle remaining blocks
    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 48;
        // Load codebook for this single block
        if (tid < 16) {
            codebook[0][tid] = ((const _Float16*)block)[tid];
        }
        const unsigned char* quants = block + 32;
        unsigned char qbyte = quants[tid & 15];
        unsigned int nibble = (tid < 16) ? (qbyte & 0xF) : (qbyte >> 4);
        float w = (float)codebook[0][nibble];
        sum += w * x[bi * 32 + tid];
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
"#;

/// Wave-cooperative Q4: use warp shuffle to distribute nibbles.
/// Same Q4_F16_G32 format (20 bytes/32 elem = 0.625 B/w).
/// 16 threads load 16 bytes, shuffle to give all 32 threads one nibble each.
/// Avoids the tid<16 conditional branch in the inner loop.
pub const GEMV_Q4WAVE_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4wave(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A + (size_t)row * blocks_per_row * 20;

    float sum = 0.0f;

    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 20;
        const float* xb = x + oi * 256;

        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 20;
            _Float16 scale = *(const _Float16*)(block);
            _Float16 mn    = *(const _Float16*)(block + 2);

            // Only threads 0-15 load a byte from memory
            unsigned int byte_val = (tid < 16) ? (unsigned int)block[4 + tid] : 0u;
            // Shuffle: thread t gets the byte that thread (t&15) loaded
            unsigned int shared_byte = __shfl(byte_val, tid & 15);
            // Extract: tid<16 gets low nibble, tid>=16 gets high nibble
            unsigned int nibble = (tid < 16) ? (shared_byte & 0xF) : (shared_byte >> 4);

            _Float16 w = (_Float16)((unsigned short)nibble) * scale + mn;
            sum += (float)w * xb[sub * 32 + tid];
        }
    }

    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 20;
        _Float16 scale = *(const _Float16*)(block);
        _Float16 mn    = *(const _Float16*)(block + 2);
        unsigned int byte_val = (tid < 16) ? (unsigned int)block[4 + tid] : 0u;
        unsigned int shared_byte = __shfl(byte_val, tid & 15);
        unsigned int nibble = (tid < 16) ? (shared_byte & 0xF) : (shared_byte >> 4);
        _Float16 w = (_Float16)((unsigned short)nibble) * scale + mn;
        sum += (float)w * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
"#;

/// Q4 stored as Q8: 4-bit precision quantized but stored in int8 (1 byte per weight).
/// Same as Q8_0 format (34 bytes per 32 elements) but values clamped to [-8,7].
/// Gets Q8 occupancy (16 VGPRs, 84% peak BW) at 4-bit quality.
/// 1.0625 bytes/weight — only useful when VRAM is not the constraint.
pub const GEMV_Q4AS8_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Identical to gemv_q8_0 — same format, just the values happen to only use 4 bits.
// The kernel doesn't know or care that values are clamped to [-8,7].
// This is here to prove that storage format determines speed, not bit precision.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4as8(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float sum = 0.0f;

    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;

        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }

    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (tid == 0) y[row] = sum;
}
"#;

/// GEMV Q8_HFQ: split-metadata row layout — scales contiguous, then values contiguous.
/// Row layout: [f16 scales × n_groups | int8 values × K | padding to 128B]
/// Pure sequential value stream with no metadata gaps every 34 bytes.
/// Narrow variant: 32 threads (1 warp), 8x unrolled, warp shuffle reduction.
pub const GEMV_Q8HFQ_SRC: &str = r#"
#include <hip/hip_runtime.h>

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q8hfq(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int row_stride
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const unsigned char* row_base = A + (size_t)row * row_stride;
    const int n_groups = K / 32;

    const unsigned char* scales_ptr = row_base;
    const signed char* values_ptr = (const signed char*)(row_base + n_groups * 2);

    float sum = 0.0f;

    const int outer_iters = n_groups / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            int gi = oi * 8 + sub;
            float d = (float)*((const _Float16*)(scales_ptr + gi * 2));
            signed char qval = values_ptr[gi * 32 + tid];
            sum += d * (float)qval * x[gi * 32 + tid];
        }
    }
    for (int gi = outer_iters * 8; gi < n_groups; gi++) {
        float d = (float)*((const _Float16*)(scales_ptr + gi * 2));
        signed char qval = values_ptr[gi * 32 + tid];
        sum += d * (float)qval * x[gi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (tid == 0) y[row] = sum;
}
"#;

/// GEMV Q8_HFQ wide: 2 warps per block, each processes one row independently.
/// Same split-metadata layout. 8x unrolled. Grid = ceil(M/2).
pub const GEMV_Q8HFQ_WIDE_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_q8hfq_wide(
    const unsigned char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int row_stride
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const unsigned char* row_base = A + (size_t)row * row_stride;
    const int n_groups = K / 32;

    const unsigned char* scales_ptr = row_base;
    const signed char* values_ptr = (const signed char*)(row_base + n_groups * 2);

    float sum = 0.0f;

    int gi = 0;
    for (; gi + 7 < n_groups; gi += 8) {
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            float d = (float)*((const _Float16*)(scales_ptr + (gi + u) * 2));
            signed char qval = values_ptr[(gi + u) * 32 + lane];
            sum += d * (float)qval * x[(gi + u) * 32 + lane];
        }
    }
    for (; gi < n_groups; gi++) {
        float d = (float)*((const _Float16*)(scales_ptr + gi * 2));
        signed char qval = values_ptr[gi * 32 + lane];
        sum += d * (float)qval * x[gi * 32 + lane];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0) y[row] = sum;
}
"#;

/// GPU argmax: find index of maximum value.
pub const ARGMAX_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void argmax_f32(
    const float* __restrict__ data, int* __restrict__ result, int n
) {
    extern __shared__ float s[];
    int* si = (int*)(s + blockDim.x);
    float lmax = -1e30f; int lidx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (data[i] > lmax) { lmax = data[i]; lidx = i; }
    }
    s[threadIdx.x] = lmax; si[threadIdx.x] = lidx;
    __syncthreads();
    for (int sz = blockDim.x/2; sz > 0; sz >>= 1) {
        if (threadIdx.x < sz && s[threadIdx.x + sz] > s[threadIdx.x]) {
            s[threadIdx.x] = s[threadIdx.x + sz]; si[threadIdx.x] = si[threadIdx.x + sz];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) result[0] = si[0];
}
"#;
