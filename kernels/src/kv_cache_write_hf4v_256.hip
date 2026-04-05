#include <hip/hip_runtime.h>

// HF4-V: Hipfire-native 4-bit V cache for RDNA.
// No FWHT rotation. No centroid LUT. Direct quantization designed for
// cache-line-friendly access during attention.
//
// Format per head per position:
//   [f32 norm (4B)] [f16 scale (2B)] [f16 min (2B)] [128 nibbles (128B)] = 136 bytes
//   (vs turbo4: 132B, vs Q8: 272B → 2.0x compression vs Q8)
//
// Quantization: affine 4-bit with per-head min/scale
//   nibble = clamp(round((val - min) / scale), 0, 15)
//   dequant = min + nibble * scale  (1 FMA, zero LUT)
//
// Math magic: values are L2-normalized before quantization (preserving
// direction), with the original norm stored. This concentrates the
// distribution and makes uniform 4-bit quantization near-optimal
// (normalized vectors have small dynamic range).
//
// Grid: [n_kv_heads]. Block: [32]. Each thread handles 8 dims.
extern "C" __launch_bounds__(32)
__global__ void kv_cache_write_hf4v_256(
    const float* __restrict__ v_src,     // [n_kv_heads * 256]
    unsigned char* __restrict__ v_dst,    // cache: [max_seq * n_kv_heads * 72]
    const int* __restrict__ pos_buf,
    int n_kv_heads, int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];
    const int bytes_per_head = 8 + head_dim / 2;  // 4(norm) + 2(scale_f16) + 2(min_f16) + 128 nibbles = 136
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    const float* src = v_src + h * head_dim + tid * 8;
    unsigned char* dst = v_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    // Load 8 values
    float v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
    float v4 = src[4], v5 = src[5], v6 = src[6], v7 = src[7];

    // L2 norm (warp reduce)
    float sq = v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7;
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor(sq, off);
    float norm = sqrtf(sq + 1e-12f);
    float inv_norm = 1.0f / norm;

    // Normalize
    v0 *= inv_norm; v1 *= inv_norm; v2 *= inv_norm; v3 *= inv_norm;
    v4 *= inv_norm; v5 *= inv_norm; v6 *= inv_norm; v7 *= inv_norm;

    // Per-head min/max of normalized values (warp reduce)
    float local_min = fminf(fminf(fminf(v0,v1),fminf(v2,v3)), fminf(fminf(v4,v5),fminf(v6,v7)));
    float local_max = fmaxf(fmaxf(fmaxf(v0,v1),fmaxf(v2,v3)), fmaxf(fmaxf(v4,v5),fmaxf(v6,v7)));
    for (int off = 16; off > 0; off >>= 1) {
        local_min = fminf(local_min, __shfl_xor(local_min, off));
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    }

    float range = local_max - local_min;
    float scale = (range > 1e-10f) ? range / 15.0f : 1e-10f;
    float inv_scale = 1.0f / scale;

    // Quantize to 4-bit (0-15)
    #define Q4(val) (unsigned char)fminf(fmaxf(roundf(((val) - local_min) * inv_scale), 0.0f), 15.0f)
    unsigned char n0 = Q4(v0), n1 = Q4(v1), n2 = Q4(v2), n3 = Q4(v3);
    unsigned char n4 = Q4(v4), n5 = Q4(v5), n6 = Q4(v6), n7 = Q4(v7);
    #undef Q4

    // Pack: 8 nibbles → 4 bytes
    unsigned char b0 = (n1 << 4) | n0;
    unsigned char b1 = (n3 << 4) | n2;
    unsigned char b2 = (n5 << 4) | n4;
    unsigned char b3 = (n7 << 4) | n6;

    // Store header (thread 0 writes norm + scale, thread 1 writes min)
    if (tid == 0) {
        *(float*)(dst) = norm;
        // Pack scale and min into bytes 4-7 as f16 pair to save space
        // Actually: store norm as f32 (4B) and pack scale+min as 2×f16 (4B)
        *(_Float16*)(dst + 4) = (_Float16)scale;
        *(_Float16*)(dst + 6) = (_Float16)local_min;
    }

    // Store nibbles (contiguous: 4 bytes per thread, 32 threads × 4 = 128 bytes... wait, head_dim/2=128)
    // Offset: 8 header bytes + tid*4
    dst[8 + tid * 4 + 0] = b0;
    dst[8 + tid * 4 + 1] = b1;
    dst[8 + tid * 4 + 2] = b2;
    dst[8 + tid * 4 + 3] = b3;
}
