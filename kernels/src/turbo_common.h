#include <hip/hip_runtime.h>

// Native ds_swizzle_b32 Hadamard butterfly: XOR swap + add/sub.
// Replaces __shfl_xor (compiles to ds_bpermute, needs VGPR for lane index).
// ds_swizzle encodes XOR pattern as immediate — no VGPR, lower latency.
#define HADAMARD_BFLY(v, pattern, stride, tid) do {                     \
    float _p = __int_as_float(                                          \
        __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pattern)));     \
    if ((tid) & (stride)) { (v) = _p - (v); }                          \
    else                  { (v) = (v) + _p; }                           \
} while(0)

// Lloyd-Max optimal centroids for N(0, 1/128) after unit-norm + FWHT(1/sqrt(128))
__constant__ float TURBO_C2[4] = {-0.133466f, -0.040022f, 0.040022f, 0.133466f};
__constant__ float TURBO_C3[8] = {-0.190685f, -0.117832f, -0.065717f, -0.021460f, 0.021460f, 0.065717f, 0.117832f, 0.190685f};
__constant__ float TURBO_C4[16] = {
    -0.241565f, -0.182875f, -0.143012f, -0.111016f, -0.083262f, -0.057983f, -0.034295f, -0.011225f,
     0.011225f,  0.034295f,  0.057983f,  0.083262f,  0.111016f,  0.143012f,  0.182875f,  0.241565f
};

// Lloyd-Max optimal centroids for N(0, 1/256) — all values = 128-dim × 1/sqrt(2)
__constant__ float TURBO_C2_256[4] = {-0.094376f, -0.028300f, 0.028300f, 0.094376f};
__constant__ float TURBO_C3_256[8] = {-0.134860f, -0.083320f, -0.046469f, -0.015176f, 0.015176f, 0.046469f, 0.083320f, 0.134860f};
__constant__ float TURBO_C4_256[16] = {
    -0.170807f, -0.129321f, -0.101134f, -0.078505f, -0.058869f, -0.041003f, -0.024249f, -0.007938f,
     0.007938f,  0.024249f,  0.041003f,  0.058869f,  0.078505f,  0.101134f,  0.129321f,  0.170807f
};

// In-place FWHT on 128 elements in registers.
// signs1/signs2 are ±1.0f arrays in global memory (uploaded once).
__device__ void fwht_forward_128(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    // Step 1: apply signs1
    for (int i = 0; i < 128; i++) x[i] *= signs1[i];

    // Step 2: Walsh-Hadamard butterfly (7 passes for n=128)
    for (int stride = 1; stride < 128; stride <<= 1) {
        for (int i = 0; i < 128; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }

    // Step 3: scale by 1/sqrt(128)
    const float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;

    // Step 4: apply signs2
    for (int i = 0; i < 128; i++) x[i] *= signs2[i];
}

// Inverse FWHT: signs2 -> butterfly -> scale -> signs1 (reverse order)
__device__ void fwht_inverse_128(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    for (int i = 0; i < 128; i++) x[i] *= signs2[i];
    for (int stride = 1; stride < 128; stride <<= 1) {
        for (int i = 0; i < 128; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128 * signs1[i];
}


// Register-only FWHT via __shfl_xor. Zero shared memory. Zero barriers.
// Each thread owns 4 of 128 elements in registers (a,b,c,d).
// signs1/signs2 are applied from constant memory.
// After this function, (a,b,c,d) are in the FWHT-rotated space.
__device__ void fwht_shfl_forward(float& a, float& b, float& c, float& d,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    int d0 = tid * 4;
    // Apply signs1
    a *= signs1[d0]; b *= signs1[d0+1]; c *= signs1[d0+2]; d *= signs1[d0+3];

    // Local butterfly: stride 1 (pairs 0↔1, 2↔3)
    float t;
    t = a; a = a + b; b = t - b;
    t = c; c = c + d; d = t - d;

    // Local butterfly: stride 2 (pairs 0↔2, 1↔3)
    t = a; a = a + c; c = t - c;
    t = b; b = b + d; d = t - d;

    // Wave-level butterfly via ds_swizzle_b32 (native XOR swap, no VGPR for lane index)
    HADAMARD_BFLY(a, 0x041F, 1, tid); HADAMARD_BFLY(b, 0x041F, 1, tid);
    HADAMARD_BFLY(c, 0x041F, 1, tid); HADAMARD_BFLY(d, 0x041F, 1, tid);
    HADAMARD_BFLY(a, 0x081F, 2, tid); HADAMARD_BFLY(b, 0x081F, 2, tid);
    HADAMARD_BFLY(c, 0x081F, 2, tid); HADAMARD_BFLY(d, 0x081F, 2, tid);
    HADAMARD_BFLY(a, 0x101F, 4, tid); HADAMARD_BFLY(b, 0x101F, 4, tid);
    HADAMARD_BFLY(c, 0x101F, 4, tid); HADAMARD_BFLY(d, 0x101F, 4, tid);
    HADAMARD_BFLY(a, 0x201F, 8, tid); HADAMARD_BFLY(b, 0x201F, 8, tid);
    HADAMARD_BFLY(c, 0x201F, 8, tid); HADAMARD_BFLY(d, 0x201F, 8, tid);
    HADAMARD_BFLY(a, 0x401F, 16, tid); HADAMARD_BFLY(b, 0x401F, 16, tid);
    HADAMARD_BFLY(c, 0x401F, 16, tid); HADAMARD_BFLY(d, 0x401F, 16, tid);

    // Scale by 1/sqrt(128) and apply signs2
    const float s = 0.08838834764831845f;
    a *= s * signs2[d0]; b *= s * signs2[d0+1]; c *= s * signs2[d0+2]; d *= s * signs2[d0+3];
}

// Inverse: signs2, butterfly, scale, signs1 (reverse order)
__device__ void fwht_shfl_inverse(float& a, float& b, float& c, float& d,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    int d0 = tid * 4;
    a *= signs2[d0]; b *= signs2[d0+1]; c *= signs2[d0+2]; d *= signs2[d0+3];

    HADAMARD_BFLY(a, 0x041F, 1, tid); HADAMARD_BFLY(b, 0x041F, 1, tid);
    HADAMARD_BFLY(c, 0x041F, 1, tid); HADAMARD_BFLY(d, 0x041F, 1, tid);
    HADAMARD_BFLY(a, 0x081F, 2, tid); HADAMARD_BFLY(b, 0x081F, 2, tid);
    HADAMARD_BFLY(c, 0x081F, 2, tid); HADAMARD_BFLY(d, 0x081F, 2, tid);
    HADAMARD_BFLY(a, 0x101F, 4, tid); HADAMARD_BFLY(b, 0x101F, 4, tid);
    HADAMARD_BFLY(c, 0x101F, 4, tid); HADAMARD_BFLY(d, 0x101F, 4, tid);
    HADAMARD_BFLY(a, 0x201F, 8, tid); HADAMARD_BFLY(b, 0x201F, 8, tid);
    HADAMARD_BFLY(c, 0x201F, 8, tid); HADAMARD_BFLY(d, 0x201F, 8, tid);
    HADAMARD_BFLY(a, 0x401F, 16, tid); HADAMARD_BFLY(b, 0x401F, 16, tid);
    HADAMARD_BFLY(c, 0x401F, 16, tid); HADAMARD_BFLY(d, 0x401F, 16, tid);

    float t;
    t = a; a = a + c; c = t - c;
    t = b; b = b + d; d = t - d;
    t = a; a = a + b; b = t - b;
    t = c; c = c + d; d = t - d;

    const float s = 0.08838834764831845f;
    a *= s * signs1[d0]; b *= s * signs1[d0+1]; c *= s * signs1[d0+2]; d *= s * signs1[d0+3];
}

// ============================================================================
// 256-element FWHT via __shfl_xor. Zero shared memory. Zero barriers.
// Each thread owns 8 of 256 elements in registers (v0..v7).
// 32 threads × 8 elements = 256.
// log2(256) = 8 butterfly passes:
//   Passes 1-3: register-local (strides 1, 2, 4 in element space)
//   Passes 4-8: warp shuffle    (strides 8,16,32,64,128 in element space
//                                = strides 1,2,4,8,16 in thread space)
//
// VGPR budget (RDNA, 32-lane wavefront):
//   v0-v7:  8 VGPRs  (the 8 data elements)
//   t:      1 VGPR   (butterfly temp, reused across all local passes)
//   d0:     1 VGPR   (base index for sign table loads, computed once)
//   s:      1 VGPR   (scale constant 1/sqrt(256), fused with signs2)
//   p0-p7:  0 VGPRs  (shuffle results alias v0-v7 via compiler; see note)
//   tid:    1 VGPR   (passed in, likely already live in caller)
//   signs:  0 VGPRs  (loaded directly into multiply operand from global)
//   -------
//   Total: 12 VGPRs kernel-private + 8 data = 20 VGPRs worst case
//
// Note on shuffle passes: each __shfl_xor result (p0..p7) is consumed
// immediately in the add/sub that overwrites v0..v7, so the compiler
// can alias them. At most 1 extra VGPR for the shuffle temporary if
// the compiler doesn't overlap. In practice expect 18-19 VGPRs on RDNA.
// ============================================================================

__device__ void fwht_shfl_forward_256(
    float& v0, float& v1, float& v2, float& v3,
    float& v4, float& v5, float& v6, float& v7,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    const int d0 = tid * 8;

    // Apply signs1 (randomize input for incoherence)
    v0 *= signs1[d0];   v1 *= signs1[d0+1]; v2 *= signs1[d0+2]; v3 *= signs1[d0+3];
    v4 *= signs1[d0+4]; v5 *= signs1[d0+5]; v6 *= signs1[d0+6]; v7 *= signs1[d0+7];

    // --- Pass 1: stride 1 in element space ---
    // Butterfly pairs: (0,1), (2,3), (4,5), (6,7)
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    // --- Pass 2: stride 2 in element space ---
    // Butterfly pairs: (0,2), (1,3), (4,6), (5,7)
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    // --- Pass 3: stride 4 in element space ---
    // Butterfly pairs: (0,4), (1,5), (2,6), (3,7)
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // --- Passes 4-8: ds_swizzle_b32 (native XOR swap, no VGPR for lane index) ---
    #define HBFLY8(pat, str) \
        HADAMARD_BFLY(v0,(pat),(str),tid); HADAMARD_BFLY(v1,(pat),(str),tid); \
        HADAMARD_BFLY(v2,(pat),(str),tid); HADAMARD_BFLY(v3,(pat),(str),tid); \
        HADAMARD_BFLY(v4,(pat),(str),tid); HADAMARD_BFLY(v5,(pat),(str),tid); \
        HADAMARD_BFLY(v6,(pat),(str),tid); HADAMARD_BFLY(v7,(pat),(str),tid)
    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8

    // Scale by 1/sqrt(256) = 1/16 = 0.0625 and apply signs2
    const float s = 0.0625f;
    v0 *= s * signs2[d0];   v1 *= s * signs2[d0+1]; v2 *= s * signs2[d0+2]; v3 *= s * signs2[d0+3];
    v4 *= s * signs2[d0+4]; v5 *= s * signs2[d0+5]; v6 *= s * signs2[d0+6]; v7 *= s * signs2[d0+7];
}

// Inverse FWHT-256: signs2 -> shuffle passes -> local passes (reversed) -> scale -> signs1
__device__ void fwht_shfl_inverse_256(
    float& v0, float& v1, float& v2, float& v3,
    float& v4, float& v5, float& v6, float& v7,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    const int d0 = tid * 8;

    // Apply signs2 (undo output randomization)
    v0 *= signs2[d0];   v1 *= signs2[d0+1]; v2 *= signs2[d0+2]; v3 *= signs2[d0+3];
    v4 *= signs2[d0+4]; v5 *= signs2[d0+5]; v6 *= signs2[d0+6]; v7 *= signs2[d0+7];

    // --- Passes 4-8: ds_swizzle_b32 ---
    #define HBFLY8(pat, str) \
        HADAMARD_BFLY(v0,(pat),(str),tid); HADAMARD_BFLY(v1,(pat),(str),tid); \
        HADAMARD_BFLY(v2,(pat),(str),tid); HADAMARD_BFLY(v3,(pat),(str),tid); \
        HADAMARD_BFLY(v4,(pat),(str),tid); HADAMARD_BFLY(v5,(pat),(str),tid); \
        HADAMARD_BFLY(v6,(pat),(str),tid); HADAMARD_BFLY(v7,(pat),(str),tid)
    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8

    // --- Pass 3 (reverse): stride 4 --- pairs (0,4), (1,5), (2,6), (3,7)
    float t;
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // --- Pass 2 (reverse): stride 2 --- pairs (0,2), (1,3), (4,6), (5,7)
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    // --- Pass 1 (reverse): stride 1 --- pairs (0,1), (2,3), (4,5), (6,7)
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    // Scale by 1/sqrt(256) and apply signs1
    const float s = 0.0625f;
    v0 *= s * signs1[d0];   v1 *= s * signs1[d0+1]; v2 *= s * signs1[d0+2]; v3 *= s * signs1[d0+3];
    v4 *= s * signs1[d0+4]; v5 *= s * signs1[d0+5]; v6 *= s * signs1[d0+6]; v7 *= s * signs1[d0+7];
}

// ============================================================================
// Scalar reference FWHT-256 (for validation / non-turbo paths)
// ============================================================================
__device__ void fwht_forward_256(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    for (int i = 0; i < 256; i++) x[i] *= signs1[i];
    for (int stride = 1; stride < 256; stride <<= 1) {
        for (int i = 0; i < 256; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    const float inv_sqrt_256 = 0.0625f;
    for (int i = 0; i < 256; i++) x[i] *= inv_sqrt_256 * signs2[i];
}

__device__ void fwht_inverse_256(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    for (int i = 0; i < 256; i++) x[i] *= signs2[i];
    for (int stride = 1; stride < 256; stride <<= 1) {
        for (int i = 0; i < 256; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    const float inv_sqrt_256 = 0.0625f;
    for (int i = 0; i < 256; i++) x[i] *= inv_sqrt_256 * signs1[i];
}


// Branchless 2-bit quantize: returns index 0-3 (thresholds for N(0, 1/128))
__device__ int turbo_quantize_2bit(float x) {
    return (x > -0.086744f) + (x > 0.0f) + (x > 0.086744f);
}

// Branchless 3-bit quantize: returns index 0-7
__device__ int turbo_quantize_3bit(float x) {
    return (x > -0.154258f) + (x > -0.091775f) + (x > -0.043589f) + (x > 0.0f)
         + (x > 0.043589f) + (x > 0.091775f) + (x > 0.154258f);
}

// Branchless 4-bit quantize: returns index 0-15
__device__ int turbo_quantize_4bit(float x) {
    return (x > -0.212220f) + (x > -0.162944f) + (x > -0.127014f) + (x > -0.097139f)
         + (x > -0.070622f) + (x > -0.046139f) + (x > -0.022760f) + (x > 0.0f)
         + (x > 0.022760f) + (x > 0.046139f) + (x > 0.070622f) + (x > 0.097139f)
         + (x > 0.127014f) + (x > 0.162944f) + (x > 0.212220f);
}

// 256-dim quantize functions — thresholds = 128-dim × 1/sqrt(2)
__device__ int turbo_quantize_2bit_256(float x) {
    return (x > -0.061352f) + (x > 0.0f) + (x > 0.061352f);
}
__device__ int turbo_quantize_3bit_256(float x) {
    return (x > -0.109068f) + (x > -0.064903f) + (x > -0.030827f) + (x > 0.0f)
         + (x > 0.030827f) + (x > 0.064903f) + (x > 0.109068f);
}
__device__ int turbo_quantize_4bit_256(float x) {
    return (x > -0.150086f) + (x > -0.115239f) + (x > -0.089815f) + (x > -0.068691f)
         + (x > -0.049935f) + (x > -0.032626f) + (x > -0.016096f) + (x > 0.0f)
         + (x > 0.016096f) + (x > 0.032626f) + (x > 0.049935f) + (x > 0.068691f)
         + (x > 0.089815f) + (x > 0.115239f) + (x > 0.150086f);
}

// ============================================================================
// 512-element FWHT via ds_swizzle_b32.  Zero shared memory.  Zero barriers.
// Each thread owns 16 of 512 elements in registers (v0..v15).
// 32 threads × 16 elements = 512.
// log2(512) = 9 butterfly passes:
//   Passes 1-4: register-local (strides 1, 2, 4, 8 in element space)
//   Passes 5-9: warp shuffle    (strides 16,32,64,128,256 in element space
//                                = strides  1, 2, 4,  8, 16 in thread space)
//
// VGPR budget estimate (RDNA wave32):
//   v0-v15: 16 VGPRs  (data)
//   t:       1 VGPR   (butterfly temp, reused across all local passes)
//   d0:      1 VGPR   (base index tid*16, computed once)
//   s:       1 VGPR   (scale 1/sqrt(512), fused with signs2 multiply)
//   swizzle: 0-1 VGPR (compiler aliases shuffle result with target register)
//   tid:     1 VGPR   (passed in, already live in caller)
//   Total: ~20 VGPRs worst case — occupancy 16 on gfx1201 at __launch_bounds__(32,16)
// ============================================================================

// Lloyd-Max optimal centroids for N(0, 1/512) — = 256-dim values × 1/sqrt(2)
__constant__ float TURBO_C3_512[8] = {
    -0.095360f, -0.058916f, -0.032859f, -0.010731f,
     0.010731f,  0.032859f,  0.058916f,  0.095360f
};

// Branchless 3-bit quantize for N(0, 1/512): returns index 0-7.
// Thresholds = 256-dim thresholds × 1/sqrt(2).
__device__ int turbo_quantize_3bit_512(float x) {
    return (x > -0.077123f) + (x > -0.045893f) + (x > -0.021798f) + (x > 0.0f)
         + (x > 0.021798f)  + (x > 0.045893f)  + (x > 0.077123f);
}

// Forward FWHT-512: 16 elements/lane in 16 VGPRs v0..v15.
// Lane tid owns dims tid*16 .. tid*16+15 (d0 = tid*16).
__device__ void fwht_shfl_forward_512(
    float& v0,  float& v1,  float& v2,  float& v3,
    float& v4,  float& v5,  float& v6,  float& v7,
    float& v8,  float& v9,  float& v10, float& v11,
    float& v12, float& v13, float& v14, float& v15,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    const int d0 = tid * 16;

    // Apply signs1 (randomize input for incoherence)
    v0  *= signs1[d0];    v1  *= signs1[d0+1];  v2  *= signs1[d0+2];  v3  *= signs1[d0+3];
    v4  *= signs1[d0+4];  v5  *= signs1[d0+5];  v6  *= signs1[d0+6];  v7  *= signs1[d0+7];
    v8  *= signs1[d0+8];  v9  *= signs1[d0+9];  v10 *= signs1[d0+10]; v11 *= signs1[d0+11];
    v12 *= signs1[d0+12]; v13 *= signs1[d0+13]; v14 *= signs1[d0+14]; v15 *= signs1[d0+15];

    // --- Pass 1: stride 1 in element space ---
    // Butterfly pairs: (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)
    float t;
    t = v0;  v0  = v0  + v1;  v1  = t - v1;
    t = v2;  v2  = v2  + v3;  v3  = t - v3;
    t = v4;  v4  = v4  + v5;  v5  = t - v5;
    t = v6;  v6  = v6  + v7;  v7  = t - v7;
    t = v8;  v8  = v8  + v9;  v9  = t - v9;
    t = v10; v10 = v10 + v11; v11 = t - v11;
    t = v12; v12 = v12 + v13; v13 = t - v13;
    t = v14; v14 = v14 + v15; v15 = t - v15;

    // --- Pass 2: stride 2 in element space ---
    // Butterfly pairs: (0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(12,14),(13,15)
    t = v0;  v0  = v0  + v2;  v2  = t - v2;
    t = v1;  v1  = v1  + v3;  v3  = t - v3;
    t = v4;  v4  = v4  + v6;  v6  = t - v6;
    t = v5;  v5  = v5  + v7;  v7  = t - v7;
    t = v8;  v8  = v8  + v10; v10 = t - v10;
    t = v9;  v9  = v9  + v11; v11 = t - v11;
    t = v12; v12 = v12 + v14; v14 = t - v14;
    t = v13; v13 = v13 + v15; v15 = t - v15;

    // --- Pass 3: stride 4 in element space ---
    // Butterfly pairs: (0,4),(1,5),(2,6),(3,7),(8,12),(9,13),(10,14),(11,15)
    t = v0;  v0  = v0  + v4;  v4  = t - v4;
    t = v1;  v1  = v1  + v5;  v5  = t - v5;
    t = v2;  v2  = v2  + v6;  v6  = t - v6;
    t = v3;  v3  = v3  + v7;  v7  = t - v7;
    t = v8;  v8  = v8  + v12; v12 = t - v12;
    t = v9;  v9  = v9  + v13; v13 = t - v13;
    t = v10; v10 = v10 + v14; v14 = t - v14;
    t = v11; v11 = v11 + v15; v15 = t - v15;

    // --- Pass 4: stride 8 in element space ---
    // Butterfly pairs: (0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15)
    t = v0;  v0  = v0  + v8;  v8  = t - v8;
    t = v1;  v1  = v1  + v9;  v9  = t - v9;
    t = v2;  v2  = v2  + v10; v10 = t - v10;
    t = v3;  v3  = v3  + v11; v11 = t - v11;
    t = v4;  v4  = v4  + v12; v12 = t - v12;
    t = v5;  v5  = v5  + v13; v13 = t - v13;
    t = v6;  v6  = v6  + v14; v14 = t - v14;
    t = v7;  v7  = v7  + v15; v15 = t - v15;

    // --- Passes 5-9: ds_swizzle_b32 (lane-strides 1,2,4,8,16) ---
    // Swizzle reach max XOR-lane = 16 — identical to FWHT-256, no cross-warp needed.
    #define HBFLY16(pat, str) \
        HADAMARD_BFLY(v0,(pat),(str),tid);  HADAMARD_BFLY(v1,(pat),(str),tid); \
        HADAMARD_BFLY(v2,(pat),(str),tid);  HADAMARD_BFLY(v3,(pat),(str),tid); \
        HADAMARD_BFLY(v4,(pat),(str),tid);  HADAMARD_BFLY(v5,(pat),(str),tid); \
        HADAMARD_BFLY(v6,(pat),(str),tid);  HADAMARD_BFLY(v7,(pat),(str),tid); \
        HADAMARD_BFLY(v8,(pat),(str),tid);  HADAMARD_BFLY(v9,(pat),(str),tid); \
        HADAMARD_BFLY(v10,(pat),(str),tid); HADAMARD_BFLY(v11,(pat),(str),tid); \
        HADAMARD_BFLY(v12,(pat),(str),tid); HADAMARD_BFLY(v13,(pat),(str),tid); \
        HADAMARD_BFLY(v14,(pat),(str),tid); HADAMARD_BFLY(v15,(pat),(str),tid)
    HBFLY16(0x041F, 1);
    HBFLY16(0x081F, 2);
    HBFLY16(0x101F, 4);
    HBFLY16(0x201F, 8);
    HBFLY16(0x401F, 16);
    #undef HBFLY16

    // Scale by 1/sqrt(512) = 0.0441941738f and apply signs2
    const float s = 0.0441941738f;
    v0  *= s * signs2[d0];    v1  *= s * signs2[d0+1];  v2  *= s * signs2[d0+2];  v3  *= s * signs2[d0+3];
    v4  *= s * signs2[d0+4];  v5  *= s * signs2[d0+5];  v6  *= s * signs2[d0+6];  v7  *= s * signs2[d0+7];
    v8  *= s * signs2[d0+8];  v9  *= s * signs2[d0+9];  v10 *= s * signs2[d0+10]; v11 *= s * signs2[d0+11];
    v12 *= s * signs2[d0+12]; v13 *= s * signs2[d0+13]; v14 *= s * signs2[d0+14]; v15 *= s * signs2[d0+15];
}

// Inverse FWHT-512: signs2 -> shuffle passes -> local passes (reversed) -> scale -> signs1
// Only needed for the lloyd3-rotated-V reduce path (v_mode=3).  Not used in the
// default q8-V (v_mode=8) path shipped in v1.
__device__ void fwht_shfl_inverse_512(
    float& v0,  float& v1,  float& v2,  float& v3,
    float& v4,  float& v5,  float& v6,  float& v7,
    float& v8,  float& v9,  float& v10, float& v11,
    float& v12, float& v13, float& v14, float& v15,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    const int d0 = tid * 16;

    // Apply signs2 (undo output randomization)
    v0  *= signs2[d0];    v1  *= signs2[d0+1];  v2  *= signs2[d0+2];  v3  *= signs2[d0+3];
    v4  *= signs2[d0+4];  v5  *= signs2[d0+5];  v6  *= signs2[d0+6];  v7  *= signs2[d0+7];
    v8  *= signs2[d0+8];  v9  *= signs2[d0+9];  v10 *= signs2[d0+10]; v11 *= signs2[d0+11];
    v12 *= signs2[d0+12]; v13 *= signs2[d0+13]; v14 *= signs2[d0+14]; v15 *= signs2[d0+15];

    // --- Passes 5-9: ds_swizzle_b32 (same 5 patterns, same order as forward) ---
    #define HBFLY16(pat, str) \
        HADAMARD_BFLY(v0,(pat),(str),tid);  HADAMARD_BFLY(v1,(pat),(str),tid); \
        HADAMARD_BFLY(v2,(pat),(str),tid);  HADAMARD_BFLY(v3,(pat),(str),tid); \
        HADAMARD_BFLY(v4,(pat),(str),tid);  HADAMARD_BFLY(v5,(pat),(str),tid); \
        HADAMARD_BFLY(v6,(pat),(str),tid);  HADAMARD_BFLY(v7,(pat),(str),tid); \
        HADAMARD_BFLY(v8,(pat),(str),tid);  HADAMARD_BFLY(v9,(pat),(str),tid); \
        HADAMARD_BFLY(v10,(pat),(str),tid); HADAMARD_BFLY(v11,(pat),(str),tid); \
        HADAMARD_BFLY(v12,(pat),(str),tid); HADAMARD_BFLY(v13,(pat),(str),tid); \
        HADAMARD_BFLY(v14,(pat),(str),tid); HADAMARD_BFLY(v15,(pat),(str),tid)
    HBFLY16(0x041F, 1);
    HBFLY16(0x081F, 2);
    HBFLY16(0x101F, 4);
    HBFLY16(0x201F, 8);
    HBFLY16(0x401F, 16);
    #undef HBFLY16

    // --- Pass 4 (reverse): stride 8 --- pairs (0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15)
    float t;
    t = v0;  v0  = v0  + v8;  v8  = t - v8;
    t = v1;  v1  = v1  + v9;  v9  = t - v9;
    t = v2;  v2  = v2  + v10; v10 = t - v10;
    t = v3;  v3  = v3  + v11; v11 = t - v11;
    t = v4;  v4  = v4  + v12; v12 = t - v12;
    t = v5;  v5  = v5  + v13; v13 = t - v13;
    t = v6;  v6  = v6  + v14; v14 = t - v14;
    t = v7;  v7  = v7  + v15; v15 = t - v15;

    // --- Pass 3 (reverse): stride 4 --- pairs (0,4),(1,5),(2,6),(3,7),(8,12),(9,13),(10,14),(11,15)
    t = v0;  v0  = v0  + v4;  v4  = t - v4;
    t = v1;  v1  = v1  + v5;  v5  = t - v5;
    t = v2;  v2  = v2  + v6;  v6  = t - v6;
    t = v3;  v3  = v3  + v7;  v7  = t - v7;
    t = v8;  v8  = v8  + v12; v12 = t - v12;
    t = v9;  v9  = v9  + v13; v13 = t - v13;
    t = v10; v10 = v10 + v14; v14 = t - v14;
    t = v11; v11 = v11 + v15; v15 = t - v15;

    // --- Pass 2 (reverse): stride 2 --- pairs (0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(12,14),(13,15)
    t = v0;  v0  = v0  + v2;  v2  = t - v2;
    t = v1;  v1  = v1  + v3;  v3  = t - v3;
    t = v4;  v4  = v4  + v6;  v6  = t - v6;
    t = v5;  v5  = v5  + v7;  v7  = t - v7;
    t = v8;  v8  = v8  + v10; v10 = t - v10;
    t = v9;  v9  = v9  + v11; v11 = t - v11;
    t = v12; v12 = v12 + v14; v14 = t - v14;
    t = v13; v13 = v13 + v15; v15 = t - v15;

    // --- Pass 1 (reverse): stride 1 --- pairs (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)
    t = v0;  v0  = v0  + v1;  v1  = t - v1;
    t = v2;  v2  = v2  + v3;  v3  = t - v3;
    t = v4;  v4  = v4  + v5;  v5  = t - v5;
    t = v6;  v6  = v6  + v7;  v7  = t - v7;
    t = v8;  v8  = v8  + v9;  v9  = t - v9;
    t = v10; v10 = v10 + v11; v11 = t - v11;
    t = v12; v12 = v12 + v13; v13 = t - v13;
    t = v14; v14 = v14 + v15; v15 = t - v15;

    // Scale by 1/sqrt(512) and apply signs1
    const float s = 0.0441941738f;
    v0  *= s * signs1[d0];    v1  *= s * signs1[d0+1];  v2  *= s * signs1[d0+2];  v3  *= s * signs1[d0+3];
    v4  *= s * signs1[d0+4];  v5  *= s * signs1[d0+5];  v6  *= s * signs1[d0+6];  v7  *= s * signs1[d0+7];
    v8  *= s * signs1[d0+8];  v9  *= s * signs1[d0+9];  v10 *= s * signs1[d0+10]; v11 *= s * signs1[d0+11];
    v12 *= s * signs1[d0+12]; v13 *= s * signs1[d0+13]; v14 *= s * signs1[d0+14]; v15 *= s * signs1[d0+15];
}

// ============================================================================
// Scalar reference FWHT-512 (for self-test / validation only)
// ============================================================================
__device__ void fwht_forward_512(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    for (int i = 0; i < 512; i++) x[i] *= signs1[i];
    for (int stride = 1; stride < 512; stride <<= 1) {
        for (int i = 0; i < 512; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    const float inv_sqrt_512 = 0.0441941738f;
    for (int i = 0; i < 512; i++) x[i] *= inv_sqrt_512 * signs2[i];
}

__device__ void fwht_inverse_512(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    for (int i = 0; i < 512; i++) x[i] *= signs2[i];
    for (int stride = 1; stride < 512; stride <<= 1) {
        for (int i = 0; i < 512; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    const float inv_sqrt_512 = 0.0441941738f;
    for (int i = 0; i < 512; i++) x[i] *= inv_sqrt_512 * signs1[i];
}

// Sign flip array for cheap decorrelation (seed=42, ±1.0)
__constant__ float TURBO_SIGNS1[128] = {
  1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f,
  1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,
 -1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f,
  1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,
  1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,
  1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f,
  1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f
};

