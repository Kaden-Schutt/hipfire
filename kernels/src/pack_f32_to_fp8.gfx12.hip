#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Activation pre-pass for FP8-WMMA GEMM kernels on gfx12 (RDNA4).
// Converts F32 elements to E4M3 FP8 bytes one-to-one with no scaling.
//
// Rationale: doing the FP16->FP8 conversion inside the GEMM inner loop
// was the dominant ALU bottleneck for the FP8 WMMA path (measured
// 0.7-0.86x FP16 with in-kernel cvt vs 1.05-1.35x FP16 with cvt stubbed
// out at K=2048 on gfx1201). Moving it to a separate, memory-bandwidth-
// bound pre-pass kernel that runs once per layer input (cached by
// src_ptr, same lifetime as ensure_fp16_x's scratch) recovers the FP8
// WMMA throughput advantage.
//
// Saturation: E4M3 max is ±448. Post-RMSNorm transformer activations
// are bounded well below this in practice (typically [-5, 5]); occasional
// outliers in deep residual streams may approach ±20, still safe. We
// rely on cvt_pk_fp8_f32's hardware saturation for the rare outlier.
//
// Layout: each thread converts 16 F32 -> 16 FP8 bytes (16 floats = 64 B
// in, 16 B = 1 dword4 out). Block = 256 threads = 4096 elements per
// block. Grid scaled to (len + 4095) / 4096.

typedef int   __attribute__((ext_vector_type(2))) v2i_t;
typedef float __attribute__((ext_vector_type(4))) v4f_t;

extern "C" __global__ void pack_f32_to_fp8_gfx12(
    const float* __restrict__ in,     // [len], F32
    unsigned char* __restrict__ out,  // [len], E4M3 FP8 (1 byte each)
    int len
) {
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    if (base >= len) return;

    if (base + 16 <= len) {
        // 4x v4f_t load = 16 F32 elements coalesced
        const v4f_t* in_v = (const v4f_t*)(in + base);
        v4f_t a = in_v[0], b = in_v[1], c = in_v[2], d = in_v[3];
        v2i_t out_pack = {0, 0};
        out_pack[0] = __builtin_amdgcn_cvt_pk_fp8_f32(a[0], a[1], out_pack[0], false);
        out_pack[0] = __builtin_amdgcn_cvt_pk_fp8_f32(a[2], a[3], out_pack[0], true);
        out_pack[1] = __builtin_amdgcn_cvt_pk_fp8_f32(b[0], b[1], out_pack[1], false);
        out_pack[1] = __builtin_amdgcn_cvt_pk_fp8_f32(b[2], b[3], out_pack[1], true);
        *(v2i_t*)(out + base) = out_pack;
        v2i_t out_pack2 = {0, 0};
        out_pack2[0] = __builtin_amdgcn_cvt_pk_fp8_f32(c[0], c[1], out_pack2[0], false);
        out_pack2[0] = __builtin_amdgcn_cvt_pk_fp8_f32(c[2], c[3], out_pack2[0], true);
        out_pack2[1] = __builtin_amdgcn_cvt_pk_fp8_f32(d[0], d[1], out_pack2[1], false);
        out_pack2[1] = __builtin_amdgcn_cvt_pk_fp8_f32(d[2], d[3], out_pack2[1], true);
        *(v2i_t*)(out + base + 8) = out_pack2;
    } else {
        // Tail elements: one-by-one.
        for (int k = base; k < len; k++) {
            unsigned int tmp = 0;
            tmp = __builtin_amdgcn_cvt_pk_fp8_f32(in[k], 0.0f, tmp, false);
            out[k] = (unsigned char)(tmp & 0xFFu);
        }
    }
}
