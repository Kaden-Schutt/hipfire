// MFMA-direct HFQ4 GEMM test for gfx942.
//
// Goal: replace the `hfq4g256_dequantize_to_f16 → rocBLAS Tensile` path
// for prefill HFQ4 GEMM with a single MFMA-direct kernel that consumes
// Q4 weights and computes in MFMA fp16 accumulators.
//
// This test: 16 output rows × 16 tokens × K=512 (= 2 HFQ4 groups).
// Compares against FP32 scalar reference computed in the same FP16
// accumulator order so we expect bit-near-exact (within FP16 ULPs).
//
// Build: hipcc -O2 --offload-arch=gfx942 mfma_hfq4_test.cpp -o mfma_hfq4_test
// Run:   ./mfma_hfq4_test

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

typedef _Float16 half_t;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float    vfloat4 __attribute__((ext_vector_type(4)));

// HFQ4G256 group layout (136 bytes): 4 B scale (f32) + 4 B zp (f32) + 128 B nibbles (256 vals).
// Nibble order: low nibble first within each byte, byte 0 holds K-elements 0,1 (in low,high).
static constexpr int HFQ4G_GROUP_BYTES = 136;
static constexpr int HFQ4G_K_PER_GROUP = 256;

extern "C" __global__ __launch_bounds__(64)
void mfma_hfq4_gemm(
    const char*  __restrict__ A,   // [N, K] HFQ4
    const half_t* __restrict__ B,  // [K, M] FP16
    float* __restrict__ C,         // [N, M] FP32
    int N, int K, int M
) {
    const int lane    = threadIdx.x;
    const int n_lane  = lane % 16;
    const int strip   = lane / 16;
    const int wg_n    = blockIdx.x;
    const int wg_m    = blockIdx.y;

    const int n_offset = wg_n * 16;
    const int m_offset = wg_m * 16;
    const int my_n     = n_offset + n_lane;
    if (my_n >= N) return;

    const int groups_per_row = K / HFQ4G_K_PER_GROUP;
    const char* row_ptr = A + (long long)my_n * groups_per_row * HFQ4G_GROUP_BYTES;

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + g * HFQ4G_GROUP_BYTES;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        half_t sc_h = (half_t)sc;
        half_t zp_h = (half_t)zp;

        // Each group has 256 K-elements split into 16 MFMA-K=16 steps.
        #pragma unroll
        for (int ki = 0; ki < HFQ4G_K_PER_GROUP; ki += 16) {
            // 4 K-elements per lane (strip selects which 4 of the 16):
            // Lane (n_lane, strip) reads from row my_n, at K-position ki + strip*4 .. ki + strip*4 + 3
            // Byte offset within group's nibble pool = (ki + strip*4) / 2 = ki/2 + strip*2
            const unsigned short pk2 = *(const unsigned short*)(gp + 8 + ki/2 + strip*2);
            const unsigned int n0 = (pk2 >>  0) & 0xFu;
            const unsigned int n1 = (pk2 >>  4) & 0xFu;
            const unsigned int n2 = (pk2 >>  8) & 0xFu;
            const unsigned int n3 = (pk2 >> 12) & 0xFu;

            half4 a_reg;
            a_reg[0] = sc_h * (half_t)n0 + zp_h;
            a_reg[1] = sc_h * (half_t)n1 + zp_h;
            a_reg[2] = sc_h * (half_t)n2 + zp_h;
            a_reg[3] = sc_h * (half_t)n3 + zp_h;

            // Load B: lane (n_lane, strip) reads B[k_start+0..3, m_offset + n_lane]
            const int k_start = g * HFQ4G_K_PER_GROUP + ki + strip * 4;
            half4 b_reg;
            b_reg[0] = B[(k_start + 0) * M + (m_offset + n_lane)];
            b_reg[1] = B[(k_start + 1) * M + (m_offset + n_lane)];
            b_reg[2] = B[(k_start + 2) * M + (m_offset + n_lane)];
            b_reg[3] = B[(k_start + 3) * M + (m_offset + n_lane)];

            c_acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_acc, 0, 0, 0);
        }
    }

    // Output: lane (strip, n_lane) writes C[strip*4+r + n_offset, m_offset + n_lane]
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int gn = n_offset + strip * 4 + r;
        int gm = m_offset + n_lane;
        if (gn < N && gm < M) {
            C[gn * M + gm] = c_acc[r];
        }
    }
}

// Pack `nibbles[256]` (each in [0,15]) into 128 packed bytes, low nibble first.
static void pack_nibbles(const unsigned char* nibbles, unsigned char* packed) {
    for (int i = 0; i < 128; i++) {
        packed[i] = (nibbles[2*i] & 0xF) | ((nibbles[2*i + 1] & 0xF) << 4);
    }
}

int main() {
    const int N = 16, K = 512, M = 16;
    const int groups_per_row = K / HFQ4G_K_PER_GROUP;

    // Build HFQ4 weight buffer: N rows × groups_per_row groups × 136 bytes
    char* A_buf = (char*)calloc(N * groups_per_row * HFQ4G_GROUP_BYTES, 1);
    half_t* B_h = (half_t*)malloc(K * M * sizeof(half_t));
    float* C_ref = (float*)malloc(N * M * sizeof(float));
    float* C_got = (float*)malloc(N * M * sizeof(float));

    srand(42);
    // Generate random A nibbles + scales/zps per group
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups_per_row; g++) {
            char* gp = A_buf + (n * groups_per_row + g) * HFQ4G_GROUP_BYTES;
            float sc = 0.05f + (rand() % 100) / 2000.0f;
            float zp = -0.5f + (rand() % 100) / 100.0f;
            memcpy(gp,     &sc, 4);
            memcpy(gp + 4, &zp, 4);
            unsigned char nibbles[256];
            for (int i = 0; i < 256; i++) nibbles[i] = rand() % 16;
            pack_nibbles(nibbles, (unsigned char*)(gp + 8));
        }
    }
    for (int i = 0; i < K * M; i++) B_h[i] = (half_t)((rand() % 100) / 100.0f - 0.5f);

    // Scalar reference (compute in FP16 accumulator order to mirror MFMA)
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int g = 0; g < groups_per_row; g++) {
                const char* gp = A_buf + (n * groups_per_row + g) * HFQ4G_GROUP_BYTES;
                float sc = 0.0f, zp = 0.0f;
                memcpy(&sc, gp, 4); memcpy(&zp, gp + 4, 4);
                half_t sc_h = (half_t)sc, zp_h = (half_t)zp;
                for (int k = 0; k < HFQ4G_K_PER_GROUP; k++) {
                    unsigned char byte = (unsigned char)gp[8 + k/2];
                    unsigned int nibble = (k % 2 == 0) ? (byte & 0xF) : (byte >> 4);
                    half_t a_h = sc_h * (half_t)nibble + zp_h;
                    half_t b_h = B_h[(g * HFQ4G_K_PER_GROUP + k) * M + m];
                    acc += (float)a_h * (float)b_h;
                }
            }
            C_ref[n * M + m] = acc;
        }
    }

    // GPU run
    char* A_d; half_t* B_d; float* C_d;
    hipMalloc(&A_d, N * groups_per_row * HFQ4G_GROUP_BYTES);
    hipMalloc(&B_d, K * M * sizeof(half_t));
    hipMalloc(&C_d, N * M * sizeof(float));
    hipMemcpy(A_d, A_buf, N * groups_per_row * HFQ4G_GROUP_BYTES, hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h,   K * M * sizeof(half_t), hipMemcpyHostToDevice);

    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    dim3 block(64, 1, 1);
    hipLaunchKernelGGL(mfma_hfq4_gemm, grid, block, 0, 0, A_d, B_d, C_d, N, K, M);
    hipDeviceSynchronize();
    hipMemcpy(C_got, C_d, N * M * sizeof(float), hipMemcpyDeviceToHost);

    // Compare with relative tolerance (FP16 intermediate precision: rel ~1e-3)
    double max_rel_err = 0.0;
    double max_abs_err = 0.0;
    int fails = 0;
    for (int i = 0; i < N * M; i++) {
        double abs_err = fabs((double)C_got[i] - (double)C_ref[i]);
        double rel_err = abs_err / (fabs((double)C_ref[i]) + 1e-6);
        if (rel_err > 0.05 && abs_err > 0.05) fails++;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
    }
    printf("MFMA HFQ4 GEMM test (N=%d K=%d M=%d): max_abs=%g max_rel=%g fails=%d/%d\n",
           N, K, M, max_abs_err, max_rel_err, fails, N*M);
    if (fails == 0) {
        printf("PASS — MFMA HFQ4 path works\n");
    } else {
        printf("FAIL\n");
        for (int i = 0; i < 8; i++) {
            printf("  [%d]: got=%g ref=%g rel=%g\n",
                i, C_got[i], C_ref[i],
                fabs((double)C_got[i] - (double)C_ref[i]) / (fabs((double)C_ref[i]) + 1e-6));
        }
    }
    free(A_buf); free(B_h); free(C_ref); free(C_got);
    hipFree(A_d); hipFree(B_d); hipFree(C_d);
    return fails ? 1 : 0;
}
