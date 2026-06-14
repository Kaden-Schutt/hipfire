// Minimal MFMA F32_16x16x16_F16 test for gfx942.
//
// Verifies the MFMA intrinsic + lane layout in isolation before building
// the full HFQ4 MFMA GEMM kernel. Computes C[16,16] = A[16,16] * B[16,16]
// (single tile, no K loop iterations) and compares against scalar reference.
//
// Build: hipcc -O2 --offload-arch=gfx942 mfma_test.cpp -o mfma_test
// Run:   ./mfma_test

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

typedef _Float16 half_t;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float    vfloat4 __attribute__((ext_vector_type(4)));

// Output layout for MFMA F32_16x16x16:
// Lane l ∈ [0,63]: holds c[r] = C[m = l % 16, n = (l / 16) * 4 + r] for r ∈ [0,3]
// Input layouts (mirror):
//   a[r] = A[m = l % 16, k = (l / 16) * 4 + r]
//   b[r] = B[k = (l / 16) * 4 + r, n = l % 16]

extern "C" __global__ __launch_bounds__(64)
void mfma_16x16x16_test(
    const half_t* __restrict__ A,    // [M=16, K=16]
    const half_t* __restrict__ B,    // [K=16, N=16]
    float* __restrict__ C            // [M=16, N=16]
) {
    const int lane = threadIdx.x;
    // Input layout (per AMD CDNA3 F32_16x16x16 reference):
    //   A operand: lane l holds a[r] = A[m = l%16,   k = (l/16)*4 + r]
    //   B operand: lane l holds b[r] = B[k = (l/16)*4 + r,   n = l%16]
    // Output layout (different — strip-major over m):
    //   C/D output: lane l holds c[r] = C[m = (l/16)*4 + r,   n = l%16]
    const int n_lane = lane % 16;            // shared by A.m, B.n, C.n
    const int strip  = lane / 16;            // 0..3 — strip selector

    half4 a_reg;
    half4 b_reg;
    for (int r = 0; r < 4; r++) {
        a_reg[r] = A[n_lane * 16 + (strip * 4 + r)];   // A[m=n_lane, k=strip*4+r]
        b_reg[r] = B[(strip * 4 + r) * 16 + n_lane];   // B[k=strip*4+r, n=n_lane]
    }

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};
    c_acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_acc, 0, 0, 0);

    // Output: C[m = strip*4 + r, n = n_lane] = c_acc[r]
    for (int r = 0; r < 4; r++) {
        C[(strip * 4 + r) * 16 + n_lane] = c_acc[r];
    }
}

int main() {
    constexpr int M = 16, N = 16, K = 16;
    half_t A_h[M*K], B_h[K*N];
    float C_h[M*N], C_ref[M*N];

    srand(42);
    for (int i = 0; i < M*K; i++) A_h[i] = (half_t)((rand() % 100) / 100.0f - 0.5f);
    for (int i = 0; i < K*N; i++) B_h[i] = (half_t)((rand() % 100) / 100.0f - 0.5f);

    // Scalar reference
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += (float)A_h[m*K + k] * (float)B_h[k*N + n];
            }
            C_ref[m*N + n] = sum;
        }
    }

    half_t *A_d, *B_d;
    float *C_d;
    hipMalloc(&A_d, M*K*sizeof(half_t));
    hipMalloc(&B_d, K*N*sizeof(half_t));
    hipMalloc(&C_d, M*N*sizeof(float));
    hipMemcpy(A_d, A_h, M*K*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h, K*N*sizeof(half_t), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(mfma_16x16x16_test, dim3(1), dim3(64), 0, 0, A_d, B_d, C_d);
    hipDeviceSynchronize();
    hipMemcpy(C_h, C_d, M*N*sizeof(float), hipMemcpyDeviceToHost);

    // Compare
    double max_err = 0.0;
    int fails = 0;
    for (int i = 0; i < M*N; i++) {
        double err = fabs((double)C_h[i] - (double)C_ref[i]);
        if (err > 0.01) fails++;
        if (err > max_err) max_err = err;
    }
    printf("MFMA 16x16x16 test: max_err=%g fails=%d/%d\n", max_err, fails, M*N);
    if (fails == 0) {
        printf("PASS — MFMA path works; lane layout correct\n");
    } else {
        printf("FAIL — layout or intrinsic issue. First few:\n");
        for (int i = 0; i < 8; i++) {
            printf("  C[%d]: got=%g ref=%g\n", i, C_h[i], C_ref[i]);
        }
    }
    hipFree(A_d); hipFree(B_d); hipFree(C_d);
    return fails ? 1 : 0;
}
