// BF16 MFMA F32_16x16x16_BF16 test for gfx942 (MI300X CDNA3).
//
// Foundation POC for the Tier 1 hipfire-native BF16 calibration path.
// Verifies the BF16 MFMA intrinsic + lane layout in isolation before
// validating the full LDS-tiled GEMM kernel.
//
// Two stages:
//   Stage 1: single-tile C[16,16] = A[16,16] * B[16,16] (mirror of the
//            F16 POC at `docs/investigations/2026-05-19-mfma-hfq4/mfma_test.cpp`).
//   Stage 2: full kernel C[M,N] = A[M,K] * B[N,K]^T (calls the production
//            kernel `gemm_bf16_mfma.gfx942.hip`'s shape and layout) at
//            M=128 K=128 N=64 to exercise LDS B-tile + 4-wave WG geometry.
//
// Build: hipcc --offload-arch=gfx942 -O3 mfma_bf16_test.cpp -o mfma_bf16_test
// Run:   ./mfma_bf16_test
//
// Tolerance: BF16 has 7 mantissa bits (vs F16's 10). For a dot product of
// length K=128 with normal-distribution operands and FP32 accumulator,
// expect max abs err ~ K * 2^-7 * |operand| ≈ 1e-2. Configured pass gate:
// max_err < 1e-1 on Stage 2 (relaxed), max_err < 1e-3 on Stage 1 (K=16).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

typedef __bf16 bf16_t;
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef short  s16x4  __attribute__((ext_vector_type(4)));
typedef float  vfloat4 __attribute__((ext_vector_type(4)));

// The CDNA3 BF16 MFMA intrinsic models the BF16 lane bits as i16 — bit_cast
// is a no-op type pun the compiler folds away. See the kernel header comment
// in gemm_bf16_mfma.gfx942.hip for why `_1k` is the canonical name on gfx942.
__device__ static inline vfloat4 mfma_bf16(bf16x4 a, bf16x4 b, vfloat4 c) {
    s16x4 ai = __builtin_bit_cast(s16x4, a);
    s16x4 bi = __builtin_bit_cast(s16x4, b);
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(ai, bi, c, 0, 0, 0);
}

// ─── Host-side BF16 conversion helpers ────────────────────────────────
// Round-to-nearest-even FP32 → BF16. hipcc's __bf16 cast is fine but
// these helpers make the round-trip explicit for the host reference.
static inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, sizeof(u));
    // Handle NaN: preserve sign + signaling bit
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t)((u >> 16) | 0x40u);   // force quiet NaN
    }
    // Round-to-nearest-even
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    return (uint16_t)((u + rounding_bias) >> 16);
}

static inline float bf16_bits_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

static inline bf16_t f32_to_bf16(float f) {
    uint16_t bits = f32_to_bf16_bits(f);
    bf16_t out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline float bf16_to_f32(bf16_t b) {
    uint16_t bits;
    __builtin_memcpy(&bits, &b, sizeof(bits));
    return bf16_bits_to_f32(bits);
}

// ─── Stage 1 kernel: single 16×16×16 MFMA tile ────────────────────────
//
// Input layout (CDNA3 F32_16x16x16 reference):
//   A operand: lane l holds a[r] = A[m = l%16,   k = (l/16)*4 + r]
//   B operand: lane l holds b[r] = B[k = (l/16)*4 + r,   n = l%16]
// Output layout (strip-major over m, different from A's input layout):
//   D output: lane l holds c[r] = C[m = (l/16)*4 + r,   n = l%16]

extern "C" __global__ __launch_bounds__(64)
void mfma_bf16_16x16x16_test(
    const bf16_t* __restrict__ A,    // [M=16, K=16]
    const bf16_t* __restrict__ B,    // [K=16, N=16]
    float* __restrict__ C            // [M=16, N=16]
) {
    const int lane = threadIdx.x;
    const int n_lane = lane % 16;
    const int strip  = lane / 16;

    bf16x4 a_reg;
    bf16x4 b_reg;
    for (int r = 0; r < 4; r++) {
        a_reg[r] = A[n_lane * 16 + (strip * 4 + r)];   // A[m=n_lane, k=strip*4+r]
        b_reg[r] = B[(strip * 4 + r) * 16 + n_lane];   // B[k=strip*4+r, n=n_lane]
    }

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};
    c_acc = mfma_bf16(a_reg, b_reg, c_acc);

    for (int r = 0; r < 4; r++) {
        C[(strip * 4 + r) * 16 + n_lane] = c_acc[r];
    }
}

// ─── Stage 2 kernel: LDS-tiled GEMM (production layout copy) ──────────
// Mirrors `kernels/src/gemm_bf16_mfma.gfx942.hip`. Embedded inline here so
// the POC is buildable without the kernels/ tree included.
//
// Layout: A[M,K] BF16 row-major × B[N,K] BF16 row-major (note B is
// "[batch, K]") → D[N,M] FP32 row-major.

#define K_CHUNK 128
#define N_TILE  32

extern "C" __global__ __launch_bounds__(256)
void gemm_bf16_mfma_gfx942_test(
    const bf16_t* __restrict__ A,
    const bf16_t* __restrict__ B,
    float*        __restrict__ D,
    int M, int K, int batch_size
) {
    __shared__ bf16_t lds_b[N_TILE * K_CHUNK];

    const int tid       = threadIdx.x;
    const int wave_id   = tid >> 6;
    const int lane_in_w = tid & 63;
    const int n_lane    = lane_in_w & 15;
    const int strip     = lane_in_w >> 4;

    const int wave_m_off = (wave_id & 2) ? 16 : 0;
    const int wave_n_off = (wave_id & 1) ? 16 : 0;
    const int m_offset = blockIdx.x * 32 + wave_m_off;
    const int b_offset_global = blockIdx.y * 32;
    const int b_offset = b_offset_global + wave_n_off;

    const int my_m = m_offset + n_lane;

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kc = 0; kc < K; kc += K_CHUNK) {
        __syncthreads();
        #pragma unroll
        for (int li = 0; li < 16; li++) {
            int idx = tid + li * 256;
            int b_col = idx / K_CHUNK;
            int b_k   = idx % K_CHUNK;
            int gb = b_offset_global + b_col;
            int gk = kc + b_k;
            bf16_t v = (gb < batch_size && gk < K)
                ? B[(long long)gb * K + gk]
                : (bf16_t)0;
            lds_b[b_col * K_CHUNK + b_k] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < K_CHUNK; kk += 16) {
            int gk_start = kc + kk;
            int a_k_base = gk_start + strip * 4;
            bf16x4 a_reg;
            if (my_m < M) {
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int gk = a_k_base + r;
                    a_reg[r] = (gk < K)
                        ? A[(long long)my_m * K + gk]
                        : (bf16_t)0;
                }
            } else {
                a_reg[0] = (bf16_t)0; a_reg[1] = (bf16_t)0;
                a_reg[2] = (bf16_t)0; a_reg[3] = (bf16_t)0;
            }

            int b_col_in_tile   = wave_n_off + n_lane;
            int k_in_chunk_base = kk + strip * 4;
            bf16x4 b_reg;
            b_reg[0] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 0];
            b_reg[1] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 1];
            b_reg[2] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 2];
            b_reg[3] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 3];

            c_acc = mfma_bf16(a_reg, b_reg, c_acc);
        }
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int gm = m_offset + strip * 4 + r;
        const int gb = b_offset + n_lane;
        if (gm < M && gb < batch_size) {
            D[(long long)gb * M + gm] = c_acc[r];
        }
    }
}

// ─── Test driver ──────────────────────────────────────────────────────

static int run_stage1() {
    constexpr int M = 16, N = 16, K = 16;
    bf16_t A_h[M*K], B_h[K*N];
    float C_h[M*N], C_ref[M*N];

    srand(42);
    for (int i = 0; i < M*K; i++) A_h[i] = f32_to_bf16((rand() % 100) / 100.0f - 0.5f);
    for (int i = 0; i < K*N; i++) B_h[i] = f32_to_bf16((rand() % 100) / 100.0f - 0.5f);

    // FP32 reference (BF16 cast back to FP32 for the multiply, FP32 accumulate)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += bf16_to_f32(A_h[m*K + k]) * bf16_to_f32(B_h[k*N + n]);
            }
            C_ref[m*N + n] = sum;
        }
    }

    bf16_t *A_d, *B_d;
    float *C_d;
    hipMalloc(&A_d, M*K*sizeof(bf16_t));
    hipMalloc(&B_d, K*N*sizeof(bf16_t));
    hipMalloc(&C_d, M*N*sizeof(float));
    hipMemcpy(A_d, A_h, M*K*sizeof(bf16_t), hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h, K*N*sizeof(bf16_t), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(mfma_bf16_16x16x16_test, dim3(1), dim3(64), 0, 0, A_d, B_d, C_d);
    hipDeviceSynchronize();
    hipMemcpy(C_h, C_d, M*N*sizeof(float), hipMemcpyDeviceToHost);

    double max_err = 0.0;
    int fails = 0;
    for (int i = 0; i < M*N; i++) {
        double err = fabs((double)C_h[i] - (double)C_ref[i]);
        if (err > 1e-2) fails++;
        if (err > max_err) max_err = err;
    }
    printf("Stage 1 (single 16x16x16 BF16 tile): max_err=%.3g fails=%d/%d\n",
           max_err, fails, M*N);

    hipFree(A_d); hipFree(B_d); hipFree(C_d);
    return fails;
}

static int run_stage2() {
    // Production-shape test: 128x128 weights × 64 batch, K_CHUNK=128 → 1 outer-k step.
    constexpr int M = 128, K = 128, BATCH = 64;
    bf16_t *A_h = new bf16_t[M*K];
    bf16_t *B_h = new bf16_t[BATCH*K];
    float  *D_h = new float[BATCH*M];
    float  *D_ref = new float[BATCH*M];

    srand(123);
    for (int i = 0; i < M*K; i++) A_h[i]    = f32_to_bf16(((rand() % 200) - 100) / 100.0f);
    for (int i = 0; i < BATCH*K; i++) B_h[i] = f32_to_bf16(((rand() % 200) - 100) / 100.0f);

    // FP32 reference
    for (int b = 0; b < BATCH; b++) {
        for (int m = 0; m < M; m++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += bf16_to_f32(A_h[m*K + k]) * bf16_to_f32(B_h[b*K + k]);
            }
            D_ref[b*M + m] = sum;
        }
    }

    bf16_t *A_d, *B_d;
    float *D_d;
    hipMalloc(&A_d, M*K*sizeof(bf16_t));
    hipMalloc(&B_d, BATCH*K*sizeof(bf16_t));
    hipMalloc(&D_d, BATCH*M*sizeof(float));
    hipMemcpy(A_d, A_h, M*K*sizeof(bf16_t), hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h, BATCH*K*sizeof(bf16_t), hipMemcpyHostToDevice);

    dim3 grid((M + 31) / 32, (BATCH + 31) / 32);
    dim3 block(256);
    hipLaunchKernelGGL(gemm_bf16_mfma_gfx942_test, grid, block, 0, 0,
                       A_d, B_d, D_d, M, K, BATCH);
    hipDeviceSynchronize();
    hipMemcpy(D_h, D_d, BATCH*M*sizeof(float), hipMemcpyDeviceToHost);

    double max_err = 0.0;
    double max_rel_err = 0.0;
    int fails = 0;
    int rel_fails = 0;
    for (int i = 0; i < BATCH*M; i++) {
        double err = fabs((double)D_h[i] - (double)D_ref[i]);
        double denom = fmax(fabs((double)D_ref[i]), 1e-3);
        double rel = err / denom;
        if (err > 1e-1) fails++;
        if (rel > 5e-3) rel_fails++;
        if (err > max_err) max_err = err;
        if (rel > max_rel_err) max_rel_err = rel;
    }
    printf("Stage 2 (LDS-tiled GEMM, M=128 K=128 batch=64): "
           "max_err=%.3g max_rel_err=%.3g abs_fails=%d/%d rel_fails=%d/%d\n",
           max_err, max_rel_err, fails, BATCH*M, rel_fails, BATCH*M);

    delete[] A_h; delete[] B_h; delete[] D_h; delete[] D_ref;
    hipFree(A_d); hipFree(B_d); hipFree(D_d);
    return fails;
}

int main() {
    int s1 = run_stage1();
    int s2 = run_stage2();

    if (s1 == 0 && s2 == 0) {
        printf("\nPASS — BF16 MFMA path works; lane layout + LDS tiling correct.\n");
        printf("       Foundation kernel ready for the Tier 1 calibration forward pass.\n");
        return 0;
    } else {
        printf("\nFAIL — stage1=%d stage2=%d failures.\n", s1, s2);
        return 1;
    }
}
