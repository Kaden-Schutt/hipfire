#include "gemm_hfq4g256_residual_mmq_gfx906_body.cuh"

#define MMQ_X_VAL 16

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_hfq4g256_residual_mmq_gfx906_x16(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int add
) {
    mmq_body_templated<MMQ_X_VAL, true, -1>(A, Xq, Y, M, K, N, add);
}

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_hfq4g256_residual_mmq_gfx906_full_add_x16(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int /*add*/
) {
    mmq_body_templated<MMQ_X_VAL, false, 1>(A, Xq, Y, M, K, N, 1);
}

extern "C" __global__ __launch_bounds__(256, 2)
void gemm_hfq4g256_residual_mmq_gfx906_full_set_x16(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N, int /*add*/
) {
    mmq_body_templated<MMQ_X_VAL, false, 0>(A, Xq, Y, M, K, N, 0);
}
