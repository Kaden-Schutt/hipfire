// MMQ residual variant — mmq_x = 32 (long-prefill regime, b128 LDS path).
#define MMQ_X 32
#define KERNEL_NAME gemm_hfq3g256_residual_mmq_x32
#include "gemm_hfq3g256_residual_mmq_body.cuh"
