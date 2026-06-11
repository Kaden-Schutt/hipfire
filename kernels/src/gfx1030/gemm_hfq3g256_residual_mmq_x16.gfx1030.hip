// MMQ residual variant — mmq_x = 16 (mid-prefill regime).
#define MMQ_X 16
#define KERNEL_NAME gemm_hfq3g256_residual_mmq_x16
#include "gemm_hfq3g256_residual_mmq_body.cuh"
