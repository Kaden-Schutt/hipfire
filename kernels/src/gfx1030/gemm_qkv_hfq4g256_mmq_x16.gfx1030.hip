// HFQ4 qkv MMQ — mmq_x=16, MMQ_Y=128. Mid-batch tile.

#define MMQ_X 16
#define KERNEL_NAME gemm_qkv_hfq4g256_mmq_x16
#include "gemm_qkv_hfq4g256_mmq_body.cuh"
