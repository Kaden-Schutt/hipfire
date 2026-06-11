// HFQ3 qkv MMQ variant — mmq_x = 32 (long-prefill regime, b128 LDS path).
#define MMQ_X 32
#define KERNEL_NAME gemm_qkv_hfq3g256_mmq_x32
#include "gemm_qkv_hfq3g256_mmq_body.cuh"
