// HFQ3 qkv MMQ variant — mmq_x = 8 (short-prefill regime).
#define MMQ_X 8
#define KERNEL_NAME gemm_qkv_hfq3g256_mmq_x8
#include "gemm_qkv_hfq3g256_mmq_body.cuh"
