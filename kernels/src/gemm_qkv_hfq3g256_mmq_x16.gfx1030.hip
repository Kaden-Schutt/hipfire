// HFQ3 qkv MMQ variant — mmq_x = 16 (mid-prefill regime).
#define MMQ_X 16
#define KERNEL_NAME gemm_qkv_hfq3g256_mmq_x16
#include "gemm_qkv_hfq3g256_mmq_body.cuh"
