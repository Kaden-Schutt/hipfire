// HFQ4 gate_up MMQ — mmq_x=16, MMQ_Y=128. Mid-batch tile.

#define MMQ_X 16
#define KERNEL_NAME gemm_gate_up_hfq4g256_mmq_x16
#include "gemm_gate_up_hfq4g256_mmq_body.cuh"
