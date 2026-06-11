// MMQ residual variant — mmq_x=32, MMQ_Y=64. Applies MQ3 phase-2 finding.
//
// On HFQ3 RDNA2 the y=64 variant cuts per-WG LDS budget (~26 KB → ~15 KB),
// doubles theoretical CU occupancy (24% → 48%), and won +8% kernel-level
// over y=128. HFQ4 has the same kernel topology and LDS layout, so the
// same occupancy mechanism applies. Microbench will confirm whether the
// magnitude transfers (HFQ4 has a cheaper unpack so the relative
// memory-bound-vs-compute-bound balance differs slightly).

#define MMQ_X 32
#define MMQ_Y 64
#define KERNEL_NAME gemm_hfq4g256_residual_mmq_x32_y64
#include "gemm_hfq4g256_residual_mmq_body.cuh"
