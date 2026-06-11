// MMQ residual variant — mmq_x=32, MMQ_Y=64. Experimental occupancy probe.
//
// Halves the row tile (128 → 64) to cut per-WG LDS budget from ~26 KB
// to ~15 KB. At 64 KB LDS/CU, 4 WGs fit per CU instead of 2 → 16 waves/CU
// instead of 8 (~48% occupancy vs ~24%). Per-WG work also halves, so 2×
// more workgroups dispatched. Test whether the latency-hiding win from
// higher occupancy outweighs the per-WG-work-halving overhead.

#define MMQ_X 32
#define MMQ_Y 64
#define KERNEL_NAME gemm_hfq3g256_residual_mmq_x32_y64
#include "gemm_hfq3g256_residual_mmq_body.cuh"
