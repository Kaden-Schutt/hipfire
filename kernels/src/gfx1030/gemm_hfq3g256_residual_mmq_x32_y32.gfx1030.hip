// MMQ residual variant — mmq_x=32, MMQ_Y=32. Push y64 further.
//
// y64 cut LDS budget from ~26KB to ~15KB (2→4 WG/CU). y32 cuts to ~10KB
// (potentially 6 WG/CU → ~72% theoretical occupancy). Per-WG work is 1/4
// of y128, so 4× more workgroups dispatched. The risk: dispatch overhead
// and reduced latency-hiding margin per WG may exceed the occupancy win.

#define MMQ_X 32
#define MMQ_Y 32
#define KERNEL_NAME gemm_hfq3g256_residual_mmq_x32_y32
#include "gemm_hfq3g256_residual_mmq_body.cuh"
