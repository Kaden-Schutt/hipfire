// HFQ3 gate_up MMQ — mmq_x=32, MMQ_Y=96. Sweet-spot occupancy probe.
//
// The y=64 variant lost (kept 50% per-WG work, 4 WG/CU = 48% occupancy)
// and y=128 wins (full per-WG work, 2 WG/CU = 24% occupancy). y=96 tests
// the middle ground:
//   - LDS budget: 96 × 40 × 4 + 96 × 8 + 32 × 36 × 4 = 20736 B (~20 KB)
//   - 3 WGs/CU fit in 64 KB cap → ~36% occupancy
//   - Per-WG work: 75% of y=128 (vs y=64's 50%)
//
// If the residual/gate_up asymmetry is about dispatch amortization vs
// occupancy tradeoff, y=96 should sit at a local optimum for gate_up.

#define MMQ_X 32
#define MMQ_Y 96
#define KERNEL_NAME gemm_gate_up_hfq3g256_mmq_x32_y96
#include "gemm_gate_up_hfq3g256_mmq_body.cuh"
