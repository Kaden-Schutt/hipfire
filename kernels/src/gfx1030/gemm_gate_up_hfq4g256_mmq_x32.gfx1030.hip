// HFQ4 gate_up MMQ — mmq_x=32, MMQ_Y=128. Long-prefill tile.
//
// MMQ_Y stays at 128 (MQ3 phase-2 finding: y64/y96 regress on gate_up
// because per-WG work doubles vs residual, magnifying cache pressure).

#define MMQ_X 32
#define KERNEL_NAME gemm_gate_up_hfq4g256_mmq_x32
#include "gemm_gate_up_hfq4g256_mmq_body.cuh"
