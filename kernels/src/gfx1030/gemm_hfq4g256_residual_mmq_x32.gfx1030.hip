// MMQ residual variant — mmq_x=32, MMQ_Y=128. Default long-prefill tile.
//
// Replaces the prior standalone `gemm_hfq4g256_residual_mmq_rdna2`
// probe with the templated body; identical kernel semantics.

#define MMQ_X 32
#define KERNEL_NAME gemm_hfq4g256_residual_mmq_x32
#include "gemm_hfq4g256_residual_mmq_body.cuh"
