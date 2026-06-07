// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! JIT smoke for the conservative gfx1151 scalar MoE bring-up kernels.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_mq_gfx1151_scalar_jit

const SRC: &str = include_str!("../../../kernels/src/moe_mq_gfx1151_scalar_batched.hip");

const KERNELS: &[&str] = &[
    "gemm_gate_up_hfq2g256_scalar_batched",
    "gemv_hfq2g256_residual_sigmoid_scaled_gpu_batched",
    "gemv_hfq2g256_moe_gate_up_k8_indexed_batched",
    "gemv_hfq2g256_moe_down_k8_indexed_batched_expanded",
    "gemm_gate_up_hfq8g256_scalar_batched",
    "gemv_hfq8g256_residual_sigmoid_scaled_gpu_batched",
    "gemv_hfq8g256_moe_gate_up_k8_indexed_batched",
    "gemv_hfq8g256_moe_down_k8_indexed_batched_expanded",
    "gemm_gate_up_mq2g256_lloyd_scalar_batched",
    "gemv_mq2g256_lloyd_residual_sigmoid_scaled_gpu_batched",
    "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched",
    "gemv_mq2g256_lloyd_moe_down_k8_indexed_batched_expanded",
    "gemm_gate_up_mq3g256_lloyd_scalar_batched",
    "gemv_mq3g256_lloyd_residual_sigmoid_scaled_gpu_batched",
    "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched",
    "gemv_mq3g256_lloyd_moe_down_k8_indexed_batched_expanded",
];

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
    if gpu.arch != "gfx1151" {
        eprintln!(
            "SKIP: scalar MoE bring-up is admitted only on gfx1151; arch={}",
            gpu.arch
        );
        return;
    }
    for &name in KERNELS {
        gpu.ensure_kernel_public(name, SRC, name)
            .unwrap_or_else(|err| panic!("failed to JIT {name}: {err}"));
    }
    eprintln!(
        "PASS: JIT compiled {} gfx1151 scalar MoE kernels",
        KERNELS.len()
    );
}
