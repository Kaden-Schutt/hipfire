#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::drop_non_drop,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::print_literal,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unusual_byte_groupings,
    clippy::useless_vec,
    clippy::unnecessary_cast
)]

//! JIT smoke for the conservative gfx1151 scalar MoE bring-up kernels.
//!
//! Run:
//!   cargo run --release -p hipfire-rdna --example test_moe_mq_gfx1151_scalar_jit

const SRC: &str = include_str!("../../../kernels/src/gfx1151/moe_mq_gfx1151_scalar_batched.hip");

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
    let mut gpu = hipfire_rdna::Gpu::init().expect("Gpu::init");
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
