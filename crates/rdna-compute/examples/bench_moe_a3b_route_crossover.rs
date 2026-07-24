//! Qwen3.6 A3B routed-MoE crossover benchmark.
//!
//! Compares the complete route-dependent projection bundle:
//!   Path 1: indexed gate_up + indexed down + combine
//!   Path 2: scatter + grouped gate_up + unscatter + grouped down + combine
//!
//! SwiGLU/FWHT is intentionally omitted because it is identical on both paths.
//! The grouped path launches the same conservative padded-row bound used by the
//! Qwen prefill scheduler, including small speculative-verify batches.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example bench_moe_a3b_route_crossover

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const DIM: usize = 2048;
const MI: usize = 512;
const NUM_EXPERTS: usize = 256;
const K_TOP: usize = 8;
const BLOCK_M: usize = 16;
const MAX_BATCH: usize = 256;

fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

fn grouped_m_total_bound(batch: usize) -> usize {
    let total_slots = batch * K_TOP;
    let live_expert_bound = total_slots.min(NUM_EXPERTS);
    align_up(total_slots + live_expert_bound * (BLOCK_M - 1), BLOCK_M)
}

fn build_hfq4g256_weight(m: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(256));
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 136;
    let mut weight = vec![0u8; m * bytes_per_row];
    for row in 0..m {
        for group in 0..groups_per_row {
            let off = row * bytes_per_row + group * 136;
            weight[off..off + 4].copy_from_slice(&0.02f32.to_le_bytes());
            weight[off + 4..off + 8].copy_from_slice(&0.0f32.to_le_bytes());
            weight[off + 8..off + 136].fill(0x87);
        }
    }
    weight
}

fn upload_bytes(gpu: &mut Gpu, bytes: &[u8]) -> GpuTensor {
    let tensor = gpu
        .alloc_tensor(&[bytes.len()], DType::Raw)
        .expect("allocate byte tensor");
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .expect("upload byte tensor");
    tensor
}

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    upload_bytes(gpu, bytes)
}

fn upload_u64(gpu: &mut Gpu, values: &[u64]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    upload_bytes(gpu, bytes)
}

fn upload_f32(gpu: &mut Gpu, values: &[f32]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    let tensor = gpu
        .alloc_tensor(&[values.len()], DType::F32)
        .expect("allocate f32 tensor");
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .expect("upload f32 tensor");
    tensor
}

fn alloc_f32(gpu: &mut Gpu, elements: usize) -> GpuTensor {
    let tensor = gpu
        .alloc_tensor(&[elements], DType::F32)
        .expect("allocate f32 scratch");
    gpu.hip
        .memset(&tensor.buf, 0, elements * std::mem::size_of::<f32>())
        .expect("zero f32 scratch");
    tensor
}

fn alloc_i32(gpu: &mut Gpu, elements: usize) -> GpuTensor {
    let tensor = gpu
        .alloc_tensor(&[elements * std::mem::size_of::<i32>()], DType::Raw)
        .expect("allocate i32 scratch");
    gpu.hip
        .memset(&tensor.buf, 0, elements * std::mem::size_of::<i32>())
        .expect("zero i32 scratch");
    tensor
}

fn upload_experts(gpu: &mut Gpu, m: usize, k: usize) -> (Vec<GpuTensor>, GpuTensor) {
    let weight = build_hfq4g256_weight(m, k);
    let mut experts = Vec::with_capacity(NUM_EXPERTS);
    let mut pointers = Vec::with_capacity(NUM_EXPERTS);
    for _ in 0..NUM_EXPERTS {
        let tensor = upload_bytes(gpu, &weight);
        pointers.push(tensor.buf.as_ptr() as u64);
        experts.push(tensor);
    }
    let pointer_table = upload_u64(gpu, &pointers);
    (experts, pointer_table)
}

fn time_iterations(gpu: &mut Gpu, iterations: usize, mut launch: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..3 {
        launch(gpu);
    }
    gpu.hip.device_synchronize().expect("warmup synchronize");

    let start = Instant::now();
    for _ in 0..iterations {
        launch(gpu);
    }
    gpu.hip.device_synchronize().expect("timed synchronize");
    start.elapsed().as_secs_f64() * 1.0e6 / iterations as f64
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    if !gpu.arch_caps.has_wmma() {
        println!("SKIP: arch {} does not expose WMMA", gpu.arch);
        return;
    }
    println!(
        "arch={} shape=A3B-MQ4 E={} topk={} gate_up=({}x{}) down=({}x{})",
        gpu.arch,
        NUM_EXPERTS,
        K_TOP,
        2 * MI,
        DIM,
        DIM,
        MI
    );

    let (_gate_up_experts, gate_up_ptrs) = upload_experts(&mut gpu, 2 * MI, DIM);
    let (_down_experts, down_ptrs) = upload_experts(&mut gpu, DIM, MI);

    let max_slots = MAX_BATCH * K_TOP;
    let max_grouped = grouped_m_total_bound(MAX_BATCH);
    let topk: Vec<i32> = (0..MAX_BATCH)
        .flat_map(|token| {
            (0..K_TOP).map(move |rank| ((token * 17 + rank * 31) % NUM_EXPERTS) as i32)
        })
        .collect();
    let topk_indices = upload_i32(&mut gpu, &topk);
    let topk_weights = upload_f32(&mut gpu, &vec![1.0 / K_TOP as f32; max_slots]);
    let x_batch = upload_f32(&mut gpu, &vec![0.125; MAX_BATCH * DIM]);
    let rot_batch = upload_f32(&mut gpu, &vec![0.125; max_slots * MI]);

    let gate_batch = alloc_f32(&mut gpu, max_slots * MI);
    let up_batch = alloc_f32(&mut gpu, max_slots * MI);
    let down_expanded = alloc_f32(&mut gpu, max_slots * DIM);
    let residual_indexed = alloc_f32(&mut gpu, MAX_BATCH * DIM);
    let residual_grouped = alloc_f32(&mut gpu, MAX_BATCH * DIM);

    let expert_token_counts = alloc_i32(&mut gpu, NUM_EXPERTS);
    let expert_offsets = alloc_i32(&mut gpu, NUM_EXPERTS + 1);
    let sorted_slot_index = alloc_i32(&mut gpu, max_grouped);
    let expert_tile_ids = alloc_i32(&mut gpu, max_grouped / BLOCK_M);
    let inverse_perm = alloc_i32(&mut gpu, max_slots);
    let y_gate_up_grouped = alloc_f32(&mut gpu, max_grouped * 2 * MI);
    let y_down_grouped = alloc_f32(&mut gpu, max_grouped * DIM);

    println!(
        "{:>5} {:>7} {:>7} {:>11} {:>11} {:>9}",
        "batch", "slots", "rows", "indexed_us", "grouped_us", "speedup"
    );
    for batch in [1usize, 2, 3, 4, 8, 16, 32, 64, 128, 256] {
        let total_slots = batch * K_TOP;
        let m_total = grouped_m_total_bound(batch);
        let iterations = if batch <= 8 {
            100
        } else if batch <= 32 {
            50
        } else {
            20
        };

        let indexed_us = time_iterations(&mut gpu, iterations, |gpu| {
            gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                &gate_up_ptrs,
                &topk_indices,
                &x_batch,
                &gate_batch,
                &up_batch,
                2 * MI,
                DIM,
                K_TOP,
                batch,
            )
            .expect("indexed gate_up");
            gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                &down_ptrs,
                &topk_indices,
                &rot_batch,
                &down_expanded,
                DIM,
                MI,
                K_TOP,
                batch,
            )
            .expect("indexed down");
            gpu.moe_down_combine_k8_batched(
                &down_expanded,
                &topk_weights,
                &residual_indexed,
                DIM,
                K_TOP,
                batch,
            )
            .expect("indexed combine");
        });

        let grouped_us = time_iterations(&mut gpu, iterations, |gpu| {
            gpu.moe_scatter_fused_k8(
                &topk_indices,
                &expert_token_counts,
                &expert_offsets,
                &sorted_slot_index,
                &expert_tile_ids,
                &inverse_perm,
                total_slots,
                NUM_EXPERTS,
                m_total,
                BLOCK_M,
            )
            .expect("grouped scatter");
            gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
                &gate_up_ptrs,
                &expert_tile_ids,
                &sorted_slot_index,
                &x_batch,
                &y_gate_up_grouped,
                2 * MI,
                DIM,
                K_TOP,
                m_total,
                batch,
            )
            .expect("grouped gate_up");
            gpu.moe_gate_up_unscatter_k8(
                &y_gate_up_grouped,
                &sorted_slot_index,
                &gate_batch,
                &up_batch,
                MI,
                K_TOP,
                m_total,
            )
            .expect("grouped gate_up unscatter");
            gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
                &down_ptrs,
                &expert_tile_ids,
                &sorted_slot_index,
                &rot_batch,
                &y_down_grouped,
                DIM,
                MI,
                1,
                m_total,
                total_slots,
            )
            .expect("grouped down");
            gpu.moe_down_combine_grouped_k8(
                &y_down_grouped,
                &inverse_perm,
                &topk_weights,
                &residual_grouped,
                DIM,
                K_TOP,
                batch,
            )
            .expect("grouped combine");
        });

        println!(
            "{batch:5} {total_slots:7} {m_total:7} {indexed_us:11.1} {grouped_us:11.1} {:8.2}x",
            indexed_us / grouped_us
        );
    }
}
