// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact-gfx1201 DS4 TP/EP MQ2-Lloyd channel.

use rdna_compute::{Gpu, GpuTensor};

const TOP_K: usize = 6;
const GATE_M: usize = 4096;
const GATE_K: usize = 4096;
const DOWN_M: usize = 4096;
const DOWN_K: usize = 2048;
const GROUP_BYTES: usize = 72;
const OWNED: [bool; TOP_K] = [true, false, false, true, false, false];

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn upload_i32(gpu: &Gpu, values: &[i32]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.upload_raw(bytes, &[bytes.len()]).expect("upload i32")
}

fn upload_u64(gpu: &Gpu, values: &[u64]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.upload_raw(bytes, &[bytes.len()]).expect("upload u64")
}

fn build_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    const CODEBOOK: [u16; 4] = [0xbc00, 0xb400, 0x3400, 0x3c00];
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * GROUP_BYTES;
    let mut state = seed;
    let mut packed = vec![0u8; m * row_bytes];
    for row in 0..m {
        for group in 0..groups_per_row {
            let base = row * row_bytes + group * GROUP_BYTES;
            for (slot, bits) in CODEBOOK.iter().enumerate() {
                packed[base + 2 * slot..base + 2 * slot + 2].copy_from_slice(&bits.to_le_bytes());
            }
            for byte in &mut packed[base + 8..base + GROUP_BYTES] {
                *byte = lcg(&mut state) as u8;
            }
        }
    }
    packed
}

fn build_signal(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|i| match i & 31 {
            0 => 0.0,
            1 => -0.0,
            2 => 0.125,
            3 => -0.25,
            _ => (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.5,
        })
        .collect()
}

fn assert_raw_equal(label: &str, reference: &[f32], observed: &[f32]) {
    assert_eq!(reference.len(), observed.len(), "{label} length");
    let mut mismatches = 0usize;
    let mut first = None;
    for (index, (&expected, &actual)) in reference.iter().zip(observed).enumerate() {
        if expected.to_bits() != actual.to_bits() {
            mismatches += 1;
            first.get_or_insert((index, expected.to_bits(), actual.to_bits()));
        }
    }
    println!(
        "CHANNEL {label} values={} raw_bit_mismatches={mismatches} first={first:?}",
        reference.len()
    );
    assert_eq!(mismatches, 0, "{label} raw-bit mismatch: {first:?}");
}

fn assert_nonowned_positive_zero(label: &str, values: &[f32], rows: usize) {
    for (slot, &owned) in OWNED.iter().enumerate() {
        if owned {
            continue;
        }
        let bad = values[slot * rows..(slot + 1) * rows]
            .iter()
            .position(|value| value.to_bits() != 0.0f32.to_bits());
        assert!(bad.is_none(), "{label} slot {slot} has non-+0 at {bad:?}");
    }
}

fn elapsed_ms<F>(gpu: &mut Gpu, repeats: usize, mut launch: F) -> f32
where
    F: FnMut(&mut Gpu),
{
    let start = gpu.hip.event_create().expect("event start");
    let stop = gpu.hip.event_create().expect("event stop");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..repeats {
        launch(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("wait stop");
    let elapsed = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") / repeats as f32;
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    elapsed
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "this channel requires exact gfx1201");

    let gate_owned0 = gpu
        .upload_raw(
            &build_weights(GATE_M, GATE_K, 0x1201_0000),
            &[GATE_M, GATE_K],
        )
        .expect("upload gate owned0");
    let gate_owned3 = gpu
        .upload_raw(
            &build_weights(GATE_M, GATE_K, 0x1201_0003),
            &[GATE_M, GATE_K],
        )
        .expect("upload gate owned3");
    let gate_row_bytes = (GATE_K / 256) * GROUP_BYTES;
    let dummy = gpu
        .zeros(&[GATE_M * gate_row_bytes / 4], rdna_compute::DType::F32)
        .expect("zero gate dummy");
    let gate_ptr_values = [
        gate_owned0.buf.as_ptr() as usize as u64,
        dummy.buf.as_ptr() as usize as u64,
        dummy.buf.as_ptr() as usize as u64,
        gate_owned3.buf.as_ptr() as usize as u64,
        dummy.buf.as_ptr() as usize as u64,
        dummy.buf.as_ptr() as usize as u64,
    ];
    let gate_ptrs = upload_u64(&gpu, &gate_ptr_values);
    let topk = upload_i32(&gpu, &[0, 1, 2, 3, 4, 5]);
    let x = gpu
        .upload_f32(&build_signal(GATE_K, 0x1201_1010), &[GATE_K])
        .expect("upload x");
    let base_gate = gpu
        .alloc_tensor(&[TOP_K, GATE_M / 2], rdna_compute::DType::F32)
        .expect("base gate");
    let base_up = gpu
        .alloc_tensor(&[TOP_K, GATE_M / 2], rdna_compute::DType::F32)
        .expect("base up");
    let candidate_gate = gpu
        .alloc_tensor(&[TOP_K, GATE_M / 2], rdna_compute::DType::F32)
        .expect("candidate gate");
    let candidate_up = gpu
        .alloc_tensor(&[TOP_K, GATE_M / 2], rdna_compute::DType::F32)
        .expect("candidate up");

    gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        &gate_ptrs, &topk, &x, &base_gate, &base_up, GATE_M, GATE_K, TOP_K,
    )
    .expect("gate baseline");
    gpu.try_gfx1201()
        .expect("exact gfx1201")
        .mq2_lloyd_moe_gate_up_ep(
            &gate_ptrs,
            &dummy,
            &topk,
            &x,
            &candidate_gate,
            &candidate_up,
            GATE_M,
            GATE_K,
            TOP_K,
        )
        .expect("gate candidate");

    let down_owned0 = gpu
        .upload_raw(
            &build_weights(DOWN_M, DOWN_K, 0x1201_2000),
            &[DOWN_M, DOWN_K],
        )
        .expect("upload down owned0");
    let down_owned3 = gpu
        .upload_raw(
            &build_weights(DOWN_M, DOWN_K, 0x1201_2003),
            &[DOWN_M, DOWN_K],
        )
        .expect("upload down owned3");
    let down_ptr_values = [
        down_owned0.buf.as_ptr() as usize as u64,
        down_owned0.buf.as_ptr() as usize as u64,
        down_owned0.buf.as_ptr() as usize as u64,
        down_owned3.buf.as_ptr() as usize as u64,
        down_owned0.buf.as_ptr() as usize as u64,
        down_owned0.buf.as_ptr() as usize as u64,
    ];
    let down_ptrs = upload_u64(&gpu, &down_ptr_values);
    let mut rot_host = vec![0.0f32; TOP_K * DOWN_K];
    for slot in [0usize, 3] {
        rot_host[slot * DOWN_K..(slot + 1) * DOWN_K]
            .copy_from_slice(&build_signal(DOWN_K, 0x1201_3000 + slot as u64));
    }
    let rot = gpu
        .upload_f32(&rot_host, &[TOP_K, DOWN_K])
        .expect("upload rotated batch");
    let base_down = gpu
        .alloc_tensor(&[TOP_K, DOWN_M], rdna_compute::DType::F32)
        .expect("base down");
    let candidate_down = gpu
        .alloc_tensor(&[TOP_K, DOWN_M], rdna_compute::DType::F32)
        .expect("candidate down");
    let lds_down = gpu
        .alloc_tensor(&[TOP_K, DOWN_M], rdna_compute::DType::F32)
        .expect("LDS down");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
        &down_ptrs, &topk, &rot, &base_down, DOWN_M, DOWN_K, TOP_K, 1,
    )
    .expect("down baseline");
    gpu.try_gfx1201()
        .expect("exact gfx1201")
        .mq2_lloyd_moe_down_expanded_ep(
            &down_ptrs,
            &gate_ptrs,
            &dummy,
            &topk,
            &rot,
            &candidate_down,
            DOWN_M,
            DOWN_K,
            TOP_K,
            1,
        )
        .expect("down candidate");
    gpu.try_gfx1201()
        .expect("exact gfx1201")
        .mq2_lloyd_moe_down_expanded_lds_ep(
            &down_ptrs, &gate_ptrs, &dummy, &topk, &rot, &lds_down, DOWN_M, DOWN_K, TOP_K, 1,
        )
        .expect("down LDS candidate");
    gpu.hip.device_synchronize().expect("channel sync");

    let base_gate_host = gpu.download_f32(&base_gate).expect("download base gate");
    let candidate_gate_host = gpu
        .download_f32(&candidate_gate)
        .expect("download candidate gate");
    let base_up_host = gpu.download_f32(&base_up).expect("download base up");
    let candidate_up_host = gpu
        .download_f32(&candidate_up)
        .expect("download candidate up");
    let base_down_host = gpu.download_f32(&base_down).expect("download base down");
    let candidate_down_host = gpu
        .download_f32(&candidate_down)
        .expect("download candidate down");
    let lds_down_host = gpu.download_f32(&lds_down).expect("download LDS down");
    assert_raw_equal("gate", &base_gate_host, &candidate_gate_host);
    assert_raw_equal("up", &base_up_host, &candidate_up_host);
    assert_raw_equal("down", &base_down_host, &candidate_down_host);
    assert_raw_equal("down-lds", &candidate_down_host, &lds_down_host);
    assert_nonowned_positive_zero("gate", &candidate_gate_host, GATE_M / 2);
    assert_nonowned_positive_zero("up", &candidate_up_host, GATE_M / 2);
    assert_nonowned_positive_zero("down", &candidate_down_host, DOWN_M);
    assert_nonowned_positive_zero("down-lds", &lds_down_host, DOWN_M);

    for _ in 0..3 {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
            &gate_ptrs, &topk, &x, &base_gate, &base_up, GATE_M, GATE_K, TOP_K,
        )
        .expect("warm gate baseline");
        gpu.try_gfx1201()
            .expect("exact gfx1201")
            .mq2_lloyd_moe_gate_up_ep(
                &gate_ptrs,
                &dummy,
                &topk,
                &x,
                &candidate_gate,
                &candidate_up,
                GATE_M,
                GATE_K,
                TOP_K,
            )
            .expect("warm gate candidate");
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    const REPEATS: usize = 20;
    let gate_base_ms = elapsed_ms(&mut gpu, REPEATS, |gpu| {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
            &gate_ptrs, &topk, &x, &base_gate, &base_up, GATE_M, GATE_K, TOP_K,
        )
        .expect("timed gate baseline");
    });
    let gate_candidate_ms = elapsed_ms(&mut gpu, REPEATS, |gpu| {
        gpu.try_gfx1201()
            .expect("exact gfx1201")
            .mq2_lloyd_moe_gate_up_ep(
                &gate_ptrs,
                &dummy,
                &topk,
                &x,
                &candidate_gate,
                &candidate_up,
                GATE_M,
                GATE_K,
                TOP_K,
            )
            .expect("timed gate candidate");
    });
    let down_base_ms = elapsed_ms(&mut gpu, REPEATS, |gpu| {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
            &down_ptrs, &topk, &rot, &base_down, DOWN_M, DOWN_K, TOP_K, 1,
        )
        .expect("timed down baseline");
    });
    let down_candidate_ms = elapsed_ms(&mut gpu, REPEATS, |gpu| {
        gpu.try_gfx1201()
            .expect("exact gfx1201")
            .mq2_lloyd_moe_down_expanded_ep(
                &down_ptrs,
                &gate_ptrs,
                &dummy,
                &topk,
                &rot,
                &candidate_down,
                DOWN_M,
                DOWN_K,
                TOP_K,
                1,
            )
            .expect("timed down candidate");
    });
    let down_lds_ms = elapsed_ms(&mut gpu, REPEATS, |gpu| {
        gpu.try_gfx1201()
            .expect("exact gfx1201")
            .mq2_lloyd_moe_down_expanded_lds_ep(
                &down_ptrs, &gate_ptrs, &dummy, &topk, &rot, &lds_down, DOWN_M, DOWN_K, TOP_K, 1,
            )
            .expect("timed down LDS candidate");
    });
    const LAYERS: f32 = 43.0;
    const PRODUCT_TOK_S: f32 = 54.903_755;
    let product_ms = 1000.0 / PRODUCT_TOK_S;
    let saved_per_token_ms = (down_candidate_ms - down_lds_ms) * LAYERS;
    let projected_product_pct = saved_per_token_ms / product_ms * 100.0;
    println!(
        "MICRO owned=2/6 repeats={REPEATS} gate_ms={gate_base_ms:.6}->{gate_candidate_ms:.6} \
         gate_speedup={:.3}x down_ms={down_base_ms:.6}->{down_candidate_ms:.6} \
         down_speedup={:.3}x down_lds_ms={down_lds_ms:.6} down_lds_speedup={:.3}x \
         saved_43_layers_ms={saved_per_token_ms:.6} projected_product_pct={projected_product_pct:.3}",
        gate_base_ms / gate_candidate_ms,
        down_base_ms / down_candidate_ms,
        down_candidate_ms / down_lds_ms,
    );
}
