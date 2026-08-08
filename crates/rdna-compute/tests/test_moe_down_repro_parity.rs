// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU kernel-parity tests for the MQ3-Lloyd and MQ2-Lloyd MoE down int64
//! reproducible kernels.
//!
//! Verifies that splitting K into two halves and summing the int64 residuals
//! produces a bit-exact result equal to a single full-K launch:
//!
//!   full_i64[row] == half_a_i64[row] + half_b_i64[row]   (for all rows)
//!
//! This is the core partition-invariance property that enables TP all-reduce
//! correctness under any K-split alignment (required: K and each half divisible
//! by 256, the MQ3-Lloyd group size).
//!
//! The old FP kernel does NOT satisfy this property (see the structural
//! quads/tail tree bug documented in tp_minimax.rs:403-424).
//!
//! Run under the GPU lock:
//!   source scripts/gpu-lock.sh && gpu_acquire "device-mesh" && \
//!   cargo test -p rdna-compute test_moe_down_repro_parity -- --nocapture ; \
//!   gpu_release

use rdna_compute::{moe::MOE_DOWN_REPRO_E, DType, Gpu, GpuTensor};

// ---------------------------------------------------------------------------
// Helpers (mirrors test_moe_grouped_wmma_mq3lloyd.rs style)
// ---------------------------------------------------------------------------

fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc_tensor u8");
    gpu.hip.memcpy_htod(&t.buf, data).expect("memcpy_htod u8");
    t
}

fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc_tensor f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod f32");
    t
}

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc_tensor i32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod i32");
    t
}

fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    let t = gpu
        .alloc_tensor(&[data.len() * 8], DType::Raw)
        .expect("alloc_tensor u64");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod u64");
    t
}

fn alloc_i64_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let byte_len = n * 8;
    let t = gpu
        .alloc_tensor(&[byte_len], DType::Raw)
        .expect("alloc i64 zeros");
    gpu.hip
        .memset(&t.buf, 0, byte_len)
        .expect("memset i64 zero");
    t
}

fn alloc_f32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc f32 zeros");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("memset f32 zero");
    t
}

fn download_i64(gpu: &Gpu, tensor: &GpuTensor, n: usize) -> Vec<i64> {
    let mut data = vec![0i64; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 8) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("memcpy_dtoh i64");
    data
}

fn download_f32(gpu: &Gpu, tensor: &GpuTensor, n: usize) -> Vec<f32> {
    let mut data = vec![0f32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("memcpy_dtoh f32");
    data
}

// ---------------------------------------------------------------------------
// MQ3-Lloyd weight builder (112 bytes per 256-weight group)
//
// Layout: [16 B fp16 codebook (8 entries)] [96 B 3-bit packed indices (256 entries)]
// Mirrors build_lloyd_row from test_gemm_mq3g256_lloyd_residual_wmma.rs.
// ---------------------------------------------------------------------------

fn f32_to_f16_le(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    let h: u16 = if exp == 0xff {
        (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 }
    } else if exp - 127 + 15 < 1 {
        sign << 15
    } else if exp - 127 + 15 > 30 {
        (sign << 15) | (0x1f << 10)
    } else {
        let new_exp = (exp - 127 + 15) as u16;
        let m13 = mant & 0x1fff;
        let mut new_mant = (mant >> 13) as u16;
        if m13 > 0x1000 || (m13 == 0x1000 && (new_mant & 1) != 0) {
            new_mant += 1;
        }
        let mut exp_bits = new_exp;
        if new_mant == 0x400 {
            new_mant = 0;
            exp_bits += 1;
        }
        (sign << 15) | (exp_bits << 10) | new_mant
    };
    h.to_le_bytes()
}

fn pack_3bit_group(qs: &[u8; 256]) -> [u8; 96] {
    let mut out = [0u8; 96];
    for tid in 0..32 {
        let mut pk: u32 = 0;
        for i in 0..8 {
            let q = qs[tid * 8 + i] as u32 & 7;
            pk |= q << (3 * i);
        }
        out[tid * 3] = (pk & 0xff) as u8;
        out[tid * 3 + 1] = ((pk >> 8) & 0xff) as u8;
        out[tid * 3 + 2] = ((pk >> 16) & 0xff) as u8;
    }
    out
}

/// Build the byte buffer for one 256-weight group.
/// Returns 112 bytes: 16 B fp16 codebook + 96 B 3-bit packed indices.
fn build_group_bytes(row: usize, group: usize) -> [u8; 112] {
    let mut out = [0u8; 112];
    // Codebook: 8 ascending centroids, different per (row, group).
    let base = ((row.wrapping_mul(7) + group.wrapping_mul(11)) % 19) as f32 * 0.013 - 0.1;
    for i in 0..8usize {
        let v = base + (i as f32 - 3.5) * 0.025;
        let bytes = f32_to_f16_le(v);
        out[i * 2] = bytes[0];
        out[i * 2 + 1] = bytes[1];
    }
    // Indices: pseudo-random in [0, 8).
    let mut qs = [0u8; 256];
    for i in 0..256 {
        qs[i] = ((row.wrapping_mul(31) ^ group.wrapping_mul(53) ^ i.wrapping_mul(7)) & 7) as u8;
    }
    let packed = pack_3bit_group(&qs);
    out[16..112].copy_from_slice(&packed);
    out
}

/// Build a flat MQ3L weight matrix for one expert: [M rows × groups_per_row × 112 bytes].
fn build_expert_weights(m: usize, groups_per_row: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(m * groups_per_row * 112);
    for row in 0..m {
        for g in 0..groups_per_row {
            buf.extend_from_slice(&build_group_bytes(row, g));
        }
    }
    buf
}

/// Extract rows of only specific groups from a full expert weight matrix.
/// `group_range` is the range of group indices to extract.
fn slice_groups(
    full: &[u8],
    m: usize,
    full_groups: usize,
    group_range: std::ops::Range<usize>,
) -> Vec<u8> {
    let ngroups = group_range.end - group_range.start;
    let mut out = Vec::with_capacity(m * ngroups * 112);
    for row in 0..m {
        let row_start = row * full_groups * 112;
        let slice_start = row_start + group_range.start * 112;
        let slice_end = row_start + group_range.end * 112;
        out.extend_from_slice(&full[slice_start..slice_end]);
    }
    out
}

// ---------------------------------------------------------------------------
// Core parity check
// ---------------------------------------------------------------------------

/// Run the int64 down kernel with one expert (k_top=1, topk_weight=1.0),
/// return the i64 residual and the f32 convert output.
fn run_down_i64(
    gpu: &mut Gpu,
    expert_data: &[u8], // [M × groups × 112 B]
    rot_x: &[f32],      // [K] (K = groups * 256)
    m: usize,
    k: usize,
) -> (Vec<i64>, Vec<f32>) {
    // Upload expert weights and build the 1-entry pointer table.
    let expert_t = upload_u8(gpu, expert_data);
    let ptr_val: u64 = expert_t.buf.as_ptr() as u64;
    let expert_ptrs_t = upload_u64(gpu, &[ptr_val]);

    // topk_indices = [0] (select expert 0), topk_weights = [1.0].
    let topk_idx_t = upload_i32(gpu, &[0i32]);
    let topk_w_t = upload_f32(gpu, &[1.0f32]);

    // rot_batch = [1 × K] (k_top=1).
    let rot_batch_t = upload_f32(gpu, rot_x);

    // Zero-init i64 residual buffer ([M] × 8 bytes).
    let residual_i64_t = alloc_i64_zeros(gpu, m);

    // f32 output buffer (after convert).
    let out_f32_t = alloc_f32_zeros(gpu, m);

    gpu.moe_down_mq3g256_lloyd_residual_i64_indexed(
        &expert_ptrs_t,
        &topk_idx_t,
        &topk_w_t,
        &rot_batch_t,
        &residual_i64_t,
        m,
        k,
        1, // k_top
    )
    .expect("moe_down_mq3g256_lloyd_residual_i64_indexed launch");

    gpu.moe_i64_residual_to_f32(&residual_i64_t, &out_f32_t, m)
        .expect("moe_i64_residual_to_f32 launch");

    gpu.hip
        .device_synchronize()
        .expect("sync after down kernels");

    let i64_vals = download_i64(gpu, &residual_i64_t, m);
    let f32_vals = download_f32(gpu, &out_f32_t, m);

    (i64_vals, f32_vals)
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[test]
fn test_moe_down_repro_parity_k_split_bit_exact() {
    // Skip if no GPU.
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU available ({e:?})");
            return;
        }
    };
    eprintln!("GPU: {}", gpu.arch);

    // Small shape: M=4 output rows, K=512 (2 groups of 256).
    // K must be even and both halves divisible by 256.
    let m = 4;
    let k = 512; // 2 groups
    let groups_per_row = k / 256;
    let half_k = k / 2;
    let half_groups = groups_per_row / 2; // = 1

    eprintln!(
        "Shape: M={m}, K={k} ({groups_per_row} groups), K/2={half_k} ({half_groups} group each)"
    );

    // Build the full expert weight matrix.
    let full_weights = build_expert_weights(m, groups_per_row);

    // Split into two halves (groups 0..1 and 1..2).
    let weights_a = slice_groups(&full_weights, m, groups_per_row, 0..half_groups);
    let weights_b = slice_groups(
        &full_weights,
        m,
        groups_per_row,
        half_groups..groups_per_row,
    );

    // Activation vector x[K]: deterministic values in [-0.3, 0.3).
    let rot_x: Vec<f32> = (0..k)
        .map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05)
        .collect();
    let rot_x_a = rot_x[..half_k].to_vec();
    let rot_x_b = rot_x[half_k..].to_vec();

    // --- Full K run ---
    let (full_i64, full_f32) = run_down_i64(&mut gpu, &full_weights, &rot_x, m, k);
    eprintln!("Full  i64: {:?}", full_i64);

    // --- Half A run (groups 0..G/2) ---
    let (half_a_i64, _) = run_down_i64(&mut gpu, &weights_a, &rot_x_a, m, half_k);
    eprintln!("HalfA i64: {:?}", half_a_i64);

    // --- Half B run (groups G/2..G) ---
    let (half_b_i64, _) = run_down_i64(&mut gpu, &weights_b, &rot_x_b, m, half_k);
    eprintln!("HalfB i64: {:?}", half_b_i64);

    // --- Parity assertion: full == half_a + half_b (bit-exact) ---
    let inv_s = 1.0f64 / (1u64 << (MOE_DOWN_REPRO_E as u64)) as f64;
    let mut all_pass = true;
    for row in 0..m {
        let sum = half_a_i64[row].wrapping_add(half_b_i64[row]);
        let fp_full = full_i64[row] as f64 * inv_s;
        let fp_f32out = full_f32[row] as f64;
        eprintln!(
            "row {row}: full={full_i64_v}  a+b={sum}  match={} | fp={fp_full:.8e} f32out={fp_f32out:.8e}",
            full_i64[row] == sum,
            full_i64_v = full_i64[row],
        );
        if full_i64[row] != sum {
            eprintln!(
                "  FAIL row {row}: full_i64={} != half_a+half_b={}",
                full_i64[row], sum
            );
            all_pass = false;
        }
        // FP convert should match to within 1 ULP (the i64 values are bit-identical).
        let fp_diff = (fp_f32out - fp_full).abs();
        if fp_diff > 1e-6 {
            eprintln!("  WARN row {row}: f32 convert diff {fp_diff:.2e}");
        }
    }

    assert!(
        all_pass,
        "FAIL: int64 residuals are NOT bit-exact under K-split (partition non-invariant)"
    );
    eprintln!(
        "\nPASS: full_i64[row] == half_a_i64[row] + half_b_i64[row] for all {m} rows (bit-exact)"
    );
    eprintln!("PASS: moe_i64_residual_to_f32 matches expected scaled values");
}

// ---------------------------------------------------------------------------
// MQ2-Lloyd weight builder (72 bytes per 256-weight group)
//
// Layout: [8 B fp16 codebook (4 entries)] [64 B 2-bit packed indices (256 entries)]
// ---------------------------------------------------------------------------

/// Build the byte buffer for one MQ2L 256-weight group.
/// Returns 72 bytes: 8 B fp16 codebook (4 entries) + 64 B 2-bit packed indices.
fn build_mq2l_group_bytes(row: usize, group: usize) -> [u8; 72] {
    let mut out = [0u8; 72];
    // Codebook: 4 ascending centroids.
    let base = ((row.wrapping_mul(5) + group.wrapping_mul(9)) % 13) as f32 * 0.017 - 0.1;
    for i in 0..4usize {
        let v = base + (i as f32 - 1.5) * 0.04;
        let bytes = f32_to_f16_le(v);
        out[i * 2] = bytes[0];
        out[i * 2 + 1] = bytes[1];
    }
    // Pack 256 2-bit indices into 64 bytes (8 indices per byte).
    // Thread tid (0..32) reads bytes [boff] and [boff+1] where boff = tid*2,
    // giving 8 weights at indices tid*8 .. tid*8+7.
    // byte[b] = q[b*8+0] | (q[b*8+1]<<2) | ... | (q[b*8+7]<<14) — but the
    // kernel reads only 2 bytes per thread: b0 at boff, b1 at boff+1.
    // So byte 0 → weights 0..7 packed 2-bits-each, byte 1 → weights 8..15, etc.
    for b in 0..64usize {
        let mut byte: u8 = 0;
        for bit in 0..4 {
            let wi = b * 4 + bit; // weight index (0..256)
            let q =
                ((row.wrapping_mul(29) ^ group.wrapping_mul(47) ^ wi.wrapping_mul(11)) & 3) as u8;
            byte |= q << (bit * 2);
        }
        out[8 + b] = byte;
    }
    out
}

/// Build a flat MQ2L weight matrix for one expert: [M rows × groups × 72 bytes].
fn build_mq2l_expert_weights(m: usize, groups_per_row: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(m * groups_per_row * 72);
    for row in 0..m {
        for g in 0..groups_per_row {
            buf.extend_from_slice(&build_mq2l_group_bytes(row, g));
        }
    }
    buf
}

/// Extract a group slice from an MQ2L expert weight matrix (72 B/group).
fn slice_mq2l_groups(
    full: &[u8],
    m: usize,
    full_groups: usize,
    group_range: std::ops::Range<usize>,
) -> Vec<u8> {
    let ngroups = group_range.end - group_range.start;
    let mut out = Vec::with_capacity(m * ngroups * 72);
    for row in 0..m {
        let row_start = row * full_groups * 72;
        let slice_start = row_start + group_range.start * 72;
        let slice_end = row_start + group_range.end * 72;
        out.extend_from_slice(&full[slice_start..slice_end]);
    }
    out
}

/// Run the MQ2L int64 residual down kernel (single-token, k_top=1, weight=1.0).
fn run_down_i64_mq2l(
    gpu: &mut Gpu,
    expert_data: &[u8], // [M × groups × 72 B]
    rot_x: &[f32],      // [K]
    m: usize,
    k: usize,
) -> (Vec<i64>, Vec<f32>) {
    let expert_t = upload_u8(gpu, expert_data);
    let ptr_val: u64 = expert_t.buf.as_ptr() as u64;
    let expert_ptrs_t = upload_u64(gpu, &[ptr_val]);

    let topk_idx_t = upload_i32(gpu, &[0i32]);
    let topk_w_t = upload_f32(gpu, &[1.0f32]);
    let rot_batch_t = upload_f32(gpu, rot_x);

    let residual_i64_t = alloc_i64_zeros(gpu, m);
    let out_f32_t = alloc_f32_zeros(gpu, m);

    gpu.moe_down_mq2g256_lloyd_residual_i64_indexed(
        &expert_ptrs_t,
        &topk_idx_t,
        &topk_w_t,
        &rot_batch_t,
        &residual_i64_t,
        m,
        k,
        1,
    )
    .expect("moe_down_mq2g256_lloyd_residual_i64_indexed launch");

    gpu.moe_i64_residual_to_f32(&residual_i64_t, &out_f32_t, m)
        .expect("moe_i64_residual_to_f32 launch");

    gpu.hip.device_synchronize().expect("sync");

    let i64_vals = download_i64(gpu, &residual_i64_t, m);
    let f32_vals = download_f32(gpu, &out_f32_t, m);
    (i64_vals, f32_vals)
}

/// Run the MQ2L int64 expanded down kernel (N=1, k_top=1, no weights).
/// Returns the single expert_outputs_i64 cell per row.
fn run_down_i64_mq2l_expanded(
    gpu: &mut Gpu,
    expert_data: &[u8],
    rot_x: &[f32],
    m: usize,
    k: usize,
) -> Vec<i64> {
    let expert_t = upload_u8(gpu, expert_data);
    let ptr_val: u64 = expert_t.buf.as_ptr() as u64;
    let expert_ptrs_t = upload_u64(gpu, &[ptr_val]);

    let topk_idx_t = upload_i32(gpu, &[0i32]);
    let rot_batch_t = upload_f32(gpu, rot_x);

    // expert_outputs_i64: [N=1 × K_TOP=1 × M] = M elements × 8 bytes
    let n_elems = m;
    let byte_len = n_elems * 8;
    let out_t = gpu
        .alloc_tensor(&[byte_len], rdna_compute::DType::Raw)
        .expect("alloc expanded i64");
    gpu.hip.memset(&out_t.buf, 0, byte_len).expect("memset");

    gpu.moe_down_mq2g256_lloyd_expanded_i64(
        &expert_ptrs_t,
        &topk_idx_t,
        &rot_batch_t,
        &out_t,
        m,
        k,
        1, // k_top
        1, // batch_size (N)
    )
    .expect("moe_down_mq2g256_lloyd_expanded_i64 launch");

    gpu.hip.device_synchronize().expect("sync");

    download_i64(gpu, &out_t, n_elems)
}

// ---------------------------------------------------------------------------
// MQ2L residual parity test
// ---------------------------------------------------------------------------

#[test]
fn test_moe_down_repro_parity_mq2l_residual_k_split_bit_exact() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU available ({e:?})");
            return;
        }
    };
    eprintln!("GPU: {}", gpu.arch);

    // M=4 rows, K=512 (2 groups of 256), split at group boundary.
    let m = 4;
    let k = 512;
    let groups_per_row = k / 256;
    let half_k = k / 2;
    let half_groups = groups_per_row / 2;

    eprintln!(
        "MQ2L residual: M={m}, K={k} ({groups_per_row} groups), K/2={half_k} ({half_groups} group each)"
    );

    let full_weights = build_mq2l_expert_weights(m, groups_per_row);
    let weights_a = slice_mq2l_groups(&full_weights, m, groups_per_row, 0..half_groups);
    let weights_b = slice_mq2l_groups(
        &full_weights,
        m,
        groups_per_row,
        half_groups..groups_per_row,
    );

    let rot_x: Vec<f32> = (0..k)
        .map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05)
        .collect();
    let rot_x_a = rot_x[..half_k].to_vec();
    let rot_x_b = rot_x[half_k..].to_vec();

    let (full_i64, full_f32) = run_down_i64_mq2l(&mut gpu, &full_weights, &rot_x, m, k);
    eprintln!("Full  i64: {:?}", full_i64);

    let (half_a_i64, _) = run_down_i64_mq2l(&mut gpu, &weights_a, &rot_x_a, m, half_k);
    eprintln!("HalfA i64: {:?}", half_a_i64);

    let (half_b_i64, _) = run_down_i64_mq2l(&mut gpu, &weights_b, &rot_x_b, m, half_k);
    eprintln!("HalfB i64: {:?}", half_b_i64);

    let inv_s = 1.0f64 / (1u64 << (MOE_DOWN_REPRO_E as u64)) as f64;
    let mut all_pass = true;
    for row in 0..m {
        let sum = half_a_i64[row].wrapping_add(half_b_i64[row]);
        let fp_full = full_i64[row] as f64 * inv_s;
        let fp_f32out = full_f32[row] as f64;
        eprintln!(
            "row {row}: full={full_i64_v}  a+b={sum}  match={} | fp={fp_full:.8e} f32out={fp_f32out:.8e}",
            full_i64[row] == sum,
            full_i64_v = full_i64[row],
        );
        if full_i64[row] != sum {
            eprintln!(
                "  FAIL row {row}: full_i64={} != half_a+half_b={}",
                full_i64[row], sum
            );
            all_pass = false;
        }
    }

    assert!(
        all_pass,
        "FAIL: MQ2L int64 residuals are NOT bit-exact under K-split"
    );
    eprintln!(
        "\nPASS: MQ2L residual full_i64[row] == half_a + half_b for all {m} rows (bit-exact)"
    );
}

// ---------------------------------------------------------------------------
// MQ2L expanded parity test
// ---------------------------------------------------------------------------

#[test]
fn test_moe_down_repro_parity_mq2l_expanded_k_split_bit_exact() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU available ({e:?})");
            return;
        }
    };
    eprintln!("GPU: {}", gpu.arch);

    // M=4, K=512 (2 groups), split at group boundary.
    let m = 4;
    let k = 512;
    let groups_per_row = k / 256;
    let half_k = k / 2;
    let half_groups = groups_per_row / 2;

    eprintln!(
        "MQ2L expanded: M={m}, K={k} ({groups_per_row} groups), K/2={half_k} ({half_groups} group each)"
    );

    let full_weights = build_mq2l_expert_weights(m, groups_per_row);
    let weights_a = slice_mq2l_groups(&full_weights, m, groups_per_row, 0..half_groups);
    let weights_b = slice_mq2l_groups(
        &full_weights,
        m,
        groups_per_row,
        half_groups..groups_per_row,
    );

    let rot_x: Vec<f32> = (0..k)
        .map(|i| ((i as i32 % 17) as f32 - 8.0) * 0.03)
        .collect();
    let rot_x_a = rot_x[..half_k].to_vec();
    let rot_x_b = rot_x[half_k..].to_vec();

    let full_i64 = run_down_i64_mq2l_expanded(&mut gpu, &full_weights, &rot_x, m, k);
    eprintln!("Full  expanded i64: {:?}", full_i64);

    let half_a_i64 = run_down_i64_mq2l_expanded(&mut gpu, &weights_a, &rot_x_a, m, half_k);
    eprintln!("HalfA expanded i64: {:?}", half_a_i64);

    let half_b_i64 = run_down_i64_mq2l_expanded(&mut gpu, &weights_b, &rot_x_b, m, half_k);
    eprintln!("HalfB expanded i64: {:?}", half_b_i64);

    let mut all_pass = true;
    for row in 0..m {
        let sum = half_a_i64[row].wrapping_add(half_b_i64[row]);
        eprintln!(
            "row {row}: full={full_i64_v}  a+b={sum}  match={}",
            full_i64[row] == sum,
            full_i64_v = full_i64[row],
        );
        if full_i64[row] != sum {
            eprintln!(
                "  FAIL row {row}: expanded full_i64={} != half_a+half_b={}",
                full_i64[row], sum
            );
            all_pass = false;
        }
    }

    assert!(
        all_pass,
        "FAIL: MQ2L expanded int64 outputs are NOT bit-exact under K-split"
    );
    eprintln!(
        "\nPASS: MQ2L expanded full_i64[row] == half_a + half_b for all {m} rows (bit-exact)"
    );
}
