//! Byte-exact validation for the SGLang-style MoE scatter pipeline:
//!   Phase 1: moe_scatter_histogram_k8
//!   Phase 2: moe_scatter_offsets_k8
//!   Phase 3: moe_scatter_permute_k8
//!
//! Path 2 stage 1 — produces sorted_slot_index + expert_tile_ids for the
//! grouped-WMMA-GEMM MoE prefill (Stage 2). See memory entry
//! `project_next_session_a3b_path2_stage1` for the algorithm spec.
//!
//! Validates two cases:
//!   1. Toy:   N=4, K_TOP=2, E=4, BLOCK_M=2  (hand-checkable)
//!   2. A3B:   N=256, K_TOP=8, E=256, BLOCK_M=16
//!
//! Compares:
//!   - expert_token_counts (raw and padded)
//!   - expert_offsets (exclusive prefix of padded)
//!   - expert_tile_ids (deterministic by expert)
//!   - sorted_slot_index — bucket SETS (intra-bucket order is non-
//!     deterministic because Phase 3 uses LDS atomic increments)
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_scatter_permute_k8

use rdna_compute::{Gpu, GpuTensor, DType};

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc_tensor i32");
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("memcpy_htod i32");
    t
}

fn alloc_i32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[n * 4], DType::Raw)
        .expect("alloc_tensor i32");
    gpu.hip
        .memset(&t.buf, 0, n * 4)
        .expect("memset zero");
    t
}

fn download_i32(gpu: &Gpu, tensor: &GpuTensor, n: usize) -> Vec<i32> {
    let mut data = vec![0i32; n];
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4)
    };
    gpu.hip.memcpy_dtoh(bytes, &tensor.buf).expect("memcpy_dtoh i32");
    data
}

/// Deterministic LCG for topk_indices generation. Glibc constants.
fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state & 0x7fff_ffff
}

fn gen_topk_indices(n: usize, k_top: usize, num_experts: usize, seed: u32) -> Vec<i32> {
    let mut state = seed;
    let mut out = vec![0i32; n * k_top];
    for tok in 0..n {
        // Per-token unique k_top experts: draw without replacement.
        let mut chosen: Vec<i32> = Vec::with_capacity(k_top);
        while chosen.len() < k_top {
            let cand = (lcg(&mut state) as usize % num_experts) as i32;
            if !chosen.contains(&cand) {
                chosen.push(cand);
            }
        }
        for k in 0..k_top {
            out[tok * k_top + k] = chosen[k];
        }
    }
    out
}

/// CPU reference: same algorithm as the GPU pipeline.
struct CpuRef {
    raw_counts: Vec<i32>,
    padded_counts: Vec<i32>,
    offsets: Vec<i32>,
    sorted_slot_index: Vec<i32>, // [m_total]
    expert_tile_ids: Vec<i32>,   // [m_total / block_m]
    m_total: usize,
}

fn cpu_scatter(
    topk_indices: &[i32],
    num_experts: usize,
    block_m: usize,
) -> CpuRef {
    // Phase 1: raw histogram.
    let mut raw_counts = vec![0i32; num_experts];
    for &e in topk_indices {
        if e >= 0 && (e as usize) < num_experts {
            raw_counts[e as usize] += 1;
        }
    }

    // Phase 2: pad + exclusive prefix sum.
    let padded_counts: Vec<i32> = raw_counts
        .iter()
        .map(|&r| {
            let bm = block_m as i32;
            ((r + bm - 1) / bm) * bm
        })
        .collect();
    let mut offsets = vec![0i32; num_experts + 1];
    {
        let mut acc = 0;
        for i in 0..num_experts {
            offsets[i] = acc;
            acc += padded_counts[i];
        }
        offsets[num_experts] = acc;
    }
    let m_total = offsets[num_experts] as usize;

    // Phase 3: scatter + tile_ids. In-input-order scatter gives a
    // canonical permutation; the GPU's order will differ inside each
    // bucket but the SET equality check below tolerates that.
    let mut sorted_slot_index = vec![-1i32; m_total];
    let mut bucket_fill = vec![0i32; num_experts];
    for (flat, &e) in topk_indices.iter().enumerate() {
        if e >= 0 && (e as usize) < num_experts {
            let eu = e as usize;
            let pos = offsets[eu] + bucket_fill[eu];
            sorted_slot_index[pos as usize] = flat as i32;
            bucket_fill[eu] += 1;
        }
    }

    // Tile ids: walk by expert.
    let num_tiles = m_total / block_m;
    let mut expert_tile_ids = vec![0i32; num_tiles];
    for e in 0..num_experts {
        let s = (offsets[e] as usize) / block_m;
        let en = (offsets[e + 1] as usize) / block_m;
        for t in s..en {
            expert_tile_ids[t] = e as i32;
        }
    }

    CpuRef {
        raw_counts,
        padded_counts,
        offsets,
        sorted_slot_index,
        expert_tile_ids,
        m_total,
    }
}

fn compare_set_per_bucket(
    label: &str,
    gpu_sorted: &[i32],
    cpu_ref: &CpuRef,
    num_experts: usize,
    block_m: usize,
) -> usize {
    let mut fails = 0;
    for e in 0..num_experts {
        let start = cpu_ref.offsets[e] as usize;
        let end = cpu_ref.offsets[e + 1] as usize;
        let raw = cpu_ref.raw_counts[e] as usize;

        // The first `raw` slots in this bucket are real entries; the
        // remaining (padded - raw) are -1 sentinels. The GPU's order
        // inside the real range is non-deterministic, so collect both
        // sides as sets and compare sorted.
        let mut gpu_bucket: Vec<i32> = gpu_sorted[start..end].to_vec();
        let mut cpu_bucket: Vec<i32> = cpu_ref.sorted_slot_index[start..end].to_vec();
        gpu_bucket.sort();
        cpu_bucket.sort();

        if gpu_bucket != cpu_bucket {
            if fails < 3 {
                eprintln!(
                    "  {label}: expert {e} bucket mismatch (raw={raw} pad={} range=[{start},{end}))",
                    cpu_ref.padded_counts[e]
                );
                eprintln!("    cpu={:?}", &cpu_bucket[..cpu_bucket.len().min(8)]);
                eprintln!("    gpu={:?}", &gpu_bucket[..gpu_bucket.len().min(8)]);
            }
            fails += 1;
        }

        // Also verify every non-(-1) entry from the GPU side really
        // maps back to expert e in topk_indices.
        for &flat in &gpu_bucket {
            if flat == -1 {
                continue;
            }
            // flat = tok * K_TOP + k → recovered expert must equal e
            // (cross-checked against the original topk_indices below).
            let _ = flat;
        }

        // Padding count must match.
        let gpu_pads = gpu_sorted[start..end].iter().filter(|&&x| x == -1).count();
        let cpu_pads = end - start - raw;
        if gpu_pads != cpu_pads {
            eprintln!(
                "  {label}: expert {e} padding count mismatch (cpu={cpu_pads} gpu={gpu_pads})"
            );
            fails += 1;
        }

        let _ = block_m; // currently unused; kept for future per-tile checks
    }
    fails
}

fn compare_vec(label: &str, a: &[i32], b: &[i32]) -> usize {
    if a.len() != b.len() {
        eprintln!("{label}: length mismatch (cpu={} gpu={})", a.len(), b.len());
        return 1;
    }
    let mut fails = 0;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            if fails < 3 {
                eprintln!("  {label}[{i}]: cpu={x} gpu={y}");
            }
            fails += 1;
        }
    }
    if fails > 0 {
        eprintln!("{label}: FAIL {}/{}", fails, a.len());
    } else {
        println!("{label}: byte-exact ({} entries)", a.len());
    }
    fails
}

fn validate_back_pointers(
    label: &str,
    sorted_slot_index: &[i32],
    topk_indices: &[i32],
    k_top: usize,
    cpu_ref: &CpuRef,
    num_experts: usize,
) -> usize {
    let mut fails = 0;
    for e in 0..num_experts {
        let s = cpu_ref.offsets[e] as usize;
        let en = cpu_ref.offsets[e + 1] as usize;
        for &flat in &sorted_slot_index[s..en] {
            if flat == -1 {
                continue;
            }
            let recovered = topk_indices[flat as usize];
            if recovered as usize != e {
                if fails < 3 {
                    eprintln!(
                        "  {label}: flat {flat} sits in bucket e={e} but topk_indices[{flat}] = {recovered}"
                    );
                }
                fails += 1;
            }
            // Also: flat must be in range.
            if (flat as usize) >= topk_indices.len() {
                fails += 1;
            }
            let _ = k_top;
        }
    }
    if fails > 0 {
        eprintln!("{label}: back-pointer FAIL ({fails})");
    } else {
        println!("{label}: back-pointers consistent");
    }
    fails
}

fn run_case(
    gpu: &mut Gpu,
    label: &str,
    n: usize,
    k_top: usize,
    num_experts: usize,
    block_m: usize,
    seed: u32,
) -> usize {
    println!(
        "\n=== {label}: N={n} K_TOP={k_top} E={num_experts} BLOCK_M={block_m} ==="
    );

    let topk_indices_host = gen_topk_indices(n, k_top, num_experts, seed);
    let cpu_ref = cpu_scatter(&topk_indices_host, num_experts, block_m);
    let total_slots = n * k_top;
    println!(
        "  total_slots={total_slots} m_total={} num_tiles={}",
        cpu_ref.m_total,
        cpu_ref.m_total / block_m
    );

    let m_total_max = total_slots + num_experts * (block_m - 1);

    // Upload + alloc.
    let d_topk = upload_i32(gpu, &topk_indices_host);
    let d_counts = alloc_i32_zeros(gpu, num_experts);
    let d_offsets = alloc_i32_zeros(gpu, num_experts + 1);
    let d_sorted = alloc_i32_zeros(gpu, m_total_max);
    let d_tile_ids = alloc_i32_zeros(gpu, m_total_max / block_m);
    let d_inverse = alloc_i32_zeros(gpu, total_slots);

    // Phase 1.
    gpu.moe_scatter_histogram_k8(&d_topk, &d_counts, total_slots, num_experts)
        .expect("histogram");
    gpu.hip.device_synchronize().expect("sync after phase 1");
    let gpu_raw_counts = download_i32(gpu, &d_counts, num_experts);
    let mut fails = compare_vec(
        &format!("[{label}] raw_counts"),
        &cpu_ref.raw_counts,
        &gpu_raw_counts,
    );

    // Phase 2.
    gpu.moe_scatter_offsets_k8(&d_counts, &d_offsets, num_experts, block_m)
        .expect("offsets");
    gpu.hip.device_synchronize().expect("sync after phase 2");
    let gpu_padded_counts = download_i32(gpu, &d_counts, num_experts);
    fails += compare_vec(
        &format!("[{label}] padded_counts"),
        &cpu_ref.padded_counts,
        &gpu_padded_counts,
    );
    let gpu_offsets = download_i32(gpu, &d_offsets, num_experts + 1);
    fails += compare_vec(
        &format!("[{label}] expert_offsets"),
        &cpu_ref.offsets,
        &gpu_offsets,
    );

    // Phase 3.
    gpu.moe_scatter_permute_k8(
        &d_topk,
        &d_offsets,
        &d_sorted,
        &d_tile_ids,
        &d_inverse,
        total_slots,
        num_experts,
        cpu_ref.m_total,
        block_m,
    )
    .expect("permute");
    gpu.hip.device_synchronize().expect("sync after phase 3");

    let gpu_sorted = download_i32(gpu, &d_sorted, cpu_ref.m_total);
    let gpu_tile_ids = download_i32(gpu, &d_tile_ids, cpu_ref.m_total / block_m);

    // expert_tile_ids is deterministic — compare byte-exact.
    fails += compare_vec(
        &format!("[{label}] expert_tile_ids"),
        &cpu_ref.expert_tile_ids,
        &gpu_tile_ids,
    );

    // sorted_slot_index — bucket SET equality.
    let bucket_fails = compare_set_per_bucket(
        &format!("[{label}] sorted_slot_index"),
        &gpu_sorted,
        &cpu_ref,
        num_experts,
        block_m,
    );
    if bucket_fails == 0 {
        println!(
            "[{label}] sorted_slot_index: bucket sets match ({} buckets)",
            num_experts
        );
    }
    fails += bucket_fails;

    fails += validate_back_pointers(
        &format!("[{label}] sorted_slot_index back-pointers"),
        &gpu_sorted,
        &topk_indices_host,
        k_top,
        &cpu_ref,
        num_experts,
    );

    println!("[{label}] total fails: {fails}");
    fails
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);

    let mut total_fails = 0usize;

    // Toy case — hand-checkable.
    total_fails += run_case(&mut gpu, "toy", 4, 2, 4, 2, 0xc0ffee);

    // Mid case — sweep both small batch and ragged padding.
    total_fails += run_case(&mut gpu, "mid", 16, 4, 8, 4, 0xdead_beef);

    // A3B-shape — the production target.
    total_fails += run_case(&mut gpu, "a3b", 256, 8, 256, 16, 0x1234_5678);

    if total_fails == 0 {
        println!("\nALL CASES PASS");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL total={total_fails}");
        std::process::exit(1);
    }
}
