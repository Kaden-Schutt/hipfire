// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Escha-W2 routed GROUPED GEMM: correctness against the slot-parallel kernel,
//! and the tile-shape sweep that chose `escha_grouped_tile`'s default.
//!
//! Both shipped A3B projections at a real prefill chunk's shape
//! (`n_exp = 256`, `slots = 256 tokens * k=8 = 2048`):
//!
//! | projection | K    | M    | trellis K | slot-parallel variant |
//! |------------|------|------|-----------|-----------------------|
//! | `gate_up`  | 2048 | 1024 | 2         | narrow (`K > 1536`)   |
//! | `down`     |  512 | 2048 | 3         | wide   (`K <= 1536`)  |
//!
//! # What the correctness arm asserts, and why it differs per projection
//!
//! The grouped kernel keeps the slot-parallel NARROW form per (token, output
//! row). So:
//!
//! * `gate_up` — the slot-parallel path is narrow too, so the two must agree
//!   **BIT FOR BIT**. Asserted as an exact `to_bits()` equality.
//! * `down` — the slot-parallel path is WIDE (four interleaved accumulators
//!   folded `(a0+a1)+(a2+a3)`); the grouped one sums the identical 512 products
//!   in one sequential chain. They cannot be bit-equal, so this arm reports the
//!   max/mean delta and asserts only that it stays inside f32 summation noise
//!   for K=512.
//!
//! The codes here are RANDOM bits, not a golden fixture: any 16-bit window
//! decodes to a valid fp16 through `escha_cba`, so random codes exercise the
//! decode arithmetic just as well while letting the fixture be the full 256
//! experts a real chunk touches. Bit-exactness of the DECODE itself against
//! the frozen oracle is G2/G7's job, not this one's — this example is about
//! the grouping.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example bench_escha_grouped_gemm
use rdna_compute::{DType, Gpu, GpuTensor};

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn ptr_table(gpu: &Gpu, owner: &GpuTensor, n_exp: usize, stride_bytes: usize) -> GpuTensor {
    let bytes: Vec<u8> = (0..n_exp)
        .map(|e| owner.buf.as_ptr() as u64 + (e * stride_bytes) as u64)
        .flat_map(|p| p.to_ne_bytes())
        .collect();
    gpu.upload_raw(&bytes, &[2 * n_exp]).expect("ptr table")
}

struct Case {
    name: &'static str,
    ic: usize,
    oc: usize,
    trellis_k: u32,
    /// True when the slot-parallel path picks the WIDE accumulator form
    /// (`ic <= 1536`) and the grouped kernel therefore cannot be bit-equal.
    slot_parallel_is_wide: bool,
}

fn main() {
    let cases = [
        Case {
            name: "gate_up",
            ic: 2048,
            oc: 1024,
            trellis_k: 2,
            slot_parallel_is_wide: false,
        },
        Case {
            name: "down",
            ic: 512,
            oc: 2048,
            trellis_k: 3,
            slot_parallel_is_wide: true,
        },
    ];
    // One real prefill chunk of the shipped A3B file.
    let n_exp = 256usize;
    let n_tokens = 256usize;
    let k_top = 8usize;
    let slots = n_tokens * k_top;
    // Sweep set = every instantiation in `escha_moe_gemm_grouped.hip`.
    let tiles = [(4, 2), (8, 2), (8, 4), (8, 8), (16, 2), (16, 4)];
    let iters = 5usize;

    let mut gpu = Gpu::init().expect("gpu");
    let mut failures = 0usize;

    // Routing: each token picks k_top DISTINCT experts, as the router does. A
    // token that could pick the same expert twice would put two slots of the
    // same token in one group, which is legal for the kernel but is not the
    // distribution the tile shape is being tuned for.
    let mut rng = Rng(0xE5C8_A000_0000_0001);
    let mut ids = Vec::with_capacity(slots);
    for _ in 0..n_tokens {
        let mut chosen: Vec<i32> = Vec::with_capacity(k_top);
        while chosen.len() < k_top {
            let e = (rng.next_u32() as usize % n_exp) as i32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        ids.extend_from_slice(&chosen);
    }
    let mut hist = vec![0usize; n_exp];
    for &e in &ids {
        hist[e as usize] += 1;
    }
    let live = hist.iter().filter(|c| **c > 0).count();
    println!(
        "fixture: n_exp={n_exp} slots={slots} live experts={live} group size min/mean/max = \
         {}/{:.1}/{}",
        hist.iter().filter(|c| **c > 0).min().unwrap(),
        slots as f64 / live as f64,
        hist.iter().max().unwrap()
    );

    let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_ids = gpu.upload_raw(&id_bytes, &[slots]).expect("ids");

    // Sort scratch — `block_m = 1`, so no padding: `expert_offsets` is the
    // exact exclusive scan and `sorted_slot_index` is a pure permutation.
    let d_counts = gpu.alloc_tensor(&[n_exp], DType::F32).expect("counts");
    let d_offsets = gpu.alloc_tensor(&[n_exp + 1], DType::F32).expect("offsets");
    let d_sorted = gpu.alloc_tensor(&[slots], DType::F32).expect("sorted");
    let d_tile_ids = gpu.alloc_tensor(&[slots], DType::F32).expect("tile ids");
    let d_inv = gpu.alloc_tensor(&[slots], DType::F32).expect("inv");
    gpu.moe_scatter_fused_k8(
        &d_ids,
        &d_counts,
        &d_offsets,
        &d_sorted,
        &d_tile_ids,
        &d_inv,
        slots,
        n_exp,
        slots,
        1,
    )
    .expect("scatter");
    gpu.hip.device_synchronize().expect("sync");

    for case in &cases {
        let words_per_expert = (case.ic / 16) * (case.oc / 16) * 16 * case.trellis_k as usize;
        let expert_bytes = words_per_expert * 2;
        let mut crng = Rng(0x5EED_0000_0000_0001 ^ case.ic as u64);
        let mut code_bytes = vec![0u8; n_exp * expert_bytes];
        for chunk in code_bytes.chunks_exact_mut(4) {
            chunk.copy_from_slice(&crng.next_u32().to_le_bytes());
        }
        let d_code = gpu
            .upload_raw(&code_bytes, &[code_bytes.len()])
            .expect("upload code");
        drop(code_bytes);
        let code_ptrs = ptr_table(&gpu, &d_code, n_exp, expert_bytes);

        let x: Vec<f32> = (0..slots * case.ic).map(|_| rng.next_f32()).collect();
        let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_x = gpu.upload_raw(&x_bytes, &[slots * case.ic]).expect("x");
        drop(x_bytes);
        let d_y_slot = gpu
            .alloc_tensor(&[slots * case.oc], DType::F32)
            .expect("y slot");
        let d_y_grp = gpu
            .alloc_tensor(&[slots * case.oc], DType::F32)
            .expect("y grouped");
        let d_y_wmma = gpu
            .alloc_tensor(&[slots * case.oc], DType::F32)
            .expect("y wmma");

        // ── the slot-parallel baseline ───────────────────────────────────────
        //
        // Re-measured immediately before EVERY tile below, not once. This box
        // drifts: the untouched slot-parallel kernel has been seen to move 25%
        // (12.95 -> 16.27 ms) between two consecutive processes, which is
        // larger than several of the differences the sweep is trying to rank.
        // Every speedup printed here is therefore a ratio against a control
        // taken in the same second, and the absolute millisecond figures are
        // only good to ~15%.
        let mut measure_slot = |gpu: &mut Gpu| -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..iters {
                gpu.hip.device_synchronize().expect("sync");
                let t = std::time::Instant::now();
                gpu.escha_gemv_native_moe_k8_indexed_batched(
                    &code_ptrs,
                    &d_ids,
                    &d_x,
                    &d_y_slot,
                    case.oc,
                    case.ic,
                    slots,
                    case.trellis_k,
                )
                .expect("slot-parallel gemv");
                gpu.hip.device_synchronize().expect("sync");
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        };
        let slot_ms = measure_slot(&mut gpu);
        let y_slot = gpu.download_f32(&d_y_slot).expect("dl slot");
        // A comparison of two all-zero buffers would report zero difference and
        // prove nothing.
        let nonzero = y_slot.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > y_slot.len() / 2,
            "{}: only {nonzero} of {} slot-parallel outputs are non-zero — the fixture is \
             degenerate and every comparison below would be vacuous",
            case.name,
            y_slot.len()
        );
        // Logical expert bytes the slot-parallel kernel moves: every slot reads
        // its whole expert.
        let slot_w_gb = (slots * expert_bytes) as f64 / 1e9;
        let slot_x_gb = ((case.oc / 16) * slots * case.ic * 4) as f64 / 1e9;
        println!(
            "\n=== {} (K={} M={} tk={} slot-parallel {}) ===",
            case.name,
            case.ic,
            case.oc,
            case.trellis_k,
            if case.slot_parallel_is_wide {
                "wide"
            } else {
                "narrow"
            }
        );
        println!(
            "  slot-parallel : {slot_ms:8.3} ms   weights {slot_w_gb:.3} GB + x {slot_x_gb:.3} GB \
             = {:.3} GB -> {:.1} GB/s",
            slot_w_gb + slot_x_gb,
            (slot_w_gb + slot_x_gb) / (slot_ms / 1e3)
        );

        for &(rows, ctiles) in &tiles {
            if case.oc % (16 * ctiles) != 0 {
                continue;
            }
            let ctl_ms = measure_slot(&mut gpu);
            // The `_tiled` entry point, not the production one: the latter
            // memoises its shape in a `OnceLock`, so a sweep through it would
            // measure the first shape six times over.
            let mut grp_ms = f64::INFINITY;
            let mut launched = true;
            for _ in 0..iters {
                gpu.hip.device_synchronize().expect("sync");
                let t = std::time::Instant::now();
                let r = gpu.escha_gemm_native_moe_grouped_tiled(
                    &code_ptrs,
                    &d_offsets,
                    &d_sorted,
                    &d_x,
                    &d_y_grp,
                    case.oc,
                    case.ic,
                    slots,
                    n_exp,
                    case.trellis_k,
                    rows,
                    ctiles,
                );
                if let Err(e) = r {
                    println!("  r{rows}c{ctiles} : launch refused: {e}");
                    launched = false;
                    break;
                }
                gpu.hip.device_synchronize().expect("sync");
                grp_ms = grp_ms.min(t.elapsed().as_secs_f64() * 1e3);
            }
            if !launched {
                continue;
            }
            // Logical expert bytes: each expert's code is read once per pass of
            // ROWS rows through its group.
            let passes: usize = hist.iter().map(|g| g.div_ceil(rows)).sum();
            let grp_w_gb = (passes * expert_bytes) as f64 / 1e9;
            let grp_x_gb = ((case.oc / (16 * ctiles)) * slots * case.ic * 4) as f64 / 1e9;
            println!(
                "  r{rows}c{ctiles} : {grp_ms:8.3} ms  \
                 {:.2}x (control {ctl_ms:.3} ms)  weights {grp_w_gb:.3} GB + x {grp_x_gb:.3} GB \
                 = {:.3} GB -> {:.1} GB/s",
                ctl_ms / grp_ms,
                grp_w_gb + grp_x_gb,
                (grp_w_gb + grp_x_gb) / (grp_ms / 1e3)
            );

            // ── WMMA arm ──────────────────────────────────────────────
            // Same grouping inputs, matrix cores instead of scalar FMAs.
            // Run once per (rows, ctiles) sweep entry is wasteful — it does
            // not take a register tile — so only do it on the first entry.
            if rows == 8 && ctiles == 4 {
                let mut w_ms = f64::INFINITY;
                let mut ok = true;
                for _ in 0..iters {
                    gpu.hip.device_synchronize().expect("sync");
                    let t = std::time::Instant::now();
                    let r = gpu.escha_gemm_native_moe_grouped_wmma(
                        &code_ptrs,
                        &d_offsets,
                        &d_sorted,
                        &d_x,
                        &d_y_wmma,
                        case.oc,
                        case.ic,
                        slots,
                        n_exp,
                        case.trellis_k,
                    );
                    if let Err(e) = r {
                        println!("  wmma  : launch refused: {e}");
                        ok = false;
                        break;
                    }
                    gpu.hip.device_synchronize().expect("sync");
                    w_ms = w_ms.min(t.elapsed().as_secs_f64() * 1e3);
                }
                if ok {
                    let yw = gpu.download_f32(&d_y_wmma).expect("dl wmma");
                    let mut worst = 0.0f32;
                    let mut sum = 0.0f64;
                    let mut nf = 0usize;
                    for (a, b) in y_slot.iter().zip(&yw) {
                        let d = (a - b).abs();
                        worst = worst.max(d);
                        sum += d as f64;
                        if !b.is_finite() {
                            nf += 1;
                        }
                    }
                    println!(
                        "  wmma  : {w_ms:8.3} ms  {:.2}x vs scalar-grouped, {:.2}x vs slot-parallel",
                        grp_ms / w_ms,
                        ctl_ms / w_ms
                    );
                    println!(
                        "            vs slot-parallel: max {worst:e}, mean {:e}, non-finite {nf}",
                        sum / yw.len() as f64
                    );
                    if nf != 0 {
                        failures += 1;
                    }
                }
            }

            let y_grp = gpu.download_f32(&d_y_grp).expect("dl grouped");
            let mut diff = 0usize;
            let mut worst = 0.0f32;
            let mut sum_abs = 0.0f64;
            for (a, b) in y_slot.iter().zip(&y_grp) {
                if a.to_bits() != b.to_bits() {
                    diff += 1;
                }
                let d = (a - b).abs();
                worst = worst.max(d);
                sum_abs += d as f64;
            }
            let mean = sum_abs / y_grp.len() as f64;
            let nonfinite = y_grp.iter().filter(|v| !v.is_finite()).count();
            println!(
                "            vs slot-parallel: {diff} differing floats of {} (max {worst:e}, \
                 mean {mean:e}), non-finite {nonfinite}",
                y_grp.len()
            );
            if nonfinite != 0 {
                failures += 1;
            }
            if case.slot_parallel_is_wide {
                // Cannot be bit-equal by construction (see the module docs);
                // hold it to f32 summation noise instead.
                if worst > 1e-3 {
                    println!("            FAIL: delta {worst:e} is beyond f32 summation noise");
                    failures += 1;
                }
            } else if diff != 0 {
                println!("            FAIL: the narrow arm must be bit-identical");
                failures += 1;
            }
        }

        for t in [d_code, code_ptrs, d_x, d_y_slot, d_y_grp] {
            let _ = gpu.free_tensor(t);
        }
    }

    for t in [d_ids, d_counts, d_offsets, d_sorted, d_tile_ids, d_inv] {
        let _ = gpu.free_tensor(t);
    }
    assert_eq!(failures, 0, "grouped GEMM sweep had {failures} failures");
    println!("\nbench_escha_grouped_gemm: OK");
}
