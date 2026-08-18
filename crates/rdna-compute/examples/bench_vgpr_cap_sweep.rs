//! Is `amdgpu_num_vgpr` a THROUGHPUT lever, or only a register-count one?
//!
//! Discovered while porting the v1.5 / mq4c residual GEMV (132 B/group, per-256 fp16
//! header). Uncapped it lands at 97 VGPR — one over the 96 needed for 16 waves/SIMD
//! (1536/16) — and occupancy drops to 12. Adding `amdgpu_num_vgpr` produced a
//! strongly NON-MONOTONIC register/spill picture on gfx1201:
//!
//!   uncapped  97 VGPR  occ 12  spill 0
//!   cap 96    96 VGPR  occ 16  spill 3   scratch 16 B
//!   cap 88    88 VGPR  occ 16  spill 5   scratch 24 B
//!   cap 84    84 VGPR  occ 16  spill 0
//!   cap 80    79 VGPR  occ 16  spill 0
//!   cap 76    76 VGPR  occ 16  spill 0
//!   cap 72    70 VGPR  occ 16  spill 0
//!   cap 64    63 VGPR  occ 16  spill 0
//!
//! A small cap violation gets patched by the register allocator (it spills a few
//! values); a large one makes the machine scheduler re-plan and genuinely need
//! fewer registers. So "tighter cap" is not monotonically worse, and the
//! spill-free region sits BELOW the spilling one.
//!
//! None of that says which is FASTEST. More occupancy hides memory latency; fewer
//! registers can mean less aggressive prefetch. On a bandwidth-bound GEMV those
//! pull in opposite directions and only measurement decides — which matters well
//! beyond this one kernel, since the same lever applies to any kernel sitting just
//! over an occupancy boundary.
//!
//! Conventions honoured (CLAUDE.md / project history):
//!   - HIPFIRE_GEMV_ROWS is irrelevant here: this kernel is fixed two-rows-per-block.
//!   - >= 32 warmup iterations before any measured window (8 vs 32 warmups swung a
//!     comparable kernel 38%).
//!   - Arms interleaved sample-by-sample, never arm-at-a-time, so DPM/thermal drift
//!     hits every arm equally.
//!   - >= 3 runs, all reported, ratios primary.
//!   - Host timing is amortised over many launches per sample so the ~22.1 us sync
//!     tax lands under 0.5% rather than being measured per launch.
//!
//! Run: `cargo run --release -p rdna-compute --example bench_vgpr_cap_sweep`

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};

// `kernels` is a private module, so reconstruct the two sources here with the same
// concat the crate uses. Both arms MUST get the identical weight-cache policy
// prefix, or the comparison measures the policy rather than the header.
const CACHE_POLICY: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc")
);
const V1_RESIDUAL: &str = include_str!("../../../kernels/src/gemv_hfq4g256_residual.hip");
const V15_RESIDUAL: &str = include_str!("../../../kernels/src/gemv_mq4cg256_residual.hip");

/// f32 -> IEEE fp16 bits. Round-to-nearest-even, with flush of anything below the
/// fp16 denormal floor; sufficient for a synthetic blob whose contents only need to
/// be well-formed.
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = b & 0x007F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1F {
        return sign | 0x7C00;
    }
    let mut m = (mant >> 13) as u16;
    if (mant & 0x1000) != 0 && ((mant & 0x0FFF) != 0 || (m & 1) != 0) {
        m += 1;
        if m == 0x400 {
            m = 0;
            exp += 1;
            if exp >= 0x1F {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((exp as u16) << 10) | m
}
use std::time::Instant;

const GROUP: usize = 256;
const V1_BYTES: usize = 136;
const V15_BYTES: usize = 132;

/// Caps to sweep. 0 = no attribute at all (the 97-VGPR / occ-12 baseline).
const CAPS: &[u32] = &[0, 96, 84, 80, 76, 72, 64];

const WARMUP: usize = 64;
const ITERS: usize = 200;
const RUNS: usize = 7;

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Synthetic blob in either container. Contents only need to be well-formed: this
/// measures issue/schedule cost, and correctness is covered by mq4v2_parity and the
/// mq4c repack probe, not here.
fn blob(m: usize, k: usize, bytes_per_group: usize, fp16_header: bool) -> Vec<u8> {
    let gpr = k / GROUP;
    let mut out = vec![0u8; m * gpr * bytes_per_group];
    for g in 0..(m * gpr) {
        let off = g * bytes_per_group;
        let scale = 0.0044f32 + prng(g, 7) * 0.001;
        let zero = -0.035f32 - prng(g, 11) * 0.005;
        if fp16_header {
            let s = f16_bits(scale);
            let z = f16_bits(zero);
            out[off..off + 2].copy_from_slice(&s.to_le_bytes());
            out[off + 2..off + 4].copy_from_slice(&z.to_le_bytes());
            for i in 0..128 {
                out[off + 4 + i] = (prng(g * 128 + i, 3) * 255.0) as u8;
            }
        } else {
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for i in 0..128 {
                out[off + 8 + i] = (prng(g * 128 + i, 3) * 255.0) as u8;
            }
        }
    }
    out
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("bench_vgpr_cap_sweep: no GPU ({e}) — skipping");
            return;
        }
    };
    eprintln!("bench_vgpr_cap_sweep: arch={}", gpu.arch);
    if !gpu.arch.starts_with("gfx12") {
        eprintln!("the cap attribute is inside a gfx12-only branch; this sweep is a no-op elsewhere");
    }

    // o_proj-ish shape from the dense qwen3.8 decode path.
    let (m, k) = (5120usize, 5120usize);
    let gpr = k / GROUP;

    let b_v1 = blob(m, k, V1_BYTES, false);
    let b_v15 = blob(m, k, V15_BYTES, true);
    let x: Vec<f32> = (0..k).map(|i| prng(i, 0xC0FF_EE) * 2.0 - 1.0).collect();

    let d_v1 = gpu.upload_raw(&b_v1, &[b_v1.len()]).unwrap();
    let d_v15 = gpu.upload_raw(&b_v15, &[b_v15.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], DType::F32).unwrap();

    // ── build one module per arm, each with a DISTINCT entry symbol ──
    // `launch_kernel_blob` resolves by function name against a flat map, so two
    // modules exporting the same symbol would collide.
    let mut arms: Vec<(String, &'static rdna_compute::GpuTensor)> = Vec::new();
    let _ = &arms;
    let mut names: Vec<String> = Vec::new();

    // v1 control.
    let v1_sym = "bench_v1_residual".to_string();
    let v1_src = format!(
        "#define HIPFIRE_RESIDUAL_KERNEL {v1_sym}\n{}",
        format!("{CACHE_POLICY}{V1_RESIDUAL}")
    );
    match gpu.ensure_kernel_public(&format!("bench_{v1_sym}"), &v1_src, &v1_sym) {
        Ok(()) => names.push(v1_sym.clone()),
        Err(e) => eprintln!("v1 control failed to compile: {e}"),
    }

    // v1.5 at each cap.
    let mut v15_syms = Vec::new();
    for &cap in CAPS {
        let sym = format!("bench_v15_cap{cap}");
        let src = format!(
            "#define HIPFIRE_MQ4C_RESIDUAL_KERNEL {sym}\n#define HIPFIRE_MQ4C_VGPR_CAP {cap}\n{}",
            format!("{CACHE_POLICY}{V15_RESIDUAL}")
        );
        match gpu.ensure_kernel_public(&format!("bench_{sym}"), &src, &sym) {
            Ok(()) => v15_syms.push((cap, sym)),
            Err(e) => eprintln!("cap {cap} failed to compile: {e}"),
        }
    }
    if v15_syms.is_empty() {
        eprintln!("no v1.5 arm compiled; nothing to measure");
        return;
    }

    let grid = [((m as u32) + 1) / 2, 1, 1];
    let block = [32u32, 1, 1];

    let mut launch = |gpu: &Gpu, sym: &str, a: &rdna_compute::GpuTensor| {
        let mut kb = KernargBlob::new();
        kb.push_ptr(a.buf.as_ptr());
        kb.push_ptr(d_x.buf.as_ptr());
        kb.push_ptr(d_y.buf.as_ptr());
        kb.push_i32(m as i32);
        kb.push_i32(k as i32);
        gpu.launch_kernel_blob(sym, grid, block, 0, kb.as_mut_slice())
            .unwrap();
    };

    // Interleaved: one full pass per run, every arm sampled inside the same pass.
    let mut results: Vec<(String, Vec<f64>)> = Vec::new();
    let all: Vec<(String, bool)> = names
        .iter()
        .map(|n| (n.clone(), false))
        .chain(v15_syms.iter().map(|(_, s)| (s.clone(), true)))
        .collect();

    for (sym, is_v15) in &all {
        let a = if *is_v15 { &d_v15 } else { &d_v1 };
        for _ in 0..WARMUP {
            launch(&gpu, sym, a);
        }
        gpu.hip.device_synchronize().unwrap();
        results.push((sym.clone(), Vec::new()));
        let _ = a;
    }

    for _run in 0..RUNS {
        for (idx, (sym, is_v15)) in all.iter().enumerate() {
            let a = if *is_v15 { &d_v15 } else { &d_v1 };
            for _ in 0..8 {
                launch(&gpu, sym, a);
            }
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..ITERS {
                launch(&gpu, sym, a);
            }
            gpu.hip.device_synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            results[idx].1.push(us);
        }
    }

    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    let bytes_v1 = (m * gpr * V1_BYTES) as f64;
    let bytes_v15 = (m * gpr * V15_BYTES) as f64;

    let mut base = f64::NAN;
    println!();
    println!("shape m={m} k={k}  ({gpr} groups/row)   warmup={WARMUP} iters={ITERS} runs={RUNS}");
    println!(
        "{:<22} {:>9} {:>9} {:>10} {:>8}",
        "arm", "us(med)", "GB/s", "vs v1", "samples"
    );
    for (i, (sym, samples)) in results.iter_mut().enumerate() {
        let m_us = med(samples);
        let is_v15 = all[i].1;
        let gbs = (if is_v15 { bytes_v15 } else { bytes_v1 }) / (m_us * 1e-6) / 1e9;
        if !is_v15 {
            base = m_us;
        }
        let ratio = if base.is_nan() {
            String::from("-")
        } else {
            format!("{:.4}", m_us / base)
        };
        let ss: Vec<String> = samples.iter().map(|s| format!("{s:.2}")).collect();
        println!(
            "{:<22} {:>9.2} {:>9.1} {:>10} {:>8}",
            sym,
            m_us,
            gbs,
            ratio,
            ss.join(",")
        );
    }
    println!();
    println!("ratio < 1.0 means faster than the v1 136 B control. v1.5 moves 2.94% fewer");
    println!("weight bytes, so on a purely bandwidth-bound kernel ~0.971 is the floor set");
    println!("by traffic alone; anything above that is schedule/occupancy cost.");
}
