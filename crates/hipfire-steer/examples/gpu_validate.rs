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
// hipfire — see LICENSE and NOTICE in the project root.
#![allow(clippy::needless_range_loop)]

//! GPU validation harness for hipfire-steer's on-GPU apply path.
//!
//! Cross-checks the on-GPU decode apply (`maybe_steer_block` → `gemv_f32` dot +
//! `scaled_add_inplace_gpu_scalar_f32` axpy) against the pure, unit-tested host
//! reference `apply_direction`, for both Steer and Ablate, across several blocks
//! and strengths. Also stress-loops Ablate to catch a missing `gemv → download`
//! stream sync, and round-trips the capture → derive path.
//!
//! Needs a real GPU. Coordinate with the GPU lock:
//!   hipfire lock acquire steer-validate && \
//!     cargo run -p hipfire-steer --example gpu_validate ; hipfire lock release
//!
//! Exits non-zero if any case exceeds tolerance, so it can gate a GPU smoke run.

use hipfire_steer::{
    apply_direction, begin_apply, begin_capture, clear, commit_capture, derive_directions,
    finish_capture, maybe_steer_block, CaptureMeans, SteerMode, SteerSpec,
};
use hipfire_rdna::Gpu;

const NUM_LAYERS: usize = 4;
// Non-power-of-two and > 256 so the gemv tree reduction exercises both the
// strided accumulation loop and the pow2 block rounding.
const HIDDEN: usize = 384;

/// Tiny deterministic LCG → roughly [-1, 1), so the harness needs no rand dep
/// and is reproducible across machines.
struct Lcg(u64);
impl Lcg {
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
    fn unit_vec(&mut self, n: usize) -> Vec<f32> {
        let mut v = self.vec(n);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init failed — is a GPU visible?");
    println!("hipfire-steer GPU validation on {}", gpu.arch);

    let mut rng = Lcg(0x5eed_1234_abcd_0001);
    let directions: Vec<Vec<f32>> = (0..NUM_LAYERS).map(|_| rng.unit_vec(HIDDEN)).collect();

    let mut failures = 0usize;
    // Steer is exact f32 axpy (tight tol). Ablate adds a tree-reduced dot, so
    // allow for reduction-order drift vs the host's sequential sum.
    let cases = [
        (SteerMode::Steer, 0.7_f32, 1e-4_f32),
        (SteerMode::Steer, 1.5, 1e-4),
        (SteerMode::Ablate, 1.0, 2e-3),
        (SteerMode::Ablate, 0.5, 2e-3),
    ];

    for (mode, strength, tol) in cases {
        let spec = SteerSpec {
            directions: directions.clone(),
            mode,
            strength,
            layer_range: 0..NUM_LAYERS,
        };
        begin_apply(spec);

        for layer_idx in 0..NUM_LAYERS {
            let x_host = rng.vec(HIDDEN);

            // Reference (CPU): the unit-tested host implementation.
            let mut expected = x_host.clone();
            apply_direction(&mut expected, &directions[layer_idx], mode, strength);

            // On-GPU: exactly the path the decode forward drives.
            let x_gpu = gpu.upload_f32(&x_host, &[HIDDEN]).unwrap();
            maybe_steer_block(&mut gpu, &x_gpu, layer_idx).unwrap();
            let got = gpu.download_f32(&x_gpu).unwrap();

            let err = max_abs_err(&got, &expected);
            let ok = err <= tol;
            failures += !ok as usize;
            println!(
                "  [{}] {:?} strength={strength} layer={layer_idx}: max_abs_err={err:.2e} (tol {tol:.0e}) {}",
                if ok { "PASS" } else { "FAIL" },
                mode,
                if ok { "" } else { "<<<" },
            );
        }
        clear();
    }

    // Sync stress: repeat one Ablate apply many times. A missing gemv→download
    // sync would surface as an intermittent coefficient read → drift on some
    // iterations. Each iter re-uploads a fresh residual.
    {
        let layer_idx = NUM_LAYERS / 2;
        let strength = 1.0;
        begin_apply(SteerSpec {
            directions: directions.clone(),
            mode: SteerMode::Ablate,
            strength,
            layer_range: 0..NUM_LAYERS,
        });
        let mut worst = 0.0_f32;
        for _ in 0..256 {
            let x_host = rng.vec(HIDDEN);
            let mut expected = x_host.clone();
            apply_direction(
                &mut expected,
                &directions[layer_idx],
                SteerMode::Ablate,
                strength,
            );
            let x_gpu = gpu.upload_f32(&x_host, &[HIDDEN]).unwrap();
            maybe_steer_block(&mut gpu, &x_gpu, layer_idx).unwrap();
            let got = gpu.download_f32(&x_gpu).unwrap();
            worst = worst.max(max_abs_err(&got, &expected));
        }
        clear();
        let ok = worst <= 2e-3;
        failures += !ok as usize;
        println!(
            "  [{}] Ablate sync-stress x256: worst max_abs_err={worst:.2e} (tol 2e-3) {}",
            if ok { "PASS" } else { "FAIL" },
            if ok { "" } else { "<<<" },
        );
    }

    // Capture → derive round-trip: feed known per-block residuals for a "good"
    // and a "bad" set, then check the derived direction matches a host compute.
    {
        let good_means = run_capture(&mut gpu, &mut rng, &[0.1, -0.2]);
        let bad_means = run_capture(&mut gpu, &mut rng, &[0.4, 0.3]);
        let dirs = derive_directions(&good_means, &bad_means, false);
        let mut worst_norm_err = 0.0_f32;
        for d in &dirs {
            let norm = d.iter().map(|x| x * x).sum::<f32>().sqrt();
            worst_norm_err = worst_norm_err.max((norm - 1.0).abs());
        }
        let ok = worst_norm_err <= 1e-4 && dirs.len() == NUM_LAYERS;
        failures += !ok as usize;
        println!(
            "  [{}] capture→derive: {} dirs, worst |‖dir‖-1|={worst_norm_err:.2e} {}",
            if ok { "PASS" } else { "FAIL" },
            dirs.len(),
            if ok { "" } else { "<<<" },
        );
    }

    if failures == 0 {
        println!("ALL PASS");
    } else {
        eprintln!("{failures} case(s) FAILED");
        std::process::exit(1);
    }
}

/// Run a small CAPTURE session: two "prompts", each a full pass over all blocks
/// with a per-prompt residual offset, then return the accumulated means.
fn run_capture(gpu: &mut Gpu, rng: &mut Lcg, offsets: &[f32]) -> CaptureMeans {
    begin_capture(NUM_LAYERS, HIDDEN);
    for &off in offsets {
        for layer_idx in 0..NUM_LAYERS {
            let mut x = rng.vec(HIDDEN);
            for v in &mut x {
                *v += off;
            }
            let x_gpu = gpu.upload_f32(&x, &[HIDDEN]).unwrap();
            maybe_steer_block(gpu, &x_gpu, layer_idx).unwrap();
        }
        commit_capture();
    }
    finish_capture().expect("capture session was active")
}
