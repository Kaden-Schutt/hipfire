// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness smoke for `hneuron_gain_layer` (H-Neurons intervention gain).
//! Verifies `ffn[t,j] *= gain[j]` in place over `positions × inter`, with a
//! per-neuron gain vector (H-Neurons scaled, others identity). No-LDS kernel;
//! must compile + run on gfx1103.

use hipfire_rdna::Gpu;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_hneuron_gain ===\n  arch = {arch}");

    let mut fails = 0usize;
    // A few shapes: decode (positions=1) and prefill batches; small + FFN-wide.
    let shapes: [(usize, usize); 4] = [(1, 8), (4, 256), (1, 21504), (37, 4096)];
    for (positions, inter) in shapes {
        // ffn[t,j] = deterministic non-trivial values.
        let ffn_host: Vec<f32> = (0..positions * inter)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.031 + 0.5)
            .collect();
        // gain[j]: H-Neurons (every 5th neuron) get a swept gain; others 1.0.
        // Mix down-weight, ablate, and up-weight across the sweep values.
        let sweep = [0.0f32, 0.5, 1.5, 2.0];
        let gain_host: Vec<f32> = (0..inter)
            .map(|j| {
                if j % 5 == 0 {
                    sweep[(j / 5) % sweep.len()]
                } else {
                    1.0
                }
            })
            .collect();

        let d_ffn = gpu.upload_f32(&ffn_host, &[positions * inter]).unwrap();
        let d_gain = gpu.upload_f32(&gain_host, &[inter]).unwrap();
        gpu.hneuron_gain_layer(&d_ffn, &d_gain, positions, inter)
            .unwrap();
        let got = gpu.download_f32(&d_ffn).unwrap();

        let mut shape_fail = 0usize;
        for t in 0..positions {
            for j in 0..inter {
                let idx = t * inter + j;
                let want = ffn_host[idx] * gain_host[j];
                if (got[idx] - want).abs() > 1e-6 {
                    shape_fail += 1;
                }
            }
        }
        eprintln!(
            "  positions={positions} inter={inter}: {}",
            if shape_fail == 0 {
                "OK".to_string()
            } else {
                format!("FAIL ({shape_fail} mismatches)")
            }
        );
        fails += shape_fail;
    }

    if fails == 0 {
        eprintln!("\ntest_hneuron_gain: PASS");
    } else {
        eprintln!("\ntest_hneuron_gain: FAIL ({fails} total mismatches)");
        std::process::exit(1);
    }
}
