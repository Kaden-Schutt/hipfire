//! Deterministic output gate for the tie-safe fast parallel sampler.
//!
//! Run this binary in two fresh processes and diff stdout:
//!
//! ```text
//! HIPFIRE_SAMPLE_FAST=0 sample_parallel_stable_parity > legacy.txt
//! HIPFIRE_SAMPLE_FAST=1 sample_parallel_stable_parity > fast.txt
//! diff -u legacy.txt fast.txt
//! ```
//!
//! The tied cases must take the fast kernel's sentinel fallback and therefore
//! exercise both the fast and exact legacy reducers when profiled.

use rdna_compute::{DType, Gpu};

const VOCAB: usize = 32768;

fn base_logits() -> Vec<f32> {
    let mut logits = vec![-100.0f32; VOCAB];
    // Strictly ordered top candidates, kept far enough apart that expf does not
    // collapse adjacent probabilities to the same f32 value.
    for (rank, slot) in (0..24usize).map(|rank| (rank, rank * 997 % VOCAB)) {
        logits[slot] = 12.0 - rank as f32 * 0.25;
    }
    logits
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    let result = gpu.zeros(&[2], DType::F32)?;
    let repeat_tokens = [0u32, 3 * 997, 10 * 997, 15 * 997];
    let repeat_bits: Vec<f32> = repeat_tokens.into_iter().map(f32::from_bits).collect();
    let repeat = gpu.upload_f32(&repeat_bits, &[repeat_bits.len()])?;

    for case in ["distinct", "max-tie", "inner-tie", "boundary-tie"] {
        let mut logits = base_logits();
        match case {
            "max-tie" => logits[997] = logits[0],
            "inner-tie" => logits[6 * 997] = logits[5 * 997],
            "boundary-tie" => logits[20 * 997] = logits[19 * 997],
            _ => {}
        }
        let logits = gpu.upload_f32(&logits, &[VOCAB])?;
        for (temperature, top_p, seed) in [
            (0.0f32, 1.0f32, 0x1234_5678u32),
            (0.7, 0.95, 0x1234_5678),
            (1.0, 1.0, 0xdead_beef),
        ] {
            let (token, rng) = gpu.sample_top_p_pf(
                &logits,
                &result,
                &repeat,
                VOCAB,
                temperature,
                top_p,
                seed,
                0,
                1.0,
                0.0,
                0.0,
                Some(20),
                None,
            )?;
            println!(
                "{case} temp={temperature:.1} top_p={top_p:.2} seed={seed:08x} token={token} rng={rng:08x}"
            );
        }
        let seed = 0x1357_9bdf;
        let (token, rng) = gpu.sample_top_p_pf(
            &logits,
            &result,
            &repeat,
            VOCAB,
            0.8,
            0.95,
            seed,
            repeat_bits.len(),
            1.1,
            1.5,
            0.1,
            Some(20),
            None,
        )?;
        println!("{case} penalties seed={seed:08x} token={token} rng={rng:08x}");
        gpu.free_tensor(logits)?;
    }

    gpu.free_tensor(repeat)?;
    gpu.free_tensor(result)?;
    Ok(())
}
