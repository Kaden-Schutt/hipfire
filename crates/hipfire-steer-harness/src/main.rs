// SPDX-License-Identifier: Apache-2.0
//! CLI for the hipfire-steer driver: load a model + the +/- prompt sets, run
//! capture → derive → apply-sweep → score, and print the Pareto front.
//!
//! ```text
//! cargo run --release -p hipfire-steer-harness -- \
//!     --hfq ~/.hipfire/models/medgemma-1.5-4b-it.q8f16.hfq \
//!     --data-dir crates/hipfire-steer/data/heretic \
//!     --limit 16 --eval-limit 16 --mode ablate --strengths 0.5,1.0,1.5
//! ```

use std::error::Error;
use std::path::{Path, PathBuf};

use hipfire_steer::driver::{load_prompts, run_driver, DriverConfig, Prompt};
use hipfire_steer::SteerMode;
use rdna_compute::Gpu;

const SYSTEM_PROMPT: &str = "You are a helpful assistant.";

struct Args {
    hfq: String,
    data_dir: PathBuf,
    limit: usize,
    eval_limit: usize,
    strengths: Vec<f32>,
    modes: Vec<SteerMode>,
    max_new_tokens: usize,
    max_seq: usize,
    orthogonalize: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut hfq = None;
    let mut data_dir = PathBuf::from("crates/hipfire-steer/data/heretic");
    let mut limit = 16usize;
    let mut eval_limit = 16usize;
    let mut strengths = vec![1.0f32];
    let mut modes = vec![SteerMode::Ablate];
    let mut max_new_tokens = 64usize;
    let mut max_seq = 2048usize;
    let mut orthogonalize = true;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut next = || it.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--hfq" => hfq = Some(next()?),
            "--data-dir" => data_dir = PathBuf::from(next()?),
            "--limit" => limit = next()?.parse().map_err(|_| "bad --limit")?,
            "--eval-limit" => eval_limit = next()?.parse().map_err(|_| "bad --eval-limit")?,
            "--max-new-tokens" => {
                max_new_tokens = next()?.parse().map_err(|_| "bad --max-new-tokens")?
            }
            "--max-seq" => max_seq = next()?.parse().map_err(|_| "bad --max-seq")?,
            "--no-orthogonalize" => orthogonalize = false,
            "--strengths" => {
                strengths = next()?
                    .split(',')
                    .map(|s| s.trim().parse::<f32>().map_err(|_| "bad --strengths"))
                    .collect::<Result<_, _>>()?;
            }
            "--mode" => {
                modes = match next()?.as_str() {
                    "steer" => vec![SteerMode::Steer],
                    "ablate" => vec![SteerMode::Ablate],
                    "both" => vec![SteerMode::Steer, SteerMode::Ablate],
                    other => {
                        return Err(format!("--mode: expected steer|ablate|both, got {other}"))
                    }
                };
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: hipfire-steer --hfq <model.hfq> [--data-dir DIR] [--limit N] \
                     [--eval-limit N] [--strengths a,b,c] [--mode steer|ablate|both] \
                     [--max-new-tokens N] [--max-seq N] [--no-orthogonalize]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    Ok(Args {
        hfq: hfq.ok_or("--hfq is required")?,
        data_dir,
        limit,
        eval_limit,
        strengths,
        modes,
        max_new_tokens,
        max_seq,
        orthogonalize,
    })
}

fn load_set(dir: &Path, name: &str, limit: usize) -> Result<Vec<Prompt>, Box<dyn Error>> {
    let path = dir.join(name);
    let mut prompts = load_prompts(&path, SYSTEM_PROMPT).map_err(|e| {
        format!(
            "loading {}: {e} (run scripts/fetch_heretic_prompts.sh)",
            path.display()
        )
    })?;
    prompts.truncate(limit);
    Ok(prompts)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let good_prompts = load_set(&args.data_dir, "good_prompts.txt", args.limit)?;
    let bad_prompts = load_set(&args.data_dir, "bad_prompts.txt", args.limit)?;
    let good_eval = load_set(&args.data_dir, "good_eval.txt", args.eval_limit)?;
    let bad_eval = load_set(&args.data_dir, "bad_eval.txt", args.eval_limit)?;
    eprintln!(
        "prompts: {} good / {} bad (direction), {} good / {} bad (eval)",
        good_prompts.len(),
        bad_prompts.len(),
        good_eval.len(),
        bad_eval.len()
    );

    eprintln!("loading {} ...", args.hfq);
    let gpu = Gpu::init()?;
    let mut harness = hipfire_steer_harness::build_harness(
        gpu,
        Path::new(&args.hfq),
        args.max_seq,
        args.max_new_tokens,
    )?;

    let cfg = DriverConfig {
        good_prompts,
        bad_prompts,
        good_eval,
        bad_eval,
        modes: args.modes,
        strengths: args.strengths,
        layer_range: 0..harness.num_layers(),
        orthogonalize: args.orthogonalize,
        markers: DriverConfig::default_markers(),
    };

    eprintln!("running driver ({} layers) ...", harness.num_layers());
    let report = run_driver(&cfg, harness.as_mut())?;

    println!("\n=== steer driver report ===");
    println!(
        "base refusals: {}/{}",
        report.base_refusals, report.n_bad_eval
    );
    println!("  (* = Pareto-optimal on refusals↓ + KLD↓)");
    for (i, t) in report.trials.iter().enumerate() {
        let star = if report.pareto.contains(&i) { "*" } else { " " };
        println!(
            "{star} {:?} strength={:.2}  refusals={:>3}/{:<3}  kld={:.4}",
            t.mode, t.strength, t.refusals, report.n_bad_eval, t.kl_divergence
        );
    }
    Ok(())
}
