// SPDX-License-Identifier: Apache-2.0
//! CLI for the hipfire-steer driver, driven through a `hipfire-daemon`
//! subprocess: load a model + the +/- prompt sets, run capture → derive →
//! apply-sweep → score (all through the daemon's correct inference, templating,
//! and KLD), and print the Pareto front.
//!
//! ```text
//! cargo run --release -p hipfire-steer-harness -- \
//!     --hfq ~/.hipfire/models/medgemma-4b-it.q8f16.hfq \
//!     --data-dir crates/hipfire-steer/data/medical \
//!     --limit 16 --eval-limit 16 --mode ablate --strengths 0.5,1.0,1.5
//! ```

use std::error::Error;
use std::path::{Path, PathBuf};

use hipfire_steer::driver::{load_prompts, run_driver, DriverConfig, ModelHarness, Prompt};
use hipfire_steer::SteerMode;
use hipfire_steer_harness::DaemonHarness;

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
    /// When set, capture+derive only and write a `.lora` adapter here (no sweep).
    export_lora: Option<PathBuf>,
    /// When set, load this `.lora` and show base vs applied vs scale-0 refusals.
    apply_lora: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut hfq = None;
    let mut data_dir = PathBuf::from("crates/hipfire-steer/data/medical");
    let mut limit = 16usize;
    let mut eval_limit = 16usize;
    let mut strengths = vec![1.0f32];
    let mut modes = vec![SteerMode::Ablate];
    let mut max_new_tokens = 64usize;
    let mut max_seq = 2048usize;
    let mut orthogonalize = true;
    let mut export_lora = None;
    let mut apply_lora = None;

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
            "--export-lora" => export_lora = Some(PathBuf::from(next()?)),
            "--apply-lora" => apply_lora = Some(PathBuf::from(next()?)),
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
                     [--max-new-tokens N] [--max-seq N] [--no-orthogonalize] \
                     [--export-lora PATH]\n  --export-lora: capture+derive only, write a \
                     rank-1 ablate adapter (scale = first --strengths) to PATH"
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
        export_lora,
        apply_lora,
    })
}

fn load_set(dir: &Path, name: &str, limit: usize) -> Result<Vec<Prompt>, Box<dyn Error>> {
    let path = dir.join(name);
    let mut prompts = load_prompts(&path, SYSTEM_PROMPT)
        .map_err(|e| format!("loading {}: {e}", path.display()))?;
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

    let daemon_bin = hipfire_daemon_adapter::find_daemon_bin_or_error()?;
    eprintln!(
        "loading {} via daemon {} ...",
        args.hfq,
        daemon_bin.display()
    );
    let tmp = std::env::temp_dir().join(format!("hipfire-steer-{}", std::process::id()));
    let mut harness = DaemonHarness::connect(
        &daemon_bin,
        Path::new(&args.hfq),
        args.max_seq,
        args.max_new_tokens,
        SYSTEM_PROMPT.to_string(),
        tmp,
    )?;

    // Export mode: capture + derive only, then write a rank-1 ablate adapter.
    if let Some(out) = args.export_lora.as_ref() {
        let strength = args.strengths.first().copied().unwrap_or(1.0);
        return export_lora(
            &mut harness,
            &good_prompts,
            &bad_prompts,
            args.orthogonalize,
            strength,
            out,
        );
    }

    // Apply mode: load a `.lora` and compare base / applied / scale-0 refusals.
    if let Some(path) = args.apply_lora.as_ref() {
        return apply_lora(&mut harness, &bad_eval, path);
    }

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
    let report = run_driver(&cfg, &mut harness)?;

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

/// Capture +/- residual means through the daemon, derive per-block directions, and
/// write a rank-1 ablate adapter (residual form) sized to the model. `strength`
/// seeds the adapter's default `scale` (the live intensity dial at load time).
fn export_lora(
    harness: &mut DaemonHarness,
    good_prompts: &[Prompt],
    bad_prompts: &[Prompt],
    orthogonalize: bool,
    strength: f32,
    out: &Path,
) -> Result<(), Box<dyn Error>> {
    use hipfire_steer::lora;

    eprintln!(
        "capturing directions ({} good / {} bad, {} layers) ...",
        good_prompts.len(),
        bad_prompts.len(),
        harness.num_layers()
    );
    harness
        .begin_capture()
        .map_err(|e| format!("begin_capture: {e}"))?;
    harness
        .capture(good_prompts)
        .map_err(|e| format!("capture good: {e}"))?;
    let good_means = harness
        .finish_capture()
        .map_err(|e| format!("finish good: {e}"))?;
    harness
        .begin_capture()
        .map_err(|e| format!("begin_capture: {e}"))?;
    harness
        .capture(bad_prompts)
        .map_err(|e| format!("capture bad: {e}"))?;
    let bad_means = harness
        .finish_capture()
        .map_err(|e| format!("finish bad: {e}"))?;

    let directions = hipfire_steer::derive_directions(&good_means, &bad_means, orthogonalize);
    let layers = harness.num_layers();
    let adapter = lora::abliteration_adapter(
        "abliterate",
        &directions,
        SteerMode::Ablate,
        strength,
        0..layers,
    )?;
    lora::write_adapter(out, &adapter)?;
    eprintln!(
        "wrote LoRA adapter: {} rank-1 deltas, default scale {strength:.2} → {}",
        adapter.deltas.len(),
        out.display()
    );
    Ok(())
}

/// Load a `.lora` adapter into the live daemon and report refusal counts on the
/// bad-eval set for three states: base (no adapter), adapter applied at its baked
/// scale, and the same adapter dialed to scale 0 (≡ base) — proving load + the GPU
/// stack apply + the live intensity knob end to end.
fn apply_lora(
    harness: &mut DaemonHarness,
    bad_eval: &[Prompt],
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    use hipfire_steer::driver::count_refusals;
    let markers = DriverConfig::default_markers();
    let n = bad_eval.len();

    let base = harness
        .generate(bad_eval)
        .map_err(|e| format!("generate base: {e}"))?;
    let base_ref = count_refusals(&base, &markers);

    harness
        .lora_load(path, None)
        .map_err(|e| format!("lora_load: {e}"))?;
    let loaded = harness.lora_list().map_err(|e| format!("lora_list: {e}"))?;
    eprintln!("loaded adapters: {loaded:?}");

    let applied = harness
        .generate(bad_eval)
        .map_err(|e| format!("generate applied: {e}"))?;
    let applied_ref = count_refusals(&applied, &markers);

    let id = loaded
        .first()
        .map(|(id, _)| id.clone())
        .ok_or("apply-lora: no adapter loaded")?;
    harness
        .lora_set_scale(&id, 0.0)
        .map_err(|e| format!("lora_set_scale: {e}"))?;
    let off = harness
        .generate(bad_eval)
        .map_err(|e| format!("generate scale0: {e}"))?;
    let off_ref = count_refusals(&off, &markers);

    println!("\n=== lora apply report ===");
    println!("base (no adapter):          refusals {base_ref}/{n}");
    println!("adapter applied (default):  refusals {applied_ref}/{n}");
    println!("adapter scale=0 (≡ base):   refusals {off_ref}/{n}");
    Ok(())
}
