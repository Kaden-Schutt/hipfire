//! Train the GLA-lite / minimal-selective-SSM PFlash drafter to reproduce the
//! qwen3.5 target's MID-layer block ranking, and measure it against the same
//! shallow-K bar + the +0.47 ATTENTION-drafter ceiling (P5). Same labels, same
//! production scoring, same ListNet/AdamW loop as `pflash_drafter_train` — only
//! the drafter body differs (gated recurrence vs attention).
//!
//! Thin client over `hipfire_train::train_loop` — the SAME loop the daemon
//! `train_drafter` op will call (docs/plans/2026-06-19-train-as-daemon-op.md).
//! This binary owns only label IO + shuffle; the loop owns epochs/loss/eval.
//!
//! Requires daemon labels (teacher/student split; real qwen3.5 target):
//!   HIPFIRE_PFLASH_DAEMON_LABELS=/tmp/pflash_q35_labels.jsonl \
//!   cargo run -p hipfire-train --release --example ssm_drafter_train
//!
//! Env knobs: HIPFIRE_PFLASH_{EPOCHS,TAU,LR,WD,NEVAL,SHUFFLE_SEED},
//!            HIPFIRE_SSM_LAYERS, HIPFIRE_SSM_H.

use hipfire_train::ssm_drafter::{SsmDrafter, SsmDrafterConfig};
use hipfire_train::train_loop::{eval_ssm_drafter, spearman, train_ssm_drafter_loop, TrainCfg};
use rdna_compute::Gpu;

const SEQ: usize = 512;
const BLOCK: usize = 64;
const N_EVAL: usize = 8;
const EPOCHS: usize = 300;
const TAU: f32 = 0.1;
const EVAL_EVERY: usize = 15;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    let nb = SEQ / BLOCK;

    let dlpath = std::env::var("HIPFIRE_PFLASH_DAEMON_LABELS")
        .map_err(|_| "set HIPFIRE_PFLASH_DAEMON_LABELS=<jsonl> (real qwen3.5 labels)")?;
    let n_eval = env_usize("HIPFIRE_PFLASH_NEVAL", N_EVAL);
    // ONE loader, shared with the daemon train_drafter op.
    let mut ls = hipfire_train::labels::load_daemon_labels(&mut gpu, &dlpath, SEQ)?;
    println!("daemon labels: {} chunks, embed [{}×{}]", ls.chunks.len(), ls.vocab, ls.h_t);
    let seed = env_usize("HIPFIRE_PFLASH_SHUFFLE_SEED", 0x5EED) as u64;
    hipfire_train::labels::shuffle_in_place(&mut ls.chunks, &mut ls.label_mid, &mut ls.base_shallow, seed);
    let (chunks, label_mid, base_shallow, embed, h_t, vocab) =
        (ls.chunks, ls.label_mid, ls.base_shallow, ls.embed, ls.h_t, ls.vocab);

    let n_chunks = chunks.len();
    let n_train = n_chunks.checked_sub(n_eval).filter(|&t| t > 0)
        .unwrap_or_else(|| panic!("n_chunks {n_chunks} ≤ n_eval {n_eval}"));

    // ── SSM drafter + shared training loop ──
    let mut dcfg = SsmDrafterConfig::tiny(10000.0, 1e-5);
    dcfg.n_layers = env_usize("HIPFIRE_SSM_LAYERS", 3);
    dcfg.h_draft = env_usize("HIPFIRE_SSM_H", 512);
    let drafter = SsmDrafter::new(&mut gpu, embed, h_t, vocab, dcfg, SEQ)?;
    let nparams: usize = drafter.param_sizes().iter().sum();

    let cfg = TrainCfg {
        seq: SEQ,
        block: BLOCK,
        n_eval,
        epochs: env_usize("HIPFIRE_PFLASH_EPOCHS", EPOCHS),
        lr: env_f32("HIPFIRE_PFLASH_LR", 1e-3),
        wd: env_f32("HIPFIRE_PFLASH_WD", 0.0),
        tau: env_f32("HIPFIRE_PFLASH_TAU", TAU),
        eval_every: EVAL_EVERY,
        report_train: std::env::var("HIPFIRE_PFLASH_REPORT_TRAIN").is_ok(),
    };

    println!("arch: {}  SEQ={SEQ} BLOCK={BLOCK} blocks={nb} train={n_train} eval={n_eval}", gpu.arch);
    println!("labels: daemon source {dlpath} (real qwen3.5 target)");
    println!(
        "SSM drafter: h={} layers={} inter={} kv={}×{}  params={} ({:.2}M)  epochs={} tau={} lr={} wd={}",
        dcfg.h_draft, dcfg.n_layers, dcfg.inter, dcfg.n_kv, dcfg.head_dim,
        drafter.param_sizes().len(), nparams as f32 / 1e6, cfg.epochs, cfg.tau, cfg.lr, cfg.wd
    );

    let bar: f32 = (n_train..n_chunks).map(|i| spearman(&base_shallow[i], &label_mid[i])).sum::<f32>()
        / n_eval as f32;
    println!("\n  bar  Spearman(shallow, mid)   [eval] = {bar:+.3}  ← drafter must beat this");
    println!("  ref  attention-drafter ceiling       ≈ +0.47  (P5, tuning-resistant)");
    let init = eval_ssm_drafter(&mut gpu, &drafter, &chunks, &label_mid, &cfg);
    println!("  init Spearman(ssm-drafter, mid)[eval] = {init:+.3}\n");

    let report = train_ssm_drafter_loop(
        &mut gpu,
        &drafter,
        &chunks,
        &label_mid,
        &base_shallow,
        &cfg,
        |ep, train_loss, corr, best, best_ep, train_corr| {
            // Print EVERY eval epoch and FLUSH — block-buffering when piped left
            // prior runs unobservable for hours. Always flush; never gate prints.
            use std::io::Write;
            let tc = train_corr.map(|t| format!("  train_ρ {t:+.3}")).unwrap_or_default();
            println!("  ep {ep:>3}  train_loss {train_loss:.4}  eval {corr:+.3}{tc}  (best {best:+.3} @ ep {best_ep})");
            let _ = std::io::stdout().flush();
        },
    )?;

    println!("\n── SSM drafter result ──");
    println!("  shallow bar       : {:+.3}", report.bar);
    println!("  attn ceiling (P5) : ≈ +0.47");
    println!("  SSM drafter BEST  : {:+.3} @ ep {}", report.best_eval, report.best_epoch);
    if report.best_eval > report.bar {
        println!("  ✓ SSM drafter BEATS the shallow bar");
    } else if report.best_eval > 0.47 {
        println!("  ~ SSM drafter beats the attn ceiling but not the shallow bar");
    } else {
        println!("  ✗ SSM drafter did not beat the attn ceiling — ablate up (conv1d / delta rule)");
    }
    Ok(())
}
