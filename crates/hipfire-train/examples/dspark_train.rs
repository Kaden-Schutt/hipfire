// SPDX-License-Identifier: Apache-2.0
//! DSpark drafter training CLI (T5a).
//!
//! Trains a 5-layer DSpark drafter (body + markov/confidence heads) against a
//! `DSLB v1` label cache (written by `examples/dspark_labels.rs`), using the
//! frozen target's shared embedding + lm-head. Produces a resumable `DSCK`
//! checkpoint. Export to `.dspark.hfq` is a separate follow-up.
//!
//! Usage:
//! ```text
//! cargo run --release -p hipfire-train --example dspark_train -- \
//!   --labels <cache.dslb> [--target <model.hfq>] [--out <ckpt.dsck>] \
//!   --epochs N --lr F [--wd F] [--markov-rank R] [--eval-frac F] \
//!   [--checkpoint-every N] [--resume] [--overfit]
//! ```
//! `--target` defaults to the path recorded in the DSLB header.
//!
//! `--overfit` is the built-in correctness check: it trains on just the first
//! 1-2 windows for `--epochs` steps (eval on the same windows) and asserts the
//! loss drops substantially, printing `OVERFIT ok: loss A -> B`. Run this on a
//! GPU box to validate the loop end-to-end.

use hipfire_rdna::{Gpu, HipResult};
use hipfire_train::dspark_loss::DsparkLossCfg;
use hipfire_train::dspark_train::{
    init_dspark_model, load_dslb, load_dspark_ckpt, load_weights_into, save_dspark_ckpt,
    train_dspark_loop, DsparkDrafterCfg, DsparkTrainCfg,
};
use hipfire_train::loader::load_llama_from_hfq;
use hipfire_train::optim::AdamW;
use std::path::Path;

struct Args {
    labels: String,
    target: Option<String>,
    out: Option<String>,
    epochs: usize,
    lr: f32,
    wd: f32,
    markov_rank: usize,
    eval_frac: f32,
    checkpoint_every: usize,
    resume: bool,
    overfit: bool,
}

fn parse_args() -> Args {
    let mut labels = None;
    let mut target = None;
    let mut out = None;
    let mut epochs = 100usize;
    let mut lr = 1e-3f32;
    let mut wd = 0.0f32;
    let mut markov_rank = 256usize;
    let mut eval_frac = 0.1f32;
    let mut checkpoint_every = 10usize;
    let mut resume = false;
    let mut overfit = false;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let a = argv[i].as_str();
        let mut next = || {
            i += 1;
            argv.get(i)
                .unwrap_or_else(|| {
                    eprintln!("missing value for {a}");
                    std::process::exit(2);
                })
                .clone()
        };
        match a {
            "--labels" => labels = Some(next()),
            "--target" => target = Some(next()),
            "--out" => out = Some(next()),
            "--epochs" => epochs = next().parse().expect("bad --epochs"),
            "--lr" => lr = next().parse().expect("bad --lr"),
            "--wd" => wd = next().parse().expect("bad --wd"),
            "--markov-rank" => markov_rank = next().parse().expect("bad --markov-rank"),
            "--eval-frac" => eval_frac = next().parse().expect("bad --eval-frac"),
            "--checkpoint-every" => {
                checkpoint_every = next().parse().expect("bad --checkpoint-every")
            }
            "--resume" => resume = true,
            "--overfit" => overfit = true,
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let labels = labels.unwrap_or_else(|| {
        eprintln!(
            "Usage: dspark_train --labels <dslb> [--target <hfq>] [--out <ckpt>] \
             --epochs N --lr F [--wd F] [--markov-rank R] [--eval-frac F] \
             [--checkpoint-every N] [--resume] [--overfit]\nmissing --labels"
        );
        std::process::exit(2);
    });
    Args {
        labels,
        target,
        out,
        epochs,
        lr,
        wd,
        markov_rank,
        eval_frac,
        checkpoint_every,
        resume,
        overfit,
    }
}

fn main() -> HipResult<()> {
    let args = parse_args();

    // ── label cache ───────────────────────────────────────────────────────────
    let mut cache = load_dslb(&args.labels).unwrap_or_else(|e| {
        eprintln!("load DSLB {}: {e}", args.labels);
        std::process::exit(1);
    });
    eprintln!(
        "cache: {} windows  vocab={} dim={} n_targets={} block={} ctx_len={} target_layers={:?}",
        cache.n_windows(),
        cache.vocab,
        cache.dim,
        cache.n_targets,
        cache.block,
        cache.ctx_len,
        cache.target_layer_ids,
    );
    let target_path = args
        .target
        .clone()
        .unwrap_or_else(|| cache.target_path.clone());

    // ── GPU + frozen target (shared embed + lm-head) ─────────────────────────
    let mut gpu = Gpu::init().expect("GPU init");
    let (target_cfg, target) = load_llama_from_hfq(&mut gpu, Path::new(&target_path))
        .unwrap_or_else(|e| {
            eprintln!("load target {target_path}: {e}");
            std::process::exit(1);
        });
    assert_eq!(
        cache.dim, target_cfg.hidden_size,
        "cache dim {} != target hidden {}",
        cache.dim, target_cfg.hidden_size
    );
    assert_eq!(
        cache.vocab, target_cfg.vocab_size,
        "cache vocab {} != target vocab {}",
        cache.vocab, target_cfg.vocab_size
    );

    // ── drafter model + optimizer ────────────────────────────────────────────
    let dcfg = DsparkDrafterCfg {
        markov_rank: args.markov_rank,
        ..DsparkDrafterCfg::default()
    };
    let model = init_dspark_model(&mut gpu, &target_cfg, cache.n_targets, cache.block, &dcfg)?;
    let mut opt = AdamW::new(
        &mut gpu,
        &model.param_sizes(),
        args.lr,
        0.9,
        0.999,
        1e-8,
        args.wd,
    )?;
    eprintln!(
        "drafter: {} params ({} tensors)  markov_rank={}",
        model.param_sizes().iter().sum::<usize>(),
        model.param_sizes().len(),
        args.markov_rank,
    );

    if args.resume {
        if let Some(path) = &args.out {
            match load_dspark_ckpt(&mut gpu, path, &model, &mut opt)? {
                Some(ep) => eprintln!("resumed from {path} @ epoch {ep}"),
                None => eprintln!("no checkpoint at {path}; starting fresh"),
            }
        }
    }

    // ── train config ─────────────────────────────────────────────────────────
    let mut eval_frac = args.eval_frac;
    if args.overfit {
        let k = cache.n_windows().min(2).max(1);
        cache.truncate(k);
        eval_frac = 0.0; // eval == train windows in overfit mode
        eprintln!(
            "OVERFIT mode: training on first {k} window(s) for {} epochs",
            args.epochs
        );
    }

    let cfg = DsparkTrainCfg {
        epochs: args.epochs,
        lr: args.lr,
        weight_decay: args.wd,
        block_size: cache.block,
        loss: DsparkLossCfg::with_block_size(cache.block),
        eval_frac,
        checkpoint_every: args.checkpoint_every,
        seed: 0,
    };

    // ── run ──────────────────────────────────────────────────────────────────
    // checkpoint only when NOT overfitting (overfit is a throwaway sanity run).
    let ckpt_path = if args.overfit {
        None
    } else {
        args.out.as_deref()
    };
    let mut first_loss: Option<f32> = None;
    let mut last_loss = 0.0f32;
    let report = train_dspark_loop(
        &mut gpu,
        &model,
        &target,
        &cache,
        &mut opt,
        &cfg,
        ckpt_path,
        |ep, train_loss, eval_loss, best_eval, best_epoch, accept| {
            if first_loss.is_none() {
                first_loss = Some(train_loss);
            }
            last_loss = train_loss;
            if ep % 10 == 0 || ep + 1 == cfg.epochs {
                eprintln!(
                    "  ep {:>4}/{}  train {:.4}  eval {:.4}  best {:.4}@{}  accept {:.3}",
                    ep + 1,
                    cfg.epochs,
                    train_loss,
                    eval_loss,
                    best_eval,
                    best_epoch,
                    accept,
                );
            }
        },
    )?;

    eprintln!(
        "done: best_eval_loss {:.4} @ epoch {}  accept {:.3}  final_train {:.4}  final_eval {:.4}",
        report.best_eval_loss,
        report.best_epoch,
        report.best_accept,
        report.final_train_loss,
        report.final_eval_loss,
    );

    // ── overfit assertion ─────────────────────────────────────────────────────
    if args.overfit {
        let a = first_loss.unwrap_or(last_loss);
        let b = last_loss;
        // "substantial" = at least halved (and strictly lower).
        if b < 0.5 * a {
            println!("OVERFIT ok: loss {a:.4} -> {b:.4}");
        } else {
            println!(
                "OVERFIT FAIL: loss {a:.4} -> {b:.4} (expected < {:.4})",
                0.5 * a
            );
            std::process::exit(1);
        }
        return Ok(());
    }

    // ── persist the BEST (generalizing) model ─────────────────────────────────
    if let Some(path) = &args.out {
        if !report.best_weights.is_empty() {
            load_weights_into(&mut gpu, &model, &report.best_weights)?;
        }
        save_dspark_ckpt(&mut gpu, path, &model, &opt, report.best_epoch as u32)?;
        eprintln!("wrote checkpoint {path} (best epoch {})", report.best_epoch);
    }

    Ok(())
}
