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
// hipfire — tokenizer-free multi-arch tiny-model probe (KLD + calibration).
//
// Drives `hipfire_serving_core::tiny_harness` for the `tiny_quant` eval battery
// and tiny gates. Three subcommands:
//
//   kld     --arch <fam> --ref <bf16.hfq> --cand <quant.hfq> [--len N --warmup W --seed S]
//           → prints `mean_kld:`, `max_kld:`, `n_scored:`, `finite:` lines.
//   collect --arch <fam> --model <bf16.hfq> --out <calib.hfq> [--len N --seed S]
//           → arms the model-agnostic Hessian/imatrix collector, runs the
//             forward, drains a `.calib.hfq` (HFQM). Prints `n_tensors:`,
//             `consistency:`, `calib_out:`.
//   ar-hash --arch <fam> --model <hfq> [--len N --prompt-len P --seed S]
//           → free-runs greedy decode after a deterministic prompt and prints
//             `logit_hash:` + `token_hash:`. This grows KV/state and is used by
//             tiny long-state/KV tripwires.
//
// Machine-readable `key: value` stdout lines; the battery parses these. A
// panic / nonzero exit is the hard-fail signal.

use std::path::Path;

use hipfire_serving_core::tiny_harness::{run_ar_hash, run_collect, run_kld, TinyArch};
use hipfire_rdna::Gpu;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn req(args: &[String], name: &str) -> String {
    flag(args, name).unwrap_or_else(|| {
        eprintln!("tiny_quant_probe: missing required flag {name}");
        std::process::exit(2);
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sub = args.first().cloned().unwrap_or_default();
    let arch = TinyArch::parse(&req(&args, "--arch")).unwrap_or_else(|e| {
        eprintln!("tiny_quant_probe: {e}");
        std::process::exit(2);
    });
    let len: usize = flag(&args, "--len")
        .map(|s| s.parse().unwrap())
        .unwrap_or(24);
    let seed: u64 = flag(&args, "--seed")
        .map(|s| s.parse().unwrap())
        .unwrap_or(42);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);

    match sub.as_str() {
        "kld" => {
            let warmup: usize = flag(&args, "--warmup")
                .map(|s| s.parse().unwrap())
                .unwrap_or(4);
            let r = req(&args, "--ref");
            let c = req(&args, "--cand");
            let out = run_kld(
                arch,
                Path::new(&r),
                Path::new(&c),
                &mut gpu,
                len,
                warmup,
                seed,
            )
            .unwrap_or_else(|e| {
                eprintln!("tiny_quant_probe kld: {e}");
                std::process::exit(1);
            });
            println!("arch: {}", arch.as_str());
            println!("mean_kld: {:.8}", out.mean_kld);
            println!("max_kld: {:.8}", out.max_kld);
            println!("n_scored: {}", out.n_scored);
            println!("finite: {}", out.finite);
        }
        "collect" => {
            let model = req(&args, "--model");
            let out = req(&args, "--out");
            let r = run_collect(
                arch,
                Path::new(&model),
                Path::new(&out),
                &mut gpu,
                len,
                seed,
            )
            .unwrap_or_else(|e| {
                eprintln!("tiny_quant_probe collect: {e}");
                std::process::exit(1);
            });
            println!("arch: {}", arch.as_str());
            println!("n_tensors: {}", r.n_tensors);
            println!("consistency: {:.6}", r.consistency);
            println!("calib_out: {}", r.out_path);
        }
        "ar-hash" | "ar_hash" => {
            let prompt_len: usize = flag(&args, "--prompt-len")
                .map(|s| s.parse().unwrap())
                .unwrap_or(4);
            let model = req(&args, "--model");
            let out = run_ar_hash(arch, Path::new(&model), &mut gpu, len, prompt_len, seed)
                .unwrap_or_else(|e| {
                    eprintln!("tiny_quant_probe ar-hash: {e}");
                    std::process::exit(1);
                });
            println!("arch: {}", arch.as_str());
            println!("logit_hash: 0x{:016x}", out.logit_hash);
            println!("token_hash: 0x{:016x}", out.token_hash);
            println!("n_steps: {}", out.n_steps);
            println!("prompt_len: {}", out.prompt_len);
            println!("last_token: {}", out.last_token);
        }
        // PROOF-OF-CONCEPT for re-pointing the probe at the REAL serving-core path:
        // load the tiny .hfq through the production `load_model` (not tiny_harness),
        // tokenize with its EMBEDDED tokenizer, and forward through the same
        // `ChunkScoredForward` seam the daemon's KLD uses. Finite logits ⇒ the tiny
        // model is a complete artifact that round-trips through production. (llama
        // only for now; generalizing = hoisting the daemon's backend-slot pick block,
        // then deleting tiny_harness.)
        "real-load" => {
            use hipfire_runtime::kld_eval::ChunkScoredForward;
            let model = req(&args, "--model");
            let prompt = flag(&args, "--prompt").unwrap_or_else(|| "hello hipfire tiny".into());
            let mut m = hipfire_serving_core::load::load_model(
                &model,
                len + 16,
                None,
                None,
                None,
                None,
                &hipfire_serving_core::model::CaskConfig::default(),
                1,
                &mut gpu,
            )
            .unwrap_or_else(|e| {
                eprintln!("tiny_quant_probe real-load: load_model: {e}");
                std::process::exit(1);
            });
            let ids = m
                .tokenizer
                .as_ref()
                .expect("tiny .hfq must carry an embedded tokenizer")
                .encode(&prompt);
            println!("tokenizer_ok: {}", m.tokenizer.is_some());
            println!("n_tokens: {}", ids.len());
            let fwd: &mut dyn ChunkScoredForward = m
                .llama_backend
                .as_mut()
                .expect("real-load proof currently wired for llama only");
            let vocab = fwd.kld_vocab_size();
            let (mut n_scored, mut n_finite) = (0usize, 0usize);
            fwd.forward_chunk_scored(&mut gpu, &ids, 0, &mut |_j, lg, _next| {
                n_scored += 1;
                if lg.len() == vocab && lg.iter().all(|x| x.is_finite()) {
                    n_finite += 1;
                }
            })
            .unwrap_or_else(|e| {
                eprintln!("tiny_quant_probe real-load: forward: {e}");
                std::process::exit(1);
            });
            println!("vocab: {vocab}");
            println!("n_scored: {n_scored}");
            println!("n_finite: {n_finite}");
        }
        other => {
            eprintln!(
                "tiny_quant_probe: unknown subcommand {other:?} (kld|collect|ar-hash|real-load)"
            );
            std::process::exit(2);
        }
    }
}
