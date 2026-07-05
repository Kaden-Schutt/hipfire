#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_div_ceil
)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! DSpark training-LABEL generation (native trainer, T4 — FORWARD only).
//!
//! Runs a DENSE Qwen3 (LLaMA-family) target's prefill forward with hidden-state
//! capture and produces the label cache a DSpark drafter (T5) trains against —
//! no external DeepSpec / 38 TB cache. This is the T4→T5 interface: it is
//! forward-only (no backward) and independent of the drafter's backward.
//!
//! Pipeline, per prompt (a pre-tokenized u32 sequence):
//!   1. one `forward_prefill_batch_capture` over the whole prompt with a
//!      [`HiddenCaptureSink`] whose `extract_layers` = the sorted-unique union of
//!      the requested `--target-layers` and the final decoder layer
//!      (`n_layers-1`, needed for the lm-head). The sink appends the post-FFN
//!      residual at each extract layer, laid out `[n_pos × num_extract × dim]`.
//!   2. sample anchor windows: a `ctx_len`-long context immediately followed by a
//!      `block`-long draft region, slid by `--stride`.
//!   3. per window emit the labels below and stream them to the cache.
//!
//! Target load + capture APIs used (all `hipfire-runtime`):
//!   * `hfq::HfqFile::open` + `hfq::config_from_hfq` + `hfq::load_weights_hfq`
//!     (a runnable, capture-compatible target `LlamaWeights` / `LlamaConfig`).
//!   * `llama::{ForwardScratch, PrefillBatchScratch, HiddenCaptureSink,
//!     forward_prefill_batch_capture}` and `kv::KvCache::new_gpu_q8`.
//!   * `llama_spec::lm_head_logits_n_rows` for the target lm-head logits over the
//!     captured final-layer hidden at the block positions.
//!
//! ── Label-cache binary format (`DSLB`, version 1) — the T4↔T5 interface ──────
//!
//! Little-endian throughout. No external serialization. Shared frozen weights
//! (token embedding + lm-head) are NOT dumped; the header stores the target path
//! and T5 reloads them from it (e.g. `hipfire_train::loader::load_llama_from_hfq`).
//! This mirrors the `PFLB`/`QEMB` cache style in `src/checkpoint.rs`/`src/labels.rs`
//! but references, rather than copies, the (large) shared matrices.
//!
//! HEADER (fixed-offset prefix, then two variable-length fields):
//! ```text
//! off  bytes  field
//!   0    4    magic          = b"DSLB"
//!   4    4    version   u32  = 1
//!   8    4    vocab     u32       (== target vocab_size, logits width)
//!  12    4    dim       u32       (== target hidden dim)
//!  16    4    n_targets u32       (== target_layer_ids.len())
//!  20    4    block     u32       (draft positions per window)
//!  24    4    ctx_len   u32       (context positions per window)
//!  28    4    flags     u32  = 0  (reserved)
//!  32    4    n_windows u32       (patched after streaming; total windows)
//!  36  4+4*k  target_layer_ids: u32 count=k, then k× u32 layer id (ascending)
//!   -   4+L   target_path: u32 byte-len L, then L bytes UTF-8 (the target .hfq)
//! ```
//! Then `n_windows` records, each (all sizes derive from the header):
//! ```text
//!   main_hidden        f32 [ctx_len * n_targets * dim]  row-major
//!                        row p (0..ctx_len) = concat over target_layer_ids
//!                        (ascending) of the target hidden at that ctx position;
//!                        feeds the drafter main_proj / context ingest.
//!   target_logits      f32 [block * vocab]              target lm-head soft labels
//!   target_next_tokens i32 [block]                      argmax(target_logits) per
//!                        block pos; -100 at invalid positions (loss ignore_index)
//!   block_tokens       u32 [block]                      actual token at each block pos
//!   prev_tokens        u32 [block]                      token immediately BEFORE each
//!                        block pos (VanillaMarkov step token)
//!   eval_mask          u8  [block]                      1 valid / 0 past sequence end
//! ```
//! `main_hidden`/`target_logits` map onto `dspark_drafter_forward_train`'s
//! `main_hidden` and `dspark_loss_forward_backward`'s `target_logits`;
//! `target_next_tokens` onto the CE hard label; `prev_tokens` onto the markov
//! head; `eval_mask` onto the loss weight mask. Drafter RoPE positions are
//! window-relative (ctx = `0..ctx_len`, block = `ctx_len..ctx_len+block`) and are
//! reconstructed by T5 from `ctx_len`/`block`, so they are not stored.
//!
//! Usage:
//! ```text
//! cargo run --release -p hipfire-train --example dspark_labels -- \
//!   --target <model.hfq> --prompts <toks.jsonl|.txt> \
//!   --target-layers 1,9,17,25,33 --block 7 --ctx-len 128 \
//!   [--stride N] [--max-windows N] --out <label_cache.dslb>
//! ```
//! `--prompts`: one token sequence per line — either a JSON array of ints /
//! `{"tokens":[...]}` (`.jsonl`) or whitespace-separated ints (`.txt`).
//!
//! NOTE: forward-only; runs on GPU (validate on `halo`, not the LDS-hazard box).

use hipfire_rdna::{Gpu, GpuTensor, HipResult};
use hipfire_runtime::hfq::{self, HfqFile};
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::llama::{self, ForwardScratch, HiddenCaptureSink, PrefillBatchScratch};
use hipfire_runtime::llama_spec::lm_head_logits_n_rows;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

// ── little-endian writers (example-local; mirror checkpoint.rs style) ────────
fn w_u32(w: &mut impl Write, x: u32) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn w_i32(w: &mut impl Write, x: i32) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn w_f32s(w: &mut impl Write, v: &[f32]) -> std::io::Result<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    w.write_all(bytes)
}

// ── CLI ──────────────────────────────────────────────────────────────────────
struct Args {
    target: String,
    prompts: String,
    target_layers: Vec<usize>,
    block: usize,
    ctx_len: usize,
    stride: usize,
    max_windows: usize,
    out: String,
}

fn parse_args() -> Args {
    let mut target = None;
    let mut prompts = None;
    let mut out = None;
    let mut target_layers: Vec<usize> = vec![1, 9, 17, 25, 33];
    let mut block = 7usize;
    let mut ctx_len = 128usize;
    let mut stride: Option<usize> = None;
    let mut max_windows = usize::MAX;

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
            "--target" => target = Some(next()),
            "--prompts" => prompts = Some(next()),
            "--out" => out = Some(next()),
            "--target-layers" => {
                target_layers = next()
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.trim().parse::<usize>().expect("bad --target-layers"))
                    .collect();
            }
            "--block" => block = next().parse().expect("bad --block"),
            "--ctx-len" => ctx_len = next().parse().expect("bad --ctx-len"),
            "--stride" => stride = Some(next().parse().expect("bad --stride")),
            "--max-windows" => max_windows = next().parse().expect("bad --max-windows"),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let missing = |name: &str| -> ! {
        eprintln!(
            "Usage: dspark_labels --target <hfq> --prompts <jsonl|txt> \
             --target-layers 1,9,17,25,33 --block 7 --ctx-len 128 \
             [--stride N] [--max-windows N] --out <cache>\nmissing --{name}"
        );
        std::process::exit(2);
    };
    Args {
        target: target.unwrap_or_else(|| missing("target")),
        prompts: prompts.unwrap_or_else(|| missing("prompts")),
        target_layers,
        block,
        ctx_len,
        stride: stride.unwrap_or(block),
        max_windows,
        out: out.unwrap_or_else(|| missing("out")),
    }
}

// ── prompt loading: one token sequence per line ──────────────────────────────
fn load_prompts(path: &str) -> Vec<Vec<u32>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("read prompts {path}: {e}");
        std::process::exit(1);
    });
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let toks: Vec<u32> = if t.starts_with('[') || t.starts_with('{') {
            let v: serde_json::Value = match serde_json::from_str(t) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("skip bad json line: {e}");
                    continue;
                }
            };
            let arr = if v.is_array() {
                v.as_array().cloned().unwrap_or_default()
            } else {
                v["tokens"].as_array().cloned().unwrap_or_default()
            };
            arr.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect()
        } else {
            t.split_whitespace()
                .filter_map(|s| s.parse::<u32>().ok())
                .collect()
        };
        if !toks.is_empty() {
            out.push(toks);
        }
    }
    out
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bestv = f32::NEG_INFINITY;
    for (j, &x) in row.iter().enumerate() {
        if x > bestv {
            bestv = x;
            best = j;
        }
    }
    best
}

fn main() -> HipResult<()> {
    let args = parse_args();

    // ── load the dense target (runtime .hfq loader → capture-compatible) ─────
    let hfq = HfqFile::open(Path::new(&args.target)).expect("open target .hfq");
    let config = hfq::config_from_hfq(&hfq).expect("config from hfq");
    let dim = config.dim;
    let vocab = config.vocab_size;
    let n_layers = config.n_layers;
    eprintln!(
        "target: dim={dim} layers={n_layers} heads={} kv_heads={} vocab={vocab}",
        config.n_heads, config.n_kv_heads
    );

    // Requested target layers, clamped to valid decoder indices, ascending-unique.
    let mut target_layers: Vec<usize> = args
        .target_layers
        .iter()
        .map(|&l| l.min(n_layers - 1))
        .collect();
    target_layers.sort_unstable();
    target_layers.dedup();
    let n_targets = target_layers.len();
    assert!(n_targets > 0, "no target layers");

    // Extract set = target layers ∪ final layer (final residual feeds the lm-head).
    let lm_layer = n_layers - 1;
    let mut extract: Vec<usize> = target_layers.clone();
    if !extract.contains(&lm_layer) {
        extract.push(lm_layer);
    }
    extract.sort_unstable();
    extract.dedup();
    let num_extract = extract.len();
    let col_of = |layer: usize| -> usize { extract.iter().position(|&x| x == layer).unwrap() };
    let target_cols: Vec<usize> = target_layers.iter().map(|&l| col_of(l)).collect();
    let lm_col = col_of(lm_layer);

    let prompts = load_prompts(&args.prompts);
    assert!(!prompts.is_empty(), "no prompts");
    let max_len = prompts.iter().map(|p| p.len()).max().unwrap();
    eprintln!(
        "{} prompts (max len {max_len}); ctx_len={} block={} stride={} extract_layers={:?}",
        prompts.len(),
        args.ctx_len,
        args.block,
        args.stride,
        extract
    );

    // ── GPU setup ────────────────────────────────────────────────────────────
    let mut gpu = Gpu::init().expect("GPU init");
    let weights = hfq::load_weights_hfq(&hfq, &config, &mut gpu).expect("load target weights");
    let scratch = ForwardScratch::new(&mut gpu, &config).expect("scratch");
    let kv_cap = max_len.max(args.ctx_len + args.block);
    let mut kv = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_cap,
    )
    .expect("kv");
    let pbs = PrefillBatchScratch::new(
        &mut gpu,
        &config,
        llama::PREFILL_MAX_BATCH.min(max_len.max(4)),
        kv_cap,
    )
    .expect("pbs");

    // ── output cache: write header, stream windows, patch n_windows ──────────
    let mut f = std::io::BufWriter::new(std::fs::File::create(&args.out).expect("create out"));
    f.write_all(b"DSLB").unwrap();
    w_u32(&mut f, 1).unwrap(); // version
    w_u32(&mut f, vocab as u32).unwrap();
    w_u32(&mut f, dim as u32).unwrap();
    w_u32(&mut f, n_targets as u32).unwrap();
    w_u32(&mut f, args.block as u32).unwrap();
    w_u32(&mut f, args.ctx_len as u32).unwrap();
    w_u32(&mut f, 0).unwrap(); // flags
    w_u32(&mut f, 0).unwrap(); // n_windows placeholder @ offset 32
    w_u32(&mut f, n_targets as u32).unwrap();
    for &l in &target_layers {
        w_u32(&mut f, l as u32).unwrap();
    }
    let tpath = args.target.as_bytes();
    w_u32(&mut f, tpath.len() as u32).unwrap();
    f.write_all(tpath).unwrap();

    let mut n_windows: u32 = 0;
    let row_stride = num_extract * dim; // f32 per captured position

    'outer: for tokens in &prompts {
        let l = tokens.len();
        if l < args.ctx_len + args.block {
            continue;
        }

        // One capturing prefill over the whole prompt (start_pos=0 overwrites KV
        // slots 0..l; leftover from a prior prompt beyond l is never read).
        let mut hidden: Vec<f32> = Vec::with_capacity(l * row_stride);
        // Per-token capturing forward (start_pos=0). The batched prefill path
        // requires the model be batch-eligible (Q8-KV etc.); a plain dense
        // Qwen3 target is not, so capture one token at a time — the per-token
        // capture appends rows in the same extract-layer-ascending layout.
        for (pos, &tok) in tokens.iter().enumerate() {
            llama::forward_scratch_embed(&mut gpu, &weights, &config, tok, pos, &scratch)?;
            let mut sink = HiddenCaptureSink {
                extract_layers: &extract,
                hidden: &mut hidden,
                hidden_gpu: None,
            };
            llama::forward_scratch_compute_capture(
                &mut gpu,
                &weights,
                &config,
                pos,
                &mut kv,
                &scratch,
                Some(&mut sink),
            )?;
        }
        let _ = &pbs;
        assert_eq!(hidden.len(), l * row_stride, "capture size mismatch");

        // Slide anchor windows.
        let last_off = l - args.ctx_len - args.block;
        let mut off = 0usize;
        while off <= last_off {
            // main_hidden [ctx_len * n_targets * dim] — concat target layers per pos.
            let mut main_hidden = vec![0.0f32; args.ctx_len * n_targets * dim];
            for p in 0..args.ctx_len {
                let src_base = (off + p) * row_stride;
                let dst_base = p * n_targets * dim;
                for (t, &col) in target_cols.iter().enumerate() {
                    let s = src_base + col * dim;
                    main_hidden[dst_base + t * dim..dst_base + (t + 1) * dim]
                        .copy_from_slice(&hidden[s..s + dim]);
                }
            }

            // Final-layer hidden at the block positions → lm-head → target_logits.
            let mut block_final = vec![0.0f32; args.block * dim];
            for i in 0..args.block {
                let s = (off + args.ctx_len + i) * row_stride + lm_col * dim;
                block_final[i * dim..(i + 1) * dim].copy_from_slice(&hidden[s..s + dim]);
            }
            let block_final_gpu: GpuTensor = gpu.upload_f32(&block_final, &[args.block * dim])?;
            let target_logits = lm_head_logits_n_rows(
                &mut gpu,
                &weights,
                &config,
                &block_final_gpu,
                args.block,
                &scratch,
            )?;
            gpu.free_tensor(block_final_gpu)?;
            debug_assert_eq!(target_logits.len(), args.block * vocab);

            // Hard labels + markov/eval side-info.
            let mut next_tokens = vec![0i32; args.block];
            let mut block_tokens = vec![0u32; args.block];
            let mut prev_tokens = vec![0u32; args.block];
            let mut eval_mask = vec![0u8; args.block];
            for i in 0..args.block {
                let pos = off + args.ctx_len + i;
                let valid = pos < l;
                eval_mask[i] = valid as u8;
                block_tokens[i] = tokens[pos];
                prev_tokens[i] = tokens[pos - 1];
                next_tokens[i] = if valid {
                    argmax(&target_logits[i * vocab..(i + 1) * vocab]) as i32
                } else {
                    -100
                };
            }

            // Emit the window.
            w_f32s(&mut f, &main_hidden).unwrap();
            w_f32s(&mut f, &target_logits).unwrap();
            for &t in &next_tokens {
                w_i32(&mut f, t).unwrap();
            }
            for &t in &block_tokens {
                w_u32(&mut f, t).unwrap();
            }
            for &t in &prev_tokens {
                w_u32(&mut f, t).unwrap();
            }
            f.write_all(&eval_mask).unwrap();

            n_windows += 1;
            if n_windows as usize >= args.max_windows {
                break 'outer;
            }
            off += args.stride;
        }
    }

    // Patch n_windows at fixed offset 32.
    f.flush().unwrap();
    let mut inner = f.into_inner().expect("into_inner");
    inner.seek(SeekFrom::Start(32)).unwrap();
    inner.write_all(&n_windows.to_le_bytes()).unwrap();
    inner.flush().unwrap();

    pbs.free_gpu(&mut gpu);
    let _ = &mut kv; // kv buffers are GPU-pool owned; process exit reclaims

    eprintln!(
        "wrote {n_windows} windows to {} (vocab={vocab} dim={dim} n_targets={n_targets} block={} ctx_len={})",
        args.out, args.block, args.ctx_len
    );
    Ok(())
}
