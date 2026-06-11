// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// DFlash FR3/FR4: streaming trainer for the LFM2.5-350M DFlash drafter on REAL
// data — warm-start body + real Qwen3.6-27B target hiddens (context) + faithful
// frozen lm_head + real tokens. Multi-sequence/multi-block sampling, block-
// diffusion position-weighted CE, AdamW with warmup+cosine, pool-scoped memory
// (stable over thousands of steps), periodic PROXY-τ eval (mean accepted prefix
// = argmax-vs-target leading run on held-out blocks ≈ spec-decode acceptance),
// and checkpointing.
//
//   dflash_train_run <embed.f32> <lmhead.f32> <data_dir> <kldref> <out_ckpt>
//     [steps] [lr] [n_train_chunks]
// data_dir holds chunk{c}.hfhs (5-layer dumps); kldref provides tokens per chunk.

use hipfire_arch_lfm2moe::dflash_train as dt;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const SEL: [usize; 5] = [2, 16, 31, 46, 61];

fn read_head(path: &str) -> (usize, usize, Vec<f32>) {
    let mut f = std::fs::File::open(path).expect("head");
    let mut hdr = [0u8; 16]; f.read_exact(&mut hdr).unwrap();
    assert_eq!(&hdr[0..8], b"DFHEAD\0\0");
    let vocab = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let mut raw = Vec::new(); f.read_to_end(&mut raw).unwrap();
    (vocab, dim, raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
fn read_hfhs(path: &str) -> (usize, usize, Vec<Vec<f32>>) {
    let mut f = std::fs::File::open(path).unwrap_or_else(|_| panic!("hfhs {path}"));
    let mut hdr = [0u8; 24]; f.read_exact(&mut hdr).unwrap();
    assert_eq!(&hdr[0..8], b"HFHS\0\0\0\0");
    let n_layers = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let n_pos = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let hidden = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
    let lb = n_pos * hidden * 4;
    let mut out = Vec::new();
    // 5-layer bulk dumps store the SEL layers as indices 0..5; legacy 64-layer
    // dumps need SEL indexing.
    let sel: Vec<usize> = if n_layers == SEL.len() { (0..SEL.len()).collect() } else { SEL.to_vec() };
    for &l in &sel {
        assert!(l < n_layers);
        f.seek(SeekFrom::Start(24 + (l * lb) as u64)).unwrap();
        let mut raw = vec![0u8; lb]; f.read_exact(&mut raw).unwrap();
        out.push(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
    }
    (n_pos, hidden, out)
}
fn read_toks(path: &str) -> Vec<u32> {
    let mut f = std::fs::File::open(path).unwrap_or_else(|_| panic!("toks {path}"));
    let mut raw = Vec::new(); f.read_to_end(&mut raw).unwrap();
    raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn up(gpu: &mut Gpu, v: &[f32], sh: &[usize]) -> GpuTensor { gpu.upload_f32(v, sh).unwrap() }
fn zt(gpu: &mut Gpu, n: usize) -> GpuTensor { gpu.zeros(&[n], DType::F32).unwrap() }
fn rndv(gpu: &mut Gpu, rows: usize, cols: usize, seed: usize, sc: f32) -> GpuTensor {
    let v: Vec<f32> = (0..rows * cols).map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 2000) as f32 / 1000.0 - 1.0) * sc).collect();
    up(gpu, &v, &[rows, cols])
}
// lm_head GEMM over vocab=248K is the training bottleneck. Use the register-
// tiled kernel (reads each 5GB weight row once per 8-batch-tile, ~8x less
// bandwidth) — validated bit-equal to the naive linear (test_lmhead_tiled).
fn lmhead_fwd(gpu: &mut Gpu, out: &GpuTensor, lmhead: &GpuTensor, lmhead_bf16: Option<&GpuTensor>, b: usize, d_tgt: usize, vocab: usize) -> GpuTensor {
    let logits = gpu.zeros(&[b, vocab], DType::F32).unwrap();
    if let Some(lmh) = lmhead_bf16 {
        let out_bf16 = gpu.zeros(&[b, d_tgt], DType::F16).unwrap();
        gpu.to_bf16_f32(out, &out_bf16, b * d_tgt).unwrap();
        gpu.gemm_bf16_mfma_splitk(lmh, &out_bf16, &logits, vocab, d_tgt, b).unwrap();
    } else {
        gpu.gemm_f32_register_tiled(lmhead, out, &logits, vocab, d_tgt, b).unwrap();
    }
    logits
}
fn lmhead_bwd(gpu: &mut Gpu, dlogits: &GpuTensor, lmhead_t: &GpuTensor, lmhead_t_bf16: Option<&GpuTensor>, b: usize, d_tgt: usize, vocab: usize) -> GpuTensor {
    let d_out = gpu.zeros(&[b, d_tgt], DType::F32).unwrap();
    if let Some(lmh_t) = lmhead_t_bf16 {
        let dl_bf16 = gpu.zeros(&[b, vocab], DType::F16).unwrap();
        gpu.to_bf16_f32(dlogits, &dl_bf16, b * vocab).unwrap();
        gpu.gemm_bf16_mfma_splitk(lmh_t, &dl_bf16, &d_out, d_tgt, vocab, b).unwrap();
    } else {
        gpu.gemm_f32_register_tiled(lmhead_t, dlogits, &d_out, d_tgt, vocab, b).unwrap();
    }
    d_out
}

fn zeros_like(gpu: &mut Gpu, src: &dt::Net) -> dt::Net {
    let mk = |g: &mut Gpu, t: &GpuTensor| g.zeros(&[t.numel()], DType::F32).unwrap();
    let layers = src.layers.iter().map(|l| dt::LW {
        op_norm: mk(gpu, &l.op_norm), ffn_norm: mk(gpu, &l.ffn_norm),
        in_proj: l.in_proj.as_ref().map(|t| mk(gpu, t)), conv_w: l.conv_w.as_ref().map(|t| mk(gpu, t)),
        out_proj: l.out_proj.as_ref().map(|t| mk(gpu, t)),
        wq: l.wq.as_ref().map(|t| mk(gpu, t)), wk: l.wk.as_ref().map(|t| mk(gpu, t)),
        wv: l.wv.as_ref().map(|t| mk(gpu, t)), wo: l.wo.as_ref().map(|t| mk(gpu, t)),
        q_norm: l.q_norm.as_ref().map(|t| mk(gpu, t)), k_norm: l.k_norm.as_ref().map(|t| mk(gpu, t)),
        w_c: l.w_c.as_ref().map(|t| mk(gpu, t)),
        w1: mk(gpu, &l.w1), w3: mk(gpu, &l.w3), w2: mk(gpu, &l.w2),
    }).collect();
    dt::Net { layers, in_proj_v: mk(gpu, &src.in_proj_v), out_proj_v: mk(gpu, &src.out_proj_v), fc: mk(gpu, &src.fc), final_norm: mk(gpu, &src.final_norm), hidden_norm: None }
}

// deterministic xorshift PRNG (Math.random unavailable; seedable + replayable)
struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x } fn usize(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize } }

struct Seq { layers5: Vec<Vec<f32>>, tokens: Vec<u32>, n_pos: usize }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let embed_path = a.get(1).map(|s| s.as_str()).unwrap_or("/workspace/qwen3.6-27b.head.f32");
    let lmhead_path = a.get(2).map(|s| s.as_str()).unwrap_or("/workspace/qwen3.6-27b.lmhead.f32");
    let data_dir = a.get(3).map(|s| s.as_str()).unwrap_or("/workspace/dflash-regen");
    let kldref = a.get(4).map(|s| s.as_str()).unwrap_or("/workspace/qwen3.6-27b-native.kldref.bin");
    let out_ckpt = a.get(5).map(|s| s.to_string()).unwrap_or("/workspace/lfm2-dflash-draft.dfnet".to_string());
    let steps: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let lr_max: f32 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(2e-4);
    let n_chunks: usize = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(16);
    let st_path = "/root/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M-Base/snapshots/9960764e30892e01f29a6dc23df2533fcd8bd5ae/model.safetensors";

    let mut gpu = Gpu::init().expect("gpu");
    let (vocab, d_tgt, embed) = read_head(embed_path);
    let (lvocab, ldim, lmhead) = read_head(lmhead_path);
    assert_eq!((lvocab, ldim), (vocab, d_tgt), "lmhead/embed dim mismatch");
    let cfg = dt::Cfg::lfm2_350m(d_tgt, vocab, 5);
    let d = cfg.d; let fc_in = 5 * d_tgt; let ws = 1.0 / (d as f32).sqrt();
    println!("vocab={vocab} d_tgt={d_tgt} d={d} steps={steps} lr_max={lr_max} chunks={n_chunks}");

    // load sequences (glob *.hfhs; works for seq{i} bulk + chunk{c}). Filter
    // out degenerate greedy generations (low unique-token ratio = loops).
    let _ = kldref;
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(data_dir).unwrap()
        .filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "hfhs")).collect();
    paths.sort();
    // pure-greedy corpora loop legitimately (production spec-decode is greedy);
    // filter only the fully-degenerate tail. Env-tunable.
    let uniq_min: f32 = std::env::var("HIPFIRE_UNIQ_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
    let mut seqs = Vec::new();
    let mut skipped = 0;
    for hp in &paths {
        let tp = hp.with_extension("toks");
        if !tp.exists() { continue; }
        let tokens = read_toks(tp.to_str().unwrap());
        let uniq: std::collections::HashSet<u32> = tokens.iter().copied().collect();
        if (uniq.len() as f32 / tokens.len() as f32) < uniq_min { skipped += 1; continue; }
        let (n_pos, hidden, layers5) = read_hfhs(hp.to_str().unwrap());
        assert_eq!(hidden, d_tgt);
        seqs.push(Seq { layers5, tokens, n_pos });
        if seqs.len() >= n_chunks { break; }
    }
    eprintln!("loaded {} coherent seqs ({skipped} degenerate skipped, uniq<{uniq_min})", seqs.len());
    let n_seq = seqs.len();
    assert!(n_seq >= 2, "need >=2 sequences (got {n_seq})");
    let n_eval = 2.min(n_seq - 1);
    let n_tr = n_seq - n_eval;
    println!("loaded {n_seq} sequences ({n_tr} train, {n_eval} eval)");

    // frozen head on GPU (faithful lm_head)
    let lm_head_g = up(&mut gpu, &lmhead, &[vocab, d_tgt]);
    // transposed copy for the backward GEMM (d_out = dlogits . lmhead)
    let lm_head_T_g = gpu.zeros(&[d_tgt, vocab], DType::F32).unwrap();
    gpu.transpose_f32(&lm_head_g, &lm_head_T_g, vocab, d_tgt).unwrap();
    eprintln!("lm_head transposed for tiled bwd (+{} GB)", (d_tgt * vocab * 4) / 1_000_000_000);
    let (lm_head_bf16, lm_head_T_bf16) = if dt::dflash_use_mfma() {
        let a = gpu.zeros(&[vocab, d_tgt], DType::F16).unwrap();
        gpu.to_bf16_f32(&lm_head_g, &a, vocab * d_tgt).unwrap();
        let bt = gpu.zeros(&[d_tgt, vocab], DType::F16).unwrap();
        gpu.to_bf16_f32(&lm_head_T_g, &bt, d_tgt * vocab).unwrap();
        eprintln!("lm_head bf16-MFMA path ON (+{} GB bf16 weights)", (vocab * d_tgt * 2 * 2) / 1_000_000_000);
        (Some(a), Some(bt))
    } else { (None, None) };
    // warm-start net + fresh adapters
    let (mut body_layers, final_norm) = dt::load_lfm2_warmstart(&mut gpu, &cfg, Path::new(st_path)).expect("warm-start");
    if std::env::var("HIPFIRE_CONV_INJECT").ok().as_deref() == Some("1") {
        let mut nc = 0;
        for li in 0..cfg.n_layers() { if !cfg.is_attn[li] { body_layers[li].w_c = Some(rndv(&mut gpu, d, d, 9100 + li, 0.01)); nc += 1; } }
        eprintln!("CONV-INJECT ON: W_c added to {nc} conv layers");
    } else { eprintln!("CONV-INJECT OFF (GQA-only injection)"); }
    let net = dt::Net {
        layers: body_layers,
        in_proj_v: rndv(&mut gpu, d, d_tgt, 1, 1.0 / (d_tgt as f32).sqrt()),
        out_proj_v: rndv(&mut gpu, d_tgt, d, 2, ws),
        fc: rndv(&mut gpu, d, fc_in, 3, 1.0 / (fc_in as f32).sqrt()),
        final_norm,
        hidden_norm: None,
    };
    let m_state = zeros_like(&mut gpu, &net);
    let v_state = zeros_like(&mut gpu, &net);

    // z-lab protocol: mask token 248070, long context (env-tunable; rung-0
    // showed 32 was crippling). Default 256.
    let bsz = 16usize;
    let n_ctx: usize = std::env::var("HIPFIRE_TRAIN_NCTX").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let mask_id = 248070usize; let gamma = 8.0f32;
    eprintln!("protocol: n_ctx={n_ctx} mask_id={mask_id}");
    let warmup = (steps / 60).max(500).min(1500);
    let (b1, b2, eps_a, wd) = (0.9f32, 0.999f32, 1e-8f32, 0.01f32);
    // seed-anchor: position 0 is the revealed seed (weight 0); predict 1..B with w_k
    let weights: Vec<f32> = (0..bsz).map(|k| if k == 0 { 0.0 } else { (-((k - 1) as f32) / gamma).exp() }).collect();
    let wsum: f32 = weights.iter().sum();
    let weights_g = up(&mut gpu, &weights, &[bsz]);
    let block_pos: Vec<i32> = (0..bsz).map(|i| (n_ctx + i) as i32).collect();
    let mut full: Vec<i32> = (0..n_ctx).map(|i| i as i32).collect(); full.extend(block_pos.iter().copied());
    let block_pos_g = dt::upos(&mut gpu, &block_pos);
    let full_pos_g = dt::upos(&mut gpu, &full);
    let conv_state = gpu.zeros(&[d, cfg.conv_k - 1], DType::F32).unwrap();
    // mask-token embedding row (constant input for masked positions)
    let mask_emb: Vec<f32> = embed[mask_id * d_tgt..(mask_id + 1) * d_tgt].to_vec();

    // build a [n_ctx, fc_in] context-hidden host buffer for a given (seq, p)
    // context = the n_ctx committed hiddens ENDING AT the seed position p
    // (inclusive). Including p's hidden is critical: it's the target state that
    // predicts the first block token p+1.
    let ctx_hiddens = |s: &Seq, p: usize| -> Vec<f32> {
        let mut c = vec![0f32; n_ctx * fc_in];
        for (ci, pos) in (p + 1 - n_ctx..p + 1).enumerate() {
            for (li, layer) in s.layers5.iter().enumerate() {
                c[ci * fc_in + li * d_tgt..ci * fc_in + (li + 1) * d_tgt].copy_from_slice(&layer[pos * d_tgt..(pos + 1) * d_tgt]);
            }
        }
        c
    };

    let mut rng = Rng(0x9E3779B97F4A7C15);
    gpu.pool_begin_scope();
    let mut running = 0f32;
    for step in 1..=steps {
        let ck = gpu.pool_checkpoint();
        // sample a train block
        let si = rng.usize(n_tr);
        let s = &seqs[si];
        let p = n_ctx + rng.usize(s.n_pos - n_ctx - bsz);
        let ctxh = ctx_hiddens(s, p);
        let targets: Vec<i32> = (0..bsz).map(|i| s.tokens[p + i] as i32).collect();
        // all-masked block input
        let mut embed_in = vec![0f32; bsz * d_tgt];
        let seed_tok = s.tokens[p] as usize; // block pos 0 = committed seed (revealed)
        embed_in[0..d_tgt].copy_from_slice(&embed[seed_tok * d_tgt..(seed_tok + 1) * d_tgt]);
        for b in 1..bsz { embed_in[b * d_tgt..(b + 1) * d_tgt].copy_from_slice(&mask_emb); }
        let ctxh_g = up(&mut gpu, &ctxh, &[n_ctx, fc_in]);
        let embed_in_g = up(&mut gpu, &embed_in, &[bsz, d_tgt]);
        let targets_g = dt::ui32(&mut gpu, &targets);
        // lr schedule: linear warmup then cosine
        let lr = if step <= warmup { lr_max * step as f32 / warmup as f32 }
                 else { let t = (step - warmup) as f32 / (steps - warmup).max(1) as f32; 0.5 * lr_max * (1.0 + (std::f32::consts::PI * t).cos()) };
        // forward
        let body_in = dt::lin(&mut gpu, &embed_in_g, &net.in_proj_v, bsz, d_tgt, d);
        let ctx = dt::lin(&mut gpu, &ctxh_g, &net.fc, n_ctx, fc_in, d);
        let (body_out, tape) = dt::body_forward(&mut gpu, &cfg, &net.layers, &body_in, &ctx, &block_pos_g, &full_pos_g, &conv_state, bsz, n_ctx);
        let fn2 = gpu.zeros(&[bsz, d], DType::F32).unwrap();
        gpu.rmsnorm_batched(&body_out, &net.final_norm, &fn2, bsz, d, cfg.eps).unwrap();
        let out = dt::lin(&mut gpu, &fn2, &net.out_proj_v, bsz, d, d_tgt);
        let logits = lmhead_fwd(&mut gpu, &out, &lm_head_g, lm_head_bf16.as_ref(), bsz, d_tgt, vocab);
        let dlogits = gpu.zeros(&[bsz, vocab], DType::F32).unwrap();
        let loss_t = gpu.zeros(&[bsz], DType::F32).unwrap();
        gpu.ce_loss_bwd_f32(&logits, &targets_g, &weights_g, &dlogits, &loss_t, bsz, vocab).unwrap();
        let loss: f32 = gpu.download_f32(&loss_t).unwrap().iter().sum::<f32>() / wsum;
        running = if step == 1 { loss } else { 0.98 * running + 0.02 * loss };
        // backward
        let d_out = lmhead_bwd(&mut gpu, &dlogits, &lm_head_T_g, lm_head_T_bf16.as_ref(), bsz, d_tgt, vocab);
        let d_fn2 = dt::lin_dx(&mut gpu, &d_out, &net.out_proj_v, bsz, d, d_tgt);
        let g_out_proj_v = dt::lin_dw(&mut gpu, &d_out, &fn2, bsz, d, d_tgt);
        let d_body_out = zt(&mut gpu, bsz * d); let g_final_norm = zt(&mut gpu, d);
        gpu.rmsnorm_bwd_f32(&body_out, &net.final_norm, &d_fn2, &d_body_out, &g_final_norm, bsz, d, cfg.eps).unwrap();
        let (d_body_in, d_ctx, glayers) = dt::body_backward(&mut gpu, &cfg, &net.layers, &tape, &d_body_out, &ctx, &block_pos_g, &full_pos_g, &conv_state, bsz, n_ctx);
        let g_in_proj_v = dt::lin_dw(&mut gpu, &d_body_in, &embed_in_g, bsz, d_tgt, d);
        let g_fc = dt::lin_dw(&mut gpu, &d_ctx, &ctxh_g, n_ctx, fc_in, d);
        let grad = dt::Net { layers: glayers, in_proj_v: g_in_proj_v, out_proj_v: g_out_proj_v, fc: g_fc, final_norm: g_final_norm, hidden_norm: None };
        let bc1 = 1.0 / (1.0 - b1.powi(step as i32)); let bc2 = 1.0 / (1.0 - b2.powi(step as i32));
        let ps = dt::net_tensors(&net); let gs = dt::net_tensors(&grad); let ms = dt::net_tensors(&m_state); let vs = dt::net_tensors(&v_state);
        for i in 0..ps.len() { let n = ps[i].numel(); gpu.adam_step_f32(ps[i], gs[i], ms[i], vs[i], lr, b1, b2, eps_a, wd, bc1, bc2, n).unwrap(); }
        gpu.pool_release_to(ck);

        if step % 100 == 0 || step == 1 {
            println!("  step {step:5}/{steps}  loss={loss:.4} (ema {running:.4})  lr={lr:.2e}");
        }
        // proxy-τ eval (mean accepted prefix) on held-out blocks
        if step % 500 == 0 || step == steps {
            let ck2 = gpu.pool_checkpoint();
            let mut acc_sum = 0f32; let mut nb = 0; let mut hit_sum = 0f32; let mut pos_total = 0f32;
            for ei in n_tr..n_seq {
                let s = &seqs[ei];
                for bi in 0..8 {
                    let p = n_ctx + bi * ((s.n_pos - n_ctx - bsz) / 8).max(1);
                    if p + bsz > s.n_pos { break; }
                    let ctxh = ctx_hiddens(s, p);
                    let targets: Vec<i32> = (0..bsz).map(|i| s.tokens[p + i] as i32).collect();
                    let mut embed_in = vec![0f32; bsz * d_tgt];
                    for b in 0..bsz { embed_in[b * d_tgt..(b + 1) * d_tgt].copy_from_slice(&mask_emb); }
                    let ctxh_g = up(&mut gpu, &ctxh, &[n_ctx, fc_in]);
                    let embed_in_g = up(&mut gpu, &embed_in, &[bsz, d_tgt]);
                    let body_in = dt::lin(&mut gpu, &embed_in_g, &net.in_proj_v, bsz, d_tgt, d);
                    let ctx = dt::lin(&mut gpu, &ctxh_g, &net.fc, n_ctx, fc_in, d);
                    let (body_out, _t) = dt::body_forward(&mut gpu, &cfg, &net.layers, &body_in, &ctx, &block_pos_g, &full_pos_g, &conv_state, bsz, n_ctx);
                    let fn2 = gpu.zeros(&[bsz, d], DType::F32).unwrap();
                    gpu.rmsnorm_batched(&body_out, &net.final_norm, &fn2, bsz, d, cfg.eps).unwrap();
                    let out = dt::lin(&mut gpu, &fn2, &net.out_proj_v, bsz, d, d_tgt);
                    let logits = lmhead_fwd(&mut gpu, &out, &lm_head_g, lm_head_bf16.as_ref(), bsz, d_tgt, vocab);
                    let lg = gpu.download_f32(&logits).unwrap();
                    // argmax per block position (1..B; pos 0 is the seed)
                    let am: Vec<i32> = (1..bsz).map(|i| {
                        let row = &lg[i * vocab..(i + 1) * vocab];
                        row.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0 as i32
                    }).collect();
                    // accepted prefix = leading run of matches = greedy-τ
                    let mut acc = 0;
                    for (k, &a) in am.iter().enumerate() { if a == targets[k + 1] { acc += 1; } else { break; } }
                    // per-position top-1 accuracy (no break) = learning diagnostic
                    let hits = am.iter().enumerate().filter(|(k, &a)| a == targets[k + 1]).count();
                    acc_sum += acc as f32; nb += 1; hit_sum += hits as f32; pos_total += (bsz - 1) as f32;
                }
            }
            let proxy_tau = if nb > 0 { acc_sum / nb as f32 } else { 0.0 };
            let per_pos = if pos_total > 0.0 { hit_sum / pos_total } else { 0.0 };
            println!("  [eval step {step}] proxy_tau(accepted prefix) = {proxy_tau:.3} | per_pos_acc = {per_pos:.3} | {nb} blocks  (baseline tau~6.3)");
            gpu.pool_release_to(ck2);
        }
        // periodic checkpoint (long run safety)
        if step % 5000 == 0 {
            let ck3 = gpu.pool_checkpoint();
            save_net(&mut gpu, &net, &cfg, &out_ckpt);
            gpu.pool_release_to(ck3);
            println!("  [ckpt step {step}] saved {out_ckpt}");
        }
    }

    // save checkpoint (DFNET: simple flat fp32 tensor container)
    save_net(&mut gpu, &net, &cfg, &out_ckpt);
    println!("saved drafter -> {out_ckpt}");
    println!("dflash_train_run: DONE ({steps} steps, final ema loss {running:.4})");
}

// DFNET container: magic + n_tensors + per tensor (name_len u32, name, numel u32, f32 data)
fn save_net(gpu: &mut Gpu, net: &dt::Net, cfg: &dt::Cfg, path: &str) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("ckpt"));
    let mut entries: Vec<(String, Vec<f32>)> = Vec::new();
    for (li, l) in net.layers.iter().enumerate() {
        entries.push((format!("layers.{li}.op_norm"), gpu.download_f32(&l.op_norm).unwrap()));
        entries.push((format!("layers.{li}.ffn_norm"), gpu.download_f32(&l.ffn_norm).unwrap()));
        for (nm, o) in [("in_proj", &l.in_proj), ("conv_w", &l.conv_w), ("out_proj", &l.out_proj), ("wq", &l.wq), ("wk", &l.wk), ("wv", &l.wv), ("wo", &l.wo), ("q_norm", &l.q_norm), ("k_norm", &l.k_norm), ("w_c", &l.w_c)] {
            if let Some(t) = o { entries.push((format!("layers.{li}.{nm}"), gpu.download_f32(t).unwrap())); }
        }
        entries.push((format!("layers.{li}.w1"), gpu.download_f32(&l.w1).unwrap()));
        entries.push((format!("layers.{li}.w3"), gpu.download_f32(&l.w3).unwrap()));
        entries.push((format!("layers.{li}.w2"), gpu.download_f32(&l.w2).unwrap()));
    }
    for (nm, t) in [("in_proj_v", &net.in_proj_v), ("out_proj_v", &net.out_proj_v), ("fc", &net.fc), ("final_norm", &net.final_norm)] {
        entries.push((nm.to_string(), gpu.download_f32(t).unwrap()));
    }
    f.write_all(b"DFNET\0\0\0").unwrap();
    f.write_all(&(entries.len() as u32).to_le_bytes()).unwrap();
    // config line: d, n_layers, nh, nkv, hd, conv_k, inter, d_tgt, vocab
    for v in [cfg.d, cfg.n_layers(), cfg.nh, cfg.nkv, cfg.hd, cfg.conv_k, cfg.inter, cfg.d_tgt, cfg.vocab] {
        f.write_all(&(v as u32).to_le_bytes()).unwrap();
    }
    for (name, data) in &entries {
        f.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
        f.write_all(name.as_bytes()).unwrap();
        f.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
        let raw: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        f.write_all(raw).unwrap();
    }
    f.flush().unwrap();
}
