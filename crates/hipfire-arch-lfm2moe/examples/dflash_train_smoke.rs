// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// DFlash step 6 SMOKE: the full real-dims trainer on REAL ingredients —
//   * warm-start LFM2.5-350M body (bf16->fp32)
//   * REAL target hidden states (Qwen3.6-27B, dumped) as the injected context
//   * REAL frozen head table [vocab 248320, d_tgt 5120] (Q8-dequant embed; the
//     faithful MQ4 lm_head swaps in for the full run)
//   * REAL target tokens (from the 27B kldref)
// Overfits ONE real block: if loss collapses, the entire real-dims pipeline
// (forward + block-diffusion CE + backward + AdamW) is proven on real data.
//
//   dflash_train_smoke <head.f32> <hfhs> <kldref> [st_lfm2]

use hipfire_arch_lfm2moe::dflash_train as dt;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::io::Read;
use std::path::Path;

fn frand(seed: usize) -> f32 { ((seed.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0 }

// read [vocab,dim] f32 head table written by dflash_extract_head (DFHEAD header)
fn read_head(path: &str) -> (usize, usize, Vec<f32>) {
    let mut f = std::fs::File::open(path).expect("head");
    let mut hdr = [0u8; 16];
    f.read_exact(&mut hdr).unwrap();
    assert_eq!(&hdr[0..8], b"DFHEAD\0\0");
    let vocab = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let mut raw = Vec::new();
    f.read_to_end(&mut raw).unwrap();
    let v: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(v.len(), vocab * dim);
    (vocab, dim, v)
}

// read selected layers from an HFHS dump: returns [n_sel][n_pos*hidden] f32
fn read_hfhs(path: &str, sel: &[usize]) -> (usize, usize, Vec<Vec<f32>>) {
    let mut f = std::fs::File::open(path).expect("hfhs");
    let mut hdr = [0u8; 24];
    f.read_exact(&mut hdr).unwrap();
    assert_eq!(&hdr[0..8], b"HFHS\0\0\0\0");
    let n_layers = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let n_pos = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let hidden = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
    let layer_bytes = n_pos * hidden * 4;
    let mut all = Vec::new();
    for &l in sel {
        assert!(l < n_layers, "layer {l} >= {n_layers}");
        use std::io::{Seek, SeekFrom};
        f.seek(SeekFrom::Start(24 + (l * layer_bytes) as u64)).unwrap();
        let mut raw = vec![0u8; layer_bytes];
        f.read_exact(&mut raw).unwrap();
        all.push(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
    }
    (n_pos, hidden, all)
}

// read chunk-0 tokens from an HFKLDR kldref
fn read_tokens(path: &str) -> Vec<u32> {
    let mut f = std::fs::File::open(path).expect("kldref");
    let mut magic = [0u8; 8]; f.read_exact(&mut magic).unwrap();
    let mut hdr = [0u8; 24]; f.read_exact(&mut hdr).unwrap();
    let n_ctx = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let mut tb = vec![0u8; n_ctx * 4]; f.read_exact(&mut tb).unwrap();
    tb.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn up(gpu: &mut Gpu, v: &[f32], sh: &[usize]) -> GpuTensor { gpu.upload_f32(v, sh).unwrap() }
fn zt(gpu: &mut Gpu, n: usize) -> GpuTensor { gpu.zeros(&[n], DType::F32).unwrap() }

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
    dt::Net { layers, in_proj_v: mk(gpu, &src.in_proj_v), out_proj_v: mk(gpu, &src.out_proj_v), fc: mk(gpu, &src.fc), final_norm: mk(gpu, &src.final_norm) }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let head_path = a.get(1).map(|s| s.as_str()).unwrap_or("/workspace/qwen3.6-27b.head.f32");
    let hfhs_path = a.get(2).map(|s| s.as_str()).unwrap_or("/workspace/dflash-smoke.hfhs");
    let kldref_path = a.get(3).map(|s| s.as_str()).unwrap_or("/workspace/qwen3.6-27b-native.kldref.bin");
    let st_path = a.get(4).map(|s| s.to_string()).unwrap_or_else(|| "/root/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M-Base/snapshots/9960764e30892e01f29a6dc23df2533fcd8bd5ae/model.safetensors".to_string());

    let mut gpu = Gpu::init().expect("gpu");

    // ---- real ingredients ----
    println!("loading head table {head_path}");
    let (vocab, d_tgt, head) = read_head(head_path);
    let cfg = dt::Cfg::lfm2_350m(d_tgt, vocab, 5);
    let d = cfg.d;
    let sel = [2usize, 16, 31, 46, 61]; // 5 of 64, layer-2 -> 3rd-to-last
    let (n_pos, hidden, layers5) = read_hfhs(hfhs_path, &sel);
    assert_eq!(hidden, d_tgt, "hfhs hidden != d_tgt");
    let tokens = read_tokens(kldref_path);
    println!("vocab={vocab} d_tgt={d_tgt} n_pos={n_pos} tokens={}", tokens.len());

    let lm_head_g = up(&mut gpu, &head, &[vocab, d_tgt]);
    println!("warm-starting LFM2.5-350M body");
    let (body_layers, final_norm) = dt::load_lfm2_warmstart(&mut gpu, &cfg, Path::new(&st_path)).expect("warm-start");
    let fc_in = 5 * d_tgt;
    let ws = 1.0 / (d as f32).sqrt();
    let rndv = |g: &mut Gpu, rows: usize, cols: usize, seed: usize, sc: f32| { let v: Vec<f32> = (0..rows * cols).map(|i| frand(i + seed) * sc).collect(); up(g, &v, &[rows, cols]) };
    let net = dt::Net {
        layers: body_layers,
        in_proj_v: rndv(&mut gpu, d, d_tgt, 1, 1.0 / (d_tgt as f32).sqrt()),
        out_proj_v: rndv(&mut gpu, d_tgt, d, 2, ws),
        fc: rndv(&mut gpu, d, fc_in, 3, 1.0 / (fc_in as f32).sqrt()),
        final_norm,
    };

    // ---- one real block ----
    let bsz = 16usize; let n_ctx = 32usize; let p = 256usize;
    assert!(p >= n_ctx && p + bsz <= n_pos);
    // context hiddens [n_ctx, 5*d_tgt] = concat the 5 selected layers at positions [p-n_ctx, p)
    let mut ctxh = vec![0f32; n_ctx * fc_in];
    for (ci, pos) in (p - n_ctx..p).enumerate() {
        for (li, layer) in layers5.iter().enumerate() {
            let src = &layer[pos * hidden..(pos + 1) * hidden];
            ctxh[ci * fc_in + li * d_tgt..ci * fc_in + (li + 1) * d_tgt].copy_from_slice(src);
        }
    }
    let ctxh_g = up(&mut gpu, &ctxh, &[n_ctx, fc_in]);
    // block: all-masked input (mask id 0), real targets, w_k weights
    let mask_id = 0usize;
    let targets: Vec<i32> = (0..bsz).map(|i| tokens[p + i] as i32).collect();
    let gamma = 8.0f32;
    let weights: Vec<f32> = (0..bsz).map(|k| (-(k as f32) / gamma).exp()).collect();
    let mut embed_in = vec![0f32; bsz * d_tgt];
    for b in 0..bsz { embed_in[b * d_tgt..(b + 1) * d_tgt].copy_from_slice(&head[mask_id * d_tgt..(mask_id + 1) * d_tgt]); }
    let embed_in_g = up(&mut gpu, &embed_in, &[bsz, d_tgt]);
    let targets_g = dt::ui32(&mut gpu, &targets);
    let weights_g = up(&mut gpu, &weights, &[bsz]);
    let block_pos: Vec<i32> = (0..bsz).map(|i| (n_ctx + i) as i32).collect();
    let mut full: Vec<i32> = (0..n_ctx).map(|i| i as i32).collect();
    full.extend(block_pos.iter().copied());
    let block_pos_g = dt::upos(&mut gpu, &block_pos);
    let full_pos_g = dt::upos(&mut gpu, &full);
    let conv_state = gpu.zeros(&[d, cfg.conv_k - 1], DType::F32).unwrap();

    let m_state = zeros_like(&mut gpu, &net);
    let v_state = zeros_like(&mut gpu, &net);
    gpu.pool_begin_scope(); // persistent tensors above; per-step allocs released below
    let (lr, b1, b2, eps_a, wd) = (2e-3f32, 0.9f32, 0.999f32, 1e-8f32, 0.0f32);
    let steps = 300usize;
    println!("overfit ONE real block (p={p}, n_ctx={n_ctx}); ln(vocab)={:.3}", (vocab as f32).ln());
    let wsum: f32 = weights.iter().sum();
    let mut first = 0f32;

    for step in 1..=steps {
        let ck = gpu.pool_checkpoint();
        // forward
        let body_in = dt::lin(&mut gpu, &embed_in_g, &net.in_proj_v, bsz, d_tgt, d);
        let ctx = dt::lin(&mut gpu, &ctxh_g, &net.fc, n_ctx, fc_in, d);
        let (body_out, tape) = dt::body_forward(&mut gpu, &cfg, &net.layers, &body_in, &ctx, &block_pos_g, &full_pos_g, &conv_state, bsz, n_ctx);
        let fn2 = gpu.zeros(&[bsz, d], DType::F32).unwrap();
        gpu.rmsnorm_batched(&body_out, &net.final_norm, &fn2, bsz, d, cfg.eps).unwrap();
        let out = dt::lin(&mut gpu, &fn2, &net.out_proj_v, bsz, d, d_tgt);
        let logits = dt::lin(&mut gpu, &out, &lm_head_g, bsz, d_tgt, vocab);
        // loss + dlogits
        let dlogits = gpu.zeros(&[bsz, vocab], DType::F32).unwrap();
        let loss_t = gpu.zeros(&[bsz], DType::F32).unwrap();
        gpu.ce_loss_bwd_f32(&logits, &targets_g, &weights_g, &dlogits, &loss_t, bsz, vocab).unwrap();
        let loss: f32 = gpu.download_f32(&loss_t).unwrap().iter().sum::<f32>() / wsum;
        if step == 1 { first = loss; }
        if step % 20 == 0 || step == 1 { println!("  step {step:4}  loss = {loss:.5}"); }
        // backward (head -> body -> adapters)
        let d_out = dt::lin_dx(&mut gpu, &dlogits, &lm_head_g, bsz, d_tgt, vocab);
        let d_fn2 = dt::lin_dx(&mut gpu, &d_out, &net.out_proj_v, bsz, d, d_tgt);
        let g_out_proj_v = dt::lin_dw(&mut gpu, &d_out, &fn2, bsz, d, d_tgt);
        let d_body_out = zt(&mut gpu, bsz * d); let g_final_norm = zt(&mut gpu, d);
        gpu.rmsnorm_bwd_f32(&body_out, &net.final_norm, &d_fn2, &d_body_out, &g_final_norm, bsz, d, cfg.eps).unwrap();
        let (d_body_in, d_ctx, glayers) = dt::body_backward(&mut gpu, &cfg, &net.layers, &tape, &d_body_out, &ctx, &block_pos_g, &full_pos_g, &conv_state, bsz, n_ctx);
        let g_in_proj_v = dt::lin_dw(&mut gpu, &d_body_in, &embed_in_g, bsz, d_tgt, d);
        let g_fc = dt::lin_dw(&mut gpu, &d_ctx, &ctxh_g, n_ctx, fc_in, d);
        let grad = dt::Net { layers: glayers, in_proj_v: g_in_proj_v, out_proj_v: g_out_proj_v, fc: g_fc, final_norm: g_final_norm };
        // adam
        let bc1 = 1.0 / (1.0 - b1.powi(step as i32));
        let bc2 = 1.0 / (1.0 - b2.powi(step as i32));
        let ps = dt::net_tensors(&net); let gs = dt::net_tensors(&grad); let ms = dt::net_tensors(&m_state); let vs = dt::net_tensors(&v_state);
        for i in 0..ps.len() {
            let n = ps[i].numel();
            gpu.adam_step_f32(ps[i], gs[i], ms[i], vs[i], lr, b1, b2, eps_a, wd, bc1, bc2, n).unwrap();
        }
        gpu.pool_release_to(ck); // return this step's allocations to the pool
    }

    // final eval + argmax
    let body_in = dt::lin(&mut gpu, &embed_in_g, &net.in_proj_v, bsz, d_tgt, d);
    let ctx = dt::lin(&mut gpu, &ctxh_g, &net.fc, n_ctx, fc_in, d);
    let (body_out, _t) = dt::body_forward(&mut gpu, &cfg, &net.layers, &body_in, &ctx, &block_pos_g, &full_pos_g, &conv_state, bsz, n_ctx);
    let fn2 = gpu.zeros(&[bsz, d], DType::F32).unwrap();
    gpu.rmsnorm_batched(&body_out, &net.final_norm, &fn2, bsz, d, cfg.eps).unwrap();
    let out = dt::lin(&mut gpu, &fn2, &net.out_proj_v, bsz, d, d_tgt);
    let logits = dt::lin(&mut gpu, &out, &lm_head_g, bsz, d_tgt, vocab);
    let lg = gpu.download_f32(&logits).unwrap();
    let mut correct = 0;
    for i in 0..bsz {
        let row = &lg[i * vocab..(i + 1) * vocab];
        let am = row.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        if am as i32 == targets[i] { correct += 1; }
    }
    let dl = gpu.zeros(&[bsz, vocab], DType::F32).unwrap(); let lt = gpu.zeros(&[bsz], DType::F32).unwrap();
    gpu.ce_loss_bwd_f32(&logits, &targets_g, &weights_g, &dl, &lt, bsz, vocab).unwrap();
    let final_loss: f32 = gpu.download_f32(&lt).unwrap().iter().sum::<f32>() / wsum;
    println!("final loss = {final_loss:.5} (from {first:.5}); argmax-correct {correct}/{bsz}");
    if final_loss < 0.3 * first && final_loss < 5.0 {
        println!("dflash_train_smoke: PASS (real-dims pipeline learns on real data)");
    } else {
        println!("dflash_train_smoke: FAIL (loss {final_loss:.4} from {first:.4})");
        std::process::exit(1);
    }
}
