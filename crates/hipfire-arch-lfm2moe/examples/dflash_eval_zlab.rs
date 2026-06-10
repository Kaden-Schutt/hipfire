// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Rung-0 harness proof: load the PUBLISHED z-lab Qwen3.6-27B DFlash drafter
// (bf16 safetensors) into the trainer Net and run the trainer per_pos eval.
// If the trainer eval / data / injection conventions match how the drafter
// was trained, per_pos should be ~0.5+ and proxy_tau in the τ≈6 ballpark.
// Far below that = harness flaw, found cheaply against known-good weights.
//
//   dflash_eval_zlab <st> <embed.f32> <lmhead.f32> <data_dir(SEL=1,16,31,46,61)>

use hipfire_arch_lfm2moe::dflash_train as dt;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::io::Read;
use std::path::Path;

fn read_head(path: &str) -> (usize, usize, Vec<f32>) {
    let mut f = std::fs::File::open(path).unwrap();
    let mut hdr = [0u8; 16]; f.read_exact(&mut hdr).unwrap();
    let vocab = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let mut raw = Vec::new(); f.read_to_end(&mut raw).unwrap();
    (vocab, dim, raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
fn read_hfhs(path: &str) -> (usize, usize, Vec<Vec<f32>>) {
    let mut f = std::fs::File::open(path).unwrap();
    let mut hdr = [0u8; 24]; f.read_exact(&mut hdr).unwrap();
    let n_layers = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let n_pos = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let hidden = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
    assert_eq!(n_layers, 5, "dump must be 5 SEL layers");
    let lb = n_pos * hidden * 4;
    let mut out = Vec::new();
    for _ in 0..5 { let mut raw = vec![0u8; lb]; f.read_exact(&mut raw).unwrap(); out.push(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()); }
    (n_pos, hidden, out)
}
fn read_toks(path: &str) -> Vec<u32> { let mut f = std::fs::File::open(path).unwrap(); let mut r = Vec::new(); f.read_to_end(&mut r).unwrap(); r.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
fn up(gpu: &mut Gpu, v: &[f32], sh: &[usize]) -> GpuTensor { gpu.upload_f32(v, sh).unwrap() }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let st_p = a.get(1).map(|s| s.as_str()).unwrap_or("/workspace/zlab-dflash/model.safetensors");
    let embed_p = a.get(2).map(|s| s.as_str()).unwrap_or("/workspace/qwen3.6-27b.head.f32");
    let lmhead_p = a.get(3).map(|s| s.as_str()).unwrap_or("/workspace/qwen3.6-27b.lmhead.f32");
    let data_dir = a.get(4).map(|s| s.as_str()).unwrap_or("/workspace/zlab-eval");
    let shift: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
    let n_ctx_arg: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(32);

    let mut gpu = Gpu::init().unwrap();
    let (vocab, d_tgt, embed) = read_head(embed_p);
    let (_lv, _ld, lmhead) = read_head(lmhead_p);
    let cfg = dt::Cfg::zlab_27b(vocab);
    let d = cfg.d;
    assert_eq!(d, d_tgt);
    let lm_head_g = up(&mut gpu, &lmhead, &[vocab, d_tgt]);
    eprintln!("loading z-lab drafter (bf16->f32, ~6.5GB)");
    let net = dt::load_zlab(&mut gpu, &cfg, Path::new(st_p)).expect("load_zlab");
    let hidden_norm = net.hidden_norm.as_ref().expect("hidden_norm");

    let bsz = 16usize; let n_ctx = n_ctx_arg;
    let mask_id = 248070usize; // z-lab mask_token_id
    let mask_emb: Vec<f32> = embed[mask_id * d_tgt..(mask_id + 1) * d_tgt].to_vec();
    let block_pos: Vec<i32> = (0..bsz).map(|i| (n_ctx + i) as i32).collect();
    let mut full: Vec<i32> = (0..n_ctx).map(|i| i as i32).collect(); full.extend(block_pos.iter().copied());
    let bpos = dt::upos(&mut gpu, &block_pos); let fpos = dt::upos(&mut gpu, &full);
    let cs = gpu.zeros(&[d, cfg.conv_k - 1], DType::F32).unwrap();
    let fc_in = 5 * d_tgt;

    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(data_dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "hfhs")).collect();
    paths.sort();
    let mut hits = 0f32; let mut tot = 0f32; let mut tau_sum = 0f32; let mut blocks = 0f32;
    let mut pos_hist = vec![0f32; bsz];
    gpu.pool_begin_scope();
    for hp in &paths {
        let toks = read_toks(hp.with_extension("toks").to_str().unwrap());
        let (n_pos, _h, layers5) = read_hfhs(hp.to_str().unwrap());
        for bi in 0..16 {
            let p = n_ctx + bi * ((n_pos - n_ctx - bsz) / 16).max(1);
            if p + bsz + shift > n_pos { break; }
            let ck = gpu.pool_checkpoint();
            let mut ctxh = vec![0f32; n_ctx * fc_in];
            for (ci, pos) in (p + 1 - n_ctx..p + 1).enumerate() {
                for (li, layer) in layers5.iter().enumerate() { ctxh[ci * fc_in + li * d_tgt..ci * fc_in + (li + 1) * d_tgt].copy_from_slice(&layer[pos * d_tgt..(pos + 1) * d_tgt]); }
            }
            let targets: Vec<i32> = (0..bsz).map(|i| toks[p + shift + i] as i32).collect();
            let mut ein = vec![0f32; bsz * d_tgt];
            let seed = toks[p] as usize;
            ein[0..d_tgt].copy_from_slice(&embed[seed * d_tgt..(seed + 1) * d_tgt]);
            for b in 1..bsz { ein[b * d_tgt..(b + 1) * d_tgt].copy_from_slice(&mask_emb); }
            let ctxh_g = up(&mut gpu, &ctxh, &[n_ctx, fc_in]);
            let body_in = up(&mut gpu, &ein, &[bsz, d_tgt]); // no in_proj (native d)
            let ctx = dt::lin(&mut gpu, &ctxh_g, &net.fc, n_ctx, fc_in, d);
            gpu.rmsnorm_batched(&ctx, hidden_norm, &ctx, n_ctx, d, cfg.eps).unwrap();
            let (body_out, _t) = dt::body_forward(&mut gpu, &cfg, &net.layers, &body_in, &ctx, &bpos, &fpos, &cs, bsz, n_ctx);
            let fn2 = gpu.zeros(&[bsz, d], DType::F32).unwrap();
            gpu.rmsnorm_batched(&body_out, &net.final_norm, &fn2, bsz, d, cfg.eps).unwrap();
            let logits = gpu.zeros(&[bsz, vocab], DType::F32).unwrap();
            gpu.gemm_f32_register_tiled(&lm_head_g, &fn2, &logits, vocab, d_tgt, bsz).unwrap();
            let lg = gpu.download_f32(&logits).unwrap();
            let mut tau = 0; let mut alive = true;
            for i in 1..bsz {
                let row = &lg[i * vocab..(i + 1) * vocab];
                let am = row.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0 as i32;
                let hit = am == targets[i];
                if hit { hits += 1.0; pos_hist[i] += 1.0; }
                tot += 1.0;
                if alive && hit { tau += 1; } else { alive = false; }
            }
            tau_sum += tau as f32; blocks += 1.0;
            gpu.pool_release_to(ck);
        }
    }
    println!("zlab drafter shift={shift}: per_pos = {:.3}  proxy_tau = {:.2}  blocks = {}", hits / tot.max(1.0), tau_sum / blocks.max(1.0), blocks as i32);
    let hist: Vec<String> = (1..bsz).map(|i| format!("{:.2}", pos_hist[i] / blocks.max(1.0))).collect();
    println!("pos1..15 acc: {}", hist.join(" "));
}
