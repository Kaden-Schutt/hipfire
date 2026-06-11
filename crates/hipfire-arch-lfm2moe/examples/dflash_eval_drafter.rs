// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Decisive diagnostic: load a trained drafter checkpoint and measure per_pos +
// position-1 accuracy on TRAIN sequences vs HELD-OUT. If train ~= held-out ~=
// low -> underfitting/capacity (data won't help). If train >> held-out ->
// overfitting -> data scale. Isolates the bottleneck.
//
//   dflash_eval_drafter <ckpt.dfnet> <embed.f32> <lmhead.f32> <data_dir>

use hipfire_arch_lfm2moe::dflash_train as dt;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const SEL: [usize; 5] = [2, 16, 31, 46, 61];

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
    let lb = n_pos * hidden * 4;
    let sel: Vec<usize> = if n_layers == SEL.len() { (0..SEL.len()).collect() } else { SEL.to_vec() };
    let mut out = Vec::new();
    for &l in &sel { f.seek(SeekFrom::Start(24 + (l * lb) as u64)).unwrap(); let mut raw = vec![0u8; lb]; f.read_exact(&mut raw).unwrap(); out.push(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()); }
    (n_pos, hidden, out)
}
fn read_toks(path: &str) -> Vec<u32> { let mut f = std::fs::File::open(path).unwrap(); let mut r = Vec::new(); f.read_to_end(&mut r).unwrap(); r.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
fn up(gpu: &mut Gpu, v: &[f32], sh: &[usize]) -> GpuTensor { gpu.upload_f32(v, sh).unwrap() }

struct Seq { layers5: Vec<Vec<f32>>, tokens: Vec<u32>, n_pos: usize }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ckpt = a.get(1).map(|s| s.as_str()).unwrap_or("/workspace/lfm2-dflash-grind.dfnet");
    let embed_p = a.get(2).map(|s| s.as_str()).unwrap_or("/workspace/qwen3.6-27b.head.f32");
    let lmhead_p = a.get(3).map(|s| s.as_str()).unwrap_or("/workspace/qwen3.6-27b.lmhead.f32");
    let data_dir = a.get(4).map(|s| s.as_str()).unwrap_or("/workspace/dflash-code2");

    let mut gpu = Gpu::init().unwrap();
    let (vocab, d_tgt, embed) = read_head(embed_p);
    let (_lv, _ld, lmhead) = read_head(lmhead_p);
    let cfg = dt::Cfg::lfm2_350m(d_tgt, vocab, 5);
    let d = cfg.d; let fc_in = 5 * d_tgt;
    let lm_head_g = up(&mut gpu, &lmhead, &[vocab, d_tgt]);
    println!("loading checkpoint {ckpt}");
    let net = dt::load_net(&mut gpu, &cfg, Path::new(ckpt)).expect("load_net");

    // load coherent seqs (same filter/order as the trainer)
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(data_dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "hfhs")).collect();
    paths.sort();
    let mut seqs = Vec::new();
    for hp in &paths {
        let tp = hp.with_extension("toks"); if !tp.exists() { continue; }
        let tokens = read_toks(tp.to_str().unwrap());
        let uniq: std::collections::HashSet<u32> = tokens.iter().copied().collect();
        if (uniq.len() as f32 / tokens.len() as f32) < 0.20 { continue; }
        let (n_pos, _h, layers5) = read_hfhs(hp.to_str().unwrap());
        seqs.push(Seq { layers5, tokens, n_pos });
    }
    let n = seqs.len();
    println!("{n} coherent seqs (trainer held out the LAST 2)");

    let bsz = 16usize;
    let n_ctx: usize = std::env::var("HIPFIRE_TRAIN_NCTX").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let mask_id = 248070usize;
    let mask_emb: Vec<f32> = embed[mask_id * d_tgt..(mask_id + 1) * d_tgt].to_vec();
    let block_pos: Vec<i32> = (0..bsz).map(|i| (n_ctx + i) as i32).collect();
    let mut full: Vec<i32> = (0..n_ctx).map(|i| i as i32).collect(); full.extend(block_pos.iter().copied());
    let bpos = dt::upos(&mut gpu, &block_pos); let fpos = dt::upos(&mut gpu, &full);
    let cs = gpu.zeros(&[d, cfg.conv_k - 1], DType::F32).unwrap();

    let mut eval = |gpu: &mut Gpu, idxs: &[usize]| -> (f32, f32) {
        let mut hits = 0f32; let mut tot = 0f32; let mut p1 = 0f32; let mut p1n = 0f32;
        for &si in idxs {
            let s = &seqs[si];
            for bi in 0..16 {
                let p = n_ctx + bi * ((s.n_pos - n_ctx - bsz) / 16).max(1);
                if p + bsz > s.n_pos { break; }
                let ck = gpu.pool_checkpoint(); gpu.pool_begin_scope();
                let mut ctxh = vec![0f32; n_ctx * fc_in];
                for (ci, pos) in (p + 1 - n_ctx..p + 1).enumerate() {
                    for (li, layer) in s.layers5.iter().enumerate() { ctxh[ci * fc_in + li * d_tgt..ci * fc_in + (li + 1) * d_tgt].copy_from_slice(&layer[pos * d_tgt..(pos + 1) * d_tgt]); }
                }
                let targets: Vec<i32> = (0..bsz).map(|i| s.tokens[p + i] as i32).collect();
                let mut ein = vec![0f32; bsz * d_tgt];
                let seed = s.tokens[p] as usize; ein[0..d_tgt].copy_from_slice(&embed[seed * d_tgt..(seed + 1) * d_tgt]);
                for b in 1..bsz { ein[b * d_tgt..(b + 1) * d_tgt].copy_from_slice(&mask_emb); }
                let ctxh_g = up(gpu, &ctxh, &[n_ctx, fc_in]); let ein_g = up(gpu, &ein, &[bsz, d_tgt]);
                let body_in = dt::lin(gpu, &ein_g, &net.in_proj_v, bsz, d_tgt, d);
                let ctx = dt::lin(gpu, &ctxh_g, &net.fc, n_ctx, fc_in, d);
                let (body_out, _t) = dt::body_forward(gpu, &cfg, &net.layers, &body_in, &ctx, &bpos, &fpos, &cs, bsz, n_ctx);
                let fn2 = gpu.zeros(&[bsz, d], DType::F32).unwrap();
                gpu.rmsnorm_batched(&body_out, &net.final_norm, &fn2, bsz, d, cfg.eps).unwrap();
                let out = dt::lin(gpu, &fn2, &net.out_proj_v, bsz, d, d_tgt);
                let logits = gpu.zeros(&[bsz, vocab], DType::F32).unwrap();
                gpu.gemm_f32_register_tiled(&lm_head_g, &out, &logits, vocab, d_tgt, bsz).unwrap();
                let lg = gpu.download_f32(&logits).unwrap();
                for i in 1..bsz {
                    let row = &lg[i * vocab..(i + 1) * vocab];
                    let am = row.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0 as i32;
                    if am == targets[i] { hits += 1.0; if i == 1 { p1 += 1.0; } }
                    tot += 1.0; if i == 1 { p1n += 1.0; }
                }
                gpu.pool_release_to(ck);
            }
        }
        (hits / tot.max(1.0), p1 / p1n.max(1.0))
    };

    let train_idx: Vec<usize> = (0..4.min(n - 2)).collect();
    let held_idx: Vec<usize> = (n - 2..n).collect();
    let (tr_pp, tr_p1) = eval(&mut gpu, &train_idx);
    let (ho_pp, ho_p1) = eval(&mut gpu, &held_idx);
    println!("TRAIN   ({} seqs): per_pos = {tr_pp:.3}  position-1 = {tr_p1:.3}", train_idx.len());
    println!("HELDOUT ({} seqs): per_pos = {ho_pp:.3}  position-1 = {ho_p1:.3}", held_idx.len());
    println!("=> {}", if tr_pp > 0.15 && ho_pp < tr_pp * 0.5 { "OVERFIT (train>>heldout) -> data scale" }
                       else if tr_pp < 0.10 { "UNDERFIT (train also low) -> capacity/task, data won't help" }
                       else { "ambiguous" });
}
