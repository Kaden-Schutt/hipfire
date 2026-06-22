// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// DFlash spike — step 3b: block-parallel LFM2.5-350M *draft body* forward with
// GQA KV-injection of the target-context feature.
//
// Processes a diffusion block of B positions in one shot (fp32), assembling the
// validated primitives:
//   conv layers  -> batched gated short-conv (conv1d_gated_batched_f32)
//   attn layers  -> naive GQA block attention over [ctx_KV ; block_KV]
//                   (attn_block_ctx_f32), the load-bearing DFlash conditioning
//   FFN          -> dense SwiGLU (linear_fwd_f32 + silu_mul_f32)
// Pre-norm residual stream throughout. Conv gate-injection is deferred to the
// refinement step (plan's spike-minimal: GQA-only injection first).
//
// This validates the *wiring* (the kernels themselves are CPU-ref validated):
//   - runs, finite, right shape
//   - deterministic across reruns
//   - the injected context actually changes the output (injection is connected,
//     not a silent no-op)  <- the thing most likely to be wrong in assembly
//   - the n_ctx = 0 (no context) path runs.

use rdna_compute::{DType, Gpu, GpuTensor};

const D: usize = 1024; // LFM2.5-350M hidden
const N_LAYERS: usize = 16;
const N_HEADS: usize = 16;
const N_KV: usize = 8;
const HEAD_DIM: usize = 64;
const CONV_K: usize = 3;
const DENSE_INTER: usize = 4608;
const ROPE_THETA: f32 = 1.0e6;
const EPS: f32 = 1.0e-5;
// layer_types from config.json: conv except attn at {2,5,8,10,12,14}
const ATTN_LAYERS: [usize; 6] = [2, 5, 8, 10, 12, 14];

fn frand(seed: usize) -> f32 {
    ((seed.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0
}

struct LayerW {
    op_norm: GpuTensor,  // [d]
    ffn_norm: GpuTensor, // [d]
    // conv
    in_proj: Option<GpuTensor>,  // [3d, d]
    conv_w: Option<GpuTensor>,   // [d, K]
    out_proj: Option<GpuTensor>, // [d, d]
    // attn
    wq: Option<GpuTensor>,     // [qd, d]
    wk: Option<GpuTensor>,     // [kvd, d]
    wv: Option<GpuTensor>,     // [kvd, d]
    wo: Option<GpuTensor>,     // [d, qd]
    q_norm: Option<GpuTensor>, // [head_dim]
    k_norm: Option<GpuTensor>, // [head_dim]
    // ffn (dense SwiGLU)
    w1: GpuTensor, // gate [inter, d]
    w3: GpuTensor, // up   [inter, d]
    w2: GpuTensor, // down [d, inter]
}

struct DraftW {
    layers: Vec<LayerW>,
}

fn rnd(gpu: &mut Gpu, rows: usize, cols: usize, seed: usize, scale: f32) -> GpuTensor {
    let v: Vec<f32> = (0..rows * cols).map(|i| frand(i + seed) * scale).collect();
    gpu.upload_f32(&v, &[rows, cols]).unwrap()
}

fn build_weights(gpu: &mut Gpu) -> DraftW {
    let qd = N_HEADS * HEAD_DIM;
    let kvd = N_KV * HEAD_DIM;
    let mut layers = Vec::new();
    for li in 0..N_LAYERS {
        let s = li * 1_000_003;
        let is_attn = ATTN_LAYERS.contains(&li);
        // norm weights ~1.0 (rmsnorm gain), small jitter
        let op_norm = {
            let v: Vec<f32> = (0..D).map(|i| 1.0 + 0.02 * frand(i + s)).collect();
            gpu.upload_f32(&v, &[D]).unwrap()
        };
        let ffn_norm = {
            let v: Vec<f32> = (0..D).map(|i| 1.0 + 0.02 * frand(i + s + 7)).collect();
            gpu.upload_f32(&v, &[D]).unwrap()
        };
        // weights scaled ~1/sqrt(fan_in) to keep activations sane
        let wscale = 1.0 / (D as f32).sqrt();
        let (in_proj, conv_w, out_proj, wq, wk, wv, wo, q_norm, k_norm) = if is_attn {
            let qn: Vec<f32> = (0..HEAD_DIM).map(|i| 1.0 + 0.02 * frand(i + s + 11)).collect();
            let kn: Vec<f32> = (0..HEAD_DIM).map(|i| 1.0 + 0.02 * frand(i + s + 13)).collect();
            (
                None, None, None,
                Some(rnd(gpu, qd, D, s + 100, wscale)),
                Some(rnd(gpu, kvd, D, s + 200, wscale)),
                Some(rnd(gpu, kvd, D, s + 300, wscale)),
                Some(rnd(gpu, D, qd, s + 400, 1.0 / (qd as f32).sqrt())),
                Some(gpu.upload_f32(&qn, &[HEAD_DIM]).unwrap()),
                Some(gpu.upload_f32(&kn, &[HEAD_DIM]).unwrap()),
            )
        } else {
            (
                Some(rnd(gpu, 3 * D, D, s + 100, wscale)),
                Some(rnd(gpu, D, CONV_K, s + 200, 0.3)),
                Some(rnd(gpu, D, D, s + 300, wscale)),
                None, None, None, None, None, None,
            )
        };
        let w1 = rnd(gpu, DENSE_INTER, D, s + 500, wscale);
        let w3 = rnd(gpu, DENSE_INTER, D, s + 600, wscale);
        let w2 = rnd(gpu, D, DENSE_INTER, s + 700, 1.0 / (DENSE_INTER as f32).sqrt());
        layers.push(LayerW {
            op_norm, ffn_norm, in_proj, conv_w, out_proj,
            wq, wk, wv, wo, q_norm, k_norm, w1, w3, w2,
        });
    }
    DraftW { layers }
}

fn upload_pos(gpu: &mut Gpu, pos: &[i32]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[pos.len()], DType::F32).unwrap();
    let bytes: Vec<u8> = pos.iter().flat_map(|p| p.to_le_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &bytes).unwrap();
    t
}

fn lin(gpu: &mut Gpu, x: &GpuTensor, w: &GpuTensor, m: usize, k: usize, n: usize) -> GpuTensor {
    let y = gpu.zeros(&[m, n], DType::F32).unwrap();
    gpu.linear_fwd_f32(x, w, &y, m, k, n).unwrap();
    y
}

/// Block-parallel draft body forward. `h` [B,d] is the residual stream (body
/// input on entry, body output on exit). `ctx` [n_ctx,d] is the post-fc target
/// context feature. Positions: block_pos[B], ctx_pos[n_ctx], full_pos[L].
#[allow(clippy::too_many_arguments)]
fn draft_body_forward(
    gpu: &mut Gpu, w: &DraftW, h: &GpuTensor, ctx: &GpuTensor,
    block_pos: &GpuTensor, full_pos: &GpuTensor, conv_state: &GpuTensor,
    b: usize, n_ctx: usize,
) {
    let qd = N_HEADS * HEAD_DIM;
    let kvd = N_KV * HEAD_DIM;
    let l = n_ctx + b;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    for li in 0..N_LAYERS {
        let lw = &w.layers[li];
        let xn = gpu.zeros(&[b, D], DType::F32).unwrap();
        gpu.rmsnorm_batched(h, &lw.op_norm, &xn, b, D, EPS).unwrap();

        if ATTN_LAYERS.contains(&li) {
            let q = lin(gpu, &xn, lw.wq.as_ref().unwrap(), b, D, qd);
            let kfull = gpu.zeros(&[l, kvd], DType::F32).unwrap();
            let vfull = gpu.zeros(&[l, kvd], DType::F32).unwrap();
            if n_ctx > 0 {
                gpu.linear_fwd_f32(ctx, lw.wk.as_ref().unwrap(), &kfull.sub_offset(0, n_ctx * kvd), n_ctx, D, kvd).unwrap();
                gpu.linear_fwd_f32(ctx, lw.wv.as_ref().unwrap(), &vfull.sub_offset(0, n_ctx * kvd), n_ctx, D, kvd).unwrap();
            }
            gpu.linear_fwd_f32(&xn, lw.wk.as_ref().unwrap(), &kfull.sub_offset(n_ctx * kvd, b * kvd), b, D, kvd).unwrap();
            gpu.linear_fwd_f32(&xn, lw.wv.as_ref().unwrap(), &vfull.sub_offset(n_ctx * kvd, b * kvd), b, D, kvd).unwrap();
            // per-head qk-norm: q over B*nH rows, K over L*nKV rows
            gpu.rmsnorm_batched(&q, lw.q_norm.as_ref().unwrap(), &q, b * N_HEADS, HEAD_DIM, EPS).unwrap();
            gpu.rmsnorm_batched(&kfull, lw.k_norm.as_ref().unwrap(), &kfull, l * N_KV, HEAD_DIM, EPS).unwrap();
            // RoPE: q by block positions, K_full by [ctx_pos ; block_pos]
            gpu.rope_batched_f32(&q, &q, block_pos, N_HEADS, 0, HEAD_DIM, ROPE_THETA, b).unwrap();
            gpu.rope_batched_f32(&kfull, &kfull, full_pos, 0, N_KV, HEAD_DIM, ROPE_THETA, l).unwrap();
            let attn_out = gpu.zeros(&[b, qd], DType::F32).unwrap();
            gpu.attn_block_ctx_f32(&q, &kfull, &vfull, &attn_out, b, l, N_HEADS, N_KV, HEAD_DIM, scale).unwrap();
            let mo = lin(gpu, &attn_out, lw.wo.as_ref().unwrap(), b, qd, D);
            gpu.add_inplace_f32(h, &mo).unwrap();
        } else {
            let bcx = lin(gpu, &xn, lw.in_proj.as_ref().unwrap(), b, D, 3 * D);
            let cy = gpu.zeros(&[b, D], DType::F32).unwrap();
            gpu.conv1d_gated_batched_f32(&bcx, conv_state, lw.conv_w.as_ref().unwrap(), &cy, b, D, CONV_K).unwrap();
            let mo = lin(gpu, &cy, lw.out_proj.as_ref().unwrap(), b, D, D);
            gpu.add_inplace_f32(h, &mo).unwrap();
        }

        // dense SwiGLU FFN
        let fnorm = gpu.zeros(&[b, D], DType::F32).unwrap();
        gpu.rmsnorm_batched(h, &lw.ffn_norm, &fnorm, b, D, EPS).unwrap();
        let g = lin(gpu, &fnorm, &lw.w1, b, D, DENSE_INTER);
        let u = lin(gpu, &fnorm, &lw.w3, b, D, DENSE_INTER);
        let act = gpu.zeros(&[b, DENSE_INTER], DType::F32).unwrap();
        gpu.silu_mul_f32(&g, &u, &act).unwrap();
        let fo = lin(gpu, &act, &lw.w2, b, DENSE_INTER, D);
        gpu.add_inplace_f32(h, &fo).unwrap();
    }
}

fn run(gpu: &mut Gpu, w: &DraftW, body_in: &[f32], ctx_v: &[f32], b: usize, n_ctx: usize) -> Vec<f32> {
    let h = gpu.upload_f32(body_in, &[b, D]).unwrap();
    let ctx = if n_ctx > 0 {
        gpu.upload_f32(ctx_v, &[n_ctx, D]).unwrap()
    } else {
        gpu.zeros(&[1, D], DType::F32).unwrap()
    };
    let block_pos: Vec<i32> = (0..b).map(|i| (n_ctx + i) as i32).collect();
    let mut full: Vec<i32> = (0..n_ctx).map(|i| i as i32).collect();
    full.extend(block_pos.iter().copied());
    let block_pos_g = upload_pos(gpu, &block_pos);
    let full_pos_g = upload_pos(gpu, &full);
    let conv_state = gpu.zeros(&[D, CONV_K - 1], DType::F32).unwrap();
    draft_body_forward(gpu, w, &h, &ctx, &block_pos_g, &full_pos_g, &conv_state, b, n_ctx);
    gpu.download_f32(&h).unwrap()
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let w = build_weights(&mut gpu);

    let b = 16usize; // block_size (train)
    let n_ctx = 24usize; // injected target-context positions

    let body_in: Vec<f32> = (0..b * D).map(|i| 0.5 * frand(i + 3)).collect();
    let ctx_v: Vec<f32> = (0..n_ctx * D).map(|i| 0.5 * frand(i + 80_000)).collect();
    let zero_ctx = vec![0f32; n_ctx * D];

    // 1) main forward
    let out = run(&mut gpu, &w, &body_in, &ctx_v, b, n_ctx);
    let finite = out.iter().all(|x| x.is_finite());
    let mean = out.iter().sum::<f32>() / out.len() as f32;
    let var = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / out.len() as f32;
    println!("forward: shape {}x{} finite={} mean={:.4} std={:.4}", b, D, finite, mean, var.sqrt());

    // 2) determinism
    let out2 = run(&mut gpu, &w, &body_in, &ctx_v, b, n_ctx);
    let max_det = out.iter().zip(&out2).map(|(a, c)| (a - c).abs()).fold(0f32, f32::max);
    println!("determinism: max|diff| across reruns = {max_det:.3e}");

    // 3) injection connectivity — ctx=0 must change the output (else inject is dead)
    let out_zero = run(&mut gpu, &w, &body_in, &zero_ctx, b, n_ctx);
    let max_inj = out.iter().zip(&out_zero).map(|(a, c)| (a - c).abs()).fold(0f32, f32::max);
    println!("injection effect: max|diff| (ctx=rand vs ctx=0) = {max_inj:.3e}");

    // 4) n_ctx = 0 path runs
    let out_noctx = run(&mut gpu, &w, &body_in, &[], b, 0);
    let finite_noctx = out_noctx.iter().all(|x| x.is_finite());
    println!("no-context path: finite={finite_noctx}");

    let ok = finite && finite_noctx && max_det == 0.0 && max_inj > 1e-3;
    if ok {
        println!("dflash_draft_forward: PASS (deterministic, finite, injection live)");
    } else {
        println!("dflash_draft_forward: FAIL (finite={finite} det={max_det:.1e} inj={max_inj:.1e})");
        std::process::exit(1);
    }
}
