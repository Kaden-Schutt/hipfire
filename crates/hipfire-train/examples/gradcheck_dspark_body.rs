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

//! End-to-end finite-difference gradcheck for the full DSpark drafter body:
//! ingest (`fc` + `hidden_norm`) → N blocks (qk-norm + bidirectional ctx++block
//! attention) → `out_norm`, producing `x_head`. Loss L = Σ X_HEAD∘G ⇒
//! d_x_head = G. Confirms param grads (in `params()`/`flat()` order) and the
//! `d_main_hidden` input grad match central differences.
//!
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire lock acquire "gradcheck-dspark-body"
//!   cargo run -p hipfire-train --release --example gradcheck_dspark_body

use hipfire_rdna::{Gpu, GpuTensor, HipResult};
use hipfire_train::dspark_drafter::{
    dspark_drafter_backward, dspark_drafter_forward_train, free_dspark_drafter_acts,
    DsparkDrafterConfig, DsparkDrafterWeights, DsparkLayerWeights,
};

const H: usize = 16;
const NL: usize = 2;
const NH: usize = 2;
const NKV: usize = 1;
const HD: usize = 8;
const INTER: usize = 32;
const BLOCK: usize = 3;
const CTX: usize = 4;
const NT: usize = 2; // n_targets → fc input = NT*H
const QD: usize = NH * HD;
const KVD: usize = NKV * HD;
const FIN: usize = NT * H;

fn cfg() -> DsparkDrafterConfig {
    DsparkDrafterConfig {
        h: H,
        n_layers: NL,
        n_heads: NH,
        n_kv: NKV,
        head_dim: HD,
        inter: INTER,
        rope_base: 10000.0,
        eps: 1e-6,
        block_size: BLOCK,
        n_targets: NT,
        qk_norm: true,
        vocab: 32,
    }
}

/// Deterministic seeded fill; distinct `seed` per tensor so weights differ.
fn seeded(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (1u64 << 31) as f32 - 1.0) * scale + off
        })
        .collect()
}

fn build(gpu: &mut Gpu) -> HipResult<DsparkDrafterWeights> {
    let mut seed = 1u64;
    let mut lin = |gpu: &mut Gpu, n: usize, scale: f32| -> HipResult<GpuTensor> {
        seed += 1;
        gpu.upload_f32(&seeded(n, seed, scale, 0.0), &[n])
    };

    let fc = lin(gpu, H * FIN, 0.06)?;
    let hidden_norm = lin(gpu, H, 0.05)?;
    let mut layers = Vec::with_capacity(NL);
    for _ in 0..NL {
        layers.push(DsparkLayerWeights {
            input_ln: lin(gpu, H, 0.05)?,
            wq: lin(gpu, QD * H, 0.06)?,
            wk: lin(gpu, KVD * H, 0.06)?,
            wv: lin(gpu, KVD * H, 0.06)?,
            wo: lin(gpu, H * QD, 0.06)?,
            q_norm: lin(gpu, HD, 0.05)?,
            k_norm: lin(gpu, HD, 0.05)?,
            post_ln: lin(gpu, H, 0.05)?,
            wgate: lin(gpu, INTER * H, 0.05)?,
            wup: lin(gpu, INTER * H, 0.05)?,
            wdown: lin(gpu, H * INTER, 0.05)?,
        });
    }
    let out_norm = lin(gpu, H, 0.05)?;
    Ok(DsparkDrafterWeights {
        fc,
        hidden_norm,
        layers,
        out_norm,
    })
}

fn param_names() -> Vec<String> {
    let mut v = vec!["fc".to_string(), "hidden_norm".to_string()];
    for li in 0..NL {
        for p in [
            "wq", "wk", "wv", "wo", "wgate", "wup", "wdown", "input_ln", "post_ln", "q_norm",
            "k_norm",
        ] {
            v.push(format!("L{li}.{p}"));
        }
    }
    v.push("out_norm".to_string());
    v
}

fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);
    let c = cfg();

    let ctx_pos: Vec<f32> = (0..CTX).map(|t| t as f32).collect();
    let blk_pos: Vec<f32> = (0..BLOCK).map(|t| (CTX + t) as f32).collect();
    let mhh: Vec<f32> = (0..CTX * FIN)
        .map(|i| ((i * 13 % 7) as f32) * 0.1 - 0.3)
        .collect();
    let beh: Vec<f32> = (0..BLOCK * H)
        .map(|i| ((i * 17 % 11) as f32) * 0.1 - 0.4)
        .collect();
    let gh: Vec<f32> = (0..BLOCK * H)
        .map(|i| ((i * 7 % 5) as f32) * 0.2 - 0.3)
        .collect();

    let weights = build(&mut gpu)?;
    let block_embeds = gpu.upload_f32(&beh, &[BLOCK * H])?;

    // Loss L = Σ x_head ∘ G, rebuilding main_hidden each call (host-perturbable).
    let loss = |gpu: &mut Gpu, w: &DsparkDrafterWeights, mh: &[f32]| -> HipResult<f32> {
        let main_hidden = gpu.upload_f32(mh, &[CTX * FIN])?;
        let acts = dspark_drafter_forward_train(
            gpu,
            w,
            &c,
            &main_hidden,
            &block_embeds,
            &ctx_pos,
            &blk_pos,
            None,
        )?;
        let xv = gpu.download_f32(acts.x_head())?;
        let l = xv.iter().zip(&gh).map(|(a, b)| a * b).sum();
        free_dspark_drafter_acts(gpu, acts)?;
        gpu.free_tensor(main_hidden)?;
        Ok(l)
    };

    // Analytic grads.
    let main_hidden = gpu.upload_f32(&mhh, &[CTX * FIN])?;
    let acts = dspark_drafter_forward_train(
        &mut gpu,
        &weights,
        &c,
        &main_hidden,
        &block_embeds,
        &ctx_pos,
        &blk_pos,
        None,
    )?;
    let d_x_head = gpu.upload_f32(&gh, &[BLOCK * H])?;
    let grads = dspark_drafter_backward(&mut gpu, &weights, &c, &main_hidden, &acts, &d_x_head)?;
    let gflat = grads.flat();
    let pflat = weights.params();
    assert_eq!(gflat.len(), pflat.len());
    let d_mh_a = gpu.download_f32(&grads.d_main_hidden)?;

    let names = param_names();
    // (param_index, element_index)
    let probes: &[(usize, usize)] = &[
        (0, 5),               // fc
        (1, 0),               // hidden_norm
        (2, 3),               // L0.wq
        (3, 1),               // L0.wk
        (4, 2),               // L0.wv
        (5, 0),               // L0.wo
        (2 + 9, 0),           // L0.q_norm
        (2 + 10, 0),          // L0.k_norm
        (2 + 11, 4),          // L1.wq
        (2 + 11 + 4, 6),      // L1.wgate
        (pflat.len() - 1, 0), // out_norm
    ];

    let h = 1e-3f32;
    let (atol, rtol) = (2e-3f32, 3e-2f32);
    let mut all_ok = true;
    println!("  param                  idx   analytic         fd        abs_err   tol    ok");
    for &(pi, ei) in probes {
        let a = gpu.download_f32(gflat[pi])?[ei];
        let mut host = gpu.download_f32(pflat[pi])?;
        let orig = host[ei];
        host[ei] = orig + h;
        gpu.memcpy_htod_auto(&pflat[pi].buf, bytemuck_cast(&host))?;
        let lp = loss(&mut gpu, &weights, &mhh)?;
        host[ei] = orig - h;
        gpu.memcpy_htod_auto(&pflat[pi].buf, bytemuck_cast(&host))?;
        let lm = loss(&mut gpu, &weights, &mhh)?;
        host[ei] = orig;
        gpu.memcpy_htod_auto(&pflat[pi].buf, bytemuck_cast(&host))?;
        let fd = (lp - lm) / (2.0 * h);
        let abs = (a - fd).abs();
        let tol = atol + rtol * fd.abs();
        let ok = abs <= tol;
        all_ok &= ok;
        println!(
            "  {:<20} {:>4} {:>12.6} {:>12.6} {:>10.2e} {:>8.2e} {}",
            names[pi],
            ei,
            a,
            fd,
            abs,
            tol,
            if ok { "OK" } else { "XX" }
        );
    }

    // d_main_hidden (ingest input grad).
    {
        let mut e = 0.0f32;
        for i in 0..mhh.len() {
            let mut hp = mhh.clone();
            hp[i] += h;
            let mut hm = mhh.clone();
            hm[i] -= h;
            let lp = loss(&mut gpu, &weights, &hp)?;
            let lm = loss(&mut gpu, &weights, &hm)?;
            let fd = (lp - lm) / (2.0 * h);
            e = e.max((d_mh_a[i] - fd).abs());
        }
        let tol = atol + rtol * 1.0;
        let ok = e < 5e-2f32;
        all_ok &= ok;
        println!(
            "  {:<20} {:>4} {:>12} {:>12} {:>10.2e} {:>8.2e} {}",
            "main_hidden",
            "-",
            "",
            "",
            e,
            tol,
            if ok { "OK" } else { "XX" }
        );
    }

    // Cleanup so the example is leak-clean under repeated runs.
    free_dspark_drafter_acts(&mut gpu, acts)?;

    if all_ok {
        println!("\n  PASS — DSpark drafter body backward matches finite differences end-to-end");
        Ok(())
    } else {
        Err("DSpark drafter body gradcheck FAILED".into())
    }
}
