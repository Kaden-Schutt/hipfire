//! End-to-end finite-difference gradcheck for the drafter training backward
//! (in_proj + blocks via block_backward_full + out_norm + score K-head + the
//! gradchecked scoring head). Tiny dims; never train on unverified glue.
//!
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "pflash-dgc"
//!   cargo run -p hipfire-train --release --example pflash_drafter_gradcheck

use hipfire_train::drafter::{drafter_backward, drafter_forward_train, Drafter, DrafterConfig};
use hipfire_train::ops::pflash_score::pflash_score_forward;
use hipfire_rdna::{DType, Gpu};

const VOCAB: usize = 32;
const H_T: usize = 8;
const SEQ: usize = 8;
const BLOCK: usize = 2;
const LAST: usize = SEQ - 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    let nb = SEQ / BLOCK;

    let mut s: u64 = 0xCAFEF00D;
    let mut rng = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (1u64 << 31) as f32 - 1.0
    };
    let embed_host: Vec<f32> = (0..VOCAB * H_T).map(|_| rng()).collect();
    let embed = gpu.upload_f32(&embed_host, &[VOCAB, H_T])?;
    let tokens: Vec<u32> = (0..SEQ)
        .map(|t| (t * 3 + 1) as u32 % VOCAB as u32)
        .collect();
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let w: Vec<f32> = (0..nb).map(|_| rng()).collect(); // loss weights

    let cfg = DrafterConfig {
        h_draft: 8,
        n_layers: 2,
        n_heads: 2,
        n_kv: 1,
        head_dim: 4,
        inter: 16,
        rope_base: 10000.0,
        eps: 1e-5,
    };
    let kvd = cfg.kv_dim();
    let drafter = Drafter::new(&mut gpu, embed, H_T, VOCAB, cfg, SEQ)?;
    let scores_dev = gpu.zeros(&[nb], DType::F32)?;
    let dscores = gpu.upload_f32(&w, &[nb])?;

    // L = Σ_b w_b · score_b
    let loss = |gpu: &mut Gpu, d: &Drafter| -> f32 {
        let a = drafter_forward_train(gpu, d, &tokens, &pos).unwrap();
        pflash_score_forward(gpu, &a.score_k, &scores_dev, SEQ, kvd, BLOCK, nb, LAST).unwrap();
        let sc = gpu.download_f32(&scores_dev).unwrap();
        sc.iter().zip(&w).map(|(x, y)| x * y).sum()
    };

    // analytic grads
    let acts = drafter_forward_train(&mut gpu, &drafter, &tokens, &pos)?;
    pflash_score_forward(
        &mut gpu,
        &acts.score_k,
        &scores_dev,
        SEQ,
        kvd,
        BLOCK,
        nb,
        LAST,
    )?;
    let grads = drafter_backward(&mut gpu, &drafter, &acts, &dscores, BLOCK, nb, LAST)?;
    let gflat = grads.flat();
    let pflat = drafter.params();
    assert_eq!(gflat.len(), pflat.len());

    // labels for the params we probe (param_idx, elem) — spread across the graph
    let names = drafter_param_names(cfg.n_layers);
    let probes: &[(usize, usize)] = &[
        (0, 3),               // in_proj
        (1, 1),               // layer0 wq
        (5, 2),               // layer0 wgate
        (7, 0),               // layer0 wdown
        (8, 0),               // layer0 norm1
        (1 + 9, 1),           // layer1 wq
        (1 + 9 + 6, 0),       // layer1 wdown
        (pflat.len() - 2, 0), // out_norm
        (pflat.len() - 1, 3), // wk_score
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
        let lp = loss(&mut gpu, &drafter);
        host[ei] = orig - h;
        gpu.memcpy_htod_auto(&pflat[pi].buf, bytemuck_cast(&host))?;
        let lm = loss(&mut gpu, &drafter);
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
            if ok { "✓" } else { "✗" }
        );
    }
    if all_ok {
        println!("\n  PASS ✓ — drafter backward matches finite differences end-to-end");
        Ok(())
    } else {
        Err("drafter gradcheck FAILED".into())
    }
}

fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn drafter_param_names(n_layers: usize) -> Vec<String> {
    let mut v = vec!["in_proj".to_string()];
    for li in 0..n_layers {
        for p in [
            "wq", "wk", "wv", "wo", "wgate", "wup", "wdown", "norm1", "norm2",
        ] {
            v.push(format!("L{li}.{p}"));
        }
    }
    v.push("out_norm".to_string());
    v.push("wk_score".to_string());
    v
}
