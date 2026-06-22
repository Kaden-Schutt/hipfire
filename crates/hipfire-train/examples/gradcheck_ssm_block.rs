//! Finite-difference gradcheck for the GLA-lite SSM block (fwd+bwd composed:
//! rmsnorm + sigmoid + gated_scan + linears + swiglu + residuals).
//! Checks dL/dx (input) and the SSM-path weight grads (w_u, w_g, w_o, norm1) plus
//! one MLP weight (wdown). Never train on an unverified gradient.
//!
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "ssm-block-gradcheck"
//!   cargo run -p hipfire-train --release --example gradcheck_ssm_block

use hipfire_train::ssm_block::{
    ssm_block_backward, ssm_block_forward, SsmBlockDims, SsmBlockWeights,
};
use rdna_compute::Gpu;

const SEQ: usize = 6;
const H: usize = 8;
const INTER: usize = 16;
const EPS: f32 = 1e-6;

fn rng_fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (1u64 << 31) as f32 - 1.0) * scale
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    let dims = SsmBlockDims {
        seq: SEQ,
        h: H,
        inter: INTER,
        eps: EPS,
    };
    println!("gradcheck ssm_block: seq={SEQ} h={H} inter={INTER}");

    // host weights (kept so we can perturb + re-upload for FD)
    let sh = 1.0 / (H as f32).sqrt();
    let si = 1.0 / (INTER as f32).sqrt();
    let mut norm1 = vec![1.0f32; H];
    let mut w_u = rng_fill(H * H, 0x11, sh);
    let mut w_g = rng_fill(H * H, 0x22, sh);
    let mut w_o = rng_fill(H * H, 0x33, sh);
    let norm2 = vec![1.0f32; H];
    let wgate = rng_fill(INTER * H, 0x44, sh);
    let wup = rng_fill(INTER * H, 0x55, sh);
    let mut wdown = rng_fill(H * INTER, 0x66, si);
    let mut x = rng_fill(SEQ * H, 0x77, 1.0);
    let w_loss = rng_fill(SEQ * H, 0x88, 1.0); // loss weights

    // upload-and-forward-and-loss closure (re-uploads ALL weights each call).
    let forward_loss = |gpu: &mut Gpu,
                        x: &[f32],
                        norm1: &[f32],
                        w_u: &[f32],
                        w_g: &[f32],
                        w_o: &[f32],
                        wdown: &[f32]|
     -> f32 {
        let xd = gpu.upload_f32(x, &[SEQ * H]).unwrap();
        let n1 = gpu.upload_f32(norm1, &[H]).unwrap();
        let wu = gpu.upload_f32(w_u, &[H, H]).unwrap();
        let wg = gpu.upload_f32(w_g, &[H, H]).unwrap();
        let wo = gpu.upload_f32(w_o, &[H, H]).unwrap();
        let n2 = gpu.upload_f32(&norm2, &[H]).unwrap();
        let wgt = gpu.upload_f32(&wgate, &[INTER, H]).unwrap();
        let wu2 = gpu.upload_f32(&wup, &[INTER, H]).unwrap();
        let wd = gpu.upload_f32(wdown, &[H, INTER]).unwrap();
        let w = SsmBlockWeights {
            norm1: &n1,
            w_u: &wu,
            w_g: &wg,
            w_o: &wo,
            norm2: &n2,
            wgate: &wgt,
            wup: &wu2,
            wdown: &wd,
        };
        let (xout, acts) = ssm_block_forward(gpu, &xd, &w, &dims).unwrap();
        let hh = gpu.download_f32(&xout).unwrap();
        hipfire_train::ssm_block::free_ssm_block_acts(gpu, acts).unwrap();
        for t in [xd, n1, wu, wg, wo, n2, wgt, wu2, wd, xout] {
            let _ = gpu.free_tensor(t);
        }
        hh.iter().zip(&w_loss).map(|(a, b)| a * b).sum()
    };

    // analytic grads
    let xd = gpu.upload_f32(&x, &[SEQ * H])?;
    let n1 = gpu.upload_f32(&norm1, &[H])?;
    let wu = gpu.upload_f32(&w_u, &[H, H])?;
    let wg = gpu.upload_f32(&w_g, &[H, H])?;
    let wo = gpu.upload_f32(&w_o, &[H, H])?;
    let n2 = gpu.upload_f32(&norm2, &[H])?;
    let wgt = gpu.upload_f32(&wgate, &[INTER, H])?;
    let wu2 = gpu.upload_f32(&wup, &[INTER, H])?;
    let wd = gpu.upload_f32(&wdown, &[H, INTER])?;
    let w = SsmBlockWeights {
        norm1: &n1,
        w_u: &wu,
        w_g: &wg,
        w_o: &wo,
        norm2: &n2,
        wgate: &wgt,
        wup: &wu2,
        wdown: &wd,
    };
    let (xout, acts) = ssm_block_forward(&mut gpu, &xd, &w, &dims)?;
    let _ = gpu.free_tensor(xout);
    let d_xout = gpu.upload_f32(&w_loss, &[SEQ * H])?; // dL/d(x_out) = w_loss
    let (d_x, grad) = ssm_block_backward(&mut gpu, &d_xout, &xd, &w, &acts, &dims)?;
    let dx_host = gpu.download_f32(&d_x)?;
    let dwu_host = gpu.download_f32(&grad.dw_u)?;
    let dwg_host = gpu.download_f32(&grad.dw_g)?;
    let dwo_host = gpu.download_f32(&grad.dw_o)?;
    let dn1_host = gpu.download_f32(&grad.dnorm1)?;
    let dwd_host = gpu.download_f32(&grad.dwdown)?;

    let hstep = 1e-3f32;
    let (atol, rtol) = (2e-3f32, 3e-2f32);
    let mut all_ok = true;
    let mut max_abs = 0.0f32;

    let mut check = |gpu: &mut Gpu,
                     name: &str,
                     vec: &mut Vec<f32>,
                     analytic: &[f32],
                     idxs: &[usize],
                     which: u8| {
        println!("\n-- dL/d{name} --\n  idx      analytic         fd        abs_err   tol      ok");
        for &i in idxs {
            let orig = vec[i];
            vec[i] = orig + hstep;
            let lp = match which {
                0 => forward_loss(gpu, vec, &norm1, &w_u, &w_g, &w_o, &wdown),
                1 => forward_loss(gpu, &x, vec, &w_u, &w_g, &w_o, &wdown),
                2 => forward_loss(gpu, &x, &norm1, vec, &w_g, &w_o, &wdown),
                3 => forward_loss(gpu, &x, &norm1, &w_u, vec, &w_o, &wdown),
                4 => forward_loss(gpu, &x, &norm1, &w_u, &w_g, vec, &wdown),
                _ => forward_loss(gpu, &x, &norm1, &w_u, &w_g, &w_o, vec),
            };
            vec[i] = orig - hstep;
            let lm = match which {
                0 => forward_loss(gpu, vec, &norm1, &w_u, &w_g, &w_o, &wdown),
                1 => forward_loss(gpu, &x, vec, &w_u, &w_g, &w_o, &wdown),
                2 => forward_loss(gpu, &x, &norm1, vec, &w_g, &w_o, &wdown),
                3 => forward_loss(gpu, &x, &norm1, &w_u, vec, &w_o, &wdown),
                4 => forward_loss(gpu, &x, &norm1, &w_u, &w_g, vec, &wdown),
                _ => forward_loss(gpu, &x, &norm1, &w_u, &w_g, &w_o, vec),
            };
            vec[i] = orig;
            let fd = (lp - lm) / (2.0 * hstep);
            let a = analytic[i];
            let abs = (a - fd).abs();
            let tol = atol + rtol * fd.abs();
            let ok = abs <= tol;
            all_ok &= ok;
            max_abs = max_abs.max(abs);
            println!(
                "  {i:>5} {a:>14.6} {fd:>12.6} {abs:>10.2e} {tol:>8.2e}   {}",
                if ok { "✓" } else { "✗" }
            );
        }
    };

    // NOTE: `x`,`norm1`,`w_u`,`w_g`,`w_o`,`wdown` are perturbed in place inside
    // `check`; the closure captures them by the explicit `vec` arg + the others
    // by ref, so we pass clones for the perturbed one. Borrow-checker: do these
    // serially with fresh clones.
    let idx_x = [0usize, 3, H, 2 * H + 1, 5 * H + 7];
    let idx_w = [0usize, 1, H + 2, 3 * H + 3, H * H - 1];
    let mut xv = x.clone();
    check(&mut gpu, "x", &mut xv, &dx_host, &idx_x, 0);
    let mut n1v = norm1.clone();
    check(
        &mut gpu,
        "norm1",
        &mut n1v,
        &dn1_host,
        &[0usize, 2, 5, 7],
        1,
    );
    let mut wuv = w_u.clone();
    check(&mut gpu, "w_u", &mut wuv, &dwu_host, &idx_w, 2);
    let mut wgv = w_g.clone();
    check(&mut gpu, "w_g", &mut wgv, &dwg_host, &idx_w, 3);
    let mut wov = w_o.clone();
    check(&mut gpu, "w_o", &mut wov, &dwo_host, &idx_w, 4);
    let mut wdv = wdown.clone();
    check(
        &mut gpu,
        "wdown",
        &mut wdv,
        &dwd_host,
        &[0usize, 1, INTER + 2, 3 * INTER + 3, H * INTER - 1],
        5,
    );

    // silence unused-mut warnings on the originals (perturbed via clones above)
    let _ = (&mut norm1, &mut w_u, &mut w_g, &mut w_o, &mut wdown, &mut x);

    println!("\n  max_abs_err={max_abs:.2e}  (atol={atol:.0e} rtol={rtol:.0e})");
    if all_ok {
        println!("  PASS ✓ (analytic gradient matches finite differences)");
        Ok(())
    } else {
        Err("gradcheck FAILED: some index exceeded atol+rtol·|fd|".into())
    }
}
