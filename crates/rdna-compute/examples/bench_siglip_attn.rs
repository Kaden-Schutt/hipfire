//! De-risk: which attention kernel for the SigLIP encoder (N=4096, 16 heads,
//! head_dim=72)? Compares `vit_attention_opt` (current) vs the flash kernels
//! `attention_dflash_f32` (arbitrary head_dim) and `attention_dflash_wmma_f32`
//! (matrix-core, needs head_dim%16==0 → pad 72→80). Reports correctness vs the
//! vit_attention_opt reference and per-call time.
//!
//! Usage: cargo run --release --example bench_siglip_attn -p rdna-compute

use rdna_compute::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5
        })
        .collect()
}

fn rel_l1(a: &[f32], b: &[f32]) -> f64 {
    let (mut sd, mut sr) = (0.0f64, 0.0f64);
    for (p, q) in a.iter().zip(b.iter()) {
        sd += (p - q).abs() as f64;
        sr += p.abs() as f64;
    }
    sd / sr.max(1e-12)
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let n = 4096usize;
    let heads = 16usize;
    let hd = 72usize;
    let hidden = heads * hd; // 1152
    let hdp = 80usize; // padded head_dim for WMMA (multiple of 16)
    let iters = 3usize;

    // Per-head q,k,v laid out [N, heads, hd].
    let q = lcg(0x11, n * heads * hd);
    let k = lcg(0x22, n * heads * hd);
    let v = lcg(0x33, n * heads * hd);

    // Fused qkv [N, 3*hidden] for vit_attention_opt: [q(hidden)|k|v] per token.
    let mut fused = vec![0f32; n * 3 * hidden];
    // Separate contiguous q/k/v [N*heads*hd] for dflash (same as q/k/v already).
    // Padded q/k/v [N*heads*hdp] (zero-filled tail) for WMMA.
    let mut qp = vec![0f32; n * heads * hdp];
    let mut kp = vec![0f32; n * heads * hdp];
    let mut vp = vec![0f32; n * heads * hdp];
    for t in 0..n {
        for h in 0..heads {
            for d in 0..hd {
                let src = (t * heads + h) * hd + d;
                fused[t * 3 * hidden + h * hd + d] = q[src];
                fused[t * 3 * hidden + hidden + h * hd + d] = k[src];
                fused[t * 3 * hidden + 2 * hidden + h * hd + d] = v[src];
                qp[(t * heads + h) * hdp + d] = q[src];
                kp[(t * heads + h) * hdp + d] = k[src];
                vp[(t * heads + h) * hdp + d] = v[src];
            }
        }
    }

    let d_fused = gpu.upload_f32(&fused, &[n * 3 * hidden]).unwrap();
    let d_q = gpu.upload_f32(&q, &[n * heads * hd]).unwrap();
    let d_k = gpu.upload_f32(&k, &[n * heads * hd]).unwrap();
    let d_v = gpu.upload_f32(&v, &[n * heads * hd]).unwrap();
    let d_qp = gpu.upload_f32(&qp, &[n * heads * hdp]).unwrap();
    let d_kp = gpu.upload_f32(&kp, &[n * heads * hdp]).unwrap();
    let d_vp = gpu.upload_f32(&vp, &[n * heads * hdp]).unwrap();
    let d_ref = gpu.zeros(&[n * hidden], DType::F32).unwrap();
    let d_df = gpu.zeros(&[n * heads * hd], DType::F32).unwrap();
    let d_wm = gpu.zeros(&[n * heads * hdp], DType::F32).unwrap();

    let bench = |gpu: &mut Gpu, label: &str, mut f: Box<dyn FnMut(&mut Gpu)>| {
        f(gpu); // warm
        gpu.hip.device_synchronize().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f(gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        eprintln!(
            "{label:32} {:.1} ms/call",
            t.elapsed().as_secs_f32() * 1000.0 / iters as f32
        );
    };

    // Reference + timing: vit_attention_opt (fused qkv).
    {
        let d_fused = &d_fused;
        let d_ref = &d_ref;
        bench(
            &mut gpu,
            "vit_attention_opt (current)",
            Box::new(move |g: &mut Gpu| {
                g.vit_attention_opt(d_fused, d_ref, n, hidden, heads, hd)
                    .unwrap();
            }),
        );
    }
    let r_ref = gpu.download_f32(&d_ref).unwrap();

    // attention_dflash_f32 (separate q/k/v, head_dim=72).
    {
        let (d_q, d_k, d_v, d_df) = (&d_q, &d_k, &d_v, &d_df);
        bench(
            &mut gpu,
            "attention_dflash_f32 (hd=72)",
            Box::new(move |g: &mut Gpu| {
                g.attention_dflash_f32(d_q, d_k, d_v, d_df, n, n, heads, heads, hd)
                    .unwrap();
            }),
        );
    }
    let r_df = gpu.download_f32(&d_df).unwrap();
    eprintln!("  dflash_f32 rel_L1 vs ref = {:.3e}", rel_l1(&r_ref, &r_df));

    // attention_dflash_wmma_f32 (padded head_dim=80).
    {
        let (d_qp, d_kp, d_vp, d_wm) = (&d_qp, &d_kp, &d_vp, &d_wm);
        bench(
            &mut gpu,
            "attention_dflash_wmma_f32 (hd=80)",
            Box::new(move |g: &mut Gpu| {
                g.attention_dflash_wmma_f32(d_qp, d_kp, d_vp, d_wm, n, n, heads, heads, hdp)
                    .unwrap();
            }),
        );
    }
    // Extract [:, :, :hd] from the padded WMMA output for comparison.
    let r_wm_pad = gpu.download_f32(&d_wm).unwrap();
    let mut r_wm = vec![0f32; n * hidden];
    for t in 0..n {
        for h in 0..heads {
            for d in 0..hd {
                r_wm[(t * heads + h) * hd + d] = r_wm_pad[(t * heads + h) * hdp + d];
            }
        }
    }
    eprintln!("  wmma_f32   rel_L1 vs ref = {:.3e}", rel_l1(&r_ref, &r_wm));
}
