//! GPU validation of the fused `ddtree_swor_walk_f32` kernel: same
//! distribution-fidelity proof as the CPU `swor_step` test, but the KERNEL does
//! the work. Depth-1 tree (root + k leaf children = the draft's WITHOUT-
//! replacement samples). Per trial: sample candidates from q on the host, run
//! the kernel, record the first emitted token; assert the marginal equals the
//! TARGET p (TV < 0.02) — the on-device distribution-exactness check.

use rdna_compute::{DType, Gpu, GpuTensor};

fn upload_i32(gpu: &Gpu, data: &[i32]) -> GpuTensor {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.upload_raw(bytes, &[data.len()]).expect("upload i32")
}

fn xorshift_unit(s: &mut u64) -> f32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    ((x >> 40) as f32) * (1.0 / 16_777_216.0)
}

// Sample k tokens WITHOUT replacement from q, in draw order.
fn swor_sample(q: &[f32], k: usize, rng: &mut u64) -> Vec<i32> {
    let mut qr = q.to_vec();
    let mut out = Vec::with_capacity(k);
    for _ in 0..k {
        let s: f32 = qr.iter().sum();
        if s <= 0.0 {
            break;
        }
        let u = xorshift_unit(rng) * s;
        let mut acc = 0.0;
        let mut pick = qr.len() - 1;
        for (i, &v) in qr.iter().enumerate() {
            acc += v;
            if u < acc {
                pick = i;
                break;
            }
        }
        out.push(pick as i32);
        qr[pick] = 0.0;
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let vocab = 6usize;
    let k = 3usize;
    let num_pos = 1usize;
    let n_slots = 1 + k; // root + k leaf children
    let p = [0.30f32, 0.05, 0.20, 0.10, 0.25, 0.10];
    let q = [0.10f32, 0.40, 0.05, 0.20, 0.05, 0.20];

    // Constant tensors (uploaded once).
    let mut tlog = vec![0f32; n_slots * vocab];
    for t in 0..vocab {
        tlog[t] = p[t].ln(); // slot 0 = ln(p)
    }
    for s in 1..n_slots {
        for t in 0..vocab {
            tlog[s * vocab + t] = (1.0f32 / vocab as f32).ln(); // leaves: uniform (unused for 1st token)
        }
    }
    let dlog: Vec<f32> = q.iter().map(|x| x.ln()).collect();
    let slot_depth: Vec<i32> = (0..n_slots).map(|s| if s == 0 { 0 } else { 1 }).collect();
    // child_of_cand[0*k+r] = r (the r-th candidate is child node r); leaves: -1.
    let mut child_of_cand = vec![-1i32; n_slots * k];
    for r in 0..k {
        child_of_cand[r] = r as i32;
    }

    let t_tlog = gpu.upload_f32(&tlog, &[n_slots * vocab]).unwrap();
    let t_dlog = gpu.upload_f32(&dlog, &[num_pos * vocab]).unwrap();
    let t_depth = upload_i32(&gpu, &slot_depth);
    let t_child = upload_i32(&gpu, &child_of_cand);
    let t_pres = gpu.alloc_tensor(&[vocab], DType::F32).unwrap();
    let t_qpos = gpu.alloc_tensor(&[vocab], DType::F32).unwrap();
    let t_pcand = gpu.alloc_tensor(&[num_pos * k], DType::F32).unwrap(); // i32-bytes
    let t_out = gpu.alloc_tensor(&[2 + num_pos], DType::F32).unwrap();

    let n_runs = 60_000u32;
    let mut hist = vec![0u64; vocab];
    let mut rng = 0xabcd_1234_5678_9001u64;
    let temp = 1.0f32;
    for run in 0..n_runs {
        let cands = swor_sample(&q, k, &mut rng);
        let cb =
            unsafe { std::slice::from_raw_parts(cands.as_ptr() as *const u8, cands.len() * 4) };
        gpu.memcpy_htod_auto(&t_pcand.buf, cb).unwrap();
        gpu.ddtree_swor_walk_f32(
            &t_tlog,
            &t_dlog,
            &t_pcand,
            &t_depth,
            &t_child,
            &t_pres,
            &t_qpos,
            &t_out,
            temp,
            k,
            vocab,
            n_slots,
            num_pos,
            0x9e37_79b9_u64.wrapping_mul(run as u64 + 1),
        )
        .unwrap();
        let raw = gpu.download_f32(&t_out).unwrap();
        let accept_len = raw[0].to_bits() as i32;
        let bonus = raw[1].to_bits() as i32;
        let first = if accept_len > 0 {
            let child = raw[2].to_bits() as i32; // accepted child node index = rank
            cands[child as usize]
        } else {
            bonus
        };
        hist[first as usize] += 1;
    }
    let tv: f64 = (0..vocab)
        .map(|t| (hist[t] as f64 / n_runs as f64 - p[t] as f64).abs())
        .sum::<f64>()
        * 0.5;
    println!("hist = {hist:?}");
    println!(
        "emitted marginal = {:?}",
        hist.iter()
            .map(|&h| h as f64 / n_runs as f64)
            .collect::<Vec<_>>()
    );
    println!("target p        = {p:?}");
    println!(
        "TV(emitted, target) = {tv:.4}  -> {}",
        if tv < 0.02 { "PASS" } else { "FAIL" }
    );
    assert!(
        tv < 0.02,
        "GPU SWOR kernel must preserve the target distribution; TV={tv:.4}"
    );
}
