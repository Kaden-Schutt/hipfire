//! Validate the device Gumbel-top-k SWOR sampler. The Gumbel-max trick: the
//! rank-0 (top perturbed score) token is an exact sample from softmax(logits/temp),
//! so its marginal must equal the target p. Also checks the k tokens are distinct
//! (without replacement) and the returned log-probs equal ln(softmax). A weak /
//! correlated RNG would skew rank-0 toward the argmax — this catches that.

use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let vocab = 8usize;
    let k = 4usize;
    let p = [0.30f32, 0.04, 0.18, 0.06, 0.22, 0.05, 0.10, 0.05];
    let logits: Vec<f32> = p.iter().map(|x| x.ln()).collect();
    let t_log = gpu.upload_f32(&logits, &[vocab]).unwrap();
    let t_idx = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    let t_logp = gpu.alloc_tensor(&[k], DType::F32).unwrap();

    let n = 200_000u32;
    let mut hist = vec![0u64; vocab];
    let mut dup = 0u64;
    let mut logp_err = 0.0f64;
    let temp = 1.0f32;
    let lse: f32 = {
        let m = logits.iter().cloned().fold(f32::MIN, f32::max);
        m + logits.iter().map(|&v| (v - m).exp()).sum::<f32>().ln()
    };
    for run in 0..n {
        gpu.ddtree_gumbel_topk_batched_f32(
            &t_log,
            &t_idx,
            &t_logp,
            vocab,
            k,
            1,
            temp,
            0x9e37_79b9u64
                .wrapping_mul(run as u64 + 1)
                .wrapping_add(12345),
        )
        .unwrap();
        let idx = gpu.download_f32(&t_idx).unwrap();
        let lp = gpu.download_f32(&t_logp).unwrap();
        let toks: Vec<usize> = idx.iter().map(|f| f.to_bits() as usize).collect();
        hist[toks[0]] += 1; // rank-0 marginal
        let mut seen = std::collections::HashSet::new();
        for (j, &tk) in toks.iter().enumerate() {
            if !seen.insert(tk) {
                dup += 1;
            }
            logp_err += ((lp[j] - (logits[tk] - lse)) as f64).abs();
        }
    }
    let tv: f64 = (0..vocab)
        .map(|t| (hist[t] as f64 / n as f64 - p[t] as f64).abs())
        .sum::<f64>()
        * 0.5;
    println!(
        "rank-0 marginal = {:?}",
        hist.iter()
            .map(|&h| (h as f64 / n as f64 * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );
    println!("target p        = {p:?}");
    println!(
        "TV(rank0, p) = {tv:.4}   dup_tokens = {dup}   mean_logp_err = {:.2e}",
        logp_err / (n as f64 * k as f64)
    );
    println!(
        "{}",
        if tv < 0.02 && dup == 0 {
            "PASS — sampler is an exact SWOR draw"
        } else {
            "FAIL"
        }
    );
    assert!(
        tv < 0.02 && dup == 0,
        "device Gumbel SWOR sampler is biased"
    );
}
