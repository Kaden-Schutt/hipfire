//! Byte-identity gate for the fused DSpark sample+accept kernel
//! (`dspark_sample_accept_lazy_f32`) vs the per-position `sample_top_p_pf`
//! reference loop it replaces.
//!
//! The kernel replicates the SINGLE-BLOCK `sample_top_p` draw per verify
//! position, threading the xorshift32 RNG and lazily early-exiting on the first
//! mismatch vs `draft[pos+1]`. This test forces the reference sampler onto its
//! single-block path (`HIPFIRE_SAMPLE_PARALLEL=0`) — exactly what the fused
//! kernel reimplements — and asserts the sampled token vector + advanced RNG are
//! byte-identical across a matrix of (n, temp, top_p, top_k, seed) and three
//! draft regimes (full-accept / immediate-mismatch / partial-accept). The
//! parallel sampler is byte-identical to the single-block one for distinct
//! logits (see `sample_top_p_parallel_impl`), so single-block parity transitively
//! covers the production path.
//!
//! Run:
//!   source ./scripts/gpu-lock.sh && gpu_acquire "sample-accept-parity"
//!   cargo run -p rdna-compute --release --example sample_accept_parity
//!   gpu_release

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_memcpy
)]

use rdna_compute::{DType, Gpu, GpuTensor};

const VOCAB: usize = 32768;

/// Deterministic distinct-ish logits in [-15, 15] via an LCG seeded by `seed`.
fn synth_logits(n: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(n * VOCAB);
    let mut s = seed ^ 0x9E37_79B9;
    for _ in 0..n * VOCAB {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (s >> 8) as f32 / 16_777_216.0; // [0,1)
        v.push((u - 0.5) * 30.0);
    }
    v
}

/// One reference draw of row `r` via the single-block `sample_top_p_pf`
/// (penalties off, matching the DSpark call site). Returns (token, new_rng).
fn ref_sample(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    result_buf: &GpuTensor,
    repeat_buf: &GpuTensor,
    row: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    rng: u32,
) -> (u32, u32) {
    let row_t = logits.sub_offset(row * VOCAB, VOCAB);
    gpu.sample_top_p_pf(
        &row_t, result_buf, repeat_buf, VOCAB, temp, top_p, rng, 0, 1.0, 0.0, 0.0, top_k, None,
    )
    .expect("sample_top_p_pf")
}

/// Reference lazy loop = the host logic in `final_norm_and_sample_all_batched_lazy`.
fn ref_lazy(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    result_buf: &GpuTensor,
    repeat_buf: &GpuTensor,
    draft: &[u32],
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    seed: u32,
) -> (Vec<u32>, u32) {
    let n = draft.len();
    let mut rng = seed;
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let (tok, new_rng) = ref_sample(
            gpu, logits, result_buf, repeat_buf, i, temp, top_p, top_k, rng,
        );
        rng = new_rng;
        ids.push(tok);
        if i + 1 < n && draft[i + 1] != tok {
            while ids.len() < n {
                ids.push(u32::MAX);
            }
            break;
        }
    }
    (ids, rng)
}

fn upload_u32(gpu: &mut Gpu, data: &[u32]) -> GpuTensor {
    let t = gpu.zeros(&[data.len()], DType::F32).expect("alloc u32 buf");
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.memcpy_htod_auto(&t.buf, bytes).expect("upload u32");
    t
}

/// δ>0 CACTUS distribution check. With `cactus_delta > 0` the fused kernel must
/// draw the committed token from the BOOSTED distribution — accept the drafted
/// token with prob `min(p_t + √(2δ·p_t·(1−p_t)), 1)`, else a residual over the
/// target support with the drafted token removed — NOT the plain target. A kernel
/// that ignored `cactus_delta` would match the δ=0 target and FAIL the `> 0.30`
/// leg. Controlled 3-token target (A=draft, p_t=0.3) so the boost is unmissable.
fn cactus_distribution_check(gpu: &mut Gpu) -> bool {
    const N_SAMPLES: usize = 10_000;
    let (a, b, c) = (100usize, 200usize, 300usize);
    let (pa, pb, pc) = (0.3f32, 0.5f32, 0.2f32); // target probs; drafted token = a
    let delta = 1.0f32;

    // n=2: pos 0 is the accept row (draft[1]=a); pos 1 is the bonus row. out[0] is
    // the CACTUS-distributed token either way (accept→a, reject→residual then stop).
    let mut logits = vec![-1.0e30f32; 2 * VOCAB];
    for base in [0usize, VOCAB] {
        logits[base + a] = pa.ln();
        logits[base + b] = pb.ln();
        logits[base + c] = pc.ln();
    }
    let logits_t = gpu
        .upload_f32(&logits, &[2, VOCAB])
        .expect("upload cactus logits");
    let draft_buf = upload_u32(gpu, &[0u32, a as u32]); // draft[1] = a
    let out_buf = gpu.zeros(&[3], DType::F32).expect("cactus out_buf");

    // Analytic distributions of out[0].
    let p_t = pa;
    let accept_prob = (p_t + (2.0 * delta * p_t * (1.0 - p_t)).sqrt()).min(1.0);
    let resid = pb + pc; // support minus the drafted token, unnormalized
    let mut theory_cactus = vec![0.0f32; VOCAB];
    theory_cactus[a] = accept_prob;
    theory_cactus[b] = (1.0 - accept_prob) * (pb / resid);
    theory_cactus[c] = (1.0 - accept_prob) * (pc / resid);
    let mut theory_delta0 = vec![0.0f32; VOCAB];
    theory_delta0[a] = pa;
    theory_delta0[b] = pb;
    theory_delta0[c] = pc;

    let mut hist = vec![0u32; VOCAB];
    for s_idx in 0..N_SAMPLES {
        // Spread seeds across u32 (Knuth multiplicative hash). The kernel's accept
        // draw u1 is ONE xorshift step from the seed; consecutive seeds through a
        // single xorshift are poorly decorrelated and would bias the empirical
        // accept rate (the real path threads a well-mixed RNG across windows).
        let seed = (s_idx as u32)
            .wrapping_mul(2654435761)
            .wrapping_add(0x9E37_79B9);
        let (ids, _rng) = gpu
            .sample_accept_lazy_f32(
                &logits_t, &draft_buf, &out_buf, 2, VOCAB, 1.0, 1.0, None, seed, delta,
            )
            .expect("sample_accept_lazy_f32 cactus");
        let t0 = ids[0] as usize;
        if t0 < VOCAB {
            hist[t0] += 1;
        }
    }
    let _ = gpu.free_tensor(out_buf);
    let _ = gpu.free_tensor(draft_buf);
    let _ = gpu.free_tensor(logits_t);

    let tv = |hist: &[u32], theory: &[f32]| -> f32 {
        let nn = N_SAMPLES as f32;
        0.5 * (0..VOCAB)
            .map(|i| (hist[i] as f32 / nn - theory[i]).abs())
            .sum::<f32>()
    };
    let tv_cactus = tv(&hist, &theory_cactus);
    let tv_delta0 = tv(&hist, &theory_delta0);
    let pass = tv_cactus < 0.03 && tv_delta0 > 0.30;
    println!(
        "[cactus δ={delta}] accept_prob={accept_prob:.4}  TV(GPU vs CACTUS)={tv_cactus:.5}  \
         TV(GPU vs δ0)={tv_delta0:.5}  (want cactus<0.03, δ0>0.30)  {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reference uses the DEFAULT parallel sampler (production path on gfx1151);
    // it is byte-identical to the single-block draw the fused kernel mirrors for
    // distinct logits (see `sample_top_p_parallel_impl`). Force single-block with
    // HIPFIRE_SAMPLE_PARALLEL=0 for an even tighter same-algorithm comparison.
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}  vocab: {VOCAB}", gpu.arch);

    let result_buf = gpu.zeros(&[2], DType::F32)?;
    let repeat_buf = gpu.zeros(&[1], DType::F32)?;

    // (n, temp, top_p, top_k)
    let configs: &[(usize, f32, f32, u32)] = &[
        (2, 0.7, 0.95, 0),
        (4, 0.7, 1.0, 0),
        (5, 0.3, 0.95, 20),
        (6, 1.0, 0.90, 40),
        (8, 0.8, 0.98, 0),
    ];
    let seeds: &[u32] = &[0x1357_9BDF, 42, 1, 0xDEAD_BEEF];

    let mut cases = 0usize;
    let mut fails = 0usize;

    for &(n, temp, top_p, top_k_raw) in configs {
        let top_k = if top_k_raw > 0 { Some(top_k_raw) } else { None };
        for &seed in seeds {
            let host = synth_logits(n, seed);
            let logits = gpu.upload_f32(&host, &[n, VOCAB])?;

            // Full no-stop sample (to construct accept-forcing drafts).
            let mut rng = seed;
            let mut full = Vec::with_capacity(n);
            for i in 0..n {
                let (tok, nr) = ref_sample(
                    &mut gpu,
                    &logits,
                    &result_buf,
                    &repeat_buf,
                    i,
                    temp,
                    top_p,
                    top_k,
                    rng,
                );
                rng = nr;
                full.push(tok);
            }

            // Three draft regimes. draft[0] is never compared (compare is draft[i+1]).
            // (A) full-accept: draft[i+1] == full[i] for all i → no early stop.
            let mut draft_a = vec![0u32; n];
            for i in 1..n {
                draft_a[i] = full[i - 1];
            }
            // (B) immediate mismatch: draft never matches → stop at pos 0 (n>1).
            let draft_b = vec![u32::MAX; n];
            // (C) partial: accept up to k=n/2, then a forced mismatch.
            let k = n / 2;
            let mut draft_c = vec![u32::MAX; n];
            for i in 1..=k.min(n - 1) {
                draft_c[i] = full[i - 1];
            }
            // draft_c[k+1] stays u32::MAX → mismatch at pos k (if k+1 < n).

            for (label, draft) in [
                ("full", &draft_a),
                ("mismatch", &draft_b),
                ("partial", &draft_c),
            ] {
                let (ref_ids, ref_rng) = ref_lazy(
                    &mut gpu,
                    &logits,
                    &result_buf,
                    &repeat_buf,
                    draft,
                    temp,
                    top_p,
                    top_k,
                    seed,
                );

                let draft_buf = upload_u32(&mut gpu, draft);
                let out_buf = gpu.zeros(&[n + 1], DType::F32)?;
                let (fused_ids, fused_rng) = gpu.sample_accept_lazy_f32(
                    &logits, &draft_buf, &out_buf, n, VOCAB, temp, top_p, top_k, seed, 0.0,
                )?;

                cases += 1;
                let ok = ref_ids == fused_ids && ref_rng == fused_rng;
                if !ok {
                    fails += 1;
                    eprintln!(
                        "FAIL n={n} temp={temp} top_p={top_p} top_k={top_k:?} seed={seed:#x} draft={label}\n  ref   ids={ref_ids:?} rng={ref_rng:#x}\n  fused ids={fused_ids:?} rng={fused_rng:#x}"
                    );
                }
                let _ = gpu.free_tensor(draft_buf);
                let _ = gpu.free_tensor(out_buf);
            }
            let _ = gpu.free_tensor(logits);
        }
    }

    // δ>0 CACTUS distribution check (the feature added on top of the δ=0 kernel).
    let cactus_ok = cactus_distribution_check(&mut gpu);
    if !cactus_ok {
        fails += 1;
    }

    println!("checked {cases} byte-parity cases + 1 cactus check, {fails} failure(s)");
    if fails > 0 {
        eprintln!("\nFAIL: fused sample+accept kernel is NOT byte-identical to sample_top_p_pf (or CACTUS check failed)");
        std::process::exit(1);
    }
    println!(
        "\nPASS: dspark_sample_accept_lazy_f32 byte-identical at δ=0 AND matches CACTUS at δ>0"
    );
    Ok(())
}
