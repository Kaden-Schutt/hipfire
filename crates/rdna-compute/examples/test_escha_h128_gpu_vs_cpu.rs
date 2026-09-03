//! G3: the H128 kernels must match escha_ref directly.
//!
//! A round-trip check (H128 . H128 == 128 I) is NOT sufficient — a wrong
//! butterfly order is also self-inverse and would pass it while being wrong.
use hipfire_quantize::escha_ref;
use hipfire_quantize::float16::f16_to_f32;

fn main() {
    let n = 2048usize;
    let x: Vec<f32> = (0..n).map(|i| ((i * 37) as f32 * 0.017).sin()).collect();
    let rin: Vec<f32> = (0..n)
        .map(|i| if i % 3 == 0 { -0.0023 } else { 0.0023 })
        .collect();
    let mut rout: Vec<f32> = (0..n).map(|i| 1.0 + (i % 5) as f32 * 0.1).collect();
    rout[7] = 0.0;
    rout[1000] = 0.0; // pruned channels must stay exactly zero

    let want_in = escha_ref::input_transform(&x, &rin);
    let want_out = escha_ref::output_transform(&x, &rout);

    let mut gpu = rdna_compute::Gpu::init().expect("gpu");
    let got_in = gpu.escha_h128_in_host(&x, &rin).expect("h128 in");
    let got_out = gpu.escha_h128_out_host(&x, &rout).expect("h128 out");

    let bad_in = want_in.iter().zip(&got_in).filter(|(a, b)| a != b).count();
    let bad_out = want_out
        .iter()
        .zip(&got_out)
        .filter(|(a, b)| a != b)
        .count();
    println!("h128_in : {bad_in} mismatched of {n}");
    println!("h128_out: {bad_out} mismatched of {n}");
    assert_eq!(
        f16_to_f32(got_out[7]),
        0.0,
        "pruned channel 7 must be exactly zero"
    );
    assert_eq!(
        f16_to_f32(got_out[1000]),
        0.0,
        "pruned channel 1000 must be exactly zero"
    );
    assert_eq!(bad_in, 0);
    assert_eq!(bad_out, 0);

    batched_vs_ref(&mut gpu, n);
    println!("G3 PASS");
}

/// G3b (Task 10): the BATCHED forms must agree with `escha_ref` element for
/// element, exactly as the per-expert forms do. Batching is an indexing
/// change — slot `s` reads row `ids[s]` of the resident `[E, n]` table — so a
/// wrong index would still produce plausible, full-magnitude output. It is
/// gated against the oracle, never against the per-expert kernel alone.
///
/// The slot list deliberately repeats an expert (two slots on row 1) and is
/// not sorted: nothing in the kernel may assume distinct or ordered ids.
fn batched_vs_ref(gpu: &mut rdna_compute::Gpu, n: usize) {
    use rdna_compute::DType;
    let n_exp = 5usize;
    let ids: Vec<i32> = vec![3, 1, 0, 1];
    let slots = ids.len();

    // Per-expert transform vectors. Expert 1 (used twice) carries the pruned
    // channels so the zero contract is exercised through the batched index.
    let mut r_in = vec![0.0f32; n_exp * n];
    let mut r_out = vec![0.0f32; n_exp * n];
    for e in 0..n_exp {
        for i in 0..n {
            r_in[e * n + i] = if (i + e) % 3 == 0 { -0.0023 } else { 0.0019 } * (1.0 + e as f32);
            r_out[e * n + i] = 1.0 + ((i + 2 * e) % 5) as f32 * 0.1;
        }
    }
    r_out[1 * n + 7] = 0.0;
    r_out[1 * n + 1000] = 0.0;

    // Broadcast activation (gate_up input side) and a per-slot one (down side).
    let x1: Vec<f32> = (0..n).map(|i| ((i * 37) as f32 * 0.017).sin()).collect();
    let xk: Vec<f32> = (0..slots * n)
        .map(|i| ((i * 11) as f32 * 0.013).cos() * 0.5)
        .collect();

    let up = |g: &rdna_compute::Gpu, v: &[f32]| {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        g.upload_raw(&b, &[v.len()]).expect("upload")
    };
    let d_rin = up(gpu, &r_in);
    let d_rout = up(gpu, &r_out);
    let d_x1 = up(gpu, &x1);
    let d_xk = up(gpu, &xk);
    let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_ids = gpu.upload_raw(&id_bytes, &[slots]).expect("ids");
    let d_out = gpu.alloc_tensor(&[slots * n], DType::F32).expect("out");

    // ── in, broadcast x ──────────────────────────────────────────────────
    gpu.escha_h128_batched(
        "escha_h128_in_batched",
        &d_x1,
        &d_rin,
        &d_ids,
        &d_out,
        n,
        slots,
        false,
    )
    .expect("in batched (broadcast)");
    let got = gpu.download_f32(&d_out).expect("dl");
    let mut bad = 0usize;
    for (s, &e) in ids.iter().enumerate() {
        let want = escha_ref::input_transform(&x1, &r_in[e as usize * n..(e as usize + 1) * n]);
        for i in 0..n {
            if got[s * n + i] != f16_to_f32(want[i]) {
                bad += 1;
            }
        }
    }
    println!(
        "h128_in_batched (broadcast x): {bad} mismatched of {}",
        slots * n
    );
    assert_eq!(bad, 0);

    // ── in, per-slot x ───────────────────────────────────────────────────
    gpu.escha_h128_batched(
        "escha_h128_in_batched",
        &d_xk,
        &d_rin,
        &d_ids,
        &d_out,
        n,
        slots,
        true,
    )
    .expect("in batched (per-slot)");
    let got = gpu.download_f32(&d_out).expect("dl");
    let mut bad = 0usize;
    for (s, &e) in ids.iter().enumerate() {
        let want = escha_ref::input_transform(
            &xk[s * n..(s + 1) * n],
            &r_in[e as usize * n..(e as usize + 1) * n],
        );
        for i in 0..n {
            if got[s * n + i] != f16_to_f32(want[i]) {
                bad += 1;
            }
        }
    }
    println!(
        "h128_in_batched (per-slot x): {bad} mismatched of {}",
        slots * n
    );
    assert_eq!(bad, 0);

    // ── out ──────────────────────────────────────────────────────────────
    gpu.escha_h128_batched(
        "escha_h128_out_batched",
        &d_xk,
        &d_rout,
        &d_ids,
        &d_out,
        n,
        slots,
        true,
    )
    .expect("out batched");
    let got = gpu.download_f32(&d_out).expect("dl");
    let mut bad = 0usize;
    for (s, &e) in ids.iter().enumerate() {
        let want = escha_ref::output_transform(
            &xk[s * n..(s + 1) * n],
            &r_out[e as usize * n..(e as usize + 1) * n],
        );
        for i in 0..n {
            if got[s * n + i] != f16_to_f32(want[i]) {
                bad += 1;
            }
        }
    }
    println!("h128_out_batched: {bad} mismatched of {}", slots * n);
    assert_eq!(bad, 0);
    // Pruned channels of expert 1 land in slots 1 and 3.
    for s in [1usize, 3] {
        for ch in [7usize, 1000] {
            assert_eq!(
                got[s * n + ch],
                0.0,
                "slot {s} channel {ch} must be exactly zero"
            );
        }
    }

    // ── batched SwiGLU ───────────────────────────────────────────────────
    // Input must be f16-representable (it is the H128 output in production);
    // reuse the transform result above so the gate feeds the real shape.
    let inter = n / 2;
    let d_h = gpu
        .alloc_tensor(&[slots * inter], DType::F32)
        .expect("h buf");
    gpu.escha_swiglu_batched(&d_out, &d_h, inter, slots)
        .expect("swiglu");
    let got_h = gpu.download_f32(&d_h).expect("dl");
    let mut ulp1 = 0usize;
    let mut worse = 0usize;
    for s in 0..slots {
        let bits: Vec<u16> = got[s * n..(s + 1) * n]
            .iter()
            .map(|&v| escha_ref::f16_rne(v))
            .collect();
        let want = escha_ref::swiglu(&bits, inter);
        for i in 0..inter {
            let gb = escha_ref::f16_rne(got_h[s * inter + i]);
            if gb != want[i] {
                let d = (gb as i32 - want[i] as i32).abs();
                if d == 1 {
                    ulp1 += 1;
                } else {
                    worse += 1;
                }
            }
        }
    }
    println!(
        "swiglu_batched vs escha_ref::swiglu: {ulp1} at 1 f16 ulp, {worse} worse, of {}",
        slots * inter
    );
    // Device `expf` and Rust `f32::exp` are both <1 ulp but not the same
    // function, so a handful of values straddle an f16 rounding boundary.
    // Anything beyond 1 ulp is a real defect (wrong half, wrong slot stride,
    // missing rounding), not a libm difference.
    assert_eq!(
        worse, 0,
        "swiglu differs from the oracle by more than 1 ulp"
    );
    assert!(
        ulp1 * 1000 <= slots * inter,
        "swiglu 1-ulp mismatches {ulp1} exceed 0.1% of {} — not a libm difference",
        slots * inter
    );
}
