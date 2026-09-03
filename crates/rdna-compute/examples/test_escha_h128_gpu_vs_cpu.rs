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
    println!("G3 PASS");
}
