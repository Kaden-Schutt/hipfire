//! G2: GPU tile decode must match escha_ref::reconstruct EXACTLY in fp16,
//! for both K. Run:
//!   cargo run --release -p rdna-compute --example test_escha_decode_gpu_vs_cpu
use hipfire_quantize::escha_ref;

fn main() {
    for (name, ic, oc, k) in [
        ("packed_gu_e0_k2.i16", 2048usize, 1024usize, 2usize),
        ("packed_down_e0_k3.i16", 512, 2048, 3),
    ] {
        let path = format!(
            "{}/../hipfire-quantize/tests/data/escha/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        let raw = std::fs::read(&path).expect("run fetch-goldens.sh first");
        let code: Vec<i16> = raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let want = escha_ref::reconstruct(&code, ic, oc, k);

        let mut gpu = rdna_compute::Gpu::init().expect("gpu");
        let got = gpu
            .escha_decode_tiles_host(&code, ic as u32, oc as u32, k as u32)
            .expect("decode");

        let bad = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        println!("{name}: {bad} mismatched of {} elements", want.len());
        assert_eq!(bad, 0, "{name}: GPU decode diverges from the CPU reference");
    }
    println!("G2 PASS");
}
