//! G2: GPU tile decode must match escha_ref::reconstruct EXACTLY in fp16,
//! for both K. Also covers the gate-widening and device-resident-vs-host
//! equivalence checks from the Task 7 review. Run:
//!   cargo run --release -p rdna-compute --example test_escha_decode_gpu_vs_cpu
use hipfire_quantize::escha_ref;
use rdna_compute::DType;

/// xorshift64* — same tiny inline PRNG convention used by the other GPU
/// parity examples in this crate (no `rand` dependency).
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
    }
}

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

    // Widen the gate: the two golden fixtures above are 2048x1024 and
    // 512x2048 — both comfortably small. Cover at least one production
    // shape from the dense 27B checkpoint at each K, on pseudo-random code,
    // to prove the tile/lane indexing generalises past the two golden
    // shapes rather than happening to work only for them.
    let mut rng = Rng(0xE5CA_5EED_1234_5678u64);
    for (ic, oc, k) in [(5120usize, 17408usize, 2usize), (17408, 5120, 3usize)] {
        let n_tiles = (ic / 16) * (oc / 16);
        let code_len = n_tiles * 16 * k;
        let code: Vec<i16> = (0..code_len).map(|_| rng.next_u32() as i16).collect();
        let want = escha_ref::reconstruct(&code, ic, oc, k);

        let mut gpu = rdna_compute::Gpu::init().expect("gpu");
        let got = gpu
            .escha_decode_tiles_host(&code, ic as u32, oc as u32, k as u32)
            .expect("wide-shape decode");

        let bad = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        println!(
            "wide shape {ic}x{oc} K={k}: {bad} mismatched of {} elements",
            want.len()
        );
        assert_eq!(
            bad, 0,
            "{ic}x{oc} K={k}: GPU decode diverges from the CPU reference at production scale"
        );
    }
    println!("wide-shape gate PASS");

    // Device-resident vs. host-roundtrip equivalence (Task 7, Finding 4):
    // the load path decodes weights that are already resident on the GPU,
    // with no host round trip. Prove the device-resident `escha_decode_tiles`
    // and the host-roundtrip `escha_decode_tiles_host` (which now calls it
    // internally) produce bit-identical output for the same input.
    {
        let (ic, oc, k) = (2048usize, 1024usize, 2usize);
        let path = format!(
            "{}/../hipfire-quantize/tests/data/escha/packed_gu_e0_k2.i16",
            env!("CARGO_MANIFEST_DIR")
        );
        let raw = std::fs::read(&path).expect("run fetch-goldens.sh first");
        let code: Vec<i16> = raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        let mut gpu = rdna_compute::Gpu::init().expect("gpu");
        let host_result = gpu
            .escha_decode_tiles_host(&code, ic as u32, oc as u32, k as u32)
            .expect("host-roundtrip decode");

        let code_bytes: Vec<u8> = code.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_code = gpu
            .upload_raw(&code_bytes, &[code.len()])
            .expect("upload code");
        let d_bare = gpu
            .alloc_tensor(&[ic * oc], DType::F16)
            .expect("alloc bare");
        gpu.escha_decode_tiles(&d_code, &d_bare, ic as u32, oc as u32, k as u32)
            .expect("device-resident decode");
        let mut raw_out = vec![0u8; ic * oc * 2];
        gpu.hip
            .memcpy_dtoh(&mut raw_out, &d_bare.buf)
            .expect("copy back device-resident result");
        let device_result: Vec<u16> = raw_out
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        assert_eq!(
            host_result, device_result,
            "device-resident escha_decode_tiles diverges from escha_decode_tiles_host"
        );
        println!(
            "device-resident vs host equivalence PASS ({} elements identical)",
            device_result.len()
        );
    }
}
