//! Gate: one escha-coded DENSE linear on GPU against the `escha_ref` oracle.
//!
//! This exists to be run BEFORE the dense path is wired into ten call sites in
//! `forward.rs`/`prefill.rs`. Every failure mode of that wiring — a missing
//! H128, the two H128s swapped, a bias applied before the output transform
//! instead of after, rin/rout transposed — produces a full-rank, finite,
//! plausible activation rather than a crash. Debugging that through a whole
//! 64-layer model is far more expensive than pinning the single linear first.
//!
//! The oracle is `escha_ref::expert_linear`, which is the same
//! input_transform -> matmul -> output_transform that `ref.py::dense_linear`
//! specifies, and which G2/G3 already gate bit-exact at the decode and H128
//! level. Bias is added on top here because the reference helper covers the
//! coded linear only.
//!
//! Usage:
//!   cargo run --release -p hipfire-arch-qwen35 \
//!     --example test_escha_dense_linear_gpu_vs_cpu -- <model.hfq> [proj]
//!
//! `proj` defaults to `mlp.gate_proj` on layer 0; pass e.g.
//! `linear_attn.in_proj_qkv` to check another.

use hipfire_arch_qwen35::qwen35::escha::{
    escha_dense_leaf, escha_dense_linear_forward, load_escha_dense_linear, EschaWeightStore,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu};

/// Same candidate expansion the loader uses, so a bare `layers.0.…` resolves
/// against the `model.language_model.…` names actually in the file.
fn candidates(name: &str) -> Vec<String> {
    hipfire_arch_qwen35::qwen35::load::qwen35_tensor_name_candidates(name)
}

fn find(hfq: &HfqFile, name: &str) -> Option<(hipfire_runtime::hfq::HfqTensorInfo, Vec<u8>)> {
    for c in candidates(name) {
        if let Some((info, data)) = hfq.tensor_data(&c) {
            return Some((info.clone(), data.to_vec()));
        }
        if let Some((info, buf)) = hfq.tensor_data_pread(&c) {
            return Some((info.clone(), buf.to_vec()));
        }
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: <model.hfq> [proj]");
    let proj = args.next().unwrap_or_else(|| "mlp.gate_proj".to_string());
    let p = "layers.0";

    let hfq = HfqFile::open(std::path::Path::new(&path))?;
    let mut gpu = Gpu::init()?;

    // Shape from the code tensor's own dims: escha stores [in/16, out/16, 16K].
    let (code_info, code_bytes) =
        find(&hfq, &escha_dense_leaf(p, &proj, "code")).expect("escha_code not found");
    let k = match code_info.quant_type {
        42 => 2usize,
        43 => 3usize,
        other => panic!("{proj}: quant_type {other} is not escha (42/43)"),
    };
    let dims: Vec<usize> = code_info.shape.iter().map(|&d| d as usize).collect();
    assert_eq!(dims.len(), 3, "{proj}: escha_code should be 3-D, got {dims:?}");
    let (ic, oc) = (dims[0] * 16, dims[1] * 16);
    println!("{proj}: ic={ic} oc={oc} K={k}");

    // ── CPU reference ────────────────────────────────────────────────────
    let code_i16: Vec<i16> = code_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let w_bits = hipfire_quantize::escha_ref::reconstruct(&code_i16, ic, oc, k);

    let read_f32 = |leaf: &str, want: usize| -> Vec<f32> {
        let (_, d) = find(&hfq, &escha_dense_leaf(p, &proj, leaf))
            .unwrap_or_else(|| panic!("{leaf} not found"));
        let v: Vec<f32> = d
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(v.len(), want, "{leaf}: {} elements, want {want}", v.len());
        v
    };
    let rin = read_f32("rin_eff", ic);
    let rout = read_f32("rout_eff", oc);

    // Deterministic input; no rand dependency and reproducible across runs.
    let x: Vec<f32> = (0..ic)
        .map(|i| (((i * 2654435761usize) % 1000) as f32 / 500.0) - 1.0)
        .collect();

    let y_ref_bits = hipfire_quantize::escha_ref::expert_linear(&x, &w_bits, &rin, &rout);
    let mut y_ref: Vec<f32> = y_ref_bits
        .iter()
        .map(|&b| hipfire_runtime::llama::f16_to_f32(b))
        .collect();

    // Bias AFTER the output transform, matching `ref.py::dense_linear`.
    let bias_name = format!("{p}.{proj}.bias");
    if let Some((info, d)) = find(&hfq, &bias_name) {
        let b: Vec<f32> = match info.quant_type {
            1 => d
                .chunks_exact(2)
                .map(|c| hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            _ => d
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        };
        assert_eq!(b.len(), oc, "bias length");
        for (yi, bi) in y_ref.iter_mut().zip(b.iter()) {
            *yi += bi;
        }
        println!("bias: present ({oc} elements), applied after the output transform");
    } else {
        println!("bias: absent (optional leaf)");
    }

    // ── GPU ──────────────────────────────────────────────────────────────
    // F16 store: the decode to fp16 is exact, so any difference is the
    // forward path rather than a re-quantisation. Q8_0 is reported after.
    for store in [EschaWeightStore::F16, EschaWeightStore::Q8_0] {
        let lin = load_escha_dense_linear(&hfq, &mut gpu, p, &proj, ic, oc, store, candidates)?;
        let xg = gpu.upload_f32(&x, &[ic])?;
        let xh = gpu.alloc_tensor(&[ic], DType::F32)?;
        let mid = gpu.alloc_tensor(&[oc], DType::F32)?;
        let yg = gpu.alloc_tensor(&[oc], DType::F32)?;
        escha_dense_linear_forward(&mut gpu, &lin, &xg, &xh, &mid, &yg)?;
        let y = gpu.download_f32(&yg)?;

        // Stage-by-stage norms: a zero at a known stage names the broken step.
        let l2 = |v: &[f32]| (v.iter().map(|a| (*a as f64) * (*a as f64)).sum::<f64>()).sqrt();
        let xh_h = gpu.download_f32(&xh)?;
        let mid_h = gpu.download_f32(&mid)?;
        println!(
            "    |x|={:.4} |xh|={:.4} |mid|={:.4} |y|={:.4} |y_ref|={:.4}  w.dtype={:?} w.m={} w.k={}",
            l2(&x), l2(&xh_h), l2(&mid_h), l2(&y), l2(&y_ref), lin.w.gpu_dtype, lin.w.m, lin.w.k
        );

        let mut worst = 0.0f32;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        let mut nonfinite = 0usize;
        for (a, b) in y.iter().zip(y_ref.iter()) {
            let (a, b): (&f32, &f32) = (a, b);
            if !a.is_finite() {
                nonfinite += 1;
            }
            let d = (a - b).abs();
            if d > worst {
                worst = d;
            }
            num += (d as f64) * (d as f64);
            den += (*b as f64) * (*b as f64);
        }
        let rel_rms = (num / den.max(1e-30)).sqrt();
        println!(
            "  store={store:?}: rel_rms {rel_rms:.3e}  worst_abs {worst:.3e}  non-finite {nonfinite}"
        );
        // F16 must be tight; Q8_0 carries its own re-quantisation error and is
        // reported rather than gated, so the two are not held to one bar.
        if matches!(store, EschaWeightStore::F16) {
            assert_eq!(nonfinite, 0, "non-finite output");
            assert!(
                rel_rms < 2e-3,
                "F16 store: rel_rms {rel_rms:.3e} exceeds 2e-3 — the dense forward disagrees \
                 with escha_ref by more than fp16 accumulation explains"
            );
        }
    }
    println!("PASS");
    Ok(())
}
