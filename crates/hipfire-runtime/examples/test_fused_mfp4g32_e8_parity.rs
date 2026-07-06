//! Isolated bit-exact parity test for the gfx1151 fused mfp4-E8 decode kernels.
//!
//! Compares `fused_gate_up_mfp4g32_e8` and `fused_qkvza_mfp4g32_e8` (single
//! launch) against N sequential `gemv_mfp4g32_e8` calls on the SAME input
//! buffers. Because both share the byte-identical `gemv_e8_row_body`, the fused
//! output MUST equal the per-projection output bit-for-bit. This is the
//! deterministic correctness gate the autoregressive harness can't provide
//! (qwen3.5 DeltaNet Q8 state + KV q8 are stochastic run-to-run).
//!
//! Run on gfx1151: `HIP_VISIBLE_DEVICES=1 cargo run --release \
//!   --example test_fused_mfp4g32_e8_parity --features deltanet -- <model.mfp4e8>`

use hipfire_arch_qwen35::qwen35::{self, HfqSource, LayerWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::model_load::Layout;
use rdna_compute::DType;
use std::path::Path;

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/kaden/.hipfire/models/qwen3.5-0.8b.mfp4e8");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open HFQ");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    if !gpu.arch_caps.is_gfx1151() {
        eprintln!(
            "WARNING: not gfx1151 — fused kernels are gfx1151-only; set HIP_VISIBLE_DEVICES=1"
        );
    }
    let mut source = HfqSource::new(&mut hfq, &config);
    let layout = Layout::single(config.n_layers);
    let weights = qwen35::load_weights(&mut source, std::slice::from_mut(&mut gpu), &layout)
        .expect("load weights");

    // First DeltaNet layer carries both the FFN (gate/up) and the LA preamble
    // (qkv/z/beta/alpha) — everything we need.
    let layer = weights
        .layers
        .iter()
        .find_map(|l| match l {
            LayerWeights::DeltaNet(d) => Some(d),
            _ => None,
        })
        .expect("no DeltaNet layer");

    // Deterministic pseudo-random x in [-1, 1]; any vector works (both paths
    // read x identically — the FWHT rotation is applied upstream of the GEMV).
    let k = layer.w_gate.k;
    assert_eq!(layer.wqkv.k, k, "all projections share K");
    let x: Vec<f32> = (0..k)
        .map(|i| {
            let v = ((i as u64).wrapping_mul(2654435761) >> 11) & 0xffff;
            (v as f32 / 32768.0) - 1.0
        })
        .collect();
    let x_dev = gpu.upload_f32(&x, &[k]).expect("upload x");

    let mut fails = 0u32;

    // ── gate+up parity ──
    assert_eq!(
        layer.w_gate.gpu_dtype,
        DType::MFP4G32E8,
        "gate must be MFP4G32E8"
    );
    assert_eq!(
        layer.w_up.gpu_dtype,
        DType::MFP4G32E8,
        "up must be MFP4G32E8"
    );
    {
        let (gm, um) = (layer.w_gate.m, layer.w_up.m);
        let yfg = gpu.alloc_tensor(&[gm], DType::F32).unwrap();
        let yfu = gpu.alloc_tensor(&[um], DType::F32).unwrap();
        gpu.fused_gate_up_mfp4g32_e8(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &x_dev,
            &yfg,
            &yfu,
            gm,
            um,
            k,
        )
        .expect("fused gate_up");
        let yrg = gpu.alloc_tensor(&[gm], DType::F32).unwrap();
        let yru = gpu.alloc_tensor(&[um], DType::F32).unwrap();
        gpu.gemv_mfp4g32_e8(&layer.w_gate.buf, &x_dev, &yrg, gm, k)
            .expect("gemv gate");
        gpu.gemv_mfp4g32_e8(&layer.w_up.buf, &x_dev, &yru, um, k)
            .expect("gemv up");
        fails += cmp("gate", &gpu, &yfg, &yrg);
        fails += cmp("up", &gpu, &yfu, &yru);
    }

    // ── qkvza parity ──
    assert_eq!(
        layer.wqkv.gpu_dtype,
        DType::MFP4G32E8,
        "wqkv must be MFP4G32E8"
    );
    let la4_e8 = layer.wz.gpu_dtype == DType::MFP4G32E8
        && layer.w_beta.gpu_dtype == DType::MFP4G32E8
        && layer.w_alpha.gpu_dtype == DType::MFP4G32E8;
    if la4_e8 {
        let (mqkv, mz, mb, ma) = (layer.wqkv.m, layer.wz.m, layer.w_beta.m, layer.w_alpha.m);
        let yfq = gpu.alloc_tensor(&[mqkv], DType::F32).unwrap();
        let yfz = gpu.alloc_tensor(&[mz], DType::F32).unwrap();
        let yfb = gpu.alloc_tensor(&[mb], DType::F32).unwrap();
        let yfa = gpu.alloc_tensor(&[ma], DType::F32).unwrap();
        gpu.fused_qkvza_mfp4g32_e8(
            &layer.wqkv.buf,
            &layer.wz.buf,
            &layer.w_beta.buf,
            &layer.w_alpha.buf,
            &x_dev,
            &yfq,
            &yfz,
            &yfb,
            &yfa,
            mqkv,
            mz,
            mb,
            ma,
            k,
        )
        .expect("fused qkvza");
        let yrq = gpu.alloc_tensor(&[mqkv], DType::F32).unwrap();
        let yrz = gpu.alloc_tensor(&[mz], DType::F32).unwrap();
        let yrb = gpu.alloc_tensor(&[mb], DType::F32).unwrap();
        let yra = gpu.alloc_tensor(&[ma], DType::F32).unwrap();
        gpu.gemv_mfp4g32_e8(&layer.wqkv.buf, &x_dev, &yrq, mqkv, k)
            .unwrap();
        gpu.gemv_mfp4g32_e8(&layer.wz.buf, &x_dev, &yrz, mz, k)
            .unwrap();
        gpu.gemv_mfp4g32_e8(&layer.w_beta.buf, &x_dev, &yrb, mb, k)
            .unwrap();
        gpu.gemv_mfp4g32_e8(&layer.w_alpha.buf, &x_dev, &yra, ma, k)
            .unwrap();
        fails += cmp("qkv", &gpu, &yfq, &yrq);
        fails += cmp("z", &gpu, &yfz, &yrz);
        fails += cmp("beta", &gpu, &yfb, &yrb);
        fails += cmp("alpha", &gpu, &yfa, &yra);
    } else {
        eprintln!("qkvza: beta/alpha/z not all MFP4G32E8 — skipping qkvza parity (fusion won't engage either)");
    }

    if fails == 0 {
        println!("PARITY PASS: fused == per-projection bit-for-bit (all outputs)");
    } else {
        println!("PARITY FAIL: {fails} output(s) differ");
        std::process::exit(1);
    }
}

#[cfg(feature = "deltanet")]
fn cmp(
    name: &str,
    gpu: &rdna_compute::Gpu,
    fused: &rdna_compute::GpuTensor,
    refr: &rdna_compute::GpuTensor,
) -> u32 {
    let a = gpu.download_f32(fused).unwrap();
    let b = gpu.download_f32(refr).unwrap();
    assert_eq!(a.len(), b.len());
    let mut ndiff = 0usize;
    let mut max_abs = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            ndiff += 1;
            max_abs = max_abs.max((x - y).abs());
        }
    }
    if ndiff == 0 {
        eprintln!("  {name:6}: {} rows, BIT-EXACT", a.len());
        0
    } else {
        eprintln!(
            "  {name:6}: {ndiff}/{} rows differ (max |Δ|={max_abs:.3e}) FAIL",
            a.len()
        );
        1
    }
}
