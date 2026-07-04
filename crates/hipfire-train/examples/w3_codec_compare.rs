//! QTIP3 vs Oq3 (op3): a 3-bit weight-codec head-to-head at equal footing (both plain
//! — no LDLQ — same FWHT-256 rotation, seeds 42/1042). Isolates the CODEC: trellis-coded
//! QTIP3 vs symmetric-int3 Oq3. Reports weight-recon SQNR (per-tensor, energy-weighted)
//! AND end-to-end held-out KLD vs the fp teacher.
//!
//! Bytes/weight: Oq3 = 98 B/256-group = 3.0625 b/w ; QTIP3 = 100 B/group = 3.125 b/w
//! (QTIP3 costs ~2% more bytes). Memory `project_lowbit_quant_findings`: trellis > affine
//! at every bit — so QTIP3 is expected to win recon; this measures by how much + whether
//! it survives to end-to-end KLD.
//!
//!   hipfire lock acquire "w3-codec-cmp"
//!   cargo run -p hipfire-train --release --example w3_codec_compare [model_dir]
//!   hipfire lock release

use hipfire_train::loader::{load_llama_fp32, LlamaWeightsF32};
use hipfire_train::model::{model_distill_backward, model_forward, LlamaModel};
use hipfire_train::ops::softmax::softmax_forward;
use hipfire_train::oqplus_quant::{oq3_simquant, oq8_simquant, oqplus_simquant};
use hipfire_train::qtip_quant::qtip_quantize_dequant;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::path::Path;

const DEFAULT_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const SEQ: usize = 16;
const N_EVAL: usize = 4;
const QTIP_BEAM: usize = 16;

fn sqnr(orig: &[f32], q: &[f32]) -> (f64, f64) {
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for (&a, &b) in orig.iter().zip(q) {
        sig += (a as f64) * (a as f64);
        let d = (a - b) as f64;
        noise += d * d;
    }
    (sig, noise)
}

/// Quantize the 7 linears/layer with `codec`, accumulating weight-recon energy/noise.
fn quantize_and_measure(
    gpu: &mut Gpu,
    w: &mut LlamaWeightsF32,
    codec: &dyn Fn(&[f32]) -> Vec<f32>,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for l in w.layers.iter_mut() {
        for t in [
            &mut l.q_proj,
            &mut l.k_proj,
            &mut l.v_proj,
            &mut l.o_proj,
            &mut l.gate_proj,
            &mut l.up_proj,
            &mut l.down_proj,
        ] {
            let host = gpu.download_f32(t)?;
            let q = codec(&host);
            let (s, n) = sqnr(&host, &q);
            sig += s;
            noise += n;
            *t = gpu.upload_f32(&q, &t.shape.clone())?;
        }
    }
    Ok((sig, noise))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_DIR.to_string());
    let dir = Path::new(&dir);
    if !dir.exists() {
        return Err(format!("model dir not found: {}", dir.display()).into());
    }
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}  model: {}", gpu.arch, dir.display());

    let (cfg, w_teacher) = load_llama_fp32(&mut gpu, dir)?;
    let vocab = cfg.vocab_size;
    let teacher = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_teacher, SEQ, 1, 1.0)?;

    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    // Held-out eval batch + clean teacher distributions.
    let eval_batch: Vec<Vec<u32>> = (0..N_EVAL)
        .map(|s| {
            (0..SEQ)
                .map(|t| (((t + 1) * 2654435761 + (s + 1000) * 40503) % vocab) as u32)
                .collect()
        })
        .collect();
    let mut teacher_p: Vec<GpuTensor> = Vec::with_capacity(N_EVAL);
    for toks in &eval_batch {
        let at = model_forward(&mut gpu, &teacher, toks, &pos)?;
        let p = gpu.zeros(&[SEQ * vocab], DType::F32)?;
        softmax_forward(&mut gpu, &at.logits, &p, SEQ, vocab)?;
        teacher_p.push(p);
    }

    // Oq3 = int3 W3A4 baseline. QTIP3 = trellis A16 ceiling. QTIP3→int4/int8 = store the
    // trellis (3-bit bytes) but expand to the int4/int8 grid the iu4/iu8 WMMA consumes.
    let oq3 = |w: &[f32]| oq3_simquant(w);
    let qtip3 = |w: &[f32]| qtip_quantize_dequant(w, 3, QTIP_BEAM);
    let qtip3_i4 = |w: &[f32]| oqplus_simquant(&qtip_quantize_dequant(w, 3, QTIP_BEAM));
    let qtip3_i8 = |w: &[f32]| oq8_simquant(&qtip_quantize_dequant(w, 3, QTIP_BEAM));
    let codecs: [(&str, &dyn Fn(&[f32]) -> Vec<f32>); 4] = [
        ("Oq3       (int3 W3A4 baseline)", &oq3),
        ("QTIP3     (trellis, A16 ceiling)", &qtip3),
        ("QTIP3-int4 (store3b, iu4 operand)", &qtip3_i4),
        ("QTIP3-int8 (store3b, iu8 operand)", &qtip3_i8),
    ];

    println!("\n  codec                             | recon SQNR |  held-out KLD");
    println!("  ----------------------------------+------------+--------------");
    for (name, codec) in codecs {
        let (_, mut w_student) = load_llama_fp32(&mut gpu, dir)?; // fresh fp weights
        let (sig, noise) = quantize_and_measure(&mut gpu, &mut w_student, codec)?;
        let recon = 10.0 * (sig / noise.max(1e-30)).log10();
        let student = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_student, SEQ, 1, 1.0)?;
        let mut kl = 0.0f32;
        for (si, toks) in eval_batch.iter().enumerate() {
            let acts = model_forward(&mut gpu, &student, toks, &pos)?;
            let (k, _g, _d) = model_distill_backward(&mut gpu, &student, &acts, &teacher_p[si])?;
            kl += k;
        }
        let kl = kl / (N_EVAL * SEQ) as f32;
        println!("  {name:<33} | {recon:7.2} dB | {kl:.4} nats/tok");
    }
    println!("\n  (both plain — no LDLQ — same FWHT-256; isolates the codec. QTIP3 costs ~2% more\n   bytes. Trellis expected to win recon; KLD shows if it survives end-to-end.)");
    Ok(())
}
