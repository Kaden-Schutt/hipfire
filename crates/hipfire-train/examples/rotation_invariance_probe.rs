//! SpinQuant R1 Phase 0: prove the rotation-invariant transform leaves the fp32
//! forward unchanged. Builds a tiny 2-layer LLaMA (tied embeddings), runs the
//! forward for baseline logits, bakes `R1` into the weights with [`apply_r1`],
//! and re-runs — the logits must match up to fp reassociation.
//!
//! Two rotations are checked:
//!   • identity  — isolates the RMSNorm-scale fold (must be near bit-exact),
//!   • random    — the full orthonormal residual rotation.
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire lock acquire "rotation-invariance-probe"
//!   cargo run -p hipfire-train --release --example rotation_invariance_probe
//!   hipfire lock release

use hipfire_train::block::BlockDims;
use hipfire_train::model::{model_forward, LayerLora, LayerWeights, LlamaModel};
use hipfire_train::rotation::{apply_r1, Rotation};
use rdna_compute::{Gpu, HipResult};

const NL: usize = 2;
const SEQ: usize = 4;
const H: usize = 16;
const NH: usize = 4;
const NKV: usize = 2;
const HD: usize = 4;
const INTER: usize = 32;
const VOCAB: usize = 12;
const QD: usize = NH * HD;
const KVD: usize = NKV * HD;

fn dims() -> BlockDims {
    BlockDims {
        seq: SEQ,
        h: H,
        n_heads: NH,
        n_kv: NKV,
        head_dim: HD,
        inter: INTER,
        rope_base: 10000.0,
        eps: 1e-6,
        lora_scale: 1.0,
        lora_rank: 2,
    }
}

fn rnd(n: usize, a: usize, b: usize, scale: f32, off: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * a % b) as f32) * scale + off).collect()
}

// Deterministic base weights, varied per layer. Norm scales are deliberately
// non-unit (0.7..1.1) so the fold actually does something.
fn base_layer(l: usize) -> [Vec<f32>; 9] {
    let s = l + 1;
    [
        rnd(H, 3 * s, 7, 0.05, 0.7),            // norm1
        rnd(QD * H, 11 * s, 7, 0.06, -0.2),     // wq
        rnd(KVD * H, 13 * s, 9, 0.06, -0.2),    // wk
        rnd(KVD * H, 5 * s, 11, 0.06, -0.2),    // wv
        rnd(H * QD, 7 * s, 13, 0.06, -0.2),     // wo
        rnd(H, 5 * s, 6, 0.05, 0.75),           // norm2
        rnd(INTER * H, 9 * s, 7, 0.05, -0.15),  // wgate
        rnd(INTER * H, 11 * s, 5, 0.05, -0.15), // wup
        rnd(H * INTER, 13 * s, 7, 0.05, -0.15), // wdown
    ]
}

fn build_model(gpu: &mut Gpu) -> HipResult<LlamaModel> {
    let embed = gpu.upload_f32(&rnd(VOCAB * H, 7, 13, 0.05, -0.1), &[VOCAB * H])?;
    let final_norm = gpu.upload_f32(&rnd(H, 3, 5, 0.05, 0.8), &[H])?;
    let mut layers = Vec::with_capacity(NL);
    for l in 0..NL {
        let b = base_layer(l);
        let lw = LayerWeights {
            norm1: gpu.upload_f32(&b[0], &[H])?,
            wq: gpu.upload_f32(&b[1], &[QD * H])?,
            wk: gpu.upload_f32(&b[2], &[KVD * H])?,
            wv: gpu.upload_f32(&b[3], &[KVD * H])?,
            wo: gpu.upload_f32(&b[4], &[H * QD])?,
            norm2: gpu.upload_f32(&b[5], &[H])?,
            wgate: gpu.upload_f32(&b[6], &[INTER * H])?,
            wup: gpu.upload_f32(&b[7], &[INTER * H])?,
            wdown: gpu.upload_f32(&b[8], &[H * INTER])?,
        };
        // LoRA B=0 ⇒ zero contribution; A arbitrary. Keeps the base forward pure.
        let ll = LayerLora {
            aq: gpu.upload_f32(&rnd(2 * H, 7, 5, 0.1, -0.2), &[2 * H])?,
            bq: gpu.zeros(&[QD * 2], rdna_compute::DType::F32)?,
            av: gpu.upload_f32(&rnd(2 * H, 13, 9, 0.1, -0.2), &[2 * H])?,
            bv: gpu.zeros(&[KVD * 2], rdna_compute::DType::F32)?,
        };
        layers.push((lw, ll));
    }
    Ok(LlamaModel {
        embed,
        lm_head: None,
        final_norm,
        layers,
        dims: dims(),
        vocab: VOCAB,
    })
}

fn logits(gpu: &mut Gpu, model: &LlamaModel, tokens: &[u32], pos: &[f32]) -> HipResult<Vec<f32>> {
    let acts = model_forward(gpu, model, tokens, pos)?;
    gpu.download_f32(&acts.logits)
}

/// max abs diff + relative-L2 between two logit vectors.
fn compare(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut maxabs = 0.0f32;
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        maxabs = maxabs.max((x - y).abs());
        num += (x - y) * (x - y);
        den += x * x;
    }
    (maxabs, (num / den.max(1e-20)).sqrt())
}

fn run_case(gpu: &mut Gpu, name: &str, rot: &Rotation, base: &[f32]) -> HipResult<f32> {
    let tokens: Vec<u32> = vec![2, 5, 1, 8];
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let mut model = build_model(gpu)?;
    apply_r1(gpu, &mut model, rot)?;
    let rotated = logits(gpu, &model, &tokens, &pos)?;
    let (maxabs, rell2) = compare(base, &rotated);
    println!(
        "  [{name}] orthonormality={:.1e}  max|Δlogit|={maxabs:.3e}  relL2={rell2:.3e}",
        rot.orthonormality_error()
    );
    Ok(maxabs)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let tokens: Vec<u32> = vec![2, 5, 1, 8];
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();

    // Baseline (untransformed) logits.
    let base_model = build_model(&mut gpu)?;
    let base = logits(&mut gpu, &base_model, &tokens, &pos)?;
    println!("baseline logits[0..4] = {:?}", &base[..4]);

    // Identity R ⇒ pure norm-scale fold: must be (near) bit-exact.
    let ident = run_case(
        &mut gpu,
        "identity (fold only)",
        &Rotation::identity(H),
        &base,
    )?;
    // Random orthonormal R ⇒ full residual rotation.
    let rnd1 = run_case(&mut gpu, "random R1 #1", &Rotation::random(H, 1), &base)?;
    let rnd2 = run_case(&mut gpu, "random R1 #2", &Rotation::random(H, 99), &base)?;

    // Fold-only should be tiny (just reassociation of the scale into the GEMM);
    // rotation adds a dense h×h GEMM of reassociation, so allow a looser bound.
    let ok = ident < 1e-4 && rnd1 < 5e-3 && rnd2 < 5e-3;
    if ok {
        println!("\nPHASE 0 PASS — R1 transform leaves the fp32 forward invariant.");
        Ok(())
    } else {
        Err(format!("PHASE 0 FAIL — ident {ident:.2e}, rnd1 {rnd1:.2e}, rnd2 {rnd2:.2e}").into())
    }
}
