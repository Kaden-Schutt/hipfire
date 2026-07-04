#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

//! Hardening run for the OQ+ / KV-noise recovery probes: real wikitext corpus
//! with a DISJOINT train/held-out split, so the reported number is held-out KL
//! (generalization), not fit-to-calibration. Distill on the TRAIN chunks only;
//! report KL on both TRAIN and HELD-OUT through training. A held-out drop that
//! tracks the train drop = real recovery; a held-out drop that lags = overfit.
//!
//! Mode (HIPFIRE_RECOVER_NOISE):
//!   oqplus  (default) — student weights OQ+ sim-quant (W4A8); KV clean.
//!   kvnoise           — student weights fp32 (clean); KVarN-4bit + CASK merge
//!                       on the student's attention forward (HIPFIRE_KVNOISE_*).
//! Recovery surface (HIPFIRE_RECOVER_MODE): lora+norms (default) | norms.
//! Teacher is always clean fp32. LR via HIPFIRE_RECOVER_LR (default 3e-4).
//!
//! Run:
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire gpu-lock acquire "recovery-gen" --watch-pid $$
//!   cargo run -p hipfire-train --release --example recovery_generalization_supra50m
//!   hipfire gpu-lock release

use hipfire_model::tokenizer::Tokenizer;
use hipfire_train::block::BlockLoraGrad;
use hipfire_train::loader::{load_llama_fp32, LlamaWeightsF32};
use hipfire_train::model::{
    flatten_norm_grads, flatten_recovery_grads, free_model_acts, model_distill_backward,
    model_distill_backward_tail, model_forward, LlamaModel,
};
use hipfire_train::ops::softmax::softmax_forward;
use hipfire_train::optim::AdamW;
use hipfire_train::oqplus_quant::oqplus_simquant;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::path::Path;

const MODEL_DIR: &str =
    "/srv/huggingface/models--SupraLabs--Supra-50M-Instruct/snapshots/77a1c2a33f386f9f4bf7151ec5f2156b62caac39";
const CORPUS_FILE: &str = "benchmarks/calib/calib-1m.txt";
const L: usize = 32;
const TRAIN_CHUNKS: usize = 16;
const HELDOUT_CHUNKS: usize = 8;
const RANK: usize = 16;
const ALPHA: f32 = 32.0;
const STEPS: usize = 100;

fn envf(key: &str, d: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Free a vec of LoRA grads (no Drop on GpuTensor → must free per-iter or the
/// pool balloons → 719 wedge over many steps).
fn free_grads(gpu: &mut Gpu, grads: Vec<BlockLoraGrad>) -> Result<(), Box<dyn std::error::Error>> {
    for g in grads {
        let BlockLoraGrad {
            daq,
            dbq,
            dav,
            dbv,
            dnorm1,
            dnorm2,
        } = g;
        for t in [daq, dbq, dav, dbv, dnorm1, dnorm2] {
            gpu.free_tensor(t)?;
        }
    }
    Ok(())
}

fn oqplus_quantize_linears(
    gpu: &mut Gpu,
    w: &mut LlamaWeightsF32,
) -> Result<(), Box<dyn std::error::Error>> {
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
            let q = oqplus_simquant(&host);
            *t = gpu.upload_f32(&q, &t.shape.clone())?;
        }
    }
    Ok(())
}

/// Distill backward — tail-scored (last `n_score` positions) when n_score>0,
/// else full-sequence. Tail scoring is the leak-free KVarN+CASK measurement.
#[allow(clippy::type_complexity)]
fn distill(
    gpu: &mut Gpu,
    model: &LlamaModel,
    acts: &hipfire_train::model::ModelActivations,
    tp: &GpuTensor,
    n_score: usize,
) -> Result<(f32, Vec<BlockLoraGrad>, GpuTensor), Box<dyn std::error::Error>> {
    if n_score > 0 {
        Ok(model_distill_backward_tail(gpu, model, acts, tp, n_score)?)
    } else {
        Ok(model_distill_backward(gpu, model, acts, tp)?)
    }
}

/// Mean held-out KL (forward + distill backward, discard grads, free acts).
fn heldout_kl(
    gpu: &mut Gpu,
    student: &LlamaModel,
    chunks: &[Vec<u32>],
    teacher_p: &[GpuTensor],
    pos: &[f32],
    n_score: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let per = if n_score > 0 { n_score } else { L };
    let mut total = 0.0f32;
    for (ci, toks) in chunks.iter().enumerate() {
        let acts = model_forward(gpu, student, toks, pos)?;
        let (kl, g, d_final) = distill(gpu, student, &acts, &teacher_p[ci], n_score)?;
        total += kl;
        free_grads(gpu, g)?;
        gpu.free_tensor(d_final)?;
        free_model_acts(gpu, acts)?;
    }
    Ok(total / (chunks.len() * per) as f32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(MODEL_DIR);
    if !dir.exists() {
        return Err(format!("model dir not found: {MODEL_DIR}").into());
    }
    std::env::remove_var("HIPFIRE_KVNOISE"); // teacher + weight-mode student stay clean

    let mode = std::env::var("HIPFIRE_RECOVER_NOISE").unwrap_or_else(|_| "oqplus".into());
    let norms_only = std::env::var("HIPFIRE_RECOVER_MODE").as_deref() == Ok("norms");
    let lr = envf("HIPFIRE_RECOVER_LR", 3e-4);
    // HIPFIRE_SCORE_TAIL=N → score only the last N query positions (leak-free for
    // KVarN+CASK: tail queries read merged cold keys strictly in their past). 0 =
    // score all positions (only valid when there's no cross-token KV merge).
    let n_score = std::env::var("HIPFIRE_SCORE_TAIL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0usize);
    let per_tok = if n_score > 0 { n_score } else { L };

    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!(
        "arch: {}  mode={mode}  recover={}  lr={lr}  score_tail={n_score}",
        gpu.arch,
        if norms_only { "norms" } else { "lora+norms" }
    );

    let tok = Tokenizer::from_hf_json(&std::fs::read_to_string(dir.join("tokenizer.json"))?)
        .map_err(|e| format!("tokenizer: {e:?}"))?;
    // Read enough text for all chunks (well under the 4.8MB file).
    let need = (TRAIN_CHUNKS + HELDOUT_CHUNKS) * L;
    let raw = std::fs::read_to_string(CORPUS_FILE)?;
    let slice: String = raw.chars().take(need * 12 + 4096).collect();
    let ids = tok.encode(&slice);
    if ids.len() < need {
        return Err(format!("corpus too short: {} < {need} tokens", ids.len()).into());
    }
    println!(
        "corpus: {} tokens → {TRAIN_CHUNKS} train + {HELDOUT_CHUNKS} held-out chunks of {L}",
        ids.len()
    );

    let (cfg, w_teacher) = load_llama_fp32(&mut gpu, dir)?;
    let (_, mut w_student) = load_llama_fp32(&mut gpu, dir)?;
    let vocab = cfg.vocab_size;
    if mode == "oqplus" {
        println!("OQ+ sim-quant of student weights (W4A8)...");
        oqplus_quantize_linears(&mut gpu, &mut w_student)?;
    }

    let teacher = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_teacher, L, RANK, ALPHA)?;
    let student = LlamaModel::from_f32_weights(&mut gpu, &cfg, w_student, L, RANK, ALPHA)?;

    // Disjoint chunks: [0, TRAIN) train, [TRAIN, TRAIN+HELDOUT) held-out.
    let pos: Vec<f32> = (0..L).map(|t| t as f32).collect();
    let total_chunks = TRAIN_CHUNKS + HELDOUT_CHUNKS;
    let mut chunks: Vec<Vec<u32>> = Vec::with_capacity(total_chunks);
    let mut teacher_p: Vec<GpuTensor> = Vec::with_capacity(total_chunks);
    for c in 0..total_chunks {
        let toks = ids[c * L..(c + 1) * L].to_vec();
        let at = model_forward(&mut gpu, &teacher, &toks, &pos)?; // teacher clean (noise off)
        let p = gpu.zeros(&[L * vocab], DType::F32)?;
        softmax_forward(&mut gpu, &at.logits, &p, L, vocab)?;
        free_model_acts(&mut gpu, at)?;
        teacher_p.push(p);
        chunks.push(toks);
    }
    let (train_p, held_p) = teacher_p.split_at(TRAIN_CHUNKS);
    let train_c = &chunks[..TRAIN_CHUNKS];
    let held_c = &chunks[TRAIN_CHUNKS..];

    // Now arm KV-noise for student forwards (no-op in oqplus mode).
    if mode == "kvnoise" {
        std::env::set_var("HIPFIRE_KVNOISE", "1");
    }

    let sizes = if norms_only {
        student.norm_param_sizes()
    } else {
        student.recovery_param_sizes()
    };
    let mut opt = AdamW::new(&mut gpu, &sizes, lr, 0.9, 0.999, 1e-8, 0.0)?;

    let kl0_train;
    let kl0_held = heldout_kl(&mut gpu, &student, held_c, held_p, &pos, n_score)?;
    {
        // measure starting train KL too
        let mut t = 0.0f32;
        for (ci, toks) in train_c.iter().enumerate() {
            let acts = model_forward(&mut gpu, &student, toks, &pos)?;
            let (kl, g, d_final) = distill(&mut gpu, &student, &acts, &train_p[ci], n_score)?;
            t += kl;
            free_grads(&mut gpu, g)?;
            gpu.free_tensor(d_final)?;
            free_model_acts(&mut gpu, acts)?;
        }
        kl0_train = t / (TRAIN_CHUNKS * per_tok) as f32;
    }
    println!("\nstart: train KL {kl0_train:.4}  held-out KL {kl0_held:.4} nats/tok");

    let mut last_train = kl0_train;
    let mut last_held = kl0_held;
    for step in 1..=STEPS {
        let mut total = 0.0f32;
        for (ci, toks) in train_c.iter().enumerate() {
            let acts = model_forward(&mut gpu, &student, toks, &pos)?;
            let (kl, grads, d_final) = distill(&mut gpu, &student, &acts, &train_p[ci], n_score)?;
            total += kl;
            if norms_only {
                let params = student.norm_params();
                let gflat = flatten_norm_grads(&grads, &d_final);
                opt.step(&mut gpu, &params, &gflat)?;
            } else {
                let params = student.recovery_params();
                let gflat = flatten_recovery_grads(&grads, &d_final);
                opt.step(&mut gpu, &params, &gflat)?;
            }
            free_grads(&mut gpu, grads)?;
            gpu.free_tensor(d_final)?;
            free_model_acts(&mut gpu, acts)?;
        }
        last_train = total / (TRAIN_CHUNKS * per_tok) as f32;
        if !last_train.is_finite() {
            println!("step {step:3}: train KL diverged (NaN/inf) — stopping; STE instability, lower HIPFIRE_RECOVER_LR");
            break;
        }
        if step % 20 == 0 || step == STEPS {
            last_held = heldout_kl(&mut gpu, &student, held_c, held_p, &pos, n_score)?;
            println!("step {step:3}: train KL {last_train:.4}  held-out KL {last_held:.4}");
        }
    }

    let rec_train = 100.0 * (kl0_train - last_train) / kl0_train.max(1e-6);
    let rec_held = 100.0 * (kl0_held - last_held) / kl0_held.max(1e-6);
    println!(
        "\n{mode}: train {kl0_train:.4}→{last_train:.4} ({rec_train:.1}%)  \
              HELD-OUT {kl0_held:.4}→{last_held:.4} ({rec_held:.1}%)"
    );
    let gen_ratio = if rec_train.abs() > 1e-3 {
        100.0 * rec_held / rec_train
    } else {
        0.0
    };
    println!("generalization: held-out recovers {gen_ratio:.0}% of the train recovery");
    if rec_held > 40.0 {
        println!("HOLDS — recovery generalizes to unseen text.");
    } else if rec_held > 10.0 {
        println!("PARTIAL — generalizes but weaker than the fit-to-calibration ceiling.");
    } else {
        println!("OVERFIT/FLOOR — train recovers but held-out barely moves.");
    }
    Ok(())
}
