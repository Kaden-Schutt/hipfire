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

//! Phase A (qwen3.5 OQ+ norm recovery): MLP-norm block-local distillation.
//!
//! For each layer, recover the `post_attention_layernorm` weight (γ) so the
//! OQ+-quantized MLP reproduces the bf16 teacher's block output — using ONLY the
//! daemon-captured residuals, so the (non-differentiable) DeltaNet/attn mixer is
//! never run. Per layer i:
//!   input  = x_mid_i  (captured `qwen35.premlp.L{i}`  — post-mixer, pre-FFN)
//!   target = x_out_i  (captured `qwen35.pertoken.L{i}` — block output)
//!   student: x_mid → rmsnorm(γ, trainable) → gate/up (OQ+ sim-quant, frozen)
//!            → swiglu → down (OQ+) → + x_mid  ==>  predict x_out
//!   loss = MSE(pred, x_out); AdamW on γ only.
//! Because the residual x_mid is identical on both sides, MSE isolates the MLP
//! quant error — exactly what γ can partially compensate.
//!
//! Capture first (see commit 402392f8):
//!   HIPFIRE_FORWARD_LOWERED=0 HIPFIRE_DUMP_HIDDEN=/tmp/residcap/qwen35 \
//!   HIPFIRE_DUMP_HIDDEN_ALL=1 HIPFIRE_DUMP_HIDDEN_ALLLAYERS=1 HIPFIRE_MAX_GEN=4 \
//!   infer_qwen35 ~/.hipfire/models/qwen3.5-0.8b-bf16.hfq --guards off "<text>"
//!
//! Run:
//!   hipfire gpu-lock acquire "qwen35-mlp-norm" --watch-pid $$
//!   cargo run -p hipfire-train --release --example qwen35_mlp_norm_recovery
//!   hipfire gpu-lock release
//! Env: HIPFIRE_CAP_DIR (default /tmp/residcap), HIPFIRE_MODEL (bf16 .hfq),
//!      HIPFIRE_RECOVER_LR (3e-4), HIPFIRE_RECOVER_STEPS (200).

use hipfire_train::hfq_patch::{bf16_bits_to_f32, parse_hfq, HfqEntry};
use hipfire_train::ops::linear::{linear_backward_x, linear_forward};
use hipfire_train::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use hipfire_train::ops::swiglu::{swiglu_backward, swiglu_forward};
use hipfire_train::optim::AdamW;
use hipfire_train::oqplus_quant::oqplus_simquant;
use hipfire_rdna::{DType, Gpu};
use std::collections::HashMap;

const QT_BF16: u8 = 16;

fn envs(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}
fn envf(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
fn envu(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Read a bf16-stored tensor from the parsed .hfq bytes as f32.
fn read_bf16(bytes: &[u8], e: &HfqEntry) -> Result<Vec<f32>, String> {
    if e.quant_type != QT_BF16 {
        return Err(format!(
            "{}: quant_type {} != bf16(16)",
            e.name, e.quant_type
        ));
    }
    let n = e.data_size / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = e.data_offset + i * 2;
        out.push(bf16_bits_to_f32(u16::from_le_bytes([
            bytes[off],
            bytes[off + 1],
        ])));
    }
    Ok(out)
}

/// Read a captured residual file `{dir}/qwen35.{tag}.L{i}` as rows × dim f32.
fn read_cap(dir: &str, tag: &str, i: usize, dim: usize) -> Result<(Vec<f32>, usize), String> {
    let p = format!("{dir}/qwen35.{tag}.L{i}");
    let raw = std::fs::read(&p).map_err(|e| format!("{p}: {e}"))?;
    if raw.len() % (dim * 4) != 0 {
        return Err(format!("{p}: {} bytes not a multiple of dim*4", raw.len()));
    }
    let rows = raw.len() / (dim * 4);
    let mut out = Vec::with_capacity(rows * dim);
    for c in raw.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok((out, rows))
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len() as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cap_dir = envs("HIPFIRE_CAP_DIR", "/tmp/residcap");
    let model = envs(
        "HIPFIRE_MODEL",
        &format!(
            "{}/.hipfire/models/qwen3.5-0.8b-bf16.hfq",
            std::env::var("HOME").unwrap()
        ),
    );
    let lr = envf("HIPFIRE_RECOVER_LR", 3e-4);
    let steps = envu("HIPFIRE_RECOVER_STEPS", 200);

    let bytes = std::fs::read(&model)?;
    let (entries, meta) = parse_hfq(&bytes).map_err(|e| format!("parse_hfq: {e}"))?;
    let by_name: HashMap<&str, &HfqEntry> = entries.iter().map(|e| (e.name.as_str(), e)).collect();

    // dims from metadata json — config nests under config.text_config.
    let mj: serde_json::Value = serde_json::from_str(&meta)?;
    let tc = mj
        .get("config")
        .and_then(|c| c.get("text_config"))
        .or_else(|| mj.get("config"))
        .unwrap_or(&mj);
    let getu = |k: &str| -> usize {
        tc.get(k)
            .or_else(|| mj.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize
    };
    let dim = if getu("dim") > 0 {
        getu("dim")
    } else {
        getu("hidden_size")
    };
    let inter = if getu("hidden_dim") > 0 {
        getu("hidden_dim")
    } else {
        getu("intermediate_size")
    };
    let n_layers = if getu("n_layers") > 0 {
        getu("n_layers")
    } else {
        getu("num_hidden_layers")
    };
    let eps = (tc
        .get("norm_eps")
        .or_else(|| tc.get("rms_norm_eps"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6)) as f32;
    println!("model: dim={dim} inter={inter} layers={n_layers} eps={eps}  lr={lr} steps={steps}");

    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}\n", gpu.arch);

    // Find the layer-prefix scheme by probing a known tensor name (qwen3.5-VL
    // nests under model.language_model.layers; text-only under model.layers).
    let prefix = |i: usize, leaf: &str| -> String {
        for base in ["model.language_model.layers", "model.layers", "layers"] {
            let cand = format!("{base}.{i}.{leaf}");
            if by_name.contains_key(cand.as_str()) {
                return cand;
            }
        }
        format!("model.language_model.layers.{i}.{leaf}") // default; error surfaces on lookup
    };

    let mut total_start = 0.0f64;
    let mut total_final = 0.0f64;
    let mut n_done = 0usize;
    let mut tuned_norms: HashMap<String, Vec<f32>> = HashMap::new();

    for i in 0..n_layers {
        // Load this layer's MLP weights + norm. Skip layers whose MLP tensors
        // aren't present as bf16 (e.g. MoE layers) — Phase A is dense-MLP norms.
        let names = [
            (prefix(i, "mlp.gate_proj.weight"), "gate"),
            (prefix(i, "mlp.up_proj.weight"), "up"),
            (prefix(i, "mlp.down_proj.weight"), "down"),
            (prefix(i, "post_attention_layernorm.weight"), "norm"),
        ];
        let mut missing = false;
        for (nm, _) in &names {
            match by_name.get(nm.as_str()) {
                Some(e) if e.quant_type == QT_BF16 => {}
                _ => {
                    missing = true;
                }
            }
        }
        if missing {
            continue;
        }
        let w_gate_h = read_bf16(&bytes, by_name[names[0].0.as_str()])?;
        let w_up_h = read_bf16(&bytes, by_name[names[1].0.as_str()])?;
        let w_down_h = read_bf16(&bytes, by_name[names[2].0.as_str()])?;
        let norm_name = names[3].0.clone();
        let mut gamma_h = read_bf16(&bytes, by_name[norm_name.as_str()])?;
        // qwen3.5 may store RMSNorm weight as (1+γ) (Gemma-style). HIPFIRE_NORM_PLUS1=1
        // adds 1.0 so the effective scale is ~1 (stored values centered near 0).
        if std::env::var("HIPFIRE_NORM_PLUS1").as_deref() == Ok("1") {
            for v in gamma_h.iter_mut() {
                *v += 1.0;
            }
        }

        // Only the FFN INPUT (premlp = what gate_up norms) is taken from the daemon
        // capture — verified bit-exact (norm + gate outputs match the model). The
        // teacher FFN output is computed IN THE TRAINER from bf16 weights, so it is
        // correct for EVERY layer (no dependence on residual-difference captures,
        // which are contaminated by the attention add for the first few layers).
        let (xin_h, rows) = read_cap(&cap_dir, "premlp", i, dim)?;
        let x_in = gpu.upload_f32(&xin_h, &[rows * dim])?;

        // Trainable γ (init = bf16 norm, with (1+γ) already folded above).
        let gamma = gpu.upload_f32(&gamma_h, &[dim])?;
        let mut opt = AdamW::new(&mut gpu, &[dim], lr, 0.9, 0.999, 1e-8, 0.0)?;

        // scratch
        let yn = gpu.zeros(&[rows * dim], DType::F32)?;
        let rinv = gpu.zeros(&[rows], DType::F32)?;
        let g = gpu.zeros(&[rows * inter], DType::F32)?;
        let u = gpu.zeros(&[rows * inter], DType::F32)?;
        let act = gpu.zeros(&[rows * inter], DType::F32)?;
        let mlp = gpu.zeros(&[rows * dim], DType::F32)?;
        let d_act = gpu.zeros(&[rows * inter], DType::F32)?;
        let d_g = gpu.zeros(&[rows * inter], DType::F32)?;
        let d_u = gpu.zeros(&[rows * inter], DType::F32)?;
        let d_yn = gpu.zeros(&[rows * dim], DType::F32)?;
        let d_xmid_unused = gpu.zeros(&[rows * dim], DType::F32)?;

        // FFN forward parameterized by weights: down(swiglu(gate(norm_γ(x_in)),up)).
        let run_fwd = |gpu: &mut Gpu,
                       wg: &hipfire_rdna::GpuTensor,
                       wu: &hipfire_rdna::GpuTensor,
                       wd: &hipfire_rdna::GpuTensor|
         -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            rmsnorm_forward(gpu, &x_in, &gamma, &yn, &rinv, rows, dim, eps)?;
            linear_forward(gpu, &yn, wg, &g, rows, dim, inter)?;
            linear_forward(gpu, &yn, wu, &u, rows, dim, inter)?;
            swiglu_forward(gpu, &g, &u, &act, rows * inter)?;
            linear_forward(gpu, &act, wd, &mlp, rows, inter, dim)?;
            gpu.download_f32(&mlp)
                .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })
        };

        // Teacher target = bf16 FFN output (computed here, not captured).
        let wg_bf = gpu.upload_f32(&w_gate_h, &[inter, dim])?;
        let wu_bf = gpu.upload_f32(&w_up_h, &[inter, dim])?;
        let wd_bf = gpu.upload_f32(&w_down_h, &[dim, inter])?;
        let ffn_target_h = run_fwd(&mut gpu, &wg_bf, &wu_bf, &wd_bf)?;
        for t in [wg_bf, wu_bf, wd_bf] {
            gpu.free_tensor(t)?;
        }

        // Student weights = OQ+ sim-quant (HIPFIRE_NO_QUANT=1 → identity sanity check).
        let q = |w: &[f32]| {
            if std::env::var("HIPFIRE_NO_QUANT").as_deref() == Ok("1") {
                w.to_vec()
            } else {
                oqplus_simquant(w)
            }
        };
        let w_gate = gpu.upload_f32(&q(&w_gate_h), &[inter, dim])?;
        let w_up = gpu.upload_f32(&q(&w_up_h), &[inter, dim])?;
        let w_down = gpu.upload_f32(&q(&w_down_h), &[dim, inter])?;

        let fwd_ffn = |gpu: &mut Gpu| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            run_fwd(gpu, &w_gate, &w_up, &w_down)
        };

        // Diagnostic: quant-vs-bf16 fidelity per layer (target is the bf16 FFN).
        if std::env::var("HIPFIRE_DIAG").as_deref() == Ok("1") {
            let pred = fwd_ffn(&mut gpu)?;
            let e = ffn_target_h.iter().map(|v| v * v).sum::<f32>() / ffn_target_h.len() as f32;
            let cos = {
                let dot: f32 = pred.iter().zip(&ffn_target_h).map(|(a, b)| a * b).sum();
                let na = pred.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb = ffn_target_h.iter().map(|x| x * x).sum::<f32>().sqrt();
                dot / (na * nb).max(1e-12)
            };
            println!(
                "  DIAG L{i:2}: rel={:.4} cos={:.4}  rows={rows}",
                mse(&pred, &ffn_target_h) / e.max(1e-12),
                cos
            );
        }
        let start_mse = mse(&fwd_ffn(&mut gpu)?, &ffn_target_h);

        let mut last_mse = start_mse;
        let n = (rows * dim) as f32;
        for _ in 0..steps {
            let pred_h = fwd_ffn(&mut gpu)?;
            last_mse = mse(&pred_h, &ffn_target_h);
            // d_mlp = 2/N·(pred_ffn − ffn_target)
            let d_mlp_h: Vec<f32> = pred_h
                .iter()
                .zip(&ffn_target_h)
                .map(|(p, t)| 2.0 * (p - t) / n)
                .collect();
            let d_mlp = gpu.upload_f32(&d_mlp_h, &[rows * dim])?;
            // backward MLP
            linear_backward_x(&mut gpu, &d_mlp, &w_down, &d_act, rows, inter, dim, false)?;
            swiglu_backward(&mut gpu, &d_act, &g, &u, &d_g, &d_u, rows * inter)?;
            linear_backward_x(&mut gpu, &d_g, &w_gate, &d_yn, rows, dim, inter, false)?;
            linear_backward_x(&mut gpu, &d_u, &w_up, &d_yn, rows, dim, inter, true)?;
            // rmsnorm backward → d_gamma (fresh zeros; bwd atomic-accumulates)
            let d_gamma = gpu.zeros(&[dim], DType::F32)?;
            rmsnorm_backward(
                &mut gpu,
                &d_yn,
                &x_in,
                &gamma,
                &rinv,
                &d_xmid_unused,
                &d_gamma,
                rows,
                dim,
            )?;
            opt.step(&mut gpu, &[&gamma], &[&d_gamma])?;
            gpu.free_tensor(d_gamma)?;
            gpu.free_tensor(d_mlp)?;
        }

        let tuned = gpu.download_f32(&gamma)?;
        tuned_norms.insert(norm_name.clone(), tuned);
        let rec = 100.0 * (start_mse - last_mse) / start_mse.max(1e-12);
        // relative error vs the FFN-output energy (mse/energy): tells "wrong
        // reconstruction" (rel~1) from "right but large-magnitude" (rel<<1).
        let energy = ffn_target_h.iter().map(|v| v * v).sum::<f32>() / ffn_target_h.len() as f32;
        let rel0 = start_mse / energy.max(1e-12);
        let rel1 = last_mse / energy.max(1e-12);
        println!("L{i:2}: FFN MSE {start_mse:.3e}→{last_mse:.3e} ({rec:5.1}%)  rel {rel0:.3e}→{rel1:.3e}  rows={rows}");
        total_start += start_mse as f64;
        total_final += last_mse as f64;
        n_done += 1;

        for t in [
            w_gate,
            w_up,
            w_down,
            x_in,
            gamma,
            yn,
            rinv,
            g,
            u,
            act,
            mlp,
            d_act,
            d_g,
            d_u,
            d_yn,
            d_xmid_unused,
        ] {
            gpu.free_tensor(t)?;
        }
    }

    let rec = 100.0 * (total_start - total_final) / total_start.max(1e-12);
    println!("\n{n_done} MLP norms recovered: Σ MSE {total_start:.4e} → {total_final:.4e}  ({rec:.1}% block-local)");
    // Persist tuned norms for the Path-A export step.
    let out = format!("{cap_dir}/tuned_mlp_norms.json");
    std::fs::write(&out, serde_json::to_string(&tuned_norms)?)?;
    println!("wrote {} tuned norm tensors → {out}", tuned_norms.len());
    Ok(())
}
