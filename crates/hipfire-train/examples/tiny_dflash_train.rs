// SPDX-License-Identifier: Apache-2.0
//! Train a tiny DFlash draft bridge (`fc.weight`) on synthetic hidden states
//! and export an HF-style DFlash draft directory (`config.json` +
//! `model.safetensors`).
//!
//! This is the first useful DFlash trainer slice: it exercises hipfire-train's
//! fp32 GEMM backward + AdamW path and emits an artifact that the existing
//! `dflash_convert` tool can pack into a runtime `.dflash.hfq` sidecar.
//!
//! Usage:
//!   cargo run -p hipfire-train --example tiny_dflash_train -- \
//!     --out /tmp/tiny-dflash-trained --steps 80

use hipfire_train::ops::linear::{linear_backward_w, linear_forward};
use hipfire_train::optim::AdamW;
use rdna_compute::{DType, Gpu};
use serde_json::json;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
struct TinyDflashCfg {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    rows: usize,
}

impl TinyDflashCfg {
    fn new(rows: usize) -> Self {
        Self {
            hidden: 256,
            inter: 512,
            vocab: 4096,
            layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
            block_size: 16,
            rows,
        }
    }

    fn num_extract(self) -> usize {
        2
    }

    fn q_dim(self) -> usize {
        self.n_heads * self.head_dim
    }

    fn kv_dim(self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

#[derive(Clone)]
struct TensorOut {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
    dtype: StDtype,
}

#[derive(Clone, Copy)]
enum StDtype {
    Bf16,
    F16,
}

impl StDtype {
    fn name(self) -> &'static str {
        match self {
            StDtype::Bf16 => "BF16",
            StDtype::F16 => "F16",
        }
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn f32(&mut self, scale: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        (u * 2.0 - 1.0) * scale
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return 0x7FC0;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn tensor_bytes(t: &TensorOut) -> Vec<u8> {
    let mut out = Vec::with_capacity(t.data.len() * 2);
    for &v in &t.data {
        let bits = match t.dtype {
            StDtype::Bf16 => bf16_bits(v),
            StDtype::F16 => hipfire_runtime::quant::f32_to_f16(v),
        };
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn write_safetensors(path: &Path, tensors: &[TensorOut]) -> Result<(), String> {
    let mut datas = Vec::with_capacity(tensors.len());
    let mut header = BTreeMap::new();
    let mut offset = 0usize;
    for t in tensors {
        let bytes = tensor_bytes(t);
        let end = offset + bytes.len();
        header.insert(
            t.name.clone(),
            json!({
                "dtype": t.dtype.name(),
                "shape": t.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
        datas.push(bytes);
    }

    let header_json = serde_json::to_string(&header).map_err(|e| e.to_string())?;
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {path:?}: {e}"))?;
    f.write_all(&(header_json.len() as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    f.write_all(header_json.as_bytes())
        .map_err(|e| e.to_string())?;
    for d in datas {
        f.write_all(&d).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn uniform_tensor(
    rng: &mut SplitMix64,
    name: impl Into<String>,
    shape: Vec<usize>,
    scale: f32,
) -> TensorOut {
    let n = shape.iter().product();
    TensorOut {
        name: name.into(),
        shape,
        data: (0..n).map(|_| rng.f32(scale)).collect(),
        dtype: StDtype::Bf16,
    }
}

fn norm_tensor(name: impl Into<String>, n: usize) -> TensorOut {
    TensorOut {
        name: name.into(),
        shape: vec![n],
        data: vec![1.0; n],
        dtype: StDtype::F16,
    }
}

fn build_tensors(cfg: TinyDflashCfg, fc_weight: Vec<f32>) -> Vec<TensorOut> {
    let h = cfg.hidden;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let mut rng = SplitMix64(0xD1FA_5A11);
    let mut tensors = Vec::new();
    tensors.push(TensorOut {
        name: "fc.weight".to_string(),
        shape: vec![h, cfg.num_extract() * h],
        data: fc_weight,
        dtype: StDtype::Bf16,
    });
    tensors.push(norm_tensor("hidden_norm.weight", h));
    tensors.push(norm_tensor("norm.weight", h));
    for layer in 0..cfg.layers {
        let p = format!("layers.{layer}");
        let sa = format!("{p}.self_attn");
        tensors.push(norm_tensor(format!("{p}.input_layernorm.weight"), h));
        tensors.push(uniform_tensor(
            &mut rng,
            format!("{sa}.q_proj.weight"),
            vec![q_dim, h],
            0.03,
        ));
        tensors.push(uniform_tensor(
            &mut rng,
            format!("{sa}.k_proj.weight"),
            vec![kv_dim, h],
            0.03,
        ));
        tensors.push(uniform_tensor(
            &mut rng,
            format!("{sa}.v_proj.weight"),
            vec![kv_dim, h],
            0.03,
        ));
        tensors.push(uniform_tensor(
            &mut rng,
            format!("{sa}.o_proj.weight"),
            vec![h, q_dim],
            0.03,
        ));
        tensors.push(norm_tensor(format!("{sa}.q_norm.weight"), cfg.head_dim));
        tensors.push(norm_tensor(format!("{sa}.k_norm.weight"), cfg.head_dim));
        tensors.push(norm_tensor(
            format!("{p}.post_attention_layernorm.weight"),
            h,
        ));
        tensors.push(uniform_tensor(
            &mut rng,
            format!("{p}.mlp.gate_proj.weight"),
            vec![cfg.inter, h],
            0.03,
        ));
        tensors.push(uniform_tensor(
            &mut rng,
            format!("{p}.mlp.up_proj.weight"),
            vec![cfg.inter, h],
            0.03,
        ));
        tensors.push(uniform_tensor(
            &mut rng,
            format!("{p}.mlp.down_proj.weight"),
            vec![h, cfg.inter],
            0.03,
        ));
    }
    tensors
}

fn write_config(out: &Path, cfg: TinyDflashCfg) -> Result<(), String> {
    let config = json!({
        "architectures": ["DFlashDraftModel"],
        "model_type": "dflash",
        "num_hidden_layers": cfg.layers,
        "hidden_size": cfg.hidden,
        "intermediate_size": cfg.inter,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "block_size": cfg.block_size,
        "num_target_layers": 4,
        "dflash_config": {
            "mask_token_id": cfg.vocab - 1,
            "target_layer_ids": [0, 1],
        },
    });
    std::fs::write(
        out.join("config.json"),
        serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write config.json: {e}"))
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        / a.len() as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        eprintln!(
            "Usage: tiny_dflash_train --out <dir> [--steps N] [--rows N] [--lr LR] [--seed U64]"
        );
        return Ok(());
    }
    let out = PathBuf::from(
        arg_value(&args, "--out").unwrap_or_else(|| "target/tiny-dflash-trained".to_string()),
    );
    let steps: usize = arg_value(&args, "--steps")
        .as_deref()
        .unwrap_or("80")
        .parse()?;
    let rows: usize = arg_value(&args, "--rows")
        .as_deref()
        .unwrap_or("64")
        .parse()?;
    let lr: f32 = arg_value(&args, "--lr")
        .as_deref()
        .unwrap_or("0.03")
        .parse()?;
    let seed: u64 = arg_value(&args, "--seed")
        .as_deref()
        .unwrap_or("12648430")
        .parse()?;

    let cfg = TinyDflashCfg::new(rows);
    let m = cfg.rows;
    let k = cfg.num_extract() * cfg.hidden;
    let n = cfg.hidden;

    let mut rng = SplitMix64(seed);
    let x_h: Vec<f32> = (0..m * k).map(|_| rng.f32(0.5)).collect();
    let teacher_w: Vec<f32> = (0..n * k)
        .map(|_| rng.f32(1.0 / (k as f32).sqrt()))
        .collect();
    let mut target_h = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x_h[row * k + kk] * teacher_w[col * k + kk];
            }
            target_h[row * n + col] = acc;
        }
    }

    let mut gpu = Gpu::init()?;
    eprintln!("tiny_dflash_train: gpu {}", gpu.arch);
    let x = gpu.upload_f32(&x_h, &[m * k])?;
    let target = gpu.upload_f32(&target_h, &[m * n])?;
    let mut init_rng = SplitMix64(seed ^ 0xA55A_5AA5);
    let w0: Vec<f32> = (0..n * k)
        .map(|_| init_rng.f32(0.25 / (k as f32).sqrt()))
        .collect();
    let w = gpu.upload_f32(&w0, &[n * k])?;
    let y = gpu.zeros(&[m * n], DType::F32)?;
    let grad = gpu.zeros(&[n * k], DType::F32)?;
    let mut opt = AdamW::new(&mut gpu, &[n * k], lr, 0.9, 0.999, 1e-8, 0.0)?;

    let mut last_loss = f32::NAN;
    for step in 0..steps {
        linear_forward(&mut gpu, &x, &w, &y, m, k, n)?;
        let pred = gpu.download_f32(&y)?;
        last_loss = mse(&pred, &target_h);
        let scale = 2.0 / (m * n) as f32;
        let dy_h: Vec<f32> = pred
            .iter()
            .zip(&target_h)
            .map(|(p, t)| (p - t) * scale)
            .collect();
        let dy = gpu.upload_f32(&dy_h, &[m * n])?;
        linear_backward_w(&mut gpu, &dy, &x, &grad, m, k, n, false)?;
        opt.step(&mut gpu, &[&w], &[&grad])?;
        gpu.free_tensor(dy)?;
        if step == 0 || (step + 1) % 10 == 0 || step + 1 == steps {
            eprintln!("  step {:>3}/{steps}  mse={last_loss:.6e}", step + 1);
        }
    }

    linear_forward(&mut gpu, &x, &w, &y, m, k, n)?;
    let final_pred = gpu.download_f32(&y)?;
    let final_loss = mse(&final_pred, &target_h);
    let fc_weight = gpu.download_f32(&w)?;
    std::fs::create_dir_all(&out)?;
    write_config(&out, cfg)?;
    write_safetensors(
        &out.join("model.safetensors"),
        &build_tensors(cfg, fc_weight),
    )?;

    eprintln!(
        "tiny_dflash_train: wrote {} (start_mse={last_loss:.6e}, final_mse={final_loss:.6e})",
        out.display()
    );
    drop(target);
    Ok(())
}
