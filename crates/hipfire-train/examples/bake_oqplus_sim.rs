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

//! Bake OQ+ sim-quant into a bf16 `.hfq` (in place, same bytes) so the daemon
//! runs the OQ+-damaged model AS bf16 — isolating the norm-recovery effect from
//! real-kernel numerics. Optionally patch the post_attention_layernorm weights
//! with recovered γ (Path-A) to measure end-to-end whether block-local norm
//! recovery reduces perplexity.
//!
//! The model uses the (1+γ) RMSNorm convention (loader/kernel adds 1 to the
//! stored weight), and the trainer's tuned γ already has +1 folded — so we store
//! (tuned − 1).
//!
//! Usage:
//!   bake_oqplus_sim <in.hfq> <out.hfq> [--layers mlp|all] [--norms tuned.json]
//! Default --layers mlp = quantize gate/up/down only (cleanest isolation of the
//! MLP-norm recovery).

use hipfire_train::hfq_patch::{bf16_bits_to_f32, f32_to_bf16_bits, parse_hfq};
use hipfire_train::oqplus_quant::oqplus_simquant;
use std::collections::HashMap;

const QT_BF16: u8 = 16;

fn is_quant_target(name: &str, all: bool) -> bool {
    let mlp = name.ends_with(".mlp.gate_proj.weight")
        || name.ends_with(".mlp.up_proj.weight")
        || name.ends_with(".mlp.down_proj.weight");
    let attn = name.ends_with(".self_attn.q_proj.weight")
        || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight")
        || name.ends_with(".self_attn.o_proj.weight");
    mlp || (all && attn)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: bake_oqplus_sim <in.hfq> <out.hfq> [--layers mlp|all] [--norms tuned.json]"
        );
        std::process::exit(1);
    }
    let (inp, outp) = (&args[1], &args[2]);
    let mut all = false;
    let mut norms_path: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--layers" => {
                all = args.get(i + 1).map(|s| s == "all").unwrap_or(false);
                i += 2;
            }
            "--norms" => {
                norms_path = args.get(i + 1).cloned();
                i += 2;
            }
            _ => i += 1,
        }
    }

    let mut bytes = std::fs::read(inp)?;
    let (entries, _meta) = parse_hfq(&bytes).map_err(|e| format!("parse_hfq: {e}"))?;

    // 1) OQ+ sim-quant the target linears (bf16 → f32 → oqplus round-trip → bf16).
    let mut nq = 0usize;
    for e in &entries {
        if e.quant_type != QT_BF16 || !is_quant_target(&e.name, all) {
            continue;
        }
        let n = e.data_size / 2;
        let mut wf = Vec::with_capacity(n);
        for j in 0..n {
            let off = e.data_offset + j * 2;
            wf.push(bf16_bits_to_f32(u16::from_le_bytes([
                bytes[off],
                bytes[off + 1],
            ])));
        }
        let q = oqplus_simquant(&wf);
        for (j, &v) in q.iter().enumerate() {
            let off = e.data_offset + j * 2;
            bytes[off..off + 2].copy_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
        }
        nq += 1;
    }
    println!(
        "OQ+ sim-quantized {nq} linears (layers={})",
        if all { "all" } else { "mlp" }
    );

    // 2) Optionally Path-A patch the post_attention_layernorm weights (store γ−1).
    if let Some(np) = norms_path {
        let tuned: HashMap<String, Vec<f32>> =
            serde_json::from_str(&std::fs::read_to_string(&np)?)?;
        let by_name: HashMap<&str, &_> = entries.iter().map(|e| (e.name.as_str(), e)).collect();
        let mut npatched = 0usize;
        for (name, vals) in &tuned {
            let Some(e) = by_name.get(name.as_str()) else {
                continue;
            };
            if e.quant_type != QT_BF16 || vals.len() * 2 != e.data_size {
                return Err(format!(
                    "{name}: norm shape/type mismatch (qt {}, len {})",
                    e.quant_type,
                    vals.len()
                )
                .into());
            }
            for (j, &v) in vals.iter().enumerate() {
                let off = e.data_offset + j * 2;
                // tuned γ has +1 folded (trainer); loader applies (1+stored) → store γ−1.
                bytes[off..off + 2].copy_from_slice(&f32_to_bf16_bits(v - 1.0).to_le_bytes());
            }
            npatched += 1;
        }
        println!("Path-A patched {npatched} post_attention_layernorm weights (stored γ−1)");
    }

    std::fs::write(outp, &bytes)?;
    println!("wrote {outp}");
    Ok(())
}
