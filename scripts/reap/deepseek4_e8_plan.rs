// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// Emit a frozen cumulative DeepSeek-V4 MQ2R P1/P2/P3 REAP plan, or the
// isolated P3 router bucket used for attribution.
// Usage: deepseek4_e8_plan <p1|p2|p3|router> <output.json>

use std::env;
use std::fs::File;
use std::io::{self, Write};

fn emit_entry(
    output: &mut File,
    first: &mut bool,
    layer: usize,
    role: &str,
    tensors: &[String],
) -> io::Result<()> {
    if !*first {
        writeln!(output, ",")?;
    }
    *first = false;
    writeln!(
        output,
        "    {{\n      \"layer\": {layer},\n      \"role\": \"{role}\",\n      \"tensors\": ["
    )?;
    for (index, tensor) in tensors.iter().enumerate() {
        let suffix = if index + 1 == tensors.len() { "" } else { "," };
        writeln!(output, "        \"{tensor}\"{suffix}")?;
    }
    write!(output, "      ],\n      \"tier\": \"mfp4e8soa\"\n    }}")
}

fn names(layer: usize, suffixes: &[&str]) -> Vec<String> {
    suffixes
        .iter()
        .map(|suffix| format!("layers.{layer}.{suffix}"))
        .collect()
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 || !matches!(args[1].as_str(), "p1" | "p2" | "p3" | "router") {
        eprintln!("usage: {} <p1|p2|p3|router> <output.json>", args[0]);
        std::process::exit(2);
    }
    let phase = args[1].as_str();
    let mut output = File::create(&args[2])?;
    writeln!(
        output,
        "{{\n  \"model_arch\": \"deepseek4\",\n  \"num_layers\": 43,\n  \"original_experts\": 256,\n  \"quant_overrides\": ["
    )?;
    let mut first = true;

    if phase != "router" {
        for layer in 0..43 {
            emit_entry(
                &mut output,
                &mut first,
                layer,
                "attention",
                &names(
                    layer,
                    &[
                        "attn.wq_a.weight",
                        "attn.wq_b.weight",
                        "attn.wkv.weight",
                        "attn.wo_a.weight",
                        "attn.wo_b.weight",
                    ],
                ),
            )?;
            emit_entry(
                &mut output,
                &mut first,
                layer,
                "shared_expert",
                &names(
                    layer,
                    &[
                        "ffn.shared_experts.w1.weight",
                        "ffn.shared_experts.w2.weight",
                        "ffn.shared_experts.w3.weight",
                    ],
                ),
            )?;
        }
    }

    if matches!(phase, "p2" | "p3") {
        for layer in 2..43 {
            let mut tensors = names(
                layer,
                &["attn.compressor.wkv.weight", "attn.compressor.wgate.weight"],
            );
            if layer % 2 == 0 {
                tensors.extend(names(
                    layer,
                    &[
                        "attn.indexer.wq_b.weight",
                        "attn.indexer.weights_proj.weight",
                        "attn.indexer.compressor.wkv.weight",
                        "attn.indexer.compressor.wgate.weight",
                    ],
                ));
            }
            emit_entry(&mut output, &mut first, layer, "attention", &tensors)?;
        }
    }

    if matches!(phase, "p3" | "router") {
        for layer in 0..43 {
            emit_entry(
                &mut output,
                &mut first,
                layer,
                "router",
                &names(layer, &["ffn.gate.weight"]),
            )?;
        }
    }
    if phase == "p3" {
        emit_entry(
            &mut output,
            &mut first,
            0,
            "lm_head",
            &["head.weight".to_string()],
        )?;
    }

    writeln!(output, "\n  ]\n}}")?;
    output.sync_all()
}
