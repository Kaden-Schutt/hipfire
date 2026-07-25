// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// Emit the frozen all-layer DeepSeek-V4 MQ2R P1 REAP plan.
// Usage: deepseek4_e8_p1_plan <output.json>

use std::env;
use std::fs::File;
use std::io::{self, Write};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: {} <output.json>", args[0]);
        std::process::exit(2);
    }
    let mut output = File::create(&args[1])?;
    writeln!(
        output,
        "{{\n  \"model_arch\": \"deepseek4\",\n  \"num_layers\": 43,\n  \"original_experts\": 256,\n  \"quant_overrides\": ["
    )?;
    for layer in 0..43 {
        let separator = if layer == 0 { "" } else { "," };
        writeln!(
            output,
            r#"{separator}    {{
      "layer": {layer},
      "role": "attention",
      "tensors": [
        "layers.{layer}.attn.wq_a.weight",
        "layers.{layer}.attn.wq_b.weight",
        "layers.{layer}.attn.wkv.weight",
        "layers.{layer}.attn.wo_a.weight",
        "layers.{layer}.attn.wo_b.weight"
      ],
      "tier": "mfp4e8soa"
    }},
    {{
      "layer": {layer},
      "role": "shared_expert",
      "tensors": [
        "layers.{layer}.ffn.shared_experts.w1.weight",
        "layers.{layer}.ffn.shared_experts.w2.weight",
        "layers.{layer}.ffn.shared_experts.w3.weight"
      ],
      "tier": "mfp4e8soa"
    }}"#
        )?;
    }
    writeln!(output, "  ]\n}}")?;
    output.sync_all()
}
