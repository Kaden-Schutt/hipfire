<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# hipfire-arch-cohere2moe — bring-up status & next steps

Port target: **CohereLabs/BLS-Mini-Code-1.0** (`model_type = cohere2_moe`,
`Cohere2MoeForCausalLM`). ~30B total / ~3B active MoE code model.

Methodology: `docs/methodology/arch-port-validation.md` (tiny random-weight
oracle, per-layer cosine). Validation box: **hiptrx GPU 3** (gfx1201/RDNA4,
32 GB; pin `HIP_VISIBLE_DEVICES=3`). The tiny-oracle phase needs **no GPU**.

## What this scaffold contains (DONE)

- `config.rs` — `Cohere2MoeConfig` parse (flat or `config`-wrapped HFQ metadata).
- `cohere2moe.rs` — `CohereWeights` / `CohereState` + loader. Single
  `input_layernorm` per layer (parallel block), per-layer dense-vs-MoE FFN
  split, minimax-style packed experts + device-pointer table, zero routing bias.
- `forward.rs` — `decode_step` / `decode_step_capture` implementing the
  **parallel block** (`h += attn(n) + ffn(n)`, one shared norm `n`), interleaved
  full-dim RoPE, full attention (SWA deferred), dense SwiGLU layer-0, sigmoid
  top-8 MoE. **Zero new HIP kernels.**
- `arch.rs` — `Architecture` impl, `arch_id = 12`, `name = "cohere2moe"`.
- `examples/dump_cohere2moe_hidden_states.rs` — per-layer HFHS dumper.
- `scripts/gen_tiny_cohere2moe.py` — tiny reference oracle (HF transformers).
- Registered in workspace `Cargo.toml` + `docs/architecture-ids.md` (id 12).

`cargo build -p hipfire-arch-cohere2moe` compiles. The crate does not yet run
end-to-end because two pieces are unwired (below).

## Required to run the oracle loop (NOT yet done)

1. **hipfire-quantize converter arm** (safetensors → `.hfq`) for `cohere2_moe`.
   The tiny oracle emits `model.safetensors` (SPLIT expert layout) + flat
   `config.json`; the converter must map tensor names (the loader reads RAW HF
   names: `model.layers.{l}.input_layernorm.weight`,
   `self_attn.{q,k,v,o}_proj.weight`, dense `mlp.{gate,up,down}_proj.weight`,
   MoE `mlp.gate.weight` + `mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`,
   `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`), quantize
   experts to MQ4G256/HFQ4G256 (FWHT-rotated) and attn/router/dense to Q8, and
   emit `arch_id = 12` in the HFQ header.

2. **Daemon dispatch arm** for `arch_id == 12` (load + generate). See
   `docs/architecture-ids.md` "Daemon dispatch sites" — wire the load arm and a
   generate branch calling `forward::decode_step`. Not needed for the oracle
   compare (the dump example calls the forward directly), only for `serve`/CLI.

## Oracle loop (once the converter arm lands)

```sh
# 1. reference oracle (no GPU) — writes model.safetensors, config.json,
#    oracle_hidden.hfhs, tokens.hfkldr
python scripts/gen_tiny_cohere2moe.py --out /workspace/cohere2moe-tiny --n-ctx 16

# 2. quantize → .hfq (cohere2moe converter arm, step 1 above)
#    e.g. hipfire-quantize --in /workspace/cohere2moe-tiny --out /workspace/cohere2moe-tiny.hfq --recipe mq4

# 3. hipfire per-layer dump (hiptrx GPU 3)
HIP_VISIBLE_DEVICES=3 cargo run -p hipfire-arch-cohere2moe --release \
  --example dump_cohere2moe_hidden_states -- \
  --model /workspace/cohere2moe-tiny.hfq \
  --ref /workspace/cohere2moe-tiny/tokens.hfkldr \
  --out /workspace/cohere2moe-tiny/hipfire_hidden.hfhs

# 4. compare per-layer cosine
python scripts/compare_hidden_states.py \
  --hf /workspace/cohere2moe-tiny/oracle_hidden.hfhs \
  --hipfire /workspace/cohere2moe-tiny/hipfire_hidden.hfhs
```

Target: per-layer `mean_cos ≥ 0.999` (Q8-grade) / `≥ 0.99` (4-bit experts).

## Known oracle-loop targets (verify in this order)

- **[T1] Routing renormalization.** `forward.rs` reuses the bias-aware top-k
  kernel, which renormalizes the top-k weights to sum 1. Cohere2Moe has
  `norm_topk_prob = false` → it must NOT renormalize (weight = `sigmoid(logit)`
  of each selected expert, un-normalized). If the MoE layer's per-layer cosine
  is the first to crater, this is why. Fix = a no-renorm top-k variant.
- **[T3] RoPE layout.** If attention (not MoE) diverges, the Q/K projection may
  need a load-time interleave permute to match `rope_partial_interleaved_f32`.
  Localize with `HIPFIRE_COHERE_CAPTURE_POSTATTN=1` (the dump captures the
  post-attention residual instead of post-layer).
- **Parallel block.** If EVERY layer is uniformly slightly off, re-check that
  the FFN reads the shared `n` (state.tmp), not a re-normalized post-attention
  hidden state, and that both sub-blocks accumulate into `h`.

## Deferred (after forward-correctness)

- **[T2] Sliding-window attention** (1:3 full:sliding cadence, window 4096).
  Full attention is correct for prompts < 4096; add windowed KV for long
  context (deepseek4 has reusable SWA ring-cache machinery).
- **Tokenizer / chat template** (Cohere `<|START_OF_TURN_TOKEN|>` framing) for
  `serve`.
- **Real-weight quantize + coherence gate** on hiptrx; **batched/prefix
  forward** for fast TTFT (mirror minimax `forward_batch`).
- **logit_scale** ≠ 1 path (no-op for this checkpoint).
