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

`cargo build -p hipfire-arch-cohere2moe` compiles.

## FORWARD VALIDATED via tiny oracle (2026-06-07, hiptrx GPU 3 / gfx1201)

Tiny 2-layer random oracle (hidden 256, hd 128, 16 experts top-8, layer-0 dense)
vs HF transformers `Cohere2MoeForCausalLM` per-layer post-residual cosine:

```
            MQ4 experts        MQ6 experts
layer    mean_cos  rel_L2    mean_cos  rel_L2
  0      0.99588   0.088     0.99851   0.052
  1      0.99571   0.086     0.99795   0.054
```

Flat across layers, rms-matched, no crater/compounding → structure correct.
Error shrinks ∝ precision (MQ4→MQ6) → residual is quant noise on a tiny random
model, NOT an arch bug (methodology step 5). [T1] routing fix moved layer 1 from
0.984 → 0.9957 exactly as predicted. Conclusion: parallel block, interleaved
RoPE, dense layer-0, GQA, sigmoid no-renorm MoE, RMSNorm are all forward-correct.

Repro: HF ref needs transformers `main` (cohere2_moe is NOT in the 5.8 release;
the model's `transformers_version` tag is misleading and there is no `auto_map`).
On hiptrx: `python -m venv --system-site-packages ~/cohere-gen-venv &&
~/cohere-gen-venv/bin/pip install --no-deps -U git+https://github.com/huggingface/transformers.git`.
Then gen → `hipfire-quantize --input ~/cohere2moe-tiny --output …hfq --format mq4
--no-kmap` (experts→MQ4G256 FWHT, attn/router→Q8, embed→Q8, norms→F16) →
`HIP_VISIBLE_DEVICES=3 dump_cohere2moe_hidden_states` → `compare_hidden_states.py`.

## DONE: hipfire-quantize converter arm

`cohere2_moe` → arch_id 12 wired in both arch-detect paths + `is_moe`. Split
per-expert tensors (`mlp.experts.{e}.{gate,up,down}_proj`) flow through the
existing 2D quant path → FWHT-rotated MQ4/MQ6; router/attn/dense → Q8; embed →
Q8; norms → F16. No new quant logic.

## Still required (NOT yet wired)

1. **Daemon dispatch arm** for `arch_id == 12` (load + generate). See
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

- **[T1] Routing renormalization — RESOLVED 2026-06-07.** Now uses
  `moe_topk_renorm_k8(norm_topk = cfg.norm_topk_prob)` (false), giving
  un-renormalized sigmoid weights, no bias. Was the bias-aware renormalizing
  kernel; cost ~0.012 layer-1 cosine on the tiny oracle.
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
