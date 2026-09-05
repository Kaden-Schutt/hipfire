# Models

**Owner:** registry-backed model surface (`docs/INDEX.md`).
**Machine sources:** curated `registry/models.json`; generated and bundled
`registry/v1.json` (loaded by `hipfire-registry`).
**Last checked:** 2026-08-24 against the published Qwen3.8 MQ V2 ladder.

This page projects **registry availability**: tags, default artifact filenames, declared download size, and declared VRAM floor. It is **not** a product admission table and **not** a guarantee that every GPU/route runs every tag.

| Concept | Meaning |
|---|---|
| Registry tag | Pull/list name resolved through the bundled v1 registry (+ aliases). |
| Default artifact | `file` field — what `hipfire pull <tag>` fetches into `~/.hipfire/models/`. |
| Runtime support | Whether the daemon/loader/arch crate can load and run the artifact shape (`arch_id`, kernels, Cargo features). Source-of-truth: runtime crates + [`architecture-ids.md`](architecture-ids.md). |
| Admission | Explicit product decision in [`admissions.yml`](admissions.yml). Schema v2 holds exactly one evidence-bound record; no inferred admissions beyond that row. |

`hipfire list -r` prints the live registry plus local availability. Prefer that command when sizes change; this page is a checked narrative, not a second registry.

---

## Pull and run

```bash
hipfire pull qwen3.5:9b
hipfire run qwen3.5:9b "hello"
hipfire list -r
```

Default serve pre-warm tag is `qwen3.5:9b` (`CONFIG.md` → `default_model`). Per-tag sampling defaults come from registry `recommended_settings` only (applied by the native CLI request resolver). Cards may set intentional `reasoning_effort` (semantic prompt strength only — never a budget). Effort-native families (Qwen3.8, Ornith 1.5, DeepSeek V4, Muse Glimmer) omit `thinking_budget`; absence means no implicit cap. Qwen3.8 and Ornith 1.5 accept hipfire's explicit integer Qwen continuation cap. DeepSeek V4 and Muse Glimmer cap fields are dropped+warned rather than translated into hand-written tokenizer closure. Registry `sampling` blocks are legacy metadata and are not promoted by the native request resolver. Full contract: [`CONFIG.md`](CONFIG.md); HTTP examples: [`SERVE.md`](SERVE.md).

---

## Registry tags (from `registry/models.json`)

Fields: **Tag**, **File** (`file`), **Size GB** (`size_gb`), **Min VRAM GB** (`min_vram_gb`), **Default KV** (`default_kv_mode` when set; else empty — global `kv_cache=auto` resolves to `q8`), **Notes** (`desc`, truncated).

### Qwen 3.5 dense / hybrid

| Tag | File | Size GB | Min VRAM | Default KV | Notes |
|---|---|---:|---:|---|---|
| `qwen3.5:0.8b` | `qwen3.5-0.8b.mq4` | 0.55 | 2.0 | q8 | MQ4 default small |
| `qwen3.5:0.8b-mq6` | `qwen3.5-0.8b.mq6` | 0.67 | 2.2 | q8 | MQ6 |
| `qwen3.5:2b` | `qwen3.5-2b.mq4` | 1.29 | 2.8 | | MQ4 (registry desc still mentions legacy HF4 naming) |
| `qwen3.5:2b-hf6` | `qwen3.5-2b.hf6` | 1.6 | 3 | | HF6 |
| `qwen3.5:2b-mq3` | `qwen3.5-2b.mq3` | 1.16 | 2.7 | | MQ3 |
| `qwen3.5:2b-mq6` | `qwen3.5-2b.mq6` | 1.63 | 3.1 | | MQ6 |
| `qwen3.5:4b` | `qwen3.5-4b.mq4` | 2.59 | 4.1 | q8 | MQ4 |
| `qwen3.5:4b-mq3` | `qwen3.5-4b.mq3` | 2.25 | 3.8 | q8 | MQ3 |
| `qwen3.5:4b-mq6` | `qwen3.5-4b.mq6` | 3.48 | 5.0 | q8 | MQ6 |
| `qwen3.5:9b` | `qwen3.5-9b.mq4` | 5.31 | 6.8 | q8 | MQ4; common default |
| `qwen3.5:9b-mq3` | `qwen3.5-9b.mq3` | 4.57 | 6.1 | q8 | MQ3 alpha (gfx11/gfx12 noted in desc) |
| `qwen3.5:9b-mq6` | `qwen3.5-9b.mq6` | 7.3 | 8.8 | q8 | MQ6 |
| `qwen3.5:27b` | `qwen3.5-27b.mq4` | 15.0 | 16 | q8 | MQ4 |
| `qwen3.5:27b-mq3` | `qwen3.5-27b.mq3` | 10.7 | 12 | | MQ3 alpha |
| `qwen3.5:27b-mq6` | `qwen3.5-27b.mq6` | 21.4 | 24 | | MQ6 |

### Qwen 3.5 / 3.6 MoE (A3B)

Sizes below are **registry declarations**, not a substitute for runtime MoE layout checks.

| Tag | File | Size GB | Min VRAM | Default KV | Notes |
|---|---|---:|---:|---|---|
| `qwen3.5:35b-a3b` | `qwen3.5-35b-a3b.mq4` | 19.7 | 22 | q8 | 35B / 3B-active |
| `qwen3.6:35b-a3b` | `qwen3.6-35b-a3b.mq4p` | 19.8 | 22 | q8 | Default graded mq4p SKU |
| `qwen3.6:35b-a3b-mq2` | `qwen3.6-35b-a3b.mq2` | 11.6 | 14 | | Floor SKU |
| `qwen3.6:35b-a3b-escha-xt` | `qwen3.6-35b-a3b.escha-xt` | 11.4 | 14 | q8 | Escha-W2 2-bit trellis, MQ4V2 dense. Fastest/smallest. PPL 8.0643 |
| `qwen3.6:35b-a3b-escha` | `qwen3.6-35b-a3b.escha` | 11.8 | 15 | q8 | **Default.** Escha-W2 2-bit trellis experts stored verbatim and decoded in the GEMV — no decode-at-load. MQ6 dense. PPL 7.6940, 725 tok/s prefill, 55 decode |
| `qwen3.6:35b-a3b-escha-pro` | `qwen3.6-35b-a3b.escha-pro` | 12.3 | 16 | q8 | Q8_0 dense, most faithful. PPL 7.6864. KLD reference for the other two |
| `qwen3.8:27b-escha-xt` | `qwen3.8-27b.escha-xt` | 10.5 | 14 | q8 | Escha-W2 2-bit trellis, MQ4V2 dense. PPL 9.7242, 122 tok/s prefill, 12.3 decode |
| `qwen3.8:27b-escha` | `qwen3.8-27b.escha` | 10.8 | 15 | q8 | **Default.** Dense arch-5 sibling of the 35B. MQ6 dense. PPL 9.6753, 119 tok/s prefill, 12.1 decode. Beats `qwen3.8-27b.mq3` on quality while smaller |
| `qwen3.8:27b-escha-pro` | `qwen3.8-27b.escha-pro` | 11.2 | 15 | q8 | Q8_0 dense, most faithful. PPL 9.6486 |
| `qwen3.6:35b-a3b-mq3p` | `qwen3.6-35b-a3b.mq3p` | 17.2 | 20 | | MQ3+P graded |
| `qwen3.6:35b-a3b-mq4p` | `qwen3.6-35b-a3b.mq4p` | 19.8 | 22 | | MQ4+P graded |
| `qwen3.6:35b-a3b-mfp4` | `qwen3.6-35b-a3b.mfp4` | 20.2 | 22 | | MFP4-E8 |
| `qwen3.6:35b-a3b-mq4r` | `qwen3.6-35b-a3b.mq4r` | 18.7 | 22 | | Uniform MQ4G256V1/qt13 Redline speed SKU; zero qt44/graded experts. Dated tok/s in registry text is not a live baseline. |
| `qwen3.6:35b-a3b-mq5` | `qwen3.6-35b-a3b.mq5` | 23.7 | 26 | | Quality SKU |
| `qwen3.6:35b-a3b-mq6` | `qwen3.6-35b-a3b.mq6` | 27.7 | 30 | | Max quality |

Several A3B entries carry an `mtp.file` sidecar name (`qwen3.6-35b-a3b.mtp`). MTP enablement is config/runtime gated (`mtp_mode`, env); registry presence alone is not admission.

### Qwen 3.6 dense

| Tag | File | Size GB | Min VRAM | Default KV | Notes |
|---|---|---:|---:|---|---|
| `qwen3.6:27b` | `qwen3.6-27b.mq4` | 15.0 | 16 | q8 | Ships `triattn.file` in registry; template not effort-native — `reasoning_effort` dropped+warned, never converted to a cap |
| `qwen3.6:27b-mq3` | `qwen3.6-27b.mq3` | 10.7 | 12 | | MQ3 alpha |

### Qwen 3.8 dense

| Tag | File | Size GB | Min VRAM | Default KV | Notes |
|---|---|---:|---:|---|---|
| `qwen3.8:27b-mq3-xt` | `qwen3.8-27b.mq3-xt` | 11.78 | 13 | q8 | MQ3V2 XT |
| `qwen3.8:27b-mq3` | `qwen3.8-27b.mq3` | 12.62 | 14 | q8 | MQ3V2 base |
| `qwen3.8:27b-mq3-pro` | `qwen3.8-27b.mq3-pro` | 13.18 | 15 | q8 | MQ3V2 Pro |
| `qwen3.8:27b-mq4-xt` | `qwen3.8-27b.mq4-xt` | 14.98 | 16 | q8 | MQ4V2 XT (speed; supersedes legacy `.mq4r`) |
| `qwen3.8:27b` | `qwen3.8-27b.mq4` | 15.66 | 17 | q8 | MQ4V2 base; default; effort-native (`low`/`medium`/`xhigh`, default `xhigh`); uncapped think span unless explicit integer cap |
| `qwen3.8:27b-mq4-pro` | `qwen3.8-27b.mq4-pro` | 16.46 | 18 | q8 | MQ4V2 Pro |
| `qwen3.8:27b-mq5-xt` | `qwen3.8-27b.mq5-xt` | 18.18 | 20 | q8 | MQ5V2 XT |
| `qwen3.8:27b-mq5` | `qwen3.8-27b.mq5` | 18.71 | 20 | q8 | MQ5V2 base |
| `qwen3.8:27b-mq5-pro` | `qwen3.8-27b.mq5-pro` | 19.32 | 21 | q8 | MQ5V2 Pro |
| `qwen3.8:27b-mq6-xt` | `qwen3.8-27b.mq6-xt` | 21.39 | 23 | q8 | MQ6V2 XT |
| `qwen3.8:27b-mq6` | `qwen3.8-27b.mq6` | 21.75 | 23 | q8 | MQ6V2 base |
| `qwen3.8:27b-mq6-pro` | `qwen3.8-27b.mq6-pro` | 22.17 | 24 | q8 | MQ6V2 Pro |

MQ2V2 is not registered. Explicit `qwen3.8:27b-mq4` aliases to `qwen3.8:27b`. Legacy `qwen3.8:27b-fast` / `qwen3.8:fast` alias to `qwen3.8:27b-mq4-xt`. Reasoning contract: [`CONFIG.md`](CONFIG.md) (effort is semantic only; named `thinking_budget` dropped on this family).

### DFlash draft artifacts (registry)

| Tag | File | Size GB | Min VRAM | Pairs with (by name) |
|---|---|---:|---:|---|
| `qwen3.5:9b-draft` | `qwen35-9b-dflash-mq4.hfq` | 0.55 | 6 | `qwen3.5:9b` |
| `qwen3.5:27b-draft` | `qwen35-27b-dflash-mq4.hfq` | 0.92 | 16 | `qwen3.5:27b` |
| `qwen3.5:27b-draft-mq3` | `qwen35-27b-dflash-mq3.hfq` | 0.67 | 12 | `qwen3.5:27b` (mq3 draft) |
| `qwen3.6:27b-draft` | `qwen36-27b-dflash-mq4.hfq` | 0.92 | 16 | `qwen3.6:27b` |
| `qwen3.6:27b-draft-mq3` | `qwen36-27b-dflash-mq3.hfq` | 0.67 | 12 | `qwen3.6:27b` |
| `qwen3.8:27b-draft-mq3` | `qwen38-27b-dflash-mq3.hfq` | 0.98 | 16 | `qwen3.8:27b*` (same-bit alt) |
| `qwen3.8:27b-draft-mq4` | `qwen38-27b-dflash-mq4.hfq` | 1.21 | 16 | `qwen3.8:27b*` (recommended controller) |
| `qwen3.8:27b-draft-mq5` | `qwen38-27b-dflash-mq5.hfq` | 1.43 | 16 | `qwen3.8:27b*` (same-bit alt) |
| `qwen3.8:27b-draft-mq6` | `qwen38-27b-dflash-mq6.hfq` | 1.66 | 16 | `qwen3.8:27b*` (same-bit alt) |
| `muse-glimmer:draft` | `muse-glimmer-30b-dflash.mq4` | 1.36 | 26 | `muse-glimmer` / `muse-glimmer:fast` |

Draft **loading** is controlled by `dflash_mode` / `speculation` / `HIPFIRE_DFLASH_DRAFT` ([`CONFIG.md`](CONFIG.md), [`env-vars.md`](env-vars.md)). Default `dflash_mode` is **off**. Filename auto-match may wire a sibling draft when present; that is discovery, not an admission that DFlash wins on every prompt.

### Qwen3 (non-3.5) dense HF4

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `qwen3:0.6b` | `qwen3-0.6b.hf4` | 0.4 | 1 | standard attention |
| `qwen3:8b` | `qwen3-8b.hf4` | 4.1 | 6 | standard attention |

### Fine-tunes on Qwen 3.5 / 3.6 families

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `carnice:9b` | `carnice-9b.mq4` | 5.0 | 6 | Hermes tool-use; `default_tool_format=hermes` |
| `carnice:9b-mq6` | `carnice-9b.mq6` | 7.3 | 8 | Hermes MQ6 |
| `carnice:27b` | `carnice-27b.mq4` | 15.0 | 16 | Hermes 27B |
| `carnice:27b-mq6` | `carnice-27b.mq6` | 21.4 | 24 | Hermes 27B MQ6 |
| `qwopus:4b` | `qwopus-4b.mq4` | 2.6 | 4 | Qwopus3.5 v3 |
| `qwopus:4b-mq6` | `qwopus-4b.mq6` | 3.8 | 5 | |
| `qwopus:9b` | `qwopus-9b.mq4` | 5.3 | 6 | |
| `qwopus:9b-mq6` | `qwopus-9b.mq6` | 7.3 | 8 | |
| `qwopus:27b` | `qwopus-27b.mq4` | 15.0 | 16 | |
| `qwopus:27b-mq6` | `qwopus-27b.mq6` | 21.4 | 24 | |
| `qwopus3.6:27b-coder` | `qwopus3.6-27b-coder.mq4` | 15.0 | 16 | q8 default KV; agentic coder finetune |
| `nex-n2:mini` | `nex-n2-mini.mq4p` | 19.82 | 22 | q8 default KV; Qwen3.5-35B-A3B agentic MoE finetune |
| `ornith-1.5:35b-a3b` | `ornith-1.5-35b-a3b.mq4` | 19.02 | 22 | q8 default KV; MQ4G256V2 quality trunk with selective MQ6/Q8 protection; semantic `low`/`medium`/`xhigh` effort (default `xhigh`), uncapped unless an explicit integer cap is set |
| `ornith-1.5:35b-a3b-mq4r` | `ornith-1.5-35b-a3b.mq4r` | 18.70 | 22 | q8 default KV; uniform MQ4G256V2 Redline SKU, 20,871 qt44 and zero qt13/qt15; same effort contract as the quality trunk |

### Other families (registry)

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `deepseek-v4-flash` | `deepseek-v4-flash-0731.mq2lloyd` | 86.2 | 96 | current 0731 release; DSpark sidecar; default low `reasoning_effort`, uncapped think span (no named budget) |
| `deepseek-v4-flash:mq2r` | `deepseek-v4-flash-0731.mq2r` | 82 | 96 | current 0731 golden MQ2R; MQ2-Lloyd routed experts + MFP4-E8 dense route; matching `.mq2r` DSpark sidecar |
| `deepseek-v4-flash-preview` | `deepseek-v4-flash.mq2lloyd` | 82 | 96 | prior nwoolmer preview package, retained under an explicit preview identity |
| `minimax-m2.7` | `MiniMax-M2.7.mq2` | 79.2 | 96 | arch_id=10 Mixtral-style MoE |
| `north-mini-code` | `north-mini-code.mq4.hfq` | 16 | 24 | Cohere2-MoE arch_id=12; registry `sampling` block is **inert metadata** today |
| `vibethinker:3b` | `vibethinker-3b.mq4.hfq` | 1.82 | 3.5 | Qwen2 MQ4 |
| `vibethinker:3b-mq6` | `vibethinker-3b.mq6.hfq` | 2.51 | 5.0 | Qwen2 MQ6 |
| `muse-glimmer` | `muse-glimmer-30b.mq4` | 18.61 | 26 | 30B dense + perception encoder; MQ4 quality trunk; always-on Onyx reasoning with strength dial; default uncapped think span |
| `muse-glimmer:fast` | `muse-glimmer-30b.mq4r` | 16.26 | 24 | MQ4R speed SKU (MQ4 body and attention, Q8 lm_head); same Onyx reasoning contract as `muse-glimmer` |

### LFM2.5 (registry)

| Tag | File | Size GB | Min VRAM | Notes (registry `desc`) |
|---|---|---:|---:|---|
| `lfm2.5:350m` | `lfm2.5-350m.q8` | 0.38 | 1.9 | 350M dense; default artifact is **Q8** file |
| `lfm2.5:1.2b` | `lfm2.5-1.2b.mq4` | 0.7 | 2.2 | 1.2B Instruct dense |
| `lfm2.5:1.2b-thinking` | `lfm2.5-1.2b-thinking.mq4` | 0.7 | 2.2 | 1.2B Thinking dense |
| `lfm2.5:8b-a1b` | `lfm2.5-8b-a1b.mq4` | 4.66 | 6.2 | 8B-A1B MoE |

Registry `recommended_settings` for LFM tags is low temperature (0.05–0.2) with `repeat_penalty` 1.05 — applied by the CLI resolver. Do not treat a registry `sampling` field as active defaults.

### Gemma4 (registry — no published artifact yet)

> **No hipfire-quantized Gemma4 `.hfq`/`.mq4` artifact has been published.** The
> rows below are the *intended* tag layout for when the quantize + upload lane
> publishes; the current `registry/models.json` intentionally lists **zero**
> `gemma4:` tags so that no registry row can point at a 404. See the report at
> the bottom of this section for the exact repos/files to publish.

**Architecture ground truth** (`crates/hipfire-arch-gemma4/src/config.rs`,
`crates/hipfire-arch-gemma4/src/lowered.rs`, and
`crates/hipfire-arch-gemma4/src/gemma4.rs`):

- **Hybrid 5:1 sliding:global** — every 6th layer is global (full) attention.
  The per-layer `layer_types` array is authoritative; on the 12B dense text
  config (`hidden_size=3840`, `num_hidden_layers=48`, `vocab_size=262144`) that
  is 40 sliding + 8 full. Do not assume the period in code.
- **Sliding-window layers:** `window = 1024` (`sliding_window`), `head_dim = 256`
  (`sliding_head_dim` / `head_dim`), `RoPE θ = 10_000` (`sliding_rope_theta`),
  `RopeType::Default` on full head_dim, `attention scale = 1.0` (kernels bake
  `1/√d` so decode pre-scales Q by `√head_dim`), Q/K per-head RMSNorm.
- **Global (full) layers:** `head_dim = 512` (`global_head_dim` / `full_head_dim`),
  `window = 0` (full causal), `RoPE θ = 1_000_000` (`full_rope_theta`),
  `RopeType::Proportional` with `partial_rotary_factor = 0.25` (only first 64 of
  512 dims rotate; remainder NoPE), `attention_k_eq_v = true` — **V shares the
  pre-`k_norm` output of `k_proj`** (no `v_proj` on those layers; `v_norm` is
  weight-less, implemented with a ones-filled scratch buffer).
- **KV head counts (variant-dependent):** 12B text — `num_attention_heads = 16`,
  `num_key_value_heads = 8` (sliding) and `num_global_key_value_heads` defaults
  to sliding when absent; 31B target (per `lowered.rs` comments) — `n_heads = 32`,
  `sliding_n_kv_heads = 16`, `full_n_kv_heads = 4`, `sliding_head_dim = 256`,
  `full_head_dim = 512`, `hidden_dim = 21504`. The 12B `hidden_dim` is
  `intermediate_size = 15360`, `dim = 3840`.
- **FFN:** SwiGLU with `gelu_pytorch_tanh` activation, `intermediate_size`
  (above) per-layer.
- **Norms:** Sandwich RMSNorm — `input_layernorm`, `post_attention_layernorm`,
  `pre_feedforward_layernorm`, `post_feedforward_layernorm` per layer plus a
  learned scalar `layer_scalar [1]` at layer end. Gemma4 uses plain `x * w`
  (weights init 1.0); the `HIPFIRE_GEMMA4_NORM_PLUS_ONE=1` toggle bakes the
  Gemma-2/3 `x * (1+w)` form at load time.
- **Embeddings:** `embed_scale = sqrt(hidden_size)` multiplied on every lookup;
  **`lm_head` is TIED to `embed_tokens`** (single GPU allocation, aliased
  `WeightTensor`; `free` skips the head to avoid double-free).
- **Output:** `final_logit_softcapping = 30.0` — `tanh(logits/30)*30` before
  sampling. `norm_eps = 1e-6`, `max_position_embeddings = 262144` (12B).
- **Stop / EOS:** HF `eos_token_id` is the **list** `[1, 106]` (`<eos>` = 1 and
  `<end_of_turn>` / `<turn|>` = 106); parsing it as a scalar drops 106 and lets
  decode loop on `<turn|>` forever. The loader resolves `eos_tok` against
  `["<end_of_turn>", …]` and masks both.
- **Sampling defaults (Gemma card):** `temperature 1.0`, `top_p 0.95`,
  `top_k 64` — the registry will carry these in `recommended_settings` /
  `sampling_profiles` when tags land (today no Gemma rows exist to carry them).
- **Thinking:** boolean only (official default **off**). No native
  `reasoning_effort`; unsupported effort/named-budget values are dropped with a
  warning. See [`CONFIG.md`](CONFIG.md) / [`SERVE.md`](SERVE.md).
- **Crate:** `hipfire-arch-gemma4` (`Gemma4Config`, `Gemma4Weights`,
  `Gemma4State`), `Gemma4Bundle { config, weights, state, eos_tok }` in
  `hipfire-loader`. `Gemma4Carrier` claims arch ids **13** (`gemma4_text`) and
  **22** (`gemma4_unified_assistant` EAGLE drafter) — see
  [`architecture-ids.md`](architecture-ids.md).

| Intended tag | Intended file | Repo (to publish) | Notes |
|---|---|---|---|
| `gemma4:12b` | `gemma4-12b.mq4` | `schuttdev/hipfire-gemma4-12b` or `hipfire-models/hipfire-gemma4-12b` | 12B dense text, MQ4 default; `hipfire quantize google/gemma-4-12B-it --arch-id 13 --format mq4` |
| `gemma4:12b-mq4` | `gemma4-12b.mq4` | same as above | alias-style explicit quant suffix |
| `gemma4:12b-hfq` | `gemma4-12b.hfq` | same repo | HFQ4 variant if published |
| `gemma4:12b-draft` | `gemma4-12b-draft.mq4.hfq` | same repo or `…-drafter` side-repo | EAGLE draft (arch_id 22) paired with `gemma4:12b` |

No Gemma4 rows are registered until the files above (or the 27B/31B MoE
variants) are actually on the Hub and pass `scripts/registry_gen.py` LFS
probing — that script is the admission gate and will fail-closed on a missing
file or mismatched `size_gb`. Publishing steps: `hipfire quantize` → upload to
the Hub → add a `models.json` entry mirroring a dense neighbour like
`qwen3.6:27b` (fields: `repo`, `file`, `size_gb`, `min_vram_gb`, `desc`,
`recommended_settings` with temp 1.0/top_p 0.95/top_k 64) → run
`scripts/registry_gen.py` to stamp `sha256`/`size_bytes`/`arch_id`/`quant`.

---

## Aliases

String redirects in `registry/models.json` → `aliases` (not separate
downloads). **Partial table** — for the complete surface read that file or run
`hipfire list -r`.

| Alias | Resolves to |
|---|---|
| `qwen3.5` | `qwen3.5:4b` |
| `qwen3.5:latest` | `qwen3.5:9b` |
| `qwen3.5:small` | `qwen3.5:0.8b` |
| `qwen3.5:large` | `qwen3.5:27b` |
| `qwen3.6` / `qwen3.6:a3b` | `qwen3.6:35b-a3b` |
| `ornith` / `ornith-1.5` / `ornith1.5` / `ornith1.5:35b-a3b` | `ornith-1.5:35b-a3b` |
| `qwen3.8` / `qwen3.8:latest` | `qwen3.8:27b` |
| `qwen3.8:fast` / `qwen3.8:27b-fast` | `qwen3.8:27b-mq4-xt` |
| `qwen3.8:27b-mq4` | `qwen3.8:27b` |
| `qwen3.8:draft` / `qwen3.8:27b-draft` | `qwen3.8:27b-draft-mq4` |
| `muse-glimmer:latest` / `muse-glimmer:quality` / `muse-glimmer:30b` | `muse-glimmer` |
| `qwen3` | `qwen3:8b` |
| `carnice` | `carnice:9b` |
| `qwopus` | `qwopus:9b` |
| `qwopus:{4b,9b,27b}-{mq4,hf4}` | matching primary `qwopus:{4b,9b,27b}` tag |
| `deepseek4` / `deepseek-v4` | `deepseek-v4-flash` |
| `deepseek4:mq2r` / `deepseek-v4:mq2r` | `deepseek-v4-flash:mq2r` |
| `deepseek4:preview` / `deepseek-v4:preview` | `deepseek-v4-flash-preview` |
| `vibethinker` | `vibethinker:3b` |
| `qwen3.5:*-mq4` / `*-hf4` / several `*-hf6` | same-size primary or mq6 tag (see registry) |
| `qwen3.5:9b:draft` etc. | matching `*-draft` tags |

---

## Runtime family map (source, not registry)

Runtime dispatch uses HFQ `arch_id` ([`architecture-ids.md`](architecture-ids.md)). Summary for operators:

| Family | arch_id | Crate | Registry examples |
|---|---:|---|---|
| LLaMA / Mistral / plain Qwen3 path | 0 / 1 | `hipfire-arch-llama` | `qwen3:8b`, many GGUF/HF4 dense |
| Qwen3.5 dense hybrid | 5 | `hipfire-arch-qwen35` | `qwen3.5:*`, `qwen3.6:27b`, carnice/qwopus dense |
| Qwen3.5 / 3.6 MoE A3B | 6 | `hipfire-arch-qwen35` | `*:35b-a3b*`, `ornith-1.5:35b-a3b`, `nex-n2:mini` |
| Qwen2 | 7 | `hipfire-arch-qwen2` | `vibethinker:3b`, `vibethinker:3b-mq6` (support, not admission) |
| DeepSeek V4 Flash | 9 | `hipfire-arch-deepseek4` | `deepseek-v4-flash` |
| MiniMax-M2 | 10 | `hipfire-arch-minimax` | `minimax-m2.7` |
| LFM2.5 dense **and** MoE | 11 | `hipfire-arch-lfm2moe` | all `lfm2.5:*` |
| Cohere2-MoE | 12 | `hipfire-arch-cohere2moe` | `north-mini-code` |
| Gemma4 text (dense) | 13 | `hipfire-arch-gemma4` | *(none yet — awaiting publish; intended `gemma4:12b`)* |
| Gemma4 unified-assistant (EAGLE drafter) | 22 | `hipfire-arch-gemma4` `drafter` | *(none yet; intended `gemma4:12b-draft` sidecar for 13)* |

**Dense LFM2.5 is supported on arch_id 11.** The LFM config parser treats `num_experts == 0` as dense SwiGLU on every layer (`crates/hipfire-arch-lfm2moe/src/config.rs`). Do **not** claim dense LFM is unsupported.

`hipfire-arch-lfm2moe` is a **non-optional** dependency of `hipfire-loader` / daemon load paths on this tree (see crate `Cargo.toml` graphs). Feature flags on `hipfire-runtime` default set do not list a separate `arch-lfm2moe` toggle the way some other arches do — loader always links the crate.

Capability features (DFlash, CASK, PP, MTP, batched prefill, n-gram) are **per-path and often narrower than “model loads”**. Spec inventory history: [`speculation-support-inventory.md`](speculation-support-inventory.md) (historical). Product claims need source + [`admissions.yml`](admissions.yml).

### LFM optimized prefill — branch-only scope

**Branch-only; not shipped** on `origin/beta@202282de8759dfa6963ea5184ad2bf2b9259cef6`.

Audited branch wording allowed for optimized LFM prefill (and nothing broader):

- Exact cohort: **350M dense MQ4** fixture path used by the branch **runtime fixture validation/guard** (`lfm2.5-350m.mq4` shape checks in `hipfire-arch-lfm2moe` forward), **not** a generic “all LFM” claim.
- GPU: **gfx1201** only for the batched opt-in path.
- Flag: explicit opt-in **`HIPFIRE_LFM2_PREFILL_BATCH=1`** (default off). Optional chunk override `HIPFIRE_LFM2_PREFILL_MAX_BATCH` (default 256, hard cap 512 in source).
- Pin when citing branch implementation: `lfm-redline@692a726dde53508cb53de1a74c720e75a7c9f33e` (or later branch commits only if re-grounded).

**Planned (not implemented claims here):** Q8-first generic completion of the optimized path, wider LFM cohorts (1.2B / 8B-A1B), multi-GPU, and Phase-4 default-on.
**Admitted (exact one row):** [`admissions.yml`](admissions.yml) schema v2 admits only the sealed gfx1201 LFM2.5-350M MQ4 retained-PM4 plain-AR product route; nothing else.
**Not a current baseline:** any exploratory tok/s tables in designs/plans.

Eager per-token prefill / decode remains the portable LFM path when the opt-in flag is off **or** the GPU is not gfx1201. On **gfx1201 with `HIPFIRE_LFM2_PREFILL_BATCH=1`**, the daemon selects the batched path from GPU+flag alone and has **no post-selection fallback**: requests outside the exact **350M dense MQ4** fixture fail closed at the runtime fixture guard. Source symbol `validate_350m_mq4_admission` names that fixture check only — it does **not** create a product admission; [`admissions.yml`](admissions.yml) remains the sole authority (schema v2, exactly one earned retained-PM4 product row for this sealed fixture).

---

## Bring your own

### HuggingFace → quantize → register

```bash
hipfire quantize Jackrong/Qwopus3.5-4B-v3 \
  --format mq4 --install --register qwopus:4b
```

See [`QUANTIZE.md`](QUANTIZE.md) and [`QUANTIZATION.md`](QUANTIZATION.md).

### Local safetensors directory

Requires `config.json` + `.safetensors`. Architectures the **engine** loads are those with arch crates / loaders above; the quantizer may accept more shapes than inference can run.

### GGUF

```bash
hipfire quantize ./model.Q4_K_M.gguf --install --register my:tag
```

Dequant path support is format-specific (common Q4_0 / Q8_0 / Q4_K / Q6_K / F16 / BF16 / F32). Unsupported GGUF quants fail closed in the quantizer.

---

## On-disk layout

```text
~/.hipfire/models/
  <registry file names>
  optional sibling drafts / .triattn*.bin sidecars
```

Extension hints (loader recognizes several): `.mq4`, `.mq6`, `.mq4p`, `.mq4r`, `.mq2`, `.mq2lloyd`, `.mq2r`, `.mfp4`, `.hf4`, `.hf6`, `.hfq`, `.q8`, and related graded names as produced by quant tooling. Exact dtype routing is loader/kernel source, not this table.

---

## Thinking / chat framing

Reasoning is a **three-axis contract** (mode / semantic effort / hard cap) owned
by config and the OpenAI request layer — not by inventing per-tag field meanings:

- Axes, defaults, and family table — [`CONFIG.md`](CONFIG.md)
- HTTP fields, warn+drop metadata, curl examples — [`SERVE.md`](SERVE.md)
- Chat template overrides — `chat_template`, `default_chatml` / env in [`env-vars.md`](env-vars.md)

Registry cards may publish `reasoning_effort` defaults. Effort-native tags should
**not** pin a hipfire named `thinking_budget`; absence means uncapped. Ornith 1.5
uses Qwen3.8-compatible `low`/`medium`/`xhigh` semantic prompt steering (default
`xhigh`) — never a budget reinterpretation. Qwen `<think>` framing is not
universal — Gemma uses boolean thinking (default off), Glimmer uses Onyx channel
strength, DeepSeek uses `thinking.type` + its own effort ladder.

---

## Related

| Topic | Owner |
|---|---|
| Config keys / defaults | [`CONFIG.md`](CONFIG.md) |
| Env vars | [`env-vars.md`](env-vars.md) |
| CLI pull/run/list | [`CLI.md`](CLI.md) |
| Arch IDs | [`architecture-ids.md`](architecture-ids.md) |
| Admissions | [`admissions.yml`](admissions.yml) (schema v2; exactly one earned record) |
| Validation routes | [`VALIDATION.md`](VALIDATION.md) |
| Dated benches | [`BENCHMARKS.md`](BENCHMARKS.md) (tables remain **historical** regardless of admission; admission and measurement classification are independent) |
