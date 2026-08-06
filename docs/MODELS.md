# Models

**Owner:** registry-backed model surface (`docs/INDEX.md`).
**Machine sources:** curated `registry/models.json`; generated and bundled
`registry/v1.json` (loaded by `hipfire-registry`).
**Last checked:** 2026-07-22 against `origin/beta@202282de8759dfa6963ea5184ad2bf2b9259cef6`.

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

Default serve pre-warm tag is `qwen3.5:9b` (`CONFIG.md` → `default_model`). Per-tag sampling defaults come from registry `recommended_settings` only (applied by the native CLI request resolver). Registry `sampling` blocks (present on e.g. `deepseek-v4-flash`, `north-mini-code`) are **metadata only today** — they are not promoted by `RecommendedSettings::config_layer` or the native request resolver. See [`CONFIG.md`](CONFIG.md).

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
| `qwen3.6:35b-a3b-mq3p` | `qwen3.6-35b-a3b.mq3p` | 17.2 | 20 | | MQ3+P graded |
| `qwen3.6:35b-a3b-mq4p` | `qwen3.6-35b-a3b.mq4p` | 19.8 | 22 | | MQ4+P graded |
| `qwen3.6:35b-a3b-mfp4` | `qwen3.6-35b-a3b.mfp4` | 20.2 | 22 | | MFP4-E8 |
| `qwen3.6:35b-a3b-mq4r` | `qwen3.6-35b-a3b.mq4r` | 18.7 | 22 | | MQ4 Redline speed SKU (registry desc includes dated tok/s — treat as registry text, not a live baseline) |
| `qwen3.6:35b-a3b-mq5` | `qwen3.6-35b-a3b.mq5` | 23.7 | 26 | | Quality SKU |
| `qwen3.6:35b-a3b-mq6` | `qwen3.6-35b-a3b.mq6` | 27.7 | 30 | | Max quality |

Several A3B entries carry an `mtp.file` sidecar name (`qwen3.6-35b-a3b.mtp`). MTP enablement is config/runtime gated (`mtp_mode`, env); registry presence alone is not admission.

### Qwen 3.6 dense

| Tag | File | Size GB | Min VRAM | Default KV | Notes |
|---|---|---:|---:|---|---|
| `qwen3.6:27b` | `qwen3.6-27b.mq4` | 15.0 | 16 | q8 | Ships `triattn.file` in registry |
| `qwen3.6:27b-mq3` | `qwen3.6-27b.mq3` | 10.7 | 12 | | MQ3 alpha |

### DFlash draft artifacts (registry)

| Tag | File | Size GB | Min VRAM | Pairs with (by name) |
|---|---|---:|---:|---|
| `qwen3.5:9b-draft` | `qwen35-9b-dflash-mq4.hfq` | 0.55 | 6 | `qwen3.5:9b` |
| `qwen3.5:27b-draft` | `qwen35-27b-dflash-mq4.hfq` | 0.92 | 16 | `qwen3.5:27b` |
| `qwen3.5:27b-draft-mq3` | `qwen35-27b-dflash-mq3.hfq` | 0.67 | 12 | `qwen3.5:27b` (mq3 draft) |
| `qwen3.6:27b-draft` | `qwen36-27b-dflash-mq4.hfq` | 0.92 | 16 | `qwen3.6:27b` |
| `qwen3.6:27b-draft-mq3` | `qwen36-27b-dflash-mq3.hfq` | 0.67 | 12 | `qwen3.6:27b` |

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

### Other families (registry)

| Tag | File | Size GB | Min VRAM | Notes |
|---|---|---:|---:|---|
| `deepseek-v4-flash` | `deepseek-v4-flash.mq2lloyd` | 82 | 96 | arch_id=9; registry lists MTP + DSpark sidecars; entry carries a `sampling` block (temp=1.0 / top_p=1.0) that is **inert metadata** today — not applied by the CLI resolver |
| `minimax-m2.7` | `MiniMax-M2.7.mq2` | 79.2 | 96 | arch_id=10 Mixtral-style MoE |
| `north-mini-code` | `north-mini-code.mq4.hfq` | 16 | 24 | Cohere2-MoE arch_id=12; registry `sampling` block is **inert metadata** today |
| `vibethinker:3b` | `vibethinker-3b.mq4.hfq` | 1.82 | 3.5 | Qwen2 MQ4 |
| `vibethinker:3b-mq6` | `vibethinker-3b.mq6.hfq` | 2.51 | 5.0 | Qwen2 MQ6 |

### LFM2.5 (registry)

| Tag | File | Size GB | Min VRAM | Notes (registry `desc`) |
|---|---|---:|---:|---|
| `lfm2.5:350m` | `lfm2.5-350m.q8` | 0.38 | 1.9 | 350M dense; default artifact is **Q8** file |
| `lfm2.5:1.2b` | `lfm2.5-1.2b.mq4` | 0.7 | 2.2 | 1.2B Instruct dense |
| `lfm2.5:1.2b-thinking` | `lfm2.5-1.2b-thinking.mq4` | 0.7 | 2.2 | 1.2B Thinking dense |
| `lfm2.5:8b-a1b` | `lfm2.5-8b-a1b.mq4` | 4.66 | 6.2 | 8B-A1B MoE |

Registry `recommended_settings` for LFM tags is low temperature (0.05–0.2) with `repeat_penalty` 1.05 — applied by the CLI resolver. Do not treat a registry `sampling` field as active defaults.

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
| `qwen3` | `qwen3:8b` |
| `carnice` | `carnice:9b` |
| `qwopus` | `qwopus:9b` |
| `qwopus:{4b,9b,27b}-{mq4,hf4}` | matching primary `qwopus:{4b,9b,27b}` tag |
| `deepseek4` / `deepseek-v4` | `deepseek-v4-flash` |
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
| Qwen3.5 / 3.6 MoE A3B | 6 | `hipfire-arch-qwen35` | `*:35b-a3b*`, `nex-n2:mini` |
| Qwen2 | 7 | `hipfire-arch-qwen2` | `vibethinker:3b`, `vibethinker:3b-mq6` (support, not admission) |
| DeepSeek V4 Flash | 9 | `hipfire-arch-deepseek4` | `deepseek-v4-flash` |
| MiniMax-M2 | 10 | `hipfire-arch-minimax` | `minimax-m2.7` |
| LFM2.5 dense **and** MoE | 11 | `hipfire-arch-lfm2moe` | all `lfm2.5:*` |
| Cohere2-MoE | 12 | `hipfire-arch-cohere2moe` | `north-mini-code` |

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

Extension hints (loader recognizes several): `.mq4`, `.mq6`, `.mq4p`, `.mq4r`, `.mq2`, `.mq2lloyd`, `.mfp4`, `.hf4`, `.hf6`, `.hfq`, `.q8`, and related graded names as produced by quant tooling. Exact dtype routing is loader/kernel source, not this table.

---

## Thinking / chat framing

Reasoning models may emit `<think>…</think>`. Visibility and budgets are **config**, not registry fields:

- `thinking`, `thinking_budget`, `max_think_tokens`, `max_total_think_tokens` — [`CONFIG.md`](CONFIG.md)
- Chat template overrides — `chat_template`, `default_chatml` / env in [`env-vars.md`](env-vars.md)
- OpenAI request extras (`enable_thinking`, etc.) — [`SERVE.md`](SERVE.md)

## Current local family status

This table reflects the model families currently present under
`~/Models` / `~/.hipfire/models` in this checkout. It is intentionally
about runnable engine support, not just whether a generated `.hfq`
artifact exists on disk. `Cactus-Compute/needle` is omitted because it
is a custom non-Hipfire architecture target.

| Family | Local examples | Runtime status | DFlash | MTP | CASK | PP | Batched prefill | KLD ref gen | Compatible / missing kernels |
|---|---|---|---|---|---|---|---|---|---|
| Qwen 3.5 / 3.6 dense hybrid | `qwen3.5-{0.8b,2b,4b,9b}`, `qwen3.6-27b` | Qwen35 dense runtime is available (`arch_id=5`); PP is partial arch-resident (GEN-001; not production-supported). | Supported for paired dense drafts when target lm_head dtype is Q8/HFQ4/MQ4, plus MQ3 on gfx11/gfx12; MQ6 targets need AR. | Present as native Qwen35 speculative-verify/MTP surfaces for validated dense paths; still correctness-first and not the same as DFlash drafts. | Supported with TriAttention/CASK sidecars on FullAttention layers. | partial arch-resident PP (GEN-001; not production-supported) | Supported via Qwen35 batched prefill path. | Smoke refs exist for `qwen3.5-{0.8b,2b,9b}`; no full refs yet. | Dense Qwen35 decode/prefill kernels cover BF16/MQ4/MQ6 and selected MQ3. Missing DFlash batched lm_head/verify support for MQ6/MQ8/MQ2/F16 targets. |
| Qwen 3.5 / 3.6 MoE | `qwen3.6-35b-a3b`, `qwen3.5-122b-a10b` | Qwen35 MoE runtime is available (`arch_id=6`); PP is partial arch-resident (GEN-001; not production-supported). | Limited: dense-style DFlash works only where target/draft dtypes hit supported batched verify paths; MQ3 MoE is refused for DFlash. | MoE MTP code exists, but admission is narrower than dense and still gated by MoE dtype/layout validation. | Supported on FullAttention layers; no MoE-specific eviction of expert weights unless using the separate pager path. | partial arch-resident PP (GEN-001; not production-supported) | Supported; MoE batched prefill admits MQ4 control and newer MQ6/MQ3 surfaces on validated arches. | `qwen3.6-35b-a3b-bf16` KLD producer is currently skipped on error; no completed refs for MoE rows. | Indexed MoE gate/up/down, shared expert, router, and grouped GEMM kernels exist for Qwen35 MoE. Missing broad MQ3/MQ2/MQ8 MoE DFlash coverage and full validation for every local MoE artifact. |
| Qwen3-MoE / Qwen3-Coder (`qwen3_moe`) | `qwen3-coder-30b-a3b-instruct`, `tiny-random/qwen3-moe` | Not currently first-class. Local Coder HFQs are stamped `arch_id=0`, but source configs are `qwen3_moe`; that does not match the Qwen35-MoE loader layout. | No. | No. | No. | No. | No. | Listed as a desired KLD target for Coder, but no completed ref. | Needs a `qwen3_moe` architecture mapping and loader/kernel audit. Existing Qwen35 MoE kernels assume Qwen3.5 hybrid layer/tensor layout, not plain Qwen3-MoE/Coder layout. |
| DeepSeek V4 Flash | `deepseek-v4-flash.mq4.hfq` | Supported as dedicated DeepSeek V4 path (`arch_id=9`). | No Qwen-style DFlash drafter. | Supported as DeepSeek V4's own optional MTP speculative decode path. | No. | No. | Supported by DeepSeek V4 chunked batched prefill / MTP fill. | Not currently targeted for KLD refs. | Dedicated DeepSeek V4 kernels cover Hyper-Connections, compressed-KV indexer, SWA attention, routed MoE, MQ2/MQ3-Lloyd expert variants, and MTP. Missing CASK, PP, and Qwen-style DFlash integration. |
| LFM2.5-MoE | `lfm2.5-8b-a1b` | Supported as LFM2.5-MoE (`arch_id=11`) when compiled with `arch-lfm2moe`. Minimal AR bring-up. | No. | No. | No. | No. | No; prefill is per-token `decode_step`. | No completed refs yet. | Short-conv, attention, router, top-4 MoE, MQ4/MQ6 expert kernels are present. Missing batched prefill, DFlash/spec decode, CASK, PP, and grammar/tool-exec integration. |
| Dense LFM2.5 | `lfm2.5-350m`, `lfm2.5-1.2b-instruct` | Currently refused; pending admission under AXIS-003. Local MQ artifacts stamped `arch_id=11` are suspect because the LFM2-MoE parser requires MoE-only fields. | No. | No. | No. | No. | No. | KLD producer currently skipped on error for dense LFM2 rows. | Needs a dense LFM2 architecture crate or a generalized LFM2 loader. Current `hipfire-arch-lfm2moe` kernels/config assume `lfm2_moe` layer types, experts, and MoE FFN fields. |
| LLaMA-family dense | `llama-3.2-1b-instruct`, `supra-50m-instruct` | Basic dense AR support through LLaMA-family path (`arch_id=0`); dense PP code exists separately and its production proof is `HW-003`. The absence of Qwen35-style batched prefill is unrelated to dense PP. | No. | No. | No. | Dense PP code exists separately; `HW-003` physical proof pending. | No Qwen35-style batched prefill. | Producer skipped on error for `llama-3.2-1b-instruct-bf16` and `supra-50m-instruct-bf16`. | Dense LLaMA/GGUF-style GEMV, Q8/HFQ/MQ weight paths exist. Missing family-specific optimized prefill, DFlash, CASK, broader TP eligibility, and per-model quality refs. |
| Gemma 4 | `gemma-4-E2B-it` | Not runnable as a Gemma architecture yet. Prompt/tool-call support scaffolding exists, but no Gemma4 architecture crate is in the workspace. | No. | No. | No. | No. | No. | Not generated. | Needs `hipfire-arch-gemma4`, config/loader/forward kernels, and stop/tool-call policy wiring. Existing Gemma parser support is not model execution support. |

### Parallelism capability policy

PP and TP are required targets for every shipped family; EP is required
for MoE. CAP-001 will enforce refusal of planned cells and normalize
dense EP to one effective replica before mesh, device, allocation, or
collective creation; those behaviors are not currently enforced. Until
then, current loader behavior remains the existing architecture-specific
gates characterized by PAR-001. Dense LFM2 Single is currently refused
pending AXIS-003 admission. Dense normalization creates no EP support
claim. Physical hardware evidence is required for production status. The
authoritative status is
`.agent-progress/device-mesh-refactor-tracker.md`; this matrix must not
override it.

| Family / arch_id | Single | PP | TP | EP |
|---|---|---|---|---|
| LLaMA (0) | implemented | implemented code; HW-003 pending | partial; `has_qk_norm=true` / Qwen3-family metadata eligible; non-qk-norm LLaMA/Mistral refused; HW-006 pending for eligible artifacts | normalized-to-single(CAP-001) |
| plain Qwen3 (1) | implemented | implemented code; HW-003 pending | implemented code; HW-006 pending | normalized-to-single(CAP-001) |
| Qwen3.5 dense (5) | implemented | partial (GEN-001; HW-004 pending) | planned (AXIS-002; HW-007 pending) | normalized-to-single(CAP-001) |
| Qwen3.5 MoE (6) | implemented | partial (GEN-001; HW-004 pending) | planned (AXIS-002; HW-007 pending) | planned (AXIS-002; HW-011 pending) |
| Qwen2 dense (7) | implemented | planned (AXIS-001; HW-003 pending) | planned (AXIS-001; HW-006 pending) | normalized-to-single(CAP-001) |
| dots.ocr (8) | implemented | planned (AXIS-004; HW-012 pending) | planned (AXIS-004; HW-013 pending) | normalized-to-single(CAP-001) |
| Qwen35-VL (arch 5 VL extension) | implemented | planned (AXIS-004; HW-012 pending) | planned (AXIS-004; HW-013 pending) | normalized-to-single(CAP-001) |
| DeepSeek4 (9) | implemented | planned (AXIS-003; HW-008 pending) | planned (AXIS-003; HW-008 pending) | implemented code; HW-001 pending |
| MiniMax (10) | implemented | planned (AXIS-003; HW-009 pending) | planned (AXIS-003; HW-009 pending) | implemented code; HW-002 pending |
| LFM2 dense (11) | currently refused; planned admission AXIS-003 | planned (AXIS-003; HW-010 pending) | planned (AXIS-003; HW-010 pending) | normalized-to-single(CAP-001) |
| LFM2-MoE (11) | implemented | planned (AXIS-003; HW-010 pending) | planned (AXIS-003; HW-010 pending) | planned (AXIS-003; HW-010 pending) |
| Cohere2-MoE (12) | implemented | planned (AXIS-003; HW-010 pending) | planned (AXIS-003; HW-010 pending) | planned (AXIS-003; HW-010 pending) |
| toy / template (0xFF) | out-of-scope | out-of-scope | out-of-scope | out-of-scope |

TP×EP composition is explicitly refused (COMP-001): `--tp` and `--ep` cannot both exceed one. Reopening it requires a concrete deployment requirement and separate implementation/physical-validation task.

An axis marked partial, planned, or awaiting hardware evidence is not
currently claimed as supported. CAP-001 will refuse planned cells when
implemented; until then, an AXIS or GEN cell remains governed by the
current architecture-specific loader gates characterized by PAR-001. An
implemented EP cell with a pending HW gate is not production status.

Qwen35-VL caveat: the current `pp > 1` HFQ path bypasses VL detection and
does not explicitly refuse the admission at load. CAP-001 must make that
admission error explicit; this is not a claim that PP+VL works.

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
