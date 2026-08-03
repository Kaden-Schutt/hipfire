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
