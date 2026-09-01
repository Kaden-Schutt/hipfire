# Configuration

**Owner:** daemon / user config keys (`docs/INDEX.md`).
**Machine sources:**

- Native schema + resolution: `crates/hipfire-config/src/lib.rs`
- Versioned CLI/daemon handoff: `hipfire_config::ProcessConfig`
- Compact runtime snapshots: `hipfire_runtime::config::RuntimeConfig` and
  `rdna_compute::feature_flags::FeatureFlags`

**Last checked:** 2026-07-21.

Persistent stores under `~/.hipfire/`:

1. **Global** — sparse typed `config.toml`; missing keys inherit schema defaults.
2. **Per-model overlay (primary)** — `models.toml`: aliases, local paths,
   registry identities, and sparse per-model overrides.
3. **Migration inputs** — `config.json`, `models.json`, and
   `per_model_config.json`. Migration writes TOML and preserves the JSON files
   for rollback.

Copyable sparse profiles for users, kernel developers, and retained-PM4 work
live in [`docs/configs/`](configs/README.md). They include direct mappings from
historical `HIPFIRE_*` inputs to their persistent TOML keys. Apply a whole
bundle with the dedicated profile command:

```bash
hipfire config profile set default
hipfire config profile set dev
hipfire config profile set redline
hipfire config profile create lab
hipfire config profile              # interactive wizard
```

Selection replaces the complete sparse global `config.toml` with the chosen
built-in or custom profile layer (no stale keys remain). Custom snapshots live
under `~/.hipfire/profiles/<name>.toml`. Profile names are not schema fields and
are never written into `config.toml`. The Settings TUI and CLI share the
hipfire-config helpers. Profiles are global-only.

Edit global settings interactively with `hipfire config`. Per-model overlays use
the typed commands: `hipfire config <tag> list|get|set|reset ...`. Global
non-interactive edits use `hipfire config list|get|set|reset ...`.

**Precedence (operator view):** one-shot CLI values **>** legacy-compatible
process env **>** per-model override **>** global TOML **>** registry card
(`recommended_settings`) **>** built-in defaults. `hipfire config explain`
prints the winning source and shadowed candidates. Direct daemon invocation
resolves the same local TOML plus compatibility environment before GPU startup.

TOML is the supported persistent surface. Environment variables named in the
schema are compatibility/one-shot overrides, not a second persistent config
system. The native CLI serializes a versioned, schema-validated
`ProcessConfig` to the daemon before the GPU is initialized. The daemon
revalidates it and lowers it directly into `RuntimeConfig` and `FeatureFlags`;
it does not materialize TOML values as process environment variables. Sparse
architecture-sensitive fields remain absent so existing compiled defaults are
unchanged.

Process and diagnostic keys are global-only. `hipfire config <model> set ...`
rejects them because the daemon snapshots these values once; claiming a
per-model override inside a long-lived serve process would be misleading.

This page is the normative **key/default/enum** table. CASK/TriAttention and
PFlash are experimental opt-ins; multi-GPU topology lives in its linked owner
rather than a duplicated matrix here.

---

## Process-wide hardware and kernel policy

Use the native config commands; the flat spelling shown in `hipfire config
list` remains accepted as a migration alias:

```bash
hipfire config set kernel.mmq auto
hipfire config set kernel.fp16 true
hipfire config set fusions.qkv_bias true
hipfire config set diagnostic.gemm_dump false
```

`auto` means the existing architecture/model policy wins. These keys default
to `auto` and do not materialize an environment value until explicitly set:

| Family | Keys |
|---|---|
| GEMV / quant | `kernel.gemv_dp4a`, `kernel.gemv_prefetch`, `kernel.gfx942_lds_gemv`, `kernel.hfq3_dp4a`, `kernel.hfq3_mmq`, `kernel.hfq4_mmq_rdna2`, `kernel.gcn5_wave64_hybrid`, `kernel.mmq`, `kernel.gfx942_gemv_v2` |
| MoE | `kernel.moe_grouped_i8`, `kernel.moe_paro_i8`, `kernel.moe_paro_i8_k8` |
| Certified arch routes | `kernel.rdna3_hfq4_qkvza_k2048`, `kernel.rdna3_hfq4_residual_stage_x32`, `kernel.rdna3_hfq4_sigmoid_buffer`, `kernel.rdna3_rmsnorm_vecsum`, `kernel.gfx942_rmsnorm_split` |

Stable/default-on and safety controls:

| Key | Default | Purpose |
|---|---:|---|
| `hardware.allow_mixed_arch` | `false` | Permit heterogeneous multi-GPU topology. |
| `hardware.tp_use_rccl` | `true` | RCCL tensor-parallel all-reduce. |
| `kernel.prefill_batched` | `true` | Batched prefill kernels. |
| `speculation.draft_f16` | `true` | FP16 DFlash draft activations. |
| `kernel.fp16` | `true` | Allow FP16 routes. |
| `kernel.lm_head_wmma` | `true` | Allow WMMA LM-head dispatch. |
| `kernel.hfq4g128_mmq` | `true` | Allow HFQ4G128 MMQ dispatch. |
| `kernel.moe_grouped_gemm` | `true` | Grouped-GEMM MoE prefill. |
| `kernel.deepseek4_q8_wmma` | `true` | DeepSeek4 Q8 WMMA prefill. |
| `kernel.deepseek4_q8_4w` | `true` | DeepSeek4 four-wave Q8 tile. |
| `speculation.ddtree_tree_la` | `true` | DDTree linearized-ancestor tape. |
| `speculation.dflash_fast_sample` | `true` | GPU DFlash sampled verification. |
| `speculation.dflash_q8_lmhead_wmma` | `true` | Q8 WMMA LM-head in DFlash verify. |
| `fusions.qkv_bias` | `true` | Fold supported QKV bias launches. |
| `kernel.rocblas_off` | `false` | Disable rocBLAS dispatch. |
| `fusions.force_unfused` | `false` | Force supported projection paths unfused. |
| `speculation.dflash_tree` | `false` | Enable DDTree tree-SWOR verification. |

The following default-off keys are experimental kernel-route overrides. They
are typed booleans, process-scoped, and visible in `hipfire config list` with
per-key help:

| Family | Keys |
|---|---|
| RDNA3 QKV/ZA | `kernel.rdna3_hfq4_qkv_wave64`, `kernel.rdna3_hfq4_qkvza_2wave`, `kernel.rdna3_hfq4_qkvza_wavepack4`, `kernel.rdna3_hfq4_qkvza_ldsx8`, `kernel.rdna3_hfq4_qkvza_reduce_chain`, `kernel.rdna3_hfq4_qkvza_hoist_x32` |
| RDNA3 residual / sigmoid / head | `kernel.rdna3_hfq4_residual_k2048`, `kernel.rdna3_hfq4_sigmoid_tight_grid`, `kernel.rdna3_hfq4_sigmoid_rows4`, `kernel.rdna3_hfq4_lm_head_k2048`, `kernel.rdna3_hfq4_moe_gate_up_k2048` |
| RMSNorm | `kernel.rmsnorm_mq_tight_lds`, `kernel.rdna3_rmsnorm_wavegrid`, `kernel.rdna3_rmsnorm_split`, `kernel.rdna3_rmsnorm_sign_lds`, `kernel.rdna3_rmsnorm_sign_const` |
| MoE | `kernel.moe_grouped_i8_k8`, `kernel.moe_grouped_i8_k4`, `kernel.moe_grouped_i8_k4_gfx12`, `kernel.moe_grouped_m2`, `kernel.moe_grouped_4w`, `kernel.moe_down_combine_vec4`, `kernel.moe_hfq6_i8`, `kernel.moe_hfq6_v2` |
| Other kernel routes | `kernel.fp8_wmma`, `kernel.dot2_gemv`, `kernel.wo_mmq`, `kernel.lm_head_overwrite`, `kernel.hfq4_mmq_gfx906_y64`, `kernel.gate_up_nosync`, `kernel.qkvza_split_tail`, `kernel.gfx942_gemv_v3`, `kernel.deterministic`, `kernel.mw16`, `kernel.q8_batched_legacy`, `kernel.rope_interleaved_legacy`, `kernel.rocblas_all_archs`, `kernel.lloyd_force_baseline` |

Diagnostic booleans all default off: `diagnostic.prompt_token_heat`,
`diagnostic.prompt_heat_json`, `diagnostic.draft_gemm_dump`,
`diagnostic.draft_subphase`, `diagnostic.mmq_quantize_only`,
`diagnostic.blob_force`, `diagnostic.gemm_dump`, and
`diagnostic.qkv_bias`.

Process-wide scalar policy is typed as well:

| Key | Default | Values / purpose |
|---|---:|---|
| `hardware.devices` | `null` | Physical device list installed as `ROCR_VISIBLE_DEVICES`; HIP receives matching post-filter logical selectors `0..N-1` before initialization, avoiding compounded nonzero filters. |
| `hardware.uniform_vram_tolerance_gb` | `null` | Free-VRAM spread override; unset uses the compiled default. |
| `generation.loop_guard_threshold` | `0` | Repeated 4-gram count that forces EOS; zero disables. |
| `generation.loop_guard_window` | `256` | Token window inspected by the loop guard. |
| `kernel.flash_partials_batch` | `null` | Prefill flash-attention scratch multiplier. |
| `kernel.lm_head_f16` | `"auto"` | `auto`, `native`, `f16`, `f32`, `fp32`, `legacy`, plus numeric compatibility aliases `1`/`0`. |
| `diagnostic.prompt_heat_limit` | `64` | Maximum token-heat rows. |

Developer-only scalar and variant controls live below `diagnostic.kernel` so
they remain visible and typed without becoming registry-authorized product
defaults:

| Key | Values when set |
|---|---|
| `diagnostic.kernel.gemv_rows` | `"1"`, `"2"`, `"4"`, `"8"` |
| `diagnostic.kernel.fp16_layer_min`, `diagnostic.kernel.fp16_layer_max` | non-negative layer index |
| `diagnostic.kernel.hfq3_mmq_layer_min`, `diagnostic.kernel.hfq3_mmq_layer_max` | non-negative layer index |
| `diagnostic.kernel.mmq_min_batch`, `diagnostic.kernel.rocblas_min_batch` | non-negative batch threshold |
| `diagnostic.kernel.ddtree_logw_cutoff` | non-negative number |
| `diagnostic.kernel.lloyd_mb4`, `diagnostic.kernel.mq3_mb4` | `"1"`, `"2"`, `"4"` |
| `diagnostic.kernel.gate_up_variant` | `ldsx`, `k4`, `ldscoop`, `2tile` |
| `diagnostic.kernel.gfx11_weight_load_policy` | `buffer`, `global`, `flat-buffer` |
| `diagnostic.kernel.gfx12_weight_load_policy` | `rt`, `global`, `ht`, `nt-rt`, `nt-ht` |
| `diagnostic.kernel.gfx942_mfma_prefill` | `"1"`, `"2"`, `"3"`, `"4"` |
| `diagnostic.kernel.rdna2_variant` | integer 1–5 |
| `diagnostic.kernel.wo_wmma_variant` | `ksplit`, `ksplit_det`, `k2`, `k2x32`, `k4`, `wmma`, `wmma2` |
| `diagnostic.compiler.hipcc_extra_flags` | advanced local compiler-flag string |

Every field currently read by the centralized `RuntimeConfig` and
`FeatureFlags` snapshots has a schema mapping and crosses the CLI/daemon
boundary in the versioned `ProcessConfig`; no child-environment projection is
used.

### Experimental developer controls

The long tail of kernel experiments and diagnostics is available in one flat,
global-only TOML namespace instead of as hundreds of ambient process switches:

```toml
[developer]
gfx1151_gate_up_wave64 = true
deepseek4_attn = "twin"
replay_pm4_queues = 4
```

The CLI accepts the same keys (`hipfire config set
developer.gfx1151_gate_up_wave64 true`) and preserves the supplied scalar
spelling. A legacy `HIPFIRE_FOO=value` one-shot maps to
`developer.foo = "value"` during startup resolution. Developer values are
snapshotted before GPU initialization, sent in `ProcessConfig`, and read only
from that immutable snapshot. They are never registry-authorized or valid in a
per-model overlay. Prefer a typed schema key whenever one exists.

Direct production reads of `HIPFIRE_*` are restricted to bootstrap paths that
must be known before config discovery or process launch: home/model roots,
binary/registry locations, kernel cache, spill directory, and quant diagnostic
output. Examples and tests may still inject compatibility variables before
startup.

---

## Generation / sampling

| Key | Default | Validated range / enum | Notes |
|---|---|---|---|
| `temperature` | `0.30` | number 0.0–2.0 | Stored default only; see send path below. |
| `top_p` | `0.80` | number (0, 1] | Stored default only; see send path below. |
| `top_k` | `20` | int 1–100000 | Registry/TOML/request top-K cutoff; the built-in is omitted so daemon/model fallback remains authoritative. |
| `min_p` | `0.0` | number 0.0–1.0 | Relative-probability cutoff; zero disables it. |
| `presence_penalty` | `0.0` | number 0.0–2.0 | OpenAI-style flat penalty over the repeat window; zero disables it. |
| `repeat_penalty` | `1.05` | number 1.0–3.0 | Kept low; higher values harm some MQ4 greedy paths (source comment). Stored default only; see send path below. |
| `max_tokens` | `4096` | int 1–393216 | Per-turn generation cap for run / OpenAI fallback; 384 Ki tokens matches DeepSeek V4 Flash 0731's documented maximum output length. |
| `max_seq` | `32768` | int 512–1048576 | KV logical capacity at load; the 1 Mi-token ceiling is serviced by VMM growth where supported. |
| `thinking` | `"on"` | `on` \| `off` | Thinking **mode** (on/off). Independent of effort and of any hard think-token cap. When off wins, effort and cap controls are dropped with a warning. |
| `reasoning_effort` | `"auto"` | `auto` \| `none` \| `low` \| `medium` \| `high` \| `xhigh` \| `max` | **Semantic** effort only — prompt / framing strength. Field meaning never changes by family and is **never** reinterpreted as a token budget. Unsupported or cross-family values are dropped with a warning (see below). |
| `thinking_budget` | `"med"` | `low` \| `med` \| `high` \| `xhigh` \| `max` \| `uncapped` | Legacy **named cap preset** only on models that still honor that config route. Not an effort dial. On effort-native models the string is dropped with a warning (no implicit cap). |
| `max_think_tokens` | *(absent)* | int 0–393216 when set | Explicit **integer** think-span cap (`reasoning.max_tokens`) on Qwen Jinja contracts. `0` or absence = uncapped. Positive N force-closes through the validated Qwen continuation path. DeepSeek, Gemma, Glimmer, and unsupported contracts drop this field with a warning rather than inventing model-specific closure behavior. Independent of mode and effort. |
| `max_total_think_tokens` | `0` | int 0–1000000 | Cross-reopen total `<think>` budget; `0` = off. |

### Reasoning contract (three independent axes)

Thinking controls are three orthogonal axes. A field's meaning never depends on
model family:

| Axis | Keys | Meaning |
|---|---|---|
| Mode | `thinking` / `reasoning.mode`; HTTP `chat_template_kwargs.enable_thinking`, `thinking.type` | Whether thinking is on or off for the turn |
| Effort | `reasoning_effort` / `reasoning.effort`; HTTP `reasoning.effort` / `reasoning_effort` | Semantic prompt strength only — **never** a budget |
| Cap | `max_think_tokens` / `reasoning.max_tokens`; legacy named `thinking_budget` / `reasoning.budget` | Hard think-token limit on contracts that support one; otherwise dropped+warned |

**Thinking budget map** (legacy named presets → integer, only when the loaded
model's contract still accepts the named-cap route):

| Preset | Tokens |
|---|---:|
| `low` | 512 |
| `med` | 2048 |
| `high` | 8192 |
| `xhigh` | 24576 |
| `max` | 32768 |
| `uncapped` | 0 (unlimited) |

**Rules:**

- Mode, effort, and cap are independent. Setting `reasoning_effort=low` does
  **not** invent a think-token cap; setting `thinking_budget=high` does **not**
  change effort.
- **Effort-native** models (Qwen3.8, Ornith 1.5, DeepSeek V4, Muse Glimmer) have **no**
  registry or built-in think-token cap by default. Qwen3.8 and Ornith 1.5 accept the
  explicit integer cap through the validated Qwen continuation path. DeepSeek V4 and Muse
  Glimmer expose effort but no parent-defined independent think-token cap, so
  integer and named cap fields are dropped with a visible warning.
- Registry cards may keep intentional `reasoning_effort` defaults. Omitting a
  budget field is the correct uncapped signal — do not pin effort-native tags
  to a hipfire named cap.
- Thinking disabled wins: effort and cap are dropped with a warning.
- Recognizable unsupported / cross-family values are normalized or dropped with
  `[WARN: INVALID CONFIG]` (server log + OpenAI response metadata under
  `hipfire.reasoning` / `hipfire.config_warnings`). Malformed types and
  out-of-range integers remain hard errors.
- Template probing only checks whether semantic effort is native; it never
  changes field meaning.

**Family examples** (full HTTP surface: [`SERVE.md`](SERVE.md)):

| Family | Mode | Effort | Cap |
|---|---|---|---|
| **Qwen3.8** | `enable_thinking=false` → native empty closed think | `low` \| `medium` \| `xhigh` (default `xhigh`); system-turn semantic prompt steering only — **not** a budget | Positive `max_think_tokens` force-closes; 0/absent uncapped. Named `thinking_budget` dropped+warned |
| **Ornith 1.5** | same Qwen on/off (`enable_thinking=false` → empty closed think) | Qwen3.8-compatible `low` \| `medium` \| `xhigh` semantic prompt steering (default `xhigh`; medium unsteered; **not** a budget reinterpretation) | Same Qwen integer continuation cap as Qwen3.8; 0/absent uncapped. Named `thinking_budget` dropped+warned |
| **Qwen3.6** (non-effort-native template) | same on/off | Effort **dropped+warned** — never converted to a cap. Set `thinking_budget` or `max_think_tokens` explicitly if you want a limit | Named preset or integer as requested |
| **DeepSeek V4** | `thinking.type` enabled/disabled (default on) | `low` \| `high` \| `max`; aliases `medium`/`xhigh`→`high`. | No parent-defined independent cap; integer/named cap fields dropped+warned |
| **Gemma4** | Boolean thinking; official default **off**; Plain/Gemma channel framing | No native effort — unsupported effort/cap dropped+warned | Unsupported; dropped+warned |
| **Muse Glimmer** | Always reasons (Onyx `to=self`); off dropped+warned | `low` \| `medium` \| `high` \| `xhigh` strength; no Qwen `<think>` framing | No parent-defined independent cap; integer/named cap fields dropped+warned |
| **Unsupported** | Controls dropped+warned | dropped+warned | dropped+warned |

**Effective sampling send order** (`request_f64` / `request_u64` / `run`): explicit CLI or HTTP request **>** per-model TOML overlay **>** registry `recommended_settings` **>** daemon/HFQ/arch fallback. This applies to `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, and `repeat_penalty`. Bare global sampling values are deliberately not transmitted on the current run/serve path; if one shadows a registry recommendation, the card value is recovered for that model. Registry entry `sampling` blocks are legacy metadata — the resolver reads `recommended_settings`; see [`MODELS.md`](MODELS.md).

`prompt.system` (legacy spelling `system_prompt`) is the corresponding request-scoped framing default. A client-supplied `system` or `developer` message wins; otherwise a per-model TOML value or registry-card `system_prompt` is inserted before the first user message.

---

## KV cache

| Key | Default | Validated values |
|---|---|---|
| `kv_cache` | `"auto"` | `auto`, `q8`, `asym4`, `asym3`, `asym2`, `fwht4`, `fwht3`, `fwht2`, `turbo`, `turbo4`, `turbo3`, `turbo2` |
| `kv_adaptive` | `"off"` | `off`, `conservative`, `balanced`, `aggressive`, or `advanced:k=<fwht4\|fwht3\|fwht2>,v=<lloyd4\|lloyd3\|lloyd2>` |

**Resolution of `auto`:** registry entry `default_kv_mode` if present and valid;
else universal fallback **`q8`**. There is no per-arch implicit FWHT table.

`turbo*` values remain accepted aliases for validation/compat; resolution maps them in `resolveKvMode`.

`kv_adaptive` is opt-in. With adaptive on, `max_seq` is the context guaranteed at the floor tier. Daemon param overrides `HIPFIRE_KV_ADAPTIVE` when set through CLI load path.

`--kv-backend vmm` enables on-demand HIP VMM mapping for single-GPU Qwen3.5
`q8`, `asym3`, and `fwht3` KV caches. The default remains `contiguous`. VMM
currently does not compose with pipeline/expert parallelism or CASK/TriAttention
eviction.

Legacy one-shot alias: `HIPFIRE_KV_MODE` (see [`env-vars.md`](env-vars.md)).

---

## Attention

| Key | Default | Values |
|---|---|---|
| `flash_mode` | `"auto"` | `auto` \| `always` \| `never` |

Lowered directly into the daemon snapshot. `HIPFIRE_ATTN_FLASH` remains a
legacy one-shot alias.

---

## Speculative decode

Only **one** mechanism runs. Canonical selector:

| Key | Default | Values |
|---|---|---|
| `speculation` | `"auto"` | `off` \| `auto` \| `ngram` \| `dflash` \| `mtp` \| `dspark` |

- **`off`** — AR only.
- **`auto`** — cascade by availability / eligibility; legacy mode knobs filter each mechanism. Under auto, DSpark can win over in-trunk MTP when a DSpark sidecar is present (CLI comments).
- Forced mechanism names bypass heuristics; missing prerequisites fall back to AR with a warning.

Legacy one-shot alias: `HIPFIRE_SPECULATION`. CLI: `--spec`.

### Mechanism knobs

| Key | Default | Values / range | Notes |
|---|---|---|---|
| `dflash_mode` | `"off"` | `on` \| `off` \| `auto` | **Default off.** `auto` enables on dense Qwen3.5-class targets and skips known-loss A3B cases. |
| `dflash_adaptive_b` | `true` | bool | Adaptive draft block size. |
| `dflash_ngram_block` | `"auto"` | `true` \| `false` \| `"auto"` | Verify-path n-gram defense; auto size-gates. |
| `mtp_mode` | `"auto"` | `off` \| `on` \| `auto` | Built-in MTP when weights present (DeepSeek path primary). Separate Qwen35 MTP env gate may apply — see env doc. |
| `mtp_k` | `3` | int 1–10 | |
| `dspark_conf_threshold` | `null` | `null` or number 0.0–1.0 | `null` ⇒ per-arch carrier default (qwen3 0.1 / deepseek4 0.3 in comments). |
| `ngram_mode` | `"off"` | `off` \| `on` \| `auto` | Model-free; byte-identical to AR when used. |
| `ngram_k` | `12` | int 2–32 | |
| `ngram_min_count` | `2` | int 1–10 | |
| `ddtree_budget` | `0` | int 0–64 | `0` = chain DFlash (no tree). |
| `ddtree_topk` | `4` | int 1–8 | |

Legacy compatibility input still wins at the top of the startup ladder for
the corresponding knobs, but the engine receives only the resolved immutable
snapshot. Full aliases: [`env-vars.md`](env-vars.md).

---

## MMQ screening

| Key | Default | Values / range |
|---|---|---|
| `mmq_screen` | `"auto"` | `off` \| `on` \| `auto` |
| `mmq_screen_threshold` | `0.10` | number (0, 1] |

MMQ policy is configured with `kernel.mmq = "auto" | true | false` and the
experimental weight-only route with `kernel.wo_mmq`. Screening only matters
when MMQ is on. Daemon arch-gates the sweep (RDNA3/3.5 family in source
comments). Legacy `HIPFIRE_MMQ` / `HIPFIRE_WO_MMQ` values remain one-shot
compatibility overrides.

---

## CASK / TriAttention eviction

| Key | Default | Range |
|---|---|---|
| `cask_sidecar` | `""` | path string; empty = disabled |
| `cask` | `false` | bool (m-fold vs plain drop) |
| `cask_budget` | `512` | int 64–65536 |
| `cask_beta` | `128` | int 0–65536 |
| `cask_core_frac` | `0.5` | number 0.0–1.0 |
| `cask_fold_m` | `2` | int 1–16 |
| `cask_auto_attach` | `false` | bool |

Developer escapes use `[developer] cask_off = true` and
`force_a3b_eviction = true`; the old names remain one-shot aliases.

TriAttention and CASK are experimental, opt-in features. A downloaded sidecar
does not attach by default. To experiment, set `cask_sidecar` explicitly or set
`cask_auto_attach=true`; set `cask=true` only when core-aware m-folding is also
intended. Sidecar generation: `hipfire sidecar-gen` — [`CLI.md`](CLI.md).

---

## Prompt processing

| Key | Default | Values |
|---|---|---|
| `prompt_normalize` | `true` | bool |

Collapse `\n{3,}` → `\n\n` at engine entry. Legacy alias:
`HIPFIRE_NORMALIZE_PROMPT` (`0`/`false`/`off`/`no` disable).

---

## PFlash speculative prefill (experimental)

| Key | Default | Range / values |
|---|---|---|
| `prefill_compression` | `"off"` | `off` \| `auto` \| `always` |
| `prefill_threshold` | `32768` | int 0–1048576 |
| `prefill_keep_ratio` | `0.05` | (0, 1] |
| `prefill_alpha` | `0.85` | [0, 1] |
| `prefill_min_keep` | `2048` | int 0–1048576 |
| `prefill_sink` | `256` | int 0–65536 |
| `prefill_recent` | `1024` | int 0–65536 |
| `prefill_block` | `128` | int 1–4096 |
| `prefill_drafter` | `""` | path |
| `prefill_drafter_device` | `-1` | int −1–15 (`-1` = same device as target) |
| `prefill_profile` | `false` | bool |
| `prefill_sparse_threshold` | `32768` | int 0–1048576 |

Off by default. Typed fields cover product PFlash policy; remaining experiments
live under `[developer]` and retain matching `HIPFIRE_PREFILL_*` one-shot
aliases. Detailed bypass reasons and serve wiring: [`SERVE.md`](SERVE.md) /
runtime PFlash module — not restated here.

---

## Server / serve admission

| Key | Default | Range |
|---|---|---|
| `host` | `"0.0.0.0"` | non-empty hostname/IP, no whitespace, ≤255 |
| `port` | `11435` | int 1–65535 |
| `idle_timeout` | `300` | int 0–86400 seconds (`0` = never unload) |
| `default_model` | `"qwen3.5:9b"` | non-empty tag/path string |
| `serve.local` | `false` | Force the current command to use a locally spawned daemon. |
| `max_request_bytes` | `67108864` (64 MiB) | int 4096–4GiB |
| `serve_max_queue` | `64` | int 0–100000 (`0` = uncapped depth) |
| `serve_queue_timeout_ms` | `30000` | int 0–3600000 (`0` = no wait timeout) |
| `experimental_budget_alert` | `false` | bool |

Serve HTTP surface: [`SERVE.md`](SERVE.md). The corresponding `HIPFIRE_MODEL`,
`HIPFIRE_IDLE_TIMEOUT`, `HIPFIRE_MAX_REQUEST_BYTES`,
`HIPFIRE_SERVE_MAX_QUEUE`, and `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS` names are
legacy one-shot aliases.

---

## Chat template overrides

| Key | Default | Validation |
|---|---|---|
| `chat_template` | `""` | empty or existing file path (`~/` expanded); existence + `isFile` only — no readability/access check |
| `default_chatml` | `true` | bool |

Lowered directly into the startup/load policy. `HIPFIRE_CHAT_TEMPLATE_FILE`
and `HIPFIRE_DEFAULT_CHATML` remain legacy one-shot aliases.

---

## Per-model overlay

```bash
hipfire config qwen3.5:9b set dflash_mode off
hipfire config qwen3.5:9b          # list resolved model policy
```

Only explicitly set keys are stored; others inherit global. Primary path:
`~/.hipfire/models.toml`. Legacy `models.json` and `per_model_config.json` are
migration inputs and are not native write targets.

---

## Legacy-compatible one-shot env overrides (examples)

Full inventory: [`env-vars.md`](env-vars.md). Prefer `hipfire config set` for
persistent policy. Parent-process values still take precedence for one-shot
bisects and compatibility:

```bash
HIPFIRE_KV_MODE=q8
HIPFIRE_ATTN_FLASH=never
HIPFIRE_NORMALIZE_PROMPT=0
HIPFIRE_SPECULATION=off
HIPFIRE_DFLASH_DRAFT=/path/to/draft.hfq
HIPFIRE_LOCAL=1
HIPFIRE_MODEL=qwen3.5:9b
```

---

## Typed daemon snapshots

The CLI resolves TOML, registry, one-shot flags, and compatibility input into a
versioned `ProcessConfig` before the GPU is initialized. The daemon revalidates
that envelope and lowers stable runtime values through
`RuntimeConfig::from_process_config`; `FeatureFlags` performs the corresponding
kernel-policy lowering. Direct daemon invocation resolves local TOML plus the
compatibility environment into the same `ProcessConfig` first. Neither path
uses ambient variables in engine hot paths.

| Field | TOML key | Legacy env | Default behavior |
|---|---|---|---|
| `normalize_prompt` | `prompt.normalize` | `HIPFIRE_NORMALIZE_PROMPT` | true unless `0`/`false`/`off`/`no` |
| `prompt_token_heat` | `diagnostic.prompt_token_heat` | `HIPFIRE_PROMPT_TOKEN_HEAT` | off unless `1` |
| `prompt_heat_json` | `diagnostic.prompt_heat_json` | `HIPFIRE_PROMPT_HEAT_JSON` | off unless `1` |
| `prompt_heat_limit` | `diagnostic.prompt_heat_limit` | `HIPFIRE_PROMPT_HEAT_LIMIT` | 64 |
| `dflash_mode` | `speculation.dflash` | `HIPFIRE_DFLASH_MODE` | `"off"` |
| `draft_f16` | `speculation.draft_f16` | `HIPFIRE_DRAFT_F16` | true unless `0` |
| `draft_gemm_dump` | `diagnostic.draft_gemm_dump` | `HIPFIRE_DRAFT_GEMM_DUMP` | off unless `1` |
| `draft_subphase` | `diagnostic.draft_subphase` | `HIPFIRE_DRAFT_SUBPHASE` | off unless `1` |
| `ddtree_budget` | `speculation.ddtree_budget` | `HIPFIRE_DDTREE_BUDGET` | 256 |
| `ddtree_topk` | `speculation.ddtree_topk` | `HIPFIRE_DDTREE_TOPK` | 8 |
| `prefill_batched` | `kernel.prefill_batched` | `HIPFIRE_PREFILL_BATCHED` | true unless `0` |
| `flash_partials_batch` | `kernel.flash_partials_batch` | `HIPFIRE_FLASH_PARTIALS_BATCH` | unset |
| `tp_use_rccl` | `hardware.tp_use_rccl` | `HIPFIRE_TP_USE_RCCL` | unset → RCCL default on; `0`/`false` opt out |
| `ngram_loop_threshold` | `generation.loop_guard_threshold` | `HIPFIRE_NGRAM_LOOP_THRESHOLD` | **0 (off)** |
| `ngram_window` | `generation.loop_guard_window` | `HIPFIRE_NGRAM_WINDOW` | 256 |
| `ngram_draft` | `speculation.ngram` | `HIPFIRE_NGRAM_DRAFT` | off |
| `ngram_k` | `speculation.ngram_k` | `HIPFIRE_NGRAM_DRAFT_K` | 12 |
| `ngram_min_count` | `speculation.ngram_min_count` | `HIPFIRE_NGRAM_MIN_COUNT` | 2 |
| `prompt_cache_capacity` | `memory.prompt_cache_capacity` | `HIPFIRE_PROMPT_CACHE_CAP` | 32 |
| `prompt_cache_unbounded` | `memory.prompt_cache_unbounded` | `HIPFIRE_PROMPT_CACHE_UNBOUNDED` | false |
| `devices` | `hardware.devices` | `HIPFIRE_DEVICES` | unset |
| `allow_mixed_arch` | `hardware.allow_mixed_arch` | `HIPFIRE_ALLOW_MIXED_ARCH` | false unless `1` |
| `uniform_vram_tolerance_gb` | `hardware.uniform_vram_tolerance_gb` | `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | unset |
| `mtp_mode` | `speculation.mtp` | `HIPFIRE_MTP_MODE` | `"auto"` |
| `mtp_k` | `speculation.mtp_k` | `HIPFIRE_MTP_K` | 3 |

`kernel.lm_head_f16` maps the live Qwen3.5/3.6 loader compatibility control
`HIPFIRE_LM_HEAD_F16`; the duplicate unused `RuntimeConfig` member was removed.
The similarly unused `RuntimeConfig::dflash_draft` member was removed rather
than promoted. Draft selection remains part of the typed speculation/load
policy and filename/registry discovery.

Note: CLI `ddtree_budget` default is `0` while bare `RuntimeConfig` default is `256` when only env/runtime path is used — CLI load params are the product path for `hipfire run`/`serve`.

Redline eligibility helper `mq4r_redline_default` (same file) is a **narrow runtime-default predicate** for replay backend selection (exact GPU arch `gfx1100`/`gfx1151`/`gfx1201` + case-insensitive `.mq4r` + pp=tp=1; model-family agnostic, no `arch_id` gate; `gfx1200` and all other arches remain opt-in), not a general config key and not Redline certification/registry admission. Built-in `hip` config profile or an explicit backend selection disables the automatic default. Policy: [`REDLINE.md`](REDLINE.md).

---

## Related

| Topic | Owner |
|---|---|
| Copyable TOML profiles | [`configs/`](configs/README.md) |
| Env inventory | [`env-vars.md`](env-vars.md) |
| Models / registry sampling | [`MODELS.md`](MODELS.md) |
| Serve API | [`SERVE.md`](SERVE.md) |
| CLI | [`CLI.md`](CLI.md) |
| Multi-GPU | [`multi-gpu.md`](multi-gpu.md) |
