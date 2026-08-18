# Environment variables

**Owner:** environment-variable inventory (`docs/INDEX.md`).
**Machine sources:** `HIPFIRE_*` token scan of Rust, Python, and shell sources under the repo (excluding `target/`, `.git/`, etc.).
**Config schema owner:** [`CONFIG.md`](CONFIG.md) (`crates/hipfire-config/src/lib.rs` + `crates/hipfire-runtime/src/config.rs`).
**Last checked:** 2026-07-31.

For persistent user configuration, prefer `hipfire config set ...` and
`~/.hipfire/config.toml`. Schema-declared environment variables are retained as
one-shot/compatibility overrides. The native CLI resolves stable typed fields
and experimental `[developer]` values into one versioned daemon snapshot; see
[`CONFIG.md`](CONFIG.md#process-wide-hardware-and-kernel-policy).

This page has two layers:

1. **Manual (normative for operators)** — precedence, product-facing knobs, LFM/Redline branch scope, and pointers. Defaults here are grounded in CLI/runtime source cited inline.
2. **Generated inventory** — exhaustive name → example source path table from the scan. Presence in the inventory means the token appears in source; it does **not** mean the knob is supported, stable, or admitted.

Detailed procedures belong in domain owners ([`CONFIG.md`](CONFIG.md), [`SERVE.md`](SERVE.md), [`multi-gpu.md`](multi-gpu.md), [`REDLINE.md`](REDLINE.md), [`VALIDATION.md`](VALIDATION.md), methodology pages). This file does not host validation matrices or benchmark floors.

Coverage check for top-level docs: `scripts/check-env-docs.py` (every
`HIPFIRE_*` in `AGENTS.md` / `README.md` / `CONTRIBUTING.md` must appear
somewhere in this file).

---

## Manual — precedence

1. **Built-in typed defaults** in `hipfire-config` / `RuntimeConfig`.
2. **Registry card** `recommended_settings` from `registry/v1.json`.
3. **Global** `~/.hipfire/config.toml`.
4. **Per-model overlay** from `~/.hipfire/models.toml`.
5. **Compatibility environment layer** for schema-declared `HIPFIRE_*` names.
6. **One-shot CLI flags** for the current command.

Legacy `config.json`, `models.json`, and `per_model_config.json` are read only
as migration inputs when their TOML successors do not exist. Invoking the
daemon binary directly still resolves local TOML plus compatibility variables
before GPU initialization, but does not add registry/per-model policy that was
not supplied by a native client.

Ladder for speculation selector (product path): `HIPFIRE_SPECULATION` / `--spec` > per-model > global > default `auto`.

Effective sampling send order (run/serve): CLI flags > per-model > registry `recommended_settings` > daemon/HFQ/arch fallback (field omitted when only a bare global default would apply). Global-only sampling edits are not sent; registry `sampling` blocks are inert metadata — see [`CONFIG.md`](CONFIG.md) / [`MODELS.md`](MODELS.md). **Chat** keeps a global session snapshot exception ([CHAT.md](CHAT.md)).

### TOML replacement for experimental variables

Production engine code no longer reads the ambient environment for kernel,
architecture, replay, or diagnostic behavior. Stable controls use typed schema
keys. Residual experiments use a flat global-only namespace:

```toml
[developer]
gfx1151_gate_up_wave64 = true
deepseek4_attn = "twin"
```

The generic compatibility rule is `HIPFIRE_FOO=value` →
`developer.foo = "value"` at startup. This preserves existing harnesses while
making TOML the persistent and navigable surface. `[developer]` values are
process-scoped, excluded from registry and per-model policy, and frozen before
GPU initialization. The only direct production environment reads left are
bootstrap locations needed to discover config or launch the process (for
example `HIPFIRE_HOME`, binary paths, registry URL, kernel cache, spill path,
and quant diagnostic output).

---

## Manual — product / operator knobs

Values and defaults below match `hipfire-config`, the native CLI, and/or `RuntimeConfig` as of the check date. Full key tables: [`CONFIG.md`](CONFIG.md).

### Paths and process

| Variable | Role | Notes |
|---|---|---|
| `HIPFIRE_HOME` | Native state/config root | Default `~/.hipfire`; used by config, registry cache, CLI, and TUI. |
| `HIPFIRE_DIR` | Script/harness compatibility | Not read by the native CLI; prefer `HIPFIRE_HOME`. |
| `HIPFIRE_MODELS_DIR` | Model discovery/lifecycle root | Overrides list/pull/remove/pre-warm and TUI model paths. |
| `HIPFIRE_MODEL` | Serve/run model tag or path | Also `default_model` config. |
| `HIPFIRE_DAEMON_BIN` | Daemon binary override | |
| `HIPFIRE_TUI_BIN` | TUI binary | |
| `HIPFIRE_ROCM_PATH` | hipfire-specific ROCm SDK root override | Highest priority (`HIPFIRE_ROCM_PATH` > `ROCM_PATH` > `HIP_PATH`). Must provide the runtime, headers, and `hipcc`. Authoritative: no fallback to another install or bare soname. |
| `ROCM_PATH` / `HIP_PATH` | ROCm/HIP compatibility root overrides | Used only when `HIPFIRE_ROCM_PATH` is unset (`ROCM_PATH` before `HIP_PATH`). `HIP_PATH=<root>/hip` normalizes to `<root>`. Multiple equally eligible roots without an override are refused — set `HIPFIRE_ROCM_PATH`. |
| `HIPFIRE_LOCAL=1` | Skip attaching to running serve | One-shot local daemon. |
| `HIPFIRE_REGISTRY_URL` | Dynamic registry fetch URL | |
| `HIPFIRE_NO_REGISTRY_FETCH=1` | Pin bundled registry | |

### KV / attention / prompt

| Variable | Default / sense | Source |
|---|---|---|
| `HIPFIRE_KV_MODE` | From config; **`auto` → registry `default_kv_mode` else `q8`** | CLI `resolveKvMode`; **not** a legacy hard-coded fwht-per-arch table |
| `HIPFIRE_KV_ADAPTIVE` | off unless set / param | Loader/CLI |
| `HIPFIRE_KV_PHYSICAL_CAP` | optional physical slot cap | Daemon |
| `HIPFIRE_ATTN_FLASH` | from `flash_mode` (`auto`/`always`/`never`) | CLI → daemon |
| `HIPFIRE_NORMALIZE_PROMPT` | on unless `0`/`false`/`off`/`no` | `RuntimeConfig` |
| `HIPFIRE_PROMPT_TOKEN_HEAT=1` | dump BPE heat | RuntimeConfig |
| `HIPFIRE_PROMPT_HEAT_JSON=1` | JSON heat | RuntimeConfig |
| `HIPFIRE_PROMPT_HEAT_LIMIT` | default **64** | RuntimeConfig |
| `HIPFIRE_LM_HEAD_F16` | default **`auto`** | Qwen loader compatibility alias for `kernel.lm_head_f16` |

### Speculation / DFlash / MTP / n-gram / DSpark

| Variable | Default / sense | Notes |
|---|---|---|
| `HIPFIRE_SPECULATION` | `off`/`auto`/`ngram`/`dflash`/`mtp`/`dspark` | Canonical selector |
| `HIPFIRE_DFLASH_DRAFT` | retired engine read | Still appears in legacy gate scripts; product draft discovery uses typed speculation/load policy and registry/filename matching. |
| `HIPFIRE_DFLASH_CTX_CAP` | **8192**; `0` restores uncapped legacy behavior | Caps draft-side context storage; over-cap requests fall back to AR |
| `HIPFIRE_DFLASH_WINDOW` | **0 / unset** (legacy), unless declared by draft metadata | Enables bounded draft SWA; refused with CASK eviction |
| `HIPFIRE_DFLASH_MODE` | RuntimeConfig default **`off`** | Distinct from config `dflash_mode` apply path — product CLI also uses load params |
| `HIPFIRE_DFLASH_NGRAM_BLOCK` | set/clear from config | |
| `HIPFIRE_DFLASH_CKPT_RESUME` / `HIPFIRE_CACHE_CKPT_*` | checkpointing | Qwen DFlash path |
| `HIPFIRE_DRAFT_MAX` | routes to active mech window | CLI |
| `HIPFIRE_DRAFT_F16` | on unless `0` | RuntimeConfig |
| `HIPFIRE_NGRAM_DRAFT` | `1` forces n-gram | Loader always honors |
| `HIPFIRE_NGRAM_DRAFT_K` / `HIPFIRE_NGRAM_MIN_COUNT` | n-gram params | |
| `HIPFIRE_NGRAM_LOOP_THRESHOLD` | default **0 (off)** | RuntimeConfig |
| `HIPFIRE_NGRAM_WINDOW` | default 256 | RuntimeConfig |
| `HIPFIRE_MTP_MODE` / `HIPFIRE_MTP_K` | auto / 3 | Config + RuntimeConfig |
| `HIPFIRE_QWEN35_MTP` / `HIPFIRE_QWEN35_MTP_K` | Qwen35 MTP opt-in gate | Loader — separate from DeepSeek MTP |
| `HIPFIRE_DEEPSEEK4_SPEC_DECODE` / `HIPFIRE_DEEPSEEK4_SPEC_K` | DeepSeek MTP legacy | |
| `HIPFIRE_DEEPSEEK4_DSPARK` / `HIPFIRE_DEEPSEEK4_DSPARK_CONF_THRESHOLD` | DSpark | |
| `HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD` / `HIPFIRE_QWEN35_DSPARK_CONF_THRESHOLD` | per-arch conf | |
| `HIPFIRE_DDTREE_BUDGET` / `HIPFIRE_DDTREE_TOPK` | tree draft | Runtime defaults 256/8 if env-only; CLI config defaults 0/4 |
| `HIPFIRE_DDTREE_*` | research/diag family | See inventory; not product defaults |

### Graph / MMQ / prefill

| Variable | Notes |
|---|---|
| `HIPFIRE_GRAPH` | hipGraph capture opt-in (quality caveats; AR-oriented) |
| `HIPFIRE_GRAPH_MOE` | MoE graph opt-in |
| `HIPFIRE_VERIFY_GRAPH` / `_TIMING` / `_TREE` | verify-side graph diag |
| `HIPFIRE_MMQ` / `HIPFIRE_WO_MMQ` | MMQ activation |
| `HIPFIRE_MMQ_SCREEN` / `HIPFIRE_MMQ_SCREEN_THRESHOLD` / `HIPFIRE_MMQ_MIN_BATCH` | screening |
| `HIPFIRE_PREFILL_COMPRESSION` and `HIPFIRE_PREFILL_*` | PFlash + batched prefill knobs (config mirrors) |
| `HIPFIRE_PREFILL_BATCHED` | RuntimeConfig: on unless `0` (Qwen-style batched prefill gate — **not** the LFM flag) |
| `HIPFIRE_FLASH_PREFILL` | Developer override for Q8 WMMA flash prefill: `0` forces off, `1` forces on; unset uses the architecture/workload envelope. |
| `HIPFIRE_FLASH_PREFILL_FIXED_HD` | Developer ablation: fixed-head-dimension specialization is on unless `0`. |
| `HIPFIRE_FLASH_PREFILL_PREFETCH_V` | Developer ablation: gfx12 V prefetch is on unless `0`. |

### LFM (arch 11) — branch-scoped optimized prefill

| Variable | Status | Scope |
|---|---|---|
| `HIPFIRE_LFM2_PREFILL_BATCH` | **opt-in**; require `=1` | **Branch-only; not shipped** as a generic product default. Audited wording: **350M dense MQ4** cohort + **gfx1201** + this flag at `lfm-redline@692a726dde53508cb53de1a74c720e75a7c9f33e` (absent from `origin/beta@202282de…`). Does **not** admit Q8/other cohorts/default-on. |
| `HIPFIRE_LFM2_PREFILL_MAX_BATCH` | default 256, cap 512 | Chunk size for batched path |
| `HIPFIRE_LFM2_GRAPH` | LFM graph experiments | Source in LFM crate/daemon |
| `HIPFIRE_LFM2_GFX1201_DECODE_FUSION` | request bit; `=0` disables | Request bit for exact gfx1201 350M dense-MQ4 decode fusion. **Unset or `=1`** requests fusion; **`=0`** (or any other set value) disables the request. Actual enablement still depends on implementation support (gfx1201, graph off, lowered forward). Recorded on the sealed LFM [`admissions.yml`](admissions.yml) evidence row; **not** an automatic Redline runtime-default opt-out (LFM has no current automatic selector). |
| `HIPFIRE_FORWARD_LOWERED` | default on (LFM and several other arches); `=0` opts out | Shared lowered forward escape hatch. For LFM: **unset or any value other than `0`** keeps the lowered path; **`=0`** forces the legacy hand loop. Lowered-on is required for exact decode fusion where that path exists. Recorded on the sealed LFM [`admissions.yml`](admissions.yml) evidence row; **not** an automatic Redline runtime-default opt-out. |
| Other `HIPFIRE_LFM*` | diag/trace | Inventory |

Eager LFM prefill remains available when the batch flag is off **or** the GPU is not gfx1201. On gfx1201 with `HIPFIRE_LFM2_PREFILL_BATCH=1`, selection is GPU+flag only with **no post-selection fallback** — unsupported cohorts fail closed at the **runtime fixture validation/guard** (exact 350M dense MQ4 fixture only). Source symbol `validate_350m_mq4_admission` is a fixture-shape check only; its name does **not** create a product admission — [`admissions.yml`](admissions.yml) remains the sole authority (schema v2; exactly one earned retained-PM4 product row).

### CASK / serve / multi-GPU

`HIPFIRE_CASK_OFF` is a retired compatibility name. The current Rust control
plane does not consume it; use an empty `memory.cask.sidecar` and keep
`memory.cask.auto_attach=false` instead. The old name remains only in a loader
diagnostic and developer harness exports pending their cleanup.

| Variable | Notes |
|---|---|
| `HIPFIRE_FORCE_A3B_EVICTION=1` | Override A3B refusal (not recommended) |
| `HIPFIRE_IDLE_TIMEOUT` | Serve idle unload seconds |
| `HIPFIRE_MAX_REQUEST_BYTES` | Body cap |
| `HIPFIRE_SERVE_MAX_QUEUE` / `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS` | Admission queue |
| `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` | Research budget nudge |
| `HIPFIRE_DEVICES` / `HIPFIRE_TP` / `HIPFIRE_TP_USE_RCCL` | Multi-GPU / TP. `HIPFIRE_DEVICES` is the compatibility alias for `hardware.devices`; startup lowers its physical list to ROCr selectors plus matching HIP logical selectors. |
| `HIPFIRE_ALLOW_MIXED_ARCH=1` | Mixed arch pairs |
| `HIPFIRE_PP_LAYERS` / `HIPFIRE_PP_PFLASH` | Pipeline parallel |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | Uniform init tolerance |

### Redline / retained replay

Policy owner: [`REDLINE.md`](REDLINE.md) (**shipped / ref-pinned**). Timing is not route-certified without same-report timed-arm proof. Registry admission is not canonical certification.

| Variable | Notes |
|---|---|
| `HIPFIRE_REPLAY_BACKEND` | `hip` / `off` / `shadow` / `auto`. Unset may select `auto` only from the automatic product default `mq4r_redline_default` — exact GPU arch `gfx1100`/`gfx1151`/`gfx1201` + case-insensitive `.mq4r` + pp=tp=1 (model-family agnostic; no `arch_id` gate; `gfx1200` and all other arches remain opt-in). Existing LFM `.mq4` registry evidence is **not** automatically selected because it is not `.mq4r`; any usable non-default retained route is explicit opt-in and must still prove route support. The sealed LFM [`admissions.yml`](admissions.yml) row is registry evidence/admission only and does not wire runtime defaults. Runtime default ≠ Redline certification/registry admission. Built-in `hip` config profile, another explicit backend selection, or `=hip` disables the automatic default. |
| `HIPFIRE_REPLAY_TRANSPORT` | `pm4` / AQL family |
| `HIPFIRE_REPLAY_MANUAL_CAPTURE` | Manual capture delimiters |
| `HIPFIRE_REPLAY_PM4_*` | PM4 research knobs — inventory |
| `HIPFIRE_REPLAY_ROUTE_PROOF_LOG` | Developer-only / one-shot compat for `diagnostic.replay.route_proof_log`. When `1`/`true`/`on` (or TOML `true`), the daemon emits one post-generate retained-route proof marker per successful request: `HIPFIRE_REPLAY_ROUTE_PROOF transport=<name> position=<n> request_id=<id> replays=<count>`. Off by default; product coherence smoke enables it only via temporary serve_harness `config.toml`, not ambient env. |

### Chat template

| Variable | Notes |
|---|---|
| `HIPFIRE_CHAT_TEMPLATE_FILE` | External jinja path |
| `HIPFIRE_DEFAULT_CHATML` | Only load-bearing at `0` (disable ChatML fallback) |
| `HIPFIRE_JINJA_CHAT` | Serve jinja path toggle |
| `HIPFIRE_QWEN35_GRAMMAR` | Tool grammar |

### Perf / diag (common)

| Variable | Notes |
|---|---|
| `HIPFIRE_DPM_WARMUP_SECS` | DPM warmup before timing |
| `HIPFIRE_HOST_TIMING=1` | Host timing JSON |
| `HIPFIRE_PROFILE` / `HIPFIRE_PROFILE_DECODE` / `HIPFIRE_PROFILE_CYCLES` | Profiling |
| `HIPFIRE_DETERMINISTIC` | Determinism toggles in dispatch |
| `HIPFIRE_DS4_DENSE_ACT_DIR` | DeepSeek4 calibration-only dump of P1 projection inputs in `collect_e8_hessian` format; direct evaluator flag `--dump-dense-acts` is preferred. |
| `HIPFIRE_HIPCC_EXTRA_FLAGS` | Compatibility alias for `diagnostic.compiler.hipcc_extra_flags` |
| `HIPFIRE_KERNEL_CACHE` | Kernel cache dir (`var_os`) |
| `HIPFIRE_*_DUMP` / `*_TRACE` / `*_PROFILE` | Diagnostic families — see inventory |

Kernel-selector and arch-specific `HIPFIRE_GFX*` / `HIPFIRE_RDNA*` /
`HIPFIRE_MOE_*` levers are **research/power-user**. Centralized
`FeatureFlags` controls now have typed TOML keys under `kernel` or
`diagnostic.kernel`; defaults and safety remain source-defined and none are
registry-authorized.

---

## Manual — top-level doc references

The canonical documentation checker requires every `HIPFIRE_*` token in `AGENTS.md`, `README.md`, and `CONTRIBUTING.md` to appear in this file. Tokens historically routed from those surfaces (keep listed even if a root file is later thinned):

`HIPFIRE_ATTN_FLASH`, `HIPFIRE_DDTREE_` (prefix family; concrete vars in inventory), `HIPFIRE_DFLASH_DRAFT`, `HIPFIRE_GRAPH`, `HIPFIRE_HOST_TIMING`, `HIPFIRE_KV_MODE`, `HIPFIRE_LM_HEAD_F16`, `HIPFIRE_LOCAL`, `HIPFIRE_NORMALIZE_PROMPT`, `HIPFIRE_PROMPT_HEAT_JSON`, `HIPFIRE_PROMPT_HEAT_LIMIT`, `HIPFIRE_PROMPT_TOKEN_HEAT`, `HIPFIRE_VERIFY_GRAPH`.

---

## Manual — TOML key → legacy environment compatibility

Compatibility map from the `hipfire-config` schema. TOML is the supported
persistent surface. The native CLI sends typed policy directly to the daemon;
it does not project these values into the child environment. The names below
are accepted only as compatibility/one-shot input to the resolver. Message-
field-only keys (temperature, thinking, …) have no compatibility alias.

Copyable user, developer, and retained-PM4 TOML profiles are in
[`docs/configs/`](configs/README.md).

| Config key | Env |
|---|---|
| `kv_cache` | `HIPFIRE_KV_MODE` |
| `flash_mode` | `HIPFIRE_ATTN_FLASH` |
| `prompt_normalize` | `HIPFIRE_NORMALIZE_PROMPT` |
| `dflash_ngram_block` | `HIPFIRE_DFLASH_NGRAM_BLOCK` |
| `experimental_budget_alert` | `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` |
| `max_total_think_tokens` | `HIPFIRE_MAX_TOTAL_THINK_TOKENS` |
| `mtp_mode` / `mtp_k` | `HIPFIRE_MTP_MODE` / `HIPFIRE_MTP_K` |
| `chat_template` | `HIPFIRE_CHAT_TEMPLATE_FILE` |
| `default_chatml=false` | `HIPFIRE_DEFAULT_CHATML=0` |
| `speculation` | `HIPFIRE_SPECULATION` |
| `default_model` / serve model | `HIPFIRE_MODEL` |
| `idle_timeout` | `HIPFIRE_IDLE_TIMEOUT` |
| `max_request_bytes` | `HIPFIRE_MAX_REQUEST_BYTES` |
| `serve_max_queue` | `HIPFIRE_SERVE_MAX_QUEUE` |
| `serve_queue_timeout_ms` | `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS` |
| `serve.local` | `HIPFIRE_LOCAL` |
| `prefill_*` | matching `HIPFIRE_PREFILL_*` |
| `mmq_screen*` | `HIPFIRE_MMQ_SCREEN*` |
| `hardware.devices` | `HIPFIRE_DEVICES`; synchronizes `ROCR_VISIBLE_DEVICES=<physical list>` with `HIP_VISIBLE_DEVICES=0..N-1` before GPU initialization |
| `hardware.allow_mixed_arch` | `HIPFIRE_ALLOW_MIXED_ARCH` |
| `hardware.tp_use_rccl` | `HIPFIRE_TP_USE_RCCL` |
| `hardware.uniform_vram_tolerance_gb` | `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` |
| `generation.loop_guard_threshold` / `generation.loop_guard_window` | `HIPFIRE_NGRAM_LOOP_THRESHOLD` / `HIPFIRE_NGRAM_WINDOW` |
| `kernel.flash_partials_batch` | `HIPFIRE_FLASH_PARTIALS_BATCH` |
| `kernel.lm_head_f16` | `HIPFIRE_LM_HEAD_F16` |
| `diagnostic.prompt_*` | matching `HIPFIRE_PROMPT_*` |
| `diagnostic.kernel.*` / `diagnostic.compiler.*` | schema-declared kernel/compiler compatibility aliases; enumerate with `hipfire config schema --json` |

---

## Manual — Cargo features (not env)

`crates/hipfire-runtime/Cargo.toml` default features (checked 2026-07-19):

`arch-qwen35`, `arch-qwen35-vl`, `arch-llama`, `arch-qwen2`, `arch-deepseek4`, `arch-cohere2moe`, `arch-dots-ocr`, `deltanet`.

`hipfire-arch-lfm2moe` / `hipfire-arch-minimax` are linked as ordinary dependencies on the loader/runtime graph (not named in that default feature list). Feature toggles are build-time, not `HIPFIRE_*` env.

---

## Generated inventory

**Do not hand-edit rows below** except by re-running the source scan.
**Generation method:** token scan over visible `*.rs`, `*.py`, and `*.sh`, excluding ignored/generated files.
**Columns:** variable; up to two lexical source paths.
**Count:** 715

| Variable | Example source path(s) |
|---|---|
| `HF_ENDPOINT` | crates/hipfire-cli/src/main.rs |
| `HIPFIRE_9B_MODEL` | scripts/bisect_9b_decode.sh |
| `HIPFIRE_ADAPTIVE_B_DOWN` | crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_ADAPTIVE_B_UNSAFE` | crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_ADAPTIVE_B_UP` | crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_AGENTIC_GATE_NO_VRAM_CHECK` | scripts/agentic-gate.sh |
| `HIPFIRE_AGENTIC_GATE_OUT` | scripts/agentic-gate.sh |
| `HIPFIRE_ALLOW_MIXED_ARCH` | crates/hipfire-runtime/src/config.rs, crates/hipfire-runtime/src/multi_gpu.rs |
| `HIPFIRE_ALLOW_MQ2` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_ALLOW_MQ2_LLOYD` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_ALLOW_MQ3_LLOYD` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_ALLOW_MQ4_LLOYD` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_ALLOW_UNIT_IMATRIX` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_AR_GRAPH` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-cli/src/main.rs |
| `HIPFIRE_ATTENTION_REDUCE_GATED_MQ_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_ATTN_FLASH` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-config/src/lib.rs |
| `HIPFIRE_AWQ_EXPERTS` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_AWQ_F1_ONLY` | crates/hipfire-quantize/src/main.rs, scripts/awq_alpha_sweep.sh |
| `HIPFIRE_A_OUT` | scripts/ab-dispatch-validation.sh |
| `HIPFIRE_A_REF` | scripts/ab-dispatch-validation.sh |
| `HIPFIRE_BASELINE_ARCH` | crates/hipfire-runtime/examples/coherence_probe.rs, scripts/kernel_atlas.py |
| `HIPFIRE_BASELINE_FORMATS` | scripts/baseline_quant_smoke.sh |
| `HIPFIRE_BASELINE_KV_MODE` | scripts/baseline_quant_smoke.sh |
| `HIPFIRE_BASELINE_MAX_GEN` | scripts/baseline_quant_smoke.sh |
| `HIPFIRE_BASELINE_MODEL` | scripts/baseline_quant_smoke.sh |
| `HIPFIRE_BASELINE_NAME` | scripts/baseline_quant_smoke.sh |
| `HIPFIRE_BASELINE_OUT` | scripts/baseline_quant_smoke.sh |
| `HIPFIRE_BASELINE_PROMPT` | scripts/baseline_quant_smoke.sh |
| `HIPFIRE_BASELINE_PROMPT_MODE` | scripts/baseline_quant_smoke.sh |
| `HIPFIRE_BASELINE_WIDE_MARGIN` | scripts/baseline_quant_smoke.sh |
| `HIPFIRE_BENCH_AB` | benchmarks/scripts/bench_dflash_27b_gfx906.sh |
| `HIPFIRE_BENCH_MAX` | scripts/adaptive_b_bench.sh, scripts/qwen36_bench.sh |
| `HIPFIRE_BENCH_N` | crates/rdna-compute/examples/bench_indexed_moe_keystone.rs |
| `HIPFIRE_BENCH_RUNS` | scripts/adaptive_b_bench.sh, scripts/bench_qwen36_ar_dflash.sh |
| `HIPFIRE_BIN` | scripts/calibrate_multigpu.sh |
| `HIPFIRE_BLOB_FORCE` | crates/rdna-compute/src/dispatch.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_BRANCH` | scripts/mi300x_bootstrap.sh |
| `HIPFIRE_B_REF` | scripts/ab-dispatch-validation.sh |
| `HIPFIRE_C2M_DUMP_PROMPT` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_C2M_EMPTY_TURN_GUARD` | crates/hipfire-arch-cohere2moe/src/spec_emit.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_C2M_NORMDUMP` | crates/hipfire-arch-cohere2moe/src/forward.rs |
| `HIPFIRE_CACHE_CKPT_INTERVAL` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_CACHE_CKPT_MAX` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_CACHE_CKPT_RESUME` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_CALIB_PROFILE` | crates/hipfire-runtime/examples/triattn_validate.rs |
| `HIPFIRE_CANARY_MODEL` | scripts/gfx906_fallback_canary.sh |
| `HIPFIRE_CANARY_PREFILL` | scripts/gfx906_fallback_canary.sh |
| `HIPFIRE_CANARY_RUNS` | scripts/gfx906_fallback_canary.sh |
| `HIPFIRE_CASK_OFF` | crates/hipfire-loader/src/lib.rs, scripts/redline_daemon_harness.py, tools/redline/product_bench.py, scripts/serve_harness.py (retired literal; not consumed by Rust config) |
| `HIPFIRE_CASK_SIDECAR` | crates/hipfire-config/src/lib.rs |
| `HIPFIRE_CHATML` | crates/hipfire-runtime/examples/probe_argmax_agreement.rs |
| `HIPFIRE_CHAT_CURRENT_DATE` | crates/hipfire-runtime/src/prompt_frame.rs |
| `HIPFIRE_CHAT_TEMPLATE_FILE` | crates/hipfire-config/src/lib.rs, crates/hipfire-loader/src/lib.rs |
| `HIPFIRE_CLI_BIN` | crates/hipfire-tui/src/hipfire/doctor.rs, crates/hipfire-tui/src/hipfire/mod.rs |
| `HIPFIRE_COHERE2MOE_Q8_SCALAR` | crates/hipfire-arch-cohere2moe/src/forward.rs |
| `HIPFIRE_COHERENCE_MAX_SEQ` | scripts/coherence-gate-cohere2moe.sh |
| `HIPFIRE_COHERENCE_OUT` | scripts/coherence-gate-cohere2moe.sh, scripts/coherence-gate-deepseek4-mtp.sh |
| `HIPFIRE_COHERENCE_TIMEOUT` | scripts/coherence-gate-deepseek4-mtp.sh, scripts/coherence-gate-deepseek4-recall.sh |
| `HIPFIRE_COHERE_DEBUG` | crates/hipfire-arch-cohere2moe/src/forward.rs |
| `HIPFIRE_COMPILER_FLAGS` | autoresearch/ar/certify/cross_arch.py, crates/rdna-compute/src/compiler.rs |
| `HIPFIRE_COMP_DUMP` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_CONTAINER` | scripts/container-gate.sh |
| `HIPFIRE_CONV_QKNORM` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_CONV_QKNORM_SHAPE` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/rdna-compute/src/norm.rs |
| `HIPFIRE_CONV_SCALAR_PREP` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_CQN_BLOCK` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_CQN_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_CQN_SCALAR_PREP` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_DAEMON` | scripts/test_pr228_spiral_check.sh |
| `HIPFIRE_DAEMON_BIN` | autoresearch/ar/certify/serve_runner.py, autoresearch/ar/gate/serve_probe.py |
| `HIPFIRE_DAEMON_NAME` | scripts/serve_harness.py |
| `HIPFIRE_DDTREE_ASSERT_MASK` | crates/hipfire-arch-qwen35/src/speculative.rs |
| `HIPFIRE_DDTREE_BUDGET` | crates/hipfire-arch-llama/src/spec_impl.rs, crates/hipfire-arch-qwen35/src/dflash_spec.rs |
| `HIPFIRE_DDTREE_DUMP_PQ` | crates/hipfire-runtime/examples/ddtree_pq_sim.rs |
| `HIPFIRE_DDTREE_FORCE_SLOW` | crates/hipfire-arch-qwen35/src/speculative.rs |
| `HIPFIRE_DDTREE_LOGW_CUTOFF` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_DDTREE_PATH_C_VERBOSE` | scripts/path-c-smoke.sh |
| `HIPFIRE_DDTREE_TAPE_DUMP` | crates/hipfire-arch-qwen35/src/speculative.rs |
| `HIPFIRE_DDTREE_TOPK` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs |
| `HIPFIRE_DDTREE_TREE_LA` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_DDTREE_VERIFY` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_DEBUG_BATCH` | crates/hipfire-arch-qwen35/src/mtp_spec.rs, crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_DEEPSEEK4_AR` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs |
| `HIPFIRE_DEEPSEEK4_ATTN` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_ATTN_PER_POS` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_ATTN_TWIN` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_BATCH_HEAD` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_BENCH_RAW` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs |
| `HIPFIRE_DEEPSEEK4_CACHE_TRACE` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_DEEPSEEK4_CACTUS` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs |
| `HIPFIRE_DEEPSEEK4_CHAT_RAW` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs |
| `HIPFIRE_DEEPSEEK4_COMP_F16_WMMA` | crates/hipfire-arch-deepseek4/src/arch.rs, crates/hipfire-arch-deepseek4/src/deepseek4.rs |
| `HIPFIRE_DEEPSEEK4_COMP_ROPE_POS` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_DSA_WMMA` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_DSPARK` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs, crates/hipfire-runtime/src/loader_api.rs |
| `HIPFIRE_DEEPSEEK4_DSPARK_CONF_THRESHOLD` | crates/hipfire-arch-deepseek4/src/dspark_speculator.rs |
| `HIPFIRE_DEEPSEEK4_DUMP_PROMPT` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_DEEPSEEK4_DUMP_STATE` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_DUMP_TOPK` | crates/hipfire-dispatch/src/families/moe.rs, crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DEEPSEEK4_E8_U4` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END` | crates/hipfire-arch-deepseek4/src/arch.rs |
| `HIPFIRE_DEEPSEEK4_F32_TRACE` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU` | crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DEEPSEEK4_GEN_TOKENS` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs |
| `HIPFIRE_DEEPSEEK4_GRAPH` | crates/hipfire-arch-deepseek4/src/deepseek4.rs, crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_HFQ4_WMMA` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_INDEXER_TOPK_SERIAL` | crates/rdna-compute/src/attention.rs |
| `HIPFIRE_DEEPSEEK4_INDEXER_WMMA` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_LOAD_DSPARK` | crates/hipfire-runtime/src/loader_api.rs |
| `HIPFIRE_DEEPSEEK4_LOAD_MTP` | crates/hipfire-arch-deepseek4/src/arch.rs, crates/hipfire-arch-deepseek4/src/deepseek4.rs |
| `HIPFIRE_DEEPSEEK4_MAX` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs |
| `HIPFIRE_DEEPSEEK4_MODEL` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs |
| `HIPFIRE_DEEPSEEK4_MOE` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_MOE_8W` | crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs, crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DEEPSEEK4_MOE_CND` | crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs, crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC` | crates/hipfire-arch-deepseek4/src/forward.rs, crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DEEPSEEK4_MOE_GROUPED` | crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE` | crates/hipfire-dispatch/src/families/moe.rs, crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W` | crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DEEPSEEK4_MOE_MMQLOAD` | crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs, crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DEEPSEEK4_MOE_N32` | crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs, crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DEEPSEEK4_MOE_NOSYNC` | crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs, crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DEEPSEEK4_MTP_ADDON` | crates/hipfire-arch-deepseek4/src/arch.rs, crates/hipfire-loader/src/carriers.rs |
| `HIPFIRE_DEEPSEEK4_MTP_HEAD_HC` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_POST_SCALE` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_PP_BATCH` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs |
| `HIPFIRE_DEEPSEEK4_PROMPT` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs, crates/hipfire-arch-deepseek4/examples/dspark_forward_smoke.rs |
| `HIPFIRE_DEEPSEEK4_Q8_4W` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_DEEPSEEK4_Q8_WMMA` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_DEEPSEEK4_REAP_KEEPMAP` | crates/hipfire-arch-deepseek4/examples/deepseek4_perplexity.rs, crates/hipfire-arch-deepseek4/src/deepseek4.rs |
| `HIPFIRE_DEEPSEEK4_ROUTE_SCALE` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_SEED` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_DEEPSEEK4_SKIP_FFN` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_SPEC_DECODE` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-loader/src/carriers.rs |
| `HIPFIRE_DEEPSEEK4_SPEC_K` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/examples/dspark_bench.rs |
| `HIPFIRE_DEEPSEEK4_TEMP` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/examples/dspark_bench.rs |
| `HIPFIRE_DEEPSEEK4_TOP_K` | crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs, crates/hipfire-arch-deepseek4/examples/dspark_bench.rs |
| `HIPFIRE_DEEPSEEK4_TOP_P` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs |
| `HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS` | crates/hipfire-arch-deepseek4/src/arch.rs, crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_WARMUP` | crates/hipfire-arch-deepseek4/examples/dspark_bench.rs |
| `HIPFIRE_DEEPSEEK4_WO_MULTIROW` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_DEEPSEEK4_WO_Q8_WMMA` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_DEFAULT_CHATML` | crates/hipfire-config/src/lib.rs |
| `HIPFIRE_DEMOTE_MQ6` | crates/hipfire-runtime/examples/hfq_splice_attn.rs |
| `HIPFIRE_DETECTED_ARCH` | scripts/_detect-gpu.sh, scripts/test-kernels.sh |
| `HIPFIRE_DETECTED_NAME` | scripts/_detect-gpu.sh |
| `HIPFIRE_DETECTED_VRAM_GB` | scripts/_detect-gpu.sh |
| `HIPFIRE_DETERMINISTIC` | autoresearch/ar/certify/serve_runner.py, crates/hipfire-runtime/examples/pp_parity_chatml.rs |
| `HIPFIRE_DEVICES` | crates/hipfire-runtime/src/config.rs, crates/hipfire-runtime/src/multi_gpu.rs |
| `HIPFIRE_DFLASH_CHAT` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_DFLASH_CKPT_RESUME` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_DFLASH_CTX_CAP` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_DFLASH_DRAFT` | crates/hipfire-runtime/src/config.rs, scripts/coherence-gate-dflash.sh |
| `HIPFIRE_DFLASH_FAST_SAMPLE` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_DFLASH_LOGIT_DUMP` | crates/hipfire-arch-qwen35/src/speculative.rs |
| `HIPFIRE_DFLASH_LOOP_BREAK` | crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS` | crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY` | crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX` | crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP` | crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER` | crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_DFLASH_LOOP_BREAK_TEMP` | crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_DFLASH_MODE` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/src/config.rs |
| `HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH` | crates/hipfire-arch-qwen35/src/speculative.rs |
| `HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD` | crates/hipfire-arch-qwen35/src/speculative.rs |
| `HIPFIRE_DFLASH_NGRAM_BLOCK` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/hipfire-config/src/lib.rs |
| `HIPFIRE_DFLASH_OFF` | scripts/serve-loop-gate.sh |
| `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` | crates/hipfire-arch-qwen35/src/speculative.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_DFLASH_REFERENCE` | scripts/dflash_ref_spec_test.py, scripts/dflash_spec_debug.py |
| `HIPFIRE_DFLASH_SEED_ORACLE` | crates/hipfire-arch-qwen35/src/speculative.rs, scripts/seed_oracle_collect.sh |
| `HIPFIRE_DFLASH_TARGET` | scripts/coherence-gate-dflash.sh |
| `HIPFIRE_DFLASH_TEMP_SPEC` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_DFLASH_TREE` | crates/hipfire-runtime/src/dflash_generic.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_DFLASH_WINDOW` | crates/hipfire-arch-qwen35/src/dflash_spec.rs, crates/hipfire-runtime/examples/dflash_spec_demo.rs, crates/hipfire-runtime/src/dflash.rs |
| `HIPFIRE_DFLASH_ZLAB_SAFETENSORS` | scripts/dflash_spec_debug.py |
| `HIPFIRE_DIR` | scripts/ab-dispatch-validation.sh, scripts/agentic-gate-jinja-tools.sh |
| `HIPFIRE_DIVERGENCE_PREFILL` | scripts/gfx906_logit_divergence.sh |
| `HIPFIRE_DIVERGENCE_TOL` | scripts/gfx906_logit_divergence.sh |
| `HIPFIRE_DN_REQUANT_PER_TOKEN` | crates/rdna-compute/src/norm.rs |
| `HIPFIRE_DN_STATE_EF` | autoresearch/ar/certify/serve_runner.py, crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_DOT2_GEMV` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_DOTS_OCR_BF16_RESIDUAL` | crates/hipfire-arch-dots-ocr/src/dots_ocr.rs |
| `HIPFIRE_DOTS_OCR_DUMP_DIR` | crates/hipfire-arch-dots-ocr/src/dots_ocr.rs, scripts/diff_dots_ocr_stages.py |
| `HIPFIRE_DOTS_OCR_TRACE` | crates/hipfire-arch-dots-ocr/src/dots_ocr.rs |
| `HIPFIRE_DPM_WARMUP_SECS` | crates/hipfire-arch-cohere2moe/examples/infer.rs, crates/hipfire-runtime/examples/bench_qwen35_mq4.rs |
| `HIPFIRE_DRAFT_F16` | crates/hipfire-runtime/src/config.rs, crates/hipfire-runtime/src/dflash.rs |
| `HIPFIRE_DRAFT_GEMM_DUMP` | crates/hipfire-runtime/src/config.rs, crates/hipfire-runtime/src/dflash.rs |
| `HIPFIRE_DRAFT_SUBPHASE` | crates/hipfire-runtime/src/config.rs, crates/hipfire-runtime/src/dflash.rs |
| `HIPFIRE_DSPARK_ADAPTIVE_BLOCK` | crates/hipfire-runtime/src/dspark_core.rs |
| `HIPFIRE_DSPARK_DEBUG` | crates/hipfire-runtime/src/dspark_core.rs |
| `HIPFIRE_DSPARK_HFQ4_WMMA` | crates/hipfire-runtime/src/dspark_core.rs |
| `HIPFIRE_DSPARK_KERNEL_PROFILE_POSITION` | crates/hipfire-runtime/src/dspark_core.rs |
| `HIPFIRE_DSPARK_PROFILE` | crates/hipfire-runtime/src/dspark_core.rs |
| `HIPFIRE_DSPARK_Q8_4W` | crates/hipfire-runtime/src/dspark_core.rs |
| `HIPFIRE_DSPARK_Q8_WMMA` | crates/hipfire-runtime/src/dspark_core.rs |
| `HIPFIRE_DSPARK_ZERO_CTX` | crates/hipfire-runtime/src/dspark_core.rs |
| `HIPFIRE_DTOH_DUMP` | crates/hip-bridge/src/ffi.rs |
| `HIPFIRE_DUMP_HIDDEN` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_DUMP_HIDDEN_POS` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_E8_DGPU_TWIN` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_E8_GFX12` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-dispatch/src/families/moe.rs |
| `HIPFIRE_E8_LDSX` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_E8_ROW_GATE` | crates/hipfire-runtime/examples/collect_e8_hessian_native.rs |
| `HIPFIRE_E8_SOA_EXPERTS` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_E8_STRIP` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_EMIT_TOKEN_IDS` | autoresearch/ar/certify/serve_runner.py, crates/hipfire-arch-cohere2moe/src/spec_emit.rs |
| `HIPFIRE_EP_DECODE_TIMING` | crates/hipfire-arch-deepseek4/src/forward.rs, crates/hipfire-arch-minimax/src/forward.rs |
| `HIPFIRE_EP_DUMP_IDX` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_EP_DUMP_POS` | crates/hipfire-arch-deepseek4/src/forward.rs |
| `HIPFIRE_EP_FAIL_RANK` | crates/hipfire-loader/src/lib.rs |
| `HIPFIRE_EP_KV_MODE` | crates/hipfire-runtime/examples/ep_decode_parity.rs |
| `HIPFIRE_EP_KV_SEQ` | crates/hipfire-runtime/examples/ep_decode_parity.rs |
| `HIPFIRE_EP_PEER_ALLREDUCE` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_EP_PEER_ALLREDUCE_DECODE` | crates/hipfire-runtime/src/ep.rs |
| `HIPFIRE_EP_PREFILL` | crates/hipfire-runtime/examples/ep_decode_parity.rs |
| `HIPFIRE_EP_PREFILL_TIMING` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_EP_PROMPT_REPEAT` | crates/hipfire-runtime/examples/ep_decode_parity.rs |
| `HIPFIRE_EP_SKIP_ALLREDUCE` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_EXPERIMENTAL_` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_FLASH_PREFILL` | crates/hipfire-dispatch/src/families/attention.rs |
| `HIPFIRE_FLASH_PREFILL_FIXED_HD` | crates/rdna-compute/src/attention.rs |
| `HIPFIRE_FLASH_PREFILL_PREFETCH_V` | crates/rdna-compute/src/attention.rs |
| `HIPFIRE_FLASH_PARTIALS_BATCH` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-runtime/src/config.rs |
| `HIPFIRE_FORCE_ANSWER_SECS` | scripts/test-qwen35-think-cap.sh |
| `HIPFIRE_FORCE_REBUILD` | crates/hipfire-cli/src/main.rs, scripts/install.sh |
| `HIPFIRE_FORCE_SPEC_GATE` | scripts/coherence-gate-dflash.sh |
| `HIPFIRE_FORCE_UNFUSED` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_FORWARD_LOWERED` | crates/hipfire-arch-deepseek4/src/forward.rs, crates/hipfire-arch-lfm2moe/src/forward.rs |
| `HIPFIRE_FORWARD_ORACLE` | crates/hipfire-dispatch/src/pipeline/superop.rs |
| `HIPFIRE_FP16` | crates/hipfire-runtime/examples/dump_logits_qwen35.rs, crates/hipfire-runtime/examples/test_hfq6_gemm.rs |
| `HIPFIRE_FP16_LAYER_MAX` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_FP16_LAYER_MIN` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_FP8_WMMA` | crates/rdna-compute/examples/test_gemm_hfp4g32_fp8.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_FUSED_GATE_UP_K1024` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_FUSED_GATE_UP_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_FUSE_QKV_BIAS` | crates/hipfire-dispatch/src/pipeline/steps.rs, crates/rdna-compute/examples/test_fused_qkv_bias_parity.rs |
| `HIPFIRE_FUSE_QKV_BIAS_DEBUG` | crates/hipfire-dispatch/src/pipeline/steps.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GATED_NORM_MQ_ROTATE` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GATE_KV_MODE` | scripts/coherence-gate-dflash.sh |
| `HIPFIRE_GATE_MODEL` | scripts/gates.sh |
| `HIPFIRE_GATE_STATE_QUANT` | scripts/coherence-gate-dflash.sh |
| `HIPFIRE_GATE_UP_BT` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GATE_UP_NOSYNC` | crates/rdna-compute/examples/bench_gate_up_nosync.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GATE_UP_VARIANT` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GATE_WORK_DIR` | scripts/gates.sh |
| `HIPFIRE_GCN5_WAVE64_HYBRID` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GDN_BENCH` | crates/rdna-compute/examples/gdn_chunk_parity.rs |
| `HIPFIRE_GDN_BLK` | crates/rdna-compute/src/norm.rs |
| `HIPFIRE_GDN_BLOCK_SIZE` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GDN_CHUNKED` | crates/rdna-compute/src/kernels.rs, crates/rdna-compute/src/norm.rs |
| `HIPFIRE_GDN_CHUNK_SIZE` | crates/rdna-compute/src/norm.rs |
| `HIPFIRE_GDN_COMPACT2` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_GDN_COMPACT2_SHAPE` | crates/rdna-compute/src/norm.rs |
| `HIPFIRE_GDN_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GDN_MIN_BLOCKS` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GDN_QK_HEAD_DIV` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GDN_TILE_ROWS` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GDN_WAVES_PER_BLOCK` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GEMM_DUMP` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GEMV_DP4A` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GEMV_PREFETCH` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GEMV_ROWS` | crates/rdna-compute/src/dispatch.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GEN` | crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs |
| `HIPFIRE_GEN_STEPS` | crates/hipfire-runtime/examples/oracle_xcheck.rs |
| `HIPFIRE_GFX1100_AWQ_NORM_DIRECT` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1100_ASYM3_Q8_PAIR` | crates/rdna-compute/src/attention.rs |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_DOT_REFORM` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_LANE0_HEADERS` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_PAIR` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_PAIR2` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_QUAD_PREFETCH` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_SETPRIO` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_STAGE_X32` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1100_ROUTER_W64` | crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_GFX1151_ATTENTION_TILE_DPP` | crates/rdna-compute/src/attention.rs |
| `HIPFIRE_GFX1151_ATTENTION_TILE_DPP_REDUCE` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GFX1151_CUMODE_MODULES` | crates/rdna-compute/src/compiler.rs |
| `HIPFIRE_GFX1151_DOWN_HYBRID_BUFFER` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_DOWN_ROW1_BUFFER` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_DOWN_ROW2_BUFFER` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_DOWN_ROW2_CLUSTERED` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_DOWN_ROW8` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_DOWN_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_ALL_BUFFER` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_HYBRID_BUFFER` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_K2048` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_LOW_VGPR` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_PAIRED_WAVES` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_PAIR_ALL_BUFFER` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_PAIR_BUFFER` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_PAIR_VGPR` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_PERSISTENT_RANK8` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_ROUTE_ALL_BUFFER` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_SPLIT` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GATE_UP_WAVE64` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_GDN_DPP` | crates/rdna-compute/src/norm.rs |
| `HIPFIRE_GFX1151_GDN_DPP_REDUCE` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GFX1151_GDN_R4X2` | crates/rdna-compute/src/norm.rs |
| `HIPFIRE_GFX1151_GDN_R8` | crates/rdna-compute/src/norm.rs |
| `HIPFIRE_GFX1151_LM_HEAD_ALL_BUFFER` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_LM_HEAD_BUFFER` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_LM_HEAD_CPOL` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_LM_HEAD_DOT2` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_LM_HEAD_HYBRID_BUFFER` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GFX1151_LM_HEAD_K2048` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GFX1151_LM_HEAD_X_BUFFER` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GFX1151_MOE_DOWN_HYBRID_BUFFER` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GFX1151_MOE_GATE_UP_HYBRID_BUFFER` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GFX1151_MOE_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_PM4_ENTRY_ACQUIRE` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_GFX1151_PM4_INITIATOR` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_GFX1151_PM4_INTERLEAVE` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_GFX1151_PM4_RESOURCE_LIMITS` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_GFX1151_QKVZA_ALL_BUFFER_CPOL` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_QKVZA_HYBRID_BUFFER` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GFX1151_QKVZA_K2048_HOIST` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_QKVZA_LDSX8_BUFFER` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_QKVZA_PAIR_BUFFER` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_QKVZA_R2` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_QKVZA_R2_BUFFER` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_QKVZA_R4_STREAM` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_QKVZA_WAVE64` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_QKVZA_WAVE64_SHARE_X` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_QKVZA_X_BUFFER_LARGE` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_QKV_ALL_BUFFER_CPOL` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_QKV_X_BUFFER` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_REDLINE_CU_COUNT` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_GFX1151_RESIDUAL_HYBRID_BUFFER` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GFX1151_RESIDUAL_K4096` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GFX1151_RESIDUAL_MULTIROW_R2` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_RESIDUAL_ROW1` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_RESIDUAL_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_RESIDUAL_WAVE64` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_DOWN` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_GATE_UP` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_LOADS` | crates/rdna-compute/src/gemm.rs, crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_QKVZA` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_RESIDUAL` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX1151_WEIGHT_BUFFER_SIGMOID` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GFX11_WEIGHT_GLOBAL_LOADS` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GFX11_WEIGHT_LOAD_POLICY` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GFX1201_ROUTER_W64` | crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_GFX12_WEIGHT_CPOL_AUX` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GFX12_WEIGHT_GLOBAL_LOADS` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GFX12_WEIGHT_LOAD_POLICY` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GFX942_GEMV_V2` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX942_GEMV_V3` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GFX942_LDS_GEMV` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_GFX942_MFMA_PREFILL` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_GFX942_RMSNORM_SPLIT` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_GPTQ_DAMPING` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_GPUS` | scripts/ab-dispatch-validation.sh |
| `HIPFIRE_GPU_0_BIG` | scripts/ab-dispatch-validation.sh |
| `HIPFIRE_GPU_LOCKFILE` | autoresearch/ar/swarm.py, scripts/container-gate.sh |
| `HIPFIRE_GPU_LOCK_OWNER` | scripts/gpu-lock.sh, scripts/pp-gate.sh |
| `HIPFIRE_GPU_TOPK` | crates/hipfire-runtime/examples/infer_qwen35.rs |
| `HIPFIRE_GQA_CHUNK` | crates/rdna-compute/src/attention.rs |
| `HIPFIRE_GQA_FUSED` | crates/hipfire-arch-qwen2/src/qwen2.rs, crates/hipfire-dispatch/src/families/kv_tier.rs |
| `HIPFIRE_GRAPH` | benchmarks/scripts/bench_pp_gfx906.sh, crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_GRAPH_MOE` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_GRAPH_PREFILL` | crates/hipfire-runtime/examples/bench_qwen35_mq4.rs |
| `HIPFIRE_HAVE_2_GPU` | crates/hipfire-arch-qwen35/tests/pp_parity.rs |
| `HIPFIRE_HFQ` | benchmarks/vision/run_bench.sh |
| `HIPFIRE_HFQ3_DP4A` | crates/hipfire-runtime/examples/verify_hfq3_batched.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_HFQ3_MMQ` | crates/hipfire-runtime/examples/bench_hfq3_mmq_sweep.rs, crates/hipfire-runtime/examples/verify_hfq3_batched.rs |
| `HIPFIRE_HFQ3_MMQ_LAYER_` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_HFQ3_MMQ_LAYER_MAX` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_HFQ3_MMQ_LAYER_MIN` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_HFQ4G128_MMQ` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_HFQ4G256_K2048` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_HFQ4G256_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_HFQ4_MMQ_GFX906_Y64` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_HFQ4_MMQ_RDNA2` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_HFQ6_EXPERT_BINS` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs |
| `HIPFIRE_HFQ6_REAL_K` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs |
| `HIPFIRE_HFQ6_REAL_LABEL` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs |
| `HIPFIRE_HFQ6_REAL_M` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs |
| `HIPFIRE_HFQ6_REAL_M_TOTAL` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs |
| `HIPFIRE_HFQ6_REAL_X_ROW_DIV` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs |
| `HIPFIRE_HF_BASE` | crates/hipfire-cli/src/main.rs |
| `HIPFIRE_HIPCC_EXTRA_FLAGS` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_HIP_WAIT` | crates/rdna-compute/src/dispatch.rs |
| `HIPFIRE_HOME` | crates/hipfire-config/src/lib.rs, crates/hipfire-registry/src/lib.rs |
| `HIPFIRE_HOST` | crates/hipfire-config/src/lib.rs |
| `HIPFIRE_HOST_TIMING` | crates/hipfire-runtime/examples/dflash_spec_demo.rs, scripts/ddtree_verify_profile.sh |
| `HIPFIRE_IDLE_TIMEOUT` | crates/hipfire-config/src/lib.rs |
| `HIPFIRE_IMAGE` | scripts/container-gate.sh |
| `HIPFIRE_JINJA_CHAT` | crates/hipfire-runtime/examples/daemon.rs, crates/hipfire-runtime/src/prompt_frame.rs |
| `HIPFIRE_JINJA_TOOLS_DRAFTER` | scripts/agentic-gate-jinja-tools.sh |
| `HIPFIRE_JINJA_TOOLS_MODEL` | scripts/agentic-gate-jinja-tools.sh |
| `HIPFIRE_KERNEL_CACHE` | crates/hipfire-cli/src/main.rs, crates/hipfire-tui/src/hipfire/dashboard.rs |
| `HIPFIRE_KLD_NGL` | crates/hipfire-runtime/examples/build_kld_ref.rs, crates/hipfire-runtime/examples/eval_gguf.rs |
| `HIPFIRE_KV` | crates/hipfire-runtime/examples/oracle_xcheck.rs |
| `HIPFIRE_KV_ADAPTIVE` | crates/hipfire-arch-qwen35/src/carrier.rs, crates/hipfire-config/src/lib.rs |
| `HIPFIRE_KV_MODE` | benchmarks/scripts/bench_pp_gfx906.sh, crates/hipfire-arch-qwen35/src/carrier.rs |
| `HIPFIRE_KV_PHYSICAL_CAP` | crates/hipfire-loader/src/carriers.rs |
| `HIPFIRE_KV_V` | crates/hipfire-arch-qwen35/src/carrier.rs, crates/hipfire-runtime/examples/eval_hipfire.rs |
| `HIPFIRE_LFM2_CAPTURE_POSTMIXER` | crates/hipfire-arch-lfm2moe/examples/dump_lfm2moe_hidden_states.rs, crates/hipfire-arch-lfm2moe/src/forward.rs |
| `HIPFIRE_LFM2_GRAPH` | crates/hipfire-arch-lfm2moe/examples/graph_parity_lfm2moe.rs, crates/hipfire-arch-lfm2moe/src/forward.rs |
| `HIPFIRE_LLOYD_FORCE_BASELINE` | crates/rdna-compute/examples/test_gemv_mq4g256_lloyd_tail.rs, crates/rdna-compute/examples/test_mq4g256_lloyd_fused_parity.rs |
| `HIPFIRE_LLOYD_GFX12` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-runtime/examples/eval_hipfire.rs |
| `HIPFIRE_LLOYD_K3` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_LLOYD_MB4` | crates/rdna-compute/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_LM_HEAD_F16` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-runtime/src/config.rs |
| `HIPFIRE_LM_HEAD_OVERWRITE` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_LM_HEAD_WMMA` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_LOCAL` | crates/hipfire-cli/src/main.rs, tests/e2e_run_reject.sh |
| `HIPFIRE_LOG` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_LOG_FORMAT` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_MAGIC` | crates/hipfire-runtime/examples/build_kld_ref.rs, crates/hipfire-runtime/examples/build_kld_ref_native.rs |
| `HIPFIRE_MAX_REQUEST_BYTES` | crates/hipfire-config/src/lib.rs |
| `HIPFIRE_MAX_TOTAL_THINK_TOKENS` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_MEMSET_DUMP` | crates/hip-bridge/src/ffi.rs |
| `HIPFIRE_META_MAX` | scripts/ddtree_meta_sweep.sh |
| `HIPFIRE_META_RUNS` | scripts/ddtree_meta_sweep.sh |
| `HIPFIRE_MINIMAX_BATCH_PREFILL` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_MINIMAX_CAPTURE_POSTATTN` | crates/hipfire-arch-minimax/src/forward.rs |
| `HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ` | crates/hipfire-arch-minimax/src/minimax.rs |
| `HIPFIRE_MINIMAX_EXPERT_` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MINIMAX_EXPERT_MQ2L` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MINIMAX_EXPERT_MQ3L` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MINIMAX_EXPERT_MQ6` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MINIMAX_GATE_OUT` | scripts/coherence-gate-minimax.sh |
| `HIPFIRE_MINIMAX_GRAPH` | crates/hipfire-arch-minimax/src/forward.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_MINIMAX_MODEL` | scripts/coherence-gate-minimax.sh |
| `HIPFIRE_MINIMAX_PROMOTE_MQ4` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MINIMAX_PROMOTE_MQ6` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MMQ` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_MMQ_DUMP` | crates/rdna-compute/examples/test_gfx906_mmq_realdata.rs |
| `HIPFIRE_MMQ_MIN_BATCH` | benchmarks/scripts/bench_dflash_27b_gfx906.sh, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_MMQ_SCREEN` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/channel_test_mmq.rs |
| `HIPFIRE_MMQ_SCREEN_THRESHOLD` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/channel_test_mmq.rs |
| `HIPFIRE_MODEL` | crates/hipfire-config/src/lib.rs, scripts/bench_humaneval_completion.sh |
| `HIPFIRE_MODELS_DIR` | autoresearch/ar/gate/run.py, benchmarks/scripts/bench_dflash_27b_gfx906.sh |
| `HIPFIRE_MODEL_PATH` | autoresearch/ar/census.py |
| `HIPFIRE_MODEL_STORE` | scripts/baseline_quant_smoke.sh |
| `HIPFIRE_MOE_AWQ` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_MOE_COMBINE_NEXT_RMS` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_MOE_COMBINE_NEXT_RMS_RENORM` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_MOE_COMBINE_RMSNORM_MQ_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_DOWN_COMBINE_VEC4` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_MOE_DOWN_CPOL` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_MOE_DOWN_FUSED` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_DOWN_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_DOWN_LAST_COMBINE` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_MOE_DOWN_MQ5` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MOE_DOWN_MQ6` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MOE_DOWN_ROW2_CLUSTERED` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_DOWN_ROW2_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_DOWN_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_MOE_EXPERTS_MQ5` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MOE_EXPERTS_MQ6` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MOE_EXPERT_STATS` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_MOE_EXPERT_STATS_OUT` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-runtime/examples/eval_hipfire.rs |
| `HIPFIRE_MOE_GATE_UP_CPOL` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_MOE_GATE_UP_FUSED` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_GATE_UP_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_GATE_UP_LOW_VGPR` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_GATE_UP_PAIR_VGPR` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_GATE_UP_RANK_INTERLEAVE` | crates/rdna-compute/src/gemv.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_GATE_UP_TIGHT_GRID` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_MOE_GATE_UP_WG2` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_MOE_GATE_UP_WG_WAVES` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_GRADED` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MOE_GROUPED_4W` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_GROUPED_GEMM` | crates/hipfire-arch-cohere2moe/src/forward.rs, crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_MOE_GROUPED_I8` | crates/hipfire-dispatch/src/families/moe.rs, crates/rdna-compute/examples/test_moe_grouped_mmq_gfx1151.rs |
| `HIPFIRE_MOE_GROUPED_I8_K4` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_MOE_GROUPED_I8_K4_GFX12` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_MOE_GROUPED_I8_K8` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_MOE_GROUPED_M2` | crates/rdna-compute/examples/test_moe_grouped_wmma_m2.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_MOE_HFQ6_I8` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_MOE_HFQ6_V2` | crates/rdna-compute/examples/test_moe_grouped_wmma_hfq6_v2.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_MOE_HOT_FRAC` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MOE_MQ6_ADMIT` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_MOE_PAIRED_WAVES` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_PARO_I8` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_MOE_PARO_I8_K8` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_MOE_PREFILL_TRACE` | crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_MOE_PROJECTION` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_PROJECTION_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_MOE_ROUTER_SHARED_FUSE` | crates/hipfire-dispatch/src/pipeline/mod.rs |
| `HIPFIRE_MOE_TIER_MAP` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_MQ3_MB4` | crates/rdna-compute/examples/test_gemm_hfq3g256_wmma.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_MTP_DEVICE_TOKEN_CHAIN` | crates/hipfire-arch-qwen35/src/mtp_spec.rs |
| `HIPFIRE_MTP_GPU_ACCEPT` | crates/hipfire-arch-qwen35/src/mtp_spec.rs |
| `HIPFIRE_MTP_HEAD_LMHEAD_WMMA` | crates/hipfire-arch-qwen35/src/mtp_head.rs |
| `HIPFIRE_MTP_K` | crates/hipfire-config/src/lib.rs, crates/hipfire-loader/src/carriers.rs |
| `HIPFIRE_MTP_MODE` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_MTP_PREFIX_CACHE` | scripts/serve_harness.py |
| `HIPFIRE_MTP_PROPOSAL_GRAPH` | crates/hipfire-arch-qwen35/src/mtp_spec.rs |
| `HIPFIRE_MTP_P_MIN` | crates/hipfire-arch-qwen35/src/mtp_spec.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_MTP_Q8_VERIFY_WMMA` | crates/hipfire-arch-qwen35/src/mtp_spec.rs |
| `HIPFIRE_MTP_SAMPLED` | crates/hipfire-runtime/examples/daemon.rs, scripts/serve_harness.py |
| `HIPFIRE_MTP_SMOKE_HEAD` | crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs |
| `HIPFIRE_MTP_SMOKE_TRUNK` | crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs |
| `HIPFIRE_MTP_SNAPSHOT_OVERLAP` | crates/hipfire-arch-qwen35/src/mtp_spec.rs |
| `HIPFIRE_MTP_TAPE_REPLAY` | crates/hipfire-arch-qwen35/src/mtp_spec.rs |
| `HIPFIRE_MTP_VERIFY_DECOUPLE` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_MW16` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_NGRAM_DRAFT` | crates/hipfire-arch-qwen2/src/spec_impl.rs, crates/hipfire-loader/src/carriers.rs |
| `HIPFIRE_NGRAM_DRAFT_K` | crates/hipfire-loader/src/spec_build.rs, crates/hipfire-runtime/src/loader_api.rs |
| `HIPFIRE_NGRAM_LOOP_THRESHOLD` | crates/hipfire-runtime/examples/daemon.rs, crates/hipfire-runtime/src/config.rs |
| `HIPFIRE_NGRAM_MIN_COUNT` | crates/hipfire-loader/src/spec_build.rs, crates/hipfire-runtime/src/loader_api.rs |
| `HIPFIRE_NGRAM_THRESHOLD` | crates/hipfire-runtime/src/arch.rs |
| `HIPFIRE_NGRAM_WINDOW` | crates/hipfire-runtime/examples/daemon.rs, crates/hipfire-runtime/src/arch.rs |
| `HIPFIRE_NORMALIZE_PROMPT` | crates/hipfire-config/src/lib.rs, crates/hipfire-runtime/examples/build_kld_ref_native.rs |
| `HIPFIRE_NO_REGISTRY_FETCH` | crates/hipfire-registry/src/lib.rs |
| `HIPFIRE_NO_SPILL` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_ORACLE_MAX` | scripts/seed_oracle_collect.sh |
| `HIPFIRE_PARITY_MAX_TOK` | scripts/forward-lowered-parity.sh |
| `HIPFIRE_PARITY_OUT` | scripts/forward-lowered-parity.sh |
| `HIPFIRE_PARO_BATCHED` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_PATH_C_OUT` | scripts/path-c-smoke.sh |
| `HIPFIRE_PFLASH_DEBUG` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_PFLASH_DRAFTER` | scripts/pflash-gate.sh |
| `HIPFIRE_PFLASH_DRAFTER_KV` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PFLASH_SCORE_LAYER` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PFLASH_TARGET` | scripts/pflash-gate.sh |
| `HIPFIRE_PORT` | crates/hipfire-config/src/lib.rs |
| `HIPFIRE_POST_LATCH_ANSWER_TOKENS` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_PP_DFLASH` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_PP_GATE_HETEROGENEOUS` | scripts/pp-gate.sh |
| `HIPFIRE_PP_GATE_INCLUDE_IGPU` | scripts/pp-gate.sh |
| `HIPFIRE_PP_GATE_MODEL` | scripts/pp-gate.sh |
| `HIPFIRE_PP_GATE_REQUIRE_SYSFS` | scripts/pp-gate.sh |
| `HIPFIRE_PP_GATE_TEST_NO_SYSFS` | scripts/pp-gate.sh |
| `HIPFIRE_PP_LAYERS` | crates/hipfire-loader/src/carriers.rs |
| `HIPFIRE_PP_PARITY_MODEL` | crates/hipfire-arch-qwen35/tests/pp_parity.rs |
| `HIPFIRE_PP_PFLASH` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_PREFILL_` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PREFILL_ALPHA` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PREFILL_BATCHED` | crates/hipfire-arch-qwen35/src/mtp_spec.rs, crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_PREFILL_BLOCK` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PREFILL_CHUNK` | crates/hipfire-runtime/examples/ep_decode_parity.rs |
| `HIPFIRE_PREFILL_COMPRESSION` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PREFILL_DRAFTER` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PREFILL_KEEP_RATIO` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PREFILL_MAX_BATCH` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_PREFILL_MIN_KEEP` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PREFILL_PROFILE` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PREFILL_RECENT` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PREFILL_REUSE_PBS` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-runtime/examples/eval_hipfire.rs |
| `HIPFIRE_PREFILL_SINK` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PREFILL_SPARSE_THRESHOLD` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PREFILL_THRESHOLD` | crates/hipfire-arch-qwen35/src/pflash.rs |
| `HIPFIRE_PROFILE` | crates/hipfire-atlas/src/profile_report.rs, crates/hipfire-runtime/examples/bench_qwen35_mq4.rs |
| `HIPFIRE_PROFILE_CYCLES` | crates/hipfire-runtime/examples/dflash_spec_demo.rs, crates/hipfire-runtime/examples/mtp_only_demo.rs |
| `HIPFIRE_PROFILE_DECODE` | crates/hipfire-runtime/examples/bench_qwen35_mq4.rs, scripts/kernel_atlas.py |
| `HIPFIRE_PROFILE_MAX` | scripts/ddtree_verify_profile.sh |
| `HIPFIRE_PROFILE_RUNS` | scripts/ddtree_verify_profile.sh |
| `HIPFIRE_PROMPT_CACHE_CAP` | crates/hipfire-loader/src/lib.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_PROMPT_CACHE_UNBOUNDED` | crates/hipfire-loader/src/lib.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_PROMPT_HEAT_JSON` | crates/hipfire-runtime/src/config.rs, crates/hipfire-runtime/src/tokenizer.rs |
| `HIPFIRE_PROMPT_HEAT_LIMIT` | crates/hipfire-runtime/src/config.rs, crates/hipfire-runtime/src/tokenizer.rs |
| `HIPFIRE_PROMPT_TOKEN_HEAT` | crates/hipfire-runtime/examples/daemon.rs, crates/hipfire-runtime/examples/dflash_spec_demo.rs |
| `HIPFIRE_Q8_BATCHED_LEGACY` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_Q8_FLASH_TILE` | crates/rdna-compute/src/attention.rs |
| `HIPFIRE_Q8_PREFILL_WMMA` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_QA_KV_MODES` | crates/hipfire-runtime/examples/test_inferenceQA.rs |
| `HIPFIRE_QKVZA_BLOCK_SIZE` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_QKVZA_CPOL` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_QKVZA_KERNEL_NAME` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_QKVZA_MIN_BLOCKS_PER_CU` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_QKVZA_SCALAR_PREP` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_QKVZA_SPLIT_TAIL` | crates/rdna-compute/src/feature_flags.rs, scripts/bench_qwen36_qkvza_split_tail_ab.sh |
| `HIPFIRE_QKVZA_WAVES_PER_BLOCK` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_QKV_KERNEL_NAME` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_QKV_WITH_BIAS` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_QUANTIZE` | scripts/stage_models.sh |
| `HIPFIRE_QUANT_DIAG_PATH` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_QUANT_THREADS` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_QWEN2_VERIFY_SEQ` | crates/hipfire-arch-dots-ocr/src/spec_impl.rs, crates/hipfire-arch-qwen2/src/spec_impl.rs |
| `HIPFIRE_QWEN35_DSPARK_CONF_THRESHOLD` | crates/hipfire-loader/src/lib.rs |
| `HIPFIRE_QWEN35_FA_EPILOGUE_FUSE` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_QWEN35_FA_PREP_FUSE` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_QWEN35_FA_PREP_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_QWEN35_FINITE_TRACE` | crates/hipfire-arch-qwen35/src/qwen35.rs |
| `HIPFIRE_QWEN35_GRAMMAR` | crates/hipfire-runtime/examples/daemon.rs, crates/hipfire-runtime/src/prompt_frame.rs |
| `HIPFIRE_QWEN35_MTP` | crates/hipfire-loader/src/lib.rs, crates/hipfire-loader/src/spec_build.rs |
| `HIPFIRE_QWEN35_MTP_K` | crates/hipfire-loader/src/spec_build.rs |
| `HIPFIRE_QWEN35_NGRAM_LEN_MIN` | crates/hipfire-arch-qwen35/src/grammar.rs |
| `HIPFIRE_QWEN35_NGRAM_MIN_REPEATS` | crates/hipfire-arch-qwen35/src/grammar.rs |
| `HIPFIRE_QWEN3_BENCH_MODE` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs |
| `HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs, crates/hipfire-loader/src/carriers.rs |
| `HIPFIRE_QWEN3_MAX` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs |
| `HIPFIRE_QWEN3_MODEL` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs |
| `HIPFIRE_QWEN3_NGRAM_K` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs |
| `HIPFIRE_QWEN3_PROMPT` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs |
| `HIPFIRE_QWEN3_RAW` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs |
| `HIPFIRE_QWEN3_TEMP` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs |
| `HIPFIRE_QWEN3_TOP_K` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs |
| `HIPFIRE_QWEN3_TOP_P` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs |
| `HIPFIRE_QWEN3_WARMUP` | crates/hipfire-arch-llama/examples/qwen3_dspark_bench.rs |
| `HIPFIRE_QWEN_CACHE_TRACE` | crates/hipfire-runtime/examples/daemon.rs, scripts/test-qwen35-abort-resume.sh |
| `HIPFIRE_QWEN_MOE_FINAL_NORM_RAW` | scripts/test_pr228_spiral_check.sh |
| `HIPFIRE_QWEN_MTP` | crates/hipfire-runtime/examples/daemon.rs, scripts/serve_harness.py |
| `HIPFIRE_QWEN_PROMPT_CACHE` | crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_RDNA2_VARIANT` | crates/hipfire-cli/src/main.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_LM_HEAD_K2048` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_MOE_GATE_UP_K2048` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_2WAVE` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_HOIST_X32` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_K2048` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_LDSX8` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_REDUCE_CHAIN` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_QKVZA_WAVEPACK4` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_QKV_WAVE64` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_RESIDUAL_K2048` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_RESIDUAL_STAGE_X32` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_SIGMOID_BUFFER` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_SIGMOID_HOIST_X16` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_RDNA3_HFQ4_SIGMOID_K512` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_RDNA3_HFQ4_SIGMOID_ROWS4` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_HFQ4_SIGMOID_TIGHT_GRID` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_MOE_GATE_UP_K2048` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_MOE_GATE_UP_ROUTE_BUFFER` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_MOE_GATE_UP_X_BUFFER` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_QKVZA_HOIST_X32` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_QKVZA_K2048` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_QKVZA_LDSX8` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_QKVZA_PAIR_BUFFER` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_QKVZA_R2` | crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_RDNA3_QKVZA_REDUCE_CHAIN` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_QKVZA_X_BUFFER` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_QKV_K2048` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_QKV_X_BUFFER` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_RESIDUAL_BUFFER_CONSUME` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_RESIDUAL_K2048` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_RESIDUAL_MATRIX_RSRC` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_RESIDUAL_STAGE_X32` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_RMSNORM_SIGN_CONST` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_RMSNORM_SIGN_LDS` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_RMSNORM_SPLIT` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_RMSNORM_VECSUM` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_RMSNORM_VECSUM_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_RMSNORM_WAVEGRID` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_RDNA3_SIGMOID_HOIST_X16` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_SIGMOID_K512` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_SIGMOID_M2048` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RDNA3_SIGMOID_ROWS4` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_REAP_PLAN` | crates/hipfire-arch-deepseek4/src/deepseek4.rs, crates/hipfire-arch-lfm2moe/src/config.rs |
| `HIPFIRE_REGISTRY_URL` | crates/hipfire-registry/src/lib.rs |
| `HIPFIRE_REMOTE` | scripts/mi300x_bootstrap.sh |
| `HIPFIRE_REPLAY_BACKEND` | crates/hipfire-cli/src/main.rs, crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPLAY_GRAPH` | crates/hipfire-arch-qwen35/src/speculative.rs |
| `HIPFIRE_REPLAY_MANUAL_CAPTURE` | crates/rdna-compute/src/replay.rs, scripts/redline_daemon_harness.py |
| `HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPLAY_PM4_DYNAMIC_GRID` | crates/rdna-compute/src/dispatch.rs |
| `HIPFIRE_REPLAY_PM4_GCR_TRIM` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPLAY_PM4_MAX_PARALLEL_PHASES` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WIDTH` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WORKGROUPS` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPLAY_PM4_NATIVE_PHASES` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPLAY_PM4_QUEUES` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPLAY_PM4_STATEFUL` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPLAY_PM4_WAIT_POLICY` | crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPLAY_ROUTE_PROOF_LOG` | crates/hipfire-config/src/lib.rs (`diagnostic.replay.route_proof_log`), crates/rdna-compute/src/replay.rs, tools/redline/product_bench.py, scripts/serve_harness.py |
| `HIPFIRE_REPLAY_TRANSPORT` | crates/hipfire-cli/src/main.rs, crates/rdna-compute/src/replay.rs |
| `HIPFIRE_REPO` | scripts/quantize-dspark.sh |
| `HIPFIRE_RESIDUAL_CPOL` | crates/rdna-compute/src/gemv.rs |
| `HIPFIRE_RESIDUAL_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RESIDUAL_MULTIROW_R2_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RESIDUAL_MULTIROW_R4_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RESIDUAL_MULTIROW_R8_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_RMSNORM_MQ_TIGHT_LDS` | crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_ROCBLAS_ALL_ARCHS` | crates/rdna-compute/src/dispatch.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_ROCBLAS_MIN_BATCH` | crates/rdna-compute/src/dispatch.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_ROCBLAS_OFF` | crates/rdna-compute/src/dispatch.rs, crates/rdna-compute/src/feature_flags.rs |
| `HIPFIRE_ROCPROF_CSV` | crates/hipfire-runtime/examples/bench_qwen35_mq4.rs, scripts/coverage-audit.py |
| `HIPFIRE_ROPE_HALFSPLIT` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_ROPE_INTERLEAVED_LEGACY` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/norm.rs |
| `HIPFIRE_ROUTER_EXACT_KERNEL` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_ROUTER_SHARED_SILU_MQ_ROTATE` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_SAMPLE_COMPARE` | crates/hipfire-runtime/examples/infer_qwen35.rs, crates/hipfire-runtime/src/llama.rs |
| `HIPFIRE_SAMPLE_FAST` | crates/rdna-compute/examples/sample_parallel_stable_parity.rs, crates/rdna-compute/src/sampling.rs |
| `HIPFIRE_SAMPLE_PARALLEL` | crates/rdna-compute/examples/sample_accept_parity.rs, crates/rdna-compute/src/sampling.rs |
| `HIPFIRE_SERVE_GATE_DFLASH` | scripts/serve-multiturn-gate.sh |
| `HIPFIRE_SERVE_GATE_OUT` | scripts/serve-multiturn-gate.sh |
| `HIPFIRE_SERVE_GATE_PORT` | scripts/serve-loop-gate.sh |
| `HIPFIRE_SERVE_LOG` | scripts/test-qwen35-abort-resume.sh, scripts/test-qwen35-think-cap.sh |
| `HIPFIRE_SERVE_MAX_QUEUE` | crates/hipfire-config/src/lib.rs |
| `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS` | crates/hipfire-config/src/lib.rs |
| `HIPFIRE_SKIP_AGENTIC_GATE` | scripts/agentic-gate.sh |
| `HIPFIRE_SKIP_BUILD` | scripts/container-gate.sh |
| `HIPFIRE_SKIP_LLAMA_COMMIT_CHECK` | crates/hipfire-runtime/src/eval_common.rs |
| `HIPFIRE_SMOKE_KV` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-runtime/examples/a3b_smoke_forward.rs |
| `HIPFIRE_SMOKE_KV_SEQ` | crates/hipfire-runtime/examples/a3b_smoke_forward.rs |
| `HIPFIRE_SMOKE_MODE` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-runtime/examples/a3b_smoke_forward.rs |
| `HIPFIRE_SMOKE_PROMPT` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-runtime/examples/a3b_smoke_forward.rs |
| `HIPFIRE_SMOKE_STEPS` | crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-runtime/examples/a3b_smoke_forward.rs |
| `HIPFIRE_SPECULATION` | crates/hipfire-config/src/lib.rs |
| `HIPFIRE_SPEC_PHASES` | crates/hipfire-arch-qwen35/src/speculative.rs |
| `HIPFIRE_SPILL_DIR` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_SWEEP_MAX` | scripts/ddtree_budget_sweep.sh |
| `HIPFIRE_SWEEP_OUT` | scripts/mq3-mq2-sweep.sh, scripts/spec_decode_genre_sweep.sh |
| `HIPFIRE_SWEEP_PROMPTS_DIR` | scripts/mq3-mq2-sweep.sh |
| `HIPFIRE_SWEEP_RUNS` | scripts/ddtree_budget_sweep.sh |
| `HIPFIRE_TARGET_ARCH` | crates/rdna-compute/src/dispatch.rs, scripts/kernel_atlas.py |
| `HIPFIRE_TEST_MODEL` | scripts/test-qwen35-abort-resume.sh, scripts/test-qwen35-think-cap.sh |
| `HIPFIRE_THINK_CONTINUATION` | crates/hipfire-arch-qwen35/src/spec_emit.rs, crates/hipfire-runtime/examples/daemon.rs |
| `HIPFIRE_TIER_RATIO` | crates/hipfire-quantize/src/main.rs |
| `HIPFIRE_TP_BENCH_ITERS` | crates/hip-bridge/examples/rccl_smoke.rs, crates/hipfire-runtime/examples/tp_allreduce_smoke.rs |
| `HIPFIRE_TP_BENCH_N` | crates/hip-bridge/examples/rccl_smoke.rs, crates/hipfire-runtime/examples/tp_allreduce_smoke.rs |
| `HIPFIRE_TP_BENCH_WARMUP` | crates/hipfire-runtime/examples/tp_allreduce_smoke.rs |
| `HIPFIRE_TP_EXPERT_ASSIGN` | crates/hipfire-runtime/src/tp_shard.rs |
| `HIPFIRE_TP_USE_RCCL` | crates/hipfire-runtime/src/config.rs, crates/hipfire-runtime/src/multi_gpu.rs |
| `HIPFIRE_TUI_BIN` | crates/hipfire-cli/src/main.rs |
| `HIPFIRE_UNIFORM_GATE_UP` | crates/hipfire-runtime/examples/hfq_splice_attn.rs |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | crates/hipfire-runtime/src/config.rs, crates/hipfire-runtime/src/multi_gpu.rs |
| `HIPFIRE_VERIFY_GRAPH` | crates/hipfire-arch-qwen35/src/mtp_probe.rs, crates/hipfire-arch-qwen35/src/speculative.rs |
| `HIPFIRE_VERIFY_GRAPH_TIMING` | crates/hipfire-arch-qwen35/src/speculative.rs |
| `HIPFIRE_VERIFY_GRAPH_TREE` | crates/hipfire-arch-qwen35/src/speculative.rs, scripts/tree_graph_bench.sh |
| `HIPFIRE_VERSION` | crates/hipfire-runtime/examples/build_kld_ref.rs, crates/hipfire-runtime/examples/build_kld_ref_native.rs |
| `HIPFIRE_VL_DUMP_DIR` | crates/hipfire-runtime/examples/infer.rs |
| `HIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_WEIGHT_CACHE_FLAT_GEMV` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_WEIGHT_CPOL_AUX` | crates/rdna-compute/src/kernels.rs |
| `HIPFIRE_WMMA_FA` | benchmarks/results/wmma-fa-probe-gfx1100.sh, benchmarks/results/wmma-fa-probe.sh |
| `HIPFIRE_WMMA_FA_MIN_BATCH` | crates/rdna-compute/src/attention.rs |
| `HIPFIRE_WO_MMQ` | crates/rdna-compute/src/feature_flags.rs, crates/rdna-compute/src/gemm.rs |
| `HIPFIRE_WO_WMMA_VARIANT` | crates/hipfire-runtime/examples/test_wmma_correctness.rs, crates/rdna-compute/src/feature_flags.rs |


---

## Maintenance

```bash
# Independently verify concrete Rust HIPFIRE_* reads against table rows.
scripts/regen-env-vars-doc.sh

# Shipped env/docs coverage and production-ownership check.
python3 scripts/check-env-docs.py
```

Also match `std::env::var_os("HIPFIRE_…")` (e.g. `HIPFIRE_KERNEL_CACHE` in `compiler.rs`) — token scan catches the name regardless of `var` vs `var_os`.

When adding a user-facing knob:

1. Prefer a typed field + validation in `crates/hipfire-config/src/lib.rs` ([`CONFIG.md`](CONFIG.md)).
2. Add the env name to product docs only if operators must set it outside config.
3. Re-scan so the generated inventory stays complete.
4. Do not document unearned or widened LFM defaults here (multi-cohort, path/extension selection of `.mq4`, automatic runtime default for non-`.mq4r`, or generic default-on beyond the exact sealed [`admissions.yml`](admissions.yml) evidence row). That LFM row is registry evidence without current automatic runtime wiring; only `mq4r_redline_default` auto-selects (`.mq4r` + exact GPU arch + pp=tp=1). Planned broader admissions may not be documented as shipped.

**Last inventory verification:** 2026-07-29.

## Batched attention (SP1)

| Variable | Default | Meaning |
|---|---|---|
| `HIPFIRE_ATTN_TILE_SIZE` | `128` | Tile size for the batched attention tile+reduce path. Must be a positive multiple of 32; anything else falls back to 128. Resolved once via `Gpu::attn_tile_size()`. **Raising it is safe; lowering it increases `max_tiles` and therefore the `partials` bytes per query row, which can exceed buffers sized elsewhere against the 128 default.** |
| `HIPFIRE_VRAM_BUDGET_BYTES` | 32 GiB | Deployment-target VRAM ceiling used by the SP1 benchmark harnesses' preflight. Read by `examples/`, not by production code. |
| `HIPFIRE_MEM_CAP` | `24G` | Read by `scripts/run-bounded.sh`, not by the binaries: cgroup `MemoryMax` for a gated run. Exit 137 means the cap fired — shrink the configuration rather than raising it. |
