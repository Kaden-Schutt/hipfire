# Configuration

**Owner:** daemon / user config keys (`docs/INDEX.md`).
**Machine sources:**

- Native schema + resolution: `crates/hipfire-config/src/lib.rs`
- Runtime env snapshot (daemon): `crates/hipfire-runtime/src/config.rs` → `RuntimeConfig`

**Last checked:** 2026-07-20.

Persistent stores under `~/.hipfire/`:

1. **Global** — sparse typed `config.toml`; missing keys inherit schema defaults.
2. **Per-model overlay (primary)** — `models.toml`: aliases, local paths,
   registry identities, and sparse per-model overrides.
3. **Migration inputs** — `config.json`, `models.json`, and
   `per_model_config.json`. Migration writes TOML and preserves the JSON files
   for rollback.

Edit interactively: `hipfire config` or `hipfire config <tag>`. Non-interactive: `hipfire config set <key> <value>` / `hipfire config <tag> set <key> <value>`.

**Precedence (operator view):** one-shot CLI values **>** compatible process env
**>** per-model override **>** global TOML **>** registry card
(`recommended_settings`) **>** built-in defaults. `hipfire config explain`
prints the winning source and shadowed candidates. Direct daemon invocation
still reads only its JSON request and process env.

This page is the normative **key/default/enum** table. Procedures for CASK profiles, PFlash bypass reasons, and multi-GPU topology live in linked owners — not duplicated matrices here.

---

## Generation / sampling

| Key | Default | Validated range / enum | Notes |
|---|---|---|---|
| `temperature` | `0.30` | number 0.0–2.0 | Stored default only; see send path below. |
| `top_p` | `0.80` | number (0, 1] | Stored default only; see send path below. |
| `repeat_penalty` | `1.05` | number 1.0–3.0 | Kept low; higher values harm some MQ4 greedy paths (source comment). Stored default only; see send path below. |
| `max_tokens` | `4096` | int 1–131072 | Per-turn generation cap for run / OpenAI fallback. |
| `max_seq` | `32768` | int 512–524288 | KV logical capacity at load. |
| `thinking` | `"on"` | `on` \| `off` | Whether visible `<think>` is kept/stripped in client paths. |
| `thinking_budget` | `"med"` | `low` \| `med` \| `high` \| `xhigh` \| `max` \| `uncapped` | Named preset → effective think cap (below). |
| `max_think_tokens` | *(absent)* | int 0–32768 when set | Optional raw override. If unset, preset drives. `0` = unlimited. |
| `max_total_think_tokens` | `0` | int 0–1000000 | Cross-reopen total `<think>` budget; `0` = off. |

**Thinking budget map** (`hipfire-config` schema, lowered by `hipfire-cli`):

| Preset | Tokens |
|---|---:|
| `low` | 512 |
| `med` | 2048 |
| `high` | 8192 |
| `xhigh` | 24576 |
| `max` | 32768 |
| `uncapped` | 0 (unlimited) |

**Effective sampling send order** (`request_f64` / `run`): explicit CLI flags **>** per-model overlay **>** registry `recommended_settings` **>** daemon/HFQ/arch fallback. Global `temperature` / `top_p` / `repeat_penalty` changed only in global `config.toml` are deliberately not transmitted on the current run/serve path; if a global value shadows a registry recommendation, the registry value is recovered for the request. Registry entry `sampling` blocks are metadata only today — the resolver reads `recommended_settings`; see [`MODELS.md`](MODELS.md).

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

Env projection: `HIPFIRE_KV_MODE` (see [`env-vars.md`](env-vars.md)).

---

## Attention

| Key | Default | Values |
|---|---|---|
| `flash_mode` | `"auto"` | `auto` \| `always` \| `never` |

Projected to `HIPFIRE_ATTN_FLASH` when unset in the environment.

---

## Speculative decode

Only **one** mechanism runs. Canonical selector:

| Key | Default | Values |
|---|---|---|
| `speculation` | `"auto"` | `off` \| `auto` \| `ngram` \| `dflash` \| `mtp` \| `dspark` |

- **`off`** — AR only.
- **`auto`** — cascade by availability / eligibility; legacy mode knobs filter each mechanism. Under auto, DSpark can win over in-trunk MTP when a DSpark sidecar is present (CLI comments).
- Forced mechanism names bypass heuristics; missing prerequisites fall back to AR with a warning.

Env: `HIPFIRE_SPECULATION`. CLI: `--spec`.

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

Legacy env still wins at top of ladder for some knobs (`HIPFIRE_NGRAM_DRAFT*`, `HIPFIRE_DFLASH_*`, DeepSeek MTP/DSpark names) — full list in [`env-vars.md`](env-vars.md).

---

## MMQ screening

| Key | Default | Values / range |
|---|---|---|
| `mmq_screen` | `"auto"` | `off` \| `on` \| `auto` |
| `mmq_screen_threshold` | `0.10` | number (0, 1] |

MMQ itself is activated via env (`HIPFIRE_MMQ` / `HIPFIRE_WO_MMQ`); screening only matters when MMQ is on. Daemon arch-gates the sweep (RDNA3/3.5 family in source comments). Legacy boolean `true`/`false` migrates to `on`/`off` on load.

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
| `cask_auto_attach` | `true` | bool |

Ops escape: `HIPFIRE_CASK_OFF=1`, `HIPFIRE_FORCE_A3B_EVICTION=1` (env doc).

Sidecar generation: `hipfire sidecar-gen` — [`CLI.md`](CLI.md).

---

## Prompt processing

| Key | Default | Values |
|---|---|---|
| `prompt_normalize` | `true` | bool |

Collapse `\n{3,}` → `\n\n` at engine entry. Env: `HIPFIRE_NORMALIZE_PROMPT` (`0`/`false`/`off`/`no` disable in runtime config).

---

## PFlash speculative prefill (experimental)

| Key | Default | Range / values |
|---|---|---|
| `prefill_compression` | `"off"` | `off` \| `auto` \| `always` |
| `prefill_threshold` | `32768` | int 0–524288 |
| `prefill_keep_ratio` | `0.05` | (0, 1] |
| `prefill_alpha` | `0.85` | [0, 1] |
| `prefill_min_keep` | `2048` | int 0–524288 |
| `prefill_sink` | `256` | int 0–65536 |
| `prefill_recent` | `1024` | int 0–65536 |
| `prefill_block` | `128` | int 1–4096 |
| `prefill_drafter` | `""` | path |
| `prefill_drafter_device` | `-1` | int −1–15 (`-1` = same device as target) |
| `prefill_profile` | `false` | bool |
| `prefill_sparse_threshold` | `32768` | int 0–524288 |

Off by default. Matching `HIPFIRE_PREFILL_*` env names exist for research overrides. Detailed bypass reasons and serve wiring: [`SERVE.md`](SERVE.md) / runtime PFlash module — not restated here.

---

## Server / serve admission

| Key | Default | Range |
|---|---|---|
| `host` | `"0.0.0.0"` | non-empty hostname/IP, no whitespace, ≤255 |
| `port` | `11435` | int 1–65535 |
| `idle_timeout` | `300` | int 0–86400 seconds (`0` = never unload) |
| `default_model` | `"qwen3.5:9b"` | non-empty tag/path string |
| `max_request_bytes` | `67108864` (64 MiB) | int 4096–4GiB |
| `serve_max_queue` | `64` | int 0–100000 (`0` = uncapped depth) |
| `serve_queue_timeout_ms` | `30000` | int 0–3600000 (`0` = no wait timeout) |
| `experimental_budget_alert` | `false` | bool |

Serve HTTP surface: [`SERVE.md`](SERVE.md). Env mirrors: `HIPFIRE_MODEL`, `HIPFIRE_IDLE_TIMEOUT`, `HIPFIRE_MAX_REQUEST_BYTES`, `HIPFIRE_SERVE_MAX_QUEUE`, `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS`, etc.

---

## Chat template overrides

| Key | Default | Validation |
|---|---|---|
| `chat_template` | `""` | empty or existing file path (`~/` expanded); existence + `isFile` only — no readability/access check |
| `default_chatml` | `true` | bool |

Project to `HIPFIRE_CHAT_TEMPLATE_FILE` / `HIPFIRE_DEFAULT_CHATML` when env unset (see CLI apply path).

---

## Per-model overlay

```bash
hipfire config qwen3.5:9b set dflash_mode off
hipfire config qwen3.5:9b          # TUI on overlay
```

Only explicitly set keys are stored; others inherit global. Primary path:
`~/.hipfire/models.toml`. Legacy `models.json` and `per_model_config.json` are
migration inputs and are not native write targets.

---

## One-shot env overrides (examples)

Full inventory: [`env-vars.md`](env-vars.md). Common operator overrides:

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

## RuntimeConfig (daemon env snapshot)

Separate from `HipfireConfig`. Fields read once from env in `RuntimeConfig::from_env` (`crates/hipfire-runtime/src/config.rs`):

| Field | Env | Default behavior |
|---|---|---|
| `normalize_prompt` | `HIPFIRE_NORMALIZE_PROMPT` | true unless `0`/`false`/`off`/`no` |
| `prompt_token_heat` | `HIPFIRE_PROMPT_TOKEN_HEAT` | off unless `1` |
| `prompt_heat_json` | `HIPFIRE_PROMPT_HEAT_JSON` | off unless `1` |
| `prompt_heat_limit` | `HIPFIRE_PROMPT_HEAT_LIMIT` | 64 |
| `dflash_draft` | `HIPFIRE_DFLASH_DRAFT` | unset |
| `dflash_mode` | `HIPFIRE_DFLASH_MODE` | `"off"` |
| `draft_f16` | `HIPFIRE_DRAFT_F16` | true unless `0` |
| `draft_gemm_dump` | `HIPFIRE_DRAFT_GEMM_DUMP` | off unless `1` |
| `draft_subphase` | `HIPFIRE_DRAFT_SUBPHASE` | off unless `1` |
| `ddtree_budget` | `HIPFIRE_DDTREE_BUDGET` | 256 |
| `ddtree_topk` | `HIPFIRE_DDTREE_TOPK` | 8 |
| `prefill_batched` | `HIPFIRE_PREFILL_BATCHED` | true unless `0` |
| `flash_partials_batch` | `HIPFIRE_FLASH_PARTIALS_BATCH` | unset |
| `tp_use_rccl` | `HIPFIRE_TP_USE_RCCL` | unset → RCCL default on; `0`/`false` opt out |
| `ngram_loop_threshold` | `HIPFIRE_NGRAM_LOOP_THRESHOLD` | **0 (off)** |
| `ngram_window` | `HIPFIRE_NGRAM_WINDOW` | 256 |
| `devices` | `HIPFIRE_DEVICES` | unset |
| `allow_mixed_arch` | `HIPFIRE_ALLOW_MIXED_ARCH` | false unless `1` |
| `uniform_vram_tolerance_gb` | `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | unset |
| `lm_head_f16` | `HIPFIRE_LM_HEAD_F16` | `"auto"` |
| `mtp_mode` | `HIPFIRE_MTP_MODE` | `"auto"` |
| `mtp_k` | `HIPFIRE_MTP_K` | 3 |

Note: CLI `ddtree_budget` default is `0` while bare `RuntimeConfig` default is `256` when only env/runtime path is used — CLI load params are the product path for `hipfire run`/`serve`.

Redline eligibility helper `gfx12_mq4r_redline_default` (same file) is a **narrow product-default predicate** for replay backend selection (gfx12 + arch_id 6 + `.mq4r` + pp=tp=1), not a general config key. Policy: [`REDLINE.md`](REDLINE.md).

---

## Related

| Topic | Owner |
|---|---|
| Env inventory | [`env-vars.md`](env-vars.md) |
| Models / registry sampling | [`MODELS.md`](MODELS.md) |
| Serve API | [`SERVE.md`](SERVE.md) |
| CLI | [`CLI.md`](CLI.md) |
| Multi-GPU | [`multi-gpu.md`](multi-gpu.md) |
