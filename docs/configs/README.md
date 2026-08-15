# TOML configuration examples

These files are sparse, standalone examples for `~/.hipfire/config.toml`.
They show persistent TOML replacements for controls that were historically
passed as `HIPFIRE_*` environment variables.

| Example | Audience | Purpose |
|---|---|---|
| [`user.toml`](user.toml) | Users | Generation, memory, prompt, speculation, and serve policy. |
| [`developer.toml`](developer.toml) | Developers | Hardware selection, kernel policy, graphs, diagnostics, and the experimental escape namespace. |
| [`redline-pm4.toml`](redline-pm4.toml) | Redline developers | Explicit retained-PM4 transport and diagnostic policy. |
| [`batched-redline-pm4-product.toml`](batched-redline-pm4-product.toml) | Redline developers | Certified fixed-slot sampled AR batch route for gfx1201 Qwen3.6 A3B MQ4R. Pins the retained-PM4 batch shape; not a built-in profile. |

Each file is a complete valid profile, but it is intentionally not a dump of
every default. Missing keys continue to inherit registry or compiled policy.
Copy only the settings you intend to pin; an explicit value will keep winning
if a future release changes its default.

The safest way to adopt an example is through the named profile command:

```bash
hipfire config profile set default   # docs/configs/user.toml
hipfire config profile set dev       # docs/configs/developer.toml
hipfire config profile set redline   # docs/configs/redline-pm4.toml
```

`hipfire config profile set <name>` replaces the entire sparse global
`config.toml` with the selected built-in or custom profile. Profile names are
control-plane identifiers only; they are never stored inside the TOML file.
Snapshot the current global layer as a custom profile with:

```bash
hipfire config profile create lab
# writes ~/.hipfire/profiles/lab.toml
hipfire config profile set lab
```

Bare `hipfire config profile` launches the interactive profile wizard. Profiles
are global-only (`hipfire config <model> profile ...` is rejected).

Individual keys can still be pinned one at a time:

```bash
hipfire config set memory.kv_cache q8
hipfire config set prompt.normalize true
hipfire config explain memory.kv_cache
hipfire config list
```

If `~/.hipfire/config.toml` does not exist, one example may instead be copied
as a starting point:

```bash
mkdir -p ~/.hipfire
cp docs/configs/user.toml ~/.hipfire/config.toml
```

Do not overwrite an existing config. TOML has no table-include or safe textual
concatenation operation; merge the desired keys or use `hipfire config set`
(or `hipfire config profile set|create`).

## Common environment migrations

| Historical one-shot input | Persistent TOML key |
|---|---|
| `HIPFIRE_KV_MODE=q8` | `memory.kv_cache = "q8"` |
| `HIPFIRE_ATTN_FLASH=never` | `attention.flash = "never"` |
| `HIPFIRE_NORMALIZE_PROMPT=0` | `prompt.normalize = false` |
| `HIPFIRE_SPECULATION=off` | `speculation.mode = "off"` |
| `HIPFIRE_DFLASH_MODE=auto` | `speculation.dflash = "auto"` |
| `HIPFIRE_MTP_MODE=on` | `speculation.mtp = "on"` |
| `HIPFIRE_MTP_K=3` | `speculation.mtp_k = 3` |
| `HIPFIRE_PROMPT_CACHE_CAP=64` | `memory.prompt_cache_capacity = 64` |
| `HIPFIRE_DEVICES=3` | `hardware.devices = "3"`; startup applies `ROCR_VISIBLE_DEVICES=3` and matching `HIP_VISIBLE_DEVICES=0` |
| `HIPFIRE_REPLAY_BACKEND=redline` | `replay.backend = "redline"` |
| `HIPFIRE_REPLAY_TRANSPORT=pm4` | `replay.transport = "pm4"` |
| `HIPFIRE_REPLAY_PM4_QUEUES=2` | `diagnostic.replay.pm4_queues = "2"` |
| `HIPFIRE_REPLAY_ROUTE_PROOF_LOG=1` | `diagnostic.replay.route_proof_log = true` |

Stable keys are typed and documented by `hipfire config schema`. A remaining
experimental `HIPFIRE_FOO=value` maps mechanically to
`developer.foo = <scalar>`:

```toml
[developer]
verify_graph = false       # HIPFIRE_VERIFY_GRAPH=0
dspark_profile = true      # HIPFIRE_DSPARK_PROFILE=1
```

`[developer]` is global process policy. It is snapshotted before GPU
initialization and cannot be stored in a per-model overlay or supplied by a
registry model card. Prefer a stable typed key whenever one exists.

## Bootstrap variables that remain environment-based

Paths needed before config discovery or process launch cannot live inside the
file they locate. These remain bootstrap inputs:

- `HIPFIRE_HOME`, `HIPFIRE_MODELS_DIR`
- `HIPFIRE_DAEMON_BIN`, `HIPFIRE_CLI_BIN`, `HIPFIRE_TUI_BIN`
- `HIPFIRE_REGISTRY_URL`, `HIPFIRE_HF_BASE` (overrides `HF_ENDPOINT`), `HIPFIRE_NO_REGISTRY_FETCH`
- `HIPFIRE_KERNEL_CACHE`, `HIPFIRE_SPILL_DIR`, `HIPFIRE_QUANT_DIAG_PATH`

For the authoritative key/default/enum reference, see
[`docs/CONFIG.md`](../CONFIG.md). For the full compatibility inventory, see
[`docs/env-vars.md`](../env-vars.md).
