# Configuration

Two layers:

1. **Global config** at `~/.hipfire/config.json` — applies to every
   model unless overlaid.
2. **Per-model overlay** at `~/.hipfire/per_model_config.json` — sparse
   keys overriding global for a specific tag.

Edit interactively with `hipfire config` (global) or `hipfire config
<tag>` (overlay). Or set non-interactively: `hipfire config set <key>
<value>`.

## Generation

| Key | Default | Range / values | Notes |
|---|---|---|---|
| `temperature` | 0.30 | 0.0–2.0 | 0.0 = greedy. |
| `top_p` | 0.80 | 0.0–1.0 | Nucleus sampling. |
| `repeat_penalty` | 1.05 | 1.0–3.0 | Default kept conservative — 1.3 causes MQ4 gibberish at low temp. |
| `max_tokens` | 512 | 1–131072 | Per-request cap. Used by `hipfire run` and as the fallback for OpenAI API requests that omit `max_tokens` in the body. Bump if you see thinking-on responses truncated with `finish_reason=stop` mid-`<think>`. |
| `max_seq` | 32768 | 512–524288 | KV cache physical capacity. |
| `thinking` | on | on / off | Whether to keep `<think>...</think>` reasoning blocks. |
| `max_think_tokens` | 0 | 0–32768 | 0 = no cap. Caps tokens emitted before `</think>` closes. |

## KV cache

| Key | Default | Values |
|---|---|---|
| `kv_cache` | auto (per arch) | auto / q8 / asym4 / asym3 / asym2 / turbo / turbo4 / turbo3 / turbo2 |

Per-arch defaults: gfx1100 → asym3, gfx1030 → asym3, gfx1010/1013 →
asym2. asym3 is rotated K (Lloyd-Max) + Q8 V — the multi-turn quality
sweet spot. Use `q8` for byte-exact reference behavior at higher VRAM
cost.

## Speculative decode (DFlash)

| Key | Default | Values | Notes |
|---|---|---|---|
| `dflash_mode` | off | on / off / auto | `auto` enables DFlash on dense Qwen 3.5+ targets and skips configs known to lose. |
| `dflash_adaptive_b` | true | true / false | Adaptive draft block size. |
| `dflash_ngram_block` | auto | true / false / auto | n-gram cache prefilling. |

DFlash speedup is genre-conditional: large on code, modest on
instruct, can be a net loss on prose. See [BENCHMARKS.md](BENCHMARKS.md)
for measured speedups. Per-model override is the most common knob:
`hipfire config qwen3.5:9b set dflash_mode off` if your workload is
mostly long-form prose.

## Attention

| Key | Default | Values |
|---|---|---|
| `flash_mode` | auto | auto / always / never |

`auto` enables FlashAttention when the seq len passes the FA-vs-vanilla
crossover for the current arch. `never` is the byte-exact reference;
`always` forces FA even on short prompts.

## CASK (TriAttention KV eviction)

CASK is the KV cache eviction system. When a `cask_sidecar` is loaded,
the engine compacts KV against the sidecar's band-centers once active
tokens exceed `cask_budget + cask_beta`, then re-triggers when the
buffer fills again. This pins physical VRAM regardless of advertised
`max_seq` — a 16 GB card can serve dense 27B with a 131k context window
because only `cask_budget + cask_beta + 256` slots are physically
allocated.

### Profiles (recommended path)

The five raw knobs interact non-obviously and have hard-rule failure
modes. Pick a profile bundle in the TUI (`hipfire config` → `cask
profile` row) or via the CLI:

```bash
hipfire config cask-profile <name>                     # global
hipfire config qwen3.6:27b cask-profile <name>         # per-model overlay
hipfire config cask-profile                            # list active + available
```

| Profile | KV footprint¹ | Use when | Constraints |
|---|---|---|---|
| `off` | full `max_seq` | A3B models, plenty of VRAM, single-turn quality | only safe profile for 35B-a3b at current R̄ |
| `balanced` | budget=1024, ≈165 MB on 27B | dense 27B on a 16 GB card, mixed-length workloads | dense only; AR or DFlash both safe |
| `conservative` | budget=2048, ≈275 MB on 27B | ≥20 GB VRAM, very long advertised contexts | dense only |
| `aggressive-vram` | budget=512, ≈96 MB on 27B | dense 27B on a 16 GB card with tight headroom; aggressive long-ctx fit | **AR only** — m-fold + DFlash has a documented attractor regression. Set `dflash_mode=off`. Not for A3B. |

¹ KV footprint estimates for dense 27B with `kv_cache=asym3` (~107 KB/token).
Scale linearly with the model's `n_layers × n_kv_heads × head_dim`.

Picking a profile rewrites the policy bundle (`cask`, `cask_budget`,
`cask_beta`, `cask_core_frac`, `cask_fold_m`) in one shot. The non-`off`
profiles **preserve** `cask_sidecar` — set the path separately with
`hipfire config set cask_sidecar /path/to/<model>.triattn.bin`.

The `off` profile additionally **clears `cask_sidecar`**: the daemon
triggers eviction whenever a sidecar path is set, regardless of the
`cask` boolean (which only switches between m-fold and drop-eviction).
Clearing the path is the only way to actually disable eviction.

### Underlying knobs (advanced — prefer profiles)

| Key | Default | Range | Notes |
|---|---|---|---|
| `cask_sidecar` | "" | path | Path to TriAttention sidecar `.bin`. Empty = eviction disabled regardless of other knobs. |
| `cask` | false | bool | true = CASK m-folding (Kim & Gwon 2026); false = plain TriAttention drop-eviction. |
| `cask_budget` | 512 | 64–65536 | Active token count post-eviction. Smaller = tighter VRAM, more frequent eviction events. |
| `cask_beta` | 128 | 0–65536 | Hysteresis. Buffer needs to fill `budget + beta` before re-triggering eviction. |
| `cask_core_frac` | 0.5 | 0.0–1.0 | Fraction of budget kept un-merged when `cask=true`. Inert otherwise. |
| `cask_fold_m` | 2 | 1–16 | m-way merge factor for non-core slots when `cask=true`. m=2 is the validated sweet spot; m=4 over-folds. Inert when `cask=false`. |

### Safety hard rules

Three failure modes documented in `.claude/.../memory/`:

1. **`cask=true` (m-fold) + DFlash → block-level attractor.** Engine
   `f16eceb` 2026-04-26: 9B at `max_tokens=1500` emitted 76+ consecutive
   reps of a 5-token block (`node.value = value\n`). Headline τ and
   tok/s looked great; output was garbage. The single-token coherence
   gate did not catch it. **Use `cask=false` whenever `dflash_mode != off`**
   until the GPU-side m-fold rewrite re-passes the three-tier dflash
   gate. Plain drop-eviction (`cask=false`) is stable on dense models
   with DFlash.

2. **Any eviction on A3B (35b-a3b-3.5 / 3.6) → confident-wrong
   hallucination.** Multi-turn smoke 2026-04-28 (R̄=0.36 / 0.39
   sidecars under eviction): A3B-3.5 attractor-looped "Safety Policy
   Check" 8×, fabricated species; A3B-3.6 inverted hydrothermal-vent
   recall to *photosynthesis*. Dense 27B-3.6 (R̄=0.610) degraded
   gracefully. **Don't enable a sidecar on A3B targets at current
   R̄.** The CLI refuses non-`off` profiles on per-model A3B configs
   (override with `HIPFIRE_FORCE_A3B_EVICTION=1`, not recommended).

3. **DFlash + eviction is quality-asymmetric vs AR + eviction.** 12
   evictions cost DFlash −28% τ but AR only −1.7% per event. For
   long-context quality-sensitive output, AR + sidecar is the
   conservative path; DFlash + sidecar is ~3× faster wall-clock but
   degrades harder.

### CASK m-fold validation (when DFlash is off)

Paper sweep (9B Q8, AR, 18 prompts):

| Config | budget=full | budget=½ | budget=¼ |
|---|---:|---:|---:|
| TriAttention drop-eviction | 89% | 83% | 61% |
| **CASK m=2, frac=0.5** | 89% | 83% | **72%** |
| CASK m=4, frac=0.5 | 89% | 83% | 67% |

m=2 is the sweet spot; m=4 over-folds. The +11 pts at the aggressive
budget (¼) is what makes `aggressive-vram` viable for tight-VRAM
configurations on AR.

## Prompt processing

| Key | Default | Values | Notes |
|---|---|---|---|
| `prompt_normalize` | true | true / false | Collapse `\n{3,}` → `\n\n` at engine entry. +24% τ on PEP-8-style code prompts; default ON since 2026-04-26. Opt out only when raw whitespace patterns are semantically load-bearing. |

## Server

| Key | Default | Range |
|---|---|---|
| `port` | 11435 | 1–65535 |
| `idle_timeout` | 300 | 0–86400 (seconds) |
| `default_model` | "" (none) | tag or path |

`idle_timeout` evicts the loaded model from VRAM after that many
seconds of no requests; the next request reloads with a 2–5 s cold
start. Set to 0 to keep weights resident forever (useful when you have
spare VRAM and want zero-latency requests).

`default_model` is what `hipfire serve` pre-warms on startup.

## Per-model overlay

```bash
hipfire config qwen3.5:9b
```

Opens the same TUI but writes to the overlay file. Rows show
`(inherited)` if the key matches global and `(overridden)` if it
diverges. A rendered overlay JSON looks like:

```json
{
  "qwen3.5:9b": {
    "dflash_mode": "off",
    "kv_cache": "q8"
  }
}
```

Only keys explicitly set are written; everything else inherits global.
Delete a row's override with the TUI's `d` key.

## One-shot env overrides

For testing without touching the config file:

```
HIPFIRE_KV_MODE=asym3
HIPFIRE_ATTN_FLASH=auto
HIPFIRE_NORMALIZE_PROMPT=0          # opt out of \n{3,} collapse
HIPFIRE_LOCAL=1                     # skip the running daemon
HIPFIRE_HIPCC_EXTRA_FLAGS="-mcumode"
HIPFIRE_PROMPT_TOKEN_HEAT=1         # dump per-position BPE merge ranks
HIPFIRE_PROMPT_HEAT_JSON=1          # the same, machine-readable
HIPFIRE_GRAPH=1                     # hipGraph capture (debug; AR-only, may degrade quality on large models)
```
