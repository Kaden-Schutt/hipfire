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

## MMQ screening

| Key | Default | Range | Notes |
|---|---|---|---|
| `mmq_screen` | auto | off / on / auto | Per-weight outlier detection for the i8 WMMA (MMQ) prefill path. `off` disables screening entirely (max throughput, risks #87 corruption when MMQ is active). `on` forces the load-time sweep on RDNA3/3.5. `auto` lets the daemon decide per arch (today identical to `on`, reserved so future versions can demote/promote per validated arch+model combo without forcing users to retune). |
| `mmq_screen_threshold` | 0.10 | 0.01–1.0 | Max per-row abs error threshold. Lower = more conservative (more fallbacks). 0.10 validated on 9B/27B for byte-identical output vs pure WMMA. |

MMQ (i8 WMMA + Q8_1 activation quantization) gives +40-50% prefill
speedup on RDNA3/3.5 but certain weight rows produce 5-9x higher
quantization error than normal. Without screening, these outliers
corrupt tool-call output (ChatML special-token leakage, ref #87).

MMQ itself is opt-in via `HIPFIRE_MMQ=1` or `HIPFIRE_WO_MMQ=1`.
`mmq_screen` only takes effect when MMQ is active; the daemon also
arch-gates the sweep to RDNA3/3.5 (`gfx1100` / `gfx1101` / `gfx1102` /
`gfx1103` / `gfx1150` / `gfx1151`).

Screening runs a batch=16 synthetic comparison (WMMA vs MMQ) per weight
matrix at load time (~0.1ms per weight, cached). On qwen3.5-9b, 25/216
weights fall back to WMMA; on qwen3.6-27b, 73/432. The remaining 83-88%
of weights keep the fast MMQ path.

Set `mmq_screen=off` only for benchmarking raw MMQ throughput. Not
recommended for production — output quality degrades on tool-call and
structured-output prompts.

Legacy boolean values from the PR #104 ship (`true`/`false`) are
auto-migrated on load: `true → on`, `false → off`.

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
| `auto` (default) | depends on discovery | fresh-default state — pull a model with a published sidecar and CASK engages on first turn | A3B targets are silently skipped from auto-attach |
| `off` | full `max_seq` | A3B models, plenty of VRAM, hard-off guarantee | only safe profile for 35B-a3b at current R̄ |
| `balanced` | budget=1024, ≈165 MB on 27B | dense 27B on a 16 GB card, mixed-length workloads | dense only; AR or DFlash both safe |
| `conservative` | budget=2048, ≈275 MB on 27B | ≥20 GB VRAM, very long advertised contexts | dense only |
| `aggressive-vram` | budget=512, ≈96 MB on 27B | dense 27B on a 16 GB card with tight headroom; aggressive long-ctx fit | **AR only** — m-fold + DFlash has a documented attractor regression. Set `dflash_mode=off`. Not for A3B. |

¹ KV footprint estimates for dense 27B with `kv_cache=asym3` (~107 KB/token).
Scale linearly with the model's `n_layers × n_kv_heads × head_dim`.

Picking a profile rewrites a bundle of CASK config keys in one shot. The
`balanced` / `conservative` / `aggressive-vram` profiles set the policy
fields and re-enable `cask_auto_attach`; they preserve `cask_sidecar` —
set the path separately with `hipfire config set cask_sidecar
/path/to/<model>.triattn.bin`, or rely on auto-attach by `hipfire pull`'ing
a model that ships one.

The `auto` profile is the fresh-default state: at load time the engine
scans for a TriAttention sidecar next to the model file (registry's
`triattn.file` first, then `<basename>.triattn*.bin` glob fallback). When
found AND target is not A3B, it attaches with drop-eviction at the
configured budget. `hipfire pull qwen3.6:27b` fetches the v3 sidecar
alongside weights, so `hipfire run` engages CASK on the first turn with
no further config.

The `off` profile is the **hard-off** guarantee: clears `cask_sidecar`
AND sets `cask_auto_attach=false` so a discoverable sidecar can't sneak
back in via the auto-attach path. Stricter than `auto`; pick this when
you want eviction provably off (e.g., on A3B targets, or for
quality-sensitive single-turn workloads).

### Underlying knobs (advanced — prefer profiles)

| Key | Default | Range | Notes |
|---|---|---|---|
| `cask_sidecar` | "" | path | Path to TriAttention sidecar `.bin`. Empty = eviction disabled regardless of other knobs. |
| `cask` | false | bool | true = CASK m-folding (Kim & Gwon 2026); false = plain TriAttention drop-eviction. |
| `cask_budget` | 512 | 64–65536 | Active token count post-eviction. Smaller = tighter VRAM, more frequent eviction events. |
| `cask_beta` | 128 | 0–65536 | Hysteresis. Buffer needs to fill `budget + beta` before re-triggering eviction. |
| `cask_core_frac` | 0.5 | 0.0–1.0 | Fraction of budget kept un-merged when `cask=true`. Inert otherwise. |
| `cask_fold_m` | 2 | 1–16 | m-way merge factor for non-core slots when `cask=true`. m=2 is the validated sweet spot; m=4 over-folds. Inert when `cask=false`. |
| `cask_auto_attach` | true | bool | When true, scan for a sidecar next to the model file at load and attach it if `cask_sidecar` is empty + target isn't A3B. Set false to guarantee no eviction (the `off` profile flips this). |

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

## PFlash speculative prefill (EXPERIMENTAL #93)

PFlash compresses long prompts via a small drafter model before the
target prefill runs. A drafter scores attention importance per source
block, the daemon emits compressed token spans, and the target
prefills the compressed stream. Decode (DFlash / DDTree / AR) is
unchanged. Off by default until per-target validation (NIAH retrieval,
coherence) clears.

| Key | Default | Range / values | Notes |
|---|---|---|---|
| `prefill_compression` | `off` | `off` / `auto` / `always` | Top-level mode. `auto` compresses only when source >= `prefill_threshold`. `always` compresses every request (research / bench). |
| `prefill_threshold` | 32768 | 0–524288 | Token cutoff for `auto` mode. Below this, requests bypass with reason `below_threshold`. |
| `prefill_keep_ratio` | 0.05 | (0, 1] | Fraction of source tokens to keep after sink + recent + top-scoring spans. Lower = more aggressive (faster TTFT, riskier retrieval). |
| `prefill_alpha` | 0.85 | [0, 1] | Block-selection strictness. |
| `prefill_min_keep` | 2048 | 0–524288 | Floor on retained tokens. Caps over-aggressive compression on short inputs. |
| `prefill_sink` | 256 | 0–65536 | Always-keep prefix tokens (system / template / first-user-turn). |
| `prefill_recent` | 1024 | 0–65536 | Always-keep tail tokens. |
| `prefill_block` | 128 | 1–4096 | Scoring block size in source tokens. |
| `prefill_drafter` | "" | path | Path to drafter HFQ artifact. Tokenizer must match the target's; mismatch surfaces as `BypassReason::TokenizerMismatch`. |
| `prefill_profile` | false | true / false | Per-stage timing logs (`score_ms / select_ms / gather_ms / total_ms`). |
| `prefill_sparse_threshold` | 32768 | 0–524288 | Phase 3 plumbing for the sparse drafter forward (kernel not yet shipped). |

Bypass / status events (only emitted when PFlash actually had a chance
to fire -- i.e. drafter loaded successfully + request reached the
generate path that wires PFlash):

| Reason | Event | Trigger |
|---|---|---|
| `mode_off` | (none, silent) | `prefill_compression=off`. |
| `below_threshold` | `pflash_bypass` | `auto` mode + source tokens below `prefill_threshold`. |
| `tool_call_request` | `pflash_bypass` | User or system prompt contains the `<tool_call>` token. |
| `tokenizer_mismatch` | `pflash_bypass` | Drafter and target tokenizer signatures differ. Load still succeeds with `tokenizer_compat:false` in the `pflash` status line; the per-request gate is what bypasses. Reload with a matched-tokenizer drafter to compress. |
| `dflash_decode_active` | `pflash_bypass` | DFlash spec-decode took the fast path; PFlash compression on that path is a follow-up. Disable `dflash_mode` if compression is required. |
| `scoring_degenerate` | `pflash_bypass` | Scorer returned non-finite or all-zero scores. |
| (drafter load failure) | `pflash_load_failed` | Drafter HFQ open / config / weights / tokenizer failed at load. PflashState stays `None` for the session; subsequent generate requests run uncompressed with no further event. Re-load with a corrected `prefill_drafter` path to retry. |
| (drafter unset) | (none, silent) | `prefill_compression != off` but `prefill_drafter` empty. CLI prints a single warning at load; no per-request event. |
| (vision request) | (none, silent) | Image-bearing requests route to `generate_vl` which does not yet wire PFlash. PFlash is implicitly bypassed for vision. |

When compression fires, `done` events embed a `pflash` field:
`{source_tokens, kept_tokens, keep_ratio, alpha, score_ms, total_ms,
source_md5, compressed_md5}`. When PFlash bypassed (skipped), the
field is `{bypass_reason, alpha}` (only on the `pflash_bypass` rows
above; the silent / load-failure rows produce a `done` event without a
`pflash` field).

CLI usage:

```bash
# Global default
hipfire config set prefill_compression auto
hipfire config set prefill_drafter ~/.hipfire/models/qwen3-0.6b.hf4

# Per-target override (recommended -- different drafters per target).
# CLI shape: `hipfire config <model-tag> set <key> <value>` (the tag
# slots in BEFORE the action, matching the existing cask / dflash UX).
hipfire config qwen3.5:9b set prefill_compression auto
hipfire config qwen3.5:9b set prefill_drafter ~/.hipfire/models/qwen3-0.6b.hf4

# Per-request env override (research / one-shot benchmarking)
HIPFIRE_PREFILL_COMPRESSION=always \
HIPFIRE_PREFILL_KEEP_RATIO=0.10 \
hipfire run qwen3.5:9b "long-context prompt..."
```

`HIPFIRE_PREFILL_*` env vars exist for every config key (mode,
threshold, keep_ratio, alpha, min_keep, sink, recent, block, drafter,
profile, sparse_threshold).

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
HIPFIRE_PREFILL_COMPRESSION=auto    # PFlash mode: off|auto|always (#93)
HIPFIRE_PREFILL_THRESHOLD=32768     # PFlash auto-mode source-token cutoff
HIPFIRE_PREFILL_KEEP_RATIO=0.05     # PFlash kept fraction in (0, 1]
HIPFIRE_PREFILL_DRAFTER=~/.hipfire/models/qwen3-0.6b.hf4
HIPFIRE_PREFILL_PROFILE=1           # PFlash per-stage timing logs
```
