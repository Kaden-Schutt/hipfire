# Models

hipfire ships with a curated registry of Qwen 3.5 / 3.6 family tags
(small + dense + MoE) and supports running any GGUF or safetensors model
you bring yourself.

## Curated tags

All entries are MQ4 (FWHT-rotated 4-bit, calibrated for the Qwen3.5
hybrid attention path) unless noted. MQ6 variants exist for the same
sizes when you want more headroom; pull with the `:<size>-mq6` suffix.

| Tag | File | VRAM floor | Notes |
|---|---|---|---|
| `qwen3.5:0.8b` | 0.55 GB | 1 GB | Tiny, hybrid DeltaNet + FullAttn |
| `qwen3.5:2b` | 1.3 GB | 2 GB | 2B, HF4 (legacy 4-bit format; `-hf6` variant available) |
| `qwen3.5:4b` | 2.6 GB | 4 GB | Best speed/quality balance |
| `qwen3.5:9b` | 5.3 GB | 6 GB | Default `serve` pre-warm |
| `qwen3.5:27b` | 15 GB | 16 GB | Needs 16 GB+ VRAM |
| `qwen3.5:35b-a3b` | 19.7 GB | 22 GB | MoE 35B / 3B-active |
| `qwen3.6:27b` | 15 GB | 16 GB | 3.6 refresh, same hybrid arch as 3.5 |
| `qwen3.6:35b-a3b` | 22.9 GB | 24 GB | 3.6 MoE refresh |
| `deepseek-v4-flash` | 82 GB | 96 GB | DeepSeek V4 Flash (arch_id=9): MQ2-Lloyd routed-expert MoE, Q8_0 attn KV, Hyper-Connections + compressed-KV indexer + tail-only RoPE. `hipfire pull` also fetches the MTP sidecar for K=2 spec-decode (+29% TG on code). |

Higher-quality variants:

| Tag pattern | Effect |
|---|---|
| `qwen3.5:<size>-mq6` | 6-bit quant, +47% file size, closer-to-Q8 quality |

DFlash speculative-decode drafts:

| Tag | Pairs with | Effect |
|---|---|---|
| `qwen3.5:9b-draft` | `qwen3.5:9b` | 2–3× decode on code/instruct prompts |
| `qwen3.5:27b-draft` | `qwen3.5:27b` | 4× decode on code (peak 218 tok/s on 7900 XTX) |
| `qwen3.6:27b-draft` | `qwen3.6:27b` | ~4× on code |

```
hipfire pull qwen3.5:27b
hipfire pull qwen3.5:27b-draft
hipfire config set dflash_mode auto       # opt in (default is off)
```

`hipfire pull <target>` prompts to also pull the matching `-draft` if
the registry has one. At inference time the CLI does **filename
auto-match**: when the target path matches
`qwen3?.?(5|6)[-_]?<size>.(mq4|mq6|...)`, the CLI looks for a sibling
file `qwen3{ver}-{size}-dflash-{quant}.hfq` next to it (in
`~/.hipfire/models/` or alongside) and wires it up as the draft
without an explicit flag. Override with `HIPFIRE_DFLASH_DRAFT=<path>`
or disable via empty string.

See [ARCHITECTURE.md](ARCHITECTURE.md#dflash-speculative-decode) for
the resolution priority and the daemon load path,
[BENCHMARKS.md](BENCHMARKS.md) for the per-genre speedup table.

Hermes / Aureth / Qwopus fine-tunes (Qwen 3.5 architecture):

| Tag | Notes |
|---|---|
| `carnice:9b` / `carnice:27b` | kai-os Hermes tool-use |
| `qwopus:4b` / `qwopus:9b` / `qwopus:27b` | Jackrong reasoning fine-tune |

`hipfire list -r` prints the full curated registry plus availability.

## Bring your own — three input shapes

### From HuggingFace

```bash
hipfire quantize Jackrong/Qwopus3.5-4B-v3 \
    --format mq4 \
    --install --register qwopus:4b
```

Downloads the safetensors, quantizes, drops the result in
`~/.hipfire/models/`, and registers a local alias so `hipfire run
qwopus:4b` works. See [QUANTIZE.md](QUANTIZE.md).

### From local safetensors

```bash
hipfire quantize ./my-finetune/ --format mq4 -o my-finetune.mq4
```

Any directory that contains a `config.json` plus one or more
`.safetensors` files. Architectures supported by the engine: `llama`,
`qwen3`, `qwen3_5`, `qwen3_5_moe`. Other architectures are accepted by
the quantizer but won't load at inference.

### From GGUF

```bash
hipfire quantize ./tinyllama.Q4_K_M.gguf \
    --install --register tinyllama:1b-gguf
```

Default format for GGUF input is `hf4` (HFQ4-G256 — the dense-safe
4-bit format with no FWHT rotation). For Qwen3.5+ family GGUFs override
with `--format mq4` to opt into the rotated hot path.

GGUF source quantizations supported by the dequant pass:

```
Q4_0  Q8_0  Q4_K  Q6_K  F16  BF16  F32
```

Q5_K, IQ-quants, and other GGUF formats aren't implemented; the
quantizer panics on encounter (port from llama.cpp's `ggml-quants.c` if
you need one). See [QUANTIZE.md](QUANTIZE.md) for format-by-arch
guidance and the double-quantization quality tradeoff.

## Thinking mode and chat templates

### Thinking mode mechanics

Qwen 3.5 / 3.6 are reasoning models: by default they emit a hidden
`<think>...</think>` reasoning block before the visible answer. hipfire's
data flow through that block:

1. The daemon receives the full token stream from the model (no daemon-side
   filter).
2. The CLI / OpenAI server layer strips the visible `<think>...</think>`
   substring from `content`. Tokens emitted while inside `<think>` are also
   re-broadcast to OpenAI streaming clients as `delta.reasoning_content`
   (a field convention shared by DeepSeek and the pi-coding-agent harness),
   so reasoning-aware UIs can render the thinking view live without it
   leaking into the assistant message.
3. After `</think>`, the leading newline is stripped and the answer
   streams as normal `delta.content`.

Two consequences worth knowing:
- `hipfire run`'s stdout shows the answer only. Thinking is invisible
  but still consumes tokens.
- Reasoning-heavy turns can sit silent on the visible-content channel
  for thousands of tokens. The OpenAI streaming server emits SSE
  comment heartbeats every 10 s during prefill and reasoning-content
  deltas during the think phase to keep the connection alive (sub-minute
  idle timeouts in OpenCode / pi-coding-agent would otherwise abort).

### `thinking: on / off`

`thinking` is a hipfire config knob, not a prompt directive. It controls
whether the visible `<think>...</think>` block is *kept* in the assistant
message. Setting `thinking=off` does NOT inject a `/no_think` directive
into the prompt.

The "advisory only" semantics are deliberate. Earlier versions of hipfire
tried injecting `/no_think` into system messages, user prefixes, mixed
positions, etc.; every placement broke a different Qwen3.5 prompt shape
with empty `<think><|im_end|>` halts (commits 3798399, 2d9c24b, 799c268,
cf2a3d8, 68b32ee, b292565, all reverted in 5533926). The current contract:

- The model decides whether to think.
- `thinking=on` (default): visible `<think>...</think>` blocks are kept
  in the assistant message stream as-is.
- `thinking=off`: the existing `<think>...</think>` filter strips the
  visible reasoning so the user only sees the answer. The model still
  thinks; you just don't see it. The TUI flashes a yellow warning when
  enabling this so the cost is visible.

### `max_think_tokens`

Cap how many tokens the model may emit before `</think>` closes. 0
(default) means no cap. When the cap is hit, the daemon force-emits
`</think>` and the model proceeds to the answer phase. Useful when:
- You want predictable latency on a thinking model.
- A specific model loops in `<think>` (the A3B family historically does
  this on hard prompts; see #89 for the long-budget block-loop attractor).

```bash
hipfire config set max_think_tokens 4096                  # global
hipfire config qwen3.6:35b-a3b set max_think_tokens 1024  # per-model
```

Per-model settings take precedence; the registry pre-applies sane caps
for known offenders.

### OpenAI / API knobs

The OpenAI server accepts three additional fields beyond the OpenAI
spec, contributed by @shilga in #79:

- `enable_thinking: bool`. Same as `thinking`, scoped to one request.
  Overrides global / per-model config for this turn only.
- `preserve_thinking: bool`. Keep the model's `<think>...</think>` in
  the assistant message it writes back to the chat history (default
  off). Useful when you're feeding the conversation back through a tool
  loop and want the model's prior reasoning visible on the next turn.
- `presence_penalty: float`. Forwarded to the sampler. Standard OpenAI
  semantics; -2.0 to 2.0 range.

`reasoning.effort: "low" | "medium" | "high"` is also accepted (OpenAI
o1-style); maps to `max_think_tokens` of 1024 / 4096 / 32768
respectively.

### Chat template

hipfire applies the **ChatML** template for Qwen 3.5 / 3.6 / Carnice /
Qwopus; the daemon expects messages already serialized by the CLI
into:

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
```

`hipfire run` and the OpenAI server both build this string from
`messages[]` before sending to the daemon. Per-model template tweaks
live in `cli/registry.json` under each model entry; you don't normally
edit them. Custom system prompts are forwarded as a `system` role
message and inserted at the top of the ChatML envelope.

One implicit normalization step: the engine collapses runs of three or
more `\n` characters down to exactly two before tokenization
(`prompt_normalize: true` by default). Eliminates the rare BPE token
1358 (`\n\n\n`) in favour of HOT token 271 (`\n\n`) on Qwen3.5/3.6,
lifting τ on PEP-8-style code prompts up to +26.7%. Set
`prompt_normalize: false` only if your input semantically depends on
preserving raw `\n{3,}` whitespace.

## Model files on disk

```
~/.hipfire/models/
├── qwen3.5-9b.mq4                  # MQ4 (FWHT-rotated, Qwen3.5 hot path)
├── qwen35-9b-dflash-mq4.hfq        # DFlash draft for qwen3.5:9b (filename auto-match)
├── tinyllama.Q4_K_M.hf4            # HFQ4 (no rotation, dense)
└── ...
```

Extension legend:

| Ext | Format | Inference path |
|---|---|---|
| `.mq4` | MQ4G256 (FWHT-rotated 4-bit) | Qwen3.5+ hot path (DeltaNet) |
| `.mq6` | MQ6G256 (FWHT-rotated 6-bit) | Qwen3.5+ higher quality |
| `.hf4` | HFQ4-G256 (raw 4-bit) | Llama / Qwen3 / Mistral / dense |
| `.hf6` | HFQ6-G256 (raw 6-bit) | Dense, higher quality |
| `.hfq` | Legacy HFQ4 (pre-0.1.5 naming) | Loads, no new files written here |

CLI discovery (`hipfire list`, fuzzy `hipfire run` lookup) recognizes
all five extensions.

## Current local family status

This table reflects the model families currently present under
`~/Models` / `~/.hipfire/models` in this checkout. It is intentionally
about runnable engine support, not just whether a generated `.hfq`
artifact exists on disk. `Cactus-Compute/needle` is omitted because it
is a custom non-Hipfire architecture target.

| Family | Local examples | Runtime status | DFlash | MTP | CASK | PP | Batched prefill | KLD ref gen | Compatible / missing kernels |
|---|---|---|---|---|---|---|---|---|---|
| Qwen 3.5 / 3.6 dense hybrid | `qwen3.5-{0.8b,2b,4b,9b}`, `qwen3.6-27b` | Supported as Qwen35 dense (`arch_id=5`). | Supported for paired dense drafts when target lm_head dtype is Q8/HFQ4/MQ4, plus MQ3 on gfx11/gfx12; MQ6 targets need AR. | Present as native Qwen35 speculative-verify/MTP surfaces for validated dense paths; still correctness-first and not the same as DFlash drafts. | Supported with TriAttention/CASK sidecars on FullAttention layers. | Supported only on Qwen35 path when incompatible features are off. | Supported via Qwen35 batched prefill path. | Smoke refs exist for `qwen3.5-{0.8b,2b,9b}`; no full refs yet. | Dense Qwen35 decode/prefill kernels cover BF16/MQ4/MQ6 and selected MQ3. Missing DFlash batched lm_head/verify support for MQ6/MQ8/MQ2/F16 targets. |
| Qwen 3.5 / 3.6 MoE | `qwen3.6-35b-a3b`, `qwen3.5-122b-a10b` | Supported as Qwen35 MoE (`arch_id=6`) when quantized in Qwen3.5-MoE tensor layout. | Limited: dense-style DFlash works only where target/draft dtypes hit supported batched verify paths; MQ3 MoE is refused for DFlash. | MoE MTP code exists, but admission is narrower than dense and still gated by MoE dtype/layout validation. | Supported on FullAttention layers; no MoE-specific eviction of expert weights unless using the separate pager path. | Supported only on Qwen35 path when incompatible features are off. | Supported; MoE batched prefill admits MQ4 control and newer MQ6/MQ3 surfaces on validated arches. | `qwen3.6-35b-a3b-bf16` KLD producer is currently skipped on error; no completed refs for MoE rows. | Indexed MoE gate/up/down, shared expert, router, and grouped GEMM kernels exist for Qwen35 MoE. Missing broad MQ3/MQ2/MQ8 MoE DFlash coverage and full validation for every local MoE artifact. |
| Qwen3-MoE / Qwen3-Coder (`qwen3_moe`) | `qwen3-coder-30b-a3b-instruct`, `tiny-random/qwen3-moe` | Not currently first-class. Local Coder HFQs are stamped `arch_id=0`, but source configs are `qwen3_moe`; that does not match the Qwen35-MoE loader layout. | No. | No. | No. | No. | No. | Listed as a desired KLD target for Coder, but no completed ref. | Needs a `qwen3_moe` architecture mapping and loader/kernel audit. Existing Qwen35 MoE kernels assume Qwen3.5 hybrid layer/tensor layout, not plain Qwen3-MoE/Coder layout. |
| DeepSeek V4 Flash | `deepseek-v4-flash.mq4.hfq` | Supported as dedicated DeepSeek V4 path (`arch_id=9`). | No Qwen-style DFlash drafter. | Supported as DeepSeek V4's own optional MTP speculative decode path. | No. | No. | Supported by DeepSeek V4 chunked batched prefill / MTP fill. | Not currently targeted for KLD refs. | Dedicated DeepSeek V4 kernels cover Hyper-Connections, compressed-KV indexer, SWA attention, routed MoE, MQ2/MQ3-Lloyd expert variants, and MTP. Missing CASK, PP, and Qwen-style DFlash integration. |
| LFM2.5-MoE | `lfm2.5-8b-a1b` | Supported as LFM2.5-MoE (`arch_id=11`) when compiled with `arch-lfm2moe`. Minimal AR bring-up. | No. | No. | No. | No. | No; prefill is per-token `decode_step`. | No completed refs yet. | Short-conv, attention, router, top-4 MoE, MQ4/MQ6 expert kernels are present. Missing batched prefill, DFlash/spec decode, CASK, PP, and grammar/tool-exec integration. |
| Dense LFM2.5 | `lfm2.5-350m`, `lfm2.5-1.2b-instruct` | Not supported as dense LFM2. Local MQ artifacts stamped `arch_id=11` are suspect because the LFM2-MoE parser requires MoE-only fields. | No. | No. | No. | No. | No. | KLD producer currently skipped on error for dense LFM2 rows. | Needs a dense LFM2 architecture crate or a generalized LFM2 loader. Current `hipfire-arch-lfm2moe` kernels/config assume `lfm2_moe` layer types, experts, and MoE FFN fields. |
| LLaMA-family dense | `llama-3.2-1b-instruct`, `supra-50m-instruct` | Basic dense AR support through LLaMA-family path (`arch_id=0`). | No. | No. | No. | No. | No Qwen35-style batched prefill. | Producer skipped on error for `llama-3.2-1b-instruct-bf16` and `supra-50m-instruct-bf16`. | Dense LLaMA/GGUF-style GEMV, Q8/HFQ/MQ weight paths exist. Missing family-specific optimized prefill, DFlash, CASK, PP, and per-model quality refs. |
| Gemma 4 | `gemma-4-E2B-it` | Not runnable as a Gemma architecture yet. Prompt/tool-call support scaffolding exists, but no Gemma4 architecture crate is in the workspace. | No. | No. | No. | No. | No. | Not generated. | Needs `hipfire-arch-gemma4`, config/loader/forward kernels, and stop/tool-call policy wiring. Existing Gemma parser support is not model execution support. |
