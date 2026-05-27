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
| `qwen3.5:4b` | 2.6 GB | 4 GB | Best speed/quality balance |
| `qwen3.5:9b` | 5.3 GB | 6 GB | Default `serve` pre-warm |
| `qwen3.5:27b` | 15 GB | 16 GB | Needs 16 GB+ VRAM |
| `qwen3.5:35b-a3b` | 18.7 GB | 22 GB | MoE 35B / 3B-active. Local-only (no HF repo yet). |
| `qwen3.6:27b` | 15 GB | 16 GB | 3.6 refresh, same hybrid arch as 3.5 |
| `qwen3.6:35b-a3b` | 18.7 GB | 22 GB | 3.6 MoE refresh. Local-only. |
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
hipfire config set-model qwen3.6:35b-a3b max_think_tokens 1024  # per-model
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
