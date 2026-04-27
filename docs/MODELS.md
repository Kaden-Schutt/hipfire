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

Higher-quality variants:

| Tag pattern | Effect |
|---|---|
| `qwen3.5:<size>-mq6` | 6-bit quant, +47% file size, closer-to-Q8 quality |

DFlash speculative-decode drafts (registry-paired):

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

`hipfire pull <target>` already prompts to also pull the matching
`-draft` if the registry has one. The CLI resolves both tags via the
registry and hands the daemon the literal target + draft file paths;
no on-disk filename pattern is involved. See
[ARCHITECTURE.md](ARCHITECTURE.md#dflash-speculative-decode) for what
DFlash is, [BENCHMARKS.md](BENCHMARKS.md) for the per-genre speedup
table.

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

## Model files on disk

```
~/.hipfire/models/
├── qwen3.5-9b.mq4                  # MQ4 (FWHT-rotated, Qwen3.5 hot path)
├── qwen35-9b-dflash-mq4.hfq        # DFlash draft for qwen3.5:9b (registry-paired)
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
