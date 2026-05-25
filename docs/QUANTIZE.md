# Quantize

`hipfire quantize` is a CPU-only tool that converts model weights into
hipfire's native quantized formats. Three input shapes are supported.
Output is a single file the daemon mmaps directly.

## Pick a format

| Format | Bitwidth | Rotation | When to use |
|---|---|---|---|
| `mq4` | 4-bit | FWHT (rotated) | Qwen 3.5+ targets — calibrated for the DeltaNet hybrid attention path. Default for safetensors input. |
| `mq6` | 6-bit | FWHT (rotated) | Qwen 3.5+ when you can spare +47% file size for quality. |
| `hf4` | 4-bit | none | Dense models (Llama, Mistral, Gemma, older Qwen). Default for GGUF input. |
| `hf6` | 6-bit | none | Dense, higher quality. |

The FWHT rotation in MQ4/MQ6 redistributes weight outliers across each
group before quantization. The Qwen 3.5 hot path applies the inverse
rotation in its kernels at inference; the Llama path can also undo it
via `gemv_mq4g256_with_rotate`, but adds runtime overhead with no
quality benefit on a model that wasn't trained against that weight
space. **MQ4 on a Llama-style dense model is correct math but slower
inference and no better quality** — pick HF4.

## From HuggingFace

```bash
hipfire quantize Jackrong/Qwopus3.5-4B-v3 \
    --both                                     # mq4 + mq6 in one shot
    --upload schuttdev/hipfire-qwopus-4b \
    --create-repo \
    --install \
    --register qwopus:4b
```

Auto-downloads the safetensors into `~/.hipfire/hf-cache/`, quantizes
once per `--format`, optionally uploads each output as its own file in
the target HF repo, copies into `~/.hipfire/models/`, and registers a
local alias.

Useful flags:

| Flag | Purpose |
|---|---|
| `--format <fmt>` | Repeatable. Defaults: `mq4` (safetensors), `hf4` (GGUF). |
| `--both` | Shorthand for `--format mq4 --format mq6`. |
| `-o, --output <path>` | Single-format output path. |
| `--output-dir <dir>` | Multi-format output directory. |
| `--stem <name>` | Override the output basename. |
| `--upload <owner/repo>` | Push outputs to HuggingFace. |
| `--create-repo` | Create the HF repo if missing. |
| `--install` | Copy outputs into `~/.hipfire/models/`. |
| `--register <tag>` | Add a local alias so `hipfire run <tag>` works. |

## From a local safetensors directory

```bash
hipfire quantize ./my-finetune/ --format mq4 -o my-finetune.mq4
```

Directory must contain `config.json` plus one or more `.safetensors`
files. Architectures the engine actually loads at inference: `llama`,
`qwen3`, `qwen3_5`, `qwen3_5_moe`. The quantizer accepts any
architecture — the file just won't run if the engine has no matching
loader.

## From GGUF

```bash
hipfire quantize ./tinyllama.Q4_K_M.gguf \
    --install --register tinyllama:1b-gguf
```

Default `--format` for GGUF is `hf4`. Override with `--format mq4` (or
`mq6`) only when the source is a Qwen 3.5+ family GGUF.

GGUF tensor names get translated to HuggingFace safetensors style at
write time so the engine's standard `load_weights_hfq` consumes the
output:

```
token_embd.weight       → model.embed_tokens.weight
output.weight           → lm_head.weight
output_norm.weight      → model.norm.weight
blk.{i}.attn_q.weight   → model.layers.{i}.self_attn.q_proj.weight
blk.{i}.ffn_gate.weight → model.layers.{i}.mlp.gate_proj.weight
...
```

The GGUF tokenizer (`tokenizer.ggml.tokens` / `merges` /
`bos_token_id` / `eos_token_id` / `model`) is preserved verbatim under
`gguf_meta` in the `.hf4` / `.mq4` metadata blob. The engine's
`Tokenizer::from_hfq_metadata` reads it directly — no need to keep the
original GGUF on disk.

Per-tensor format selection in the GGUF pipeline:

| Tensor shape | Format |
|---|---|
| 1D norm / scale (`*_norm.weight`) | F16 (precision-sensitive, small) |
| `token_embd.weight` (the embedding) | Q8F16 (Q4-grade is too lossy) |
| 2D weight, `K % 256 == 0` | per `--format` (mq4 / mq6 / hf4 / hf6) |
| 2D weight, K not divisible by 256 | HFQ4-G128 (no rotation fallback) |

Source GGUF dequant types supported:

```
Q4_0  Q8_0  Q4_K  Q6_K  F16  BF16  F32
```

Q5_K, Q5_0, Q5_1, IQ2_*, IQ3_*, IQ4_* are not implemented. The
quantizer panics on encounter. Adding one is a ~150-line port from
llama.cpp's `ggml-quants.c` to `crates/hipfire-quantize/src/gguf_input.rs`.

## Quality caveat for GGUF

GGUF input is double-quantization: the source weights are already
quantized once (typically Q4_K_M), and you're requantizing the
dequantized values. Each step accumulates error. Expect lower output
quality than quantizing from full-precision safetensors of the same
model. Mitigations:

- Pick HF6 / MQ6 if you have the disk space — the extra two bits
  absorb most of the double-quant noise.
- If you have access to the original safetensors, prefer that
  pipeline.

## Runtime

Quantization is CPU-bound and memory-bandwidth limited. Approximate
wall times on a modern desktop CPU:

| Model size | Wall time |
|---|---|
| 1B | 5–10 s |
| 4B | 30–60 s |
| 9B | 1–2 min |
| 27B | 4–8 min |

`hipfire-quantize` defaults to 80% of available cores; cap with
`--threads N` or `HIPFIRE_QUANT_THREADS=N`. Memory peak is roughly
`max(weight tensor size) × 4` (a single tensor dequantized to f32).

## After quantizing: generating a sidecar for CASK eviction

After you've quantized your own model, generate the KV cache calibration
sidecar so that CASK can perform intelligent key-value retention on long
context prompts. Without it, all positions are treated equally and early
tokens (which often carry critical instructions) may be evicted.

```bash
hipfire sidecar-gen my-finetune.mq4 --corpus /path/to/corpus.txt
```

The sidecar is written as `my-finetune.mq4.triattn.bin` next to your
model file by default. The daemon auto-discovers it when you enable a
CASK profile with `hipfire config cask-profile balanced`. See
[CLI.md](CLI.md) for full flag details.

> **Tip:** If you're quantizing from a HuggingFace safetensors source,
> use `--install` to copy your model into the models directory, then
> generate the sidecar using either the file path or the registered tag:
>
> ```bash
> hipfire quantize ./my-finetune/ --format mq4 -o my-finetune.mq4 \\
>     --install --register finetune:1b
> # Then generate the sidecar (either works):
> hipfire sidecar-gen my-finetune.mq4 --corpus /path/to/corpus.txt
> # or, if you prefer using the alias:
> hipfire sidecar-gen finetune:1b --corpus /path/to/corpus.txt
> ```
>
> The tag-based form works because `sidecar-gen` resolves local model
> aliases (registered via `--register`). For HuggingFace models not yet
> installed locally, use the file path instead.
