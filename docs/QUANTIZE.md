# Quantize

`hipfire quantize` is the user-facing wrapper around the CPU-only
`hipfire-quantize` binary. It converts HuggingFace safetensors, a local
model directory, or a GGUF file into hipfire’s HFQM container (extensions
`.mq4`, `.hf4`, `.mq6`, …). The daemon mmaps the result directly.

Design / math: [QUANTIZATION.md](QUANTIZATION.md).
HFP4 family detail: [quant-formats/hfp4.md](quant-formats/hfp4.md).

## Pick a format

| Format | Bitwidth | Rotation | Typical use |
|---|---|---|---|
| `mq4` | 4 | FWHT | Qwen 3.5+ (DeltaNet hybrid). **Default for safetensors / HF input.** |
| `mq6` | 6 | FWHT | Qwen 3.5+ when you can spare ~+47% size for quality |
| `hf4` (`hfq4`, `hfq4g256`) | 4 | none | Dense Llama / Mistral / Gemma / older Qwen. **Default for GGUF.** |
| `hf6` (`hfq6`, `hfq6g256`) | 6 | none | Dense, higher quality |
| `q8` / `q8f16` | 8 | none | Reference / debug (safetensors path) |
| `mq3` | 3 | FWHT | Sub-4-bit — **`hipfire-quantize` direct only** (thin `hipfire quantize` rejects it); quality-sensitive — see QUANTIZATION.md |
| `hfp4` / `mfp4` / E8 / Lloyd | varies | see docs | Advanced; usually via `hipfire-quantize` directly |

FWHT (the “M” in MQ) applies the two-sign rotation
`R = D2 · H · D1 / 16` (seeds 42 / 1042) before quantization. Qwen 3.5+ kernels
apply the **same** `R` to activations (`rotate_x_mq`), not an inverse. On
Llama-style dense models MQ is correct math but adds rotate cost with little
quality benefit — prefer HFQ.

**Thin CLI allowlist** (`hipfire quantize`): `mq4`, `mq6`, `q8`, `q8f16`, and
`hf4`/`hf6` aliases only. **GGUF input** further narrows to `hf4`, `hf6`,
`mq4`, `mq6`. Formats such as `mq3`, HFP/MFP, Lloyd, and graded MoE need
`hipfire-quantize` directly.

**Not produced by the thin CLI path:** graded per-expert MoE recipes
(`mq4p`, tiered Lloyd, imatrix/Hessian GPTQ-E8, REAP overlays). Those need
extra flags (`--imatrix`, `--hessian-dir`, `--allow-mq*-lloyd`, …) on
`hipfire-quantize` directly. See the binary’s `--help` and QUANTIZATION.md.

Research / reserved formats require explicit opt-in on the binary:

| Flag / env | Opens |
|---|---|
| `--allow-mq2` / `HIPFIRE_ALLOW_MQ2=1` | uniform `mq2` (reserved — collapses) |
| `--allow-mq2-lloyd` | `mq2-lloyd` |
| `--allow-mq3-lloyd` | `mq3-lloyd` |
| `--allow-mq4-lloyd` | `mq4-lloyd` |

## From HuggingFace

```bash
hipfire quantize Jackrong/Qwopus3.5-4B-v3 \
    --both \
    --upload schuttdev/hipfire-qwopus-4b \
    --create-repo \
    --install \
    --register qwopus:4b
```

Downloads into `~/.hipfire/hf-cache/`, quantizes once per `--format`, optionally
uploads, copies into `~/.hipfire/models/`, and registers a local alias.

## From a local safetensors directory

```bash
hipfire quantize ./my-finetune/ --format mq4 -o my-finetune.mq4
```

Directory needs `config.json` plus one or more `.safetensors` files. The
quantizer accepts many architectures; inference only runs if a carrier/loader
exists for that `arch_id` (see [architecture-ids.md](architecture-ids.md)).

## From GGUF

```bash
hipfire quantize ./tinyllama.Q4_K_M.gguf \
    --install --register tinyllama:1b-gguf
# default --format hf4 → …hf4

hipfire quantize ./qwen3.5.Q4_K_M.gguf --format mq4 \
    --install --register q35:9b-gguf
```

GGUF tensor names are rewritten to HuggingFace-style names so
`load_weights_hfq` works. Tokenizer fields (`tokenizer.ggml.tokens` / merges /
bos/eos / model) are preserved under `gguf_meta` in the metadata blob;
`Tokenizer::from_hfq_metadata` reads them — the original GGUF need not stay on
disk.

Per-tensor selection in the GGUF pipeline:

| Tensor | Format |
|---|---|
| 1D norm / scale | F16 |
| embedding (`token_embd`) | Q8F16 |
| 2D weight, `K % 256 == 0` | chosen `--format` |
| 2D weight, K not multiple of 256 | HFQ4-G128 fallback (no rotation) |

Supported source GGUF dequant types: `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`,
`F16`, `BF16`, `F32`. Unsupported (quantizer errors on encounter): `Q5_0`,
`Q5_1`, and IQ* families.

### Quality caveat (GGUF)

GGUF is **double quantization**: dequant already-lossy weights, then requantize.
Expect worse quality than full-precision safetensors of the same model. Prefer
HF6/MQ6 if disk allows, or quantize from original safetensors.

## CLI flags (`hipfire quantize`)

| Flag | Purpose |
|---|---|
| `--format <fmt>` | Repeatable. Default: `mq4` (safetensors); `hf4` (GGUF). |
| `--both` | Shorthand for `--format mq4 --format mq6` |
| `-o, --output <path>` | Single-format output path |
| `--output-dir <dir>` | Multi-format output directory |
| `--stem <name>` | Override output basename |
| `--upload <owner/repo>` | Push outputs to HuggingFace |
| `--create-repo` | Create HF repo if missing |
| `--install` | Copy into `~/.hipfire/models/` |
| `--register <tag>` | Local alias for `hipfire run <tag>` |

Build the binary if missing:

```bash
cargo build --release -p hipfire-quantize
```

The CLI searches `target/release/hipfire-quantize` and
`~/.hipfire/bin/hipfire-quantize`.

## Direct `hipfire-quantize` (advanced)

Same formats plus MoE/REAP/AWQ/GPTQ controls. Common extras:

| Flag | Role |
|---|---|
| `--input` / `--output` | Paths |
| `--format` | Full alias set (`mq3`, `hfp4`, `mfp4`, `mfp4e8`, `mq3lloyd`, …) |
| `--imatrix <gguf>` | llama.cpp imatrix for activation-aware recipes |
| `--hessian-dir <dir>` | GPTQ-E8 Hessians |
| `--awq` / `--awq-alpha` | AWQ pre-scale (default alpha 0.55) |
| `--kmap-dense` / `--kmap-mode` / `--no-kmap` / `--uniform` | K-map promotion policy |
| `--q8-router` / `--no-q8-conv1d` | Protect routers / conv1d |
| `--reap-overlay` / `--reap-bake` / `--reap-out` / `--reap-arch` | REAP plan paths |
| `--arch-id` / `--force-arch-id` | Override written arch_id (qwen3* pillar guarded) |
| `--threads` / `HIPFIRE_QUANT_THREADS` | CPU thread cap (default ~80% cores) |
| `--include-vision` / `--vision-quant` | Vision tensors |
| `--allow-mq2` / `--allow-mq*-lloyd` | Research gates |

Graded MoE and E8 recipes are intentionally outside the thin `hipfire quantize`
help surface so accidental low-quality artifacts are harder to produce.

## Runtime cost

CPU- and memory-bandwidth bound. Rough desktop wall times (order-of-magnitude,
not a guarantee):

| Model size | Wall time |
|---|---|
| ~1B | tens of seconds |
| ~4B | ~1 min |
| ~9B | a few minutes |
| ~27B | several–tens of minutes |

Peak RAM is roughly `max(tensor elements) × 4` (one tensor dequantized to f32).

## After quantizing: CASK sidecar

For long-context CASK eviction calibration:

```bash
hipfire sidecar-gen my-finetune.mq4 --corpus /path/to/corpus.txt
# or via registered tag:
hipfire sidecar-gen finetune:1b --corpus /path/to/corpus.txt
```

Writes `my-finetune.mq4.triattn.bin` beside the model by default. The daemon
does not attach it by default. Set `cask_sidecar` to the exact path, or opt into
sibling discovery with `cask_auto_attach=true`. Set `cask=true` separately only
when core-aware m-folding is intended. See [CONFIG.md](CONFIG.md).

## Related

- Formats and KV: [QUANTIZATION.md](QUANTIZATION.md)
- Models / aliases: [MODELS.md](MODELS.md)
- CLI surface: [CLI.md](CLI.md)
- Sources: `crates/hipfire-cli/src/main.rs`, `crates/hipfire-quantize`
