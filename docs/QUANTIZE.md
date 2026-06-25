# Quantizing Models

`hipfire-quantize` converts Hugging Face safetensors directories, GGUF files, or
source-precision `.hfq` files into HipFire `.hfq` artifacts.

## Basic Usage

```bash
cargo run --release -p hipfire-quantize -- \
  --input /srv/huggingface/models--Qwen--Qwen3.5-9B/snapshots/<snapshot> \
  --output ~/.hipfire/models/qwen3.5-9b-oq4.hfq \
  --format oq4
```

The output filename should use the canonical artifact convention:

```text
<family>-<version>-<size>[-<variant>][-<role>]-<format>[+<features>].hfq
```

Examples:

```text
qwen3.5-9b-oq4.hfq
qwen3.5-9b-oq4+.hfq
qwen3.5-9b-oq8.hfq
```

## Opus Quant Names

Public Opus Quant spellings are:

| Format | Meaning | Quantizer aliases |
|---|---|---|
| `oq4` | 4-bit Opus Quant | `opus` |
| `oq4+` | `oq4` plus AWQ+LDLQ calibration | none; use `--format oq4+` with calibration flags |
| `oq8` | 8-bit Opus Quant | `opus8` |

The Rust code still uses older internal enum names such as `Oq4G256` and
`Oq8G256`; those are implementation details.

`oq4+` is not just a 4-bit Opus Quant artifact. The plus means the OQ4 weights
were produced with AWQ plus LDLQ calibration. Without those inputs, the artifact
is plain `oq4`.

For a quality-gated `oq4+` artifact, provide calibration inputs:

```bash
cargo run --release -p hipfire-quantize -- \
  --input <source-model> \
  --output ~/.hipfire/models/<name>-oq4+.hfq \
  --format oq4+ \
  --awq \
  --ldlq \
  --hessian <model>.hessian.bin
```

Current caveat: `--ldlq` for Opus Quant reads the legacy HFHS
`*.hessian.bin` sidecar. The newer unified `*.calib.hfq` collector format is not
yet wired into this specific OQ4 LDLQ path.

## Useful Flags

| Flag | Use |
|---|---|
| `--chat-template-file <path>` | Override the chat template embedded from the source model. |
| `--threads <n>` | Set Rayon worker threads. Defaults to 80% of host cores. |
| `--imatrix <path>` | Load llama.cpp imatrix GGUF data for activation-aware calibration. |
| `--awq` / `--awq-alpha <f>` | Enable activation-aware weight pre-scaling. Requires imatrix data or a Hessian-derived imatrix. |
| `--ldlq` | Enable full-Hessian error-feedback packing for OQ4. Requires `--hessian`. |
| `--arch-id <id>` | Override the architecture id stamped in the `.hfq` header. |

After producing a portable OQ4 artifact, use `hipfire repack` to pre-pack it for
a specific GPU architecture:

```bash
hipfire repack ~/.hipfire/models/qwen3.5-9b-oq4.hfq --arch gfx1103
```
