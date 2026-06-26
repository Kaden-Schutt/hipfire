# Quantizing Models

`hipfire-quantize` converts Hugging Face safetensors directories, GGUF files, or
source-precision `.hfq` files into HipFire `.hfq` artifacts.

## Basic Usage

```bash
cargo run --release -p hipfire-quantize -- \
  --input /srv/huggingface/models--Qwen--Qwen3.5-9B/snapshots/<snapshot> \
  --output ~/.hipfire/models/Qwen3.5-9B.oq4.hfq \
  --format oq4
```

The output filename should use the canonical artifact convention:

```text
<family>[-]<version>-<size[-effective/active]>[-tag1][-tag2...][.feature1[-feature2...]].<format>[.arch].hfq
```

Examples:

```text
Qwen3.5-9B.oq4.hfq
Qwen3.5-9B.oq4+.hfq
Qwen3.5-9B.oq4++.hfq
Qwen3.5-9B.mq4++.gfx1103.hfq
```

## Quant Token Taxonomy

Quant tokens describe weight encoding only:

```text
<family><bitwidth>[+][+]
```

- `mq` / `MQ` is affine Magnum Quant.
- `oq` / `OQ` is symmetric Opus Quant. Do not use `op` for new artifacts.
- A first `+` means clip-search, SmoothQuant, AWQ, or a comparable
  activation-aware clipping/scaling pass.
- A second `+` means Hessian/LDLQ error feedback.
- Mixed precision includes a decimal place in the bitwidth, for example
  `mq4.5+` or `oq4.25++`.

Do not use `+` for bundled runtime features or sidecars. Encode those as dot
groups before the quant token, for example `Qwen3.5-9B.mtp-vl.oq4.hfq` or
`Gemma-4-8B.dflash-triattn.oq4++.gfx1151.hfq`.

## Public Quant Names

| Format | Meaning | Quantizer aliases |
|---|---|---|
| `mq4` | 4-bit affine Magnum Quant | none |
| `mq4+` | `mq4` plus clip-search/SmoothQuant/AWQ-style calibration | none |
| `mq4++` | `mq4+` plus Hessian/LDLQ error feedback | none |
| `oq4` | 4-bit symmetric Opus Quant | legacy `op4` parser path, `opus` |
| `oq4+` | `oq4` plus clip-search/SmoothQuant/AWQ-style calibration | none |
| `oq4++` | `oq4+` plus Hessian/LDLQ error feedback | legacy `op4+` parser path |
| `oq8` | 8-bit symmetric Opus Quant | legacy `op8` parser path, `opus8` |

The Rust code still uses older internal enum names such as `Oq4G256` and
`Oq8G256`; those are implementation details.

The plus marks are positional. `oq4+` means activation-aware clipping/scaling
without Hessian/LDLQ feedback. `oq4++` means the same first-stage calibration
plus Hessian/LDLQ feedback. Without those inputs, the artifact is plain `oq4`.

For a quality-gated `oq4++` artifact, provide calibration inputs:

```bash
cargo run --release -p hipfire-quantize -- \
  --input <source-model> \
  --output ~/.hipfire/models/<name>.oq4++.hfq \
  --format oq4++ \
  --awq \
  --ldlq \
  --hessian <model>.hessian.bin
```

If the local quantizer parser has not yet been renamed, use the legacy parser
flag for the same path while keeping the output artifact canonical:

```bash
  --format op4    # canonical artifact token oq4
  --format op4+   # canonical artifact token oq4++
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
| `--awq` / `--awq-alpha <f>` | Enable the first `+`: activation-aware weight pre-scaling. Requires imatrix data or a Hessian-derived imatrix. |
| `--ldlq` | Enable the second `+`: full-Hessian error-feedback packing. Requires `--hessian`. |
| `--arch-id <id>` | Override the architecture id stamped in the `.hfq` header. |

After producing a portable OQ4 artifact, use `hipfire repack` to pre-pack it for
a specific GPU architecture:

```bash
hipfire repack ~/.hipfire/models/Qwen3.5-9B.oq4++.hfq --arch gfx1103
```
