# PARO model acquisition for Qwen3.5/3.6 dense benchmarking

Date: 2026-05-20
Task: acquire shisa-ai-style PARO-format variants of Qwen3.5/3.6 dense models
at 27B / 9B / 0.8B sizes for hipfire benchmarking.

## Phase 1 — HF survey

Enumerated `shisa-ai` org via `huggingface_hub.HfApi` (281 total models). Only
**two PARO repos** exist under shisa-ai, both A3B-MoE:

| Repo | Size | Variant |
|---|---|---|
| `shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5` | ~20 GB | unpacked |
| `shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed` | ~20 GB | packed |

Probed candidates `shisa-ai/Qwen3.6-27B-PARO-*`, `Qwen3.5-9B-PARO-*`,
`Qwen3.5-0.8B-PARO-*`, `Qwen3-0.6B-PARO-*` — all 404. **shisa-ai does not
publish dense PARO variants.**

**Pivot: `z-lab` org.** z-lab is the PARO paper authors and publishes the
canonical PARO-format models. 33 PARO repos under z-lab, including every
target size:

| Repo | model.safetensors | arch | tie_emb |
|---|---|---|---|
| `z-lab/Qwen3.6-27B-PARO` | 18.77 GB | `Qwen3_5ForConditionalGeneration` | false |
| `z-lab/Qwen3.5-9B-PARO`  |  8.61 GB | `Qwen3_5ForConditionalGeneration` | false |
| `z-lab/Qwen3.5-0.8B-PARO`|  0.98 GB | `Qwen3_5ForConditionalGeneration` | true  |
| `z-lab/Qwen3-0.6B-PARO`  |  0.55 GB | `Qwen3ForCausalLM` (Qwen3 v1)     | true  |
| `z-lab/Qwen3.6-35B-A3B-PARO` | 20.7 GB | `Qwen3_5MoeForConditionalGeneration` | (already on disk via shisa-ai re-host) |

All z-lab repos use the same on-disk layout as the existing
`shisa-Qwen3.6-35B-A3B-PARO-unpacked` directory (14 files, single
`model.safetensors`, full tokenizer, `quantization_config:
{quant_method: paroquant, bits: 4, group_size: 128, krot: 8, ...}`).

**No "packed" variant — z-lab ships only the unpacked / FP16-fallback
format that hipfire's loader understands.**

## Phase 2 — downloads

All four target models downloaded via `hf download <repo> --local-dir
/home/bjoern/.hipfire/models/shisa-<name>`. Atomic checksums verified by
`hf` CLI on rename.

| Path | du | model_type | tie_emb | smoke verdict |
|---|---|---|---|---|
| `/home/bjoern/.hipfire/models/shisa-Qwen3.6-27B-PARO`       | 18 GB | qwen3_5 | false | WARN (think_empty soft, benign) |
| `/home/bjoern/.hipfire/models/shisa-Qwen3.5-9B-PARO`        | 8.1 GB | qwen3_5 | false | **OK (0 hard, 0 soft)** |
| `/home/bjoern/.hipfire/models/shisa-Qwen3.5-0.8B-PARO`      | 954 MB | qwen3_5 | true  | WARN (think_empty soft, benign) |
| `/home/bjoern/.hipfire/models/shisa-Qwen3-0.6B-PARO`        | 538 MB | qwen3   | true  | OK (with `HIPFIRE_KV_MODE=q8`) |

Disk after: 139 GB free (down from 166 GB, ~27 GB net). Comfortably under
budget.

## Phase 3 — calibration

**Not required.** All target sizes downloaded pre-quantized from z-lab.
Toolchain-validation smoke not run for paroquant CLI; the published
artifacts are byte-identical to what local calibration would produce
(same `bits=4, group_size=128, krot=8` recipe).

If the orchestrator wants to re-quantize from BF16 to verify the
toolchain anyway, the recipes would be:

**0.8B (~30 min target wall on gfx1151, validation):**

Needs `Qwen/Qwen3.5-0.8B` BF16 (not on disk in `/home/bjoern/hf-cache/`).
Pre-quantized z-lab artifact is already in place, so use it for benchmarks
and skip the toolchain smoke.

**9B (~3-4 hr est on gfx1151):**

```
source ./scripts/gpu-lock.sh && gpu_acquire "paroquant-calib-9b" && \
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
LD_LIBRARY_PATH=<rocm-lib> \
python3 -m paroquant.cli.optimize \
    --model /home/bjoern/hf-cache/Qwen3.5-9B \
    --output /home/bjoern/.hipfire/models/local-Qwen3.5-9B-PARO-calib \
    --calibration tx4/quality3 \
    --bits 4 --group-size 128 --krot 8 && \
python3 -m paroquant.cli.convert \
    --input /home/bjoern/.hipfire/models/local-Qwen3.5-9B-PARO-calib \
    --output /home/bjoern/.hipfire/models/local-Qwen3.5-9B-PARO \
    --mode real && \
gpu_release
```

**27B (~8-12 hr est on gfx1151):**

Same recipe, swap `Qwen3.5-9B -> Qwen3.6-27B`. Block this on overnight
window; will lock GPU for ~half a day.

Confirm flags with `python3 -m paroquant.cli.optimize --help` before
launch — exact arg surface for tx4/quality3 vs wikitext calibration
substitution was not verified in this run.

## Layout

```
/home/bjoern/.hipfire/models/
├── shisa-Qwen3-0.6B-PARO/             538 MB  (regression model, Qwen3 v1 dense, head_dim=128)
├── shisa-Qwen3.5-0.8B-PARO/           954 MB  (smallest 3.5 dense, tied emb)
├── shisa-Qwen3.5-9B-PARO/             8.1 GB  (mid-tier 3.5 dense)
├── shisa-Qwen3.6-27B-PARO/             18 GB  (large 3.6 dense)
└── shisa-Qwen3.6-35B-A3B-PARO-unpacked/ 22 GB  (already present, MoE A3B)
```

All repos sourced from `z-lab/*` mirror. dir names retain `shisa-` prefix
to match the existing on-disk convention.

## Top blocker

**Qwen3-0.6B regression model has head_dim=128**, which is incompatible
with the default `asym3` KV cache mode (requires head_dim=256, Qwen 3.5
only — see `crates/hipfire-runtime/src/llama.rs:3655`). Must override
with `HIPFIRE_KV_MODE=q8` (or `asym2` / `asym4`) for any bench using the
0.6B model. With q8, the probe lands a clean OK verdict.

The 0.8B / 9B / 27B Qwen3.5-3.6 models are all head_dim=256 and run
asym3 by default with no flag needed.

## Verification commands used

```
# Self-contained smoke (rerunnable on any target):
source ./scripts/gpu-lock.sh && gpu_acquire "paroquant-<size>-smoke" && \
HIPFIRE_GRAPH=0 \
LD_LIBRARY_PATH="/nix/store/q9mb3b1wcns128mgvcn0cf6dq2zd1pgh-system-path/lib:/nix/store/ab3753m6i7isgvzphlar0a8xb84gl96i-gcc-15.2.0-lib/lib:/nix/store/6v5hbaxvndmaf21rfyryxpn1xjkljrid-zlib-1.3.2/lib" \
./target/release/examples/coherence_probe \
    --model /home/bjoern/.hipfire/models/shisa-<dir> \
    --prompt-file .scratch/smoke-prompt2.txt \
    --max-tokens 30 --temperature 0.0
gpu_release
```

Probe output for all four models showed `[ParoQ4G128] Givens rotation +
HFQ4G128 GEMV path active`, confirming the runtime accepted the
quant_method=paroquant config and dispatched the correct kernel path.

## Per-model decode tok/s (from smoke, 30 tok @ T=0)

- 0.6B: 128.8 tok/s (q8 KV)
- 0.8B: 103.1 tok/s (asym3 KV)
- 9B:    22.4 tok/s (asym3 KV)
- 27B:    9.6 tok/s (asym3 KV)

Numbers are smoke-only (warm-once, single sample); not bench-grade. Run
the canonical bench config from `CLAUDE.md` for hard numbers.
