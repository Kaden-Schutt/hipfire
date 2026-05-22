# Branch: feat/paro-g256-perfmax — Agent Brief

> READ THIS FIRST. The accompanying `GOAL.md` has the structured mission. This
> file has the *what's here, where to start* orientation.

## Branch state

This branch is `feat/paro-g256-perfmax`, the union of:

1. **Björn Bösel (fivetide)**'s ParoQuant runtime PRs:
   - **PR #316** `feat/paroquant-native` — Qwen3.6-A3B-MoE working through hipfire, KLD **0.0933** PPL **6.39** (10× over MQ4 baseline 0.946). Three forward-path fixes: GemmaRMSNorm `(1+w)` bake on PARO load path, MoE loader with aliased PARO sidecars, lm_head conditional load. +5 supporting commits.
   - **PR #317** `fix/moe-hipgraph-atomicadd` — atomic-free MoE down for hipGraph determinism (task #100).
   - **PR #318** `feat/paroquant-graph-capture` — full hipGraph capture support, **+30% decode** (30.0 → 38.9 tok/s) byte-identical to direct. Root-caused via 8 hypotheses to flash_mode=2 default on gfx11/gfx12.

2. **codex/paro-reconcile-milestone** — G256 reconciliation work:
   - `docs/plans/paroquant-g256-milestone.md` — G256 milestone PRD
   - `docs/plans/astrea-paro4-model-agnostic-pipeline.md` — production pipeline plan
   - `docs/investigations/2026-05-14-paroquant-hiptrx-baseline/README.md` — measured baseline on hiptrx gfx1201 (paroquant 186.6 tok/s decode with PARO4G128T engine layout)
   - `scripts/paroquant_g256_probe.py` — CPU-only G256 format probe
   - `scripts/paroquant_import.py` — safetensors → HFQ importer with `--layout native|engine`
   - `scripts/paroquant_oracle.py` — bit-exact source/HFQ verification
   - `scripts/paro_layer0_debug.py` — layer-0 debug tool

## Two coexisting PARO runtime paths

Per `docs/plans/paroquant-g256-milestone.md`:

| Path | DType | Status | Owner |
|---|---|---|---|
| **New** | `ParoQ4G128` (HFQ4G128 + ParoRotation sidecars) | Active production path via PR #318. Graph-capture works. A3B verified KLD 0.093. | Björn (#316-#318) |
| **Old** | `PARO4G128` / `PARO4G128T` (qtype-28/29) | Investigation/probe path. PARO4G128T engine layout = +84% over native (186.6 vs 101.3 tok/s on gfx1201). | codex local probe |

The PRD says: *"keep both visible until the G256 gate decides whether to invest in a production `PARO4G256_MQ` runtime."* That gate decision is what this branch is for.

## The four permutations to evaluate

| Format | Group size | Layout | Status | Notes |
|---|---:|---|---|---|
| `PARO4G128` | 128 | native | EXISTS (`kernels/src/gemv_paro4g128.hip`, 1081 LOC) | qtype-28, baseline |
| `PARO4G128T` | 128 | transposed (engine) | EXISTS in `dispatch.rs:2302-2772` | qtype-29, **+84%** over native |
| `PARO4G256` | 256 | native | **NEW — needs creation** | half the per-group metadata, coarser quant |
| `PARO4G256T` | 256 | transposed | **NEW — needs creation** | engine-layout variant of G256 |

G256 trade-off: smaller scale/zero metadata (~50% reduction on sidecars) means more BW-efficient inference but potentially worse quality per group (256 elems sharing one scale vs 128). The G256 probe at `scripts/paroquant_g256_probe.py` evaluates the quality cost CPU-side without committing to runtime work.

## Two perf levers already named in the baseline doc

From `docs/investigations/2026-05-14-paroquant-hiptrx-baseline/README.md`:

**1. Fuse `paro4g128t_rotate` into subsequent GEMV.**
Currently 79.8 ms / 24.3% of decode at 6.7 GiB/s standalone. Mirror MQ4's `fused_rmsnorm_mq_rotate` pattern. Estimated **+30% tok/s** if rotate cost goes to zero.

**2. Batched QKV GEMV.**
Currently 3 separate `gemv_paro4g128t_prerotated` calls per layer. MQ4 collapses to one `fused_qkvza_hfq4g256` at 265.5 GiB/s. Same fusion structurally applies.

## Where to start

```bash
# verify HEAD
git log -1 --oneline   # should show: a09af869 merge: integrate codex/...

# build sanity (in this worktree)
cargo build --release --bin hipfire-quantize -p hipfire-quantize
cargo build --release --example eval_hipfire --features arch-qwen35,deltanet -p hipfire-runtime
cargo build --release --example test_gemv_paro4g128 -p rdna-compute

# CPU-only G256 probe (decides if format is worth runtime work)
python3 scripts/paroquant_g256_probe.py --help
```

## Inputs available on droplet `mi300`

- `/workspace/paroquant/a3b-paro/` (shisa-ai Qwen3.6-35B-A3B-PARO-packed, 20 GB)
- `/workspace/hf-models/qwen3.6-27b/` (52 GB BF16)
- `/workspace/hf-models/qwen3.6-35b-a3b/` (67 GB BF16)
- HF cache: 0.8B + 9B in `/root/.cache/huggingface/hub/`
- BF16 GGUFs (for tokenizer parity): `/workspace/qwen3.{5,6}-*-bf16.gguf`
- KLDrefs: `/workspace/kldref/*.kldref.bin`
- Mix-v1 corpus: `/workspace/calibration-mix-v1.txt` (md5 `68a1d2e62117e692e0e04c2811349aaf`)
- Mix-v1 Hessians: `/workspace/qwen3.5-0.8b.mix.ctx{2048,4096}.hessian.bin`
- Mix-v1 imatrix: `/workspace/qwen3.5-0.8b.mix.ctx4096.imatrix.gguf`

## Existing perf baselines (for comparison)

| Model | Format | Arch | Prefill tok/s | Decode tok/s |
|---|---|---|---:|---:|
| A3B uniform-MQ4 | MQ4G256 | gfx1201 (R9700) | 2966 (256 ctx) | 57 |
| A3B uniform-MQ4 | MQ4G256 | gfx1151 (Strix Halo) | 1352 | — |
| A3B-PARO native (Björn #316) | PARO native | gfx1151 | 31.4 | 30.8 |
| A3B-PARO + #318 graph capture | PARO native | gfx1151 | 31.4 | **38.9** |
| 0.8B uniform-PARO (codex probe) | PARO4G128T | gfx1201 | 193.4 | 186.6 |

The gap: A3B-PARO at 31 tok/s prefill vs A3B-MQ4 at 1352 = ~43× slower today. Engine layout + the two named fusions are predicted to close most of that.

## Quality baseline already measured

| Variant | KLD | NLL | PPL |
|---|---:|---:|---:|
| A3B mq4-kmap1 (no AWQ) | 0.9566 | 2.6482 | 14.13 |
| A3B mq4-awq-f1-a025 (best F1 AWQ) | 0.9460 | 2.6336 | 13.92 |
| A3B hfq6 (pure 6-bit) | 0.9500 | 2.6486 | 14.13 |
| **A3B-PARO-unpacked (Björn #316)** | **0.0933** | **1.8552** | **6.39** |
| 0.8B-PARO -32-36% PPL vs MQ4 (from local probe investigation) | — | — | — |

Per PR #316: AWQ+GPTQ has a **structural floor** at ~0.95 KLD on MoE because per-row weight optimizers can't capture routing-conditioned activation variance — needs rotation-layer DOF, which ParoQuant provides. This branch is the path to ship that quality at competitive perf.

## Non-negotiables

- This is a **research / perfmax branch**, not master-merge candidate yet.
- For perf claims: follow `docs/methodology/perf-benchmarking.md` — fresh process per measure, gpu-tcas coordinated, byte-identical prompt with md5.
- The Δ ≥ 5% investigation rule applies: any kernel-level perf delta crossing ±5% needs warming verify (3-5 fresh runs, median).
- Coherence-gate required for any default-flip; not required for opt-in research kernels.
- All commits use git author `noreply@anthropic.com` per `feedback_git_identity_noreply` memory.
- mi300x droplet billed hourly — autonomous operation expected once the agent has the mission.
