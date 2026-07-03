# Dispatch 1.2 — Paro fused-kernel GPU verification handoff

**Branch:** `feature/dispatch-unification`
**Commits:** `6da3c7bb` (commit 1) → `284c119e` (commit 2) → `1662eb5c` (commit 3)
**Who:** Any dev with a gfx1100/gfx1150/gfx1201 GPU + ≥24 GB VRAM
**Models:** Downloaded at `/local/models/z-lab/` on this workstation

## What changed

Ship 1.2 adds three ParoQ4G128T fused-kernel entries to the dispatch pipeline:
- `FusedGateUpParo4G128T` — 2-way gate+up (FFN)
- `FusedQkvzaParo4G128T` — 4-way QKVZA (DeltaNet linear attention)
- `FusedQkvParo4G128T` — 3-way QKV (FullAttn, via 4-way kernel with `m3=0`)

Before 1.2, ParoQ4G128 fell through to per-op GEMVs (4 launches per QKVZA, 2
per gate+up). After 1.2, the fused path fires: `fused_qkvza_paro4g128t` (2
launches for 4 projections) and `fused_gate_up_paro4g128t` (2 launches for 2
projections). This restores master's fused path through the pipeline.

## Models on disk

| Model | Path | Size | Needs VRAM |
|---|---|---|---|
| Qwen3.5-9B-PARO | `/local/models/z-lab/Qwen3.5-9B-PARO` | 8.1 GB | ~12 GB |
| Qwen3.6-27B-PARO | `/local/models/z-lab/Qwen3.6-27B-PARO` | 18 GB | ~24 GB |

Transfer these to the GPU bench machine (rsync, scp, etc.) or re-download
from `z-lab/Qwen3.5-9B-PARO` / `z-lab/Qwen3.6-27B-PARO` on HuggingFace.

## Verification checklist

### 1. Build from branch

```bash
git checkout feature/dispatch-unification
cargo build --release --example coherence_probe -p hipfire-runtime
cargo build --release --example dflash_spec_demo -p hipfire-runtime  # optional
```

### 2. Smoke test — fused path fires

```bash
./target/release/examples/coherence_probe \
    --model /path/to/Qwen3.5-9B-PARO \
    --prompt "Write a Python function to reverse a linked list" \
    --max-tokens 30 --temperature 0.0
```

In debug builds, you should see `[dispatch] Paro fused arm fired: FusedGateUpParo4G128T`
and `[dispatch] Paro fused arm fired: FusedQkvzaParo4G128T` on stderr for each
forward pass. If you see *only* per-op GEMV output and no "Paro fused arm fired",
the guards didn't match — that's a silent perf no-op (Risk #1).

### 3. Byte-identical vs master (primary oracle)

Run the same prompt at temp 0.0 on both `master` and this branch, capture
`HIPFIRE_EMIT_TOKEN_IDS=1` output, and `md5sum` the token ID sequences. They
must be **byte-identical** — same fused kernel, same accumulation order.

```bash
# On master:
git stash && git checkout master
cargo build --release --example coherence_probe -p hipfire-runtime
HIPFIRE_EMIT_TOKEN_IDS=1 \
./target/release/examples/coherence_probe \
    --model /path/to/Qwen3.5-9B-PARO \
    --prompt "The capital of France is" \
    --max-tokens 64 --temperature 0.0 \
    2> /tmp/master_tokens.txt

# On branch:
git checkout feature/dispatch-unification && git stash pop
cargo build --release --example coherence_probe -p hipfire-runtime
HIPFIRE_EMIT_TOKEN_IDS=1 \
./target/release/examples/coherence_probe \
    --model /path/to/Qwen3.5-9B-PARO \
    --prompt "The capital of France is" \
    --max-tokens 64 --temperature 0.0 \
    2> /tmp/branch_tokens.txt

# Compare:
md5sum /tmp/master_tokens.txt /tmp/branch_tokens.txt
```

### 4. Force-unfused: coherence + cosine (NOT byte-exact)

Per-op and fused are different kernels with different FP reduction order.
`HIPFIRE_FORCE_UNFUSED=1` must produce coherent output, but token IDs will
differ from the fused path.

```bash
HIPFIRE_FORCE_UNFUSED=1 \
./target/release/examples/coherence_probe \
    --model /path/to/Qwen3.5-9B-PARO \
    --prompt "The capital of France is" \
    --max-tokens 64 --temperature 0.0
```

Verify: output is coherent (not garbage, not stuck in a loop). Cosine
similarity of the token-embedding vectors between fused and unfused should
be ≥ 0.9999 (but exact token IDs will differ).

### 5. Perf comparison: fused vs per-op

```bash
# Fused (branch default):
./target/release/examples/coherence_probe \
    --model /path/to/Qwen3.5-9B-PARO \
    --prompt "Write a Python quicksort" \
    --max-tokens 256 --temperature 0.0

# Per-op fallback:
HIPFIRE_FORCE_UNFUSED=1 \
./target/release/examples/coherence_probe \
    --model /path/to/Qwen3.5-9B-PARO \
    --prompt "Write a Python quicksort" \
    --max-tokens 256 --temperature 0.0
```

Fused should show a measurable tok/s gain over per-op (fewer kernel launches).
If fused is slower or identical, the guard may not have fired — check debug
stderr for the "Paro fused arm fired" messages.

### 6. Coherence gate

```bash
./scripts/coherence-gate.sh
./scripts/coherence-gate-dflash.sh  # if DFlash draft is available
```

### 7. Multi-GPU path (if ≥2 GPUs)

The multi path (`forward_scratch_layers_multi`) is still per-op (deferred to
Ship 5). Verify it still works without regression on a multi-GPU setup if
available — not blocking for single-GPU verification.

## Arch coverage

| Arch | GPU example | Must test |
|---|---|---|
| gfx1100 | RX 7900 XTX | Yes (dp4a) |
| gfx1150 | Radeon 8060S | Yes (dp4a) |
| gfx1151 | Radeon PRO W7900 | Yes (dp4a) |
| gfx1201 | RX 9070 XT | Yes (dp4a) |

All Paro keys use `ArchPredicate::HasDp4a` — must verify on at least one
dp4a-capable arch. Ideally two (gfx1100 + gfx1201).

## Known facts (from code review)

- Step pattern for Paro: `[Rmsnorm(None), Gemv(Raw)×N]` — confirmed in all
  three `qwen35.rs` helpers (qkvza_via_execute_steps, qkv_via_execute_steps,
  gate_up_via_execute_steps)
- Rotation is `None` (plain rmsnorm), not `Givens` — the kernel rotates
  per-weight internally. Double-rotation would produce garbage.
- Gate+up: 1 explicit `rot_scratch[0]` + kernel-internal `mq_x_rot` as
  `x_rot_up`. Debug assert that these don't alias.
- QKV 3-way: synthesized via 4-way kernel with `m3=0` (no 4th write),
  `a3=wq` (aliased), `y3=q` (aliased), `x_rot3=rs[0]` (aliased, unused).
- Alignment: `m%8==0` and `k%128==0` — guaranteed by ParoQ4G128 group-128
  quant and qwen35 hidden dims.

## What to report

1. Token ID md5 (master vs branch) — must be identical
2. Force-unfused coherence verdict — must be coherent (not byte-exact)
3. Tok/s: fused vs per-op vs master — fused should match or beat master
4. Any panics, garbage output, or missing "Paro fused arm fired" messages
5. GPU model + ROCm version + arch (gfx####)
