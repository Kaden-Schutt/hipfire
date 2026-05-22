# Goal — feat/paro-g256-perfmax

> Read `AGENT-BRIEF.md` first for branch context + inventory.

## Mission

Decide the **G256 gate** (per `docs/plans/paroquant-g256-milestone.md`) and **perfmax the ParoQuant runtime** so that PARO-quantized models approach uniform-MQ4 perf on both **A3B-MoE** and **dense** trunk models (0.8B / 9B / 27B / 27B-3.6).

You are autonomous. mi300x is billed hourly; keep it busy. No suspending for input until either the goal is satisfied or the work is structurally blocked. Use branches, commit often, push when stable, don't push to master.

## Exit conditions

Either of:

A. **All four permutations evaluated** (`PARO4G128`, `PARO4G128T`, `PARO4G256`, `PARO4G256T`) AND **at least one** perf lever shipped (rotate-fusion OR batched-QKV) so that A3B-PARO decode reaches ≥ 60% of A3B-MQ4 decode, OR

B. **G256 gate decided** (with empirical data, not opinion) — quality probe says G256 is unworkable (KLD blow-up > 3× G128 baseline) and you ship the G128-only perfmax stack with rotate-fusion + batched-QKV instead.

Either exit is acceptable. The bad outcome is "we still don't know."

## Phases

### Phase 1 — Quality probe: do PARO4G256 / PARO4G256T even make sense?

Use the CPU-only probe (no GPU needed for this phase):

```bash
python3 scripts/paroquant_g256_probe.py --help
# Run on shisa-ai 0.8B PARO checkpoint:
python3 scripts/paroquant_g256_probe.py \
    --paro-src ~/.hipfire/models/shisa-Qwen3.5-0.8B-PARO/ \
    --report-out docs/investigations/paro-g256-perfmax/g256-probe-0.8b.md
```

Outputs comparison of:
1. Source PARO4G128 oracle
2. PARO4G256-style AWQ regrouping
3. PARO4G256_MQ row-major HFQ4-G256 with same Paro rotation

**Gate criterion:** if G256 regrouping NRMSE / KLD-proxy ≤ 1.2× G128 baseline → invest in G256 runtime. If 1.5-2× → marginal, ship G256T only (engine layout has more BW headroom to compensate). If > 3× → kill G256, double down on G128 perfmax.

Run the probe on **at least** 0.8B and 9B PARO sources. A3B-MoE probe is optional (expert weights are smaller per-tensor, may have different G256 behavior).

Commit the probe outputs in `docs/investigations/paro-g256-perfmax/`.

### Phase 2 — Implement PARO4G256 / PARO4G256T (only if Phase 1 says go)

Pattern after the existing G128 implementation. Files to mirror:

- `kernels/src/gemv_paro4g128.hip` (1081 LOC) → `gemv_paro4g256.hip`
- Add `DType::PARO4G256` + `DType::PARO4G256T` to `crates/rdna-compute/src/dispatch.rs` (the G128 variants live around lines 2302-2772 — read them and pattern-match)
- Extend `scripts/paroquant_import.py` to emit the new qtype on `--group-size 256`
- Add a `test_gemv_paro4g256.rs` example (clone `test_gemv_paro4g128.rs`)
- Wire `qwen35.rs` load + dispatch for the new dtypes

Engine layout (G256T) gets the same two optimizations as G128T:
- `qweight` transposed to `[M/8, K]` for coalesced GEMV
- `theta` precomputed as `sincos_f32`

### Phase 3 — Perf levers (apply to whichever G-size survives Phase 1)

Two named fusions from `docs/investigations/2026-05-14-paroquant-hiptrx-baseline/README.md`:

**Lever 1 — Fuse `paro4g128t_rotate` into subsequent GEMV.**
Currently 79.8 ms / 24.3% of decode at 6.7 GiB/s standalone. The pattern to mirror is MQ4's `fused_rmsnorm_mq_rotate` (find it in `kernels/src/`). Create `gemv_paro4g128t_prerotated_fused_rmsnorm.hip` (or similar) that does rmsnorm + rotate + GEMV in one kernel. Estimated **+30% decode tok/s**.

**Lever 2 — Batched QKV GEMV.**
Currently 3 separate `gemv_paro4g128t_prerotated` calls per layer (Q, K, V). MQ4 collapses to one `fused_qkvza_hfq4g256` (see `kernels/src/`) at 265.5 GiB/s. Create `fused_qkv_paro4g128t.hip` (or `_paro4g256t` if G256 wins Phase 1) that batches the 3 GEMVs into one kernel with shared input read. Estimated **+15-25% decode tok/s** stacked.

### Phase 4 — A3B-specific perf work

PR #316's structural finding: ParoQuant gated rotations are required for MoE quality. Björn already shipped the loader + correctness for A3B-PARO (`paro_load_moe_ffn`, expert-aliased sidecars). The bottleneck is now the MoE FFN kernels themselves.

Required: per-expert `gemv_paro4g128t_moe_indexed_*.hip` kernels mirroring the existing MQ4 / HFQ4 MoE-indexed kernels (search `kernels/src/` for `*moe_*_indexed*`). For A3B (256 experts, k=8 active per token), the MoE down kernel is the prefill bottleneck.

Goal: A3B-PARO decode ≥ 60% of A3B-MQ4 decode (= ≥34 tok/s if MQ4 is at 57 on gfx1201).

### Phase 5 — Dense perf parity (0.8B / 9B / 27B / 27B-3.6)

For dense models, PARO4G128T is already at engine layout (+84%). Layered on top of Phase 3 fusions:

- 0.8B: target ≥ 90% of MQ4 decode (= ≥190 tok/s on gfx1201 if MQ4 hits ~210 — verify baseline first)
- 9B: target ≥ 80% of MQ4 decode (= ≥18 tok/s if MQ4 hits ~22 on gfx1201)
- 27B + 27B-3.6: target ≥ 75% of MQ4 decode

### Phase 6 — DFlash compatibility (stretch)

PARO-quantized drafters + DFlash spec-decode should work. May require porting fused tree-attention kernels to the PARO format. Optional — only if Phases 1-5 are complete with time remaining.

## Validation gates

For every kernel change:
1. `test_gemv_paro4g128.rs` (or G256 sibling) byte-exact on at least 4 K/M shapes
2. `paroquant_oracle.py` bit-exact source-vs-HFQ
3. Coherence smoke: `coherence_probe --model <hfq> --prompt-file recursion-defn.txt --max-tokens 120`
4. KLD bench: `eval_hipfire --model <hfq> --ref <bf16-kldref> --kv-mode q8 --max-chunks 32` (matched to Björn's 0.0933 number for A3B)

For every perf claim:
1. **Read `docs/methodology/perf-benchmarking.md`** first
2. Use `scripts/probe_commits.sh` for fresh-process cross-commit verification
3. Median of 3-5 measurements (within-session A/B is noisy on gfx1100/gfx1151 ±10-15%)
4. Coherence gate is mandatory before any default-flip — `./scripts/coherence-gate.sh`

## Hardware allocation (autonomous mode)

- **mi300 droplet (gfx942)** — primary compute for A3B (massive VRAM helps MoE per-expert work) and for the largest dense (27B). Billed hourly; keep it warm.
- **hiptrx (gfx1201, R9700 × 4)** — the original baseline reference; verify PARO4G128T engine layout perf matches the May 14 baseline (186.6 tok/s) on every commit that touches kernels.
- **hipx (gfx1100, 7900 XTX)** — secondary verification host; Björn's #316 numbers are on this arch indirectly via codex tuning.
- **Strix Halo gfx1151** — Björn's reference host; matches PR #316's 31.4/30.8 tok/s baseline.

If the perf gate fails on gfx1151 (Björn's reference) you've broken his work — investigate before committing.

## Reporting cadence

- Investigation doc per phase: `docs/investigations/paro-g256-perfmax/phase-{N}-result.md`
- KLD/perf delta numbers in commit messages (no claims without numbers)
- On Goal exit: write a final summary table in `docs/investigations/paro-g256-perfmax/SUMMARY.md` covering all 4 permutations × all 4 model sizes (or document which were skipped per Phase 1 gate)

## What NOT to do

- Don't merge to master.
- Don't push to fivetide's branches (work on `feat/paro-g256-perfmax` only).
- Don't break Björn's PRs #316/#317/#318 — verify byte-equivalence on his reference smoke before committing kernel changes.
- Don't chase grad-scale-learning — that's a separate experiment on a separate branch.
- Don't add F2 scope or autoawq formula — both falsified on the parent investigation.

## Operating discipline

- Δ ≥ 5% investigation rule: any kernel-level perf delta crossing ±5% needs the fresh-probe 3-5x median verify before claim
- Coherence gate `./scripts/coherence-gate.sh` blocks default-flips
- `--no-verify` authorized on commits if hooks complain
- Git author for all commits: `noreply@anthropic.com` (per `feedback_git_identity_noreply`)
- Push to origin OK, push to hiptrx OK, NOT to master
