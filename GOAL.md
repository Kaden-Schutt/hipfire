# Goal — feat/paro-g256-perfmax

> Read `AGENT-BRIEF.md` first for branch context + inventory.

## Mission

Decide the **G256 gate** (per `docs/plans/paroquant-g256-milestone.md`) and **perfmax the ParoQuant runtime to the gfx12 asymptote first**, then port the proven kernel shapes to other RDNA archs.

You are autonomous. Use best-judgement calls. Don't suspend for input until either the goal is satisfied or the work is structurally blocked. Use branches, commit often, push when stable, don't push to master.

## Hardware allocation — gfx12 (hiptrx) only for now

**Primary (only for now):** `hiptrx` host — R9700 / **gfx1201 / RDNA4 / "gfx12"**. This is the dedicated GPU for this mission. ssh alias `hiptrx`. The May 14 baseline (`docs/investigations/2026-05-14-paroquant-hiptrx-baseline/README.md`) was measured here at PARO4G128T engine layout = 186.6 tok/s decode; that's your reference point.

**NOT AVAILABLE:** `mi300` droplet (gfx942 / CDNA3) — the human is using this for a separate mission. Don't ssh to mi300. Don't run on mi300.

**Conditional (post-asymptote):** `hipx` (gfx1100 / RDNA3 / 7900 XTX) and `k9lin` (gfx1100 also) — usable for porting kernel shapes from gfx12, BUT ONLY after:
1. gfx12 asymptote is reached (you've explicitly documented no remaining +5% perf delta from any tested lever)
2. You SSH to the target host and run `ps -ef | grep -E "hipfire|engine|bench"` to verify no active hipfire GPU tasks are running. If anything is busy, defer that arch and continue on what's idle.
3. If both hipx and k9lin are busy when you check, just stay on gfx12 and write down what you'd test on the others.

`Strix Halo gfx1151` (RDNA3.5) is Björn's PR #316 reference host. Use it for regression smoke verification of his PRs, not for primary perfmax.

## Sequencing

**You always start with gfx12 (hiptrx). Don't begin any gfx11/gfx1151 port work until gfx12 is at the asymptote.** "Asymptote" = you've shipped both named perf levers (rotate-fusion + batched-QKV) AND tried at least 3 additional fusion/tile-shape variants AND the last 3 attempts produced <5% perf delta each. Document the asymptote in `docs/investigations/paro-g256-perfmax/gfx12-asymptote.md` before considering ports.

## Exit conditions

Either of:

A. **All four permutations evaluated** (`PARO4G128`, `PARO4G128T`, `PARO4G256`, `PARO4G256T`) on gfx12 AND **at least one** perf lever shipped (rotate-fusion OR batched-QKV) so that A3B-PARO decode reaches ≥ 60% of A3B-MQ4 decode on gfx1201, OR

B. **G256 gate decided** (with empirical data, not opinion) — quality probe says G256 is unworkable (KLD blow-up > 3× G128 baseline) and you ship the G128-only perfmax stack with rotate-fusion + batched-QKV instead.

Either exit is acceptable. The bad outcome is "we still don't know."

The optional Phase 6/7 port work (gfx11 / gfx1151) only fires if gfx12 asymptote is documented AND the target host is verified idle.

## Phases

### Phase 1 — G256 quality probe (CPU only, no GPU)

```bash
python3 scripts/paroquant_g256_probe.py --help
python3 scripts/paroquant_g256_probe.py \
    --paro-src <0.8B-PARO-safetensors-dir> \
    --report-out docs/investigations/paro-g256-perfmax/g256-probe-0.8b.md
```

Output compares PARO4G128 oracle vs PARO4G256 AWQ regrouping vs PARO4G256_MQ row-major HFQ4-G256.

**Gate criterion:**
- G256 NRMSE / KLD-proxy ≤ 1.2× G128 baseline → invest in G256 runtime (Phase 2 fires)
- 1.2-1.5× → marginal; ship G256T (engine layout) only — its BW headroom may compensate
- > 1.5× → kill G256, skip Phase 2, go straight to Phase 3 with G128 only

Run on at least 0.8B and 9B PARO sources. A3B optional. Commit probe outputs in `docs/investigations/paro-g256-perfmax/`.

### Phase 2 — Implement PARO4G256 / PARO4G256T on gfx12 (only if Phase 1 says go)

Mirror the existing G128 stack:
- `kernels/src/gemv_paro4g128.hip` (1081 LOC) → `gemv_paro4g256.hip`
- Add `DType::PARO4G256` + `DType::PARO4G256T` to `crates/rdna-compute/src/dispatch.rs` (G128 variants live around lines 2302-2772 — read + pattern-match)
- Extend `scripts/paroquant_import.py` to emit new qtype on `--group-size 256`
- Add `crates/rdna-compute/examples/test_gemv_paro4g256.rs` (clone the G128 test)
- Wire `qwen35.rs` load + dispatch for the new dtypes

Engine layout (G256T): qweight transposed `[M/8, K]` for coalesced GEMV + theta precomputed as `sincos_f32` (mirror G128T's tricks).

Verify on gfx1201 against the G128 baseline. Goal: kernel correctness (byte-exact on test) + initial perf number.

### Phase 3 — Perf levers on gfx12

Two named fusions from `docs/investigations/2026-05-14-paroquant-hiptrx-baseline/README.md`:

**Lever 1 — Fuse `paro4g{128,256}t_rotate` into subsequent GEMV.**
Currently 24.3% of decode at 6.7 GiB/s standalone. Pattern: MQ4's `fused_rmsnorm_mq_rotate`. Create `gemv_paro4g{128,256}t_prerotated_fused_rmsnorm.hip`. Estimated **+30% decode tok/s**.

**Lever 2 — Batched QKV GEMV.**
Currently 3 separate `gemv_paro4g{128,256}t_prerotated` calls per layer. MQ4 collapses to one `fused_qkvza_hfq4g256` at 265.5 GiB/s. Create `fused_qkv_paro4g{128,256}t.hip`. Estimated **+15-25% decode tok/s** stacked.

Both levers ship default-on with opt-out env var (e.g. `HIPFIRE_PARO_FUSE_ROTATE=0`).

### Phase 4 — A3B-specific perf work on gfx12

Björn shipped correctness in PR #316; you ship perf parity. Required: per-expert `gemv_paro4g{128,256}t_moe_indexed_*.hip` kernels mirroring MQ4 / HFQ4 MoE-indexed kernels (search `kernels/src/*moe_*_indexed*`). For A3B (256 experts, k=8 active per token), the MoE down kernel is the prefill bottleneck.

**Goal:** A3B-PARO decode ≥ 60% of A3B-MQ4 decode on gfx1201 (= ≥34 tok/s if MQ4 hits 57). Document the gap if you can't close it; identify the next bottleneck via rocprof.

### Phase 5 — Dense perf parity on gfx12 (0.8B / 9B / 27B / 27B-3.6)

PARO4G128T already at engine layout (+84%). Layered with Phase 3 fusions:
- 0.8B: target ≥ 90% of MQ4 decode on gfx1201
- 9B: target ≥ 80% of MQ4 decode on gfx1201
- 27B + 27B-3.6: target ≥ 75% of MQ4 decode on gfx1201

### Phase 6 — Document gfx12 asymptote

Before ANY port work to other archs, write `docs/investigations/paro-g256-perfmax/gfx12-asymptote.md`:
- Final tok/s per format/model
- The 3+ additional fusion/tile experiments you ran post-Lever-1+2 (rocprof-named, not blind tries)
- The deltas per experiment
- Why the last 3 attempts each produced <5% perf delta (the asymptote signature)

Only after this doc is committed: proceed to Phase 7.

### Phase 7 — Port to gfx11 / gfx1151 (conditional)

**Gate check before each host:**
```bash
ssh <host> 'ps -ef | grep -E "hipfire|engine|bench" | grep -v grep'
```
If output non-empty → target host busy, skip to next host or write down what would be tested and stop.

Port priority order:
1. `hipx` (gfx1100 / RDNA3) — most production-impact arch
2. `Strix Halo gfx1151` (RDNA3.5) — Björn's reference; verify his PR #316 numbers still hold
3. `k9lin` (gfx1100, second instance) — sanity-check via re-running #1's results

Port = take the proven gfx12 kernel shape and adapt for the target arch's wave size, LDS budget, MFMA/WMMA availability. Document deltas in `docs/investigations/paro-g256-perfmax/port-{arch}.md`.

### Phase 8 — DFlash compatibility (stretch)

PARO-quantized drafters + DFlash spec-decode. Optional, only if Phases 1-7 done with time remaining.

## Validation gates

Every kernel change:
1. `test_gemv_paro4g{128,256}.rs` byte-exact on at least 4 K/M shapes
2. `paroquant_oracle.py` bit-exact source-vs-HFQ
3. Coherence smoke: `coherence_probe --model <hfq> --prompt-file recursion-defn.txt --max-tokens 120` — verdict OK
4. KLD bench: `eval_hipfire --model <hfq> --ref <bf16-kldref> --kv-mode q8 --max-chunks 32` — Björn's A3B baseline 0.0933 must hold

Every perf claim:
1. Read `docs/methodology/perf-benchmarking.md` first
2. Use `scripts/probe_commits.sh` for fresh-process cross-commit verification
3. Median of 3-5 measurements (within-session A/B is noisy on gfx1100/gfx1151 ±10-15%)
4. Coherence gate `./scripts/coherence-gate.sh` mandatory before any default-flip

## Reporting cadence

- Investigation doc per phase: `docs/investigations/paro-g256-perfmax/phase-{N}-{result|asymptote|port-arch}.md`
- KLD/perf delta numbers in commit messages (no claims without numbers)
- On Goal exit: final summary table in `docs/investigations/paro-g256-perfmax/SUMMARY.md` covering tested permutations × tested model sizes × tested arches (or document skips per Phase 1 / Phase 7 gates)

## Input sourcing (since mi300 is unavailable)

The mi300 inputs in AGENT-BRIEF.md are NOT directly accessible. For gfx12 work on hiptrx, fetch/build inputs there:

- BF16 GGUFs / HF safetensors: `hf download Qwen/<model>` directly on hiptrx (the droplet had fast HF bandwidth; hiptrx may be different — verify and adapt)
- KLDrefs: rebuild via `scripts/build_kldref.py` (or local hipfire example) if not already present on hiptrx
- Mix-v1 corpus: copy from the user's local `.claude/worktrees/agent-a16d8b7781c781e4c/benchmarks/quality-baselines/slice/calibration-mix-v1.txt` (md5 `68a1d2e62117e692e0e04c2811349aaf`)
- Mix-v1 imatrix + Hessian for 0.8B can be rebuilt on hiptrx via `collect_imatrix` / `collect_hessian` binaries (10-20 min each on gfx1201)

Document any missing-input blockers; don't suspend — work around them by rebuilding on hiptrx.

## What NOT to do

- Don't ssh to `mi300` (the droplet) for ANY work — human is using it.
- Don't push to master.
- Don't push to fivetide's branches.
- Don't break Björn's PRs #316/#317/#318 — verify byte-equivalence of his reference smoke before committing kernel changes.
- Don't chase grad-scale-learning — that's a separate experiment on a separate branch.
- Don't add F2 scope or autoawq formula — both falsified on the parent investigation.
- Don't begin porting to gfx11 / gfx1151 until gfx12 asymptote is documented.

## Operating discipline

- Δ ≥ 5% investigation rule: any kernel-level perf delta crossing ±5% needs the fresh-probe 3-5x median verify before claim
- Coherence gate `./scripts/coherence-gate.sh` blocks default-flips
- `--no-verify` authorized on commits if hooks complain
- Git author for all commits: `Kaden Schutt <151092359+Kaden-Schutt@users.noreply.github.com>` (GitHub no-reply; per `feedback_git_identity_noreply` memory). NOT `noreply@anthropic.com` (Anthropic default) and NOT `kaden@schutt.dev` (personal).
- Push to origin OK, push to hiptrx OK, NOT to master
- Hardware gate check before any non-hiptrx host: `ps -ef | grep -E "hipfire|engine|bench"`
