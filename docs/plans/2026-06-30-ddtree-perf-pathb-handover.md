# Handover — qwen35 ddtree/DFlash perf: GPU-residency results, paper diff, Path B + Qwen3-8B threads

**Branch:** `feature/speculator-ddtree` · **HEAD:** `c2b95f56` (clean tree) · **Box:** gfx1151 (Strix Halo, 128 GB UMA)
**Worktree:** `/home/bjoern/hipfire/.worktrees/speculator-ddtree`
**Predecessor handover:** `docs/plans/2026-06-29-qwen35-spec-perf-handover.md` (the perf chase this continues)

> GPU is shared with a long-running dspark bench. ALWAYS go through `scripts/gpu-lock.sh`
> (`gpu_acquire`/`gpu_release`); never `rm` the lockfile. Currently free.

---

## 0. TL;DR

This session chased ddtree/DFlash perf, did a GPU-residency refactor, then **measured the impact**
— which caught a catastrophe and produced the real verdict:

- **On this UMA box the whole effort is throughput-NEUTRAL.** Every per-cycle copy I eliminated is
  cheap on unified memory; the wins are dGPU-PCIe-only and remain **unmeasured** (need k9lin
  gfx1100 / hiptrx gfx1201).
- **chain (DFlash-linear) beats ddtree on every genre at temp 0 / 0.7 / 1.0.** ddtree never wins at
  the shipped b8/k2. Both beat AR by 1.3–3.6×.
- A committed **Stage 3b/3c catastrophe** (ddtree −87%) was found by the tok/s A/B and **reverted**.
- Two open threads remain (both about *making ddtree actually pay off* / understanding the gap):
  **Path B** (kill the ~50% slow-path 2nd-forward) and the **Qwen3-8B pure-attention** isolation.

---

## 1. Branch state — what landed, what reverted

```
c2b95f56 docs: Stage 3b/3c perf catastrophe (why reverted)        <- HEAD
93c81e2e Revert Stage 3b (tree build on-GPU)                       REVERT
b8ae0900 Revert Stage 3c (greedy follow on-GPU)                    REVERT
0ab24be7 Stage 3c greedy follow on-GPU         [REVERTED — catastrophe]
77d4cbaa Stage 3b tree build on-GPU            [REVERTED — catastrophe]
201315b8 Stage 3a attn-mask on-GPU (parallel kernel)               KEPT (neutral)
1317d16f C8 chain temp>0 on-GPU accept (kill 2×9MB probs D2H)       KEPT
3f056f1c C8 samplers (validated in isolation)                      KEPT
137e7600 ddtree Stage 2: del 37MB naive D2H, drain→stream-sync,…   KEPT
670eb12f ddtree Stage 1 hidden D2D scatter                         KEPT
ca6691b0 gfx1151 (32,8) launch_bounds on HFQ4G256 verify GEMMs     KEPT (noise-level)
3df5da3d (branch base — pre-work)
```

**KEPT = correctness-preserving + coherence-gate-dflash + serve-multiturn PASS, byte-identical at
temp0 where applicable.** All 6 kept commits are throughput-neutral on UMA; their value is PCIe on
dedicated-VRAM GPUs (unmeasured here).

**Stage 3b/3c were a −87% regression** — moving the ddtree tree BUILD + greedy FOLLOW to
*single-thread* GPU kernels (sequential heap/walk on one GPU lane = ms-scale to save sub-µs copies).
Byte-identity + gates + copy-counts were all green; only a tok/s A/B caught it. See
`docs/plans/ddtree-stage3bc-perf-catastrophe.md`. **Lesson: a copy-elimination that adds a
SEQUENTIAL GPU kernel is a loss; "perf neutral" must be MEASURED, not deduced.**

---

## 2. The measured perf (the deliverable)

Daemon, `qwen3.6-27b.mq4` + `qwen36-27b-dflash-mq4.hfq`, q8 KV, warm + median-of-3, byte-identical
prompts. ddtree at **shipped b8/k2** (`DEFAULT_TREE_BUDGET=8`/`TOPK=2`). temp>0 = **bare temperature
(no top_p)** — required or ddtree SWOR disables → AR fallback (see §6).

**temp 0 — before(3df5da3d) → after(reverted HEAD), same-session A/B:** essentially FLAT.
AR ~14.4 (flat). chain code 34.3→34.2 / reason 50.6→51.3 / prose 30.4→30.7 / factual 49.0→49.8.
ddtree code 24.2→24.6 / reason 32.2→32.8 / prose 23.0→23.2 / factual 32.0→32.4 (τ unchanged).

**temp 0.7 (shipped b8/k2):**
| genre | AR | chain tok/s (τ) | ddtree tok/s (τ) |
|---|---|---|---|
| code | 14.4 | **29.5** (2.85) | 25.8 (3.63) |
| reason | 14.5 | **51.7** (5.73) | 34.0 (4.61) |
| prose | 14.5 | **24.4** (2.21) | 19.1 (2.21) |
| factual | 14.5 | **48.7** (5.42) | 29.0 (3.90) |

**temp 1.0 (shipped b8/k2 — recommended qwen3.6 temp):**
| genre | AR | chain tok/s (τ) | ddtree tok/s (τ) |
|---|---|---|---|
| code | 14.4 | **30.8** (3.06) | 23.9 (3.32) |
| reason | 14.5 | **41.6** (4.73) | 32.4 (4.24) |
| prose | 14.4 | **24.7** (2.23) | 20.4 (2.32) |
| factual | 14.5 | **38.7** (4.10) | 29.5 (3.85) |

Verdict: **chain wins everywhere.** At b8/k2 ddtree often lacks even a τ edge (chain's strong linear
drafter out-accepts it on reason/factual). The shipped default (chain; ddtree is opt-in via
`HIPFIRE_DDTREE_BUDGET>0`) is correct. Raw data: `/home/bjoern/{perf_ab_results,temp_shipped}.jsonl`.

---

## 3. DDTree paper diff (arxiv 2604.12989 / liranringel.github.io/ddtree)

We **faithfully port** the paper: DFlash block-diffusion drafter (`dflash.rs` is the z-lab reference —
non-causal 5-layer Qwen3 decoder cross-attending to target hidden layers), best-first heap tree,
fixed budget, single-pass **ancestor-mask** verify, SWOR sampling. The divergences:

1. **Target is a DeltaNet/linear-attention HYBRID** (Qwen3.6-27B: 48 `linear_attn` + 16 full-attn),
   not pure softmax-attention Qwen3. The ancestor mask is exact only for softmax attention; the 48
   recurrent layers can't be masked → the tree forward's GDN state-advance through the linearized
   order **cross-contaminates siblings** ("linear-replay approximation", `speculative.rs` ~5110
   doc). So our tree-verify is **exact on 16/64 layers, approximate on 48**, and needs a **2nd
   forward** (GdnTape re-verify) on non-spine commits — the paper needs neither.
2. **Outcome inverted:** paper DDTree beats linear DFlash **+2.13×** (6–8× total); ours **loses** to
   chain.
3. **AMD RDNA + 4-bit MQ4 + q8 KV**, vs the paper's NVIDIA at higher precision.
4. **Tiny budget (b8/k2)** vs the paper's larger fixed budget — we measured a **~30% tok/s cliff at
   b16** (the 27B MQ4 verify forward grows faster than the acceptance gain).
5. **temp>0 SWOR honors temperature only** (top_p/top_k/penalty → AR fallback); paper applies the
   target's full decoding rule.

Core insight: for a DeltaNet target the tree's value proposition is structurally weakened — you can't
get exact tree posteriors in one forward, so the acceptance gain the paper relies on is partly lost.

---

## 4. OPEN THREAD A — Path B (kill the slow-path 2nd verify). **Highest-upside ddtree lever.**

**Why it matters:** the 2nd verify forward fires on **~50% of ddtree cycles** at b8/k2 (measured
`HIPFIRE_DDTREE_TAPE_DUMP=1`: fast=11 / slow=11 over 22 cycles). It's the **DeltaNet tax** — every
accepted rank>0 (non-spine) sibling needs a re-verify to fix the recurrent state + the racing-sibling
KV slots. Killing it ≈ −33% verify-forwards → could close the chain-vs-ddtree gap on code/prose.

**The cycle structure** (`spec_step_ddtree_batched`, all line #s at HEAD c2b95f56):
- `verify_dflash_block_tree` (~5434): the single tree forward, captures GdnTape inline.
- `spine_accept` (5523) = greedy walk took rank-0 child every step.
- `fast_tape_ok` (5527) → fast path (5536/5550): restore + `replay_gdn`, **no 2nd forward**.
- `else if HIPFIRE_DDTREE_PATH_B_CAPTURE=1 && !pre_rope_k.is_empty()` (5561): **Path B** — gather
  committed innovations from the tree tape (`gather_accepted`) + re-RoPE committed-slot K + gather V
  + quant-write to KV, then `replay_gdn`. **No 2nd forward.**
- `else` (5788): default slow path — `verify_dflash_block` 2nd forward (~40–50 ms) + replay.
- `HIPFIRE_DDTREE_FORCE_SLOW=1` (5522) forces slow on every cycle (diagnostic reference).
- `pre_rope_k` capture is gated on the same env (5408) → the env flag activates both capture + Path B.

**STATUS: Path B PANICS on q8 KV.** `speculative.rs:5772`: *"Path B Phase 2 only supports asym3 KV
today (got Q8)"*. The `match kv.k_tier()` (5720) implements only the **asym3** arm (5721–5762);
`other =>` panics. Q8 is the canonical DFlash KV. So Path B is genuinely **unfinished**, not just
opt-in.

**THE CONCRETE NEXT TASK — add the Q8 arm to Path B Phase 2 (`speculative.rs` ~5762):**
- The Q8 batched KV writer EXISTS: `Gpu::kv_cache_write_q8_0_batched(dst, src, positions, n_kv_heads,
  head_dim, batch_size)` — `crates/rdna-compute/src/attention.rs:1350`.
- **CAUTION: its signature ≠ asym3.** asym3 (`kv_cache_write_asym3_batched`) writes K+V together with
  Givens cos/sin rotation; the q8 writer is **single-tensor, no Givens** → write K and V separately.
  For the Q8 arm: write the re-RoPE'd K (`pbs.fa_k_batch`) to `kv.k_gpu[layer_idx]` at committed
  positions (`scratch.kv_gather_indices`) via `kv_cache_write_q8_0_batched`, then do the V byte-gather
  exactly like the asym3 arm (5752–5762: `kv_compact_gather` from raced V slots → `memcpy_dtod_at` to
  committed slots; V has no rotation). Confirm the q8 V slot byte-stride (`v_bpp`, set at 5661 as
  `n_kv_heads*(head_dim/32)*34` — that's the Q8 V layout, so V handling already assumes q8).

**VALIDATION PROTOCOL (do NOT skip — Path B is the repo's most-falsified zone; Path A shipped
attractors):**
1. **Dual-path assert, byte-identical at temp0:** capture committed token-ids (HIPFIRE_EMIT_TOKEN_IDS=1)
   for `HIPFIRE_DDTREE_PATH_B_CAPTURE=1` vs the correct reference, b8/k2, 4 prompts, max 200. MUST be
   byte-identical to the reference. **BUT FIRST resolve the reference ambiguity below.**
2. If byte-identical → perf (Path B vs default tok/s, median-3 — should be faster, killing ~50%
   2nd-forwards). 3. → `scripts/coherence-gate-dflash.sh` + temp0.7 attractor eyeball.
- Reuse `/home/bjoern/pathb_validate.sh` (already written — 3 configs slow/pathb/default, md5 assert +
  perf). After the q8 arm lands, `pathb` should stop producing empty output.

**⚠ REFERENCE AMBIGUITY (resolve before trusting the assert):** this session found **`FORCE_SLOW`
output ≠ default output** at temp0 (different token md5s). I.e. the ddtree *fast* path (used on spine
cycles, ~50%) is **NOT byte-exact with a full re-verify** — it leaves the racing sibling's KV at
committed slots (last-write-wins). It's coherent (passes gates) but not bit-exact. So "which output is
correct?" is genuinely open: the re-verify (slow) is arguably *more* correct; the fast path is what
ships. Decide the correctness target (likely: Path B must match `FORCE_SLOW`, and separately
investigate whether the fast path's KV-race staleness matters) before declaring Path B "correct".

---

## 5. OPEN THREAD B — Qwen3-8B pure-attention (isolate the DeltaNet penalty)

Goal: run our ddtree on a **pure-attention** target (the paper's regime) to see if the DeltaNet
hybrid is *the* reason ddtree underperforms.

**State:** `qwen3-8b.mq4` (target, Qwen3ForCausalLM 36-layer) + `qwen3-8b-dflash.hfq` (drafter,
config-identical to z-lab/Qwen3-8B-DFlash-b16: `target_layer_ids:[1,9,17,25,33]`, block 16, bf16) both
on disk. It **loads + runs** via a separate **"DFlash generic speculator … arch 1"** path
(`dflash_generic.rs` + `crates/hipfire-arch-llama/src/spec_impl.rs`) — no DeltaNet, so the tree-verify
there *is* exact. **BUT τ=0.27** (drafter rejected ~always; a working drafter is τ>2).

Ruled out: ingest layers (correct, read from drafter metadata via `set_dflash_extract_layers`); missing
capture (llama `spec_impl.rs` passes `extract_layers` into the forward at 103/147/181/217 — real impl).
Remaining suspects: **(a)** a bad weight-conversion in the on-disk `.hfq`; **(b)** bf16-trained drafter
vs **mq4** target hidden-state mismatch (but the 27B works at mq4, weakening this); **(c)** a bug in the
**generic/llama DFlash port** ("real port, not a load" per memory `dflash-drafter-assets-llama-qwen3`).
Leading hypothesis: **(c)** the generic path is an unvalidated port.

**Done:** downloaded `z-lab/Qwen3-8B-DFlash-b16` → `/home/bjoern/dflash-z-lab-8b/` (2.1 GB, 9 files,
complete). **Next:** re-convert it (`cargo run --release --bin dflash_convert -p hipfire-quantize --
--input /home/bjoern/dflash-z-lab-8b --output ~/.hipfire/models/qwen3-8b-dflash-v2.hfq`; keep f16, do
NOT quantize the drafter — quality drives τ), re-smoke (recipe below), and if τ stays <1, debug the
generic/llama DFlash path (capture point / hidden projection / RoPE / mask-token 151669 handling) —
likely diff it against the working qwen35 capture path. `hf` CLI is the tool (`huggingface-cli` is dead).

Smoke recipe: daemon, load `qwen3-8b.mq4` + draft `qwen3-8b-dflash.hfq` (q8 KV), env
`HIPFIRE_DDTREE_BUDGET=8 HIPFIRE_DDTREE_TOPK=2`, temp 0, a committed prompt; watch the `done` event τ.

---

## 6. Reproduction, env knobs, artifacts

**Canonical bench:** daemon (`./target/release/examples/daemon`, build `cargo build --release --example
daemon --features deltanet -p hipfire-runtime`), `qwen3.6-27b.mq4` + `qwen36-27b-dflash-mq4.hfq`, q8 KV,
warm + median-of-3, byte-identical prompts (`benchmarks/prompts/{lru_cache_pep8_strict,trains-meet,
prose_river_short,bare_factual}.txt`, md5 df5dedc8/db92b572/07a78809/1d32df5f).

**Spec-mode env** (read at daemon startup → fresh daemon per config): AR = no `params.draft`; chain =
draft + `HIPFIRE_DDTREE_BUDGET=0`; ddtree = draft + `HIPFIRE_DDTREE_BUDGET=8 HIPFIRE_DDTREE_TOPK=2`.
**temp>0 ddtree REQUIRES bare temperature** (omit top_p/top_k) or SWOR disables → AR fallback (log:
"temp>0 DFlash spec disabled -> AR … ddtree SWOR verify honors temperature only"). chain temp>0 honors
top_p (C8 path, `HIPFIRE_DFLASH_FAST_SAMPLE=1`).

**Diagnostic knobs:** `HIPFIRE_DDTREE_TAPE_DUMP=1` (fast/slow 2nd-verify counts), `HIPFIRE_DDTREE_
FORCE_SLOW=1` (always re-verify), `HIPFIRE_DDTREE_PATH_B_CAPTURE=1` (Path B), `HIPFIRE_EMIT_TOKEN_IDS=1`
(committed token-id events for byte-identical asserts).

**Scripts I wrote** (`/home/bjoern/`, throwaway): `perf_ab.sh` (before/after 2-commit A/B — handles the
`git checkout`/build/restore dance + trap), `confirm_revert.sh`, `temp_run.sh <temp>` /
`temp_shipped.sh` (bare-temp AR/chain/ddtree sweep), `pathb_validate.sh` (Path B dual-path assert +
perf). All gpu-locked, median-of-3, fixed parse. **Watch out:** `pkill -f` self-matches the agent's own
shell (cost an exit-144 + an orphaned daemon this session) — kill daemons by PID.

**Committed docs this session** (`docs/plans/`): `ddtree-cycle-static-map`, `ddtree-verify-kernel-
occupancy`, `ddtree-budget-sweep-results`, `mq4lloyd-occupancy-patch-plan`, `hfq4g256-occupancy-ab-
results`, `ddtree-overhead-fix-scope`, `gpu-resident-spec-roadmap`, `ddtree-d2d-stage1-results`,
`ddtree-gpu-resident-stage2-results`, `c8-gpu-sampler-scope`, `c8-sampler-kernel-validation`,
`c8-integration-results`, `c8-tau-diagnosis`, `ddtree-stage3{a,b,c}-*`, `ddtree-stage3bc-perf-
catastrophe`. The sweep script `scripts/ddtree_budget_sweep_daemon.sh` is committed.

**Guardrails (mandatory):** `scripts/coherence-gate-dflash.sh` + `scripts/serve-multiturn-gate.sh`;
byte-identical temp0 token-ids for transport/state-only changes; for any spec-decode tok/s claim use the
daemon (NOT `dflash_spec_demo` — it under-reports + gives a different τ regime: demo τ8.0 vs daemon τ3.16
on the same code prompt). Don't `cargo fmt`; don't rustfmt `speculative.rs`/`qwen35.rs` (fmt debt).

**Session traps now in memory** (`qwen35-spec-gpu-residency`): single-thread GPU kernel for sequential
work = catastrophe; "perf neutral" must be measured; demo-vs-daemon τ; "stochastic τ" → 20-run A/B vs
host; dual-path shadow-assert for byte-exact ports; 27B `.mq4` uses HFQ4G256 not mq4g256_lloyd.

---

## 7. Recommended next moves (ranked)

1. **Finish Path B q8 arm** (§4) — concrete, scoped (one match arm + the q8 writer), highest ddtree
   upside; gate it hard (dual-path assert + coherence). Resolve the fast-vs-slow reference ambiguity.
2. **Measure the kept work on a real dGPU** (k9lin gfx1100 / hiptrx gfx1201) — the copy-elimination's
   only payoff is PCIe; it's currently unmeasured. This is the one number that justifies the kept commits.
3. **Fix the Qwen3-8B generic DFlash path** (§5) to enable the pure-attention DeltaNet-isolation experiment.
4. Branch hygiene: the 8→6-effective commits + 2 reverts are a slightly messy history for PR #483 (draft);
   consider whether to keep the revert trail (documents the failure) or squash before review.
