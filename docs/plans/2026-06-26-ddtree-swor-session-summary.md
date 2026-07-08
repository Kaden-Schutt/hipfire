# Session summary — temp>0 distribution-correct DFlash spec decode (qwen35 ddtree SWOR)

**Branch:** `feature/speculator-ddtree` (off `fivetide/feature/speculator-abstraction` @ `4298c16b`)
**Session commit range:** `4298c16b..b19940ea` (HEAD)
**Date:** 2026-06-26

## One-paragraph state

The qwen35 DFlash *ddtree* speculative-verify now does **distribution-preserving
SWOR sampling at temp>0** (Sequoia/SpecTr sampling-without-replacement), fully on
GPU, and is wired into serve so temp>0 requests get the spec speedup instead of
falling back to plain AR. Naive sampling is the fallback; greedy/argmax is a
diagnostic-only mode (it ignores temperature). It is **not a correctness fix** —
serve temp>0 was already correct via AR — it is a **perf feature** that SWOR makes
safe (the spec speedup without distorting the sampled distribution).

## The arc (each step taught something — keep all of it)

1. **banding (`e0767840`)** — `build_ddtree_tree_bounded` node-count floor. A/B no-op
   under greedy verify; parked. (lesson: tree breadth is wasted under argmax accept.)
2. **naive sampling (`e7ec4cda`, `681c7c48`)** — SpecInfer "naive sampling": draw
   `x~target`, accept the drafted child it lands on. Distribution-EXACT by
   construction (every emit is a target draw), reduces to greedy at temp→0. The real
   first fix. Proven: temp→0 byte-equiv + multi-temp + conditional MC tests (TV<0.01).
3. **q-exploiting plan + Phase-0 gate (`6bb181b6`, `f57fa17a`)** — cheap CPU sim on
   dumped `(p,q)` to bound the upside before GPU work. Said STOP (+6% upper bound).
4. **approximate SWOR (`f51848bb`)** — fake +20% τ "win" → it was over-acceptance
   bias from a reduced `{children,TAIL}` residual + q-desc order (the classic
   synth-win). Coherence passed but the number was a shortcut artifact.
5. **verbatim SWOR (`11d072e5`)** — exact full-vocab residual `norm(relu(p−q_j))`,
   true draw order, exact bonus, full per-position `q` from the draft. **Proven
   distribution-exact** (`swor_preserves_target_distribution`, 600k MC). Reversed the
   win: exact τ only +3% vs greedy, and −31% tok/s **on CPU**.
6. **fused GPU kernel (`206941bb`)** — `ddtree_swor_walk_f32`: the whole walk
   (per-slot softmax + recursive residual + categorical draw + descent) in one
   single-workgroup kernel. Recovered tok/s (16→23). The CPU loss was an artifact.
7. **broad sweep (`c82efbed`)** — 8 workloads: SWOR vs naive **+48% τ / +26% tok/s**;
   the real story is **naive collapses on high-entropy prompts** (math/creative/list
   τ≈1.2–1.7) while SWOR holds near greedy.
8. **SWOR default + device Gumbel sampler (`46dbab41`)** — made SWOR the temp>0
   verify default; `ddtree_gumbel_topk_batched_f32` does Gumbel-top-k SWOR draft
   sampling on-device (murmur3 RNG) so draft logits stay resident — killed the
   12 MB/cycle D2H. (this commit was amended to drop an earlier correctness overclaim.)
9. **serve wiring (`5884a894`, `b19940ea`)** — `Speculator::step` gains `temp`,
   threaded daemon-wide; `Speculator::supports_temp_verify()` gates routing so temp>0
   reaches spec ONLY where SWOR is active. Fixed a pre-existing
   `verify_scratch max_n < b` panic for ddtree-under-serve. Validated by a temp>0
   serve smoke (A spec / B AR-control / C greedy — all coherent).

## Key artifacts

| What | Where |
|------|-------|
| Naive + SWOR verify, `swor_step`, banded build, tests | `crates/hipfire-runtime/src/ddtree.rs` |
| Fused SWOR walk kernel | `kernels/src/ddtree_swor_walk.hip` |
| Device Gumbel-top-k SWOR sampler | `kernels/src/ddtree_gumbel_topk_batched.hip` |
| Kernel launchers | `crates/rdna-compute/src/sampling.rs` |
| `spec_step_ddtree_batched`, draft sampler, `swor_walk_gpu` | `crates/hipfire-arch-qwen35/src/speculative.rs` |
| `DflashSpeculator`, `supports_temp_verify`, scratch sizing | `crates/hipfire-arch-qwen35/src/dflash_spec.rs` |
| `Speculator` trait (`step` temp, `supports_temp_verify`) | `crates/hipfire-runtime/src/spec.rs` |
| Daemon routing gate + temp threading | `crates/hipfire-runtime/examples/daemon.rs` (gate ~6270) |
| GPU validation examples | `crates/rdna-compute/examples/{ddtree_swor_kernel_check,ddtree_gumbel_check}.rs` |
| Full history + A/B tables | `docs/plans/2026-06-26-ddtree-q-exploiting-verify.md` |
| Bench harnesses + results (NOT in repo) | `/home/bjoern/hipfire-ab/` |

## Flags

- `HIPFIRE_DDTREE_VERIFY=naive` — opt out of SWOR back to naive (default swor at temp>0).
- `HIPFIRE_DDTREE_GREEDY_VERIFY=1` — force greedy argmax verify (diagnostic; ignores temp).
- `HIPFIRE_DFLASH_TEMP_SPEC=0` — force temp>0 back to AR at serve (disable the routing).
- `HIPFIRE_DDTREE_BUDGET` / `HIPFIRE_DDTREE_TOPK` — enable ddtree mode + tree shape.

## Validation status

- ddtree unit tests: 30/30 (distribution-exactness for naive + SWOR).
- GPU: `ddtree_swor_kernel_check` TV=0.0026; `ddtree_gumbel_check` TV=0.002, 0 dups.
- Broad 8-workload sweep: all 24 runs coherent.
- Serve smoke: temp0.7 SWOR coherent + uses spec; AR control; temp0 unregressed.
- coherence-gate-dflash (greedy) 4/4 earlier; serve-multiturn-gate PASS.

## Open items / known limitations

1. **Single-block SWOR walk kernel** — uses 1 CU; per-cycle ~200µs, small vs the
   verify forward, but leaves parallelism on the table. Multi-block needs a global
   barrier between sequential slots (hard).
2. **Per-prompt τ realization variance** — SWOR acceptance is noisy run-to-run
   (RNG realization, dominated by rank-0). Aggregate win is robust; single prompts vary.
3. **n-gram spec temp-ignoring (pre-existing, separate)** — arch 7/10/11/12 with
   opt-in n-gram spec route temp>0 to GREEDY n-gram verify (ignores temperature).
   Out of scope here; n-gram verify can't do SWOR. Worth a separate fix (route to AR).
4. **Per-prompt byte-identical prompts** for cross-session perf claims (CLAUDE.md).

## Reuse for other model families (the point of the session)

The verify-side machinery is **arch-generic and ready to lift**:
- `ddtree::{swor_step, sample_verified_tree_swor, sample_verified_tree}` are pure CPU,
  arch-free (operate on logits + a `DdTree`).
- `ddtree_swor_walk_f32` + `ddtree_gumbel_topk_batched_f32` are arch-free GPU kernels
  (consume target/draft logits + tree metadata).
- `Speculator::supports_temp_verify()` is the clean routing seam — any arch whose
  drafter builds a `DdTree` and verifies it can return true and get temp>0 serve
  routing for free.

**Candidate next ports:** deepseek4 MTP-tree (already has a ddtree builder), minimax
(MoE, no drafter today → would need a tree drafter). The blocker for non-qwen35 is
arch-specific: building the verify forward + the per-slot target logits on device in
the `verify_scratch.logits` shape the walk kernel expects. The kernels + accept
logic transfer unchanged.

## Review findings (2026-06-26, 3-agent review: correctness / missed-ops / boundaries)

**RESOLVED in `60c73395`:** #1 (explicit sampling controls → AR), #2 (GREEDY_VERIFY
disqualifies temp>0 spec), #3 (n-gram temp>0 → AR), #4 (Gumbel→swor_step composition
distribution gate test), AND the ddtree draft settings now ride the unified CLI
(`ddtree_budget`/`ddtree_topk` config keys + load params, `--draft-max`→budget;
env-wins-else-param in the loader). Serve smoke confirms: bare-temp→SPEC,
explicit-top_p→AR, GREEDY_VERIFY→AR. **STILL OPEN:** #5 (drop arch_id allowlist),
#6 (hoist swor_walk_gpu out of qwen35 crate), #7 (parallelize kernel bonus scan) —
the reuse-prereq + cheap-perf items below.

Ranked. Two are behavior regressions introduced by the serve wiring (`5884a894`)
and should be fixed before calling this done. The seam fixes also de-risk reuse.

**MUST-FIX (serve-wiring regressions, both S-effort, fix at the routing seam):**
1. **SWOR silently drops `top_p`/`top_k`/`min_p`/repeat/presence/frequency penalties.**
   temp>0 now routes to spec, but `generate_dflash`/`Speculator::step` carry only
   `temp` — a `temp=0.7, top_p=0.9` request silently loses top_p (+penalties) it would
   get on AR. Fix (cheap/correct): narrow the gate (`daemon.rs:6275`) to also require
   no sampling control beyond temperature; else fall back to AR. Full fix: thread
   top_p/min_p into `step` + a kernel truncate-renorm pass. (missed-ops #2)
2. **n-gram arches 7/10/11/12 route temp>0 to GREEDY n-gram spec** (ignores temp,
   no error). Pre-existing, but the same routing-seam class. Fix: gate those branches
   (`daemon.rs:5916/6030/6092/6152`) on `temp<=1e-6 || supports_temp_verify()`. (missed-ops #1)
3. **`HIPFIRE_DDTREE_GREEDY_VERIFY=1` + temp>0 serve = silent greedy output.**
   `supports_temp_verify()` doesn't consult it. Fix: return false when that flag is
   set. (correctness #2)

**BEFORE REUSE on other families (prerequisites):**
4. **No end-to-end Gumbel→`swor_step` distribution test.** Exactness is *conditional*
   on the Gumbel candidates being a true SWOR sample; a weak draft sampler reintroduces
   TV≈0.3 bias while still passing every current test + looking coherent (the recurring
   trap). Existing tests inject an exact host sampler, bypassing the murmur3 kernel.
   Add a committed CPU-mirror test (murmur3 Gumbel → swor_step, assert emitted-marginal
   TV at several temps/k) + promote `examples/ddtree_{swor,gumbel}_check` into a
   `coherence-gate-dflash-temp.sh`. (correctness #1, missed-ops #4)
5. **Arch_id allowlist AND-ed with the clean `supports_temp_verify` gate** (`daemon.rs:6282`)
   — a new temp-capable arch (deepseek4=4, minimax=10) is silently dropped to AR
   without a daemon edit. Duplicate "which arch does spec" decision. Drop the literal;
   let the capability bool be the sole gate. (boundaries, MEDIUM)
6. **Hoist `swor_walk_gpu` + `upload_i32` out of the qwen35 crate** into
   `hipfire-runtime::ddtree` (they're arch-free; `speculative.rs:4074/4061`). Both ports
   duplicate or cross-dep otherwise. (boundaries, LOW-MED)

**CHEAP PERF / HYGIENE:**
7. Kernel bonus/leaf draw is a serial thread-0 O(vocab) scan
   (`ddtree_swor_walk.hip:84-88,169-172`) — fires most cycles; block-parallelize with
   the existing LDS machinery. Cheapest real kernel win. (missed-ops #3)
8. Defensive: dedup-candidate skip in `swor_step`/kernel (latent, unreachable today);
   `debug_assert(dflash_batched_lm_head_supported)` on the SWOR branch to prevent a
   future dtype reading uninitialized `verify_scratch.logits`. (correctness #3/#4)
9. Re-run the existing banding A/B harness under SWOR once (breadth may now pay; was a
   greedy no-op) and close it permanently either way. (missed-ops #7)

**REUSE CONTRACT** (what a new family implements vs gets free) — see boundaries
review: provide a `Speculator` impl returning `supports_temp_verify()=true` that (a)
draws SWOR candidates via the reusable Gumbel kernel, (b) builds a `DdTree`, (c) runs
ITS OWN verify forward emitting device `target_logits` as `[(1+N)×vocab]` slot-major,
(d) calls `swor_walk_gpu`, (e) commits partial-accept arch state. Free: the entire
`ddtree.rs` CPU core, both `.hip` kernels + launchers, the temp routing/threading, the
distribution-exactness guarantee. Irreducibly per-arch: the tree-masked verify forward
+ KV/recurrent snapshot/rewind. Recommended refactors #5+#6 BEFORE the first port; a
`SpecTarget::verify_tree_logits` interface only AFTER two instances exist (avoid
speculative generality).

**Not worth it (correctly characterized as low-upside):** multi-block SWOR walk
(intrinsically sequential), realization-variance work, Gumbel+walk fusion (transfers
already tiny).
