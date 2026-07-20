# Case studies — dated evidence, not live floors

Historical kernel-tuning outcomes from the hipfire git log and branch
campaigns. Each entry is **measured** or **historical**: a commit (or
explicit branch note), a fixture scope, a disposition, and a durable
lesson.

**Not** product floors, admissions, or “current best” numbers. Live
baselines live in `tests/speed-baselines/<arch>.txt` and are owned by
the speed-gate workflow. Methodology owners:

- Perf protocol: [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md)
- Capability vs perf-variant selection: [`docs/methodology/perf-arch-discipline.md`](../../../docs/methodology/perf-arch-discipline.md)
- Validation route selector: [`docs/VALIDATION.md`](../../../docs/VALIDATION.md)

Disposition vocabulary used below:

| Disposition | Meaning |
|---|---|
| **shipped** | Landed on the tuned path for the stated scope |
| **reverted** | Candidate was rejected after measurement; keep the lesson |
| **rejected / null** | Tried, did not win; may remain opt-in only |
| **correctness fix** | Restored numeric correctness; not a perf claim |
| **harness / measurement** | Root cause was fixture or process, not a kernel lever |

---

## CS-1 — wave64 CDNA3 port (shipped win)

| Field | Value |
|---|---|
| **Date / era** | 2026-04 (commit `4105035`) |
| **Commit** | `4105035` — *perf(cdna3): full wave64 port of all hot HFQ4 kernels — MI300X decode 48.6 → 96 tok/s* |
| **Fixture** | MI300X / gfx942; A3B 3.5-35B MQ4 decode; Fourier-explanation prompt; ctx=512, max=64 |
| **Lever** | Wave-size port (`.wave64.hip` / wave64-native dispatch) |
| **Disposition** | **shipped** on wave64-native CDNA3 path |

**What happened:** gfx94x is wave64-native; wave32 HFQ4 kernels left half
the lanes masked. Porting the hot HFQ4 set with 2-rows-per-block wave64
lane decomposition roughly doubled decode on the recorded MI300X fixture
in that commit message (48.6 → 96 tok/s).

**Validation (as of that work):** cold same-prompt smoke on MI300X was
byte-exact vs 7900 XTX (gfx1100) on the Fourier prompt. Warm outputs
could diverge at a single token from MoE-down `atomicAdd` ordering
(already present on the wave32 base path; not introduced by the port).
That commit does **not** record a CPU-reference synthetic channel-test
as its acceptance. Do not treat retired `coherence-gate-*.sh` batteries
as current acceptance — see `docs/VALIDATION.md`.

**Lesson:** wave-size mismatch is a cliff, not a small inefficiency.
Gate the path with `ArchCaps::is_wave64_native()` (or equivalent atom
allowlist) so RDNA wave32 dispatch stays untouched.

---

## CS-2 — nontemporal weight-load (reverted fake win)

| Field | Value |
|---|---|
| **Date / era** | 2026-04 |
| **Commits** | `0532579` (candidate) → `34eb024` (revert) |
| **Fixture** | gfx1100; 9B MQ4 decode; within-session A/B then clean-baseline bisect |
| **Lever** | `__builtin_nontemporal_load` on hot decode weight reads |
| **Disposition** | **reverted** |

**What happened:** Within-session A/B looked like a small decode win and
was committed. Bisect against the committed speed-gate baseline showed a
large regression (commit message order-of-magnitude: on the order of
−13% decode on that 9B MQ4 / 7900 XTX fixture). Warm L2 / skewed GPU
state hid the real cold-path behavior.

**Hypothesis retained in the revert message:** nontemporal loads on
RDNA3 defeated wave-level coalescing/prefetch the default path got for
free.

**Lesson:**

1. Bisect against the **committed** baseline / fresh process, not the
   last bench in the same shell.
2. Reasonable ISA intuition is not evidence.
3. Reverts are first-class: keep the WHY so the lever is not retried
   blindly.

---

## CS-3 — k2x32 wider-row lm_head (null result, opt-in kept)

| Field | Value |
|---|---|
| **Date / era** | pre-2026-05 (commit `f670e16`) |
| **Commit** | `f670e16` — *experiment(gemm): k2x32 wider-row lm_head — null result* |
| **Fixture** | gfx1100 / 7900 XTX; 27B MQ4 lm_head at M=248320, K=5120, B=16; k2 baseline vs k2x32 |
| **Lever** | Wider-row / deeper multi-row WMMA issue |
| **Disposition** | **rejected / null** on that gfx1100 lm_head fixture; auto-dispatch remained **k2** for that fixture/commit (`M>=8192` per `fe4ccb4`); variant retained behind `HIPFIRE_WO_WMMA_VARIANT=k2x32` |

**What happened:** 32-row blocks vs 16-row were supposed to amortize
X-fragment loads on the 27B lm_head shape. Measured slower on that
gfx1100 fixture (k2 ~1564–1587 µs / ~446–452 GB/s vs k2x32 ~2280–2297 µs
/ ~307–310 GB/s). Root cause attributed to doubled accumulators + dequant
live ranges → register pressure / occupancy loss (latency-bound, not
BW-bound).

**Later, separate evidence (not a reversal of this null):** current
`crates/rdna-compute/src/gemm.rs` auto-selects `k2x32` on RDNA3.5
(`gfx115x`) for a **distinct** small-M / prefill residual shape
(`m < 8192`). That is a different arch + shape + dated path; it does not
un-reject the original gfx1100 large-M lm_head null.

**Lesson:** past a VGPR wall, “do more parallel WMMA” loses on the shape
you measured. Negative results with a named env override
(`HIPFIRE_WO_WMMA_VARIANT=k2x32` in that era) save the next pass from
rediscovery. Null ≠ delete history; later wins on other atoms/shapes need
their own dated cases.

---

## CS-4 — gfx11 WMMA C-mapping silent corruption (correctness fix)

| Field | Value |
|---|---|
| **Date / era** | fixed in `b7ac66a` after ~6 weeks latent |
| **Commit** | `b7ac66a` — *wmma correctness fix + MQ6 family + cross-arch prefill + gate framework* |
| **Fixture** | gfx11 WMMA path; synthetic deterministic channel-test inputs |
| **Lever** | none (mapping bug) |
| **Disposition** | **correctness fix** |

**What happened:** C-output mapping
`acc[j] = C[2*j + (tid>>4)][tid & 15]` was wrong. Speed floors and
English-shaped smoke still passed; quality looked like “quant loss.”
Element-wise channel-test vs CPU reference exposed a row-mod-16 mismatch
pattern.

**Lesson:**

1. Channel-test / numeric oracle is the load-bearing correctness route
   for kernels — not speed floors and not retired coherence batteries
   (`docs/VALIDATION.md`).
2. Cooperative-lane mappings are silent-corruption magnets.
3. Row-mod-16 (or equivalent dimensional) histograms belong in every
   WMMA/MFMA channel-test.

Cited by `.agents/skills/hipfire-arch-port/` as the cautionary tale for
new matrix-engine ports.

---

## CS-5 — 27B DFlash recovery: deleted residual-WMMA kernels (shipped)

| Field | Value |
|---|---|
| **Date / era** | recovery commit `9a2c667` |
| **Commit** | `9a2c667` — *perf-recovery: restore 27B DFlash perf + flip prompt_normalize default ON + DFlash speed-gate* |
| **Fixture** | gfx1100 / 7900 XTX; 27B-3.5 DFlash LRU-code decode (broken master ~95 tok/s → post-fix ~199 tok/s) |
| **Lever** | none new — restore load-bearing residual-WMMA sources removed as “dead” |
| **Disposition** | **shipped** recovery; DFlash metric added to speed-gate |

**What happened:** Decode looked ~half speed (95 → 199 tok/s after fix on
that fixture). Root cause in `9a2c667`: PR #32 cleanup deleted
`gemm_hfq4g256_residual_wmma{,2,_k4}.hip` as dead weight, but they were
load-bearing on the K4 dispatch path for 27B verify-shape GEMMs (per-cycle
cost on 64-layer × B=16 verify jumped ~57 → 100+ ms). Restoring those
kernels recovered DFlash.

**Not the root cause of this regression:** prompt whitespace / shape.
Prompt normalization was measured earlier in `8a4a211` (PEP-8 raw
157.1 → normalized 199.0 tok/s on 27B-3.5 DFlash) and was only
**default-flipped ON** as a separate deliverable inside `9a2c667`. Treat
prompt-shape τ sensitivity as its own harness lesson (commit prompts,
record md5 — see perf-benchmarking); do not attribute the `9a2c667`
kernel-deletion regression to prompt bytes.

**Lesson:**

1. Kernel-cleanup PRs must prove DFlash (not only AR) before/after; AR-only
   speed-gate missed a ~40% DFlash hit until this recovery added the metric.
2. “Unused” residual-WMMA siblings can still be selected by shape/dispatch —
   deletion needs call-graph + bench evidence, not name hygiene.
3. Spec-decode τ is also prompt-byte-sensitive (`8a4a211`); embed prompts as
   committed files and record prompt md5 — orthogonal to this kernel case.

---

## CS-6 — wave64 residual gemv on MI300X (small additive win)

| Field | Value |
|---|---|
| **Date / era** | 2026-04-28 era (gfx942 baseline capture notes residual wave64 close-out after `4105035`) |
| **Commit** | branch-era residual port; baseline file `tests/speed-baselines/gfx942.txt` cites the residual gemv lever in its header narrative — verify tip hash before external citation |
| **Fixture** | MI300X gfx942; 27B / A3B mq4 decode; rocprof attribution on residual gemv |
| **Lever** | Wave64 port of residual gemv family |
| **Disposition** | **shipped** additive; wall-clock lift small on BW-saturated shapes |

**What happened:** Hot residual gemv still wave32 after the main wave64
port. Per-call kernel time improved in the recorded rocprof; end-to-end
decode moved only a few percent (inside or near noise on some rows)
because the shape was already HBM-bound.

**Lesson:** wave64 pays most when lanes are under-utilized (multi-row
fused projections). On saturated per-row gemv, ship if correctness-safe
and additive, but do not promise another 2× decode.

**Cross-arch:** gate with `is_wave64_native()`; RDNA unchanged.

---

## CS-7 — LDS-staged X share on gate_up (null; opt-in retained)

| Field | Value |
|---|---|
| **Date / era** | issue #60 campaign (gfx1100, ROCm 7.2 era) |
| **Commit** | historical branch experiment — verify hash in your checkout before external citation |
| **Artifact** | `kernels/src/gemm_gate_up_hfq4g256_wmma_ldsx.hip`; opt-in `HIPFIRE_GATE_UP_VARIANT=ldsx` |
| **Fixture** | Qwen 3.5 9B MQ4; gfx1100; `HIPFIRE_PROFILE=1`; pp32/128/512 |
| **Lever** | LDS-staged X share to hide VMEM before WMMA B |
| **Disposition** | **rejected / null** for default; infrastructure kept opt-in |

**Measured shape (that campaign):** per-call gate_up and end-to-end
prefill both regressed at every recorded pp; effective BW collapsed as
batch grew. ISA Gate 0 looked clean (VGPR healthy, barrier elided on
single-wave blocks); microbench still failed.

**Why ISA-clean lost:** baseline already hid much of `vmcnt(0)` via
wave-level ILP; LDSX added more stall events the scheduler hid less
well.

**Lesson:** pair ISA inspection with wall-time microbench before
committing a rewrite. Do not re-try this exact design; different
mechanisms (weight-side LDS, gfx12 prefetch, deeper independent WMMA
issue) are separate experiments.

---

## CS-8 — K4 output-mapping fix (correctness; no auto-dispatch win)

| Field | Value |
|---|---|
| **Date / era** | issue #60 follow-on; fix commit `48aa9d5` |
| **Commit** | `48aa9d5` — *fix(gemm): K4 output mapping — was swapped relative to canonical wave32 WMMA C-mapping* |
| **Artifact** | `kernels/src/gemm_hfq4g256_residual_wmma_k4.hip`; env `HIPFIRE_WO_WMMA_VARIANT=k4` |
| **Fixture** | Qwen 3.5 9B MQ4; gfx1100; channel-test K×batch matrix + residual microbench |
| **Lever** | K-tile depth (K4) — blocked until mapping fixed |
| **Disposition** | **correctness fix**; remains opt-in; auto-dispatch unchanged |

**What happened:** K4 used a transposed mental model of wave32 WMMA C
layout (same class as CS-4). Broken channel-test showed near-total bad
cells; fixed mapping matched K2. At m&lt;8192 on that fixture, K4 did not
beat ksplit; not promoted.

**Methodology bugs caught alongside the kernel bug:**

1. Stale precompiled HSACO / hash sidecars can make a “fix” look
   bit-identical-wrong — force invalidate compiled blobs and confirm a
   recompile log line.
2. Row-invariant synthetic weights hide row-shuffle bugs at batch=1 —
   vary every dimension the kernel could permute.
3. Multi-reviewer plans beat single-reviewer confident misdiagnosis on
   high-stakes mapping work.

---

## CS-9 — capability predicate selected wrong perf variant (shipped discipline)

| Field | Value |
|---|---|
| **Date / era** | 2026-06-12 mandate; fix `24e4baa9` |
| **Commits** | `303d69e9` (`ldscoop` falsified on gfx1100); `e3232034` (nosync on gfx1150/1151, `ldscoop` catch-all on others); `24e4baa9` (restore plain on RDNA3 dGPU) |
| **Fixture** | gfx1100 DFlash / gate_up hfq4g256 WMMA sub-variants (`plain` vs `ldscoop` vs `ldscoop_nosync`) |
| **Lever** | none new — selection polarity |
| **Disposition** | **shipped** selection fix; full ledger discipline in methodology doc |

**What happened:** The measured Strix (gfx1150/1151, LPDDR) winner was
`gemm_gate_up_hfq4g256_wmma_ldscoop_nosync`, not plain `ldscoop`.
`e3232034` defaulted nosync on gfx1150/1151 and used `ldscoop` as the
catch-all “others” default — which routed gfx1100 dGPU (GDDR6 + Infinity
Cache) onto `ldscoop`, a variant already falsified slower on that chip
(`303d69e9`). rocprof attributed most of a ~14% DFlash decode regression
to the wrong variant. `24e4baa9` restored **plain** WMMA on RDNA3 dGPU
via `is_rdna3_dgpu()`; Strix kept nosync; RDNA4 left on `ldscoop` pending
its own measurement.

**Rule (durable):** capability predicates answer ISA correctness.
Perf sub-variants use an explicit arch allowlist + measured ledger
defaulting to the portable choice. See
`docs/methodology/perf-arch-discipline.md`. Do not copy the 2026-06-12
ledger table into this skill as if it were live — the methodology owner
and any machine ledger file are authoritative.

---

## How to add a case study

Append a new `CS-N` section. Required fields:

- **Date / era** and **commit** (or explicit “verify hash” branch note)
- **Fixture** — arch, model/quant, prompt or bench identity when known
- **Lever** — name from `levers.md` or “none”
- **Disposition** — shipped / reverted / rejected / correctness / harness
- **Lesson** — what a future contributor must not re-learn the hard way

Rules:

- Quote numbers only as **that fixture’s measured record**. Never
  restate them as current floors or admissions.
- Rejected and null results stay rejected — do not soften into “maybe
  ship later” without new dated evidence.
- Prefer linking methodology and `docs/VALIDATION.md` over inventing a
  universal gate list here.
