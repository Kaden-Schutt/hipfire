# hipfire autoresearch machinery — review pack for external gap analysis

**Date:** 2026-07-07. **Audience:** a detached reviewer with NO repo access. Everything needed to
reason is in prose below. **Ask:** find gaps, unsound assumptions, missing checks, measurement
biases, and better designs across the loop engineering, the A/B certify, the bill-of-debt census,
and the decode-optimization research. Sections end with the specific questions we most want
pressure-tested.

---

## 0. What this system is

**hipfire** is a Rust-native LLM inference engine for AMD RDNA consumer GPUs (RDNA1 gfx1010 →
RDNA4 gfx1201), single HIP/ROCm-direct compute backend, no Python in the hot path. Kernels are HIP
C++ source compiled at runtime (JIT via hipcc) and cached per-arch. A Bun/TypeScript **CLI**
(`hipfire serve`) exposes an OpenAI-compatible HTTP endpoint and spawns an inference **daemon**
(a JSON-lines stdio server) as its backend.

**Autoresearch** is an automated loop that discovers kernel-level performance improvements: an LLM
agent (OpenAI **Codex**, driven headless) writes a candidate variant of a hot HIP kernel, an
**A/B certify** measures it against a baseline, wins are banked, and a **rollover** consolidates
banked wins into a new baseline. The whole thing is steered by a per-architecture **bill of debt**
(a measured census of where decode time goes). The orchestrator (Claude) sets up campaigns; Codex
does the per-kernel edits; a Python supervisor tracks state and prevents runaways.

The workload under optimization is **single-request decode** (batch size 1, autoregressive token
generation) — the latency-critical path users feel. The current test model is a 35B
mixture-of-experts with ~3B active params ("a3b", Qwen3.6-35B-A3B) at a 4-bit quantization
("mq4r"), which uses a **DeltaNet** linear-attention variant with **recurrent state** (this
matters a lot later).

---

## 1. Bill of Debt (BOD) — the per-arch census that steers everything

**Purpose.** Before optimizing, know *where the time goes* and *what kind of lever fits*. The BOD
is a per-GPU-architecture list of the hot decode kernels, each row carrying measured counters.

**Per kernel we measure:** `wall_pct` (share of total decode wall-clock the kernel consumes),
`l2_hit_pct` (L2 cache hit rate), `mem_busy` (memory system utilization %), `occ` (achieved
occupancy = resident waves / max), `vgpr` (vector registers per thread), `lds` (shared memory).

**Derived, never stored: `bound_class`.** At query time we classify each kernel:
- **DRAM-thrash** (L2-hit < ~40%): the kernel streams weights from DRAM and misses cache. Lever
  class = *traffic reduction* (read fewer total bytes).
- **cache-resident** (L2-hit > ~60%): data stays in L2. Lever class = *ALU / occupancy /
  instruction-selection*.
This is a deliberate "data not tags" principle: we store raw counters and derive the class at
read time, so re-thresholding never requires re-measuring.

**How it's measured (a hard-won detail).** AMD's `rocprofv3` PMC counters read **zero** for
%-busy metrics under the default `auto`/`high` DPM (dynamic power management) perf level, because
the perfmon clock is gated. The fix is to profile under a `profile_standard` perf level that pins
clocks so the counters actually count. Caveat: `profile_standard` pins clocks *low*, so it's valid
for counter **ratios** (L2-hit, occupancy) but not absolute tok/s — absolute throughput is measured
separately at `auto` clocks. (Related: on the RDNA4 part, `perf_level=high` actually *underclocks*
vs `auto`, so all absolute-tok/s measurement is done at `auto`.)

**Current state.** BOD censuses exist for three arches: gfx1100 (RDNA3, discrete 7900XTX, ~960
GB/s), gfx1201 (RDNA4, R9700, ~640 GB/s, but L2-resident), gfx1151 (RDNA3.5, Strix Halo integrated,
unified LPDDR5X ~256 GB/s, DRAM-thrashing). They're stored as JSON snapshots and ingested into a
SQLite store. The hottest a3b decode kernels are quantized GEMV kernels (moe_gate_up, moe_down,
qkv projections), the DeltaNet recurrent kernel, flash-attention tiles, rmsnorm, and MoE top-k.

**Questions for review (BOD):**
- Is L2-hit% the right discriminator for lever class, or does it conflate distinct regimes
  (e.g. latency-bound-but-cache-resident vs compute-bound-cache-resident)?
- We treat `wall_pct` as the candidate-priority signal. For batch-1 decode where many kernels are
  latency-bound, is wall% share the right thing to chase, or is critical-path/serialization the
  better target?
- Are we missing a counter that would separate "occupancy-starved" from "memory-latency-bound"
  more cleanly than occ + mem_busy?

---

## 2. The autoresearch loop (loop engineering)

**One-line flow:** `BOD census → coverage digest → Codex writes a kernel variant → A/B certify →
ledger → exhaustion update → (repeat) → rollover consolidates wins`.

**The driver.** A per-arch loop process ("driver_v3") runs rounds. Each round:
1. Check global stop: if every candidate kernel (those with `wall_pct ≥ 3%`, not already folded)
   has hit **K=5 consecutive dead attempts**, self-terminate.
2. Build a **coverage digest** and hand it to a Codex agent. The digest lists candidate kernels
   with `{wall%, exhaustion N/K, the last several levers tried on that kernel + their verdicts}`.
   It instructs: pick the fewest-attempts un-exhausted kernels, never re-try a lever already marked
   dead, skip exhausted kernels.
3. Codex reads the target kernel's HIP source, writes a value-preserving variant, and invokes the
   certify. It appends a one-line result (`kernel/lever/verdict/delta/learned`) to a progress log
   that feeds the *next* round's digest — so the loop accumulates institutional memory of what's
   been tried and why it failed.
4. Update per-kernel exhaustion counters from the round's verdicts (a WIN resets a kernel's counter
   to 0; each dead/inconclusive increments it).
5. Periodically, rollover (below).

**Runaway prevention (defense in depth).** This has burned us: an early keep-alive script ran away
and re-fired Codex for hours. The current guarantees: (a) the driver **self-terminates** on global
exhaustion — no keep-alive; (b) it's a single long-lived process with a `SAFETY_CAP` round backstop;
(c) every certify is bounds-checked *mechanically* (an exhausted/off-target/over-budget submission
is refused, exit 3) rather than relying on prompt discipline Codex sometimes ignores; (d) a Python
supervisor tracks per-run **codex-call budget + wall-TTL** and can stop past either.

**Branch lifecycle (also burned us).** Each card (a GPU worker slot) has a `loop/cardN` git branch;
wins commit there. The driver uses **create-or-resume** (`checkout existing || create from
baseline`), NEVER `checkout -B` — because a force-reset-to-baseline on every restart silently wiped
an accumulated stack of banked wins once (recovered only via reflog). A `loop/cardN_recovered`
safety branch is fast-forwarded each round so a stack can't be garbage-collected.

**The supervisor** (a small Python tool, "hipfire_ar") is the non-ephemeral control/persistence
layer. It holds a SQLite store (`attempts`, `bod`, `runs` tables), can `start`/`stop`/`status` a
remote loop over ssh (launch the driver + capture its real PID; liveness via `kill -0`; kill on
stop), and `ingest` the git-committed ledger so the DB reflects reality. Roles are scoped:
operator (Claude) can start/stop/status/ingest; agent (Codex) can only read state + submit a
bounds-checked certify.

**State discipline.** Durable research record = a git-committed JSONL **ledger** (one file per
`arch × kernel`, one row per attempt with verdict/perf/profile). Derived/ephemeral state
(exhaustion counters, progress log, per-arch BOD snapshot) lives in a persistent, **arch-scoped**
state dir — NOT in `/tmp` (a shared `/tmp` file once bled one arch's guidance into another arch's
agent, so everything is now arch-suffixed and off-`/tmp`; only genuinely-transient per-round scratch
stays in `/tmp`).

**Questions for review (loop):**
- The digest is the only steering signal Codex gets. Is "fewest-attempts-first + never-retry-dead"
  a good search policy, or does it under-explore promising kernels / over-explore hopeless ones?
- K=5 consecutive deads = exhausted. Is a fixed consecutive-dead counter the right stopping rule,
  or should it be confidence/expected-value based?
- Codex is a single agent per card doing free-form kernel edits. Is there a better decomposition
  (e.g. a proposer/critic split, or a fixed lever menu) than "here's the digest, go"?

---

## 3. A/B certify — the measurement, and the crutch we just found

**What it does.** Given a candidate kernel variant, decide WIN / DEAD / etc. vs a baseline. A WIN
must be (a) **value-preserving** (same output as baseline) and (b) a **real speedup**.

**The current (now-deprecated) mechanism** piped JSON-lines directly to the raw `daemon` binary,
ran generation at **greedy (temperature 0)**, and:
- **Parity gate:** compared the variant's committed **token-ids** against the baseline's at temp 0.
  If they differ, reject as `PARITY_FAIL` before any timing — this catches a change that is coherent
  but numerically wrong (e.g. a kernel that drops half its output rows but still reads fluently).
- **Adaptive sampling:** measured decode tok/s over 4–16 A/B rounds until a Mann–Whitney dominance
  statistic `f` resolved decisively (win prob ≥ 0.90 = WIN, ≤ 0.65 = DEAD, else INCONCLUSIVE),
  with a clock-skew VOID guard (reject if DPM clocks drifted > 4% between base and variant).
- **Per-variant profile feedback:** re-profiled the variant so the agent sees *why* it won/lost
  (occupancy dropped, VGPR rose, L2 unchanged, etc.) and what to try next.

**THE CRUX — this path is a biased crutch.** It bypasses the entire production surface: the CLI
serve HTTP layer, chatml/jinja chat templating, prompt normalization, and the model's *registry
sampling recipe* (which for this model is **temperature 1.0** with a presence penalty, NOT greedy).
And greedy decode structurally *hides attractors* (repetition/loops) because argmax is maximally
predictable. So a "win" measured this way may not survive the real path.

We proved it: the promoted gfx1151 win-stack, when finally run through the **real CLI path at the
production recipe (temp 1.0)** on a 5-prompt multiturn chain, threw a **repetition attractor on the
prose turn (1 of 5)** — behavior the greedy raw-daemon A/B could never surface. Every autoresearch
win to date was validated on the crutch path.

**The redesign (in progress) — serve_harness, two arms.** All measurement now goes through
`serve_harness`, a harness that spawns the real CLI `hipfire serve` (pointing it at a specific
daemon binary via an env var) and drives it over the OpenAI HTTP endpoint, capturing per-turn:
finish_reason (stop vs length=runaway), content/reasoning word splits, cached-token count (prefix
cache), decode tok/s, and a **tiered attractor detector** (unique-token-ratio and max-token-frequency
over the first-128 and last-128 tokens, plus a 3-gram repetition density over the second half).

Two arms, and a win must clear both:
- **Greedy, one-by-one** (temp 0, single independent requests): value-preservation + perf.
- **Sampled, multiturn** (temp 1.0 registry recipe, one growing same-session conversation):
  coherence under the *real* sampling — and, because the model has recurrent DeltaNet state, this
  also catches **cross-request state bleed** (state leaking from one request into the next, which a
  single-request check structurally cannot see).

**A determinism problem the redesign forced into the open.** Two issues:
1. Q8 DeltaNet **stochastic rounding** makes even greedy output non-reproducible run-to-run, so a
   text/token diff would false-fail parity. The greedy parity arm therefore runs under a
   deterministic mode (FP32 DeltaNet state + a determinism flag) — a path chosen to *isolate the
   kernel's math* from rounding noise, accepting that it's not production precision.
2. The sampled arm at temp 1.0 has **no per-request seed** in the engine, so a single
   variant-vs-baseline comparison is RNG-dependent (the prose attractor could be luck). We're
   **adding a per-request seed** (thread `seed` from the HTTP request → the GPU sampler RNG) so
   temp-1.0 output is reproducible and variant-vs-baseline is apples-to-apples in one run each.

**A note on parity via the HTTP path.** The HTTP endpoint returns *text*, not token-ids, so
greedy parity becomes **byte-exact text** rather than token-id-exact. At greedy+deterministic these
are effectively equivalent for value-preservation, marginally weaker in theory.

**Questions for review (certify):**
- The greedy parity arm runs at FP32+deterministic (not production Q8). Is checking value-
  preservation on a *different numerical path* than production sound, or can a kernel be parity-clean
  in FP32-deterministic yet diverge under production Q8 stochastic rounding in a way that matters?
- The coherence arm defines a win as "variant introduces no attractor the baseline didn't at the
  same seed." With one seed, is that enough, or do we need a small seed-set to avoid a single
  unlucky/lucky seed? (We chose single-seed reproducibility over N-run rate for speed.)
- Perf is now measured on the CLI path (HTTP + chatml overhead → lower absolute tok/s). We rely on
  the *relative* delta being faithful. Is there a bias where CLI overhead compresses or inflates the
  relative kernel delta vs the raw compute path?

---

## 4. Advancing baseline + composite rollover + fixer (the current design)

This is the newest design layer, motivated by a **stack-dilution failure** we observed: the loop
kept banking wins *past* its peak, and later per-kernel wins — each measured against the *original*
baseline — **overwrote better banked wins with marginally-winning worse ones**, dragging a verified
+3.07% cumulative down to +1.00%. Root cause: every variant was measured against the original
baseline, not against the accumulated stack, so "beats original" ≠ "beats current best."

**The fix — three roles:**
- **Loop agent (per card): hill-climb an OWN advancing baseline `B_a`.** `B_a` starts as the
  original baseline and is *replaced by the winning variant on every WIN* (its daemon becomes the
  next round's comparison baseline). A variant must beat `B_a`, so the agent's baseline
  *monotonically climbs* — a later sub-par variant can never overwrite a better banked win. Dilution
  is structurally excluded, not merely caught late. Agents never see the original baseline.
- **Rollover: the ONLY original-vs-composite judge.** It assembles the composite of banked wins
  (union across cards, best-per-kernel merge), and judges the composite against the *original*
  baseline on both arms (perf + coherence over a persistent guard-prompt set). Coherent → promote
  as the new original baseline. Incoherent → delegate to the fixer.
- **Fixer: correct the offending kernel.** On an incoherent composite, localize (by ablation) which
  banked kernel-win induces the attractor, then **fix that kernel's coherence bug while keeping its
  perf**, and re-certify through both arms. Ablate-and-re-open (drop the win, re-open the kernel for
  a fresh search) is the *fallback* only if the kernel can't be corrected. The offending prompt is
  persisted into a per-arch **guard-prompt set** that every future certify (coherence arm) and every
  future rollover must pass — so a fixed failure can never silently regress.

**Questions for review (rollover/fixer):**
- Is monotonic per-agent hill-climbing the right local policy, or does it trap the agent in a local
  optimum that a temporary regression would escape? (Simulated-annealing-style acceptance?)
- Best-per-kernel composite merge assumes per-kernel wins compose additively. They don't always
  (two independently-good kernel changes can interact negatively). How should the composite be
  validated/assembled to catch cross-kernel interactions before promotion?
- The fixer "corrects the offending kernel while keeping perf." Is that reliably possible, or is
  perf often *causally* tied to the coherence bug (the thing that made it fast is the thing that
  broke it)? What's the right expected success rate before we lean on the ablate fallback?

---

## 5. The research findings so far (AR decode)

Context for whether the whole kernel-decode-autoresearch premise is even worth continuing:

- **Decode is GPU-bound but batch-1 latency-limited.** Isolated decode windows show ~100% GPU busy,
  but throughput spans ~8× across model sizes because batch-1 under-saturates bandwidth and compute
  units (a small model runs 6× below bandwidth peak). Being latency-bound, it's largely
  arch-invariant — RDNA3 ≈ RDNA4 on the same model.
- **On the DRAM-thrash discrete arches (gfx1100) and the cache-resident arch (gfx1201), kernel
  micro-tuning is near-tapped for batch-1 decode:** fresh-process wins net ~0% after warming. The
  compiler already pipelines the simple gathers; manual tuning mostly runs "in the shadow" of the
  memory latency.
- **The register-cliff finding (consistent across DRAM-thrash arches):** the intuitive
  "traffic-reduction" lever for a cache-missing kernel — bigger row-tiles that reuse the activation
  across more output rows — *raises VGPR pressure → drops occupancy → loses*. The bytes saved don't
  beat the occupancy lost. This falsified our own prior for gfx1151.
- **gfx1151 (the integrated, bandwidth-starved arch) is the exception and produced the first
  fresh-process-surviving win: +3.07%** — but via a *different* lever than predicted: **latency /
  LDS-barrier removal on the catastrophically under-occupied kernels** (attention at ~1.4%
  occupancy, rmsnorm at ~0.4%). Removing shared-memory staging + barriers cut wall time without the
  register penalty. The loop *discovered* this after its traffic-reduction hypothesis died — i.e.
  the loop corrected our lever thesis from data. (This win now needs re-validation through the new
  two-arm gate; the prose attractor above is the open question.)
- **Strategic read:** for batch-1 decode, the big remaining levers are *architectural*
  (speculative decoding / multi-token prediction to batch the verify; quantization-format redesign),
  not kernel micro-tuning — EXCEPT on occupancy-starved arches like gfx1151 where real kernel
  headroom remains.

**Questions for review (findings):**
- Given "kernel micro-tuning ~tapped for batch-1 on 2 of 3 arches," is continued per-kernel
  autoresearch the right investment, or should the loop's search space shift to
  cross-kernel/fusion/format-level changes (bigger structural edits than a single-kernel variant)?
- Is there a principled way to *predict* which kernels have real headroom (like gfx1151's
  under-occupied ones) vs which are compiler-saturated, so the loop doesn't waste rounds on the
  latter?

---

## 6. Hard-won measurement rules (so the reviewer knows our guardrails)

- **Warm the kernel JIT cache + DPM state before measuring** (cold first run is 3–7× slower).
- **JIT tax is per-(config × kernel-shape):** warming one A/B cell doesn't warm its neighbors; a
  slowdown that *survives a rerun* is NOT JIT, it's real.
- **Within-session A/B is optimistic below ~2%;** sub-percent "wins" often net ~0 on a fresh-process
  re-measure (a 15-win within-session stack once netted −0.34% fresh-process). Cross-commit claims
  use fresh processes.
- **Byte-parity requires FP32 state + a determinism flag** (Q8 stochastic rounding breaks
  bit-exactness).
- **Prompt bytes matter:** one newline can swing a spec-decode acceptance metric 17% (same token
  *count*, different token *sequence* → different distribution). All perf claims use byte-identical
  prompts with recorded md5.
- **Coherence is not optional:** a tok/s win that ships an attractor/loop/special-token-leak is a
  regression on the output axis hiding behind a number. Multiple "synthetic-win → production-falsify"
  episodes are on record. This is *why* the serve_harness redesign exists.

---

## 7. The single most important things to pressure-test

1. **Is the two-arm serve_harness gate actually crutch-free, or did we just move the crutch?**
   (e.g. does FP32-deterministic parity + single-seed coherence introduce a *new* blind spot?)
2. **Is the advancing-baseline + composite + fixer design sound**, or does monotonic local
   hill-climbing + best-per-kernel merge have a failure mode we haven't hit yet?
3. **Is per-kernel batch-1 decode autoresearch worth continuing** given 2 of 3 arches look tapped,
   or should the machine be repointed at a higher-leverage search space — and if so, what
   measurement + gating would that need?
4. **What are we not measuring** that would change these conclusions (a counter, a workload shape
   like batch>1 or long-context, a coherence axis beyond attractors)?
