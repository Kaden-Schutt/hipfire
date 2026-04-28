# Case studies — wins, losses, and the methodology in action

Five worked examples from the actual hipfire git log. Each shows
the workflow from `playbook.md` running on real engineering — a
mix of decisive wins, fake wins caught by discipline, and silent
corruption caught by gating. The lessons are the durable artifact;
the numbers will move as the engine evolves.

---

## §1 — wave64 CDNA3 port (decisive 2× win)

**Commit**: `4105035` — "perf(cdna3): full wave64 port of all hot
HFQ4 kernels — MI300X decode 48.6 → 96 tok/s"

**Bottleneck**: MI300X (gfx94x) is wave64 native, but hipfire's
HFQ4 kernels were wave32. On a wave64 wave running a wave32 kernel,
half the lanes silently mask out — the kernel produces correct
output but at 50% effective throughput.

**Lever**: per-arch wave64 variant. Ten kernels ported with
2-rows-per-block wave64 lane decomposition.

**Numbers**:

```
A3B decode pre-port:  48.6 tok/s on MI300X (gfx942)
A3B decode post-port: 96.0 tok/s — matches 7900 XTX on the same model
```

**Validation path**:
- Channel-test against CPU reference on synthetic HFQ4 weights
  (caught a wave-lane-mapping bug pre-merge).
- Coherence-gate ran against the standard model matrix.
- Speed-gate on the MI300X showed no regression on RDNA archs
  (the wave64 path is gated by `arch.starts_with("gfx94")`).

**Lesson**: wave-size mismatch is a 2× perf cliff, not a small
inefficiency. Worth a proper port any time the target arch's wave
size differs from your kernel's. The pattern (separate `.wave64.hip`
or `.gfx942.hip` file) keeps RDNA dispatch unaffected.

---

## §2 — nontemporal weight-load fake win (caught by clean-baseline bisect)

**Commits**: `0532579` (the candidate) → `34eb024` (the revert).

**Setup**: an experiment to use `__builtin_nontemporal_load` for
weight reads on hot decode kernels, intuition being that decode
weights are streaming-read (each token re-reads them once) and
shouldn't pollute L2.

**Initial measurement** (within-session A/B): +2.0% decode tok/s on
9B MQ4. Looked plausible, committed.

**Bisect against committed speed-gate baseline** (April 12 anchor):
**−13% decode**. 131 → 113 tok/s on 7900 XTX 9B MQ4.

The within-session A/B happened in a GPU state already skewed by
many preceding bench runs — preceding warmup put L2 in a state
where the nontemporal change *appeared* to win, but a fresh process
with a cold cache showed the actual regression.

**Hypothesis** (in the revert commit message): on RDNA3, the
nontemporal load path bypasses cache-line allocation but ALSO
defeats wave-level coalescing/prefetch behavior the default load
path gets for free. Each wave was issuing one coalesced 128-byte
transaction for 32 packed-u32 weight reads; the nontemporal hint
broke that coalescing pattern.

**Lessons**:
1. **Always bisect against the committed baseline**, not your last
   bench run. The speed-gate baseline file
   (`tests/speed-baselines/<arch>.txt`) exists for exactly this reason.
2. **Hypothesis without measurement = noise**. The nontemporal
   intuition was reasonable on paper. The hardware behavior was
   different.
3. **Reverts are first-class commits**. The revert commit message
   captures the WHY so the next contributor doesn't try the same
   thing for the same reasons.

---

## §3 — k2x32 wider-row variant (null result, kept for posterity)

**Commit**: `f670e16` — "experiment(gemm): k2x32 wider-row lm_head —
null result"

**Hypothesis**: on the M=248320 lm_head kernel, a 32-row block
(versus the 16-row default) would halve block count and amortize
X-fragment loads across 2 WMMA issues per K-tile.

**Result**: 46% **slower** at the target shape. 1564 µs (k2 baseline)
→ 2280 µs (k2x32). Effective BW dropped from 446 GB/s to 307 GB/s.

**Root cause**: doubled accumulator (`float8_t × 2`) plus 4× dequant
live ranges pushed wave register pressure past the compiler's
budget, forcing spills or reducing effective occupancy. 310 GB/s
(32% of 960 peak) signals latency-bound, not BW-bound — more
parallel WMMAs don't help when you can't pipeline them.

**Why kept**: the kernel + `HIPFIRE_WO_WMMA_VARIANT=k2x32` env
override stayed in the tree even though auto-dispatch routes around
it. A future revisit with LDS-staged B-share + manual register
budgeting might unlock it. The negative result is a known-checkpoint
that future tuning passes don't have to re-discover.

**Lesson**: register pressure is the gating constraint past a
certain point. More parallel work does not help when the compiler
can't pipeline the issue chain. When you measure a kernel at
~30% peak BW and the obvious "do more" lever loses, the bottleneck
is latency, not BW — different lever class.

---

## §4 — gfx11 WMMA C-mapping silent corruption (caught only by channel-test)

**Commit**: `b7ac66a` — "wmma correctness fix + MQ6 family +
cross-arch prefill + gate framework"

**Setup**: gfx11 (RDNA3) WMMA was the WMMA workhorse for hipfire
since the v0.1.4 line. The C-output mapping
(`acc[j] = C[2*j + (tid>>4)][tid & 15]`) was silently wrong for
**~6 weeks**.

**How it stayed hidden**:
- All speed-gates passed — the kernel produced numbers, just wrong
  ones in the same ballpark.
- Coherence-gates passed — output was English-shaped, on-topic-ish,
  no panics or zero-tokens or attractor loops.
- Functional tests passed — comparing kernel output to itself
  doesn't catch a systematic mapping error.
- Real-model tok/s didn't regress noticeably — quality degradation
  was within "MQ4 is lossy by nature" range.

**How it got caught**: a channel-test that compared kernel output
**element by element against a CPU reference on synthetic
deterministic inputs** flagged a row-mod-16 pattern of mismatches.
The histogram diagnostic that landed in PR #56's gfx12 channel
tests is the tool that would have caught this in 30 seconds.

**Lessons**:
1. **Channel-test is the load-bearing correctness gate**, not
   speed-gate or coherence-gate. The other two are weaker signals
   that miss systematic errors.
2. **Per-lane mappings are silent-corruption magnets.** WMMA, MFMA,
   and any cooperative-thread reduction has implicit mapping
   conventions that you can get wrong without any obvious symptom.
3. **The row-mod-16 histogram diagnostic is reusable** — every
   future WMMA / MFMA channel-test should include it (it would
   have caught this in seconds).

The arch-port skill (`.skills/hipfire-arch-port/`) explicitly cites
this commit as the cautionary tale for new contributors. PR #56
followed that guidance and avoided the trap entirely on gfx12.

---

## §5 — 27B DFlash perf recovery (root-causing a real regression)

**Commit**: `9a2c667` — "perf-recovery: restore 27B DFlash perf +
flip prompt_normalize default ON + DFlash speed-gate"

**Setup**: 27B DFlash decode regressed 30-40% suddenly. Looked
catastrophic.

**Investigation path** (over 6 hours of bisecting):

1. Suspect rocBLAS — null. `HIPFIRE_ROCBLAS_OFF=1` made no difference.
2. Suspect DKMS / firmware — null. `dmesg` clean, kernel firmware
   versions matched.
3. Suspect mold / sccache — null. Clean rebuild reproduces.
4. Suspect DPM / thermal — null. `pp_dpm_sclk` looked normal.
5. **Found it**: prompt structure. A whitespace-cleanup edit to a
   bench script changed `\n\n\n` → `\n\n`. Same prompt by token
   count, totally different by token sequence. τ collapsed from
   9.42 to 8.07; tok/s from 199 to 161.

**Lessons** (now codified in CLAUDE.md and AGENTS.md):
1. **Prompt structure dictates τ.** One newline character can swing
   τ by 17%. Embed prompts as committed files, record prompt md5
   alongside results.
2. **Tight stddev on a spec-decode bench is suspicious, not
   reassuring.** The "before" measurement had tight stddev
   suggesting a deterministic attractor; real acceptance is wider.
3. **Bisect attribution is hard when the cause is in the test
   harness, not the engine.** Always reproduce the regression on
   a different prompt before deep-diving the engine.

The fix: implement engine-side `\n{3,}` → `\n\n` collapse default-on
(`prompt_normalize` config key, commit `9a2c667`). +24% τ on PEP-8-
style code prompts vs the opt-out path.

---

## §6 — wave64 residual gemv on MI300X (small win, BW-saturation ceiling)

**Commit**: this branch — "perf(cdna3): wave64 port of gemv_hfq4g256_residual"

**Bottleneck**: rocprof on 27B 3.6 mq4 decode (50 gen tokens, asym3 KV)
showed `gemv_hfq4g256_residual.kd` at 19.2% of GPU time — the largest
non-wave64 kernel after the 2026-04-17 (`4105035`) wave64 port. The
original commit ported 10 hot HFQ4 kernels but missed the residual
variants of gemv (`_residual` and `_wide`).

**Hypothesis**: wave64 port should give 1.5-2× per-call speedup
(matching the original commit's win on the same kernel family).

**Lever**: §1 wave-size port. New `gemv_hfq4g256_residual_wave64.hip`
with 2-rows-per-block layout (warp_id selects row, lane drives the
32-lane reduction unchanged). Dispatch routes via `has_wave64_native(arch)`.

**Numbers**:

```
27B 3.6 decode pre-port:  66.0 tok/s on MI300X (gfx942)
27B 3.6 decode post-port: 68.1 tok/s     (+3.2%, within noise)

per-call kernel time:
  pre-port:   28783 ns/call (single-row wave32 on wave64 hardware)
  post-port:  25222 ns/call (two-rows-per-block wave64)              -12.4%

A3B 3.6 decode pre/post: 194.6 → 198.0 tok/s (+1.7%, within noise)
```

**Why the small wall-clock delta despite -12% kernel time**: residual
gemv on this shape (M ~ 5120, K ~ 5120, single output row per warp32) is
**bandwidth-bound, not lane-bound**. Each row already saturates a wide
HBM3 read on MI300X regardless of wave size — the wave32 kernel was
issuing one coalesced 128-byte transaction every 32 packed-u32 weight
reads, and the new wave64 kernel pays the same BW for half the lanes.
The 12% per-call drop is real (less ALU pressure on the unused upper
lanes) but the wall-clock is dominated by the BW transfer, not the
compute pipeline.

**Lesson**: wave64 port wins biggest on kernels that are
**lane-utilization-bound** (multi-row fused projections like qkv, where
each lane has its own row-output to compute). On per-row gemv shapes
that are already BW-saturated, the win is incremental — ship it because
it's correctness-preserving and additive with future fusion work, but
don't expect 2× decode.

**Cross-arch**: gated by `has_wave64_native(&self.arch)`, so
gfx908/gfx940/gfx941/gfx942 only. RDNA archs unchanged. Speed-gate on
gfx1100 should pass byte-exact.

## How to add a case study

If you land a real perf win or revert worth documenting, append a
new §N section here. Required fields:

- **Commit** — the canonical commit hash.
- **Bottleneck** — what the profile said.
- **Lever** — which entry from `levers.md` you used.
- **Numbers** — before / after with binary md5 + prompt md5.
- **Validation path** — which gates ran, what they showed.
- **Lesson** — the durable insight a future contributor needs.

Negative results (null lift, fake win caught) are equally
valuable — they save the next person from re-running the same
experiment. Don't omit them just because they "didn't ship."
