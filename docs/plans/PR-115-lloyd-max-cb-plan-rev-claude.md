# Adversarial review — `PR-115-lload-max-codebooks-mq3.md`

**Reviewer:** Claude (Opus 4.7, 1M ctx)
**Date:** 2026-05-06
**Scope:** Structural review of the implementation plan, plus
adjudication of two prior reviews
(`PR-115-lloyd-max-cb-plan-rev-gemini.md`,
`docs/plans/PR-115-lloyd-max-cb-plan-rev-glm5.md`). Each cross-review
claim is validated or rejected against the actual code below.

I built nothing and ran nothing. I read the plan, both reviews, the
two kernels, the dispatcher, the kernel selector, the perplexity /
benchmark harnesses, the empirical findings doc, and verified each
factual claim against the source.

---

## TL;DR

The plan's direction is right (K4 unroll + fp32 LDS-resident codebook,
new `gfx1100.hip` file behind a kernels.rs arch-selector). What's
missing or wrong:

| ID | Severity | Issue | Source |
|----|----------|-------|--------|
| C1 | **Blocker** | Bench harness doesn't support Lloyd-MQ3 — step 4 not runnable as written | mine |
| C2 | **Blocker** | Existing Lloyd-MQ3 dispatch uses raw `self.hip.launch_kernel` — not graph-capture safe under `HIPFIRE_GRAPH=1` | glm5 (validated) |
| C3 | Major | Bottleneck attribution unverified by disassembly; should split K4 vs LDS into bisectable commits | mine |
| C4 | Major | 0.8B Lloyd-MQ3 (ppl=155.22) is the wrong numerical anchor; use 4B (ppl=22.56) | mine |
| C5 | Major | Bank-conflict arithmetic wrong (conclusion still right) | mine + glm5 (validated) |
| C6 | Major | Cooperative load isn't a single coalesced 64 B txn; is 4 cachelines @ 112 B stride | mine |
| C7 | Major | No logits-Δ check vs old kernel — ppl smooths over kernel bugs | mine + gemini (validated) |
| C8 | Medium | Tail iterations under K4 loading 4 codebooks → potential OOB on missing groups | gemini (validated, but indirect) |
| C9 | Medium | `coherence-gate.sh` likely needs `HIPFIRE_ALLOW_MQ3_LLOYD=1` env to actually exercise Lloyd path | mine |
| C10 | Medium | Pre-commit hook can't be a validation-ordering tool; fires on commit only | glm5 (validated) |
| C11 | Medium | Tail K-sweep test missing — Qwen K's may not exercise all `groups_per_row % 4` cases | gemini (validated, location wrong) |
| C12 | Minor | Arch selector mirrors HFQ3's narrow gfx1100/1/2; gfx115x/gfx12 unaddressed | glm5 (partially validated; a follow-up) |
| C13 | Minor | "Mirror byte-for-byte" overstates symmetry — header parsing is structurally different | mine |
| C14 | Minor | VGPR budget unconsidered | mine |
| C15 | Minor | LDS layout hardcoded for 8-entry; MQ4-Lloyd next will want 16 | mine |
| C16 | Minor | Function-name vs module-name explicit statement missing | glm5 (validated) |
| C17 | Minor | Cooperative load thread→entry mapping not stated | glm5 (validated) |

Rejected as false alarms:

| ID | Source | Why rejected |
|----|--------|--------------|
| R1 | **glm5 B1** ("HFQ3 tail bug inherited") | Arithmetic error in glm5's analysis. `quads*4` is divisible by 4 by construction, so `g % 4 == i` for tail iter `i`. HFQ3's `acc0/1/2` for tail [0]/[1]/[2] **is** `acc[g % 4]`. Verified by case analysis. |
| R2 | gemini §3.1 (residual-variant gap) | Plan explicitly defers residual until a use site appears. Lloyd-MQ3 is research-gated behind `--allow-mq3-lloyd`; no production caller currently uses a Lloyd-MQ3 residual. The gap is acknowledged, not an oversight. |
| R3 | glm5 M1 (use coherence-gate-dflash) | Lloyd-MQ3 is a plain GEMV change, not a spec-decode change. `coherence-gate-dflash.sh` is for DDTree/spec-decode token-attractor regressions. Plain `coherence-gate.sh` is the right gate for this kernel. |
| R4 | glm5 L4 (18.52 ppl alarming) | Format is `--allow-mq3-lloyd`-gated, research-only. The gate is "don't regress vs PR's 18.52", not "is 18.52 production-shippable". |

The rest of this doc explains each item.

---

## C1 (Blocker, mine) — bench harness doesn't support Lloyd-MQ3

Validation step 4: `scripts/probe_commits.sh <baseline> HEAD` on
9B Lloyd-MQ3 decode.

`scripts/probe_commits.sh` is hardcoded to `bench_qwen35_mq4`:

```bash
target/release/examples/bench_qwen35_mq4 "$HOME/.hipfire/models/qwen3.5-9b.mq4"
```

There is no `bench_qwen35_mq3_lloyd.rs` or equivalent in
`crates/hipfire-runtime/examples/`. The PR's "44 tok/s" came from the
*perplexity harness*, which is a single-window NLL pass — not a
steady-state decode tok/s harness.

As written, step 4 silently benches MQ4 (unchanged path) and reports
no progress. Plan must **either** add a `bench_qwen35_mq3_lloyd.rs`
example mirroring `bench_qwen35_mq4.rs` and teach `probe_commits.sh`
to take a model+example pair, **or** confirm that
`bench_qwen35_mq4.rs` auto-detects the dtype from the loaded weights
(I didn't read it deeply enough; the plan should).

Per CLAUDE.md prompt-md5 rule, the bench prompt should be a committed
file (not a heredoc), with md5 logged alongside results.

---

## C2 (Blocker, glm5 — validated) — graph-capture safety

glm5 caught this and they're right. `dispatch.rs:2073`:

```rust
unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
```

The HFQ3 path (`dispatch.rs:2626`) uses `launch_maybe_blob`, which is
the graph-capture-safe pattern (kernargs go to a `KernargBlob`, not a
stack-resident `Vec<*mut c_void>`). CLAUDE.md is explicit: under
`HIPFIRE_GRAPH=1`, captured stack-pointer kernargs dangle past
`end_graph_capture` → silent corruption.

`probe_commits.sh` exports `HIPFIRE_GRAPH=1`. If the new perf gate
runs the Lloyd path under graph mode, the existing dispatch is
unsafe. Even ignoring the graph case, fixing this is in-scope because
the plan touches the function in question (step 4: replacing
`ensure_kernel`). Add an explicit step:

> **Step 4b.** Migrate `gemv_mq3g256_lloyd` from
> `self.hip.launch_kernel` to `self.launch_maybe_blob` with a
> `KernargBlob` closure, mirroring the pattern at
> `dispatch.rs:2626-2640`. Apply same to `gemv_mq2g256_lloyd` for
> consistency with the existing Lloyd surface.

Note: many other dispatch arms (e.g. `dispatch.rs:1561`, `1581`) are
also raw — this isn't unique to Lloyd, and there's no claim that
those are graph-mode targets. Limit C2's scope to the Lloyd functions
touched by this PR.

---

## C3 (Major, mine) — bottleneck attribution is folklore

Plan: "compiler emits ~14 instructions per lookup × 8 lookups ≈ 112
inst of pure dispatch overhead per group."

No disassembly cited. Findings doc says "**Likely** recoverable with
the same K4 unrolling pattern" — note "likely", and **K4 only**, no
LDS. The 3× regression could be primarily single-accumulator dep-chain
stall, in which case K4 alone is enough and the LDS staging is
unjustified surface area.

Recommend:

1. Land K4-only first as commit A. Measure decode tok/s.
2. If still under target, layer LDS as commit B. Measure again.
3. Bisect-friendly history; cleaner perf accounting.

Or at minimum: dump `llvm-objdump -d` for the existing kernel's
inner switch and confirm 14×8 cmp/cndmask. AMD's compiler is allowed
to emit `v_perm_b32` / vector LUT / register-file indirection on
RDNA3; if it does, the plan's instruction count is off by ~10×.

---

## C4 (Major, mine) — 0.8B Lloyd-MQ3 is the wrong correctness anchor

Plan step 2: "expect ppl ≈ 155.22 ±0.5". That number is
text-collapsed already (a healthy 0.8B sits ~25-30; the findings
doc's own table shows `0.8B Lloyd factor 1.94×` with absolute ppl 155
which is well past coherence). A subtle reconstruction bug could
shift ppl by 5-10 points and either fail by drift or pass by luck on
an already-noise-dominated distribution.

Replace with: **4B Lloyd-MQ3 ppl = 22.56 ±0.3** as the primary
numerical anchor. 4B is the smallest size where the ppl number is a
real quality signal. Pair with the logits-Δ check (C7).

---

## C5 (Major) — bank-conflict arithmetic is wrong

I and glm5 both flagged this; we agree on the correct numbers.

Plan claims:

> At fp16 (2 B), 8 entries map to 2 banks → up to 4-way bank conflict

RDNA LDS has **32 banks × 4 B/bank**. 8 fp16 entries = 16 B → **4
banks** (entries `(0,1)→bank0, (2,3)→bank1, (4,5)→bank2, (6,7)→bank3`).
Worst-case 32 threads → 8 threads/bank → **8-way conflict, not
4-way**.

At fp32: 8 entries × 4 B = 32 B → 8 unique banks → up to 4-way
conflict per bank. **Both numbers in the plan are off by 2×.**

Conclusion (use fp32) is correct — fp32 is strictly better, and
128 B is trivial on 64 KB LDS. Either fix the numbers or drop them
and keep the conclusion.

---

## C6 (Major, mine) — cooperative load isn't a single coalesced txn

Plan: "32 threads × 1 fp16 each = exactly the 32 codebook entries
for the 4 groups."

Codebooks are at `gp0`, `gp0 + 112`, `gp0 + 224`, `gp0 + 336`.
Threads `0..7` read from `gp0`, `8..15` from `gp1`, etc. That's
**4 distinct ~16 B reads from 4 cachelines @ 112 B stride**, not a
coalesced 64 B transaction. RDNA3 will service this fine (4 cachelines
@ 64 B is small), but the bandwidth-savings argument vs. "each
thread loads its own 8 codebook entries from a single cached header"
is weaker than implied:

- **Existing**: 32 threads × 8 fp16 reads from same 16 B header — L1
  serves it with effective bandwidth ≈ 16 B/group.
- **Proposed**: 4 cachelines once per quad → ~16 B/group amortized.
  Comparable.

The structural win is **replacing the cmp/cndmask tree with
`ds_read_b32` indexed by `q`**, not bandwidth. Frame the plan that
way.

---

## C7 (Major) — no logits-Δ check (mine + gemini §5.1, validated)

ppl-vs-table catches gross correctness loss. It does NOT catch subtle
reconstruction bugs that shift logits by < 1% — invisible at
perplexity-aggregate level but visible as token-attractor / drift on
long generations. Per CLAUDE.md `feedback_attention_precision.md`
note, 5% attention error cascades into attractor within ~10 tokens
under greedy decode.

Add: run a single short-prompt forward pass through old vs new
kernel, diff the logits, log max-abs Δ. Cheap, rigorous, catches
the bug class ppl can't see.

Note gemini's specific recommendation
(`scripts/gfx906_logit_divergence.sh`) — I didn't verify whether that
script exists or whether a gfx1100 equivalent exists. The
*requirement* is right; the specific script reference may be wrong.

---

## C8 (Medium, gemini §2.1 — validated, but indirect)

gemini: tail K4 cooperative load reads codebooks for non-existent
groups → potential OOB.

Strict-OOB analysis: if the tail loop reuses the K4 cooperative
load (which fetches gp0..gp3 in parallel) when only 1-3 groups are
present, gp1/gp2/gp3 may point past the row's allocated bytes. That's
a real OOB.

Mitigated-OOB analysis: if the tail loop runs with single-group
cooperative loads (all 32 threads stage one 8-entry codebook from a
known-valid `gp`), threads 8..31 over-read into the codebook block's
own packed-index region — same group, in-bounds, garbage data
unwritten because LDS slots 8..31 are unused.

The plan says "Tail iterations: same LDS-staged path, one group at a
time" — implies the second pattern, which is OOB-safe but
underspecified. Make it explicit: tail iterations use a per-group
load (e.g., only first 8 threads load; barrier; index `cb_lds[q]`).

---

## C9 (Medium, mine) — coherence-gate Lloyd guard

The Lloyd-MQ3 quantizer is gated behind `HIPFIRE_ALLOW_MQ3_LLOYD=1`
(`crates/hipfire-quantize/src/main.rs:2011-2013`). I did **not**
verify whether the runtime / coherence-gate also gates loading a
`.mq3-lloyd` model. If the daemon refuses to load without the env,
coherence-gate runs against a non-Lloyd model and proves nothing
about the new kernel.

Plan should specify the env explicitly:

```
HIPFIRE_ALLOW_MQ3_LLOYD=1 HIPFIRE_MODEL=...mq3-lloyd ./scripts/coherence-gate.sh
```

(verify the actual model-selection mechanism first). If the
pre-commit hook doesn't inherit the env, it needs to.

---

## C10 (Medium, glm5 M2 — validated)

Plan line 100: "Pre-commit hook runs it automatically on kernel
change."

glm5 is right that this can't carry validation ordering. `cargo
check` (step 1) doesn't run the hook — only `git commit` does. The
hook is convenience, not a gate between steps 2 and 4. Run
`coherence-gate.sh` as an explicit step in the validation order. The
hook is fine as a backup catch on the `git commit` itself.

---

## C11 (Medium, gemini §5.2 — validated; location wrong)

gemini: "ppl tests on 9B/4B might not hit all tail cases (groups % 4
== 1, 2, 3)." Correct concern — Qwen 3.5+ projection K values may
land on multiples of 1024 (= 4-quad clean), in which case the tail
path never executes during normal inference and the bug-class glm5
(falsely) raised in B1 would only show up on a future model with a
different K.

Add a small standalone test that calls the kernel directly with
`groups_per_row ∈ {4, 5, 6, 7, 8}` (clean + each tail size + clean+1
quad). Compare against the existing kernel for bit-equivalence at
fp32 reconstruction.

Reject gemini's specific path suggestion: `cli/chat_pure.test.ts` —
this is a Rust HIP project, no `cli/` test directory, no TypeScript.
Likely a path hallucinated from another codebase. Put the test in
`crates/rdna-compute/tests/` or alongside the existing kernel
sweep tests.

---

## C12 (Minor, glm5 H1 — partially validated)

glm5: HFQ3 selector matches only `gfx1100|1101|1102`, missing
`gfx115x` (RDNA 3.5) and `gfx12xx` (RDNA 4) which AGENTS.md
allegedly lists as production for MQ3.

Verified narrowness: `kernels.rs:451-458` is exactly that match. I
did **not** verify the AGENTS.md claim about gfx115x/12xx production
status — glm5 should be checked on that part. Either way:

- Mirroring HFQ3's narrow match is consistent (defensible).
- Widening here without widening HFQ3 is inconsistent (worse).
- Fixing both in a separate commit is better.

Treat as follow-up, not blocking on this PR.

---

## C13 (Minor, mine) — "byte-for-byte" overstates symmetry

Plan: "Mirror HFQ3 file lines 76-120 byte-for-byte modulo lookup
substitution."

HFQ3 parses **8 B header (fp32 scale + fp32 zero)** and uses
`sc * q + zp`. Lloyd parses **16 B header (8 × fp16 codebook)** and
uses `cb[q]`. The control flow (4 quads + 3 tail) ports verbatim;
header parsing and reconstruction body are different in size and
shape. Specifically: HFQ3 data starts at gptr+8, Lloyd data starts at
gptr+16. A "byte-for-byte" copy will silently load 8 B from the wrong
offset.

Re-word as: "Mirror the K4 + tail control-flow structure from
HFQ3:76-120; substitute the 16 B fp16 codebook header parse + LDS
cooperative stage + `cb_lds[q]` lookup for the 8 B (scale, zero) parse
+ affine recon."

---

## C14 (Minor, mine) — VGPR budget unconsidered

Adding 4 accumulators + 4 packed indices + 4 group pointers + LDS
staging temps pushes VGPR count vs the current single-acc kernel.
`__launch_bounds__(32, 16)` constrains VGPR/wave to roughly the HFQ3
budget. HFQ3 ships with this same bound and works; **probably** fine,
but a register spill regresses instead of helping.

Recommend: preserve `__launch_bounds__(32, 16)` explicitly, dump
`hipcc --save-temps` and check `.amdgcn_target` VGPR count ≤ HFQ3
count. One-line check.

---

## C15 (Minor, mine) — LDS layout hardcoded; MQ4-Lloyd needs 16-entry

`devlog_20260506_lloyd_mq4_extension.md` (the doc this PR slots into)
targets MQ4-Lloyd next: 16 fp16 centroids per group. Plan's
`__shared__ float cb_lds[8 * 4] = 128 B` is hardcoded for 8-entry.
MQ4 will want `[16 * 4] = 256 B` and a different lookup index.

Spend 30 minutes parameterizing now (`#define CB_ENTRIES 8` + index
math) so the MQ4-Lloyd PR can drop in a new kernel header without
rewriting the LDS path. Nice-to-have, not blocking.

---

## C16 (Minor, glm5 L3 — validated)

Plan proposes module name `gemv_mq3g256_lloyd_rdna3` but doesn't
state the kernel function name. Verified pattern at
`gemv_hfq3g256.gfx1100.hip:13`: the kernel function is
`gemv_hfq3g256` (unsuffixed), module is `gemv_hfq3g256_rdna3`.

So for Lloyd: kernel function stays `gemv_mq3g256_lloyd`, module is
`gemv_mq3g256_lloyd_rdna3`. The `ensure_kernel` call expects this
distinction. State it explicitly in plan §4 step 3.

---

## C17 (Minor, glm5 L2 — validated)

Plan: "32 threads × 1 fp16 each = 32 codebook entries for 4 groups."
Math is right but mapping is unspec'd. Make explicit:

```
thread tid loads cb_h[tid % 8] from group pointer gp[tid / 8],
writes to cb_lds[tid] as fp32 (after __half2float).
```

Without this, a "all 32 threads load from gp0" bug is a one-typo
distance away.

---

## Other gemini items (validated but lower priority)

- **gemini §2.2 (fp16→fp32 conversion latency)**: real cost but
  trivial — `v_cvt_f32_f16` is single-cycle on RDNA3, 32 lanes do it
  in 1 cycle. Pipelining suggestion is fine but the magnitude is lost
  in cache-miss noise. Validated as low-priority.
- **gemini §2.3 (LDS broadcast vs conflict)**: the math is wrong (see
  C5) but the structural point — fp32 LDS is "limited conflict",
  4-way worst case, far better than 112-cycle ternaries — is right.
- **gemini §3.2 (gfx1101/1102 verification)**: target is gfx1100;
  1101/1102 share ISA; ship gate is "≥120 tok/s on gfx1100". Worth a
  callout in the plan that "the gate is gfx1100; 1101/1102 are
  expected to inherit but unverified." Not blocking.
- **gemini §4.1 (`__syncthreads()` overkill)**: plan acknowledges.
  `__builtin_amdgcn_s_waitcnt(0)` may save a cycle. Diminishing
  returns; both work.
- **gemini §4.2 (codebook prefetching)**: real future-work optimization
  — prefetch next quad's codebook during current quad's FMAs. Not in
  scope for this PR.

## Other glm5 items

- **glm5 L1 (`__syncthreads()` is a CUDA-ism)**: HIP supports both
  `__syncthreads()` and `barrier()`; both compile to the same ISA on
  single-wave. Comment in the new kernel for clarity. Trivial nit.

## Rejected glm5 items (detailed)

### glm5 B1 — HFQ3 tail bug inherited

glm5 claims: "When `groups_per_row = 5` (quads=1, tail=1), the tail
group should go into `acc[1 % 4] = acc1`, not `acc0`."

This is arithmetically wrong. Walk it through:

- `groups_per_row = 5` → `quads = 5 >> 2 = 1`, `tail = 5 & 3 = 1`
- The full quad processes groups 0-3 (acc0-acc3) ✓
- Tail iter `i=0` processes group `g = (quads << 2) + 0 = 4`
- HFQ3 puts it in `acc0`. Check: `g % 4 = 4 % 4 = 0` → `acc0` ✓

glm5 confused `tail` (the count, =1) with the group index `g` (=4).
The "acc[g % 4]" comment matches the actual code under the structural
invariant that `quads * 4` is divisible by 4 by construction.

Cross-check with `groups_per_row = 7` (quads=1, tail=3):
- Tail i=0: g=4, acc0, 4%4=0 ✓
- Tail i=1: g=5, acc1, 5%4=1 ✓
- Tail i=2: g=6, acc2, 6%4=2 ✓

HFQ3 tail logic is correct. **B1 rejected.** Plan's "mirror byte-
for-byte modulo lookup substitution" inherits a working pattern.

### glm5 M1 — wrong gate (use coherence-gate-dflash)

Lloyd-MQ3 is a plain GEMV kernel change, not a DDTree/spec-decode
structural change. Per CLAUDE.md, `coherence-gate-dflash.sh` is for
"any DDTree / spec-decode / slow-path-kill change that claims a τ or
tok/s improvement." This PR claims a decode tok/s improvement on a
GEMV kernel — different layer. `coherence-gate.sh` (the 4-prompt
battery) is the correct gate. M1 rejected.

(Caveat: if the bench example chosen for step 4 happens to use a
DFlash spec-decode harness internally, then the dflash gate **does**
apply. Worth verifying when picking the bench example for C1, but
not a default requirement.)

### glm5 L4 — 18.52 ppl quality gap not assessed

The PR is research-gated behind `--allow-mq3-lloyd` /
`HIPFIRE_ALLOW_MQ3_LLOYD=1`. The quality gate is "don't regress vs.
PR's published 18.52 (within ppl noise)", not "is 18.52 production-
shippable" — those are different questions, and the second is
out of scope here. L4 rejected.

### gemini §3.1 — residual-variant gap

Plan explicitly defers (`§What this plan does NOT do`, line 132):
"No residual variant for Lloyd-MQ3 (PR doesn't ship one; defer until
a use site appears)." The format is research-gated; no Lloyd-MQ3
residual call site exists. Adding a residual variant before there's
a caller is speculative scope expansion, exactly what the plan is
supposed to avoid. Reject as written. (If a Lloyd-MQ3 residual call
site lands later, the residual GFX1100 kernel becomes its own PR.)

---

## Things the plan got right

- Splitting GFX1100 into a separate file alongside the generic
  baseline (parallels HFQ3 layout).
- Module name `gemv_mq3g256_lloyd_rdna3` (matches HFQ3 selector
  convention at `kernels.rs:454`) — though function name should be
  stated explicitly (C16).
- Keeping MQ2-Lloyd out of scope (bit-width-bound at 2 bpw).
- Keeping the `--allow-mq3-lloyd` guard removal as a separate commit.
- Single-wave block + `__syncthreads()` for readability — correct
  call; cost is a no-op `s_barrier` on wave32.
- fp32 in LDS over fp16 — right call (despite the bank-conflict
  arithmetic glitch).
- Calling out the tail-accumulator trap up front (even though the
  `g % 4` formulation is structurally equivalent to HFQ3's
  `acc0/1/2`-by-tail-index — a future implementer reading either
  formulation will write correct code).
- Acknowledging the `active_stream` gotcha is irrelevant here — saves
  the reader from chasing a red herring from CLAUDE.md.

---

## Suggested concrete plan amendments (consolidated)

1. **Step 0 (new, before existing step 1):** Verify or extend the
   bench harness — `bench_qwen35_mq3_lloyd.rs`, taught into
   `probe_commits.sh`, with a committed prompt file (md5 logged
   alongside results). [C1]
2. **Step 4b (new):** Migrate `gemv_mq3g256_lloyd` (and
   `gemv_mq2g256_lloyd` for consistency) from
   `self.hip.launch_kernel` to `self.launch_maybe_blob` with a
   `KernargBlob` closure. [C2]
3. **Replace step 2 anchor:** "0.8B ppl ≈ 155.22 ±0.5" → "4B ppl ≈
   22.56 ±0.3", and add a logits-Δ check vs. the existing kernel on
   a single short-prompt forward pass. [C4, C7]
4. **Step 2.5 (new):** dump `--save-temps` for the new kernel; confirm
   VGPR count ≤ HFQ3's. [C14]
5. **Step 2.6 (new):** call the kernel directly with
   `groups_per_row ∈ {4, 5, 6, 7, 8}` against the existing kernel for
   bit-equivalent reconstruction in fp32. [C11]
6. **Split into two commits:** K4-only first, LDS layered second, so
   perf attribution is bisectable. (Or document why bundling is
   preferred — but split should be the default.) [C3]
7. **Step 3 explicit env:** spell out `HIPFIRE_ALLOW_MQ3_LLOYD=1`
   (and the model selection) for `coherence-gate.sh`. [C9]
8. **Drop the pre-commit-hook claim** as part of validation ordering;
   keep it as a backup catch on commit. [C10]
9. **Fix the bank-conflict arithmetic** (or drop the specific
   numbers). [C5]
10. **Rephrase the cooperative-load coalescing claim** to match how
    the wave actually issues the load (4 cachelines @ 112 B stride,
    not a coalesced 64 B txn). [C6]
11. **Briefly disassemble the existing kernel's switch tree** before
    committing to "112 inst per group" attribution. [C3]
12. **State the kernel function name explicitly** in §4 step 3
    (`gemv_mq3g256_lloyd`, unchanged) and the cooperative-load
    thread→entry mapping. [C16, C17]
13. **Re-word "byte-for-byte modulo lookup substitution"** to make
    the structural header difference (8 B affine vs 16 B codebook)
    visible to a future implementer. [C13]
14. **Note in the plan** that `__shared__ float cb_lds[8 * 4]` is
    8-entry-specific; MQ4-Lloyd extension will need re-parameterization.
    [C15]

None of these change the *direction*. They make the diff land cleanly
and the perf claims defensible after the fact.
