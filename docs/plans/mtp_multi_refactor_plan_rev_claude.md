# Adversarial review: docs/plans/mtp_multi_refactor.md

Reviewer's stance: I wrote the plan being reviewed. This pass is
intentionally hostile — looking for what's wrong, optimistic, missing,
or unsound, not what's good. I checked the actual code rather than
trusting the plan's claims.

**Bottom line up front:** the plan is structurally reasonable but
contains several material misstatements about today's code, glosses
over a real semantic gap in `forward_prefill_batch_multi`, and
underbudgets validation by ~5×. Do not start it without the changes
below.

This document is now **rev2**: incorporates findings from external
adversarial reviews (`mtp_multi_refactor_plan_rev_glm5.md`,
`mtp_multi_refactor_plan_rev_gemini.md`) — see §A at the bottom
for the verdict on each external claim.

User scope clarification (2026-05-28):
**DFlash must keep working after the refactor but needs no specific
multi-gpu handling — PP-DFlash was already proven a loss in prior
experiments. So `Path::Dflash` in the unified function is delegate-
to-existing-`generate_dflash`, not an inline implementation. This
removes ~491 LOC from the refactor scope.**

---

## P0 findings (would cause the refactor to fail or mislead)

### P0-1: LOC estimates are wrong, and one whole function is missing from scope

Plan's table (lines 34-39):
| function | claimed LOC | actual LOC |
| --- | ---: | ---: |
| `generate` | ~600 | **991** |
| `generate_multi` | ~486 | 485 |
| `generate_mtp` | ~500 | 454 |
| (claimed total) | ~1586 | **actual: 1930** |

`generate_dflash` (491 LOC) is not in the plan's table at all, yet the
proposed `SpecPath` enum lists `Dflash` as one of its arms.

**Rev2 update (user input):** DFlash stays as a separate function
because PP-DFlash is a known perf loss and the dispatch arm just
delegates to today's `generate_dflash`. Sub-scope removed: 491 LOC
no longer in scope, but the dispatch enum still needs a
`Path::DFlashSingle` arm that calls `generate_dflash` and returns —
not an inline body. **Plan must say this explicitly.**

**Fix:** redo the LOC accounting against the actual file. Decide
explicitly that `generate_dflash` stays as a delegate, NOT inlined.

### P0-2: `generate` is already a top-level dispatcher, not a sibling to the other two

The plan repeatedly says "unify generate / generate_multi / generate_mtp
into one function." But `generate` (line 4622) is already the
top-level entry point that:
- routes to `generate_qwen2` when `arch_id < 5`
- routes to `generate_multi` when `m.pp > 1`
- routes to `generate_dflash` when `m.dflash.is_some()`
- routes to `generate_mtp` when `m.mtp.is_some()`
- falls through to its own AR implementation

So we are not unifying three peers; we are inlining three children
back into their parent. That's still a valid refactor, but the
framing in the plan misrepresents the topology and therefore the
risk: the `generate` function body is BOTH a dispatcher AND a
fifth path implementation. Pulling that apart is messier than
pulling apart three already-leaf functions.

**Fix:** rewrite the plan's "Current state" section to reflect that
`generate` is a dispatcher with an inline AR implementation. Step 4
("Path enum + dispatcher") is actually mostly already done — what's
new is extracting `generate`'s inline AR body into its own match arm
before the unification can proceed.

### P0-3: `forward_prefill_batch_multi` does not have the params MTP needs

The `_multi` and single signatures genuinely diverge:

```
forward_prefill_batch(gpu, weights, config, tokens, start_pos, kv,
    dn, scratch, hidden_rb, per_token_hidden_out, gdn_tape, tree_verify)
forward_prefill_batch_multi(gpus, weights, config, tokens, start_pos,
    kv, dn, scratch_set)
                          ^ NO hidden_rb, per_token_hidden_out, gdn_tape, tree_verify
```

The MTP serve path uses `per_token_hidden_out` semantics (via
`capture_prev_hidden_from_scratch_tmp` after prefill at
daemon.rs:3904) AND `gdn_tape` (mtp_spec's rollback path). The plan
says step 3 introduces a `PrefillContext` enum at ~150 LOC and that
"both single-gpu and PP paths now use the same call site." That is
false at the API level: the multi entry point does not accept the
parameters MTP needs. You cannot wrap them behind a method.

To make MTP-under-PP actually work, `forward_prefill_batch_multi`
itself must be extended to accept `per_token_hidden_out` (which is a
GpuTensor on which device?) and `gdn_tape` (which references trunk
DN state that is now spread across bands). Neither extension is
trivial:
- `per_token_hidden_out` lives on `output_device` in the multi path
  because that's where post-output-norm runs. The MTP seed in
  `capture_prev_hidden_from_scratch_tmp` needs to read from
  `scratch_set.per_device[output_device].tmp`, not the single-gpu
  `scratch.tmp`. The plan does not mention this.
- `gdn_tape` records LinearAttention innovations per layer; layers
  are spread across devices, so the tape needs per-band capture +
  cross-device assembly OR the rollback path needs to know which
  device owns each tape slot.

**Fix:** the plan needs a new "Step 0: extend
forward_prefill_batch_multi" item before any of the unification work.
Estimated size depends on `gdn_tape` semantics under PP — could be
20 LOC (just pipe per_token_hidden_out through, leave tape as None
in multi mode and force MTP-PP to use the tape-less replay path) or
200+ LOC (full per-band tape capture). Best case is the simpler
"force tape-less replay" option, which costs a small amount of
acceptance-rate loss; we should bench that vs. full tape capture
before committing.

### P0-4: `gpu` passed to `generate` is a SEPARATE handle from `m.pp_gpus.devices[0]`

The plan's proposed signature is:
```
fn generate_qwen35(
    m: &mut LoadedModel,
    gpu: &mut Gpu,                              // dev 0 for both pp=1 and pp>1
    drafter_gpu: Option<&mut Gpu>,
    ...
)
```

The comment "dev 0 for both pp=1 and pp>1" is misleading. In the
actual daemon, the main `gpu` is created once at daemon startup
(daemon.rs:527, 555) via `Gpu::init()`. When pp>1, `load_model_pp`
creates a SEPARATE `Gpus { devices: Vec<Gpu>, ... }` via
`construct_devices` → `Gpu::init_with_device`. So in pp=2 we have
**three Gpu instances on two devices**:

  - main `gpu` (constructed by daemon) → device 0
  - `m.pp_gpus.devices[0]` → ALSO device 0
  - `m.pp_gpus.devices[1]` → device 1

The two device-0 handles do not share kernel-module caches, stream
pools, or `LAST_BOUND_DEVICE` state across the boundary. The plan
glosses over this entirely. In hetero MTP we already saw stream-
affinity bugs from less than this; here it is structural.

Worse: the plan's `gpu: &mut Gpu` arg is the main-loop handle, but
the prefill_multi path uses `m.pp_gpus.devices[0]`. Inside the
unified function, taking `&mut` on both would require interior-
mutability gymnastics or careful temporal access patterns.

**Fix:** the plan needs to explicitly decide:
  (a) When pp>1, ignore the `gpu` parameter inside the unified
      function and route all work through `m.pp_gpus`. The main
      `gpu` becomes a parked handle for non-generate utility calls
      (daemon stats, etc.).
  (b) Have `LoadedModel` store the main `gpu` too, making it
      consistently reachable. (Major refactor of struct ownership.)

Option (a) is simpler but means the function signature lies — the
`gpu` arg is unused in some paths. Document it. Option (b) is
cleaner long-term but explodes the refactor scope further.

### P0-5 (NEW, from external review + Stage 2a evidence): same-device borrow conflict is a real blocker

Both external reviews (gemini §1 "Borrow Checker Fragility", glm5
§7 "Borrow-checker blocker with dual `&mut Gpu`") identify this. I
initially missed it because I was thinking about the runtime
behavior; they thought about the type signature.

**Evidence from Stage 2a code I wrote yesterday:** in
`daemon.rs:2677-2681` I had to write `unsafe { &mut *dev0_ptr }` to
satisfy peer_clone_tensor's `(&Gpu, &mut Gpu)` signature when both
were the same device. The reviewer-flagged borrow issue is **not
hypothetical — it has already cost us an unsafe block** in shipped
code. Propagating that pattern through the unified function would
metastasize.

**Fix:** the unified function design must NOT use `(gpu: &mut Gpu,
drafter_gpu: Option<&mut Gpu>)`. Two options:
  (a) Pass a device index pair `(target_dev: usize, drafter_dev:
      usize)` plus `&mut Gpus` (the same struct already used for PP).
      All access goes through `gpus.devices[i]`. The function body
      uses `split_at_mut` when it needs two distinct device handles.
      Same-device case becomes `if target_dev == drafter_dev` and
      uses a single `&mut` to that one device.
  (b) Refactor `Gpus` to expose `disjoint_pair_mut(i, j)` and
      `single_mut(i)` that handle the same-device shortcut internally,
      so callers don't write split_at_mut themselves.

Option (b) is cleaner and matches the same-device shortcut we
already encoded in `peer_clone_tensor` and `seed_prev_hidden`. Add
this as a prerequisite refactor (Step −1: clean Gpus access pattern).

---

## P1 findings (would cause real bugs or scope creep)

### P1-1: Eviction story is unsound — `generate_mtp` ALREADY has defensive eviction code

The plan dismisses eviction with "disable at load, same as today." But
today's `generate_mtp` (line 3668+) contains defensive per-cycle
eviction handling:

```rust
// daemon.rs:374-385 (inside generate_mtp's decode loop)
// Per-cycle eviction. MTP + cask/eviction is refused at load (the
// eviction position accounting), so this is defensive: never `.unwrap()`
if let Some(ref ev) = m.eviction {
    match ev.maybe_evict(gpu, &mut target.kv_cache, position) {
        Ok(...) => {},
        Err(e) => eprintln!("[hipfire-daemon] mtp eviction error (ignored): {e}"),
        // also: "WARNING: eviction fired under MTP — position accounting
        // not reconciled; output may degrade (MTP+cask should be refused
        // at load)"
    }
}
```

So today's MTP path does NOT refuse eviction at load — it warns if
eviction fires. The plan's "disable at load, same as today" is just
wrong. After the unification, the inherited code will fire on every
PP-MTP cycle since eviction-at-load is not refused, and we will start
seeing the warning on every long-context conversation.

**glm5 partially disagreed** (§14 "Eviction open question is
backward"): claims eviction is refused at load for pp>1 in
daemon.rs:773-777. **Verdict: glm5 is right that PP+eviction is
refused (line 774 errors on CASK + pp>1), but I was talking about
MTP+eviction at pp=1, which is NOT refused.** Both points stand:
the MTP path has defensive eviction code that the plan should NOT
inherit silently. Combined PP+MTP gets eviction-disabled for free
via the pp>1 refusal — but single-gpu MTP+eviction still has the
defensive-warn bug today. The unified function inherits it.

**Fix:** either actually refuse `mtp + eviction` at load for ALL pp,
or carry the defensive eviction handling through into the unified
function with a clear comment. The plan needs to pick.

### P1-2: "Enable PFlash + MTP silently" is too cavalier given known DFlash+PFlash refusal

The plan's open questions section says of PFlash+MTP: "Suggest:
enable the combo silently, add a coherence test in the validation
step." But today's code (line 4681-4685) explicitly bypasses PFlash
on DFlash with a `pflash_bypass` event:
```
"pflash_bypass":"dflash_decode_active (pflash compression on the
DFlash path is a follow-up; set dflash_mode=off to compress with
AR decode)"
```

That documented refusal exists because the PFlash+DFlash combo has
not been validated. The PFlash+MTP combo has the SAME risk profile:
PFlash compresses the prompt (changes what tokens enter prefill),
MTP runs a speculative chain on top. They have not been jointly
tested. "Enable silently" is overconfident.

**Gemini agrees strongly** (§2 "PFlash composition"): "PFlash changes
the KV cache layout/indexing for the prompt. MTP relies on specific
KV offsets for speculative verification. If they don't agree on the
'ground truth' KV state, we'll get silent coherence failures that
are extremely hard to debug." This adds a concrete mechanism
(KV-offset mismatch) my review didn't articulate.

**Fix:** add an explicit pflash_bypass for MTP, mirroring the
DFlash bypass. Wire the integration test as a separate follow-up
once the basic PP+MTP combo is shipped. The plan should not roll
a second unvalidated combo into the same delivery.

### P1-3: Validation budget is 5× under

The plan repeatedly says "validated by coherence-gate" between each
step, and lists steps 2-6. Looking at `scripts/coherence-gate.sh`:
- 9 model tests in the matrix
- Each test = 27B-class load (~45s) + generation up to 800 tokens
  (~20s) = ~65s
- Battery wall: ~10 minutes
- Plus `coherence-gate-dflash.sh` (~5 min more on the dflash-relevant
  paths)
- Plus a not-yet-written PP-AR + PP-MTP coherence test the plan
  hand-waves into existence

Steps 2-6 with intermediate commits: realistically 12-15 commits.
At 15+ minutes per validation: **3+ hours of pure gate runs**, not
counting the time to read the markdown reports. That is BEFORE the
debugging time when something breaks.

**glm5 §10 raised a related point** ("coherence-gate.sh is
deprecated"). **VERDICT: glm5 is partially wrong.** AGENTS.md
line 166-170 says **`quality-gate.sh` is deprecated**, NOT
`coherence-gate.sh`. The dflash variant is "the canonical correctness
gate" per AGENTS.md but the non-dflash `coherence-gate.sh` is treated
as authoritative in CLAUDE.md:225 and `.githooks/pre-commit`. glm5
misread the deprecation. **However glm5's underlying suggestion is
sound — we should rely on `coherence-gate-dflash.sh` as the canonical
spec-decode gate since this whole refactor IS about a spec-decode
path.**

**Fix:** budget validation time honestly. Pick which gate runs are
mandatory between steps and which can defer to a single big run at
the end of step 5. My recommendation: gate after step 2 (helpers
extraction), full gate after step 5 (the surgery), single-run check
after each of 3, 4, 6. That cuts gate wall to ~45 min total. Use
`coherence-gate-dflash.sh` as the primary gate (per spec-decode
relevance), and run the non-dflash gate as a final smoke before merge.

Also: BUILD the PP-AR coherence test (step 0) before claiming
"validated against PP-AR" at every step. It does not exist today.

### P1-4: Step 5 line counts are inconsistent with the rest of the plan

Step 5 says "~500 LOC of moved-around code, ~200 LOC genuinely
deleted." But the plan's earlier "Total existing: ~1586 LOC ...
Estimated duplication: ~300 LOC" implies that unifying into one
function should net-delete ~300 LOC, not ~200. Where does the
delta come from? Either the duplication estimate was wrong, OR
the step 5 estimate undercounts the genuine deletion, OR the
PrefillContext/SpecPath plumbing (steps 3-4) adds back ~100 LOC
that has to be subtracted. The plan does not say.

**glm5 §9 goes further** ("Common prelude/postlude: ~300 LOC removed
is overstated"): claims realistic savings are ~100-120 LOC, not 300,
because each function's prelude/postlude has path-specific divergence
(generate_mtp has no budget_alert; generate_dflash has PFlash bypass
in prelude; generate_multi has PFlash compression in prelude;
postlude events differ per path: `mtp_tau` vs `dflash_stats` vs
`pp_progress`). **VERDICT: glm5's analysis is more detailed and
plausibly closer to truth.** The savings are ~half what the plan
claimed.

**Fix:** redo the LOC accounting end-to-end with the actual file
sizes. The 1930-LOC reality (P0-1) plus an honest duplication audit
will give a real before/after. Expect the duplication-removed number
to drop from 300 to ~120 LOC. That weakens the "this is worth doing"
argument — needs honest reassessment.

### P1-5 (NEW from external): parameter explosion needs a context struct

**glm5 §4** ("Parameter explosion — 18+ params, no context struct
proposed") and **gemini §1** ("State Explosion") both flag this.

Param counts confirmed against actual code:
- `generate`: 21 params
- `generate_multi`: 19 params
- `generate_mtp`: 14 params
- `generate_dflash`: 13 params

Union: 23+ params. **My original review missed this entirely.**
The plan's signature sketch with `gpu, drafter_gpu, stdout, id,
prompt, system_prompt, temp, top_p, max_tokens, repeat_penalty,
repeat_window, budget_alert_at_tok, budget_alert_text,
max_think_tokens, assistant_prefix, pflash_state, pflash_cfg,
tools, messages_history` is already at 19 params and dodging some.

**VERDICT: both reviews are correct.** A unified function with 23+
params is unreadable and easy to mis-order at call sites.

**Fix:** propose a `GenerateCtx<'a>` struct that bundles
`(stdout, id, prompt, system_prompt, sampling: SamplingCfg, ...)`
and pass it by `&mut`. Then the function signature is `(m, gpus_or_gpu,
ctx)`. This is a prerequisite design decision, not a step-5
implementation detail.

### P1-6 (NEW from external): the "one surgery commit" rollback story is fragile

**glm5 §6** ("Step 5 ... is critically underspecified") and **gemini
§2** ("The Surgery Commit") both flag that step 5 is the highest-risk
step yet gets one paragraph. Gemini calls out: "In complex Rust
codebases, 'surgery' usually means 'everything is broken for 12
hours.'"

My own §P2-3 noted that the rollback story ("git revert step 5") is
not literally possible because step 5 changes call sites. The
external reviews add that the surgery itself can't be incremental:
once `generate` body is dismantled, you're committed.

**Gemini §1 Recommendation #4** ("Split the Surgery: Instead of one
'surgery commit,' migrate one path at a time"). **VERDICT: agree
strongly.** Migrate AR first (simplest), then MtpSingle, then
MtpHetero, then PpAr, then PpMtp. Each migration is a separate
commit that swaps one call site at a time. Lets us bisect a
regression to the specific path that broke.

**Fix:** rewrite step 5 as a series of 5 commits, not one. Each
commit removes one of the four existing helper functions (`generate`'s
inline AR body, then `generate_mtp`, then `generate_multi`, then the
PFlash bypass arm) and migrates its call site to a new arm in the
unified function. After all 5 commits, the helper functions are
dead and can be deleted in a 6th commit.

---

## P2 findings (smaller, but worth flagging)

### P2-1: The "decision gate at step 3" is impossible to execute

The plan says: "After step 3 (PrefillContext): if the diff is too
noisy, abort and ship the existing functions with light cleanup. The
refactor only pays off if the unification gets to step 5 cleanly."

But step 3 is "extract PrefillContext"; the diff being noisy is not
a function of step 3 quality, it is a function of how much of the
existing function bodies use the prefill path. You won't know "is
the unification going to be clean" until step 4 or later. The gate
as written cannot fire when the plan claims it should.

**Fix:** move the gate to after step 4 (PathSpec enum is wired but
the bodies haven't merged yet) — at that point we know if the
match-on-path skeleton is plausible without yet committing to the
big surgery.

### P2-2: No mention of `Architecture` trait

The codebase already has an `Architecture` trait (line 2166 in
daemon.rs) that LLaMA uses to dispatch config/load/scratch. The
plan does not mention whether the unified `generate_qwen35` would
also be a method on a `Qwen35` arch impl, or stay as a free function.
Given the trait exists and is the documented direction for the
arch dispatch, the plan should at least say "not using it for v1
because X" rather than ignoring it.

### P2-3: No rollback story for partial commits

The plan says "each [intermediate commit] is independently
revertable without losing the refactor work." That's true ONLY if
each commit is a strict superset of behavior — but step 5 (the
surgery) explicitly changes call sites from `generate` to
`generate_qwen35`. After step 5 lands, reverting just step 5 (the
surgery) leaves the call sites pointing at a function that no longer
exists. The plan's "git revert step 5" is not literally possible.

**Fix:** name the actual rollback story: a series of pre-step-5 prep
commits that COULD be force-reverted (since each is behavior-
preserving), and a single surgery commit at step 5 that is
all-or-nothing. A failed step 5 means "revert that one commit
back to the prep-completed state, then ship option X as the
intended outcome."

**Superseded by P1-6 above** — gemini's "split the surgery"
recommendation makes this concern moot. The new sequence has 5
small commits in step 5 instead of one.

### P2-4: No mention of VL/qwen2 paths

`generate_qwen2` (line 5632) and `generate_vl*` (lines 5789, 6190)
exist as parallel generate functions for non-qwen35 archs. The plan
names the unified function `generate_qwen35` — which is honest
that it scopes to qwen35-arch — but does not say what the daemon's
top-level `generate` dispatcher looks like after the unification.
Specifically: does the dispatcher remain (calling out to
`generate_qwen2` / `generate_vl_dots_ocr` / `generate_qwen35`)?
The plan should explicitly diagram this.

**glm5 §3 raises this too** ("plan ignores generate's dual role as
dispatcher + AR impl ... doesn't discuss what happens to
generate_qwen2 (arch_id=7) or generate_vl (arch_id=8)"). Same point.

### P2-5 (NEW from external): hot-path CPU branchiness

**Gemini §4** ("Latency of the Match"): "placing a large match
statement inside the token-by-token decode loop adds CPU latency.
In high-throughput scenarios (small models, fast GPUs), this can
become a bottleneck."

**VERDICT: technically true but quantitatively negligible.** Modern
match on a 5-arm enum compiles to a single jump table; the CPU cost
is sub-microsecond. Our cycle wall is ~38 µs (MTP same-device) to
~164 ms (trunk verify). Match overhead is in the noise.

However the spirit of the concern is right: introducing branches
inside a hot loop can cause subtle perf regressions via icache
pressure or branch-prediction misses. Worth a microbench after the
refactor as a sanity check, NOT a blocker.

**Fix:** add to the validation checklist: "after step 5 (or
equivalent), single-run tok/s on AR-only (small model) must be
within 1% of pre-refactor baseline." If we see >1%, investigate.

### P2-6 (NEW from external): "Generator trait or strategy pattern"

**Gemini Recommendation #1** ("Decompose before Unifying: Instead
of one giant function, create a Generator trait or a series of
strategy-specific structs that share a BaseGenerator for the
prelude/postlude").

**VERDICT: this is a fundamentally different design proposal**,
not a tweak to the unification plan. Trait-based dispatch via
`dyn Generator` adds vtable overhead and erases the type info that
makes today's free functions inlinable. A `BaseGenerator` struct
with strategy impls is essentially a class hierarchy in Rust —
verbose and not idiomatic.

**REJECT** as the primary design, but the spirit (decompose
shared helpers) is the right intuition. The plan's current
approach (extract pure helpers in step 2, then unify in step 5)
already captures that intuition without a trait hierarchy.

### P2-7 (NEW from external): cancellation/cleanup semantics

**Gemini §5** ("Cancellation: If a user cancels a request
mid-generation, we need to ensure the KV cache state is consistent
for the next request").

**VERDICT: scope-creep concern.** Cancellation is not implemented
in any of today's generate functions — the daemon JSON protocol
doesn't support mid-request cancellation. This is a future feature
that the unification doesn't make harder. Park as a follow-up.

### P2-8 (NEW from external): refusal matrix incompleteness

**glm5 §8** ("refusal matrix in the match statement is incomplete"):
plan shows 7 match arms with 2 "Refused"; actual load-time refusals
cover 6 combos (DFlash+pp>1, CASK+pp>1, PFlash+pp>1, MTP+DFlash,
MTP+CASK, VL+pp>1, arch_id<5+pp>1).

**VERDICT: glm5 is right but the plan's framing is sloppier than
broken.** The match in the unified function only needs to handle
combos that PASS the load-time refusal. The dispatch by the time
generate is called has already filtered out refused combos. The
plan's match should be exhaustive over VALID combos, not all
combos. The plan's two "Refused" arms are noise — they shouldn't
appear in a function that's only reachable for valid combinations.

**Fix:** clarify in the plan that the match is exhaustive over
"validated combos only" (a smaller set), not over the cartesian
product of (pp, mtp, dflash, pflash). The two "Refused" arms get
deleted from the plan; if they're ever reachable that's a load-handler
bug worthy of `unreachable!()` not a match arm.

### P2-9 (NEW from external): PFlash state isn't on LoadedModel

**glm5 §11**: PFlash state lives at daemon-main-loop scope
(daemon.rs:564-574), not on LoadedModel. The plan's signature
suggests these are "model-level fields."

**VERDICT: correct factual catch, but consequentially small.** The
unified function needs PFlash state threaded as a param either way;
glm5 just clarifies it comes from daemon-locals, not from `m`. No
design change needed, just a doc fix to the plan.

### P2-10 (NEW from external): Option X is undersold

**glm5 §13** ("Option X is undersold as the fallback"): claims a
separate `generate_mtp_multi` of ~600 LOC follows established
convention (one function per combo), has lower risk than touching
2425 LOC of working code, and the plan's own "decision gates"
section concedes Option X is the safety net.

**VERDICT: partially right.** Option X IS the lower-risk option for
the immediate Stage 2b deliverable. The unification's payoff is
maintenance cost reduction over time, which only matters if we add
more combos. Today's combo count is small enough that Option X is
defensible.

**HOWEVER**: my P0-3 (forward_prefill_batch_multi gap) and P0-5
(borrow-checker via dual &mut Gpu) apply to BOTH Option X and
the unification. Option X doesn't dodge them — it still needs the
prefill_multi extension AND the Gpus access-pattern cleanup. The
unification only adds the additional cost of touching the existing
helpers.

So the real question is: do we want to pay the Option X cost
(~600 LOC new helper, ~200 LOC prerequisites) OR the unification
cost (~700 LOC moved + ~120 LOC saved, ~200 LOC prerequisites)?
Both have ~similar prerequisite work. The unification gives
slightly less code at higher near-term risk; Option X gives
more code at lower near-term risk. With user input clarifying
DFlash stays delegated, Option X looks more attractive than my
review initially concluded.

**Fix:** revise the plan's "Alternative if 2-3 days is too much"
section to give Option X equal billing as a credible first
option, not a fallback. Frame the choice honestly.

---

## What the plan got right

- The overall direction (unify into one path-dispatched function) IS
  correct in principle. The duplication is real even if the LOC
  accounting is off.
- The risk callouts about touching working serve code are honest.
- The "ship Option X as a fallback if unification fails" escape
  hatch is well-placed — provides a real exit.
- The decision-gate idea (check at intermediate points whether to
  continue) is the right governance pattern, even if the specific
  gate at step 3 is impossible to execute (see P2-1).

---

## Suggested rework before starting

Don't start step 1 until the plan:

1. Has accurate LOC accounting against the actual files (P0-1, P1-4).
2. Confirms `generate_dflash` stays as a delegate (per user
   clarification 2026-05-28), and the unified function's
   `Path::DFlashSingle` arm calls it and returns.
3. Has a Step 0 that extends `forward_prefill_batch_multi` with
   `per_token_hidden_out` + a tape-less-replay decision (P0-3).
4. Decides the `gpu` vs `m.pp_gpus.devices[0]` ownership question
   (P0-4) and writes it down.
5. Designs a `Gpus::disjoint_pair_mut(i, j)` / `single_mut(i)` API
   that handles the same-device shortcut WITHOUT requiring unsafe
   aliasing in callers (P0-5). Lands as a prerequisite refactor.
6. Resolves the eviction story (P1-1) — refuse at load or carry
   defensive handling.
7. Demotes PFlash+MTP from "silent enable" to "bypass with explicit
   event, validate later" (P1-2, gemini agrees with concrete
   mechanism).
8. Budgets gate runs honestly (P1-3) and includes building the
   PP-AR coherence test as a step 0 alongside the prefill_multi
   extension. Uses `coherence-gate-dflash.sh` as the spec-decode
   canonical.
9. Introduces a `GenerateCtx` struct to tame the 23+ param union
   (P1-5).
10. Rewrites step 5 as **5 sub-commits** (one per migrated path),
    not one surgery commit (P1-6, gemini's split-surgery
    recommendation).
11. Clarifies the post-unification top-level dispatcher (P2-4):
    `generate` keeps its arch-id and combo-routing role,
    `generate_qwen35` replaces the inline AR body of `generate`
    plus the three children for the qwen35 arch.
12. Adds a tok/s sanity check to the validation criteria (P2-5):
    AR baseline must stay within 1% of pre-refactor.
13. Removes the "Refused" arms from the match-statement design
    (P2-8) — those combos don't reach the function.
14. Reframes Option X as a credible first option with honest
    cost comparison vs the unification (P2-10).

Estimated cost of this rework: 60-90 min of plan rewriting, plus
maybe 30 min of code reading to nail down the prefill_multi
gdn_tape question (P0-3). That investment is much cheaper than the
cost of finding any of the P0s mid-refactor.

## What I am NOT certain of

- The `gdn_tape` claim in P0-3 specifically — I haven't read the
  full DN rollback path under multi-gpu, just the spec_step's tape
  capture call. There may be a simpler way to thread it that I'm
  missing. Worth ~30 min of focused code-read before either
  accepting or rejecting the "tape-less replay forced in multi
  mode" option.
- The validation time estimate (P1-3) — I'm estimating 65s/test from
  the gate's max_tokens settings but haven't measured. Could be
  faster on warm load (cached weights) or much slower if 27B coh
  tests take 90+ seconds each.
- The Option X vs unification cost trade (P2-10) — both reviews give
  enough to question my initial bias toward unification, but neither
  side of the trade has been costed against actual file diffs. Worth
  a half-day spike sketching Option X as a draft PR to see how big
  it really gets before committing to the unification.

These uncertainties don't change the recommendations above; they just
mean the P1-3 and P0-3 specifics should be tightened before they
become design constraints.

---

## §A — Verdict on each external review claim

### glm5 review claims

| # | claim | verdict | notes |
|---|---|---|---|
| 1 | LOC counts wrong (1933 not 1586) | ✅ accept | Matches my P0-1. glm5's 1933 figure cleaner than my "actual: 1930" — both ~the same. |
| 2 | generate_dflash missing from scope | ✅ accept | Per user clarification: dflash stays as delegate. Plan must say so. |
| 3 | generate is dispatcher + AR impl | ✅ accept | Matches my P0-2. |
| 4 | 18+ param explosion, no context struct | ✅ accept | I missed this; promoted to my P1-5. |
| 5 | PrefillContext hides 5-param divergence | ✅ accept | Matches my P0-3. glm5's enumeration of the missing params is more precise. |
| 6 | Step 5 critically underspecified (decode-loop merge, state divergence, event divergence) | ✅ accept | Matches my P1-4 conceptually. glm5's three-axis breakdown (decode, state, events) is more useful. |
| 7 | Borrow-checker blocker with dual &mut Gpu | ✅ accept | Matches my (rev2-promoted) P0-5. Already shipped unsafe code in Stage 2a confirms. |
| 8 | Refusal matrix incomplete (6 combos, plan only shows 1) | ⚠ partial | Right that refusal matrix has 6 combos. Wrong about the design — match in unified fn only sees post-refusal validated combos; Refused arms shouldn't appear. Demoted to P2-8. |
| 9 | "300 LOC saved" overstated; realistic ~100-120 | ✅ accept | Matches my P1-4 with more detail. |
| 10 | coherence-gate.sh is deprecated | ❌ reject | glm5 misread AGENTS.md. What's deprecated is `quality-gate.sh`. coherence-gate.sh is still authoritative. BUT the underlying suggestion (rely on coherence-gate-dflash for spec-decode work) is sound — adopt. |
| 11 | PFlash state on daemon locals, not LoadedModel | ✅ accept | Doc fix. Demoted to P2-9. |
| 12 | "2-3 days" estimate is unrealistic; 4-7 days | ✅ accept | With my P0-3/P0-5 additions and gemini's split-surgery sequencing, 4-7 days is realistic. |
| 13 | Option X is undersold | ⚠ partial | True that plan's framing biases against Option X. False that Option X dodges all costs — still hits P0-3 and P0-5. Reframe as credible first option, not dismiss. Promoted to my P2-10. |
| 14 (a) | "[upcoming]" commit placeholder | ✅ accept | Minor doc fix. (Stage 2a is now commit 54e18194.) |
| 14 (b) | Stage 2a claims need verification | ✅ accept (now verified) | Confirmed against branch. |
| 14 (c) | Eviction "for free" claim is backward | ⚠ partial | glm5 right that pp>1 + eviction is refused at load. But MTP+eviction at pp=1 has a defensive-warn bug today. Both points stand. |
| 14 (d) | output_device extraction is a borrow concern | ✅ accept | Subset of my P0-5. |

### gemini review claims

| # | claim | verdict | notes |
|---|---|---|---|
| §1 God-function trap (state explosion) | ✅ accept | Promoted to my P1-5 (param explosion). |
| §1 Borrow checker fragility | ✅ accept | Promoted to my P0-5. |
| §1 Implicit refusal logic | ⚠ partial | Concern is real but the existing load-time refusals already exist; consolidating them into one match-on-path isn't WORSE than today's scattered guards, just different. |
| §2 PrefillContext leakage | ✅ accept | The decode loop's Gpu vs Gpus distinction won't fully go away. Adopted into my P0-3. |
| §2 MTP speculative state divergence | ✅ accept | Real concern; needs explicit per-path state handling. |
| §2 PFlash composition (KV-offset mismatch) | ✅ accept | Gemini provides concrete mechanism I missed; strengthens my P1-2. |
| §3 Permutation problem (7 paths) | ✅ accept | Combined with my P1-3 (gate budget) and the existing 6 refusal combos, the test surface is large. |
| §3 Performance: 0% AR regression target | ⚠ partial | 0% is too aggressive (run-to-run noise is ~1-3%). Promote to my P2-5 with 1% target. |
| §4 VRAM/scratch re-allocation concern | ❌ reject | Speculative; load_model_pp already handles scratch alloc, generate doesn't re-allocate. Not a real risk for the refactor. |
| §4 Match-statement CPU latency | ⚠ partial | Technically true, quantitatively negligible. Promoted to my P2-5 with sanity-check, not blocker. |
| §5 Device failure handling | ❌ reject | Out of scope; no GPU failure handling exists today, refactor doesn't make it worse. |
| §5 Cancellation handling | ❌ reject | Out of scope; no cancellation today, refactor doesn't make it harder. Promoted to my P2-7. |
| §5 Eviction story | ⚠ partial | Gemini's "explicit delegate" suggestion is reasonable, but eviction-at-pp>1 is already refused. Concern overlaps my P1-1. |
| Rec #1 Generator trait/strategy pattern | ❌ reject | Adds vtable overhead and class-hierarchy verbosity; not idiomatic Rust. Promoted to my P2-6 with rejection rationale. |
| Rec #2 Explicit PFlash+MTP guard | ✅ accept | Matches my P1-2 and gemini's own §2. |
| Rec #3 Benchmark early | ✅ accept | Matches my P2-5. |
| Rec #4 Split the surgery into per-path migrations | ✅ accept | Strong recommendation. Promoted to my P1-6. **This is the most valuable single change from external review.** |
| Rec #5 Audit drafter_gpu ownership | ✅ accept | Matches my P0-5. |

### Summary of external review impact

External reviews added **3 P0/P1 findings I missed**:
- P0-5 borrow-checker (both reviewers)
- P1-5 param explosion (both reviewers)
- P1-6 split the surgery (gemini's #4 recommendation, the most
  actionable single suggestion across both reviews)

External reviews **partially or fully rejected**:
- glm5 #10 coherence-gate-deprecated (misread AGENTS.md)
- gemini #4 vram concern (speculative)
- gemini Rec #1 trait pattern (not idiomatic)
- gemini #5 device failure / cancellation (out of scope)

External reviews **refined my existing P0/P1s with better mechanism
or more precision**:
- glm5 #5 (PrefillContext gap detail) into my P0-3
- gemini §2 (KV-offset mechanism for PFlash+MTP) into my P1-2
- glm5 #9 (path-specific divergence in prelude/postlude) into my P1-4

Net: my review caught 4 of the 5 P0/P1 findings (the borrow-checker
issue was independently called out by both reviewers and I missed
it). The external reviews caught the structural-decomposition concern
(split-the-surgery) that materially changes the execution plan. Both
external reviewers' tendency to propose alternative designs (Generator
trait, GenerateCtx struct) is useful pressure even when individual
proposals don't pan out, because it forces the plan to defend its
design instead of asserting it.
