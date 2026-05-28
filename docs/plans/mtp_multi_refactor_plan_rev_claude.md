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
proposed `SpecPath` enum lists `Dflash` as one of its arms. Either
DFlash is in scope (and the plan undercounts another 491 LOC) or it
is not (and the plan should explicitly say "DFlash stays as a
separate function for v1"). Right now the plan reads "we'll unify
three functions" but reasons about a dispatch matrix that includes a
fourth function the plan never opens.

**Fix:** redo the LOC accounting against the actual file. Decide
explicitly whether `generate_dflash` is in or out of scope. If in,
the project is 50% bigger than estimated. If out, the `SpecPath::Dflash`
arm in §3 of the plan is a delegation, not an inline impl, and that
needs to be in the design.

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

**Fix:** either actually refuse `mtp + eviction` at load (one line
in the load handler), or carry the defensive eviction handling
through into the unified function. The plan needs to pick.

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

**Fix:** budget validation time honestly. Pick which gate runs are
mandatory between steps and which can defer to a single big run at
the end of step 5. My recommendation: gate after step 2 (helpers
extraction), full gate after step 5 (the surgery), single-run check
after each of 3, 4, 6. That cuts gate wall to ~45 min total.

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

**Fix:** redo the LOC accounting end-to-end with the actual file
sizes. The 1930-LOC reality (P0-1) plus an honest duplication
audit will give a real before/after.

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

### P2-4: No mention of VL/qwen2 paths

`generate_qwen2` (line 5632) and `generate_vl*` (lines 5789, 6190)
exist as parallel generate functions for non-qwen35 archs. The plan
names the unified function `generate_qwen35` — which is honest
that it scopes to qwen35-arch — but does not say what the daemon's
top-level `generate` dispatcher looks like after the unification.
Specifically: does the dispatcher remain (calling out to
`generate_qwen2` / `generate_vl_dots_ocr` / `generate_qwen35`)?
The plan should explicitly diagram this.

---

## What the plan got right

- The overall direction (unify into one path-dispatched function) IS
  correct. The duplication is real even if the LOC accounting is off.
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
2. Decides scope on `generate_dflash` (in or out, with rationale).
3. Has a Step 0 that extends `forward_prefill_batch_multi` with
   `per_token_hidden_out` + a tape-less-replay decision (P0-3).
4. Decides the `gpu` vs `m.pp_gpus.devices[0]` ownership question
   (P0-4) and writes it down.
5. Resolves the eviction story (P1-1) — refuse at load or carry
   defensive handling.
6. Demotes PFlash+MTP from "silent enable" to "bypass with explicit
   event, validate later" (P1-2).
7. Budgets gate runs honestly (P1-3) and includes building the
   PP-AR coherence test as a step 0 alongside #3.
8. Diagrams the post-unification top-level dispatcher (P2-4).

Estimated cost of this rework: 30-60 min of plan rewriting, plus
maybe 30 min of code reading to nail down the prefill_multi gdn_tape
question (P0-3). That investment is much cheaper than the cost of
finding any of the P0s mid-refactor.

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

These uncertainties don't change the recommendations above; they just
mean the P1-3 and P0-3 specifics should be tightened before they
become design constraints.
