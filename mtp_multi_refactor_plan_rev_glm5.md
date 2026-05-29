# Adversarial Review: `docs/plans/mtp_multi_refactor.md`

Reviewer: glm-5-turbo (automated adversarial pass)
Date: 2026-05-28
Verdict: **Plan needs significant revision before execution. Multiple
factual errors in the LOC/duplication analysis, a missing function from
the scope, an underspecified core surgery step, and at least one
borrow-checker blocker that the plan doesn't acknowledge.**

---

## 1. LOC counts are materially wrong

The plan's "current state of the four functions" table:

| function | Plan says | Actual |
|---|---|---|
| `generate` | ~600 | **992** (daemon.rs:4622–5613) |
| `generate_multi` | ~486 | 486 (correct) |
| `generate_mtp` | ~500 | **455** (daemon.rs:3668–4122) |

The total existing LOC is stated as ~1586. The three functions alone sum
to **1933**. That's a 22% understatement. If `generate_dflash` (492 LOC,
see §2) is included, the real total is **2425** — 53% higher than
claimed.

This matters because the plan's cost-benefit argument rests on "300 LOC
duplication removed" against "1586 LOC existing." If the base is 2425,
the duplication ratio is different, and the refactoring surface is
larger than the plan leads the reader to expect.

## 2. `generate_dflash` is entirely absent from the analysis

There are **four** existing generate functions, not three:

| function | LOC | file |
|---|---|---|
| `generate_dflash` | 492 | daemon.rs:3174–3665 |
| `generate_mtp` | 455 | daemon.rs:3668–4122 |
| `generate_multi` | 486 | daemon.rs:4134–4619 |
| `generate` | 992 | daemon.rs:4622–5613 |

The plan's table lists only `generate`, `generate_multi`,
`generate_mtp`, and the planned `generate_mtp_multi`. It never mentions
`generate_dflash`. Yet the proposed unified function includes
`Path::DFlash` in its match matrix and the decode-loop sketch lists
`DFlash → spec_step_dflash` as a dispatch arm.

**Impact:** The unified function would need to absorb FIVE function
bodies, not four. DFlash has unique state management (draft model
lifecycle, tree-verify batching, DDTree paths, budgeted-think handling)
that doesn't exist in the other functions. The LOC estimate and
complexity assessment are both understated.

## 3. The plan ignores `generate`'s dual role as dispatcher + AR impl

`generate()` at line 4622 is not just an "AR function" — it is the
**top-level dispatcher** for all four paths:

```
arch_id == 7  → generate_qwen2 (return)
pp > 1        → generate_multi (return)
dflash+greedy → generate_dflash (return)
mtp           → generate_mtp (return)
else          → inline AR body (the remaining ~900 LOC)
```

Each sub-call returns immediately. The plan proposes collapsing all of
this into a single `generate_qwen35` function, but doesn't discuss what
happens to `generate_qwen2` (arch_id=7) or `generate_vl` (arch_id=8).
These are separate arches that share the same dispatch point in `main()`.

The refactoring scope is unclear: is `generate_qwen35` a qwen35-only
extract, or does it replace the entire `generate` dispatcher? If the
former, the four-function table is wrong (generate's inline AR body is
qwen35-specific, but generate itself is arch-agnostic). If the latter,
the plan needs to address arch_id routing.

## 4. Parameter explosion — 18+ params, no context struct proposed

The unified signature sketched in the plan has ~18 parameters. In Rust,
this is a serious code smell. Four existing functions have **different
parameter sets**:

- `generate`: 21 params (including pflash_state, pflash_cfg, tools, messages_history)
- `generate_multi`: 19 params (includes pflash_state, pflash_cfg; omits drafter_gpu)
- `generate_mtp`: 14 params (no pflash, no budget_alert, no drafter_gpu)
- `generate_dflash`: 13 params (unique pflash_bypass_reason, pflash_alpha; no temp/top_p/repeat)

Unification forces the **union** of all these — roughly 23+ parameters.
The plan doesn't propose a `GenerateCtx` or similar struct to tame this,
nor does it acknowledge the param-set divergence.

## 5. PrefillContext enum hides a 5-parameter divergence

The plan mentions that `forward_prefill_batch` and
`forward_prefill_batch_multi` have different shapes, but understates the
gap:

- **Single:** `(gpu, weights, config, tokens, start_pos, kv_cache, dn_state, scratch, hidden_rb, per_token_hidden_out, gdn_tape, tree_verify)` — 12 params
- **Multi:** `(gpus, weights, config, tokens, start_pos, kv_cache, dn_state, scratch_set)` — 8 params

Multi lacks `hidden_rb`, `per_token_hidden_out`, `gdn_tape`, and
`tree_verify`. These aren't cosmetic — `hidden_rb` is the ring buffer
for MTP's per-token hidden states; `gdn_tape` is the DeltaNet recording
for speculative decode; `tree_verify` is the DDTree verification
context.

The plan's `PrefillCtx` enum would need to either:
(a) carry Option<> wrappers for the single-gpu-only params (ugly, leaks
abstraction), or
(b) have the multi variant silently ignore them (surprise at the call
site).

The "+200 LOC" estimate for PrefillCtx is optimistic; the method
dispatch alone to paper over this gap is substantial.

## 6. Step 5 (the unification surgery) is critically underspecified

Step 5 is described as: "collapse generate / generate_multi /
generate_mtp bodies into the new generate_qwen35. ~500 LOC of
moved-around code, ~200 LOC genuinely deleted."

This is the highest-risk step and it gets one paragraph. There is no
discussion of:

- **How the decode loops merge.** AR is a simple per-token loop.
  MTP loops over draft-verify-accept cycles with token accounting that
  has no AR analogue. DFlash has its own speculative loop with draft
  model management, tree verification, and budgeted-think caps. These
  are not "moved-around code" — they are structurally different control
  flows.
- **State management divergence.** `generate_mtp` uses `ModelSlot`,
  `MtpSpecState`, and compressed-vocab sidecar state. `generate_dflash`
  uses `ModelSlot`, `Phase2Snapshots`, `SpecStats`, and a draft model.
  `generate_multi` uses `Gpus`, `Qwen35ScratchSet`, and boundary-copy
  logic. These are not interchangeable.
- **Event emission divergence.** Each function emits a different JSON
  event stream: `mtp_tau`, `dflash_stats`, `pp_progress`, AR's
  `tok` events. The "common postlude" is much smaller than the ~60 LOC
  claimed.

## 7. Borrow-checker blocker with dual `&mut Gpu`

The proposed signature includes:

```rust
gpu: &mut Gpu,
drafter_gpu: Option<&mut Gpu>,
```

In the MTP hetero path, `drafter_gpu` may be the **same physical device**
as `gpu` (the Stage 2a same-device shortcut in `peer_clone_tensor` and
`seed_prev_hidden` fires when `device_id` matches). But Rust's borrow
checker prohibits two `&mut` to the same allocation. The current code
avoids this because `generate_mtp` only takes one `&mut Gpu` (the
same-device shortcut is internal to the spec_step functions, which take
`trunk_gpu: &mut Gpu` and `drafter_gpu: &mut Gpu` as separate params
received from different callers).

The plan doesn't discuss how the unified function would handle this at
the call site. The caller in `main()` would need to decompose `Gpus`
into two borrows — but if they're the same device, that's UB under
Rust's aliasing rules.

## 8. The refusal matrix in the match statement is incomplete

The plan shows 7 match arms, 2 of which are "Refused." The actual
load-time refusal logic (daemon.rs:766–826) has 6 refusal combos:

| Combo | Refusal |
|---|---|
| DFlash + pp>1 | Error unless HIPFIRE_PP_DFLASH=1 |
| CASK + pp>1 | Hard refusal |
| PFlash + pp>1 | Error unless HIPFIRE_PP_PFLASH=1 |
| MTP + DFlash | Hard refusal |
| MTP + CASK | Hard refusal |
| VL + pp>1 | Refused in load_model_pp |
| arch_id < 5 + pp>1 | Refused in load_model_pp |

The plan's match at step 3 collapses these to `(_, true, true) =>
Path::Refused` (MTP+DFlash). But MTP+CASK, CASK+pp>1, VL+pp>1, and
arch-gated refusals aren't in the match. The plan needs to either:
show the complete refusal matrix, or acknowledge that refusal is handled
at load time and the match only sees already-valid combinations.

## 9. "Common prelude/postlude: ~300 LOC removed" is overstated

The plan claims ~300 LOC of duplicated prelude/postlude across the
functions. Examining the actual bodies:

- **Prelude** (prompt encoding, chatml framing, capacity check): each
  function does this slightly differently. `generate_mtp` has no
  budget_alert. `generate_dflash` has PFlash bypass logic in its
  prelude. `generate_multi` has PFlash compression in its prelude. The
  common subset is maybe 60–80 LOC, not 100+130.
- **Postlude** (token emit, done event, tok/s): each function has
  path-specific event emission (mtp_tau, dflash_stats, etc.). The
  truly common part (push to conversation_tokens, emit done event) is
  ~30–40 LOC.

Realistic savings: ~100–120 LOC, not 300. The plan's efficiency argument
is inflated by ~2.5×.

## 10. `coherence-gate.sh` is deprecated

The plan's "Risks" section and "Suggested sequence" both reference
`scripts/coherence-gate.sh` as a validation gate. Per AGENTS.md (§0,
Hard Rules): "Quality-gate.sh is deprecated — its byte-exact baselines
drift faster than the engine evolves." The plan should reference only
`coherence-gate-dflash.sh` and propose a NEW PP+MTP coherence gate.

## 11. PFlash state is not on LoadedModel — parameter threading gap

The plan's function signature shows `pflash_state, pflash_cfg` as
parameters. In the actual code, PFlash state lives on **daemon-level
locals** (daemon.rs:564–574: `pflash_state: Option<PflashState>`,
`pflash_cfg: Option<PflashConfig>`), not on `LoadedModel`. The unified
function would still need these threaded from the daemon's main loop.
This isn't a design error per se, but the plan presents it as if these
are model-level fields.

## 12. The "2-3 days" estimate is unrealistic

Adjusting for the factual corrections above:

- Real LOC base: ~2425 (not 1586)
- Number of function bodies to merge: 5 (not 3, counting generate_dflash
  and generate's inline AR body)
- Underspecified PrefillCtx divergence: +1 day for design alone
- Step 5 unification is 2–3× harder than described (decode-loop
  divergence, state management, event emission)
- Missing borrow-checker analysis for dual &mut Gpu

Realistic estimate: **4–7 days** of focused work. The plan's 2–3 days
assumes the best case on every dimension simultaneously.

## 13. Option X is undersold as the fallback

The plan frames Option X (separate `generate_mtp_multi`, ~600 LOC) as
"pays back the duplication debt the FIRST time we want a 5th combo."
But:

- The codebase already has a one-function-per-combo pattern (generate,
  generate_dflash, generate_mtp, generate_multi). Adding a 5th follows
  established convention.
- The dispatch routing in `generate()` already handles this cleanly —
  add one `if` arm.
- ~600 LOC of new code is lower risk than touching ~2425 LOC of working
  code.
- The plan's own "Decision gates" section says: "if any of {AR,
  MtpSingle, DFlash} regress on coherence, revert the unification
  commit and ship Option X as a hotfix." This concedes that Option X
  is the safety net — but it should be presented as a credible first
  option, not a last resort.

## 14. Minor issues

- **"[upcoming]" commit placeholder** (line 3): the plan references a
  commit hash that doesn't exist. Should be filled in before the plan
  is actioned.
- **"Stage 2a foundation" claims** (lines 10–17): the plan lists
  Stage 2a features (peer_clone_tensor same-device shortcut,
  MtpState.drafter_state, load_model_pp MTP head load) as "already
  shipped." Verify these are actually on the branch being planned
  against — the "[upcoming]" placeholder suggests they may not be.
- **Eviction open question is backward** (lines 171–175): the plan
  says "if MTP+PP gets eviction 'for free' via the unified function,
  we'd need to either disable it or test it." But eviction is refused
  at load time for pp>1 (daemon.rs:773–777). The unified function
  doesn't change the load-time refusal. MTP+PP will NOT get eviction
  "for free" — it's gated before the function is ever called.
- **`output_device` extraction** (line 80): the plan says
  `target_gpu=output_device, drafter_gpu=output_device` for the PpMtp
  path. But `output_device` is a concept inside `Gpus`, not a separate
  handle. The plan doesn't discuss how to extract it as a `&mut Gpu`
  from `&mut Gpus` while also borrowing other devices for the trunk
  pipeline stages — another borrow-checker concern.

## 15. Summary of recommendations

1. **Correct the LOC counts** and re-derive the cost-benefit.
2. **Include `generate_dflash`** in the function table and duplication
   analysis, or explicitly scope it out with a rationale.
3. **Address the borrow-checker issue** with dual `&mut Gpu` — this is
   a potential hard blocker for the unified approach.
4. **Expand Step 5** with a per-decode-path divergence analysis showing
   exactly what differs in state management, event emission, and control
   flow.
5. **Propose a `GenerateCtx` struct** to handle the 23+ parameter union.
6. **Re-estimate timeline** to 4–7 days, or acknowledge the 2–3 day
   estimate assumes zero complications.
7. **Present Option X more neutrally** — it's a viable first option with
  lower risk, not just a fallback.
8. **Fix the eviction open question** — pp>1 eviction is refused at
   load, not at generate time.
9. **Reference `coherence-gate-dflash.sh`** instead of the deprecated
   `coherence-gate.sh`.
10. **Fill in the "[upcoming]" commit hash** before actioning.
