# Generate-function comparison map (Stage 2b step 1)

Per-section structural analysis of the four generate functions in
`daemon.rs`. Output: a cheat sheet for step 2 (helper extraction)
through step 5 (per-path migration). No code changes in this step.

## Function ranges (verified)

| function | lines | LOC | role |
| --- | --- | ---: | --- |
| `generate_dflash` | 3170-3661 | 492 | DFlash spec-decode (single-gpu, greedy-only) |
| `generate_mtp` | 3664-4118 | 455 | MTP spec-decode (single-gpu OR hetero via drafter_state) |
| `generate_multi` | 4130-4615 | 486 | PP=2 AR + PFlash compress |
| `generate` | 4618-5609 | 992 | top-level dispatcher + inline AR impl |

Total: **2425 LOC**, of which `generate_dflash` is delegated (per user
2026-05-28). In-scope for unification: ~1933 LOC.

## Top-level dispatcher (`generate`, lines 4618-4716)

`generate` is the public entry point called from the daemon's request
handler at daemon.rs:1226. Dispatches by arch_id / pp / dflash / mtp
guards BEFORE running its own inline AR body:

```
generate(m, gpu, drafter_gpu, ...)
├─ arch_id < 5 → generate_qwen2(...) return
├─ m.pp > 1 → generate_multi(...) return
├─ m.dflash.is_some() && temp≈0 && !budgeted_thinking → generate_dflash(...) return
├─ m.mtp.is_some() && (5|6) && !budgeted_thinking_needs_ar && !mtp_blocked_by_penalty
│       → generate_mtp(...) return
└─ inline AR body (~900 LOC starting line 4718)
```

The inline AR body has PFlash compression + chat-frame + capacity check
+ prefill + decode loop. Effectively a 5th function pretending to be
half of `generate`. **Stage 2b's `Path::Ar` arm corresponds to this
inline body, NOT to `generate_multi`.**

## Per-section comparison table

The four bodies share a common skeletal pattern; this table aligns
their sections side-by-side so step 2 (extract helpers) can pull the
right substring out of each.

### S1. Tokenize & multi-turn rollover (prelude)

| function | site | divergence |
| --- | --- | --- |
| inline AR (in generate) | 4723-4737 | `m.eviction.is_none() && m.seq_pos+prompt_est+max_tokens > m.max_seq` → reset (DN+kv compact_offset+llama_kv). Single-gpu DN reset via `gpu.hip.memset`. |
| generate_multi | 4151-4179 | Same threshold check, but DN reset walks `pp_dn_la_to_device` to bind the right band per LA layer. ALSO clears `kv.compact_offset`. |
| generate_mtp | 3683-3695 | No eviction check (it didn't have llama_kv branch); just `seq_pos+prompt_est+max_tokens > m.max_seq`. **NO DN reset here** — done later at line 3768-3783 conditionally on `seq_pos==0`. |
| generate_dflash | (inside its prompt-build block ~3190-3287) | Has its own conversation_tokens clear; pattern differs further because dflash has draft model state too. |

**Extraction candidate `auto_reset_conversation(m, gpu_or_gpus, prompt_est)`:**
3 of the 4 functions (AR, multi, mtp) share the rollover-and-reset
skeleton. The single-gpu form passes `&gpu`; multi passes
`(&mut gpus, &la_to_device)`. Helper signature would need a path
discriminant or two overloads (`_single` and `_multi`). ~25 LOC
saved per function = ~75 LOC total.

**Note for generate_mtp specifically:** its DN reset at line 3768-3783
ALSO resets `mtp.spec_state.mtp_kv` — that's MTP-specific cleanup the
others don't do. Helper must accept an optional "extra cleanup"
closure OR keep the MTP-specific cleanup inline.

### S2. ChatML / Jinja framing

| function | site | divergence |
| --- | --- | --- |
| inline AR | (later, ~4844-4960 area) | Full Jinja path + Plain fallback. `assistant_prefix` passed through. |
| generate_multi | 4260-4323 | Full Jinja path + Plain fallback. Same shape as AR. PFlash-compressed q_tokens used in Plain fallback (not raw). |
| generate_mtp | 3697-3763 | Full Jinja path + Plain fallback. Same shape. **NO PFlash** (MTP is not yet PFlash-aware). |
| generate_dflash | (in build_prompt block) | Has its own framing. Variant of the same shape. |

**Extraction candidate `build_prompt_frame(m, tokenizer, ctx) -> Vec<u32>`:**
all four functions do essentially the same Jinja + Plain branch.
Differences: which `user_tokens` slice they feed (raw vs PFlash-
compressed), and whether `system` resets on continuation. Helper
takes `&[u32] user_tokens` (caller does PFlash before calling) and
`Option<&str> system` (caller decides reset). ~70 LOC saved per
function = ~210 LOC total. **Largest single win.**

### S3. Capacity check

| function | site | divergence |
| --- | --- | --- |
| inline AR | 4725 (combined with reset) | `seq_pos + prompt_est + max_tokens > max_seq` triggers reset, NOT error. |
| generate_multi | 4326-4334 | Hard error: `seq_pos + new_tokens + max_tokens + trailer > physical_cap` |
| generate_mtp | 3819-3831 | Hard error: `start_pos + new_tokens + max_n + 1 > physical_cap` (NB: max_n adds K MTP candidates) |
| generate_dflash | ~3338-3367 | Hard error sequence with multiple `return;` early-outs |

**Two different patterns:** AR uses the auto-reset (S1) but doesn't
hard-error; the spec paths hard-error because they need exact
position math. Don't extract this as one helper — keep per-path. ~5
LOC each, not worth a helper.

### S4. Prefill kernel call

| function | call | divergence |
| --- | --- | --- |
| inline AR | (eventually) `qwen35::forward_prefill_batch` | single-gpu |
| generate_multi | 4366-4372 | `qwen35::forward_prefill_batch_multi` |
| generate_mtp | 3878-3893 | `qwen35::forward_prefill_batch` (single) |
| generate_dflash | (inside body) | `forward_prefill_batch` |

After step 0b: PpMtp will call `forward_prefill_batch_multi_with_caps`
with `per_token_hidden_out + gdn_tape + tree_verify=None`. This is the
ONE site where the prefill dispatch differs by path.

**Extraction candidate `PrefillCtx` enum (per v2.1 plan, ~150 LOC):**
```rust
enum PrefillCtx<'a> {
    Single { gpu: &'a mut Gpu, scratch: &'a Qwen35Scratch },
    Multi  { gpus: &'a mut Gpus, scratch_set: &'a Qwen35ScratchSet,
             gdn_tape: Option<&'a mut GdnTape>,
             per_token_hidden_out: Option<&'a GpuTensor> },
}

impl<'a> PrefillCtx<'a> {
    fn forward_batch(&mut self, weights, config, tokens, start_pos,
                     kv, dn, hidden_rb, per_token_hidden_out_single,
                     gdn_tape_single, tree_verify) -> HipResult<()>;
}
```
NB: per_token_hidden_out and gdn_tape exist on BOTH single and multi
sides; the difference is just the dispatcher target. Cleaner: have
forward_batch take ALL the spec-decode args and route internally.

### S5. Seed token (first token after prefill)

| function | site | divergence |
| --- | --- | --- |
| inline AR | (immediately starts decode loop, no separate seed) | first sample on the post-prefill logits inside the loop |
| generate_multi | 4386-4412 | First sample on `scratch_set.per_device[dev_last].logits` via `sampler::sample` |
| generate_mtp | 3905-3914 | Download trunk logits, argmax → first_token. Greedy. (Sampling path inside spec_step.) |
| generate_dflash | ~3403-3450 | First token from prefill argmax + emit_committed of seed token |

**Different enough not to share a helper.** AR has no separate seed
emit; MTP and DFlash both emit a seed before the decode loop starts.
Generate_multi samples (not just argmax) because it's AR.

### S6. Decode loop

| function | shape | divergence |
| --- | --- | --- |
| inline AR | `while generated < max_tokens` — calls `forward_scratch` per token, samples on post-logits, emits | Sampling + repeat_penalty + max_think_tokens enforcement + budget_alert |
| generate_multi | `while generated < max_tokens` — calls `forward_scratch_multi` per token | Same as AR but multi-gpu dispatch; NO budget_alert in current code |
| generate_mtp | `while generated < max_tokens && !first_token_terminal` — calls `spec_step_mtp_compressed_serial` per cycle | Per-cycle accept_count accounting; ngram loop_guard; max_think enforcement different |
| generate_dflash | `while generated < max_tokens` — calls `spec_step_dflash` (or ddtree variants) per cycle | DFlash spec stats, draft-rejection accounting, BUDGET-CAPped think handling |

**This is the heaviest divergence.** Four structurally different
decode loops, only the per-token streaming/emission/filter logic is
shared (~30 LOC per loop). The `loop_guard.observe(...)` ngram-attractor
check is in three of them (AR, multi, mtp); dflash has its own
analogous guard.

**Extraction candidate `decode_loop_postlude_step(stdout, tokenizer,
filter, ctx, tok)`:** the per-token emit + filter-update + decoded-bytes
tracking is identical across all four loops. ~20 LOC per function = ~80
LOC total.

But the SURROUNDING loop body diverges hard — Stage 2b's per-path
migration commits each move one whole loop in.

### S7. max_think_tokens enforcement

| function | site | divergence |
| --- | --- | --- |
| inline AR | ~4960+ inside decode loop | Decoded-text scan + close-tokens emission |
| generate_multi | 4451-4500 | Identical pattern to AR's (decoded-text scan + close emit), but `forward_scratch_multi` for the close-token forwards |
| generate_mtp | (different — enforced inside its own loop) | seq_pos-aware; close tokens forwarded via `forward_scratch` |
| generate_dflash | (BUDGETED — enforced INSIDE spec_step_dflash) | Different mechanism per the function's comments |

**Extraction candidate `enforce_max_think(scope, ...)`:** AR and multi
share this almost verbatim. MTP has a variant. DFlash has its own
budget mechanism. ~40 LOC saved across AR+multi.

### S8. Done event emission

| function | shape |
| --- | --- |
| inline AR | `{"type":"done","id":..,"tokens":N,"tok_s":..,"prefill_*","decode_tok_s":..,"ttft_ms":..}` + optional `pflash`/`pflash_bypass_reason` JSON fragment |
| generate_multi | Same shape as AR (mostly identical — copy-pasted) |
| generate_mtp | Same shape PLUS `"spec_path":"mtp","mtp_k":K,"tau":T,"accept_rate":A,"cycles":C,"mtp_sampling":bool` |
| generate_dflash | Same shape PLUS `"dflash":true,"tau":T,"cycles":C` |

**Extraction candidate `emit_done_event(ctx, base_stats, path_extras)`:**
base shape is shared; the path-specific extras are appended after
the base fields. Helper takes a base `DoneStats` struct + an
`Option<String>` for path-specific JSON-fragment. ~30 LOC per function
= ~120 LOC total.

### S9. Conversation/seq_pos bookkeeping

| function | site | divergence |
| --- | --- | --- |
| inline AR | scattered throughout decode loop | `m.conversation_tokens.push(tok); m.seq_pos += 1;` per token |
| generate_multi | similar scattered | Same |
| generate_mtp | per cycle | `m.conversation_tokens.extend_from_slice(&step.committed); m.seq_pos += step.advance;` |
| generate_dflash | per cycle | Per dflash cycle's commit count |

**Not extractable** — these are 1-2-line operations interleaved with
control flow. Leave in place.

## Honest LOC accounting (post-step-1 revision)

Per-section extraction estimates:

| section | extraction | LOC saved |
| --- | --- | --- |
| S1 auto-reset | helper with single/multi overloads | ~75 |
| S2 chat frame | helper takes pre-PFlash q_tokens | ~210 |
| S3 capacity | none | 0 |
| S4 prefill | PrefillCtx enum | (refactor, neutral or slight loss) |
| S5 seed | none | 0 |
| S6 decode-loop body | per-token postlude step | ~80 |
| S7 max_think | helper for AR+multi | ~40 |
| S8 done event | helper with base + path-extras | ~120 |
| S9 bookkeeping | none | 0 |
| **Total saved** | | **~525 LOC** |

**This is higher than v2.1's ~120 LOC estimate.** Why? v2.1
under-counted S2 (chat frame at ~210 LOC dedup) and S8 (done event
at ~120 LOC dedup) — those are mostly-identical bodies across all
four functions.

Revised: **the unification's real LOC savings are closer to 400-500
LOC** (after subtracting helper struct + PrefillCtx overhead of
~100 LOC), not 120. The audit was pessimistic.

## Step 2 commit plan (helper extraction order)

Order helpers by extraction complexity + blast radius (smallest first):

1. **`emit_done_event` helper** (~30 LOC × 4 = 120 dedup). Self-contained
   string formatting. Lowest risk; coherence-gate sanity per commit.
2. **`build_prompt_frame`** (~70 LOC × 3 = 210 dedup). Pure function,
   no GPU access. AR + multi + mtp use this; dflash keeps its own
   (delegated by user decision).
3. ~~**`auto_reset_conversation`** (~25 LOC × 3 = 75 dedup).~~
   **SKIPPED 2026-05-28 after measurement.** Closer look reveals the
   shared part across the three call sites is only ~5 LOC (log +
   `seq_pos = 0` + `conversation_tokens.clear()`); everything else
   diverges (eviction gate AR-only, DN-reset single vs multi vs
   none-here, kv.compact_offset AR+multi only, llama_kv.compact_offset
   AR only, log prefix). Helper would save ~5 LOC net. Not worth a new
   4-arg function. Path-specific reset code stays inline; step 5
   per-path migrations move it under each Path arm naturally.
4. **`enforce_max_think`** (~40 LOC × 2 = 80 dedup). AR + multi only.
5. **`decode_loop_postlude_step`** (~20 LOC × 4 = 80 dedup). Per-token
   emit shared across all four loops.

Each commit changes ONE function to use the new helper, leaves the
others using their inline version. After step 2 lands, helpers exist
and 3-4 of the 4 functions use them; the remaining inline copies get
removed in step 5 per-path migrations.

## Step 3 GenerateCtx scope (per v2.1 plan)

The 23-param union across the four functions bundles cleanly into:

```rust
struct GenerateCtx<'a> {
    // I/O
    stdout: &'a mut std::io::Stdout,
    id: &'a str,
    // Prompt
    prompt: &'a str,
    system_prompt: Option<&'a str>,
    tools: Option<&'a [serde_json::Value]>,
    messages_history: Option<&'a [Message]>,
    assistant_prefix: AssistantPrefix,
    // Sampling
    sampling: SamplingCfg,  // temp, top_p, repeat_penalty, repeat_window
    // Decode budget
    max_tokens: usize,
    max_think_tokens: usize,
    budget_alert: Option<BudgetAlert>,  // at_tok + text
    // Co-GPU resources
    drafter_gpu: Option<&'a mut Gpu>,
    pflash_state: Option<&'a mut PflashState>,
    pflash_cfg: Option<&'a PflashConfig>,
}
```

13 fields → from 23+ named-arg slots → drops every signature to
3 args: `(m, gpus_or_gpu, ctx)`.

## Step 4 SpecPath enum (per v2.1 plan)

```rust
enum SpecPath {
    Ar,            // m.pp==1, !mtp, !dflash, [any arch]
    PpAr,          // m.pp>1,  !mtp, !dflash
    MtpSingle,     // m.pp==1, mtp.is_some(), drafter_state.is_none()
    MtpHetero,     // m.pp==1, mtp.is_some(), drafter_state.is_some()
    PpMtp,         // m.pp>1,  mtp.is_some()         ← Stage 2b payoff
    DFlashSingle,  // m.pp==1, dflash.is_some()       ← delegates to generate_dflash
}
```

PP-DFlash is refused at load (unless HIPFIRE_PP_DFLASH=1 escape, which
the spec function ignores in v1). PFlash+MTP refused at load with
bypass event. No combos with dflash + mtp (refused at load).

`fn pick_path(m: &LoadedModel) -> SpecPath` is the central dispatcher.
Called from `generate_qwen35`'s entry. Step 4's commit can land this
function PLUS a `debug_assert!(pick_path(m) == EXPECTED)` at the top
of each existing generate body — catches load-handler refusal-matrix
bugs before any further migration.

## Step 5 per-path migration commits (one per arm)

Sequenced cheapest → most valuable:

| commit | scope | new arm | replaces |
| --- | --- | --- | --- |
| 5a | extract AR | `Path::Ar` | inline AR body in `generate` |
| 5b | extract PP-AR | `Path::PpAr` | `generate_multi` (delete after) |
| 5c | extract MTP | `Path::MtpSingle` + `Path::MtpHetero` | `generate_mtp` (delete after) |
| **5d** | **add PP+MTP** | **`Path::PpMtp` (NEW)** | **— first new functionality** |
| 5e | DFlash wrap | `Path::DFlashSingle` | thin wrapper around `generate_dflash` (keeps function intact) |

Each commit is independently revertable. After 5d the user-visible
ship has landed. 5e is cleanup.

## Open observations for the user

1. **MTP-side mtp_kv reset in S1 is MTP-specific cleanup.** The
   auto-reset helper either takes a closure for path-specific extras
   OR the MTP arm runs it after the helper. The latter is simpler;
   recommend that.

2. **Generate_multi does NOT have `budget_alert` wiring** that AR
   has. After unification, PpAr arm should inherit AR's budget_alert.
   That's a small bug-fix-by-unification, not a regression. Worth
   documenting in the step 5b commit.

3. **MTP has no PFlash integration today.** Per user decision, refuse
   the combo with bypass event. Already documented in v2.1 plan.

4. **DFlash has BUDGETED max_think_tokens (inside spec_step_dflash)**
   while the others enforce via decoded-text scan in the decode loop.
   Step 5e (DFlash wrap) just delegates so this stays intact.

5. **The ngram loop_guard wiring** is shared across AR/multi/mtp.
   Could be extracted as a helper but it's per-cycle vs per-token
   semantics differ between AR (per token) and MTP (per cycle).
   Leave per-path for now; revisit if it becomes noisy.

## Conclusion

Step 1 confirms the v2.1 plan's structure is sound. The honest LOC
savings are **higher** than the plan estimated (~500 LOC dedup vs
v2.1's ~120). The decode-loop divergence is the highest-risk
section; step 5 per-path migration is the right governance pattern.

Ready for step 2 (helper extraction, starting with `emit_done_event`).
