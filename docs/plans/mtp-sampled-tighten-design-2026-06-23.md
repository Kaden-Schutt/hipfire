# Design: Sampled + Tightened MTP Spec-Decode (hipfire)

> Source: `mtp-tighten-and-sampled-design` workflow (recon -> design -> adversarial review), 2026-06-23.
> **Status: design, no code landed.** Review verdict: **feasible-with-corrections.**
>
> WARNING / CRITICAL CORRECTION (from the review, see section "(a)"): the design body sections 1.3/1.4 prescribe a SAME-NUCLEUS truncation (apply the draft-derived tau to the target) plus a NEW kernel `softmax_temp_with_tau_batched_into_f32`. This is WRONG and diverges from shipped DFlash. The CORRECT fix to BUILD: INDEPENDENT per-side nuclei -- truncate the target with `softmax_temp_topp_batched_into_f32` using the TARGET top_p (its own, NOT the draft-derived tau; mirrors speculative.rs:3733-3747); residual = `sample_residual(target_nucleus, draft_nucleus)` across mismatched supports; NO new kernel. Match DFlash exactly -- it is the coherence-validated convention.

---

Both `sample_residual` and `softmax_temp_into` are private (module-local) in `speculative.rs`. They'd need to be promoted to `pub(crate)` or moved to a shared module to be reused by `mtp_spec.rs`. That's a concrete action item. Now I'll produce the final design doc.

# Design: Sampled + Tightened MTP Spec-Decode (hipfire)

Status: design (no code landed). Targets `arch_id` 5 (qwen35 dense) / 6 (A3B MoE) MTP serve path. All file:line anchors verified against HEAD (spec-graph branch).

---

## 0. Verified ground truth (recon confirmed against code)

- **MTP draft is sampled truncated, accepted untruncated.** Draft token drawn via `gpu.sample_top_p(...)` with on-device `top_k(=20) + top_p` nucleus (`mtp_spec.rs:2511-2520`), but the accept-ratio probs `p_draft`/`p_target` are gathered with `softmax_prob_gather_batched_f32` over the **full untruncated** vocab (`mtp_spec.rs:2537-2553` for draft; `mtp_spec.rs:2866-2895` for target). Accept rule is `min(1, p_t/p_d)` on those untruncated probs (`mtp_spec.rs:2899-2913`). This is the distribution bug.
- **MTP bonus is sampled from the full trunk, not the residual.** On reject/all-accept, `gpu.sample_top_p(bonus_row)` over trunk logits (`mtp_spec.rs:2917-2936`) — not `(p_target − p_draft)₊`.
- **Daemon gates sampled MTP off.** MTP routes only when `temp <= 1e-6` (`daemon.rs:7125`); the comment at `daemon.rs:12231-12232` confirms "no sampler wired." Sampled MTP code is compiled but unreachable.
- **DFlash sampled path is shipped and correct.** Leviathan accept `u*p_d < p_t` + `sample_residual` on reject (`speculative.rs:3771-3844`), GPU softmax+nucleus fast path `softmax_temp_topp_batched_into_f32` emitting per-row `tau_cut`/`Z` (`speculative.rs:3458-3490`, `3730-3755`), CACTUS optional (`speculative.rs:3771-3829`, `daemon.rs` passes `cactus_delta=0.0` for lossless serve). DFlash dispatch: `temp <= 1e-6 || (fast_sample_on && !topk_or_minp)` (`daemon.rs:7184-7188`).
- **Reusable helpers are private.** `sample_residual` (`speculative.rs:2001`) and `softmax_temp_into` (`speculative.rs:1880`) are module-private `fn` — must be promoted to reuse from `mtp_spec.rs`.

---

## 1. SAMPLED MTP — distribution-preserving fix

The fix mirrors DFlash exactly but adapted to MTP's **serial K-chain** shape (DFlash is block-parallel B-rows). The principle: **accept and residual must both live in the same truncated nucleus subspace as the draft was sampled from.**

### 1.1 Promote DFlash helpers to shared scope (prereq)

- `speculative.rs:1880` `softmax_temp_into` → `pub(crate)`
- `speculative.rs:2001` `sample_residual` → `pub(crate)`

(Or move both into a new `spec_sample.rs` shared module; lower-churn option is just widening visibility since both crates are the same `hipfire-arch-qwen35` crate — `mtp_spec.rs` can `use crate::speculative::{sample_residual, softmax_temp_into}`.)

### 1.2 Draft side — capture the nucleus mask (mtp_spec.rs:2511-2553)

The draft is already sampled from the nucleus by `gpu.sample_top_p`. Problem: `softmax_prob_gather_batched_f32` (`mtp_spec.rs:2537`) gathers `p_draft` over the **full** vocab, so it's an untruncated prob. Fix:

- Replace the per-step full-vocab draft-prob gather with a **GPU truncated softmax** that reuses DFlash's `softmax_temp_topp_batched_into_f32` on the single draft row, emitting `tau_cut`/`Z` for that position. Store `(tau_cut_k, Z_k)` per draft step in new `MtpSpecState` fields (`mtp_draft_tau: Vec<f32>`, `mtp_draft_z: Vec<f32>`), parallel to the existing `draft_probs` vec (`mtp_spec.rs:2265`).
- The draft prob then becomes `p_draft_trunc = softmax_full(logit_k)/Z_k` for tokens inside the nucleus (`logit_k ≥ tau_cut_k`), else 0. Since the sampled draft is by construction in-nucleus, `p_draft_trunc[draft] = p_draft_full[draft]/Z_k`.

### 1.3 Verify side — truncate target into the SAME nucleus (mtp_spec.rs:2866-2913)

After the trunk verify produces `verify_logits` at the K candidate positions:

- Per position k, recompute the **target** softmax **truncated to the draft's nucleus mask** (the set of vocab indices with draft-logit ≥ `tau_cut_k`). Concretely: `p_target_trunc[i] = p_target_full[i]/Z_target_k` for `i` in nucleus, else 0, where `Z_target_k = Σ_{i∈nucleus} p_target_full[i]`.
  - The cleanest implementation reuses DFlash's `softmax_temp_topp_batched_into_f32` **on the target verify logits with the draft's tau threshold**, not the target's own top_p — i.e. apply the *draft-derived* `tau_cut_k` as the cut. This requires a small kernel variant `softmax_temp_with_tau_batched_into_f32(logits, tau_cut[], out, Z[])` that takes a *given* per-row tau rather than computing it from top_p. (If a fully GPU path is too much for v1, gather the full target row D2H and apply the truncation host-side with `softmax_temp_into` + manual nucleus mask — slower but correct; optimize in Phase 2.)

### 1.4 Accept rule (replace mtp_spec.rs:2899-2913)

```
for k in 0..drafts_generated {
    p_t = p_target_trunc[k][candidate_k]   // truncated, same nucleus
    p_d = p_draft_trunc[k][candidate_k]    // truncated, same nucleus
    u   = rng.next_uniform_f32()
    if u * p_d < p_t { commit candidate_k }
    else {
        bonus = sample_residual(&p_target_trunc[k], &p_draft_trunc[k], u2)  // (1.5)
        commit bonus; break
    }
}
```

This is byte-for-byte the DFlash math (`speculative.rs:3796-3828`), now scale-correct because numerator and denominator are truncated identically.

### 1.5 Bonus on reject AND on all-accept (replace mtp_spec.rs:2917-2936)

- **On reject at k:** `sample_residual(&p_target_trunc[k], &p_draft_trunc[k], u2)` — the corrective draw from `(p_t−p_d)₊` over the nucleus. Degenerate (`Σ ≤ 0`) falls back to argmax of `p_target_trunc[k]` (already handled inside `sample_residual`, `speculative.rs:2011-2018`).
- **On all-accept:** sample the (B-1)-th bonus from `p_target_trunc[last]` (the trunk's truncated nucleus), which is what the existing `gpu.sample_top_p(bonus_row)` approximates — keep `gpu.sample_top_p` here since the bonus IS just a nucleus draw from the target (no residual involved). This one line is already correct; leave it.

### 1.6 Dispatch-gate change (daemon.rs:7110-7129)

Current MTP gate is `temp <= 1e-6` only. Change to mirror DFlash (`daemon.rs:7184-7188`):

```rust
let mtp_fast_sample_on = std::env::var("HIPFIRE_MTP_FAST_SAMPLE").ok().as_deref() != Some("0");
let topk_or_minp = top_k.map(|k| k>0).unwrap_or(false) || min_p.map(|p| p>0.0).unwrap_or(false);
if m.qwen35_mtp_head.is_some()
    && (m.arch_id == 5 || m.arch_id == 6)
    && !budgeted_thinking_needs_ar && !force_ar_chat
    && (temp <= 1e-6 || (mtp_fast_sample_on && !topk_or_minp))
{
    // route to generate_qwen35_mtp with temp/top_p threaded
}
```

Thread `temp` and `top_p` into `generate_qwen35_mtp` (`daemon.rs:5193`, `mtp_spec.rs:2148`) the same way DFlash threads them (`daemon.rs:7248-7250`). `cactus_delta = 0.0` on the serve path (lossless). `top_k`/`min_p` still route to AR (warn once, as DFlash does at `daemon.rs:7197-7210`). Update the stale comment at `daemon.rs:12231-12232` and the `mtp_spec.rs:30` "no rejection sampling" header.

---

## 2. MTP PATH TIGHTENING — bring up to DFlash standard

Ranked by impact (host-overhead reduction × frequency). Each is independent of §1 except where noted.

| # | Gap | Anchor (MTP) | DFlash reference | Concrete change |
|---|-----|--------------|------------------|-----------------|
| **T1** | **No GPU softmax fast path — draft & target probs D2H'd per step, host softmax** | `mtp_spec.rs:2511-2553`, `2866-2895` | `speculative.rs:3452-3490`, `3730-3755` | Replace per-step `softmax_prob_gather_batched_f32` + host work with `softmax_temp_topp_batched_into_f32` (GPU softmax + nucleus tau/Z). This is the **same change as §1.2/1.3** — the distribution fix and the perf fix are the same kernel swap. ~89% host-softmax reduction (DFlash measured: structured 1.45-1.67×, prose 1.23-1.27×). **Do this first; it unblocks both.** |
| **T2** | **No residual GPU/host parity — bonus from full trunk not residual** | `mtp_spec.rs:2917-2936` | `sample_residual` `speculative.rs:2001`, used `3828` | Covered by §1.5. Correctness + removes the "wrong distribution" attractor risk. |
| **T3** | **Host accept loop over D2H argmax (greedy path)** | `mtp_spec.rs:2967-2983` | `gpu.greedy_accept_from_argmax_i32` `speculative.rs:2626`, already partially wired in MTP at `mtp_spec.rs` legacy block (saw `greedy_accept_from_argmax_i32` call) | MTP already has a `mtp_gpu_greedy_accept_enabled_from_env()` GPU-accept arm in the legacy block — **flip it default-on** (it's env-gated). Removes ~100µs D2H + host loop per cycle on the greedy path. |
| **T4** | **No verify graph capture — per-cycle CPU kernel dispatch** | `mtp_spec.rs:2706-2833` | `verify_graph_launch` `speculative.rs:2600,2697` | Capture trunk-verify + lm_head + argmax (+ on sampled path, the GPU softmax) into a hipGraph keyed by `(B, dtype, arch)`. **Heed `feedback_hipgraph_kernarg_snapshot_rocm72`**: ROCm 7.2 snapshots kernarg at instantiate → strict-match-or-evict cache, no rewrite-in-place. MoE (arch 6) benefits most (~5-10ms CPU dispatch). Highest effort; do last. |
| **T5** | **Proposal graph off-by-default — first-cycle capture latency** | `mtp_spec.rs:158-180`, `2287-2366` | DFlash steady-state prewarm | Move `MtpProposalGraphPolicy::Auto` → on for q8 (header at `:165` says "token-identical"). Prewarm one throwaway cycle at request start so the first real cycle doesn't pay ~1-3ms capture. |
| **T6** | **Fresh per-request MtpSpecState alloc/free churn** | `daemon.rs:5425` `new_for_slot_with_kv_mode`, alloc `mtp_spec.rs:600` | DFlash bundle-resident (LoadedModel) | Make MTP KV + scratch **bundle-resident** in `LoadedModel` like DFlash, init once, reset (memset) per request instead of alloc/free. Removes GPU-fragmentation churn under burst load. Touches the daemon reset/checkpoint state machine → **must pass `serve-multiturn-gate.sh`** (the #462 cross-request-bleed class). |
| **T7** | **Nucleus tracking absent (top_p only via sample_top_p kernel)** | `mtp_spec.rs:2501-2520` | `topp_active` tau/Z `speculative.rs:3147,3464-3475` | Subsumed by T1 (the fast-sample kernel emits tau/Z). No separate work. |

**Sequencing note:** T1 = T2 = §1 (one kernel-swap deliverable). T3/T5 are cheap independent wins. T6 is the riskiest tightening (state lifecycle). T4 is the biggest but lowest-priority perf item and ROCm-7.2-fragile.

---

## 3. VOCAB-TIE / UPDATED-HEAD decision

**Recommendation: do NOTHING to the head. Vocab is already tied; ship §1 + §2 only. No retrain.**

Rationale (recon-confirmed):
- The MTP head has **no separate lm_head or vocab projection** — it reuses the trunk's `lm_head` and `embed_tokens` by design (`mtp_head.rs:9`, forward sig `mtp_head.rs:1223-1232` takes `lm_head_weights: &WeightTensor` from the caller; embed via `embed_lookup_into(trunk_weights, ...)` `mtp_head.rs:1481`). The `tie_word_embeddings` flag (`mtp_head.rs:104-106`) just documents the trunk aliasing.
- Vocab-size is asserted equal to trunk (`mtp_spec.rs:569-578`). Draft and target are **numerically identical post-lm_head**. There is no draft↔target vocab mismatch to fix — tying is the existing architecture, not a lever.
- The **only** vocab knob is the optional FastMTP `lm_head_draft` compression sidecar (`has_compressed_lm_head_draft`, `mtp_head_forward_compressed` `mtp_head.rs:1290`, `mtp_head_apply_lm_head_draft` `mtp_head.rs:1323`): a top-K-row compressed projection (~7.7× BW reduction). Correctness is preserved because trunk verify still uses the **full** lm_head and rejects out-of-K tokens — it trades a little τ for draft speed. This is a **separate, orthogonal** speed knob, not part of the distribution fix.

**Tradeoff table:**

| Option | τ impact | Effort | Verdict |
|--------|----------|--------|---------|
| Keep tied (status quo) + §1 sampled fix | maximal τ (draft≡target post-projection) | the §1 work | **SHIP THIS** |
| Compressed `lm_head_draft` sidecar | −small τ, +draft BW | already implemented, opt-in | optional speed knob, decouple from §1 |
| Retrain/update MTP head | speculative; only helps if current head under-accepts on sampled paths | days + GPU train | **NOT NEEDED** — under-acceptance on sampled paths is the §1 distribution bug, not a head-quality deficit. Fix the math first; only revisit a retrain if §1+§2 sampled τ still trails greedy τ by a wide margin after coherence-clean validation. |

**Clear call:** the sampled fix (§1) + tightening (§2) suffice. A retrained head is a Phase-3 *contingency*, gated on §1 sampled τ measurably underperforming after it ships coherence-clean — not a planned deliverable.

---

## 4. DFLASH KV-BLOAT

**Answer: FIXED. No open bloat. No remaining action.**

Three real bloats were all fixed on HEAD:
- S-tape ~10GB (`PrefillBatchScratch::new_opt(cap_gdn_tape)`, plain-prefill passes `false`) — **d7471243** (`qwen35.rs:5338-5555`).
- KV alloc for all 64 layers when only 16 carry KV (`*_filtered` constructors, 1-elem placeholders for LinearAttention layers) — **33fe5ab4** (`llama.rs` `alloc_k_v_filtered`, wired `speculative.rs:205-250`).
- `mq_x_rot` FWHT scratch oversize 1.74GB → chunked to `MQ_X_ROT_CHUNK_ROWS=1024` (~100MB), FWHT-linear-identical — **1b16aade** (`dflash.rs:756-792`).

Non-bloats (architectural, working as designed): verify-block stale KV slots (recycled next cycle, `speculative.rs:5656`), rejected-draft KV ring (monotonic write head, DN-state-only rollback `speculative.rs:5659`), S-tape correctly provisioned only on spec-decode (`new()` default `cap_gdn_tape=true`).

The spec-decode path is clean. **One forward-looking note for §2-T6:** when MTP KV becomes bundle-resident, apply the same `*_filtered` discipline (`33fe5ab4`) so the resident MTP KV doesn't re-introduce the all-layers allocation for the 48 non-KV LinearAttention layers.

---

## 5. PHASED PLAN

### Phase 1 — Sampled MTP fix (unblocks temp>0) — ~2-3 days
**Deliverable:** §1 + T1 + T2 (they are the same kernel-swap + accept/residual rewrite).
1. Promote `sample_residual` + `softmax_temp_into` to `pub(crate)` (§1.1).
2. Add `softmax_temp_with_tau_batched_into_f32` kernel variant (given-tau truncation) OR host-side truncation fallback for v1 (§1.3).
3. Swap MTP draft + target prob path to GPU truncated softmax, store per-step tau/Z (§1.2-1.3).
4. Rewrite accept rule (§1.4) + residual/all-accept bonus (§1.5).
5. Dispatch-gate change + thread temp/top_p (§1.6); update stale comments.
6. **Gates:** `coherence-gate.sh` + **`coherence-gate-dflash.sh` (mandatory)** + `serve-multiturn-gate.sh`. Eyeball decoded text per the three-tier thresholds.

**Exit criteria:** sampled MTP serves temp>0, passes all three coherence tiers, τ ≥ greedy-MTP τ on structured prompts, no attractor on prose.

### Phase 2 — Tightening — ~2-3 days, only after Phase 1 is coherence-clean
T3 (default-on GPU greedy accept) and T5 (proposal-graph auto) first (cheap, low-risk). Then T6 (bundle-resident state — **the risky one**, gated on `serve-multiturn-gate.sh`). T4 (verify-graph capture) last, ROCm-7.2 kernarg-snapshot-aware.

**Per-step rule:** every perf claim uses byte-identical committed prompts + md5 (CLAUDE.md prompt-structure rule); warm each A/B matrix cell; fresh-process probe via `scripts/probe_commits.sh`; any ≥5% delta investigated, not hand-waved.

### Phase 3 — Head/vocab — CONTINGENT, likely skipped
Only if Phase-1 sampled τ trails greedy by a wide margin after coherence-clean validation. Default: do nothing (§3). Compressed `lm_head_draft` sidecar is an independent opt-in speed knob, not a Phase-3 dependency.

### Hardest risks
1. **Distribution preservation (the core risk).** The truncated-accept-ratio must use the *identical* nucleus mask on draft and target. Getting the draft-derived tau applied to the target softmax wrong (e.g. using target's own top_p) silently biases the posterior — invisible in a one-shot run, surfaces as **block-level structural attractors** (the CASK m-fold τ=8.98-passed-first-128-then-emitted-1500-token-garbage class). **Mandate `coherence-gate-dflash.sh` three-tier** (first-128 unique-ratio<0.15/maxfreq>0.50 hard fail; last-128 <0.30/>0.50 hard fail; full-output 3gram>50% soft flag) on every sampled-MTP change. **Tight stddev on the τ bench is SUSPICIOUS, not reassuring.**
2. **Measure on the DAEMON, not `mtp_only_demo`.** Demos under-report ~40% and miss chatml+thinking predictability; a "fails on prose" demo verdict is a harness artifact (`feedback_test_via_daemon`, `project_dflash_prose_retrain_bound_2026_06_23`). Confirm τ through `serve`.
3. **T6 cross-request state bleed (#462 class).** Bundle-resident MTP state that doesn't reset at every checkpoint/abort/reset site bleeds into a `</think>` thinking-loop attractor — catastrophic on spec-decode. `serve-multiturn-gate.sh` is the only gate that catches it (single-request gates structurally cannot).
4. **GPU-softmax last-ULP accept flips.** DFlash notes fast-sample is distribution-parity not byte-parity (`daemon.rs:7177-7180`) — a borderline accept can rarely flip. Acceptable (validated coherent on DFlash), but means sampled MTP can't be byte-diffed against a host reference; validate via coherence + τ, not byte-parity.

**Key reusable code:** `sample_residual` (`speculative.rs:2001`), `softmax_temp_into` (`speculative.rs:1880`), `softmax_temp_topp_batched_into_f32` (GPU softmax+nucleus, `speculative.rs:3464-3475`), `greedy_accept_from_argmax_i32` (`speculative.rs:2626`), DFlash dispatch pattern (`daemon.rs:7184-7250`). The sampled-MTP fix is overwhelmingly a *port of shipped DFlash machinery* into MTP's serial-chain shape, not new algorithm work.

---

## Adversarial review (authoritative — supersedes design sections 1.3/1.4)

I now have all the ground truth I need. The recon and design doc are accurate on most facts, but I've found the one load-bearing error in the core fix. Let me confirm the key divergence point precisely by re-reading the design's §1.3 against what DFlash actually does, then verify the residual `p_draft` semantics in DFlash (does it use the truncated draft softmax `draft_softmaxes[i]`?).

From line 3828: `sample_residual(&target_probs, &draft_softmaxes[i], u2)` — where `target_probs` is target-nucleus-truncated and `draft_softmaxes[i]` is draft-nucleus-truncated. So DFlash's residual subtracts a *draft-nucleus* distribution from a *target-nucleus* distribution. These are **different supports**. This is the actual shipped, coherence-validated behavior. The design doc's §1.3 "same nucleus on both sides" is a *more* aggressive correctness claim than DFlash itself implements.

I have enough to write the verdict. No need to read further — the design's central technical claim is internally inconsistent with the very reference it cites, and that's the headline finding.

---

# Adversarial Review: Sampled + Tightened MTP Spec-Decode

**VERDICT: feasible-with-corrections**

The plan is directionally right, the recon is unusually accurate, and the "port shipped DFlash machinery" strategy is sound. But the **core distribution-fix (§1.3/§1.4) is specified WRONG** — it contradicts the very DFlash implementation it claims to mirror. As written, §1.3 would ship a *different* (and harder-to-validate) accept rule than the coherence-validated DFlash path, while the doc asserts they are "byte-for-byte the DFlash math." That contradiction must be resolved before Phase 1 starts, or you risk implementing a novel, unproven truncation scheme under the false belief that it's the battle-tested one. Everything else (vocab, KV-bloat, tightening, gating) is fundamentally sound with minor caveats.

---

## (a) Is the sampled-MTP fix actually distribution-preserving? — **NO as written; the spec is self-contradictory.**

This is the headline finding. I read the shipped DFlash accept path (`speculative.rs:3766-3858`) and the draft-build path (`speculative.rs:3463-3525`). **DFlash does NOT do what §1.3 describes.**

**What §1.3 proposes:** compute the draft's nucleus `tau_cut_k`, then truncate the *target* into the *draft's* nucleus via a new kernel `softmax_temp_with_tau_batched_into_f32(logits, tau_cut[], ...)`. It explicitly warns against "using target's own top_p."

**What DFlash actually ships:** each side computes its **own independent** nucleus.
- Draft (`speculative.rs:3463-3504`): `softmax_temp_topp_batched_into_f32` over **draft logits** → draft's own `tau/Z` → `apply_topp_trunc` → `draft_softmaxes[i]` lives in the **draft's** nucleus.
- Target (`speculative.rs:3728-3747, 3777-3778`): `softmax_temp_topp_batched_into_f32` over **target verify logits** → target's own `tau/Z` → `apply_topp_trunc` → `target_probs` lives in the **target's** nucleus.
- Accept (`3792-3793`): `p_d = draft_probs_at_drafted[i]` (draft-nucleus prob), `p_t = target_probs[t]` (target-nucleus prob). Accept `u·p_d ≤ p_t`.
- Residual (`3828`): `sample_residual(&target_probs, &draft_softmaxes[i], u2)` — subtracts the **draft-nucleus** dist from the **target-nucleus** dist. Different supports, on purpose.

So the design's §1.3 ("truncate target into the SAME nucleus as the draft, using the draft-derived tau") describes a **different algorithm** than the shipped one, while §1.4 simultaneously claims it is "byte-for-byte the DFlash math (`speculative.rs:3796-3828`)." **Both cannot be true.** The §1.4 code block (`u*p_d < p_t` + `sample_residual(p_target_trunc, p_draft_trunc)`) matches DFlash only if "trunc" means *each side's own nucleus* — i.e. exactly the thing §1.3 tells you NOT to do.

**Is the shipped DFlash approach itself lossless?** Strictly, no — neither scheme is exactly lossless vs. an *untruncated* target under top_p. Once you sample the draft from a truncated nucleus, the Chen-Leviathan lossless guarantee is already relative to the *truncated proposal*, not the full target. DFlash's choice (independent per-side nucleus + residual across mismatched supports) is the **pragmatic, coherence-validated** convention that has shipped and passed the three-tier gate. It is "distribution-parity, not byte-parity" by its own doc (`daemon.rs:7177-7180`). That is the bar MTP should hit — **match DFlash exactly**, not invent a stricter same-nucleus variant.

**Concrete consequence:** the proposed new kernel `softmax_temp_with_tau_batched_into_f32` is **unnecessary and a net negative** — it's net-new GPU code that diverges from the validated path, must be separately coherence-proven, and the doc's own §1.3 fallback ("gather target D2H, truncate host-side") would *also* be implementing the wrong (same-nucleus) rule. Drop it. Reuse `softmax_temp_topp_batched_into_f32` on the target verify logits with **the target's own top_p**, exactly as DFlash does at `speculative.rs:3733-3747`.

The design did correctly identify the *bug* (current MTP at `mtp_spec.rs:2899-2913` uses **untruncated** `p_target/p_draft` from `softmax_prob_gather_batched_f32` while drafting from a nucleus — confirmed at `mtp_spec.rs:2879-2913`, and the daemon comment at `daemon.rs:7110-7122` independently documents exactly this). It just prescribed a *third* scheme as the fix instead of the proven DFlash scheme.

---

## (b) Is "tie MTP vocab to target vocab" sound? — **The question is moot; the design's "do nothing" answer is CORRECT.**

Confirmed against `mtp_head.rs`: the MTP head has **no own lm_head or embed_tokens**. `tie_word_embeddings` is metadata (`mtp_head.rs:104-106, 167-188`); the head reuses trunk `embed_tokens` + `lm_head` by construction; vocab is asserted equal (`mtp_spec.rs:569-578`). Draft and target are numerically identical post-projection. **There is no vocab mismatch to fix, so tying cannot corrupt the draft — it's already the architecture.** §3's recommendation (ship §1+§2, no retrain, treat `lm_head_draft` compression as an orthogonal opt-in speed knob) is correct and well-justified.

One sharpening: the doc frames a retrain as a "Phase-3 contingency if sampled τ trails greedy." Given (a), the more likely cause of any sampled-τ shortfall is the **accept-rule math**, not head quality — so the contingency should be re-ordered: re-audit the truncation convention *before* ever considering a retrain. The design half-says this ("fix the math first") but should make it an explicit gate: *no retrain investigation until the accept rule is byte-identical to DFlash's and coherence-clean.*

---

## (c) Do the tightening changes risk coherence? Is the gate plan adequate? — **Gate plan is adequate; two tightenings carry real risk; one is mis-scoped.**

The coherence plan is genuinely strong: it mandates `coherence-gate-dflash.sh` three-tier, `serve-multiturn-gate.sh` for T6, daemon-not-demo measurement, and flags tight-stddev-is-suspicious. That directly targets the #1 failure mode (sampled spec-decode shipping a block-level attractor). **Adequate.** Specific risks:

- **T6 (bundle-resident MTP state) is correctly flagged as the #462-class risk** but the doc *understates* it. Moving `MtpSpecState` from per-request (`new_for_slot_with_kv_mode`, `daemon.rs:5425`) to bundle-resident touches *every* reset/checkpoint/abort site. The forward-looking note in §4 (apply `*_filtered` discipline so resident MTP KV doesn't re-bloat 48 LinearAttention layers) is a good catch but is *new scope* that the §2 effort estimate doesn't budget. T6 should be its own commit with `serve-multiturn-gate.sh` *and* the DFlash arm of it, and probably its own design pass.

- **T3 (flip GPU greedy-accept default-on) is gated on the wrong axis.** It's an env flag today presumably because it wasn't proven equal. Flipping a default is a "default-behavior change" per the project's PR-gating policy (`feedback_pr_gating_policy`) and needs a byte-identical greedy A/B before flip — not just "it exists, turn it on." Low risk but don't treat it as free.

- **T1 = §1**: correct that the perf fix and distribution fix are the same kernel swap. But since (a) shows §1's kernel spec is wrong, **T1's "~89% host-softmax reduction" depends on reusing the existing `softmax_temp_topp_batched_into_f32`, not the proposed new given-tau kernel.** Fixing (a) actually makes T1 *cheaper* (no new kernel).

- **GPU-softmax last-ULP accept flips** (risk #4 in the doc): correctly characterized. Acceptable, validate via coherence not byte-parity.

---

## (d) Is the KV-bloat conclusion correct? — **YES, with one unverified forward-looking claim.**

The three fixes (S-tape gating `d7471243`, `*_filtered` KV `33fe5ab4`, `mq_x_rot` chunking `1b16aade`) and the two non-bloats (verify-block stale KV, rejected-draft KV ring) are accurately described and the "no open bloat on HEAD" conclusion is sound for the **DFlash** path. Caveat: the §4 note that bundle-resident MTP KV must adopt `*_filtered` discipline is **a prediction about code that doesn't exist yet** (T6). It's the right instinct, but it's an action item for T6, not a verified property of HEAD. Don't let "KV-bloat is FIXED" lull T6 into skipping a fresh VRAM audit once MTP KV goes resident — MTP's KV-layer geometry must be re-checked against the resident allocation.

---

## (e) Effort realism — **Optimistic by ~1.5–2×; Phase 1 is understated, Phase 2 is mis-ordered.**

- **Phase 1 "~2-3 days":** optimistic. Removing the bogus new-kernel requirement (per (a)) *helps*, but threading temp/top_p through `generate_qwen35_mtp` → `spec_step_mtp_compressed_serial`, replacing the gather path, rewriting accept+residual, the dispatch-gate change, AND clearing all three coherence gates (each requiring a 27B target+draft on the right box, daemon-measured, byte-identical prompts with md5) is realistically **3-5 days** including the iteration when the first coherence run flags something. The recon's own note that sampled MTP is "compiled but unreachable" means there is **zero existing runtime coverage** — first activation often surfaces latent bugs.
- **Phase 2 "~2-3 days":** T6 alone is a multi-day risk item (state lifecycle + multiturn gate + VRAM audit). T4 (verify-graph capture under ROCm 7.2 kernarg-snapshot constraints) is explicitly the "highest effort, ROCm-fragile" item and per memory (`project_gemv_graph_cache_pr3`, `feedback_hipgraph_kernarg_snapshot_rocm72`) hipGraph has repeatedly measured as a **net loss** on this stack. T4 should be **demoted to research/contingency**, not a planned Phase-2 deliverable — budget it at "likely null, prove the win on fresh-probe before landing."

---

## Risks ranked by severity

1. **[BLOCKER for Phase 1] §1.3/§1.4 internal contradiction — the prescribed accept rule diverges from shipped DFlash.** The doc proposes a same-nucleus (draft-tau-on-target) truncation + a new kernel, while claiming it's "byte-for-byte DFlash." DFlash actually uses **independent per-side nuclei** (`speculative.rs:3463-3504` draft, `3728-3747` target, residual across mismatched supports at `3828`). **Correction: delete the new-kernel requirement; truncate the target with `softmax_temp_topp_batched_into_f32` using the target's own top_p, mirroring `speculative.rs:3733-3747` exactly; residual = `sample_residual(target_nucleus, draft_nucleus)`.** This makes the fix simpler, cheaper, and validated-by-precedent. Implementing the doc's literal §1.3 ships an unproven distribution.

2. **[High] T6 bundle-resident state = #462 cross-request-bleed class + un-budgeted VRAM re-audit.** Correctly flagged but under-scoped. Make it a standalone commit gated on `serve-multiturn-gate.sh` (AR + DFlash arms) with a fresh `*_filtered` MTP-KV audit. Don't bundle with cheap tightenings.

3. **[Medium] Coherence validation cost/realism.** All three gates need a 27B target+draft, daemon-measured, byte-identical committed prompts + md5. The "tight stddev is suspicious" framing is correct; enforce decoded-text eyeball per three-tier. This is the only thing standing between a silent posterior bias (from risk #1 if mis-implemented) and a shipped attractor.

4. **[Medium] T4 hipGraph capture is probably a null/regression on ROCm 7.2.** Demote to contingency; require fresh-probe proof before landing per `project_gemv_graph_cache_pr3`.

5. **[Low] T3 default-flip needs a byte-identical greedy A/B** (default-behavior change per PR-gating policy), not just "enable existing flag."

6. **[Low] Effort estimates optimistic ~1.5-2×.** Phase 1 realistically 3-5 days; Phase 2 dominated by T6.

**Net:** the strategy (port DFlash machinery into MTP's serial-chain shape, do nothing to vocab, gates as specified) is correct and worth doing. Fix the §1.3 truncation spec to *actually* match DFlash (independent per-side nucleus, no new kernel), re-scope T6 as standalone, demote T4, and the plan is sound.

**Key file:line anchors (verified against HEAD `24f28c3f`, spec-graph):**
- Bug confirmed: `crates/hipfire-arch-qwen35/src/mtp_spec.rs:2879-2913` (untruncated p_target/p_draft accept ratio)
- DFlash draft nucleus: `crates/hipfire-arch-qwen35/src/speculative.rs:3463-3504`
- DFlash target nucleus (own top_p, NOT draft-tau): `crates/hipfire-arch-qwen35/src/speculative.rs:3728-3747, 3777-3778`
- DFlash residual across mismatched supports: `crates/hipfire-arch-qwen35/src/speculative.rs:3828`
- Helpers private (promotion needed, confirmed): `speculative.rs:1880` `softmax_temp_into`, `speculative.rs:2001` `sample_residual`
- Vocab tying (no own lm_head): `crates/hipfire-arch-qwen35/src/mtp_head.rs:104-106, 167-188`; assert `mtp_spec.rs:569-578`
- Daemon gate + accurate bug doc: `crates/hipfire-runtime/examples/daemon.rs:7110-7129`
