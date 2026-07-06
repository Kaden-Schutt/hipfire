# qwen35 MTP Draft Audit (Phase 0+1) vs ds4 — findings + Phase 2 proposal

> Source: mtp-draft-audit-vs-ds4 workflow (2026-06-24), read-only audit (5 readers + synthesis).
> Phase 0 CONFIRMED on-disk (~/.hipfire/models): served qwen3.6-27b MTP = FULL-VOCAB TIED
> (qwen3.6-27b.mtp + .mq4-mtp: has_compressed_lm_head_draft=false). Only qwen3.6-35b-a3b cvs16384 is compressed.
> Phase 1 headline: the ds4-class position/RoPE/SWA off-by-one is ABSENT in qwen35 (correct cur_pos+k everywhere).

---

I have enough evidence across the five reader reports to synthesize this without additional file reads. All claims below are grounded in the cited file:line references from the reports; I distinguish code-verified from runtime-needs-confirmation throughout.

# qwen35 MTP Draft Audit — Phase 0 + Phase 1 + Phase 2

## PHASE 0 — Does the served 27B-3.6 run compressed or full-vocab tied? Tau-clunk or net win?

### Verdict: full-vocab tied is the production default; the on-disk served `.mtp` cannot be confirmed from the repo alone.

**What the code path determines (verified).** The mode switch is a single line:
`mtp_spec.rs:2181` — `let use_full_vocab = head.weights.lm_head_draft.is_none();`. `lm_head_draft` is populated only when the `.mtp` carries metadata `has_compressed_lm_head_draft: true` (`mtp_head.rs:894–935`), which is written only when `mtp_extract` was run with `--vocab-sidecar` (`mtp_extract.rs:1368–1369`). Bundled `.mq4-mtp` trailer files set `lm_head_draft: None` → full-vocab path.

**What the registry says (verified).** The registry-distributed `qwen3.6:27b` trunk is a plain `.mq4` with no bundled MTP trailer (`cli/registry.json`). The MTP head therefore comes from an on-disk `<trunk>.mtp` sidecar on k9lin. Agent memory states the head "has NO own lm_head/embed — reuses the trunk's by construction" (`mtp_head.rs:104–106`; vocab asserted == trunk at `mtp_spec.rs:569–578`), which describes the full-vocab/tied architecture as the default. The serve config label "compressed-serial" refers to the **function name** `spec_step_mtp_compressed_serial` (daemon.rs:5553) — that function serves BOTH modes via the `use_full_vocab` branch; the name does not imply compression is active.

**The unresolved fact (needs file inspection, NOT determinable from repo).** Whether the specific `.mtp` on k9lin was baked with `--vocab-sidecar`. If it was a research-era compressed sidecar (e.g. a May-bench `cvs16384`/`cvs32K` file), the served path would be `use_full_vocab = false`, cvs = 16384/32768.

**Exact check to settle it** (dump the length-prefixed JSON header of the on-disk sidecar):
```bash
python3 -c "
import struct, json
path = '/path/to/qwen3.6-27b.mq4.mtp'   # the actual sidecar on k9lin
with open(path, 'rb') as f:
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    meta = json.loads(f.read(hdr_len))
print('has_compressed_lm_head_draft:', meta.get('has_compressed_lm_head_draft'))
print('compressed_vocab_size:', meta.get('compressed_vocab_size'))
"
```
Runtime alternative: `mtp_head_apply_lm_head_draft` (`mtp_head.rs:1323`) is reached only at `mtp_spec.rs:2465` when `!use_full_vocab`. The loader's "MTP head loaded" log prints `n_embd`/`vocab` but NOT `compressed_vocab_size`, so the log alone is insufficient — add a cvs print or dump the header.

### Tau-clunk or net win?

**Compression is a tau-clunk, not a net win, in the regimes that matter — and a wash-to-slight-positive only for stock greedy.** Verified from the measurement corpus (`mtp-fastmtp-v1-v5-complete-2026-05-22.md`, `mtp-fastmtp-v6-v7-progress-2026-05-22.md`):

| Config | tok/s | τ | vs stock |
|---|---|---|---|
| K=2, compressed-serial, p_min=0.65, **stock** | 51 | 2.55 | baseline |
| K=2, compressed-serial, p_min=0.65, **v5 trained** | 40 | 2.19 | **−20%** |
| K=2, full-vocab, p_min=0.65, stock | 26.6 | 2.00 | baseline |
| K=2, full-vocab, p_min=0.65, v5 trained | 26.6 | 1.98 | parity |
| K=5, compressed-serial, p_min=0.65, stock | 47.20 | 3.00 | baseline |
| K=5, compressed-serial, p_min=0.65, v7 trained | 46.72 | 2.91 | −1% (parity only at K=5) |

- **Net positive only for stock + greedy:** draft BW is 7.5× cheaper (verified, see below), τ preserved at τ2.55 (K=2). This is the shipped baseline. The daemon durability doc (`spec-decode-durability-2026-06-23.md:148–150`) shows stock compressed-serial K=3/p_min=0.4 clearing all genres: code 1.93× (τ3.38), prose 1.31× (τ2.43), fiction 1.26–1.48× (τ2.37–2.79).
- **Tau-clunk for trained heads:** τ drops 14–20% at K=2 even after v5–v7 recursive training; gap closes to −1% only at K=5 (and that v7 used 20K samples, ~20× below FastMTP reference scale). Mechanism (verified, `mtp-fastmtp-v6-v7-progress-2026-05-22.md:80–85`): confident trained heads hit the p_min log-softmax early-exit differently in cvs-space vs full-vocab space — the cvs softmax distorts top-1 prob. The "20% gap is EXCLUSIVE to compressed-serial."
- **BW saving (verified):** draft lm_head GEMV 635 MB → 84 MB MQ4, ≈7.56× (`mtp_spec.rs:1762–1764, 2132–2137, 1868`); head comment's 7.7× is the raw 248K/32K vocab ratio (`mtp_head.rs:251–252`). **Caveat that reframes the win:** the saving applies ONLY to the draft lm_head. **Verify still runs the full 248K lm_head over all K+1 positions** (`mtp_spec.rs:2706, 2726`) — that BW is unreduced. At realistic K the verify GEMM dominates, so the 7.5× draft saving is real but not the system bottleneck.
- **Out-of-K forced reject (verified):** the target's argmax outside the cvs top-K has no row in `lm_head_draft` and no `vocab_map` slot, so the draft cannot propose it; that step rejects, and the trunk's full-vocab argmax becomes the lossless bonus token (`mtp_spec.rs:2937–2940`). Losslessness is preserved, but τ is upper-bounded by how often the target's argmax lands in the cvs set. For cvs=32K on a 248K-vocab model, 87% of vocab is unexplorable by the draft (Zipf makes this tolerable on code/structured, worse on prose tail). **No isolated out-of-K rejection-rate metric exists in the corpus** — needs measurement.

**Bottom line:** keep stock + greedy compressed-serial → net win banked. The compression only becomes load-bearing-bad when (a) you try to improve on stock with trained sidecars, or (b) you turn on sampling (Phase 1 C-D below). Whether the served file is compressed at all is the Phase 0 open item to check on k9lin.

---

## PHASE 1 — RANKED qwen35 MTP draft weaknesses

### ★★★ POSITION / RoPE / SWA OFF-BY-ONE: NONE FOUND (the ds4-class bug is ABSENT) ★★★

**This is the highest-value finding and it is a clean negative — verified.** The ds4 fix established `position = last_position + step` (`deepseek4/src/spec_decode.rs:236–246`: "position passed to mtp_forward is `last_position + step`, NOT `last_position + 1 + step`").

qwen35 **independently implements the identical correct convention.** Every dispatch path passes `cur_pos + k` (where `cur_pos == last_position` in ds4 terms):
- `mtp_spec.rs:1342, 1355` (lossy `spec_step_mtp` k=0/k>0)
- `mtp_spec.rs:2405, 2418` (compressed-serial full-vocab k=0/k>0)
- `mtp_spec.rs:2448, 2461` (device-token-chain)
- `mtp_spec.rs:2475, 2487` (compressed mode)
- `mtp_spec.rs:900` (graph-capture: `positions[k] = cur_pos + k`)
- `mtp_spec.rs:958, 973` (graph body q8)

RoPE encodes exactly `cur_pos + k` (`mtp_head.rs:1543`, `rope_partial_interleaved_f32(pos_buf=cur_pos+k)`). KV/SWA slot writes at `dispatch_pos = pos = cur_pos + k` and attends `[0..cur_pos+k+1)` (`mtp_head.rs:1574, 1592`; `seq_len_hint = pos + 1` at `mtp_head.rs:1425`). **No off-by-one, no SWA ring mis-alignment.** If qwen35 MTP acceptance lags, the cause is one of the items below — NOT position/phase/slot.

### Ranked weaknesses (mis-tuning + clunk):

**[W1] ★ Compressed-vocab path breaks the sampled accept-ratio support invariant — documented open bug.** *(verified code; impact deferred until sampling ships)*
The sampled accept gathers `p_draft` over the **compressed cvs** softmax (`mtp_spec.rs:2537–2553`, `logits_for_argmax = logits_compressed`) but gathers `p_target` over the **full 248K** verify softmax (`mtp_spec.rs:2878–2895`), then compares `min(1, p_t/p_d)` across incommensurable supports. This is the "distribution bug" (`mtp-sampled-tighten-design-2026-06-23.md:20`). It is worse than the DFlash convention (DFlash at least uses independent per-side nuclei + `sample_residual`; MTP has no truncation on the target side at all). **ds4 contrast:** ds4's spec path is greedy+grammar only — no sampled-accept divergence surface. **Impact:** would bias acceptance (over-accept when p_draft is a large cvs-prob vs tiny full-vocab p_target, or under-accept on rare tokens). **Currently unreachable in production** — daemon routes temp>0 entirely to AR (`daemon.rs:7125`, gate `temp ≤ 1e-6`), so it cannot ship silently, but it blocks any future sampled-MTP feature. **Highest-value latent bug.**

**[W2] ★ p_min early-exit operates in cvs-space, costing ~14–20% τ on trained heads.** *(verified)*
The p_min log-softmax threshold is evaluated over the cvs-compressed distribution, not the full vocab (`mtp_spec.rs:2564–2603`; comment `mtp_head.rs` / `mtp_spec.rs:2162–2178`: "softmax is over the 32K compressed vocab, which dilutes top-1 prob signal and breaks --mtp-p-min"). Confident trained heads cross the truncation threshold differently → the −20% K=2 clunk above. **ds4 contrast:** ds4 has no compressed projection; its acceptance is computed on the genuine distribution. **Impact:** −14 to −20% τ at K=2 with trained weights (the headline tau-clunk).

**[W3] Lossy embedding-arm chaining in the batched paths (`spec_step_mtp` / `spec_step_mtp_compressed`).** *(verified)*
For k>0 these pass the previous step's post-FFN activation `t_mtp_out[k-1]` as BOTH `prev_hidden` AND `next_token_embed` (`mtp_spec.rs:1353–1354, 1850–1851`; `mtp_head.rs:1476–1479` copies it into `tok_embd` then RMSNorm-enorm's it). The head was trained on discrete `embed[token_id]`, so the embedding arm is OOD. **ds4 contrast:** ds4 always re-embeds the discrete predicted token at each k (`spec_decode.rs:230–235`) — never substitutes an activation. **Mitigant (verified):** the production path `spec_step_mtp_compressed_serial` is NOT lossy — it does a proper discrete embedding lookup (`mtp_spec.rs:2372–2389`, full-vocab `2409–2420`, device-chain `2451–2463`). The hidden-state chain itself (`t_mtp_out[k-1]` as `prev_hidden`) is structurally identical to ds4's `mtp_last_hidden` and is NOT lossy in any path. **Impact:** affects only the demo-only batched paths; production is clean. Lower priority but a correctness trap if those paths are ever promoted.

**[W4] Five intertwined boolean mode-flags in one 939-line function — maintainability cliff + runtime-panic surface.** *(verified)*
`spec_step_mtp_compressed_serial` (mtp_spec.rs:2148–3087) is gated by `use_full_vocab` (2181), `use_sampling` (2258), `use_p_min` (2244), `use_device_token_chain` (2279), `use_proposal_graph` (2288), interacting multiplicatively across 40+ branch points; the draft-loop body (2368–2648) has 4 separate token-pick branches. `use_p_min + use_sampling` is enforced by a **runtime `panic!`** (line 2263), not a compile error. **ds4 contrast:** `speculative_decode_impl` (spec_decode.rs:166–416) is a single ~250-line control flow, one for-loop body (230–275), greedy+grammar only. **Impact:** latent crash on a disallowed flag combo; every new mode threads another boolean through five sites. Maintainability, not τ — but it is the substrate that hides W1/W2.

**[W5] Per-step host D2H round-trips on the p_min and sampling sub-paths.** *(verified)*
The default device-token-chain path keeps argmax on-device and does one D2H after K steps (`mtp_spec.rs:2651–2667`), but `mtp_device_token_chain_eligible_for` (198–209) **disables it when `use_sampling` or `use_p_min`**. So: p_min path = 2×4 B D2H/step (`mtp_topk_idx`+`mtp_topk_logp`, 2574–2585); sampling path = 1×4 B D2H/step for `p_draft_host` (2545–2552) because the accept decision is host-side (`r < accept_ratio`, 2902, using host RNG). **ds4 contrast:** ds4's greedy loop keeps logits→argmax→push without a forced per-step D2H barrier. **Impact:** K synchronous PCIe barriers/cycle on those paths (K=5 → 5 round-trips); latency, not τ. Structural to the host-side accept rule.

**[W6] Dead host-sample code allocating up to ~1 MB/call.** *(verified)*
`draft_logits_host` (2273–2278) allocates `vocab`-or-`cvs` floats per call when `use_sampling=true`, then is discarded (`let _ = &draft_logits_host`, 2563; comments 2560–2563, 2855–2856 confirm "unused on the GPU sampling path"). `sample_from_logits` (240, 100 lines) and `softmax_prob_at_temp` (351, 21 lines) are public but called only from the demo-only `spec_step_mtp_compressed` (1770). **Impact:** ~1 MB transient alloc on the sampled path + public dead code. Cleanup.

**[W7] Three public entry points for one production path.** *(verified)*
`spec_step_mtp` (1282), `spec_step_mtp_compressed` (1770) are demo-only (`mtp_only_demo.rs:537–579`); `spec_step_mtp_trunk_spine` (1210) is a 9-line wrapper that just calls `spec_step_mtp_compressed_serial`. Only the latter is wired into daemon.rs (5553). **ds4 contrast:** ds4 exposes one public wrapper over a private impl. **Impact:** API-surface clutter; no τ cost.

**Mis-tuning summary:** the τ losses are W2 (cvs-space p_min, −14–20% on trained heads) and W1 (sampled support mismatch, latent). The depth difference vs ds4 (qwen35 MTP = single GQA+SwiGLU block; ds4 = full MLA+MoE block with HC plumbing — `mtp_head.rs:1468–1661` vs `forward.rs:2590–...`) is an **architectural design choice, not a bug**, per the recurrence report.

---

## PHASE 2 — TIGHTENING PROPOSAL

### The load-bearing keep/drop-compression decision

**Recommendation: KEEP full-vocab tied as the production default; DROP compressed-vocab to a legacy compile-gated opt-in (do not delete yet).**

Tradeoff:
- **Drop-compression PRO:** eliminates W1 (sampled support bug) and W2 (cvs-space p_min −14–20%) at the root; unblocks proposal-graph capture (`mtp_spec.rs:190–191` — `!use_full_vocab` blocks graph eligibility, and `run_mtp_proposal_graph_body_q8` at 912 is dead for compressed); removes the vocab_map indirection (2554–2558, 2592–2595, 2632–2636); collapses W4's branch count. Full-vocab is parity-with-stock at any K (table above) and is already what bundled `.mq4-mtp` ships use.
- **Drop-compression CON:** loses the 7.5× **draft** BW saving. But (verified) verify-GEMM BW (full 248K lm_head over K+1 positions, `mtp_spec.rs:2706/2726`) dominates at realistic K, so the system-level cost is modest, not 7.5×. Net: the draft saving is real but not the bottleneck.
- **Why gate rather than delete:** the stock-greedy compressed path is a measured net win (τ2.55 @ K=2) for BW-starved configs; some on-disk sidecars may still be compressed. Keep it reachable behind a flag, off by default.

**Prerequisite gate:** before flipping any default, run the Phase 0 header dump on the k9lin `.mtp`. If the served file is already full-vocab, this is a no-op for production and pure cleanup. If it's compressed, the flip is a behavior change requiring the durability + coherence re-pin below.

### Ranked concrete fixes (with sequence)

1. **F1 — Confirm Phase 0 ground truth (BLOCKING, do first).** Dump the on-disk `.mtp` header on k9lin (snippet above). Determines whether everything downstream is cleanup or a behavior change. *No GPU needed beyond file access.*

2. **F2 — Make full-vocab the explicit production default; demote compressed to `HIPFIRE_MTP_COMPRESSED=1` opt-in.** Touches `mtp_spec.rs:2181` (and the `lm_head_draft.is_some()` auto-detect) + loader. Eliminates W2 from the default path and unblocks graph capture. **Prerequisite for the sampling fix (F5):** removes the cvs/full support mismatch substrate so sampled-accept operates on one distribution.

3. **F3 — Decompose `spec_step_mtp_compressed_serial` by mode (W4).** Split into greedy-device-chain / p_min-greedy / sampled functions; convert the line-2263 `panic!` flag-combo into a compile-time-impossible state. **Prerequisite for F5** (sampled-accept fix is far safer in an isolated function than threaded through 5 booleans).

4. **F4 — Delete dead host-sample code (W6).** Drop `draft_logits_host` alloc (2273–2278), `sample_from_logits` (240), `softmax_prob_at_temp` (351); collapse W7's 3 public entry points to one (`spec_step_mtp_trunk_spine` → private impl; demos → `pub(crate)`/`#[cfg(test)]`). Pure cleanup; do alongside F3.

5. **F5 — Fix the sampled accept-ratio support invariant (W1) — DEPENDS ON F2 + F3.** Per `mtp-sampled-tighten-design-2026-06-23.md:198–216`: use `softmax_temp_topp_batched_into_f32` independently per side (draft over its support with its own temp/top_p, target over full-vocab with its own), then `sample_residual` across them — matching the coherence-validated DFlash convention. **Must NOT ship before F2/F3.** Keep the daemon temp≤1e-6 gate (`daemon.rs:7125`) until F5 is coherence-validated.

6. **F6 (optional, lower priority) — Reduce per-step D2H on p_min/sampling paths (W5).** Move the p_min top-k compare on-device, or batch the p_draft gather. Latency-only; defer until F2–F5 land.

7. **F7 (defer/avoid) — Do NOT promote the lossy batched paths (W3).** Leave `spec_step_mtp`/`spec_step_mtp_compressed` demo-only; if ever promoted, they must first re-embed discrete tokens like the serial path.

### Validation protocol (mandatory before any τ claim)

- **Greedy MTP τ re-pin, full-vocab vs compressed, on the DAEMON (not `mtp_only_demo`).** Memory rule: demos under-report ~40% and miss chatml+thinking effects; measure on `serve` (`feedback_test_via_daemon.md`, `feedback_dflash_bench_no_chatml.md`). The corpus has **no daemon-measured same-weights/same-K compressed-vs-full A/B** — produce it. Canonical config analog: `max=256`, q8 KV (or full-vocab tied), greedy, `prompt_normalize=true`.
- **Byte-identical prompts + md5 recorded** alongside every number (CLAUDE.md: one newline swings τ 17%; agent-to-agent claims without md5 are unverifiable). Use committed prompt files, not heredocs.
- **Tight stddev is SUSPICIOUS, not reassuring** — eyeball decoded text.
- **Coherence gates:** `./scripts/coherence-gate.sh` (ChatML/AWQ qwen) on any forward/dispatch touch, AND `./scripts/coherence-gate-dflash.sh` for the spec-decode three-tier attractor battery (first-128 unique<0.15/maxfreq>0.50 hard-fail; last-128 unique<0.30 hard-fail; full-output 3gram>50% soft-flag). `./scripts/serve-multiturn-gate.sh` if any reset/checkpoint/state-bundle site is touched (F3 refactor risk).
- **F5 specifically** additionally requires the DFlash coherence gate green at temp>0 before lifting the daemon temp gate, since sampled-accept changes are exactly the attractor-prone surface.

### Verified vs needs-GPU/runtime confirmation

- **Verified from code/corpus:** the `use_full_vocab` switch and detection; full-vocab is the architectural default; the 7.5× draft-BW saving and unchanged verify-BW; the −14–20% trained-head cvs-space p_min clunk (offline `mtp_only_demo`); the sampled support-mismatch bug (code + design doc); the daemon temp≤1e-6 gate; **the absence of the ds4 position/RoPE/SWA off-by-one** (every path uses `cur_pos+k`).
- **Needs file inspection (Phase 0 open item):** whether the served k9lin `.mtp` is compressed or full-vocab — dump the header.
- **Needs GPU/daemon confirmation:** the daemon-measured compressed-vs-full A/B at matched weights/K (corpus has only demo numbers); the actual out-of-K rejection rate per genre (never isolated); any τ uplift from F2/F5 (must be re-pinned per the protocol above).
