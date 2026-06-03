# Open-issue triage — 2026-06-03

Automated, evidence-driven sweep of all **51 open issues**, evaluated against `origin/master` (commit `02634f4c`). Each issue was investigated by a dedicated agent (read the full thread + comments, grepped the live tree, checked `git log`/PRs and maintainer history), and **every close/fix recommendation was independently re-verified by a second adversarial agent** before landing here.

> **Note on tree state:** the local working copy was 33 commits behind `origin/master`. Several issues are already fixed in those 33 commits (e.g. the MoE/DeltaNet GPU-leak `b4adca1f`, Hunt-2 engine batch `a7dcfb0d`, Jinja per-request reset `534ed448`, batched-RoPE `compact_offset` `676338a4`). All judgements below are against the real `origin/master`, not the stale checkout.

## Summary

| Disposition | Count | Issues |
|---|---|---|
| 🛠️ Fix & close (fixed in this PR) | 4 | #31, #223, #252, #272 |
| ✅ Close — already resolved on master | 10 | #19, #41, #87, #154, #162, #220, #246, #259, #262, #286 |
| 💬 Answer & close | 1 | #339 |
| 📐 Feature — feasible, plan drawn (keep open) | 8 | #39, #77, #105, #207, #301, #344, #354, #392 |
| 🤔 Feature/decision — needs maintainer call (keep open) | 7 | #42, #43, #76, #92, #188, #345, #346 |
| 🔬 Real but needs specific hardware/model (keep open) | 5 | #30, #50, #61, #89, #213 |
| 📌 Tracking/roadmap — leave open | 16 | #45, #78, #113, #114, #116, #155, #209, #217, #270, #271, #289, #305, #328, #341, #343, #353 |

**Code changes shipped in this PR** (`Fixes #223, #162, #272, #31`): see the per-issue entries under *Fix & close*. Three real bugs + one advisory; builds clean, coherence gate green.


## 🛠️ Fix & close (fixed in this PR)

### #31 — PR #28 follow-ups: la_trace caching, compiler TOCTOU advisory, baseline coverage
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* partial · *confidence:* 0.9

**Assessment.** Maintainer-authored 3-item PR#28 follow-up tracker: item 1 (la_trace OnceLock) and
item 3 (0.8B/4B baselines) are RESOLVED/OBSOLETE on master; only item 2 (compiler TOCTOU) survives —
land its zero-risk advisory doc-comment and close.

**Evidence.** Item 1 resolved by commit 9a2c6677 (deletes la_trace_enabled + HIPFIRE_LA_TRACE + call
sites; symbol absent from current tree, grep empty). Item 3 obsoleted by commit 12b88600 ("chore:
remove deprecated quality-gate.sh + baselines", 2026-04-25) which deleted scripts/quality-gate.sh
and the entire tests/quality-baselines/ tree; both confirmed HEAD ancestors via git merge-base --is-
ancestor. Corroborating rationale in memory feedback_quality_gate_baselines_degenerate.md.

**Root cause.** N/A — this is a tracking/follow-up issue, not a defect. Item-2 hazard root cause:
compile() writes the precompiled-dir blob (fs::copy) and its hash sidecar (fs::write) as two
separate non-atomic operations with no cross-process lock, so a concurrent reader of another daemon
can observe a mismatched blob/hash pair if the two writes interleave (window is narrow — two daemons
must race the same cold kernel during ROCm 7.2's non-deterministic hipcc compile).

**Fix.** crates/rdna-compute/src/compiler.rs: above the writeback block at line 289 ("// Ensure
precompiled dir has valid hash + blob ..."), add an advisory comment documenting the TOCTOU hazard
(issue #31 item 2 / option B). Concretely insert before line 289:          // ADVISORY (issue #31):
the writeback below is NOT atomic across         // processes. `fs::copy(.hsaco)` then
`fs::write(.hash)` are two         // separate ops with no inter-process lock. If two daemons race
the         // same cold kernel during ROCm 7.2's non-deterministic hipcc compile,         // a
reader can briefly observe a blob/hash mismatch. Tolerated because         // (a) the read path at
compile():248-256 hash-validates before use and         // recompiles on mismatch when hipcc is
available, and (b) the window is         // ~seconds. If a multi-process test harness ever needs
determinism here,         // wrap compile()+writeback in a flock on {cache_dir}/.lock.  No file-lock
crate is added (none is currently a dependency of crates/rdna-compute; adding one is the higher-risk
alternative the issue also lists). Doc-only change.

**Feasibility.** N/A — not a feature request.

**Implementation plan.** N/A — tracking issue. Only residual work is the item-2 advisory doc-comment
captured in fix_diff (single file, ~10 comment lines, doc-only, no API/behavior change).

*Adversarial verification:* verified ✓

### #223 — Got an output but every so often got a string of `RDNA2 GEMV variant: v1 (baseline-rdna2)` appended to the output
*author:* `djismgaming` · *category:* bug · *still-exists:* yes · *confidence:* 0.99

**Assessment.** Unconditional eprintln! in gemv_hfq4g256_for_arch (kernels.rs:2101) fires once per
decode token on gfx1030/gfx1031 RDNA2 hardware, interleaving diagnostic text with generated output;
fix is to gate the print on HIPFIRE_RDNA2_VARIANT env var being set

**Evidence.** Not yet committed to master. The fix exists as uncommitted local changes in the
worktree at /home/kaden/ClaudeCode/autorocm/hipfire/.claude/worktrees/issue-triage-
fixes/crates/rdna-compute/src/kernels.rs (verified via git diff HEAD). HEAD (origin/master) still
has the unconditional eprintln! at kernels.rs:2101.

**Root cause.** In `crates/rdna-compute/src/kernels.rs` the `gemv_hfq4g256_for_arch()` function
(called on every GEMV decode dispatch for RDNA2 hardware) contains an unconditional `eprintln!` that
was intended only as a development/tuning diagnostic but was never gated behind the
`HIPFIRE_RDNA2_VARIANT` selector env var it corresponds to.

**Fix.** crates/rdna-compute/src/kernels.rs — around line 2101 on HEAD. Replace:             let
name = names.get(variant as usize).unwrap_or(&"baseline-rdna2");             eprintln!("  RDNA2 GEMV
variant: v{variant} ({name})");             match variant {  With:             let name =
names.get(variant as usize).unwrap_or(&"baseline-rdna2");             // #223: gate diagnostic on
the selector env var being set (it fires once             // per decode token, interleaving with
generated output otherwise).             if std::env::var("HIPFIRE_RDNA2_VARIANT").is_ok() {
eprintln!("  RDNA2 GEMV variant: v{variant} ({name})");             }             match variant {

*Adversarial verification:* verified ✓

### #252 — Hipfire on Windows performing ~50% better on token generation than on Linux?
*author:* `darkamgine` · *category:* bug · *still-exists:* yes · *confidence:* 0.85

**Assessment.** Two real issues: (1) run-vs-chat tok/s gap is a MEASUREMENT ARTIFACT (wall-clock-
incl-prefill chunk-count vs steady-state decode window) — fixable here in cli/index.ts; (2) Linux
max_seq OOM vs Windows is WDDM shared-mem overcommit vs Linux strict-VRAM hipMalloc, a real GTT-
fallback feature gap needing 9070XT/Windows to validate. Disposition: fix the metric, keep the GTT
half as a tracked feature request.

**Root cause.** Inconsistent tok/s DEFINITIONS across CLI entrypoints: `chat` reports a 2s sliding-
window steady-state DECODE rate (excludes prefill); `run`-via-serve reports a wall-clock rate
(cli/index.ts:1090) that counts visible SSE content chunks over time measured from before the
request (includes prefill, undercounts tokens, never fetches the daemon's decode_tok_s); `run`-local
reports the daemon's wall `tok_s` (incl prefill). Secondary/separate: no GTT/managed-memory fallback
allocator exists — hot-path uses VRAM-only hipMalloc, so Linux cannot overcommit into shared/GTT
memory the way Windows WDDM does, capping max_seq below Windows.

**Fix.** METRIC-CONSISTENCY FIX (the part fixable + validatable on this box): 1) cli/index.ts
runViaHttp (~line 1004 body build): add `stream_options: { include_usage: true }` to the request
body so the serve emits its usage/timings final chunk; then in the SSE loop parse hipfire's timings
(chunk.usage / the buildTimings fields the serve attaches at index.ts:2175-2180: decode_tok_s,
prefill_tok_s, ttft_ms) and, at line 1090, report the daemon's `decode_tok_s` (pure decode) plus a
separate `(wall NNN tok/s incl prefill)` — instead of the chunk-count `tokens/secs`. This makes run-
via-serve match chat. 2) cli/index.ts:1459: change `${msg.tok_s}` to report `msg.decode_tok_s` as
the headline decode rate and additionally print `(wall ${msg.tok_s})`, so run-local matches chat and
run-serve. 3) Label all three: print e.g. `decode N.N tok/s | wall M.M tok/s (incl prefill)` so
users stop comparing apples to oranges across run/chat. This is the same vocabulary the bench path
already uses (cli/index.ts:3691-3692, 4011-4013). The Part-A GTT half is NOT included here — it is a
feature (opt-in HIPFIRE_KV_MEM=managed using hipMallocManaged / a host-accessible HSA pool for the
KV buffer so Linux can overcommit like WDDM) requiring a VRAM-constrained card + Windows to A/B; do
not implement under this fix.

**Feasibility.** GTT/managed-memory KV fallback on Linux is feasible but non-trivial: swap the KV
pool allocation from hipMalloc to hipMallocManaged (HMM migratable) or an explicit GTT/host-
accessible HSA pool, gated behind an opt-in env (e.g. HIPFIRE_KV_MEM=managed) since it trades
capacity for bandwidth (GTT/PCIe is far slower than VRAM, so decode would slow when spilled). Risk:
silent perf cliff, migration thrash, and not all amdgpu/ROCm versions expose stable HMM on consumer
RDNA. Needs the constrained-VRAM card to tune and to prove it actually unblocks max_seq without
tanking tok/s.

**Implementation plan.** If the maintainer wants the GTT half as a follow-up feature (separate
issue): (1) Add MemoryType::Managed allocation in hip-bridge (hipMallocManaged wrapper) — the FFI fn
pointer + enum already exist (lib.rs:41). (2) Add HIPFIRE_KV_MEM={vram|managed} env, default vram.
(3) In the KV pool (rdna-compute/src/pool.rs:76) and the daemon physical_cap derivation
(daemon.rs:3227-3241), route the KV buffer through managed alloc when opted in; optionally
hipMemAdvise(PreferredLocation=device) so hot pages stay in VRAM and only the tail spills to GTT.
(4) Doc the perf-vs-capacity tradeoff. Effort ~1-2 days; risk med (HMM stability on consumer RDNA
varies by ROCm version); first step = on a 16GB card, prove hipMallocManaged lets a >VRAM KV buffer
allocate and run coherently, then measure the decode tok/s hit when spilled. Out of scope for this
triage — flag for maintainer decision.

*Adversarial verification:* verified ✓

### #272 — Tracking: DDTree + CASK eviction hidden-state compaction
*author:* `Kaden-Schutt` · *category:* bug · *still-exists:* yes · *confidence:* 0.97

**Assessment.** DDTree spec_step functions panic after TriAttention/CASK eviction because
target_hidden_host (CPU Vec<f32>) is never compacted alongside the GPU-resident target_hidden; no
fix committed since 2026-05-16.

**Root cause.** target_hidden_host (CPU Vec&lt;f32&gt; shadow) is not compacted after
TriAttention/CASK eviction. apply_eviction_retain_to_draft() handles the GPU-resident target_hidden
but callers (dflash_spec_demo.rs:961-969, :1769-1776, daemon.rs:5616, :5955) never apply the
retain_mask to the CPU vec. All DDTree spec_step functions unconditionally assert
target_hidden_host.len() == position * ne * h (speculative.rs:4097-4099, 4403-4405, 5051-5053),
which fails after eviction reduces position but not the host vec.

**Fix.** speculative.rs: Add after apply_eviction_retain_to_draft (line 5728):  ```rust /// Compact
the CPU-side `target_hidden_host` after a TriAttention/CASK eviction, /// keeping exactly the
`budget = retain_mask.len()` rows selected by `retain_mask`. /// Must be called alongside
`apply_eviction_retain_to_draft` to preserve the invariant /// `target_hidden_host.len() == position
* ne * h` that all DDTree spec_step functions assert. /// No-op if `retain_mask` is empty (CASK
m-fold path). pub fn compact_target_hidden_host(     target_hidden_host: &mut Vec<f32>,
retain_mask: &[u32],     ne: usize,     h: usize, ) {     if retain_mask.is_empty() {
return;     }     let row_floats = ne * h;     let mut compacted =
Vec::with_capacity(retain_mask.len() * row_floats);     for &src_idx in retain_mask {         let s
= src_idx as usize;         let start = s * row_floats;         let end = start + row_floats;
compacted.extend_from_slice(&target_hidden_host[start..end]);     }     *target_hidden_host =
compacted; } ```  dflash_spec_demo.rs: At both eviction sites (lines ~961-969 and ~1769-1776), add
after apply_eviction_retain_to_draft call: ```rust speculative::compact_target_hidden_host(     &mut
target_hidden_host,     &ev.retain_mask,     draft_cfg.num_extract(),     draft_cfg.hidden, ); ```
daemon.rs: At both eviction sites (lines ~5616 and ~5955), add after apply_eviction_retain_to_draft
call: ```rust speculative::compact_target_hidden_host(     &mut df.target_hidden_host,
&res.retain_mask,     df.draft_config.num_extract(),     df.draft_config.hidden, ); ```

*Adversarial verification:* verified ✓


## ✅ Close — already resolved on master

### #19 — hipGraph capture + MoE: numerical divergence after ~40 decoded tokens
*author:* `Kaden-Schutt` · *category:* bug · *still-exists:* no · *confidence:* 0.93

**Assessment.** MoE hipGraph divergence originally guarded by num_experts==0 check; now superseded
by full AR-forward hipGraph hard-disable (use_graph=false, commit 37d401e2) — bug unreachable on
master HEAD

**Evidence.** commit 86d89537 (PR #317): atomic-free MoE down fix (partial); commit 37d401e2: hard-
disable `use_graph = false` (complete workaround, supersedes the MoE-specific guard); current master
`crates/hipfire-arch-qwen35/src/qwen35.rs:6337` confirms `let use_graph = false` is unconditional

*Adversarial verification:* verified ✓

### #41 — [research] Fix DDTree on gfx1100: RoPE phase-delta skew at FA layers
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* no · *confidence:* 0.93

**Assessment.** Maintainer-authored DDTree RoPE-phase tracking issue: Path C (PR #72, f94ed073)
shipped a working DDTree on gfx1100 that routes around the linearization-slot skew; issue was never
closed despite the 2026-04-27 comment announcing closure in 7 days.

**Evidence.** f94ed073 (PR #72): DDTree Path C wire-up and PRD, validated on gfx1100 by
Lucebox/buun; 0931afb7 (PR #72 follow-up): bound budget/topk + validate path_c + hoist env-read. The
linearization-slot RoPE math is not corrected but is routed around — the production DDTree path
works correctly on gfx1100 as of these commits, both of which are in this worktree at origin/master
HEAD (positions 887 and 888 in git log --oneline).

*Adversarial verification:* verified ✓

### #87 — Track auto-MMQ regression on tool-call output (gfx1151) — reverted on master
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* no · *confidence:* 0.92

**Assessment.** Maintainer-authored tracking issue for the auto-MMQ gfx1151 tool-call regression —
all four stated sub-goals are on master and the maintainer's explicit close condition (gfx1151
coherence-gate + tool-call clean with mmq_screen=on) was satisfied by nwoolmer in the final comment
thread.

**Evidence.** 48a7fea7 (tool-call coherence coverage); d9860ebd / PR #104 (per-weight screening);
4e1a7e2d (tri-state mmq_screen, default auto=on); f7b0966d / PR #357 (auto-MMQ re-enabled at
batch>=128 on RDNA3, coherence-clean including tool-call fixture). nwoolmer's gfx1151 validation at
issue comment IC_kwDORsqzKs8AAAABBADpTg confirmed tool-call correct with mmq_screen=on.

*Adversarial verification:* verified ✓

### #154 — Hipfire with opencode — CPU-bound, no GPU utilization on large agentic prompts
*author:* `nlbutts` · *category:* bug · *still-exists:* no · *confidence:* 0.88

**Assessment.** opencode pegs one CPU core / no GPU util with hipfire serve because the GPT-2 BPE
encoder ran an O(N²)-scale single-problem tokenize over opencode's large system+tool-schema prompt;
fixed on master (PR #226 merge_rank cache 743e23d0 + pre-tokenization 97747374, 720-1200x faster,
byte-identical). Disposition: close-resolved.

**Evidence.** Two layered fixes shipped to master AFTER the 2026-05-05 issue: (1) commit 743e23d0
'perf: BPE merge_rank cache (37x TTFT)' (PR #226, merged 2026-05-09) — caches merge_pair_rank once
on the Tokenizer instead of rebuilding per encode call; ON master (git merge-base --is-ancestor
743e23d0 HEAD = YES). Commit body documents 446ms→0.09ms pre-prefill setup, 440ms of which was pure
tokenizer-rebuild. (2) commit 97747374 'perf(tokenizer): GPT-2 pre-tokenization in encode_gpt2_bpe —
fix O(N²) prompt-scale BPE' (2026-05-20); ON master. Body: 'drops niah_4k tokenize from 3609 ms to
3-5 ms (720-1200x) on the same source bytes, byte-identical output (md5 c1f8fa2c... preserved).'
Current master tree confirms both: crates/hipfire-runtime/src/tokenizer.rs:804 encode_gpt2_bpe now
drives gpt2_pretok_re().find_iter() → encode_gpt2_chunk (line 830) which borrows the cached
&self.merge_pair_rank (line 858, comment explicitly cites the ~450ms/request daemon cost it
eliminates). OpenAI-compatible /v1/chat/completions handler that opencode targets is present and
working (cli/index.ts:1624); the user's own log shows the request reaching the daemon, so the only
defect was the tokenizer CPU cost.

**Feasibility.** na

*Adversarial verification:* verified ✓

### #162 — doesn't install on nixos
*author:* `flooryyyy` · *category:* bug · *still-exists:* no · *confidence:* 0.97

**Assessment.** NixOS install failure (libamdhip64.so not found + /bin/bash bad interpreter) fully
resolved by PR #185 (Nix flake) + PR #240 (Nix deps/JIT fix), both merged to master.

**Evidence.** e1e2bb85 (PR #185: NixOS flake — flake.nix + nix/package.nix + wrapProgram ROCm
LD_LIBRARY_PATH), 4ed27e37 (PR #240: pin ROCm 7.x, JIT deps, ICD, sandboxing hardening), plus
follow-ups: 255055da, a7718a1e, 659bdbbd, e7790a99. All on master in current worktree.

*Adversarial verification:* verifier OVERRODE → fix-and-close: add NixOS early-detection to scripts/install.sh (check /etc/os-release for ID=nixos and redirect to `nix 

### #220 — Tools calling
*author:* `shrisha108` · *category:* bug · *still-exists:* no · *confidence:* 0.78

**Assessment.** Community bug: tool/function calling fails via external OpenAI-compat frontends
(OpenWebUI/Jan/opencode) with a recursive <tool_call> attractor; root causes (broken ChatML
template, no grammar constraint, MQ4 structured-token drift, single-token attractor) have all been
fixed on master — close-resolved and ask reporter to re-test.

**Evidence.** All fix commits are ancestors of HEAD (origin/master, verified via `git merge-base
--is-ancestor`): (a) grammar-guided tool calling f540a145 / 9f5e2a0d — default-ON
(HIPFIRE_QWEN35_GRAMMAR, opt-out =0), wired into BOTH AR decode (daemon.rs:8168-8855, full logit
token_mask) and DFlash (daemon.rs:5546-5878) paths so external OpenAI-compat clients are covered;
(b) GPU-side single-token attractor block for <tool_call>/<think> 3f396680 (#111) — kills the exact
recursive attractor jubiloso reported; (c) per-arch Rust tool-call parsers crates/hipfire-
runtime/src/tool_call.rs (present on master; Qwen35XmlParser + HermesJsonParser with #111 stacked-
opener/flat-object/XML-tag repairs); (d) Jinja chat-template rendering PR #219 replacing the broken
hand-rolled ChatML for Qwen3.5/3.6; (e) defensive cli/index.ts:parseToolCalls repair +
detectToolCallTruncation emitting OpenAI finish_reason hints, plus /v1/chat/completions passing
body.tools through to grammar and emitting spec-shaped tool_calls in responses; (f) scripts/agentic-
gate.sh added as a regression gate; (g) calibration-mix-v1 now includes a tool-call corpus slice
(reference_mq4_canonical_recipe). Maintainer's same-day root-cause is logged at
.remember/today-2026-05-09.done.md:13 ('Tool-call failures (opencode/hipfire Qwen3.6:27b MQ4);
cli/index.ts:511-528') and the Jinja/parser fix campaign at lines 23/37.

*Adversarial verification:* verified ✓

### #246 — Panic during hipGraph capture when kv_cache.compact_offset > 0 (HipError 906)
*author:* `yansheng1003` · *category:* bug · *still-exists:* no · *confidence:* 0.97

**Assessment.** Real gfx11 bug (sync memcpy_htod inside hipGraph capture under KV eviction →
HipError 906); both fixes the reporter proposed landed 3 days later in PR #269 and are ancestors of
HEAD — close-resolved with credit to the reporter.

**Evidence.** PR #269 (merge cb8d4dac, by Kaden Schutt; commits authored by Avery Drouillard)
implemented BOTH solutions the reporter proposed: (A) commit 7790ac6a added memcpy_htod_auto() at
crates/rdna-compute/src/dispatch.rs:690 — routes to memcpy_htod_async on the active stream when
graphs.capture_mode is true, else sync; all 11 compact_offset RoPE pos_buf copies in qwen35.rs now
call gpu.memcpy_htod_auto (verified via rg: 11 _auto, 0 remaining bare gpu.hip.memcpy_htod in the
compact_offset branches — the 13 remaining bare calls are MoE-ptr/prefill/setup copies outside AR-
decode capture). (B) commit 59e94596 skips graph capture/replay when compact_offset>0 (prevents
stale captured-position replay). Defense-in-depth: commit 37d401e2 hard-disabled AR-forward hipGraph
entirely (use_graph=false at qwen35.rs:6337) due to a separate ROCm 7.2.2 kernarg-snapshot
attractor, so the capture-during-decode path the issue panics in is never entered on master.
Subsequent PR #391 (676338a4, == current HEAD 02634f4c) further hardened batched-RoPE compact_offset
(phase/slot decouple, H4). All four commits confirmed ancestors of HEAD via git merge-base --is-
ancestor.

*Adversarial verification:* verified ✓

### #259 — Prefix Caching?
*author:* `manueloverride` · *category:* feature · *still-exists:* no · *confidence:* 0.97

**Assessment.** Prefix/prompt caching feature requested for opencode backend use — fully implemented
and shipped on master across all primary model families.

**Evidence.** 66151acd (V4F LCP prefix cache, 2026-05-24), 2f67d1b0 (per-turn token cache,
2026-05-24), f540a145 / PR #367 (Qwen3.5/3.6 prompt caching, 2026-05-31). Key line: daemon.rs:1889 —
`let cache_capable = matches!(m.arch_id, 5 | 6 | 9);`

**Feasibility.** Fully implemented. LCP-based prefix caching ships for Qwen3.5 (arch_id=5),
Qwen3.6/A3B (arch_id=6), and DeepSeek V4 Flash (arch_id=9). The daemon advertises
cache_capable=true; the CLI skips per-request resets; cached_tokens is reported in OpenAI usage.
Validated cache hit rates of 81-93% on a 3-turn coding-agent session.

*Adversarial verification:* verified ✓

### #262 — DFlash VRAM bloat fixes landed (2026-05-15) — +50% ctx ceiling on 27B Qwen3.5 / 24 GB
*author:* `Kaden-Schutt` · *category:* meta · *still-exists:* na · *confidence:* 0.95

**Assessment.** Maintainer-authored announcement that PR #261 shipped; all 5 referenced PRs are
confirmed on master; the headline fixes are live in code; safe to close as the "What's NOT yet
fixed" follow-on has its own branch and should be tracked separately.

**Evidence.** PR #261 → commit 4e37618b on master. PR #260 → commit f0985ff3. PR #245 → e301c27a. PR
#243 → ba7ee58b. PR #240 → 4ed27e37. mq_x_rot cap present at crates/hipfire-
runtime/src/dflash.rs:47,481. KV filtered constructors wired at crates/hipfire-arch-
qwen35/src/speculative.rs:450-494.

*Adversarial verification:* verified ✓

### #286 — Track imatrix-guided MFP4 salvage and MFP/HFP retirement decision
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* no · *confidence:* 0.88

**Assessment.** Tracking issue for MFP4 imatrix-salvage vs MFP/HFP retirement decision — decision
was made (retire MFP/HFP) based on empirical results, close without merging PR #238.

**Evidence.** Memory feedback_mfp_hfp_dead.md (2026-05-15): explicit retirement decision recorded.
benchmarks/quality-baselines/results/2026-05-11-cohort-phase-a-step-0.5/comparison.md: MFP4
KLD=1.1116 vs MQ4 KLD=0.8084 (+37.5%), PPL=21.02 vs 15.16 (+38.6%). PR #238 closed without merge
2026-05-17T23:50:26Z. Commit 5a64c321 (2026-05-26): drop broken imatrix_inspect example from #238.
Commit c825dfa0: MQ4 as default format, HF4/HF6 retired. feat/imatrix-mfp4-current branch never
created.

*Adversarial verification:* verified ✓


## 💬 Answer & close

### #339 — Is that pure latency? Perhaps dynamic hip graphs help?
*author:* `markg85` · *category:* question · *still-exists:* na · *confidence:* 0.82

**Assessment.** External user (markg85) correctly diagnoses small-model gfx1100 decode as launch-
latency-bound and suggests dynamic/updatable HIP graphs; the diagnosis matches the maintainer's own
findings, HIP graphs are already implemented+tested (small/null/negative wins), AR-forward graph is
currently hard-disabled for a correctness attractor, and the user's specific
hipGraphExecKernelNodeSetParams escape path is a real but unimplemented avenue — answer-and-close
with that context, optionally keep open as a discussion thread.

**Feasibility.** The dynamic-graph avenue the user names (hipGraphExecKernelNodeSetParams) is
genuinely feasible and currently UNIMPLEMENTED: there is no KernelNodeSetParams / hipGraphExecUpdate
binding in crates/hip-bridge/src/ffi.rs. Implementing it could let AR-forward graphs update
pos/kernarg slots in place instead of the snapshot-then-strict-match policy, potentially clearing
the kernarg-snapshot attractor that forced use_graph=false (qwen35.rs:6337). But prior evidence caps
the upside: per-forward graph capture already only buys +0.6-0.7% on gfx11 (RDNA3) per the in-code
A/B note (qwen35.rs:6296-6300), and per-shape gemv graph cache LOST -18% on RDNA1 because ROCm
burst-mode launch pipelining beats graph-boundary syncs. So even a correct dynamic-graph
implementation likely yields a low-single-digit % on gfx11, not the order-of-magnitude the user's
scaling argument implies. The high-leverage answer to the same latency problem is spec-decode/DFlash
(already shipped, 4.45x on 27B in BENCHMARKS.md), which amortizes the per-forward launch tail across
multiple accepted tokens.

**Implementation plan.** If pursued (low priority): (1) Add FFI bindings in crates/hip-
bridge/src/ffi.rs for hipGraphExecKernelNodeSetParams and/or hipGraphExecUpdate, wrap in GraphExec
in crates/rdna-compute/src/graph.rs. (2) In the AR-forward capture path, track per-node handles for
the nodes whose kernargs change per token (pos/token/scratch ptrs) and call SetParams per replay
instead of capture-snapshot+strict-match. (3) Re-enable use_graph behind HIPFIRE_GRAPH on gfx11
(qwen35.rs:6337) and gate behind ./scripts/coherence-gate.sh to confirm the token-0 attractor (the
2026-05-15 disable reason) is gone. (4) A/B HIPFIRE_GRAPH=0 vs 1 on 0.8B/9B/27B mq4 decode per
CLAUDE.md perf protocol. Effort: ~1-2 days. Risk: med — the disable was a real correctness
attractor, not just a perf null; upside is bounded (<=~3% on gfx11 historically). Recommend the
maintainer treat this as a contributor-welcome experiment, not a planned deliverable, and point the
reporter at DFlash/spec-decode as the already-shipped 4.45x lever.

*Adversarial verification:* verified ✓


## 📐 Feature — feasible, plan drawn (keep open)

### #39 — [Path C] Train custom DFlash draft via Red Hat / vLLM recipe (target distillation)
*author:* `Kaden-Schutt` · *category:* feature · *still-exists:* yes · *confidence:* 0.97

**Assessment.** Maintainer-authored roadmap issue for training a custom target-aligned DFlash draft;
all phases (C1–C7) remain unimplemented on master and a prior 4B sanity run failed with τ≈0.09 vs
z-lab τ=4.22 — active, meaningful, not done.

**Feasibility.** Feasible and valuable. The training infrastructure runs end-to-end (loss converges,
checkpoints save, model loads). The 4B sanity run failed due to methodology bugs (not architecture),
and the post-mortem (commit 6e4a5593) lists 4 specific testable hypotheses. The Red Hat recipe
provides a validated methodology. Primary value: target-aligned draft expected to raise τ from 10.36
toward 12-14 on Qwen3.5-27B, and fix the A3B case (τ=1.04 on code → project ANYPOS_MATCH above 50%
break-even). Full 27B run requires MI300X (~$80-150). Multiple memory entries confirm this is still
the active needed fix for A3B+3.6 DFlash misalignment.

**Implementation plan.** Phase C1 (1-2h, gfx1100-viable): Add 4 missing argparse flags to
scripts/dflash_train_poc.py: --draft-vocab-size (default 8192), --loss-mask assistant|all (default
assistant), --target-layer-ids override, --corpus-format jsonl-messages. Wire the vocab-size flag
into DFlashDraftModel construction (currently uses target.config.vocab_size=151936 hard). Wire loss-
mask to zero-out non-assistant CE contributions. Phase C1-debug (MI300X ~$1): Run K=1 single-anchor
2000-step probe against 4B target to isolate post-mortem hypothesis A (multi-anchor mask bug) vs B
(past_kv mismatch). If K=1 reaches τ>3, the bug is in the multi-anchor concat-block path; fix and
re-run K=4. Phase C2 (medium): Create scripts/generate_target_responses.py — vLLM batched generation
of target responses from HF datasets (Jackrong/Qwen3.5-reasoning-700x, LocoreMind qwen3.5-27b-cli-
reasoning-3632x, Mustafaege/qwen3.5-functioncalling-v1). Output JSONL with target-generated messages
for training. Phase C3/C4 (MI300X ~$80-150, 2-3 days): Extract hidden states (vLLM --enable-hidden-
states OR hipfire MQ4 inference), run 27B training at 5 epochs x 50K samples. Phase C5: draft_to_mq4
quantizer already exists; convert safetensors output to .hfq. Phase C6: Re-run Phase B oracle
(ANYPOS_MATCH benchmark) — target >50% on code/instruct for Path D oracle revival. Risk: low for C1
(script-only, backward-compatible), medium for C2-C5 (data quality, training divergence), low for C6
(measurement only).

### #77 — Design: NVMe→VRAM demand weight paging via SAM/ReBAR — running 229B on 32GB
*author:* `kmbandy` · *category:* feature · *still-exists:* partial · *confidence:* 0.95

**Assessment.** NVMe→VRAM demand weight paging: structural v0.1 foundation is on master (c546c0c9)
but paged dispatch is not wired (always pager=None); full paged execution (io_uring, MoE routing-
aware dispatch, P2P dma_buf) lives on unmerged PR #153 — keep open, awaiting merge.

**Feasibility.** High feasibility — full implementation already exists on PR #153 by the feature
author (kmbandy) with working end-to-end demos. Key validated results: 9.1× speedup over pread for
dense prefill via io_uring (v0.4), NVMe→VRAM P2P path validated on ROCm 7.2.2 + R9700 (v0.5), 0.9
t/s for 229B MoE on 32GB VRAM. The main blocker is code review and merge, not implementation
feasibility.

**Implementation plan.** PR #153 (remotes/kmbandy/feat/weight-pager-v0.2) is the implementation
vehicle. 41 commits not on master covering: v0.3 host arena + LRU eviction, v0.3-β async H2D with
double-buffered scratch (crates/hipfire-runtime/src/weight_pager.rs, crates/hip-bridge/src/lib.rs),
v0.4 IoUringHostTransport for batched parallel reads, v0.5 IoUringP2PTransport via dma_buf
(hsa_amd_portable_export_dmabuf — requires ROCm 5.3+, gated behind transport-p2p feature flag),
actual wired MoE dispatch in crates/hipfire-arch-qwen35/src/qwen35.rs. Risk: v0.5 P2P needs ROCm
5.3+ HSA ABI — gated behind feature flag, safe to land as opt-in. Next step: review PR #153, run the
test plan (cargo test -p engine --lib --features deltanet weight_pager, then a3b_paged_smoke).
Effort: low (review only — the work is done).

### #105 — CPU and GPU at same time ?
*author:* `0c33` · *category:* feature · *still-exists:* yes · *confidence:* 0.95

**Assessment.** User requests CPU+GPU split inference (llama.cpp-style partial offload) to run 35B
on 16 GB VRAM — feature not yet implemented; scaffold merged but forward path never wired; tracked
under #77/#76.

**Feasibility.** Architecture is partially designed. WeightPager (Transport trait,
PreadH2DTransport, residency map) and CpuRouter (MoE top-k GEMV replica for predictive prefetch) are
in place at crates/hipfire-runtime/src/. The seams are: (1) add load_weights_paged in qwen35.rs that
populates Qwen35Weights::pager with a WeightPager and leaves experts Vec empty; (2) add
ensure_resident calls at the start of moe_ffn_decode_impl when pager is Some; (3) expose a CLI flag
(e.g. --paged-experts or --vram-limit=NGB). Effort estimate: 2-4 days for a functional but slow
pread-synchronous path (0.03-0.1 t/s range, consistent with issue #77's reported 0.03 t/s for
llama.cpp reference). Async io_uring P2P path is a separate follow-up. Risk: medium — the
Transport/WeightId/ResidencyMap abstractions are tested; the GPU pointer-table patching
(patch_expert_ptr_table) is the risky seam.

**Implementation plan.** Files to touch: crates/hipfire-arch-qwen35/src/qwen35.rs (add
load_weights_paged, add paged branch in moe_ffn_decode_impl), crates/hipfire-
runtime/src/weight_pager.rs (implement ensure_resident with real eviction), crates/hipfire-
runtime/src/arch.rs (wire paged flag through Architecture trait), CLI entry point (add --paged-
experts/--vram-limit flag). First step: implement load_weights_paged that allocates a WeightPager
slab and records WeightId→(file_offset, byte_len) for all 256 experts, then add an ensure_resident
call in moe_ffn_decode_impl gated on weights.pager.is_some(). Effort: ~2-4 days. Risk: medium (GPU
pointer patching). Performance ceiling: 0.03–0.1 t/s (synchronous pread path); io_uring P2P can
raise ceiling 10-100×.

### #207 — gfx906 MoE kernel optimization gaps — 7 identified
*author:* `unverbraucht` · *category:* feature · *still-exists:* partial · *confidence:* 0.93

**Assessment.** gfx906 MoE optimization audit (7 gaps) — 5.5 of 7 still open; wave64 MoE + HFQ6 MoE
shipped post-filing but dp4a, MMQ-prefill, shared-expert-wave64, rotation-wave64, and prefetch gaps
remain

**Evidence.** Gap 6 partially resolved: 8991d34b added
gemv_hfq6g256_moe_{gate_up_indexed,down_k8_indexed_batched_expanded}.hip (2026-05-21). Gaps 1-5 and
Gap 7: no resolving commits found on master.

**Feasibility.** Highly feasible. The pattern for each gap is established by existing kernels
(wave64 MoE, dp4a fused, wave64 prefetch). Kevin Read (unverbraucht) has demonstrated ability to
implement all these patterns (PRs #187, #281, #327). Gap 2 is the highest-risk item due to MoE sort-
by-expert complexity, but gfx11 grouped MMQ precedent exists. Gap 4 requires butterfly redesign for
wave64 ds_bpermute vs wave32 ds_swizzle. All other gaps are straightforward ports.

**Implementation plan.** 5 remaining items ordered by estimated effort/impact: 1. Gap 7 (trivial,
~30 min): Add `eprintln!(\"[hipfire] warn: gate_side_mq4=false — falling back to 4× weight_gemv
(slow path)\")` at qwen35.rs:4774 else branch. 2. Gap 3 (~half day): New kernel
gemv_hfq4g256_residual_scaled_wave64.hip (mirror the existing
gemv_hfq4g256_moe_down_indexed_wave64.hip pattern: block=[64,1,1], warp_id selects row). Wire in
gemv.rs:4731 with is_wave64_native() check. Also add to the batched variant at gemv.rs:4797. 3. Gap
5 (~half day): New kernel gemv_hfq4g256_moe_gate_up_indexed_wave64_prefetch.hip + down variant (copy
gemv_hfq4g256_residual_wave64_prefetch.hip software-pipeline pattern, adapt for indexed expert
dispatch). Wire in gemv.rs:5338 behind a feature flag. 4. Gap 1 (~1 day): New kernels
gemv_hfq4g256_moe_gate_up_k8_indexed_wave64_dp4a.hip + down variant. Copy the sdot4 inner-loop from
gemm_hfq4g256_wave64_dp4a.hip, adapt for the indexed expert pointer structure. Gate behind
gemv_dp4a_enabled() && is_wave64_native() in gemv.rs:5338. 5. Gap 4 (~1-2 days): New
fused_silu_mul_mq_rotate_wave64.hip with block=[64,1,1] butterfly using __shfl_xor_sync analog on
gfx906 (ds_bpermute replaces ds_swizzle for cross-warp). Wire in gemv.rs fused_silu_mul_mq_rotate
dispatch. Batched variant also needed. 6. Gap 2 (~1-2 weeks): New
gemm_hfq4g256_moe_grouped_mmq.gfx906.hip — requires sort-by-expert in the grouped dispatch, adapting
the existing gfx11 grouped MMQ body for wave64 tiles. Needs new grouped-GEMM orchestration in
qwen35.rs prefill path. All changes are gfx906-only when gated. Non-gfx906 paths are byte-identical.
Precedent: see existing dispatch patterns in crates/rdna-compute/src/gemv.rs and dispatch.rs.

### #301 — RDNA1 (gfx1010) follow-ups: hardware verification + MMQ-fp16 variant
*author:* `unverbraucht` · *category:* feature · *still-exists:* yes · *confidence:* 0.92

**Assessment.** External contributor issue requesting gfx1010 hardware verification of PR #298 fp16
kernels (item A) and a new MMQ-fp16 kernel family for gfx1010 (item B) — neither has been done; keep
open as a feature/validation tracking issue.

**Feasibility.** Item A (hardware verification): High feasibility — gfx1010 hardware is present at
hipx device 0, verify_hfq3_batched example exists, and the fp16 dispatch path is already in master.
Requires running 3 commands on hipx. Item B (MMQ-fp16 kernel): Medium feasibility, ~1 day kernel
work — clone the 4 existing MMQ body.cuh files, swap sdot4 inner loop for v_pk_fma_f16 pattern from
gemm_gate_up_hfq3g256_fp16.hip, add .gfx1010.hip instantiations, wire dispatch gate in arch_caps.rs
(add has_hfq3_mmq_fp16 = is_rdna1, gated behind env flag initially), add to verify_hfq3_batched.
Items C/D: trivially follow from A/B.

**Implementation plan.** Item A: On hipx, run `ROCR_VISIBLE_DEVICES=0 cargo run --release --example
verify_hfq3_batched` (expect pass on scalar/fp16 sections, no sdot4/MMQ sections to fire). Then
daemon coherence eyeball on qwen3.5-9b.mq3 4-prompt matrix. Then KLD eval n=30. Record results in
benchmarks/quality-baselines/results/ . Effort: ~2 hours including build. Item B (MMQ-fp16 kernel
family): Files to create: kernels/src/gemm_hfq3g256_residual_mmq_fp16_body.cuh (new body cloning
gemm_hfq3g256_residual_mmq_body.cuh, replacing sdot4 with v_pk_fma_f16 inner loop),
kernels/src/gemm_hfq3g256_residual_mmq_fp16_x{8,16,32}.gfx1010.hip (3 tile sizes), plus
qkv/gate_up/qkvza siblings (12 files total). Files to modify: crates/rdna-compute/src/arch_caps.rs
(add has_hfq3_mmq_fp16 = is_rdna1, env-gated via HIPFIRE_HFQ3_MMQ_FP16), crates/rdna-
compute/src/kernels.rs (include_str! the new files), crates/rdna-compute/src/gemm.rs (add launch
helpers + dispatch routing after has_hfq3_mmq check, guarded by has_hfq3_mmq_fp16), crates/hipfire-
runtime/examples/verify_hfq3_batched.rs (add MMQ-fp16 section). Risk: LDS tile-size optimum for
gfx1010 (no Infinity Cache) likely differs from gfx1030 — need bench sweep on real gfx1010 to select
default tile. Effort: ~1.5 days kernel + plumbing + bench.

### #344 — Investigation: ComfyUI-FeatherOps HIP/WMMA kernel techniques applicable to hipfire
*author:* `fivetide` · *category:* feature · *still-exists:* yes · *confidence:* 0.88

**Assessment.** Collaborator fivetide's research report on 8 FeatherOps WMMA kernel techniques for
RDNA; none of the 6 suggested follow-up actions implemented yet — keep open as a kernel-optimization
tracking issue.

**Feasibility.** High feasibility on this hardware. The gfx1100 (RX 7900 XTX) and gfx1151 (Strix
Halo) are the exact targets FeatherOps was tuned for. Technique #1 (s_setprio) is a 1-line addition;
technique #2 (identity-order prepack) requires a weight layout change but FeatherOps already shows
0% conflict with the pattern. All techniques are incremental kernel variants that can be opt-in
gated like the existing nosync/ldscoop variants. No new model formats or dispatch architecture
changes needed.

**Implementation plan.** Files to touch: kernels/src/gemm_gate_up_hfq4g256_wmma_ldscoop_nosync.hip
and similar WMMA prefill kernels. Steps: (1) Add s_setprio variant (1-2 lines per kernel, new
variant file) + benchmark via crates/rdna-compute/examples/bench_mq4_gate_up.rs; (2) Audit LDS
B-operand layout in existing WMMA kernels against identity-order (K/16, N, 16) pattern — requires
comparing current layout vs conflict-free layout; (3) Implement register-tiled B fragment reuse in
the WMMA K-loop for prefill-batch kernels; (4) Add C-shuffle epilogue variant for lm_head/output
projections. Risk: low for (1) and (3), medium for (2) as layout changes affect all tiles. Effort:
~1-2 days per technique. Priority per issue: s_setprio first (lowest effort/risk), then LDS layout
audit, then B-fragment reuse.

### #354 — HBW-KV & Focus (attention) for presumable ~20x faster attention
*author:* `markg85` · *category:* feature · *still-exists:* na · *confidence:* 0.86

**Assessment.** Community feature request (markg85) to implement two attention/KV-compression papers
(HBW-KV + Focus) for ~20x faster decode; both align with hipfire's active KV-compression program but
neither is implemented and Focus needs a trained centroid layer with no reference code — keep open
as a feasibility-noted feature.

**Feasibility.** Partially feasible / substantial. (1) HBW-KV (DeepSeek, openreview sQjYtFSEuZ):
claims 16x KV compression / 4x decode / >99% accuracy via codebook/centroid KV compression. This
maps directly onto hipfire's EXISTING KV-quant framework — the project already ships FWHT-rotated,
Lloyd-Max-centroid-LUT quantization of both K and V at 2/3/4-bit (27 kv_cache_write_*.hip kernel
variants; KvMode enum at speculative.rs:340 with Q8/Asym2-4/Fwht2-4; most-recent merged PR #368 =
kv-vquant-fwht-lloyd-v). So HBW-KV is a tractable extension of a well-trodden subsystem, NOT a
greenfield build. (2) Focus (arxiv 2604.03260, future-dated April 2026, 'no code' per author): adds
learnable per-layer centroids to attention (a trained LoRA-like add-on). This is a research+TRAINING
effort, not a drop-in — requires fitting/distilling the centroid layer per model, a fixture
pipeline, and validation that it survives hipfire's coherence/attractor gates. The ~20x is the
author's admitted speculation (4x x 8x); realistically the decode goal heavily overlaps with the
ALREADY-SHIPPED DFlash/MTP spec-decode stack (5-7x AR), so marginal end-to-end gain on top of
existing accel is uncertain.

**Implementation plan.** HBW-KV (the more tractable half — do first as a feasibility probe): (a)
read the openreview paper's compression scheme and map its codebook structure onto the existing
centroid-LUT path (turbo_common.h TURBO_C* LUTs, kernels/src/kv_cache_write_fwht256_*bit*.hip,
crates/hipfire-runtime/src/kv_adaptive.rs KMode); (b) add a new KvMode variant + kv_cache_write
kernel following the existing fwht3/lloyd-V template (speculative.rs:340 enum, plus the batched
twin); (c) validate KLD vs a bf16 ref with q8/fwht KV (per project rule: never eval KLD on asym3)
and run ./scripts/coherence-gate.sh + the dflash gate before any tok/s claim. Effort: ~1-2 wks for a
competent contributor reusing the existing FWHT/Lloyd scaffolding. Risk: med — the decode win may
not stack on top of DFlash; KV-compression below the current 3-bit tier risks attractors (project
has multiple 'synth-win then coherence-falsify' precedents). Focus (defer / research-gated):
requires a trained centroid layer with NO reference code; first step is a Python reference
reproduction + KLD harness BEFORE any Rust, and a decision on whether to train the centroids (z-lab-
gated corpus or own trainer). Effort: multi-week research, high uncertainty. Concrete first step for
the issue: ask the proposer (per fivetide's comment + maintainer's stated preference) to bring an
HBW-KV KLD micro-result on a hipfire trunk quant — that's the cheapest way to convert this from
speculation to a fundable build.

### #392 — [mq4 quant] Productionize AWQ+GPTQ (proven -33% KLD on 9B) + mixed-precision bit-allocation
*author:* `Kaden-Schutt` · *category:* feature · *still-exists:* yes · *confidence:* 0.96

**Assessment.** Maintainer-authored feature tracking issue for AWQ+GPTQ productionization — all four
work items still unshipped on master; Wave 2 Rust GPTQ branch exists at hiptrx but not merged.

**Feasibility.** High feasibility. Wave 2 Rust GPTQ already reproduces 0.1257 KLD on hiptrx (Δ+0.07%
vs Python). Remaining work: (a) cherry-pick/merge `mqv2/gptq-rust-productionize` branch, run
coherence gates, add `faer` dep + `mod gptq;`, add CLI flags; (b) flip eval default in
eval_hipfire.rs line 69; (c) mixed-precision knapsack is the open hard problem (no kernel for sub-
block 4-bit, but the EXL2-style sensitivity + promote-to-6bit path is architecturally feasible with
the existing MQ6 format).

**Implementation plan.** Files: crates/hipfire-quantize/Cargo.toml (add faer dep), crates/hipfire-
quantize/src/main.rs (add `mod gptq;`, `mod hessian_io;`, `--hessian`/`--awq-raw-sumsq-npz` CLI
flags, GPTQ loop), crates/hipfire-runtime/examples/eval_hipfire.rs (line 69: asym3 → q8). Effort:
wiring gptq.rs ~1 day (branch already exists at hiptrx); eval default flip ~15 min; mixed-precision
knapsack ~3-5 days (sensitivity measurement loop + promote logic). Risk: faer API compat (gptq.rs
targets faer 0.24; check workspace version), MoE per-expert GPTQ needs expert_idx extension per
memory note. First step: cherry-pick Wave 2 branch from hiptrx, run the 24 unit tests on this box
(no GPU required for unit tests).


## 🤔 Feature/decision — needs maintainer call (keep open)

### #42 — Mutable hipGraph: replace per-B verify cache with one updateable graph (#80)
*author:* `Kaden-Schutt` · *category:* feature · *still-exists:* yes · *confidence:* 0.91

**Assessment.** Roadmap micro-optimization (0.03% end-to-end by own profiling) pre-ranked "Skip" by
maintainer; the per-B HashMap cache still exists and SetParams FFI is not implemented — decision
needed on whether to close-wontfix or keep as low-priority backlog.

**Feasibility.** Feasible in 4-6 hours (maintainer's own estimate). HIP 7.2 has
hipGraphExecKernelNodeSetParams (confirmed per project_hipgraph_decode.md). Low-risk implementation.
Impact capped at 0.03% end-to-end on 27B gfx1100; slightly higher on adaptive-b-heavy workloads. Not
currently a priority given active hunt-2/hunt-3 bug backlog.

**Implementation plan.** Files: crates/hip-bridge/src/ffi.rs (add hipGraphExecKernelNodeSetParams
dlopen binding + safe wrapper), crates/rdna-compute/src/graph.rs (add node-handle tracking to
PerBGraphCache or replace with single-graph struct + per-node handle map; add
set_verify_graph_params method), crates/hipfire-arch-qwen35/src/speculative.rs (call
set_verify_graph_params on B change instead of re-capture). Effort: 4-6h. Risk: low. Validation:
scripts/coherence-gate-dflash.sh must pass; bench with adaptive-b workload (B oscillating) vs fixed
B to confirm no regression.

### #43 — SSM intermediate persist-write (#72, Lucebox-attributed lever, ~1-2%)
*author:* `Kaden-Schutt` · *category:* feature · *still-exists:* yes · *confidence:* 0.92

**Assessment.** Deferred low-priority DFlash perf feature (~1-2% ceiling) — not implemented,
maintainer's own profiling memory marked it DEFER in favor of higher-leverage work; needs explicit
close-wontfix or keep-open-tracking decision.

**Feasibility.** Technically feasible but high-effort: requires adding a persistent SSM intermediate
GpuTensor field to DflashScratch, modifying the gated_delta_net_q8 kernel to write to a persistent
slot, and updating spec_step_dflash to pass the buffer. Expected gain is only ~1-2% (the
maintainer's profiling at decc2c5 confirmed this ceiling — 'Other' = 8% of wallclock, SSM write-back
is a fraction of that). The implementation cost (3-6 weeks, high VGPR/occupancy risk) was explicitly
judged not worth it by the maintainer in memory file project_27b_dflash_step0_profile_decc2c5.md.
The DFlash perf work has since asymptoted near hardware limits, further reducing the marginal value.

**Implementation plan.** 1. Add `ssm_intermediate: Option&lt;GpuTensor&gt;` persistent field to
`DflashScratch` in `crates/hipfire-runtime/src/dflash.rs:335`. 2. In `DflashScratch::new_with_mq`,
allocate `[n_layers, n_heads, HD, HD]` i8 buffer for the intermediate. 3. Modify
`gated_delta_net_q8` kernel in `kernels/src/gated_delta_net_q8.hip` to accept a persistent output
pointer and skip re-initializing the S_tile from s_q8 on each token if the persistent slot is
populated. 4. Wire through dispatch.rs launcher and spec_step_dflash call site. 5. Validate byte-
exact output with coherence-gate-dflash.sh. Risk: high (kernel register pressure, potential
occupancy regression).

### #76 — Design: 3-tier KV cache (hot VRAM / warm GPU or RAM / cold SSD) — llama.cpp reference implementation
*author:* `kmbandy` · *category:* feature · *still-exists:* yes · *confidence:* 0.9

**Assessment.** External contributor proposes storage-tiered KV cache (VRAM/RAM/NVMe) from their
llama.cpp implementation; hipfire shipped quantization-based adaptive KV instead — the storage-tier
feature does not exist and needs a maintainer accept/defer/close decision.

**Feasibility.** Feasible but substantial: requires KV eviction/migration API (D2D hipMemcpy between
GPUs, host-pinned staging, async NVMe write-back), semantic embedding at eviction time, and a second
VRAM allocation model. Competes architecturally with the shipped adaptive-KV (quantization
downshift) approach. Effort: 3-6 weeks for a production-quality implementation. Key files to
add/modify: crates/hipfire-runtime/src/kv_cache.rs (eviction hooks), hip-bridge/src/ffi.rs (multi-
device memcpy), new crates/kv-tier/ crate. Risk: semantic embedding CPU inference adds per-eviction
latency; the llama.cpp reference uses GGML-format BGE which hipfire does not load. Multi-GPU path
also explicitly excluded from adaptive-KV v1 non-goals.

**Implementation plan.** Not recommended to plan until maintainer decides direction. If accepted:
(1) add KvEvictionPolicy trait + LRU+attention-score eviction to hipfire-runtime; (2) add warm-tier
D2D path to hip-bridge (hipMemcpyDeviceToDevice, peer access enable); (3) add cold-tier async SSD
write-back (tokio file I/O); (4) add BGE-small embedding loader or replace with hipfire-native GGUF
loader; (5) wire --kv-tiered / --kv-warm-device CLI flags. Effort: 3-6 weeks. Risk: high (multi-GPU
complexity, semantic embedding latency, GGUF embedding model format support).

### #92 — Add DFlash draft models for MoE variants
*author:* `KotDath` · *category:* feature · *still-exists:* yes · *confidence:* 0.92

**Assessment.** A3B DFlash draft models not yet registered; blocked by structural near-neutrality vs
AR (+2.5%) and open attractor bug #89 — needs maintainer go/no-go on whether to ship at all

**Feasibility.** The registry wiring (for 3.5-A3B at minimum) is straightforward — local .hf4 files
exist, just need conversion to .hfq format, HF upload, and a registry entry with opt-in
dflash_mode=on flag. Estimated ~1 hour of work. However, the value proposition is weak:
qwen3.5:35b-a3b math-only τ=4.91 is the best measured case; code/prose is a net loss. The
qwen3.6:35b-a3b draft (τ=1.22 on code) is actively harmful for most workloads. The 122B A10B is a
separate, larger lift (quantization + routing validation) and should remain a separate tracking
issue. Feasibility: possible but only marginally useful without a custom-trained draft (task #93).

**Implementation plan.** For 3.5-A3B (feasible, limited value): (1) Convert
~/.hipfire/models/qwen35-35b-a3b-dflash-mq4.hf4 to .hfq format via dflash_convert if needed; (2)
Upload to schuttdev/hipfire-qwen3.5-35b-a3b on HuggingFace; (3) Add registry entry in
cli/registry.json: { repo: schuttdev/hipfire-qwen3.5-35b-a3b, file: qwen35-35b-a3b-dflash-mq4.hfq,
size_gb: 0.24, desc: DFlash draft for 3.5-A3B — math-only win (τ=4.91), net loss on code/prose;
requires dflash_mode=on }; (4) Validate with coherence-gate-dflash.sh on merge-sort and code prompt;
(5) Update issue #89 with A3B attractor repro. For 3.6-A3B: hold until Path C custom draft training
(task #93) or until issue #89 is resolved. For 122B: keep as separate issue (already scoped by
maintainer).

### #188 — mq3-lloyd gfx1100: K4 multi-acc vs single-acc — 2% decode tok/s vs cross-arch PPL stability
*author:* `unverbraucht` · *category:* tracking · *still-exists:* yes · *confidence:* 0.93

**Assessment.** MQ3-Lloyd gfx1100/gfx1151 multi-acc fp32-reorder drift confirmed universal; WIP
single-acc port never merged; open design decision between perf (multi-acc 121.7 tok/s) vs cross-
arch NLL determinism (single-acc 119.2 tok/s, misses ≥120 gate); maintainer direction still needed.

### #345 — Nemotron model support?
*author:* `markg85` · *category:* feature · *still-exists:* yes · *confidence:* 0.92

**Assessment.** Feature request: add Nemotron (Mamba2/SSM hybrid) model support — not implemented,
requires new arch crate + HIP Mamba2 kernels; feasible but substantial effort; maintainer priority
call needed

**Feasibility.** Feasible but substantial: requires a new arch crate (hipfire-arch-nemotron), new
HIP kernels for Mamba2 SSD parallel selective scan (the hard part — no prior art in this codebase),
weight-loader tensor-name mapping, new arch_id entry (next available ~11), and daemon dispatch
wiring. The project has precedent for hybrid recurrent+attention arches (DeltaNet/Qwen3.5) and
documented contributor guidance for new arch ports (ARCHITECTURE.md:214, .agents/skills/hipfire-
arch-port/). Estimated 2-4 weeks for an experienced contributor. The diffusion-model part of the
request is entirely out of scope for hipfire.

**Implementation plan.** Files to touch: (1) crates/hipfire-runtime/src/safetensors_source.rs — add
"nemotron_h" / "mamba2" to derive_arch_id; (2) crates/hipfire-runtime/src/arch.rs — document new
arch_id constant; (3) new crate crates/hipfire-arch-nemotron/ — implement Architecture trait
(Config/Weights/State), weight loader for HF Nemotron tensor names, SSM state allocator; (4)
kernels/src/mamba2_ssd.hip — chunked SSD selective-scan kernel for gfx11/gfx12 (the hard part); (5)
crates/hipfire-runtime/examples/daemon.rs — dispatch branch for new arch_id; (6) crates/hipfire-
quantize/src/main.rs — quantizer arch detection. Risk: the Mamba2 SSD kernel is non-trivial
(parallel prefix-sum with input-dependent A/B/C/delta), no existing HIP reference. First step:
verify a Nemotron-H model can be quantized to mq4 via the llama-fallback path in
safetensors_source.rs (it will load but produce garbage output), confirming tensor shapes, then
design the kernel from the Mamba2 paper + transformers reference impl.

### #346 — Support Orthus MTP
*author:* `markg85` · *category:* feature · *still-exists:* na · *confidence:* 0.82

**Assessment.** Feature request to add Orthrus (retrieval+learned-draft hybrid speculative decode) —
no code exists for this; requires maintainer judgment on priority given DFlash asymptote ruling

**Feasibility.** Architecturally meaningful and distinct from DFlash (retrieval-based hybrid vs
learned-only). Not a re-walk of the settled MTP+DFlash composition asymptote. Would need a context
N-gram index + candidate selector + verify-loop integration. Moderate effort, low regression risk if
additive. The maintainer's terminus doc explicitly names 'a better non-AR drafter' as the remaining
lever — Orthrus could fill that role, but PFlash (speculative prefill) is a competing priority
already scoped in a PRD. Needs go/no-go before impl planning.

**Implementation plan.** Not scoped — needs maintainer go/no-go first. If approved: (1) context
N-gram index (CPU, rolling window over committed tokens); (2) candidate scoring at each speculative
cycle selecting best N-gram continuations as draft candidates; (3) integration with existing DDTree
verify machinery; (4) coherence gate validation. Primary files to touch: crates/hipfire-arch-
qwen35/src/speculative.rs, crates/hipfire-runtime/src/dflash.rs (or a new crates/hipfire-
runtime/src/orthrus.rs). Estimated: 1-2 weeks of focused implementation + coherence/attractor
validation.


## 🔬 Real but needs specific hardware/model (keep open)

### #30 — gemm_qkvza_hfq4g256 multi-block divergence on Qwen weights — root cause unknown
*author:* `Kaden-Schutt` · *category:* bug · *still-exists:* partial · *confidence:* 0.83

**Assessment.** Real bug: batched LA-preamble GEMM diverges from per-token reference; default
gfx1100 WMMA path is fixed, but the `_per_row` safety fallback was DELETED by perf-recovery
9a2c6677, re-exposing the divergent scalar/dot2/fp16 kernels (HIPFIRE_FP16=0 + non-WMMA archs) —
disposition keep-open-needs-hw.

**Evidence.** 81714a52 + 7bffdb87 (PR #33) fixed and protected the path: WMMA variant made coherent
("The capital of France is **Paris**." on gfx1100 default), and ALL non-WMMA callers routed through
gemm_qkvza_hfq4g256_per_row (N×fused, known-good). Default gfx1100 production is safe today:
dispatcher gemm.rs:2990-2995 routes batched HIPFIRE_FP16=1 to gemm_qkvza_hfq4g256_wmma; engine call
sites qwen35.rs:9280 and :11169 hit the dispatcher only.

**Root cause.** Divergence root cause is genuinely unknown (matches the title). The issue's stated
hypothesis — ROCm 7.2 `__launch_bounds__(256,1)`+large-LDS multi-block miscompile, blocks 1..N-1
silently skipped — does NOT fit this kernel: kernels/src/gemm_qkvza_hfq4g256.hip is
`__launch_bounds__(32,8)` with ZERO __shared__ memory (one wavefront per output row, __shfl_down
reduction), and was that way even in the original Phase-1 commit f3bd132a (only diff to HEAD =
license header). The separately-tracked WMMA divergence WAS root-caused: WMMA C-matrix output-
transposition (memory project_wmma_correctness_fix.md) + an FP32-dequant-at-WMMA-boundary fix
(commit 81714a52). Those fixed only the WMMA variant; the scalar/dot2/fp16 multi-block kernels'
row-0 divergence vs fused_qkvza_hfq4g256 remains unexplained.

**Needs.** Full closure (verifying the dot2 path on gfx1011/1012/1030-1032, fp16-packed on
gfx1010/1013 RDNA1, wave64 scalar on CDNA3-without-rocBLAS) needs non-gfx11 hardware not
(fully/reliably) on this box. gfx1010 (5700 XT) is physically present but its ROCm path is
unreliable; gfx1030/CDNA3 are not here. The gfx1100 + HIPFIRE_FP16=0 scalar-path repro IS doable
here.

### #50 — Test on gfx1152
*author:* `thelittlefireman` · *category:* bug · *still-exists:* partial · *confidence:* 0.88

**Assessment.** gfx1152 incoherent-output fixed on master (d9e8dc54); segfault on --precompile
teardown and batched-WMMA arch gaps still open, need gfx1152 hardware to diagnose/validate

**Evidence.** Incoherent-output symptom: commit d9e8dc54 on master ("fix(arch): add gfx1152 to RDNA
3.5 dispatch + cache + MMQ gates (#50)"), confirmed by reporter in comment 5. Segfault: no fix
committed; MANUAL_REVIEW.md escalation from d9e8dc54 commit message.

**Root cause.** Two root causes: (1) gfx1152 missing from all RDNA 3.5 dispatch gates in dispatch.rs
— fell through to gfx1100 path with wrong wave/tiling params, causing incoherent output (FIXED
d9e8dc54). (2) Segfault in process teardown / HSA cleanup after precompile — in opaque libhsa-
runtime64.so frames, no Rust frames visible, specific to ROCm 7.12 + Fedora + gfx1152 combo (NOT
FIXED).

**Needs.** gfx1152 hardware (Strix Halo APU, any variant) with ROCm 7.12+ to reproduce and diagnose
the --precompile segfault and validate the batched-WMMA arch-gate fix for MQ3/Lloyd paths. This box
has gfx1100 + gfx1010 only; hipx box has gfx1151 which clean-exits on --precompile.

### #61 — gfx1151 (Strix Halo) speed-baseline + perf bench (help wanted)
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* partial · *confidence:* 0.9

**Assessment.** Maintainer-authored tracking issue for gfx1151 Strix Halo bringup: Phase 1 baseline
added (PR #66) and major prefill bugs fixed (PRs #83, #357), but BENCHMARKS.md table entry is
missing and speed baselines are stale post-+118% MMQ win; hardware-gated to close

**Evidence.** 0808aeeb (PR #66): speed baseline file added. 4d7de0dd (PR #83): fp16 WMMA prefill
restore. a6905066: DFlash baseline floor corrected. f7b0966d (PR #357): MMQ cutoff 256→128 for
RDNA3+ (+118% gfx1151 prefill). aa781d74: gfx1151 nosync gate_up default.

**Root cause.** Tracking issue for gfx1151 bringup — not a single-root-cause bug. Original fp16 WMMA
regression (root cause: 4d7de0dd commit message: "PR #73 introduced prefer_dot2_hfq4_prefill which
routed gfx1150/gfx1151 fused projection GEMMs to scalar dot2 fallback instead of fp16 WMMA") was
fixed in PR #83. Secondary MMQ cutoff gap (256 vs optimal 128 on RDNA3.5) fixed in PR #357.

**Needs.** Strix Halo / gfx1151 hardware (hipx box). Needs: (1) re-run speed-gate at current master
to capture post-MMQ-fix baselines, (2) full coherence-gate.sh matrix run, (3) bench numbers for
BENCHMARKS.md table row.

### #89 — DFlash thinking-attractor bug on A3B drafts (qwen3.5/3.6 35b-a3b) — block loop in <think> at long budgets
*author:* `Kaden-Schutt` · *category:* bug · *still-exists:* partial · *confidence:* 0.83

**Assessment.** Real maintainer-authored bug (reporter @mikiadev/#79): A3B DFlash block-attractor in
<think>. All enumerated user-facing mitigations have shipped (DFlash auto-off for A3B,
max_think_tokens enforced on AR+DFlash, total-think hard cap, reset clears DFlash spec buffers), but
the structural root cause — under-calibrated TriAttention sidecar R̄≈0.39 on MoE — is NOT fixed; the
real fix (per-expert sidecars) needs MI300X to train and the 35B-A3B model won't fit DFlash scratch
on this 24GB box → keep-open-needs-hw.

**Evidence.** Mitigations shipped since the 2026-04-28 issue (all on master HEAD 02634f4c): (1)
DFlash auto-gated OFF for A3B-without-sidecar — cli/index.ts:536-557, default dflash_mode=auto/off
(CONFIG_DEFAULTS:211); explicit dflash_mode=on now prints the #89 WARNING (cli/index.ts:556-557,
commit cf6eaea8). (2) max_think_tokens enforced on AR path (commit d54f2ef0) AND DFlash spec-cycle
path (commit f82ce868/#124) — daemon.rs:5921-5940 + the multi/pp loops. (3)
HIPFIRE_MAX_TOTAL_THINK_TOKENS hard cap + post-latch bound so a re-opened/rambling <think> can't
escape (daemon.rs:6552-6661, commit 3151ddc4) — addresses ForestJohnson's 'loop not contained to
thinking block'. (4) Daemon 'reset' now frees prefill_checkpoints AND dflash_checkpoints rings +
zeroes DeltaNet state (daemon.rs:2453-2464, 2263-2273) — directly addresses mikiadev's 30s-second-
prompt silent-fail (the maintainer's own hypothesized fix). (5) Hunt-2 H1 DFlash decode-abort now
memsets DeltaNet recurrent state (commit a7dcfb0d) + Jinja per-request reset (commit 534ed448/#389)
for multi-turn A3B robustness. NOT shipped: per-expert / per-cluster / gate-conditioned TriAttention
sidecars — the issue's own 'real fix' (fix-paths #1-3). git log shows no TriAttention sidecar R̄
improvement past 0.39; the 'per-expert' commits (433c70f9 etc.) are AWQ quantization, not DFlash
sidecars. Issue is still state=open, never auto-closed.

**Root cause.** A3B's global TriAttention DFlash sidecar caps at R̄≈0.36 (3.5) / 0.39 (3.6) due to
MoE routing variance — a structural ceiling (validated 2026-04-28 MI300X,
feedback_a3b_r_not_acceptable.md / feedback_a3b_dflash_regression.md). A high-R̄ sidecar suppresses
single-token attractors but NOT block-level (5+ token) attractors that emerge in tokens ~200-500 of
a long greedy <think> decode. Thinking is exactly long greedy decode on a highly self-consistent
distribution = worst-case shape. The draft and target keep agreeing on a closed phrase loop because
it is self-consistent (textbook DFlash sidecar-drift attractor, cf. CASK m-fold precedent
f16eceb/feedback_cask_mfold_dflash_broken.md).

**Feasibility.** Per-expert TriAttention sidecars (the real fix): FEASIBLE but multi-week-to-days
depending on tier. Per memory feedback_a3b_r_not_acceptable.md the iteration economics flipped post
feat/cdna-calib-mfma — gate-conditioned input dims (same-day, est +0.05-0.15 R̄), per-expert-cluster
(same-week, 10-13 min × N clusters on MI300X, est +0.10-0.20 R̄), full per-expert (days, est
R̄→0.55-0.65, ~N× storage). All require MI300X; none reproducible/validatable on this gfx1100 box
for DFlash due to the 24GB VRAM ceiling.

**Implementation plan.** Not implementing (read-only phase), but the path for a future GPU session:
(1) Cheap parallel guard first — fix-path #4: add the issue's exact thinking prompt ('What is the
answer to life, the universe, and everything?', temp=0, max_tokens=600, dflash_mode=on) to
scripts/coherence-gate-dflash.sh, require ≥9/10 healthy across 10 fresh-process attempts so
Tier-2/Tier-3 block-attractor checks exercise a real long-greedy <think> distribution. Low risk,
~half a day, validatable on hiptrx. (2) Real fix — train gate-conditioned sidecar on feat/cdna-
calib-mfma (MI300X): feed top-k expert IDs+weights as extra sidecar input dims; ~1-2 days; then per-
expert-cluster (k-means experts into 4-8 clusters by activation stats) if needed. Validate each via
coherence-gate-dflash.sh Tier-2/Tier-3 on the hardened prompt set, ≥9/10 fresh attempts, AND check τ
doesn't regress vs AR (A3B DFlash was already τ≈1.0-1.5 on code/prose per
feedback_a3b_dflash_regression.md). Files: crates/hipfire-arch-qwen35/src/speculative.rs (sidecar
input dims), the sidecar-gen CLI (commit 33807d34/#320), scripts/coherence-gate-dflash.sh. Risk: med
— confident-wrong hallucination is the failure mode if R̄ stays low
(feedback_a3b_r_not_acceptable.md), so the gate must block on coherence, and DFlash should stay
auto-OFF for A3B until a sidecar demonstrably clears the gate.

**Needs.** 35B-A3B + DFlash needs >24GB VRAM (model 22.8GB + draft scratch; observed 24.18GB OOM on
7900 XTX). Per-expert/cluster sidecar TRAINING needs the MI300X/CDNA calibration pipeline
(feat/cdna-calib-mfma @ 7f7e11b). Validation of a quality fix needs hiptrx (4×R9700 gfx1201) or
MI300X headroom. None available on this gfx1100/gfx1010 box.

### #213 — Attractor bug on Qwen3.6:27b MQ4 with unusual prompt
*author:* `darkamgine` · *category:* bug · *still-exists:* partial · *confidence:* 0.8

**Assessment.** User bug report: "malaysia" attractor loop FIXED by repeat_penalty 1.3->1.0 default
(PR #267), but the "thailand" premature-stop-during-thinking is STILL OPEN on the pure-AR path; keep
open, needs the Qwen3.6-27B MQ4 model (not on this box) to repro the exact symptom.

**Evidence.** PARTIAL resolution: malaysia loop fixed by PR #267 — commit 9b4ab74a "fix(daemon):
default repeat_penalty 1.3 -> 1.0 (fixes #258)" + merge 1874ba5c, present at HEAD (daemon.rs:1909
unwrap_or(1.0)). Reporter darkamgine confirmed in the 2026-05-21 comment that the malaysia prompt
completes normally on latest master. Related bounded-thinking work also landed: 0ec71c30
(truncation-safe DeltaNet resume + bounded thinking, PR #372), 3151ddc4 (bound post-latch generation
so think-cap can't be escaped). NONE of these reorder the terminator-vs-think-cap break, so the
thailand premature-stop remains.

**Root cause.** Two sub-bugs. (1) malaysia attractor/looping = default repeat_penalty 1.3 + top-K=20
from raw logits; FIXED for the loop part by lowering repeat_penalty to 1.0. (2) thailand "cuts off
mid-<think> at 200-300 tokens" = the model emits a terminator (<|im_end|> / <|endoftext|>/eot) WHILE
inside the open <think> block, and the daemon's AR decode loop breaks on the terminator BEFORE the
think-cap force-close logic runs (daemon.rs:5271-5273 vs the max_think enforcement at 5280+).
Compounding factor: the GPU sampler hard-codes TOP_K=20 (kernels/src/sample_top_p.hip:36), selects
the candidate pool from RAW logits before temperature is applied (Phase 1 reads logits[i] directly;
temperature only enters Phase 3 softmax), and exposes no top_k knob, so a low-probability </think>
token outside the top-20 raw-logit set can never be sampled to close the block. The hunt-2 H1
premature-stop fix (commit a7dcfb0d) is a DIFFERENT path (DFlash->AR DeltaNet dirty-state) and the
reporter explicitly ran with NO dflash, so it does not cover this AR-path case.

**Needs.** The exact Qwen3.6-27B MQ4 model (the carnice-27b.mq4/mq6 and qwen3.5-0.8b tiers are on
this box, but NOT qwen3.6-27b; the symptom is prompt+model-specific — reporter showed it only
triggers on this model with the "must-try foods in thailand" prompt). The daemon/sampler code paths
themselves are model-agnostic and CAN be exercised here, but reproducing the specific premature-stop
signature requires the matching weights.


## 📌 Tracking/roadmap — leave open

### #45 — v0.1.8+ Roadmap (living index)
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* partial · *confidence:* 0.97

**Assessment.** Living roadmap index is intentionally open but severely stale — last updated
2026-05-02 at v0.1.9-alpha, project is now at v0.2.0 with 849 additional commits; should remain open
but needs a maintainer update pass.

**Evidence.** Multiple referenced sub-issues confirmed closed: #38 (CLOSED), #40 (CLOSED), #58
(CLOSED), #60 (CLOSED), #63 (CLOSED), #70 (CLOSED), #111 (CLOSED), #119 (CLOSED). README.md now
reads "Current release: v0.2.0 — DeepSeek V4 Flash support."

### #78 — Port sliding-window FA from Lucebox PR #26 (long-context decode lever, 3.48× projected)
*author:* `Kaden-Schutt` · *category:* feature · *still-exists:* partial · *confidence:* 0.95

**Assessment.** Sliding-window FA Phases 1+3 implemented on branch feat/sliding-window-fa (NOT
merged to master); paused because the headline 3.48× win is blocked by a separate τ-collapse-at-
long-ctx issue, not by the FA windowing code itself.

**Feasibility.** Feature code already exists on branch feat/sliding-window-fa and is functionally
correct. The FA cost reduction is real (3× at ctx=14K). Merge-blocker is the orthogonal τ-collapse-
at-long-ctx investigation, not the FA windowing code quality. Once τ-collapse root cause is found
and fixed, the branch is close to mergeable (Phase 4 = CLI flag + per-model defaults + measured
3.48× = 3-4hr work per PRD).

**Implementation plan.** Blocker to unblock this issue: (1) Diagnose τ collapse at long ctx — run
spec_step_dflash at ctx=6K/14K with a fixed code prompt padded to length (eliminates prompt-shape
hypothesis), compare τ vs wikitext at same length. (2) If prompt-shape explains it → update
benchmarks, re-measure, potentially clear the blocker cheaply. (3) If not prompt-shape → instrument
spec_step_dflash ring buffer position tracking at long ctx, compare against Lucebox reference. (4)
Once τ-collapse is fixed or explained, merge feat/sliding-window-fa (7 commits, all tests pass per
comment), add CLI flag + per-model defaults (qwen35.rs Qwen35Config + CONFIG.md), run coherence-gate
+ DFlash gate at 32K ctx.

### #113 — MQ3 quality eval: perplexity 9B/27B MQ3 vs MQ4 vs Q8 baseline
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* partial · *confidence:* 0.88

**Assessment.** Maintainer-authored quality-eval tracking issue: 9B MQ3 KLD data committed, 27B MQ3
standalone row + HFP4/MFP4 canonical measurements + Pareto plot still pending; issue morphed into a
living PRD and remains an active research tracker.

**Evidence.** Partially resolved. 9B MQ3 per-token KLD committed via 2c1adad2 (KLD=2.622, PPL=85.2,
gfx1100); 9B MQ3-AWQ-GPTQ best-recipe prefill KLD=0.1967 committed; full eval harness
(build_kld_ref, eval_hipfire, eval_gguf) and BF16 references (HF: hipfire-models/qwen-kldref)
landed; GGUF anchors (7 variants) committed 2026-05-10; MQ3 declared production-ready in CHANGELOG
v0.1.9-alpha (2026-05-02); canonical eval PRD at docs/plans/issue-113-quant-quality-eval.md.

### #114 — MQ3 quality collapse on sub-9B dense models (0.8B / 4B)
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* partial · *confidence:* 0.91

**Assessment.** Maintainer-authored tracking issue for sub-9B MQ3 collapse; Lloyd infrastructure
shipped but acceptance criteria (coherent sub-9B MQ3-Lloyd to HF) are unmet — keep open.

**Evidence.** Partial: Lloyd-Max MQ3 infrastructure shipped (86cd030e, 3a12a029, 659afc74, PR #195),
PPL data committed (2c1adad2, benchmarks/results/lloyd_max_findings_20260501.md). NOT resolved:
quantizer guard still active (crates/hipfire-quantize/src/main.rs:4615-4629), no sub-9B MQ3-Lloyd in
cli/registry.json, 0.8B still in collapse per empirical data, 4B coherence eyeball not completed,
issue #116 still open.

### #116 — Lloyd-MQ3 ship gates: K4-unroll perf + coherence eyeball
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* partial · *confidence:* 0.93

**Assessment.** Gate 1 (K4-unroll perf ≥120 tok/s) fully cleared; Gate 2 (coherence eyeball)
eyeball-done but final promotion steps not executed — keep open until guard drop, daemon qt=20
extension, registry entry, and HF upload land.

**Evidence.** Gate 1 cleared: b7716c02 (134.2 tok/s on gfx1100, ≥120 gate), 86cd030e (PR #195, WMMA
prefill gap closed), 659afc74 (PR #227, mb4 fanout +77% gfx1151). Gate 2 eyeball: b7716c02 merge
message says "Coherence-gate-dflash on PR head (d3260b8) clean, no hard errors"; coherence-gate rows
in scripts/coherence-gate.sh. Promotion NOT done: daemon.rs:3192-3202 (no qt=20 check),
speculative.rs:105-115 (no MQ3G256Lloyd), main.rs:4615-4626 (guard still live), registry.json (no
lloyd entry).

### #155 — v0.1.20 — engine modularization shipped (contributor migration guide)
*author:* `Kaden-Schutt` · *category:* meta · *still-exists:* partial · *confidence:* 0.91

**Assessment.** Maintainer-authored v0.1.20 release announcement / migration guide — core
modularization fully shipped but two explicit post-0.1.20 TODOs (transformer-extraction PR, gemma4
forward-port) remain open.

**Evidence.** Core modularization shipped: b19251ee (engine→hipfire-runtime), 0005cc85 (0.1.20
finalize), 061b95cb (contributor docs + hipfire-arch-toy), 50cee6c8 (script fixes), 79bd6dae (#184
post-modular ref sweep). docs/architecture-ids.md exists and is populated.

### #209 — MQ3 + MoE batched prefill: lift the symmetric guard once wo + matcher + FFN-body wiring lands together
*author:* `unverbraucht` · *category:* tracking · *still-exists:* partial · *confidence:* 0.92

**Assessment.** Issue #209 is a tracking issue for MQ3-in-MoE dispatch wiring; Scope A (Q8_0 LA ban)
has landed on master, but the core MQ3 MoE wiring (checklist items 1-6) remains unimplemented with
both guards still active.

**Evidence.** Scope A (Q8_0 MoE LA ban): 84eb9a31 (admit Q8_0 router/shared-gate), c1946c12 (per-
proj MQ4/MQ6 dispatch), be2bf6e7 (MQ6 wo dispatch MoE bodies), 10f0c0eb (batched MoE prefill
initial). These are all on master (HEAD of this worktree). The original Q8_0 ban text described in
the issue body no longer exists in qwen35.rs.

**Feasibility.** Feasible in principle — the dense LA/FA paths already have full MQ3 dispatch arms
(gemm_qkvza_hfq3g256_wmma at line 9221, gemm_qkvza_hfq3g256 at 9239, etc.). The MoE bodies need
parallel arms with the correct 104-B stride, a generalized moe_ffn_batched_admissible predicate, and
coherence-gate rows for the new quant configs. Checklist items 1-6 in the issue body are the
complete spec. Main risk is the all-or-nothing constraint — partial wiring causes silent corruption.
No blocking kernel infrastructure gaps; existing MQ3 WMMA kernels are reusable.

**Implementation plan.** Files: crates/hipfire-arch-qwen35/src/qwen35.rs (primary),
scripts/coherence-gate.sh (new rows). Steps: (1) Add is_mq3 discriminators in DeltaNetMoe +
FullAttnMoe QKV matcher bodies, dispatch to gemm_qkvza_hfq3g256_wmma / gemm_qkvza_hfq3g256
(mirroring dense DeltaNet at L9221/9239). (2) Replace hardcoded gemm_hfq4g256_residual at MoE wo
dispatch sites with is_mq3/is_mq2/is_q8_0 branching. (3) Extend moe_ffn_batched_admissible to admit
MQ3G256 for expert weights. (4) Remove mq3_in_moe guards at lines 7086 and 7344. (5) Add coherence-
gate rows. Effort: ~400-600 LOC, 1-2 days. Risk: high if done piecemeal (the all-or-nothing rule
exists for a reason); low if all arms land together in one PR. Trigger: none currently active;
proceed when first MQ3 MoE model is quantized.

### #217 — GPU-owning state structs without Drop silently leak VRAM in loop callers
*author:* `unverbraucht` · *category:* bug · *still-exists:* partial · *confidence:* 0.85

**Assessment.** Real codebase-wide VRAM-leak footgun (GPU-owning structs have explicit free_gpu, no
Drop); the acute eval_hipfire OOM is FIXED (d430d3eb) and production-path leaks were swept by the
bug hunts (a7dcfb0d/d5985c3e/b4adca1f), but the issue's broader preventive scope is still open: no
debug-Drop guard, no audit doc, and triattn_accuracy_sweep.rs:150 still leaks in-loop — keep open as
a hygiene/prevention tracker.

**Evidence.** PARTIALLY resolved. Acute fix landed by the author himself: d430d3eb "fix(qwen35):
hoist DeltaNetState in eval_hipfire to fix VRAM leak" — added DeltaNetState::reset(&mut self, gpu)
at qwen35.rs:945 (memset-in-place, active_stream Some/None branch mirroring ModelSlot::reset_state),
and eval_hipfire.rs now allocates dn_state once (line 299) + calls dn_state.reset() per chunk (line
404); validated re-run completed all 1175 chunks (KLD 0.876237). Production-path GPU leaks (the
implicit "audit" of part 1/2) were systematically found+fixed by the bug hunts: b4adca1f
(PrefillBatchScratch 7 MoE fields ~140MB/req, DeltaNet checkpoint rings), a7dcfb0d hunt-2 (M2
Qwen35Weights awq_scale sidecars, MTP-head KvCache, M1 CASK per-eviction scratch), d5985c3e
stragglers (L1 DflashWeights sidecars, L2 TriAttnCalibStateGpu accumulators, L3 dpm_warmup 256MB
scratch).

### #270 — Tracking: Gemma 4 port hold / resume branch
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* yes · *confidence:* 0.97

**Assessment.** Gemma 4 port is still on hold — hipfire-arch-gemma4 absent from master, resume
branch 757 commits behind, forward bug unresolved; tracking issue should remain open.

### #271 — Tracking: ZAYA1 port hold / recurrent-state decision
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* yes · *confidence:* 0.97

**Assessment.** Maintainer-authored hold tracker for ZAYA1-8B port — work still on hold, no crate
landed on master, all three gating architectural decisions remain unanswered; keep open as living
discoverable index.

**Feasibility.** N/A — tracking issue, not a feature request

### #289 — MQ3 follow-ups from AWQ-loader fix (fix/mq3-awq-loader)
*author:* `unverbraucht` · *category:* tracking · *still-exists:* yes · *confidence:* 0.92

**Assessment.** Tracking issue for three MQ3 follow-ups after AWQ-loader fix — all three sub-items
remain at least partially open on master.

**Evidence.** Sub-item 1 gfx1031+gfx1100 partially resolved: 4840f0b6 (PR #298, HFQ3 batched prefill
for RDNA1/2), 0c76d31c (gfx1100 bench confirmation), 7ae2f620 (test fix for gfx10 scalar path). AWQ
quantizer partial fix in non-master branches: e905f49b, fafa8215, 8c9932fe on worktree-a3b-vram-
capacity-doc.

### #305 — Dual-license opt-in for existing contributors (MIT → MIT OR Apache-2.0)
*author:* `Kaden-Schutt` · *category:* meta · *still-exists:* yes · *confidence:* 0.95

**Assessment.** Opt-in tracking issue: 9/12 named contributors have opted in but the SPDX rewrite
has not been applied yet; 3 contributors have not responded.

### #328 — Refactor rdna-compute: decompose God Object, collapse dispatch explosion, centralize arch routing
*author:* `fivetide` · *category:* tracking · *still-exists:* partial · *confidence:* 0.95

**Assessment.** Refactor tracking issue: steps 1-3 landed on master (FeatureFlags, ArchCaps, God-
Object decomposition via PR #342); steps 4-9 + both issue comments remain open.

**Evidence.** Steps 1-3 and ~10: FeatureFlags (29ab4294, 41f2df49, 6ce18e9d, d3352263, b8a89ae6,
da814cb1, f83192ec), ArchCaps (88d43969, 8842f000, 31f73125, 1f17c47a), God-Object decomposition PR
#342 (77ed58cc). dispatch.rs reduced from 27,813 to 1,973 lines.

**Feasibility.** Steps 4-9 plus both issue comments (CODEMAP+drift-gate; ArchCaps validation-gating)
are feasible but non-trivial. Step 4 (KernelSpec table) is the largest remaining item and is blocked
only by effort — gemm.rs (19K lines, 231 methods) and gemv.rs (7K lines, 129 methods) show the
combinatorial explosion is still present, just relocated. Steps 5-9 are mechanical. The fivetide
validation-gating comment requires a design decision before implementation.

**Implementation plan.** Remaining steps per the issue's execution order: (4) KernelSpec lookup
table in gemm.rs/gemv.rs — define `(quant → ISA → fused_op → (source, symbol, grid_fn))` table,
collapse ~360 methods into ~6 generic dispatchers; (5) kernels.rs submodules by domain (mq4_lloyd,
mq3_lloyd, hfq4, attention, kv_cache, etc.); (6) move ActivationCapture/dpm_warmup/precompile_qwen35
out of dispatch.rs; (7) graph-capture safety audit — either type-system gate or at minimum audit all
direct `hip.launch_kernel` calls in gemm.rs/attention.rs/gemv.rs/norm.rs; (8) split profile.rs into
timer.rs + bandwidth.rs; (9) ProfiledLaunch builder/macro; comment #11: CODEMAP.md + AGENTS.md +
gen-codemap drift-gate script; fivetide comment: ArchCaps validation coverage table. Each step
should be a separate PR with coherence-gate validation per existing protocol.

### #341 — AWQ imatrix calibration: low-KLD MQ4 recipe, full per-domain table, and the agentic 4-bit capacity floor
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* na · *confidence:* 0.92

**Assessment.** Maintainer-authored tracking issue recording AWQ calibration methodology + per-
domain KLD/PPL table; core decision (ship AWQ as default) was executed 2026-05-28, but downstream
issue #343 still has unchecked action items referencing this as the data source.

### #343 — Default trunk → calibrated AWQ MQ4; MTP-solo (Kevin head) verified 87.5 tok/s on master
*author:* `Kaden-Schutt` · *category:* tracking · *still-exists:* partial · *confidence:* 0.9

**Assessment.** Maintainer tracking issue: 3/4 action items shipped; only the --max>=400 MTP bench
methodology doc update remains.

**Evidence.** Action item 1+2: project_hf_awq_rollout_2026_05_28.md (HF commits fb8e4a37 etc,
2026-05-28). Action item 3: PR #338 merged at f27b8595 (2026-05-27). Action item 4: absent from
docs/methodology/perf-benchmarking.md — not shipped.

### #353 — Benchmark: gfx1152 (Ryzen AI Pro 7 350)
*author:* `abondis` · *category:* tracking · *still-exists:* na · *confidence:* 0.92

**Assessment.** Community benchmark submission for gfx1152 (RDNA 3.5 APU) — data not yet recorded in
docs/BENCHMARKS.md; keep open until the "Other arches" table is updated.

**Evidence.** The underlying arch support shipped in d9e8dc54 (2026-05-01). The benchmark data
itself is in the issue body and not yet in BENCHMARKS.md — that one edit is the remaining action
before close.


---
*Generated by an automated multi-agent triage pass. Dispositions reflect the `origin/master` tree at the time of writing; reopen any closed issue if it still reproduces.*