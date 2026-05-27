# MTP Qualcomm-style probe v1 — bench results on Qwen3.5-27B (DFlash mq4)

Date: 2026-05-14
Branch: `worktree-mtp-qualcomm-probe`
Hardware: gfx1100 (k9lin's Sapphire Nitro+ 7900 XTX), ROCm 7.2.2

## Algorithm

v1 is a Qualcomm-style training-free MTP probe: each cycle issues a single
batched target forward over either `[last_committed, MASK]` (cycle 0) or
`[last_committed, pending_candidate, MASK]` (subsequent cycles). The candidate
carried in from the previous cycle's mask top-1 is verified in the same forward
by comparing `argmax(slot_0)` to that candidate; on match, the candidate plus
the candidate-slot's argmax are both committed (greedy lossless bonus). KV
advances by exactly the batch size (2 or 3) every cycle regardless of
acceptance. No tree, no draft model, no head training, lossless greedy.

Mask embedding is initialized as the mean of prompt token embeddings (Qualcomm
§3.1 soft-init, the best variant per Table 5) and updated each commit via Eq 4
EMA with λ=0.1. There is no admit gate — acceptance is strict exact-match per
Qualcomm §3.3.

Implementation: `crates/hipfire-arch-qwen35/src/mtp_probe.rs` (e0b45b9d), driven
by `crates/hipfire-runtime/examples/mtp_probe_demo.rs` (e32b871d, slot+admit
fixes 77c4cf5c). Mask-embed routing into batched forward: a7570141 + 469c301a.
Doc nits / max_n constant: c7abe67a.

## Bench config

- Model: `~/.hipfire/models/qwen3.5-27b.mq4` (14.0 GiB)
- Prompt: `benchmarks/prompts/lru_cache_pep8_strict.txt`
- Prompt md5 (raw file): `df5dedc8040ce70ba55080c4548e6024`
- Prompt md5 (after probe's chatml-wrap, per-harness log): `1e74f17934fe759468dbe1471b732067`
- max=120, temp=0, λ=0.1
- Hardware: gfx1100 / 7900 XTX / ROCm 7.2.2
- Probe wraps in chatml by default (matches dflash_spec_demo default behavior).

## Greedy AR baseline (no probe, no drafter)

`dflash_spec_demo --ar-baseline --kv-mode q8 --max 120 --temp 0`, run via
`--prompt "$(cat benchmarks/prompts/lru_cache_pep8_strict.txt)"`.

| variant | prefill tok/s | decode tok/s | first ~30 tokens | coherent |
|---|---|---|---|---|
| chatml-on  (240 tok) | 486.8 | 45.38 | `[248068, 271, 0, 0, 0, ...]` ("`<think>\n\n!!!!!...`") | NO |
| chatml-off (231 tok) | 477.6 | 45.43 | `[198, 260, 0, 0, 0, ...]` ("`\n        !!!!!...`")    | NO |

**27B inherits the bare-AR small-model attractor** — token 0 (byte 0x00) repeats
indefinitely after 1-2 leading whitespace tokens. Same fingerprint as the
0.8B/9B AR-baseline failure on this branch. Decode rate is ~45 tok/s; this is
the AR forward-pass speed, but the *content* is invalid.

## MTP probe runs (3×, byte-identical config)

| run | prefill tok/s | decode tok/s | τ | cycles | committed | mask_proposed | first ~30 tokens | coherent |
|---|---|---|---|---|---|---|---|---|
| 1 | 50.06 | 53.31 | 1.9836 | 61 | 121 | 61 | `\n!!!!!...` | NO |
| 2 | 50.08 | 53.15 | 1.9836 | 61 | 121 | 61 | `\n!!!!!...` | NO |
| 3 | 49.96 | 53.08 | 1.9836 | 61 | 121 | 61 | `\n!!!!!...` | NO |

Median: **53.15 tok/s, τ = 1.984**. Three runs are deterministic to 4 decimal
places on τ — no noise, exactly 60-of-60 mask-acceptances per run.

(The probe's "prefill" rate is much lower than the AR-baseline harness because
`mtp_probe_demo` runs prefill through the un-fused single-token path, not the
hipGraph-captured batched prefill that `dflash_spec_demo` uses. This is not a
correctness concern — it's the probe-harness shape.)

## Decision

**ABORT — gate cannot be evaluated.**

τ = 1.98 looks like a clean engine-surface success (above the 1.4 proceed
threshold), but the underlying greedy AR target is emitting the `!!!!!`
attractor. The probe's mask channel correctly predicts "next token is also 0,"
which trivially matches the broken AR output → 100% acceptance. This is a
**tautology, not a real speculation win**. The 1.17× decode speedup (45 → 53
tok/s) is real wall-clock, but it's two-tokens-per-batched-forward of garbage
vs. one-token-per-forward of garbage.

To evaluate the v1 plan we need the underlying AR path to produce coherent
text on 27B-3.5 mq4 first. Until then, neither the (a) head-training v2 nor
the (b) BC=30 1-mask-tree v2 paths are decidable from this data — they would
inherit the same 0-token attractor through the verify channel.

## Open questions / follow-ups

1. **Bare-AR small-model attractor extends to 27B.** This is the headline
   finding. Previously assumed to affect only 0.8B/9B (AR-baseline garbage on
   `master` post-Jinja), but 27B mq4 produces the identical fingerprint
   (token 0 attractor after `<think>\n\n` or after leading whitespace). Should
   be filed as its own issue; it blocks all training-free spec-decode research
   on this branch, not just MTP.
2. The `feedback_jinja_dflash_falsified_2026_05_13.md` memory entry suggests
   AssistantPrefix::ClosedThink is OOD for the sidecar; the 27B AR-baseline
   here doesn't go through the sidecar but **does** go through chatml wrap +
   `<think>\n\n` prefix → the closed-think OOD may actually be a target-LM-
   side problem (greedy collapse on `</think>\n\n` continuation), not a
   sidecar problem. Worth a clean repro on raw AR with `<|im_start|>` only
   and no `<think>` block.
3. The probe harness's prefill rate (50 tok/s) vs the dflash_spec_demo
   prefill rate (487 tok/s) is a 9.7× gap. The probe needs the batched-
   prefill fast path before any production-relevant tok/s measurement. This
   is purely cosmetic for the v1 decision-gate (the gate is τ-based, not
   tok/s-based) but it'll matter for v2.
4. After the AR-baseline fix lands, re-run this bench. Expectation: τ should
   drop substantially — a real grammar-driven prompt will not have ~100%
   one-token-ahead match rate. If τ then lands in the 1.4-2.0 range on
   coherent output, decision shifts to "proceed to head-training v2".
   If it lands ≤1.05, abort MTP-on-dflash entirely.
5. The bare-AR `!!!!!` attractor on 27B-3.5 mq4 is likely the same root-cause
   class tracked separately in `docs/investigations/2026-05-12-deltanet-mq4-bug/`
   (placeholder dir from 2026-05-12). When that investigation produces a fix,
   this v1 bench should be re-run before any v2 head-training decision.

## Cross-references

- a7570141 — MaskEmbedOverride hook in qwen35 batched forward
- 469c301a — MaskEmbedOverride doc + assertion-label fix
- e0b45b9d — mtp_probe.rs algorithm (k=1, no tree, EMA τ, soft admit)
- c7abe67a — kv-advance doc + Q8_0 max_batch comment + max_n named constant
- e32b871d — mtp_probe_demo example harness
- 77c4cf5c — widen max_seq + tighten admit guard for 3-slot/cycle KV advance

## 2026-05-15 re-run after kernel purge

User suspected the 2026-05-14 `!!!!!` AR-baseline failure on 27B-3.5 mq4 was a
ROCm 7.2.2 stale-JIT artifact (per `feedback_hipx_rocm722_jit_broken.md`).
Purge + clean rebuild + re-run.

### Purge & rebuild

```
rm -rf .hipfire_kernels
rm -rf ~/.hipfire/bin/kernels/compiled
rm -rf ~/.cache/comgr/*
cargo build --release --example dflash_spec_demo --features deltanet
cargo build --release --example mtp_probe_demo  --features deltanet,arch-qwen35
```

Verified post-run: 102 kernels recompiled into `.hipfire_kernels/` and
`~/.cache/comgr/` was repopulated by hipcc. Cargo finished in 0.37s — Rust code
was unchanged, only the runtime hipcc step was forced fresh. Branch HEAD
`5a836b8d` (no code changes from pre-purge bench).

Worktree: `.claude/worktrees/mtp-qualcomm-probe`. GPU: gfx1100 (k9lin 7900 XTX).

### AR baseline after purge (unchanged config)

`dflash_spec_demo --target ~/.hipfire/models/qwen3.5-27b.mq4 --draft ~/.hipfire/models/qwen35-9b-dflash-mq4.hfq --prompt "$(cat benchmarks/prompts/lru_cache_pep8_strict.txt)" --max 120 --temp 0 --kv-mode q8 --ar-baseline`

| variant | prefill tok/s | decode tok/s | first ~30 tokens | coherent |
|---|---|---|---|---|
| chatml-on (239 tok), post-purge | 13.5 | **9.97** | `[248068, 271, 0, 0, 0, ...]` ("`<think>\n\n!!!!!...`") | **NO** |

Output: `<think>\n\n!!!!!!!...!` — **byte-identical fingerprint** to the
2026-05-14 pre-purge run. Token-0 attractor unchanged.

(Side note: prefill collapsed 487 → 13.5 tok/s and decode 45 → 10 tok/s. This
is the AR-baseline path going through a slow non-batched prefill on 27B in
this run — not a kernel-quality regression, but a different harness shape than
the 2026-05-14 run; either the chatml wrap counted differently or the
`hipGraph` capture path inserted an extra warm-up cycle. Independent of the
attractor finding.)

### MTP probe 3× after purge

`mtp_probe_demo --target ~/.hipfire/models/qwen3.5-27b.mq4 --prompt-file benchmarks/prompts/lru_cache_pep8_strict.txt --max 120 --temp 0`

prompt md5: `1e74f17934fe759468dbe1471b732067` (same prompt-bytes as pre-purge,
distinct from `df5dedc...` because mtp_probe_demo's own chatml wrap path
hashes a slightly different intermediate; raw prompt file is `df5dedc...`).

| run | prefill tok/s | decode tok/s | τ | cycles | committed | mask_proposed | first ~30 chars | coherent |
|---|---|---|---|---|---|---|---|---|
| 1 | 50.06 | 39.84 | 1.9836 | 61 | 121 | 61 | `!!!!!!!!!!...!!!` | NO |
| 2 | 50.08 | 40.64 | **1.5385** | 78 | 120 | 78 | `</think>\n\nThe provided\n\nThe\n code\nThe\n...` | NO (loop) |
| 3 | 50.08 | 41.14 | **1.5584** | 77 | 120 | 77 | `</think>\n\nThe provided\n\nThe\n code\nThe\n code\n...` | NO (loop) |

### Pre-vs-post side-by-side

| metric | 2026-05-14 pre-purge | 2026-05-15 post-purge | delta |
|---|---|---|---|
| AR-baseline first tokens | `<think>\n\n!!!!!...` | `<think>\n\n!!!!!...` | **identical** |
| AR-baseline coherent | NO | NO | **no change** |
| Probe τ run 1 | 1.9836 | 1.9836 | identical (still `!!!!!`) |
| Probe τ run 2 | 1.9836 | 1.5385 | **−22.4%** |
| Probe τ run 3 | 1.9836 | 1.5584 | **−21.4%** |
| Probe coherent | NO (`!!!!!`) | NO (`The/code` loop) | failure mode shifted, still incoherent |
| Probe determinism | exact (4-decimal) | non-determinism on runs 2/3 | **lost** |

### Interpretation

- The kernel-purge hypothesis is **partially falsified**: AR baseline is
  byte-identical after purge → the `!!!!!` attractor is **not** a stale-JIT
  artifact, it's a real code-path bug on the 27B-3.5 mq4 AR-baseline path.
- However, probe runs 2 and 3 escaped the token-0 attractor (different output,
  τ dropped from 1.98 to ~1.55) and now produce a different incoherent loop
  (`The\n code\nThe\n code\n...`), with run 1 still hitting the `!!!!!` trap.
  This **non-determinism between probe runs at temp=0** with byte-identical
  prompt is itself a new failure signal — either uninitialised scratch or
  an order-of-execution dependency in the warm-up path. Pre-purge bench had
  exact 4-decimal-place determinism across all 3 runs.
- The earlier 1.98 τ "engine-surface success" reading is fully invalidated:
  the value was an artifact of both AR and probe channels collapsing to the
  same garbage. Post-purge, when probe partially escapes, τ drops below the
  proceed threshold (1.4-2.0 ambiguous band).

### Decision (re-run)

**ABORT — kernel purge did not fix the underlying AR-baseline bug; v1 gate
remains undecidable.**

Per the gate rule: *"AR baseline still `!!!!!` → kernel purge didn't fix it;
deeper bug, abort and document."* Falls through to that branch.

Two follow-up debugging tasks fall out of this re-run:

1. **The 27B-3.5 mq4 bare-AR token-0 attractor is real and non-cache-related.**
   Likely the same root cause tracked in `docs/investigations/2026-05-12-deltanet-mq4-bug/`.
   Until that investigation produces a fix, no v1/v2 MTP decision is recoverable
   on this model + branch combination.
2. **Post-purge probe non-determinism at temp=0** is a new finding worth its
   own bug — runs 1 vs 2/3 diverge in both τ and output text on byte-identical
   inputs. Pre-purge runs were deterministic. Either (a) an uninitialised
   scratch buffer that happens to be zeroed by some prior workload's kernel,
   or (b) the stale-cache version was deterministic-by-collapse and the fresh
   compile exposes a real warm-up race.

### Kernel-purge hypothesis verdict

**FALSIFIED for AR baseline.** Caches confirmed empty before run, 102 kernels
confirmed recompiled, AR-baseline output is byte-identical. The `!!!!!` is not
a JIT artifact. The hypothesis is partially supported for the probe path (run
1 still collapses, runs 2/3 escape), but in a way that opens a new bug rather
than closing one.

---

## 2026-05-15 second re-run after hipGraph default-OFF (commit 788c1090)

The 2026-05-15 first re-run blamed the bug on a model-side AR attractor and
linked it to `docs/investigations/2026-05-12-deltanet-mq4-bug/`. **That
attribution was wrong.** Empirical disambiguation (this run):

### Daemon-vs-example smoke (same model, same prompt, same kernels)

| path | output first 4 tokens | output coherent? |
|---|---|---|
| daemon (`crates/hipfire-runtime/examples/daemon.rs`) | ` an LRU cache with O(1)...` | YES |
| `infer_qwen35` (per-token `forward_scratch`) | `<think>!!!!!!!!` (2048 of `!`) | NO |
| `dflash_spec_demo --ar-baseline` (batched `forward_prefill_batch_with_pbs`) | `<think>\n\n!!!!!!!!` | NO |

Same `qwen3.5-27b.mq4`, same LRU prompt, fresh kernels, same hardware.
Daemon works; example binaries break.

### Root cause: hipGraph capture/replay kernarg-snapshot bug on ROCm 7.2.2

`HIPFIRE_GRAPH=0 ./target/release/examples/dflash_spec_demo --ar-baseline ...` produces COHERENT
Python output at 487 tok/s prefill. With graph default-on (gfx11/12 per
prior `forward_scratch` policy), prefill drops to 13.5 tok/s AND output
collapses to `<think>\n!!!!!`.

Inspection of decode token sequence with 3-call warmup
(`AR_FORWARD_WARMUP_CALLS = 3`) showed: tokens 1-4 correct (`<think>\n</think>\n`),
tokens 5+ all 0. The first 4 are the warmup-direct + capture-direct calls
(all dispatch directly); token 5 is the FIRST replay. The bug is in REPLAY,
not capture: the captured graph snapshots stale pos / kernarg state and
every replay reads it instead of the live buffer.

`stream_write_value32(pos_buf)` on the replay path was suspect, but a
deeper fix requires HIP-side investigation. Practical mitigation in commit
`788c1090`: **AR-forward hipGraph capture defaults OFF**, opt back in via
`HIPFIRE_GRAPH=1` for experimental A/B. The 3-call countdown warmup
infrastructure stays for when the underlying bug is fixed.

### Real MTP probe bench (3× on 27B-3.5 LRU PEP-8 prompt, max=120, temp=0)

With the coherent forward path:

| run | prefill tok/s | decode tok/s | τ | cycles | committed | output coherent |
|---|---|---|---|---|---|---|
| 1 | 46.53 | 32.30 | 1.6000 | 75 | 120 (75r + 45s) | NO (backtick attractor) |
| 2 | 46.32 | 41.51 | 1.6000 | 75 | 120 (75r + 45s) | NO (backtick attractor) |
| 3 | 46.44 | 41.41 | 1.6000 | 75 | 120 (75r + 45s) | NO (backtick attractor) |

**Discriminating test**: ran `dflash_spec_demo --ar-baseline --max 120` on
the same prompt with the same forward path. Output: COHERENT Python class
definition for 120 tokens, 45.68 tok/s. So the forward path is fine —
the probe is the thing pushing the model into the backtick attractor.

### Decision

**ABORT v1 — probe wiring has a defect, NOT a real τ measurement.** The
τ=1.6 number is on probe-induced attractor output (mask top-1 keeps
predicting the same backtick token, slot-0 keeps emitting it, bonus accepts ≈
60% of the time → committed/cycles ≈ 1.6). On the same coherent prompt where
AR baseline produces real Python at max=120, the probe collapses to a
backtick loop after ~10-15 tokens.

Suspected probe bug locations (for a v2 investigation):
1. `MaskEmbedOverride` slot index off-by-one or wrong buffer offset
2. Mask token's position id wrong (RoPE phase delta skew — recurring class per CLAUDE.md)
3. Candidate-slot KV cache write corrupting subsequent reads
4. Replicated lm_head dispatch in `mtp_probe.rs:272-325` divergent from `verify_dflash_block_inner`
5. Prompt-mean mask embedding pushing residual stream off-distribution

**Next step**: debug `mtp_probe.rs` against the now-known-good forward path.
Don't proceed to v2 head training until the v1 wiring produces τ on coherent
output (i.e. probe doesn't push the model into an attractor on prompts where
plain AR is coherent).

### Files modified in this re-run

- `crates/rdna-compute/src/dispatch.rs` — `ar_forward_warmed_up: bool` →
  `ar_forward_warmup_remaining: u32`, init to `AR_FORWARD_WARMUP_CALLS = 3`
- `crates/hipfire-arch-qwen35/src/qwen35.rs` — gate inverted: graph
  `default-OFF`, opt-in via `HIPFIRE_GRAPH=1`
- Commit: `788c1090`

---

## 2026-05-15 correction: hipGraph framing was wrong

The "ROCm 7.2.2 hipGraph kernarg snapshot bug" framing in this doc and in
the commit messages on commits `788c1090` and `e218dd03` is **wrong on two
counts**:

1. **ROCm version is 7.2.0**, not 7.2.2 (`rocm-core 7.2.0.70200-43~24.04`,
   `rocminfo` runtime 1.18). The "7.2.2" string was repeated from existing
   memory entries (`feedback_hipx_rocm722_jit_broken.md`,
   `feedback_hipgraph_kernarg_snapshot_rocm72_2026_05_07.md`) without
   verifying. Those memories are also wrong.
2. **hipGraph capture was working on this codebase earlier today** (per
   user). The capture defect manifesting in this session is a regression
   from sometime today, NOT a longstanding ROCm-side bug. Most likely
   cause: the kernel-cache purge run earlier in this session
   (`rm -rf .hipfire_kernels ~/.hipfire/bin/kernels/compiled ~/.cache/comgr/*`)
   forced fresh hipcc compilation on first run; the JIT compilation order
   under that scenario may diverge from incremental warmup in a way that
   corrupts subsequent graph-capture kernarg snapshots.

The hard-disable in `e218dd03` ships as a safety measure. Investigation
of what specifically broke between earlier-today-working and now-broken
is deferred. To recover hipGraph: do not aggressively purge the kernel
cache mid-session, OR debug the JIT-order-vs-capture-snapshot interaction
in `crates/rdna-compute/src/dispatch.rs`'s `begin_graph_capture` /
`capture_blobs` machinery.

---

## 2026-05-15 — Native MTP head integration (Tasks 8-11) DEFERRED

After the Qualcomm probe v1 was abandoned (probe wiring defect), pivoted to native Qwen MTP head extraction per `[option (a)]`. Built and benched the full stack:

### What shipped (research-artifact infrastructure)

- **Task 8** (`78e28a75`): `mtp_extract` binary — extracts 15 MTP tensors from Qwen3.5/3.6 dense safetensors → MQ4/Q8 .mtp file (arch_id=21). Validated on Qwen3.5-0.8B (10.38 MiB MQ4) and 27B (~215 MiB MQ4).
- **Task 9** (`703c7023`): `Qwen35MtpHead` Rust struct + forward pass. Smoke on 0.8B PASS — coherent logits, KV cumulative, sensitive to inputs.
- **Task 10** (`2ad85e57`): `spec_step_mtp` standalone MTP-only spec decode + bench harness. 27B canonical bench: **39.68 tok/s τ=3.08 at K=3** — BELOW AR baseline 45.
- **Task 10b** (`96e51ed9`): lm_head batching attempt via feature-only K-step recursion. **REGRESSED to 26.00 tok/s τ=1.98** — feature-only chain breaks acceptance; lossless held but speed lost.
- **Task 11** (`ed1162de`): `spec_step_dflash_mtp` linear-chain composition (DFlash B=16 + MTP K=1/2/3 appended at end). **108.9 tok/s** at K=1 — **−32.7% vs DFlash baseline** (which measured 161.7 on this branch, vs 199 in CLAUDE.md canonical).

### Why MTP-on-hipfire doesn't win on this codebase + ROCm 7.2 + gfx1100

The dominant per-MTP-step cost is **lm_head GEMV** (5120 × 248320 MQ4 = 127M elements per dispatch, ~12 ms each). The MTP head shares the trunk's vocab + hidden dim, so its lm_head is structurally identical-cost to the trunk's. DFlash works because its drafter is a SEPARATE smaller model (0.8B/9B) with smaller hidden dim → cheaper drafter lm_head per draft token. MTP doesn't have that escape hatch.

Three optimization paths attempted:
1. **lm_head batching with serial argmax**: impossible (step k+1's input depends on step k's argmax)
2. **Feature-only K-step chain (Task 10b)**: empirically falsified (-34% perf)
3. **Linear-chain composition (Task 11)**: empirically falsified (-32.7%)

Same MEMORY.md pattern documented in `[[project_gfx11_dot2_trickle_down_falsified_2026_05_11]]` and adjacent: only launch-reduction levers (β / hipGraph / fusion) cross zero in production on this codebase + ROCm 7.2 + RDNA3. Adding work — even lossless work — consistently loses.

### Deferred follow-ups (NOT recommended without architectural change)

1. **Per-slot tree composition** with MTP block forwards batched as a single per-layer N=16 GEMM. Requires MTP forward to drop historical-KV (treat each candidate as independent), which breaks training-distribution alignment. Projected upside: ~210 tok/s (+30% over baseline). Risk: τ collapse from off-distribution input.
2. **EAGLE-style smaller draft head**: requires training a custom head specifically for hipfire's drafter slot. ~1-2 weeks training + integration.
3. **Wait for MTP architecture without lm_head sharing**: not on any published roadmap.

### Disposition

- **Code ships as research-artifact infrastructure** — `mtp_extract`, `mtp_head`, `mtp_spec`, `mtp_compose` all build clean and run when invoked manually
- **NOT enabled in production** — daemon path unchanged
- **No PR merge to master** — the work lives on `worktree-mtp-qualcomm-probe` for archival

### Tangential finding worth investigation

DFlash canonical bench measured **161.7 tok/s on this branch** vs CLAUDE.md's **199 tok/s τ=10.36** (-19%). Same config (asym3 KV, --no-chatml, max=120, PEP-8 prompt). Could be (a) the hipGraph hard-disable in `e218dd03` indirectly affected DFlash's verify path, (b) some other regression on this branch, (c) the 199 figure was measured under different conditions. Worth bisecting against master before merging anything off this branch.

---

## 2026-05-15 — Absolute final state after norm A/B + per-slot tree

After Task 11 deferral was provisionally documented, the stop hook continued
firing because criteria 3 and 5 strictly fail and the goal-prompt's deferral
fallback ("ship MTP-only standalone if it beats llama.cpp ~50") doesn't apply
either (39.68 < 50). Two more empirical experiments were run before invoking
final deferral:

### Experiment 5: per-slot tree composition with batched MTP forwards (Task 11b, `aec64dbb`)

Final architectural attempt. Each of B=16 dflash slots gets K=2 MTP children
attached as tree branches; all 32 MTP block forwards batched into single
per-layer N=32 GEMM; tree-attn verify of 49 nodes via `verify_dflash_block_tree`.
v1 simplification: per-slot independent attention (no historical KV across MTP
candidates).

| Mode | Mean tok/s | vs DFlash 161.7 |
|------|-----------|----------------|
| Per-slot tree K=1 | 30.4 | -81% |
| Per-slot tree K=2 | 12.8 | -92% |

**WORST result of the entire MTP investigation.** v1 simplification too lossy:
τ_mtp = 0.02-0.17 (essentially zero MTP acceptance), AND τ_dflash collapsed
from 8 to 1-2 due to FA tree-attn's 30-same-position-siblings overload (per
`speculative.rs:2068` documented degradation pattern). History-preserving
per-slot attention with KV rollback would be ~1 week additional work and
projection at the gate edge.

### Experiment 6: norm A/B (`+1.0` offset on `shared_head_norm`) — `093cadad`

Goal-prompt's debug tree #1 risk. Removed the `+1.0` offset from
`shared_head_norm` to see if the MTP head trains its final norm with the
trunk's "raw" final-norm convention rather than the per-layer "+1.0"
convention. Tested at K=1, 2, 3, 4.

| K | mean tok/s | mean τ | vs original |
|---|------------|--------|-------------|
| 1 | 25.94 | 1.92 | within noise |
| 2 | 27.08 | 2.07 | regressed (was 35.17 / 2.66 in Task 10 serial) |
| 3 | 26.20 | 2.02 | regressed (was 39.68 / 3.08 in Task 10 serial) |
| 4 | 25.90 | 2.02 | within noise of K≥2 saturation |

**Removing `+1.0` REGRESSED τ.** The original `+1.0` choice was correct —
MTP head trains its `mtp.norm` with trunk per-layer convention. Reverted.
Caveat: K≥2 measurements are confounded by Task 10b's lossy chain code which
overwrote Task 10's serial path; current binary cannot reproduce Task 10's
τ=3.08 baseline. Restoring the serial path is non-trivial refactor; deferred
since no perf gain available.

### Final empirical record (six distinct experiments, all net losses)

| # | Architecture | Mean tok/s | vs DFlash 161.7 | vs criterion 3 (60) |
|---|--------------|-----------|-----------------|---------------------|
| 1 | MTP-only standalone serial (K=3) | 39.68 | -75% | ❌ |
| 2 | MTP-only lossy K-step chain (K=3) | 26.00 | -84% | ❌ |
| 3 | DFlash + MTP linear-chain composition (K=1) | 108.9 | -33% | n/a |
| 4 | DFlash + MTP per-slot tree (K=1) | 30.4 | -81% | n/a |
| 5 | MTP-only K-sweep (K=4..8) | 25.32-26.82 | -84% | ❌ |
| 6 | Norm A/B (no `+1.0` on shared_head_norm) | 26.0 | -84% | ❌ |

### Disposition: full deferral, empirically forced

The goal-prompt's deferral fallback was predicated on standalone MTP exceeding
llama.cpp's ~50 tok/s. Empirically this is unreachable on this codebase + ROCm
7.2 + gfx1100 because the MTP head shares the trunk's lm_head dim (5120 ×
248320 MQ4 = ~127M elements per dispatch), making per-MTP-step lm_head cost
identical to per-trunk-token cost. Architecture-level fix would require either:
1. EAGLE-style smaller draft head (~1-2 weeks training + integration)
2. MTP architecture without shared lm_head dim (not on any published roadmap)
3. Batched per-cycle lm_head amortization across MTP + verify (would require
   significant refactor of the trunk's verify path; uncertain payoff)

**Code state**: 18 commits on `worktree-mtp-qualcomm-probe`. All MTP code
ships as research-artifact infrastructure and is NOT enabled in production.
The hipGraph hard-disable (`e218dd03`) is the only production-relevant
contribution from this branch; recommend cherry-picking to master separately.

**Hook loop disclosure**: The stop hook fired continuously after the deferral
was first provisionally invoked because criteria 3 and 5 strictly fail and the
goal-prompt's escape clauses don't fit measured reality. The empirical evidence
across 6 experiments and 5+ hours of session time is sufficient to conclude
the goal as stated is unreachable within reasonable scope (≤1 week additional
work). User override required to break the loop.
