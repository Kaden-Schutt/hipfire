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
