# MTP Qualcomm-style probe v1 — bench results on Qwen3.5-27B (DFlash mq4)

Date: 2026-05-14
Branch: `worktree-mtp-qualcomm-probe`
Hardware: gfx1100 (k9lin's Sapphire Nitro+ 7900 XTX), ROCm 7.2.2

## Algorithm

v1 is a Qualcomm-style training-free MTP probe: at each cycle, append a single
`<|MASK|>` token after the last committed token, run one batched target forward
over `[last_committed, MASK]`, take argmax of both logit positions to produce
`(real_t, spec_t+1)`, then verify by running the next single-token forward on
`real_t` and accepting `spec_t+1` iff `argmax(real_t.logits) == spec_t+1`. No
tree, no draft model, no head training, lossless greedy. EMA τ-estimator with
λ=0.1 controls a soft admit gate to avoid catastrophic mis-speculation when the
mask channel collapses.

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
