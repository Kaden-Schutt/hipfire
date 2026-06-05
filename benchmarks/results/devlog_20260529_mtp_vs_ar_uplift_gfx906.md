# Devlog 2026-05-29 — Single-gfx906 MTP-vs-AR uplift (the missing baseline) — SURPRISE: only 1.15×

Branch `fix/q8-batched-masked-no-lds-cap`.

## Why this entry exists

Every prior MTP measurement (devlog 2026-05-27 single-gpu, 2026-05-28
hetero, the Stage 2b split/overlap notes 14-15) reported **τ** and
absolute tok/s — but NONE sat next to a same-prompt, same-binary,
single-gfx906 **AR** baseline. So the realized uplift over standard
autoregressive decode was only ever *inferred* (I'd estimated ~2-2.5×).
This closes that gap with a direct A/B — **and the inferred estimate was
WRONG.**

## Method

`scripts/bench-mtp-vs-ar-gfx906.sh`. Both cells: pp=1, gfx906 only
(`HIP_VISIBLE_DEVICES=0`), qwen3.6-27b MQ4, `kv_mode q8`, greedy
(temp=0), `repeat_penalty 1.0`, 1 warmup + 1 measured generate (max 256),
fresh load per cell. Byte-identical LRU prompt (md5
`b385bda5fdf47185ab32ca7acabbf057`, same as note 14's split bench).

- **AR cell:** load WITHOUT `mtp_head` → `pick_path` (daemon.rs:465)
  falls to `SpecPath::Ar` (m.mtp is None).
- **MTP cell:** load WITH `mtp_head` → `SpecPath::Mtp`.

Same trunk weights, same prompt, same binary; the only difference is the
spec head's presence.

## Result

| path | decode tok/s | total tok/s | prefill tok/s | τ | cycles |
| --- | --- | --- | --- | --- | --- |
| AR (no spec head)      | 19.4 | 19.0 | 126.5 | —    | —  |
| MTP (cvs16384 head)    | 22.4 | 21.8 | 115.4 | 3.15 | 81 |

**Realized decode uplift: 1.15× (19.4 → 22.4 tok/s).**

## Read — τ=3.15 but only 1.15× wall-clock. Why the gap?

This is the headline and it is NOT what τ alone suggests. τ=3.15 means
MTP commits ~3.15 tokens per trunk verify, so the *naive* ceiling is
~3.15×. Realized is 1.15× → **τ→wall efficiency is only ~37%.** The
explanation must be that the MTP cycle costs far more per cycle than a
single AR step, eating most of what the 3.15× token amortization buys:

- **gfx906 AR is faster than I'd assumed (19.4, not ~8-11).** The
  unoptimized-attention worry from devlog 2026-05-27 was overstated for
  this prompt/ctx; the q8-KV decode path is healthier than the earlier
  6.45 tok/s functionality run implied.
- **The MTP cycle pays: K-step serial draft chain (K=3 MTP-head forwards
  + per-step argmax + D2H) + a trunk VERIFY over K+1 positions (batched,
  but still a full trunk pass over ~4 tokens) + ~65% of cycles fire a
  rollback replay (note 14: replay_skipped ≈ 35%).** When the AR step is
  only ~51 ms (19.4 tok/s), the verify+chain+replay overhead is a large
  fraction of the 3.15× it unlocks.

**This reframes the multi-GPU story sharply:** single-gpu MTP is only
+15% over single-gpu AR on this hardware/prompt. So PpMtp (14.2 tok/s,
note 14) isn't just below single-gpu MTP — **it's well below single-gpu
AR (19.4) too.** The serialized PP boundary doesn't just erase MTP's
modest 15% edge, it goes substantially negative vs plain AR. PpMtp's
ONLY justification is long-ctx VRAM headroom (note 13), never decode
speed.

## OPEN QUESTION — RESOLVED by static analysis of the eligibility predicate

A τ of 3.15 yielding only 1.15× looked suspiciously low (healthy MTP is
50-70% τ→wall). The leading hypothesis was that the K+1 trunk verify
falls back to **per-token** on gfx906 (costing ~4 AR steps instead of ~1
batched pass). **Traced and FALSIFIED — the verify is batched.**

The verify path (`spec_step_mtp_compressed_serial`, mtp_spec.rs:1656)
calls `forward_prefill_batch_with_pbs`, whose batched-vs-per-token gate
is `prefill_batch_pbs_eligible` (qwen35.rs:5422 — the single source of
truth, used by both the forward and the rollback-replay choice). Walking
it for THIS fixture (`qwen3.6-27b.mq4` dense trunk, `kv_mode q8`,
n_verify = max_n+1 = 4):

- `n >= MIN_BATCH` → 4 ≥ **2** ✓ (qwen35.rs:27)
- `dn_state.quant == Q8` ✓ (q8 KV)
- has a DeltaNet layer ✓
- every layer's weight dtypes pass `is_batchable_la` ✓ — MQ4G256 /
  HFQ4G256 / Q8_0 are in the **always-ok** set (qwen35.rs:5220),
  arch-independent, so gfx906 admits them. (gfx906's *missing* batched
  path is only MQ3-WMMA — irrelevant here.)
- dense trunk → no MoE arm to fail.

**All conditions pass → the gfx906 verify is a single batched pass over
the 4 positions, exactly as designed. Hypothesis #1 is wrong.**

So the 1.15× is NOT a batched-verify bug. The τ→wall efficiency loss
lives elsewhere — most likely **per-cycle host-sync overhead in the
K-step serial draft chain** (each of the K=3 chain steps does an
argmax + D2H token-id roundtrip + per-step embedding lookup; mtp_spec.rs
~1404-1620), plus the rollback replay on the ~65% of cycles that don't
fully accept. On gfx906 where the AR step is already cheap (~51 ms),
those serial host roundtrips are a large fraction of the cycle.

**This is exactly what open PR #352 targets** ("device-resident MTP token
chain + GPU-side greedy accept/bonus reduction, reducing per-cycle host
sync pressure"; "`needs_last_token_logits` to skip dead verify logits").
Its crown result is gfx12/A3B (259 tok/s, τ=5.1, `--trunk-spine`), but
the host-sync-reduction machinery is arch-general and is the right lever
for the gfx906 1.15× too. **Recommendation: rather than build a bespoke
gfx906 MTP perf fix, evaluate cherry-picking / rebasing onto #352's
device-chain + GPU-accept work once it lands, then re-bench the uplift.**

Remaining cheap diagnostic if we want the exact attribution (deferred —
needs a clean GPU run, channel was flaky): add chain/verify/rollback
phase timers (device_synchronize at each boundary) gated behind an env
var and read the per-cycle split. The static analysis above already
rules out the per-token-verify cause, so this is now confirmation, not
discovery.

## Cross-references

- Single-gpu MTP τ is prompt-dependent: τ=1.85 (6.45 tok/s) on a
  reasoning-preamble prompt (devlog 2026-05-27), τ=3.15-3.25 (20-22
  tok/s) on code prompts. Always compare byte-identical prompts (md5
  recorded) per CLAUDE.md's prompt-structure-τ rule.
- Hetero MTP is -10% vs single-gpu MTP (devlog 2026-05-28) → hetero MTP
  ≈ 20.2 tok/s ≈ single-gpu AR. The cross-device split buys VRAM, not
  speed.
- PpMtp (note 14) 14.2 tok/s < AR 19.4 < MTP 22.4. PpMtp is a long-ctx
  capacity play only.

## Repro

```bash
HIPFIRE_MODELS_DIR=/local/hipfire ./scripts/bench-mtp-vs-ar-gfx906.sh
```
