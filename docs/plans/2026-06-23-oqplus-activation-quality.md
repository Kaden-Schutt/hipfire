# Plan 1: OQ+ (W4A8) quality — activation-aware + weight-error-feedback levers

Date: 2026-06-23. Branch: chaingun. Follows the recovery-FT exploration, which
established that **FT-based recovery is NOT the lever** for OQ+ on qwen3.5
(block-local norm recovery was a wash end-to-end: ppl 27.25→27.28; see
[[project_qwen35_norm_recovery_phaseA]]). The quality must come from the
**offline quantizer**, not post-hoc fine-tuning. AWQ already recovers OQ+ ~1.9×
(KLD 0.1536→0.0813, commit cf387d42). This plan pushes further.

## Goal

Minimize OQ+ (W4A8) KLD-vs-bf16 toward the W8A8 floor (oq8 = 0.00156), and decide
whether OQ+ is production-viable vs the mq4+ baseline. Headline gap to close:
OQ+awq **0.0813** → ideally <0.03 (the gap from W4 weight quant, since A8 is clean).

## Key hypothesis (the cheap high-value test)

**LDLQ (full-Hessian GPTQ/OBS error-feedback weight quant) should help OQ+ even
though it did NOT help oq4 (W4A4).** For W4A4 the dominant error was the runtime
int4 ACTIVATION quant (AWQ's target), so LDLQ-on-weights added ~nothing over AWQ
(RTN 32.16 / LDLQ 30.74 / AWQ 29.45 / LDLQ+AWQ 29.48 ppl; see
[[project_opus_w4a4_status]]). OQ+ is W4A8 — activations are clean int8, so the
**weight** int4 quant is now the dominant error (cf387d42 sweep: W8A8 0.00156 vs
OQ+ W4A8 0.1536 — the entire gap is the W8→W4 weight step). Weight error-feedback
should therefore pay off here. **If true, LDLQ+AWQ is the new best OQ+ recipe.**

## What's already built (no new codec work to start)

- `HfqInputFormat::OqPlus` (`--format oq+`, qt=33) — plain W4A8.
- `OqPlusTiered` (`--format oq+t`, qt=34) and `OqPlusCompact` (`--format oq+c`,
  qt=36) — magnitude-tiered: int4 bulk + sparse int8 outliers (`N_out =
  round(w8_frac·256)`/group); loaders qt 34/36 expand+overlay. `quantize_oqplus_compact`.
- `--awq` (AWQ/SmoothQuant sidecar) and `--ldlq` (full-Hessian) both compose with
  all three: `ldlq::oq4_ldlq_pack` (plain), `oqplus_compact_ldlq_pack`,
  `oqplus_tiered_ldlq_pack` (main.rs ~3124/3222/3224). `--hessian <h>` feeds both.
- Hessian: `~/.hipfire/hessians/qwen3.5-0.8b.hessian.bin` (full [K,K], `HfhsFull`).

So the front-end exists; the deliverable is the **eval program** that finds the
best recipe and validates the LDLQ-helps-W4A8 hypothesis.

## Eval methodology (per the perf/quality rules)

- **KLD-vs-bf16** (primary, low-noise): `build_kld_ref*` from bf16
  (`crates/hipfire-eval/src/quality.rs`), then KLD per recipe. ≥16 chunks (top-K
  KLD is noisy at 2 — the lesson from the KV work).
- **ppl ctx=2048** (`perplexity` example, default lowered path — NOT
  FORWARD_LOWERED=0, which breaks scoring; a 512-tok window is too noisy, use 2048).
- **Coherence**: `./scripts/coherence-gate.sh` on the winning recipe (no attractor /
  list-primes loop — the failure mode plain oq4 had).

## Steps (priority order)

1. **Recipe sweep on qwen3.5-0.8b** (cheap, all front-end exists):
   `oq+ {RTN, AWQ, LDLQ, LDLQ+AWQ}` × baseline. KLD + ppl2048 each. Confirms or
   kills the LDLQ-helps-W4A8 hypothesis directly. ~8 quantize runs + evals.
2. **Tiered/compact sweep:** `oq+c` (and `oq+t`) at `w8_frac ∈ {0.01, 0.03, 0.06}`
   × {AWQ, LDLQ+AWQ}. Measures the quality/byte Pareto of sparse-int8-outlier
   protection (the natural fix if pure W4 leaves a floor). Plot KLD vs bits/weight.
3. **Pick the knee** of the Pareto; run coherence gate; compare to mq4+ (the
   incumbent W4A8 — same iu8 kernel, affine-u4+clip+SmoothQuant) on equal footing.
4. **Only if headroom remains after 1–3** (new work, gated on the sweep):
   - AWQ α search (current α is fixed; sweep α∈{0.25,0.5,0.75} — `compute_awq_scales`).
   - Per-group (vs per-token) int8 activation quant — tighter A8, if A8 turns out
     non-negligible at the W4-recovered operating point.
   - Better/learned rotation in place of fixed FWHT-256 (only if rotation shows as
     the residual bottleneck via an SQNR-by-stage breakdown).

## Decision this produces

A single recommended OQ+ recipe + its KLD/ppl/coherence, and a go/no-go vs mq4+:
- If LDLQ+AWQ (or compact) gets OQ+ KLD into mq4+ territory at ≤ mq4+ bits → OQ+
  ships as the W4A8 production format (symmetric-int4, the cleaner iu8 path).
- If it can't beat mq4+ → OQ+ stays a research format; mq4+ remains production W4A8.

## RESULTS (2026-06-23, qwen3.5-0.8b, KLD-vs-bf16 991 positions, held-out calib)

Hypothesis **CONFIRMED** — LDLQ helps W4A8 strongly (opposite of W4A4):

| recipe | KLD | ppl | size |
|--------|----:|----:|-----:|
| OQ+ RTN | 0.1568 | 38.95 | 538MB |
| OQ+ AWQ (prior best) | 0.0912 | 35.99 | 538MB |
| OQ+ LDLQ | 0.1181 | 37.84 | 538MB |
| **OQ+ LDLQ+AWQ** | **0.0459** | 33.15 | 538MB |
| OQ+ compact w8=3% LDLQ+AWQ | **0.0408** | 33.24 | 569MB |
| OQ+ compact w8=6.25% LDLQ+AWQ | 0.0411 | 33.28 | 601MB |
| — mq4+ (incumbent W4A8) | 0.0780 | 33.58 | 550MB |
| — mq4 | 0.1439 | 34.14 | 549MB |
| — bf16 (ref) | 0 | 31.32 | — |

- **LDLQ+AWQ nearly halves AWQ-alone** (0.046 vs 0.091); LDLQ-alone beats RTN
  (0.118 vs 0.157). For W4A4 LDLQ added ~nothing over AWQ — the W4A8 clean-int8-
  activation regime is exactly why weight error-feedback now pays. Hypothesis holds.
- **OQ+ LDLQ+AWQ beats the mq4+ incumbent by ~1.7× lower KLD** at equal (4-bit)
  weight memory and the same iu8 kernel — and is coherent ("...is **Paris**.",
  clean `<|im_end|>`, no attractor; the recovery fixes plain oq4's loop failure).
- Compact (sparse int8 outliers) knee at **w8=3%**: KLD 0.0408 (+6% size, −11% KLD);
  past ~3% outliers no further gain (0.0625 flat). Plain LDLQ+AWQ is the smallest
  recipe that already beats mq4+.

**DECISION: OQ+ LDLQ+AWQ is the recommended W4A8 recipe** (compact-3% if the +6%
storage is acceptable for the extra quality). It supersedes the AWQ-only OQ+ and
beats mq4+. Remaining (lower priority): formal `coherence-gate.sh`, larger-model
(9B/27B) confirmation, AWQ-α / per-group-activation only if more headroom is wanted.

## BENCH + FULL-EVAL before promotion (2026-06-23) — NOT PROMOTABLE AS-IS

Benching before promoting (correctly) caught two blockers that KLD-alone missed:

**Perf (gfx1103, warmed, fixed prompt, HIPFIRE_MAX_GEN=128, infer_qwen35 raw path):**
| model | prefill tok/s | decode tok/s |
|-------|-------------:|------------:|
| OQ+ LDLQ+AWQ | 12 | **14.1** |
| mq4+ (incumbent) | 60 | **59.1** |
| bf16 | 16 | 16.0 |
OQ+ decode is **4.2× slower than mq4+** (slower than bf16). No generic-GEMV warn
fired → not the catch-all fallback; it's a structurally slow Oq8 W8A8 DECODE path —
almost certainly **unfused per-projection Oq8 GEMVs** (mq4+ has fused qkv/gate-up
decode kernels; Oq8 W8A8 apparently lacks them, and gfx1103 decode is memcpy-sync-
bound so unfused = many more per-token syncs). Engineering gap, fixable.

**Daemon load: NOT a real blocker — it was a STALE INSTALLED DAEMON (RESOLVED).**
Initial diagnosis (slab loader lacks qt=33) was WRONG. The slab loader correctly
returns None for qt=33 (`slab_dtype_for_quant`) and falls back to
`load_weight_tensor_raw`, which HAS the qt=33 arm — no code change needed. The real
cause: `find_daemon_bin` (hipfire-daemon-adapter) prioritizes `~/.hipfire/bin/
hipfire-daemon` OVER `target/release`, and the installed copy was from 06-22 18:08
(before cf387d42 added qt=33 at 20:00). Every eval/chat spawned that stale daemon →
"unsupported quant_type 33" panic at a stale line. FIX: refreshed `~/.hipfire/bin/`
with current binaries. OQ+ then loads + serves coherently on the daemon. (Lesson:
when the daemon panics but infer_qwen35/perplexity don't, suspect the installed
`~/.hipfire/bin` daemon, not the source.)

**Full daemon eval (smoke,coherence,quality,speed; fresh daemon):** OQ+ and mq4+
BOTH 9 pass / 1 fail / 1 skip — the one fail is the SAME `tool_call_read_file`
coherence detector on both (parity, not an OQ+ regression; 0.8b fast-tier quirk),
quality skipped at fast tier (KLD already measured separately). Daemon speed
confirms the perf gap on the production path: OQ+ decode 14.3 / prefill 16.0 tok/s
vs mq4+ decode 59.7 / prefill 104.8 — **4.2× slower decode, 6.5× slower prefill.**

**Quality (confirmed, the motivation):** KLD 0.046 (1.7× better than mq4+),
coherent on capital/reasoning prompts, no attractor.

**VERDICT: do NOT promote OQ+ yet — ONE blocker remains (perf).** Quality wins
decisively (KLD 0.046 vs mq4+ 0.078), loads + serves + coheres at parity on the
daemon, but is 4.2× slower decode / 6.5× slower prefill. Promotion is gated on a
single engineering item, not quality and not loading:
1. **Fused Oq8 W8A8 kernels** — fused qkv/gate-up DECODE (the 4.2× gap) and
   BATCHED-PREFILL (the 6.5× gap), mirroring the mq4 fused path. gfx1103 decode is
   memcpy-sync-bound, so unfused per-projection Oq8 GEMVs = many extra syncs.
(The earlier "blocker #2: slab-loader qt=33" was a misdiagnosis — it was a stale
installed daemon, now fixed; OQ+ serves fine.)
Until the fused Oq8 kernels land, mq4+ remains production W4A8. The LDLQ+AWQ recipe
+ its KLD win are the standing motivation to fund that kernel work.

## PROMOTED as OQ8+ (2026-06-23) — decode blocker substantially closed

The fused Oq8 decode kernels landed (commits d810f54b, 15a48ba9). Decode went
**14.1 → 42.7 tok/s** (3.0×) via:
- GEMV-demux fused QKVZA + gate-up (one launch, one wave/output-row, no WMMA
  N-tile waste at B=1), B=1 wo/down residual via gemv (not gemm_oq8).
- **W4A16 decode**: the rotated f32 activation is consumed directly and the int8
  weight is dequantized inline (mq4-style) — no quantize_act_oq8 launch.

**Profiling correction (rocprofv3, 0.8B):** this decode path is **96.7% kernel
time, 3.3% memcpy** — NOT memcpy-sync-bound (that finding was the daemon path on
bigger models). The residual gap to mq4+ (42.7 vs 59) is **weight-bandwidth**:
OQ+ stores 4-bit weights but expands them to int8 in VRAM at load, so decode
reads ~2× the bytes mq4 does. Per-launch the oq8 in-proj GEMVs are 1.4–1.5×
slower, tracking the int8-vs-int4 byte ratio. No decode-side fusion can close
this — only 4-bit-resident weights can.

**DECISION (user, 2026-06-23): PROMOTE OQ8+** at decode 42.7 / KLD 0.046330
(W4A16-decode numerics; still 1.7× better than mq4+ 0.078; coherent). The
int8-resident weight-bandwidth gap is accepted; closing it is the **OQ4+**
(4-bit-resident W4A8) follow-on, tracked separately.

- Canonical artifact: `~/.hipfire/models/qwen3.5-0.8b-oq8+.hfq`
  (md5 94ad6e0be70b1768af4b5c6342a6900c; from `/tmp/oqp-ldlqawq.hfq`,
  LDLQ+AWQ recipe, qt=33).
- Registry: `oq8` quant status experimental → **opt-in**; label corrected to
  W4A8 (weight_bits=4, act_bits=8). Batched-prefill WMMA path remains
  `partial`/parity-gated (the [[gate]] entry is unchanged — Tier 2 still open).

## OQ4+ supersedes OQ8+ at decode (2026-06-23)

Applying the same W4A16 decode to oq4's **4-bit-resident** (nibble-packed) weights
gives the prize OQ8+ couldn't: mq4-class decode speed AND OQ8+ quality.

| format | weight residency | decode KLD | decode tok/s | VRAM |
|--------|------------------|-----------:|-------------:|-----:|
| mq4+   | 4-bit (int4)     | 0.078      | 59           | 0.55 GB |
| OQ8+   | int8 (expanded)  | 0.046330   | 42.7         | 0.54 GB (loads int8) |
| **OQ4+** | **4-bit (int4)** | **0.046337** | **54.9**   | **0.54 GB** |

OQ4+ = `oq4` (Oq4G256) 4-bit-resident + LDLQ+AWQ weights + **W4A16 decode**
(gemv_oq4_grouped unpacks the nibble weight inline × f32 act; no quantize_act_oq4,
no WMMA waste). Commit 48a8d07b.
- KLD 0.046337 == OQ8+ 0.046330 (same 4-bit weights) → 1.7× better than mq4+.
- Decode 54.9 (93% of mq4+, +29% over OQ8+) — the 4-bit-resident weight halves the
  decode read bytes vs OQ8+'s int8 expansion, closing the bandwidth gap.
- **W4A4 int4-activation attractor is GONE** (list-primes loop): W4A16 decode
  sidesteps the int4 acts that were the W4A4 damage.
- **OQ4+ DOMINATES OQ8+ at decode**: same quality, faster, half the loaded weight
  bytes. Canonical artifact `~/.hipfire/models/qwen3.5-0.8b-oq4+.hfq`
  (md5 5f91fba8…, oq4-ldlqawq weights). Registry `oq4` label updated to OQ4+.
- Remaining: W4A4 **batched-prefill** still parity-gated (the [[gate]] entry) — the
  W4A16/W8A8 batched-prefill path is the next perf item; decode is the win here.

## OQ4+ prefill investigation (2026-06-24)

After the decode wins, looked at the prefill axis. Findings (daemon path; note
`infer_qwen35` prefill is per-token-sequential and is NOT a batched-prefill harness):

- **Daemon OQ4+ prefill ≈ 92 tok/s** (240-token speed battery) vs **mq4+ ≈ 105** —
  already **88% of mq4 parity**, much better than the old `infer_qwen35` per-token
  number (~55) suggested. Decode 55.6 (matches the standalone measurement).
- **W4A4 batched prefill output is COHERENT on OQ4+** and byte-identical to the
  per-token path on the long prompts tested (agentic_hermes 318-tok + BST question:
  clean BST paragraph, identical with `HIPFIRE_OQ4_BATCHED_PREFILL` 0 and 1). The
  memory's batched-prefill divergence warning was on PLAIN oq4 (RTN-ish weights);
  the well-conditioned LDLQ+AWQ OQ4+ weights do not flip greedy argmax into
  incoherence on these inputs. Fast-tier coherence battery: identical pass set
  flag 0 vs 1.
- **Caveat (unresolved):** the speed battery showed prefill 92 tok/s for BOTH flag
  values, so it's not yet confirmed the batched WMMA *projections* are the source of
  the 92 (vs chunked prefill with per-token projections + batched attention). The
  env-flag's effect through the eval-spawned daemon child wasn't isolated. Treat 92
  as the current OQ4+ daemon prefill rate; whether batched-projection WMMA raises it
  further is unverified.

**Remaining lever to mq4 prefill parity = W4A16 batched prefill** (the clean,
divergence-free path, matching the W4A16 decode quality): dequant the 4-bit weight
to f16 inline and use f16×f16 WMMA against f16 activations — exactly mq4's
`gemm_*_hfq4g256_wmma` family (reads f16 X, `a_reg = sc*nib + zp`,
`wmma_f32_16x16x16_f16_w32`). For OQ4+ this is a 4-kernel port (qkv / qkvza /
gate_up / residual) with the symmetric split layout (scale-only, sign-extend, no
zp) + dispatch wiring + is_batchable_la(Oq4G256) un-gating. This avoids the int4-act
divergence entirely (no act quant) and would match OQ4+ decode quality. Scoped, not
yet built — decode was the priority and is done.

## Cross-cutting note

This is **offline-quant** quality work — orthogonal to and cheaper than the
FT-recovery path that this session showed doesn't transfer. No GPU training loop,
no capture; just quantize → eval. Reuses the validated codecs + Hessian infra.
