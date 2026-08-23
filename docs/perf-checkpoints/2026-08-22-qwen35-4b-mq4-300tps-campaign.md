# Qwen3.5-4B MQ4 — 300 tok/s campaign (lm_head HFQ4 lever + fold negative)

**Date:** 2026-08-22 · **Lifecycle:** historical
**Branch:** `tune/iter3-gate-up-bt2` · **Worktree:** `~/projects/hipfire-tune4b`
**Fixture:** `Qwen/Qwen3.5-4B-MQ4/qwen3.5-4b.mq4`, md5 `712b69f8cf1016081cfa507c4d50e33d`
**GPU:** gfx1100 RX 7900 XTX 24 GiB, exclusive via `/home/ghazni/gpu-coord` lock.
**Constraint scope:** plain AR, q8 KV, no speculative decoding, containerized
(one-shot compose/docker-run; images built from this branch).

## Result

| Config | tg128@64 | tg128@2048 | Source |
|---|---|---|---|
| Session start (`ff5b1e36`) | 207.9 | 203.0 | `hipfire bench --kv-mode q8 --spec off`, 5 runs |
| **+ `HIPFIRE_LM_HEAD_HFQ4=1` (2b2b39fe)** | **226.75** | **220.72** | same method |

Commit `2b2b39fe`: load-time requant of the tied-embedding output projection
from Q8_0 to HFQ4G256 (embedding buffer untouched), gated behind
`HIPFIRE_LM_HEAD_HFQ4=1`. Per-kernel: `gemv_q8_0` 776 µs → `gemv_hfq4g256`
385 µs @ 820 GiB/s. Conversion cost 0.85 s at load. Greedy temp-0 quality
checked on factual/code/haiku prompts in both arms: coherent and factually
correct; planet-fact wording differs (both valid).

## Negative result: qkvza consumer-fold (PARKED)

Attempted bit-exact fold of rmsnorm+AWQ+FWHT into the qkvza GEMV prologue
(`HIPFIRE_QKVZA_FUSEDNORM=1`). First implementation: fusednorm 116 µs/call vs
30 µs base consumer (+12 µs producer saved) → net −25 % decode, and greedy
output degenerated. Even a perfect fold saves ≤ 0.29 ms/token (+6 %): the
redundant per-block reduction+rotation is not near-free. Code removed from the
branch (ratchet bypass ceiling); design preserved in session notes. Any retry
must start from a bitwise probe per docs/kernel-tune-decode-campaign-2026-08.md.

## Ceiling analysis (why 300 was not reached this session)

Serialized GPU time/token after the lm_head lever ≈ 4.5 ms (≈222 tok/s
effective). Remaining budget lines are all at documented local optima:
residual GEMV 404 GiB/s (6 falsified schedule attempts), latency-bound
producer chain 0.75 ms (wavegrid/direct variants explored), lm_head now at
820 GiB/s (~86 % of the 960 GB/s combined effective ceiling). 300 tok/s
requires 3.33 ms/token ⇒ ~90 % sustained effective bandwidth through
latency-bound kernels; consistent with the prior session's revised floor
analysis that even 288 is unreachable on this arch generation without
speculative decoding.

## Method notes

- All GPU windows serialized via `/home/ghazni/gpu-coord/gpu-ctl run`.
- A/B env vars must be passed as separate `-e` flags: `-e "A=1 B=2"` sets one
  variable named `A` with value `1 B=2` (voided one measurement window).
- The main checkout gets branch-switched by outside agents; builds must come
  from the dedicated worktree after verifying HEAD.

## Addendum (same day): HFQ3 lm_head lever — 230.2 tok/s

Commit `53753f80` added `HIPFIRE_LM_HEAD_HFQ3=1` (streaming Q8_0→HFQ3G256
output-projection requantizer); commit `869dc424` admitted gfx110x to
`GemvHfq3G256` via the new `HasHfq3G256Gemv` predicate (the table's old
`HasSdot4` gate predated the RDNA3 wave32 kernel from the MQ3 production path).

Product path, 5 runs: **tg128@64 = 230.22 tok/s** (stdev 0.25),
tg128@2048 = 224.55 (stdev 0.32). Example-path pairs: HFQ3 217.1–218.5 vs
HFQ4 212.4–213.5, all HFQ3 > all HFQ4 (+2.3 % over HFQ4, +11 % over the
209.9 session start). Greedy temp-0 quality on planets / haiku / seasons /
Romeo-and-Juliet prompts: coherent and factually correct in both arms.

## Addendum 2 (same day): memory-clock state during decode — null

Sampled `pp_dpm_mclk` / `pp_dpm_sclk` every 2 s while the decode bench ran
(default DPM): MCLK reaches its top state (1249 MHz) within ~8 s of load and
holds it for the whole gen phase; SCLK auto-boosts to ~3.16 GHz. Forcing
performance level high + fixed clocks changed nothing material (forced sclk
index actually settled lower than auto boost). Memory/Infinity Cache clocks
are not a limiting factor — the bandwidth ceiling in the analysis above is
architectural, not a power-management artifact.

## Addendum 3 (same day): qkvza consumer-fold lands — 231.0 tok/s

Commit `3abbc05e`: `HIPFIRE_QKVZA_FUSEDNORM=1` folds rmsnorm+AWQ+FWHT into
the fused_qkvza GEMV prologue, bit-exact (probe bitwise on all four outputs;
greedy e2e text byte-identical). Root cause of the first attempt's corruption:
`__shfl_down` semantics in the emulated reduction tree — explicit-source
`__shfl` stages fix it. Net gain is small (+0.7 % example; launch saving
mostly offset by inline prologue cost) but consistent.

Product path with all levers (`HIPFIRE_LM_HEAD_HFQ3=1 HIPFIRE_QKVZA_FUSEDNORM=1`),
5 runs: **tg128@64 = 231.00 tok/s** (stdev 0.16), tg128@2048 = 225.27
(stdev 0.25); prefill 3857 / 4419 tok/s. Campaign total: 209.9 → 231.0
(+10.1 % this session; 190.6 → 231.0 = +21 % across both sessions).

## Addendum 4 (same day): post-restart re-confirmation + ops notes

Re-confirmation after a session restart, same shipped config
(HIPFIRE_LM_HEAD_HFQ3=1 HIPFIRE_QKVZA_FUSEDNORM=1, image
hipfire-rocmfp4:bench-final): tg128@64 = 228.50 tok/s (stdev 0.25),
tg128@2048 = 222.79 (stdev 0.27), prefill 3776.9 / 4349.2. Within ~1% of
the earlier 231.00/225.27 reading; both runs confirm all levers hold.

Ops notes for future sessions:
- A stray `daemon` process left by a crashed bench holds its full VRAM
  allocation until killed - check `/proc/*/fd` for /dev/kfd holders when
  hipMalloc reports 0 MB free with no obvious GPU user.
- docker `-e "A=1 B=2"` sets ONE variable named A with value "1 B=2" -
  pass multiple env vars as separate `-e` flags.
- The main hipfire checkout gets branch-switched by outside agents;
  build only from the dedicated worktree after verifying HEAD.

## Addendum 5 (same day): lm_head HFQ2 — perf-neutral, quality-degraded, not shipped

`HIPFIRE_LM_HEAD_HFQ2=1` (commit bc444998) completes the quant curve.
vs HFQ3 config, fold on: decode NEUTRAL (example p50 4.38-4.40 vs
4.40-4.41 ms; product 229.6 @64 / 224.0 @2048 vs 230.2/225.3 for HFQ3) —
gemv_hfq2g256's lower achieved bandwidth eats the byte savings. Quality:
coherent but a factual regression appeared on the planets prompt. The
shipped recommendation stays HIPFIRE_LM_HEAD_HFQ3=1 (231.0 tok/s).

## Addendum 6 (next day): fa_prep KV-write fold lever — +0.7-0.9% decode, shipped behind `HIPFIRE_FA_KVWRITE_FOLD=1`

Commit (this branch): folds the single-token Q8_0 K/V cache write into the
gfx1100 FA-prep epilogue (`qwen35_fa_prep_kvwrite_gfx1100`, own translation
unit). Grid 16Q + 2x4KV workgroups; the K workgroups quantize their finished
rope'd row after a trailing `__syncthreads()`, and NKV tail workgroups quantize
`fa_v` directly. Q8_0 arithmetic is expression-identical to
`kv_cache_write_q8_0_pair` (shfl_xor amax butterfly, f16 scale,
`__float2int_rn`, +-127 clamp, same cache layout), so cache bytes are
bit-identical. Removes one `kv_cache_write_q8_0_pair` launch per full-attention
layer. Gated OFF by default; requires non-compact decode route (pos_buf must
hold the physical position when the folded writer reads it).

Certification: `probe_fa_kvwrite` bitwise on fa_q/fa_gate/fa_k AND both cache
buffers; `test_kernels` 16/16; greedy temp-0 text byte-identical vs fold-off;
fresh-process example-path pairs all-positive across two windows
(OFF 198.6/199.4/199.2 vs FOLD 199.3/201.2/200.6; then order-alternating
256-token pairs OFF 197.9/199.3/197.7 vs FOLD 199.8/199.2/199.3 — five wins,
one tie, zero losses; +0.5-1.0%). Product `hipfire bench --kv-mode q8 --spec
off` x5v5: decode median 204.3 (stdev 0.29) OFF vs 204.1 (stdev 0.24) FOLD —
neutral within noise on the daemon sampling path today. Engagement proven by
decode profile on the serve path: pair kernel absent under FOLD=1,
launches 4496 -> 4368 per profiled segment, profiled wall 165.0 -> 161.0 ms.
Dispatch plumbing adds an attend-only twin
(`AttentionFamily::run_attend_only`) so `kv_cache_attention_dispatch` can skip
the write when the fold pre-wrote it.

## Addendum 7 (same): DN compact2 DPP-reduce port — REVERTED (graph-capture parity failure)

Ported the gfx1151 DPP-reduce compact2 variant to gfx1100 behind
`HIPFIRE_DN_COMPACT_FAST=1` (same source + defines, renamed kernel).
Direct-execution result was bit-exact: greedy temp-0 output byte-identical to
the shipping compact2_b2 over 300 tokens with `HIPFIRE_VERIFY_GRAPH=0`, and the
ef-residual probe scenario matched bitwise across multi-step state evolution.
BUT under the default graph-captured daemon path the output degenerated into
attractor garbage after ~40 tokens (reproducible, deterministic). Root cause not
isolated before time-box: something in the AR-forward capture/replay path
treats the newly added compact-GDN kernel module differently (kernarg blob
retention or frame patch are the suspects; note OFF itself differs graph-vs-
nograph because capture changes the GDN stochastic frame trajectory — with
production EF-residual state being deterministic, that alone should be inert).
Probe methodology note for future attempts: the ef=None (stochastic) probe arm
MUST bracket calls with `gdn_requant_frame_checkpoint()` /
`restore_gdn_requant_frame_checkpoint()` — the global `GDN_REQUANT_FRAME`
counter otherwise differs between arms and fakes byte diffs. Reverted without a
trace in the tree; measured +0.5-1% decode while it ran, so re-landing is
worthwhile once the capture-path root cause is found.
