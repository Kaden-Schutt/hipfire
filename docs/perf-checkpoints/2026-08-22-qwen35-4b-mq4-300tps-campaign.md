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
