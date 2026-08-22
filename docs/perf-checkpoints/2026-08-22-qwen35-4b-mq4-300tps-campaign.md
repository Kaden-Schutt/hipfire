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
