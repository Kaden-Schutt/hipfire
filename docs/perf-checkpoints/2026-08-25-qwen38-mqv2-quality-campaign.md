# Qwen3.8 MQ V2 quality campaign — evidence and invalidated BenchLocal methodology

**Date:** 2026-08-25  
**Lifecycle:** `historical`  
**Disposition:** **KLD, HumanEval, and transport measurements retained. BenchLocal tier/effort scores rejected.**

This is fixture-bound evidence from the Qwen3.8-27B MQ V2 quality campaign. It
is not a current product baseline, standing admission decision, SLA, or claim
that the rejected BenchLocal totals compare quant tiers. Newest file != current
baseline.

## Scope

The campaign measured the MQ3V2–MQ6V2 XT/base/Pro product ladder, added native
Qwen3.8 reasoning and OpenAI tool controls, exercised a multi-GPU BenchLocal
campaign on hiptrx, and replaced hipfire's native HTTP transport after the
campaign exposed cancellation/admission failures.

The model artifacts and measured KLD/HumanEval results remain valid. The
BenchLocal effort/tier comparison changed token and timeout contracts during
measurement and is rejected in full, including superficially favorable MQ4-XT
rows.

## Retained quant evidence

### MQ4 tier artifact evidence

| tier | size (GB) | effective bpw | WT2 teacher KLD |
|---|---:|---:|---:|
| MQ4-XT | 14.980 | 4.456 | 0.057449 |
| MQ4 base | 15.663 | 4.659 | 0.039033 |
| MQ4 Pro | 16.464 | 4.897 | 0.032495 |

KLD is monotonic in the expected direction.

### HumanEval 164 pass@1

| tier | AR | MTP | DFlash |
|---|---:|---:|---:|
| MQ4-XT | 156/164 | 157/164 | 157/164 |
| MQ4 base | 160/164 | 160/164 | 159/164 |
| MQ4 Pro | 160/164 | 160/164 | 160/164 |

These 36 product × decode-mode arms used the committed HumanEval method already
recorded by the product ladder. They are the retained task-quality comparison.

## Why BenchLocal was rejected

Two incompatible reasoning contracts were mixed.

The early MQ4-XT run used natural endpoint-bounded reasoning:

- `thinking_max_tokens=0`;
- case timeout 86,400 seconds;
- timeout ceiling disabled;
- model-turn timeout disabled.

Later MQ4-XT effort runs and MQ4 base/Pro runs used pack/default or repaired
limits around 16K output and 300/600-second request deadlines. Successful early
XT and external comparator trajectories exceeded both limits, so the bounded
runs could not reproduce their solution paths.

The harness also contained independent retry multipliers:

- scenario failure retries;
- HTTP transient retries for 5xx responses;
- replacement-process retries.

A deterministic open-think validation 500 could run four times at roughly 516
seconds per request before the scenario returned, and then repeat in a fresh
replacement process. Static shard assignment concentrated such cases on one
GPU while others idled. Invalidation initially discarded successful replacement
overlays on resume. Docker sandbox port conflicts caused skipped rows and extra
repair waves.

The original telemetry proxy blocked in `HTTPConnection.getresponse()` before
upstream headers, so a downstream timeout could not close the native upstream
connection. The abandoned request retained single-daemon admission and caused
later 503s. This was a control-plane failure, not quantization evidence.

**Rejected:** all BenchLocal aggregate/tier/effort scores from this campaign.
They may be used only as forensic pointers to the transport and harness defects.

## Reasoning-token forensic evidence

The following are raw thinking-token counts from three cases where MQ4-XT low
passed. Comparator runs used unseeded temperature-1 trajectories, so these are
not minimum required budgets and are not causal quant comparisons.

| model / effort | LCBv6-3688 | LCBv6-3696 | LCBv6-3771 |
|---|---:|---:|---:|
| MQ4-XT low | 9,013 pass | 13,308 pass | 27,316 pass |
| UD-Q4_K_M low | 39,380 pass | 15,341 pass | 37,890 pass |
| UD-Q4_K_M medium | 27,873 pass | 12,686 pass | 20,588 pass |
| UD-Q4_K_M xhigh | 79,516 pass | 23,813 pass | 28,502 pass |
| UD-Q4_K_XL low | 13,473 pass | 26,479 pass | 23,654 pass |
| UD-Q4_K_XL medium | 10,991 pass | 19,092 pass | 10,935 pass |
| UD-Q4_K_XL xhigh | 62,743 pass | 23,027 pass | 30,224 pass |
| AutoRound low | 15,111 pass | 22,467 pass | 36,221 fail; retry 14,589 pass |
| AutoRound medium | 16,989 pass | 17,095 pass | 15,805 pass |
| AutoRound xhigh | 67,157 pass | 17,670 pass | 47,492 pass |

On matched LCB pass@1 rows, XT used fewer completion tokens in 15/26 pairs
versus Q4_K_M, 14/25 versus Q4_K_XL, and 14/26 versus AutoRound. The advantage
was not universal; it was driven by several large heavy-tail savings.

A universal 16K ceiling is incompatible with successful trajectories across
all four formats. Medium/xhigh token usage is not monotonic on individual
unseeded runs. Future reasoning-budget work requires fixed-seed, prompt-identical
native traces rather than aggregate reruns.

## Durable product changes

### Native reasoning and OpenAI tools

- `b482d9514` — model-native reasoning controls;
- `6187bd10e` — schema-aware tool-argument normalization;
- `3307ccf63` — OpenAI `tool_choice` semantics.

The reasoning contract keeps semantic effort independent from an explicit
integer think cap and reports effective policy/warnings. Tool normalization
repairs values according to request schemas on both streaming and non-streaming
paths.

### Native HTTP and cancellation

- `709c3f2be` — first non-stream cancellation transaction; later superseded as
  the transport solution;
- `158ff5ec6` — replace tiny_http with Hyper/Tokio, remove the local tiny_http
  fork, add cancellable async admission, socket-flush terminal acknowledgement,
  queued-future cleanup, streaming/non-stream disconnect propagation, and
  multi-slot dropped-client handling.

Validation on `158ff5ec6`:

- hipfire-client: 49 tests passed;
- hipfire-cli: 194 tests passed;
- direct real-model disconnect → immediate follow-up 200 in 0.17 s;
- proxied disconnect → follow-up 200 in 0.19 s, queue depth 0;
- serve-harness battery and chain produced coherent decoded output;
- no orphan hipfire/daemon/container remained after validation.

Deployed hiptrx binaries for the final transport validation:

- `hipfire` SHA-256 `53e59f25595abde8f1a48fa9edabde66edae42c0f480412e35107fc84528a19f`;
- `daemon` SHA-256 `97b44e667ce2326e5417c218768efbde46361094e9bf0bc2cf16bf2cff68fe00`.

### Transport performance screen

Same host, model, device, request bytes, and new TCP connection per request:

| measurement | tiny_http p50 | Hyper p50 | disposition |
|---|---:|---:|---|
| `/health` | 0.131 ms | 0.054 ms | Hyper about 2.4× faster |
| one-token generation | 69.856 ms | 69.428 ms | no regression; ~0.6% lower |

GPU generation remained in the prior band; the gain is control-plane latency
and reliability, not a kernel throughput claim.

### Developer harness

The external campaign harness gained isolated GPU homes/ports, immutable shard
attempts, readiness gating, exact-union atomic merges, SSE telemetry, downstream
FIN propagation before and after upstream headers, and explicit case/model-turn
deadline projection. These are developer tools, not user-facing runtime
surfaces.

## Decisions

- Publish KLD, HumanEval, artifact size/bpw, and fixed-shape performance evidence.
- Do not publish or cite the campaign's BenchLocal totals.
- Do not infer a Q8 kernel defect from open-think/token-limit failures.
- Do not integrate Gigatoken for inference: the prior hipfire measurement was
  about 0.1 tok/s at ~20K context, below the maintenance threshold.
- Coordinate future multi-slot parity changes with open PR #609; #626 is
  superseded and #636 is the separate VL work.
- If reasoning benchmarking resumes, use exact fixed-seed native traces with one
  frozen token/time contract and no hidden retry layers.

## Discovery pointers

These host-local paths are not durable authorities and may rotate:

- hiptrx campaign results: `/home/kaden/qcal/qwen38-quality-results/`;
- campaign harness checkout: `/home/kaden/qcal/qwen38-quality-current/`;
- model-card staging: `/home/kaden/qcal/modelcard-refresh/card/`.

This ledger entry is the durable disposition. It does not authorize promoting
rejected BenchLocal totals into `docs/BENCHMARKS.md` or a model card.
