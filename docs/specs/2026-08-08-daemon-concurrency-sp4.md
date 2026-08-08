# Daemon concurrency and admission control (SP4)

- **Date:** 2026-08-08
- **Base:** `feat/batched-attn-impl` (SP1 + SP2 + SP3)
- **Status:** design

## 1. Goal

The programme goal: **3-4 coding agents running concurrently on one R9700
(32 GB)**. SP1 built the kernels, SP2 the state, SP3 the forward pass and
scheduler. SP4 is what makes it reachable by a client — concurrent request
handling, per-slot sessions, admission control, and enforcement of the 32 GB
budget.

**This is the sub-project the user actually experiences.** Everything before it
is invisible from outside the process.

## 2. Starting point

`crates/hipfire-runtime/examples/daemon.rs` is ~14,000 lines and single-session:
one request at a time, JSONL over stdin, one global model state, no session map
(`grep -c "session" ... | grep -i "struct\|HashMap"` finds nothing).

SP4 does **not** rewrite it. It adds a concurrent path alongside the existing
sequential one, for the same reason SP3 adds a parallel forward entry point:
breaking the working single-user daemon is a far worse outcome than duplication,
and the existing path is what everyone uses today.

## 3. Components

### 3.1 Session table

`SessionId -> SlotId` plus per-session state: token history, sampling
parameters (SP2's `SlotSampleParams`), prefix-cache handle, and generation
status. A session outlives any single request — that is what makes multi-turn
agent conversations cheap, via prefix cache reuse.

### 3.2 Admission control

Decides whether a new session can be admitted, and what context it may claim.
This is where the 32 GB budget stops being advisory:

- Reject or queue when `SlotPool` is full.
- Reject when the requested context would exceed the remaining budget.
- Report the reason to the client rather than failing opaquely.

The budget arithmetic is already established and measured:

| model | weights | KV/token | 4 agents × 128K |
|---|---|---|---|
| qwen3.6:27b | 15.0 GB | 34 KB | 33.25 GB — **does not fit** |
| qwen3.6:35b-a3b | ~20 GB | 10.6 GB/128K | 25.8 GB — fits |

So on the 27B, admission must cap context: 4 × ~96K fits at 28.69 GB. **asym3
would have relaxed this but was rejected on quality** (SP2 gate: ~30% of top-1
token choices change), so the cap is real and not a temporary limitation.

### 3.3 Concurrent request handling

Accept and stream several requests at once, each mapped to a slot, each
streaming its own tokens back. The existing JSONL protocol gains a session or
request id on each frame; the single-session path is unchanged when only one
request is in flight.

### 3.4 KV swap-on-idle

Specified in the SP1 spec §15 and carried here. Page an idle slot's whole KV to
host and back on resume, so more sessions can be admitted than fit resident.

**Bounded by measurement, not enthusiasm:**
- **Per-step streaming is not viable and must not be built.** Every slot reads
  its entire KV every decode step — 4.56 GB/slot at 128K on the 27B, which is
  7.1 ms from VRAM against ~91 ms over PCIe. Layer-wise prefetch does not rescue
  it (5.7 ms transfer against 0.44 ms compute per layer).
- **Swap-on-idle is viable** because the transfer is paid once per activation
  and amortised over a whole generation. Coding agents are bursty: typically 1-2
  of 4 are decoding at any instant.

**Prerequisite before building it:** confirm 4 × ~96K is genuinely insufficient.
A swap subsystem is real work — eviction policy, transfer scheduling, and the
correctness risk of a half-swapped slot being read — and ~96K × 4 fits today.

**Hazard:** Strix Halo has unified memory, so a swap prototype shows near-zero
transfer cost on the dev box and falls off a cliff on the R9700 where it crosses
PCIe. Any latency claim needs R9700 access. This is the same hazard that
invalidated the expert-spill idea.

## 4. Non-goals

- No rewrite of the existing daemon path.
- No multi-node, no request routing beyond one process.
- No new quantisation or kernel work.
- Swap-on-idle is **conditional** on §3.2's prerequisite, not automatic.

## 5. Success criteria

1. Three to four concurrent clients each receive their own streamed tokens from
   one daemon on one GPU.
2. Admission control refuses over-budget requests with a clear reason rather
   than OOMing.
3. A session's second turn reuses its prefix cache.
4. The existing single-session path is byte-for-byte unaffected when only one
   request is in flight.
5. The 32 GB budget is enforced, not merely documented.

## 6. Risks

- **The daemon is large and load-bearing.** It is what the user runs daily; a
  regression here is immediately visible.
- **Memory discipline is not optional here.** The measured reality: the cgroup
  does **not** contain amdgpu GTT, so a runaway admission decision takes down the
  desktop rather than the process. Admission control *is* the memory gate in
  production, exactly as `preflight_alloc` is in the harnesses.
- **Testing concurrency on a shared dev box is hard.** Several resident sessions
  is precisely the state that exhausts this machine, and one resident model
  already takes MemAvailable from ~58 GiB to ~19 GiB.
- ~~**The end-to-end win is unmeasured.**~~ **Measured — see §7.**

## 7. Measured end-to-end throughput (2026-08-08)

First full-forward measurement, closing the §6 risk. gfx1151 (Strix Halo dev
box), `qwen3.6-35b-a3b.mq4r`, 4096-token context per slot, 512-token prefill
chunks, 48 generated tokens per slot, first decode step discarded as warmup.
Harness: `demo_multislot_generate` with `TARGET_PROMPT_TOKENS=4096
PREFILL_CHUNK=512`.

| slots | decode ms/step | aggregate decode | per slot | prefill (total) |
|---|---|---|---|---|
| 1 | 22.77 | 43.92 tok/s | 43.92 | 3.9 s |
| 2 | 34.38 | 58.17 tok/s | 29.09 | 7.6 s |
| 3 | 40.22 | 74.59 tok/s | 24.86 | 11.1 s |
| 4 | 47.82 | 83.65 tok/s | 20.91 | 14.6 s |

**1.90× aggregate at 4 concurrent agents.** Marginal cost is ~8.3 ms per added
slot against a ~14 ms fixed step cost, so the curve is still climbing at 4.

Read this as the shape of the win, not as an R9700 number: gfx1151 is unified
memory and gfx1201 is not, and the two arches take different attention
dispatches (see below). The programme's target hardware remains unmeasured.

### What this measurement found

Batching was initially *negative* at 2 users — 34.95 tok/s against 43.19 at 1
user — because every decode step with 2+ active slots dispatched the WMMA
flash-*prefill* kernel on a batch of `active_slots` rows, against a 16-row
tile. A flat ~21 ms per-step penalty. Fixed in `bbe244b5` by gating the tile
path on `n > active_slots` (prefill work remains) rather than on
`active_slots > 1`.

Two traps worth recording, both of which produced a confident wrong answer
before being caught:

- `HIPFIRE_FLASH_PREFILL=0` moves the reference arm *and* the candidate arm,
  so a golden-gate pass under it does not show that two kernels agree — it
  shows that one kernel agrees with itself.
- A diagnostic flag forcing the multi-slot path at one active slot produced a
  plausible-looking 22.75 ms/step, from a state
  (`single_slot=None, n_tiles=None, eligible=true`) production never reaches
  and whose output was wrong by 225× tolerance. Reverted rather than kept.

### Confirmed, not projected

- The gfx1201 dispatch outcome — `q8_flash_prefill_wmma_eligible` requires
  `!has_wmma_w32_gfx12()`, so on the R9700 the tile path never fires and the
  descriptor path runs unaided — passes the golden gate 10/10 at 0.000×. The
  deployment target's path is correct, though still unmeasured for speed.
- Prefill genuinely wants the WMMA kernel: 14.6 s against 19.2 s at 4 slots
  with the path disabled outright.

### Batched lm_head (done)

The per-slot `rmsnorm + Step::Gemv` loop was the last per-slot-serialised op;
everything else was already one launch across all rows. `gemv_hfq4g256_xbatch`
makes it one weight pass over B activation vectors.

| slots | before | after | ms/step |
|---|---|---|---|
| 1 | 43.92 | 42.33 tok/s | unchanged path (noise) |
| 2 | 58.17 | 59.98 tok/s | 34.38 → 33.34 |
| 3 | 74.59 | 77.24 tok/s | 40.22 → 38.84 |
| 4 | 83.65 | **89.32 tok/s** | 47.82 → 44.79 |

**Aggregate scaling is now ~2.1× at 4 concurrent agents.**

Correcting an earlier claim in this document's follow-up list: the lm_head was
described as "roughly the whole ~8.3 ms marginal per-slot cost". It is not.
`dim` is 2048, not 4096, so the weight pass is 270 MiB ≈ 1.27 ms per slot —
about 15% of the marginal. The rest is MoE expert traffic (~3B active params
per token ≈ 1.6 GB), which is irreducible here: four slots picking top-8 of 256
experts overlap by only ~5%, so there is no dedup win to take.

It was still worth doing, for a reason the arithmetic does not show: it was the
only structurally serial op left in the forward.

### Profile of a 4-slot decode step (rocprofv3)

Kernel trace, 35B-A3B, 4 slots, 4096-token context, decode phase isolated by
timestamp. After the fixes below, 36.2 ms/step wall, ~35.4 ms GPU busy,
**1393 dispatches per step**:

| kernel | ms/step | % | calls | us/call |
|---|---|---|---|---|
| `gemm_hfq4g256_moe_grouped_mmq` | 12.63 | 35.7 | 80 | 158.0 |
| `attention_q8_0_kv_batched` | 5.86 | 16.6 | 10 | 586.4 |
| `gemm_qkvza_hfq4g256_wmma` | 2.94 | 8.3 | 30 | 98.3 |
| `gated_delta_net_q8_fast` | 2.81 | 7.9 | 120 | 23.4 |
| `gemm_hfq4g256_residual_wmma_k2` | 2.33 | 6.6 | 40 | 58.3 |
| `gemv_hfq4g256_xbatch` (lm_head) | 1.49 | 4.2 | 1 | 1492.0 |
| launch gaps (GPU idle) | 5.08 | 12.6 | — | — |

Two findings came out of it, both landed:

1. **`sigmoid_mul_f32` ran over the whole batch buffer.** It takes its element
   count from the tensor handed to it, and both FA call sites passed the full
   `fa_attn_out_batch` — sized for the largest prefill batch. A 4-row decode
   step did a full prefill chunk's work: 8388608 elements, 427.6 us x 10 calls
   = 4.28 ms/step on 2044 dead rows. The reference has the same pattern but
   only reaches it during prefill, where `n` really is the batch.
2. **The flash-attention crossover was on the wrong side.** See the commit; the
   scalar kernel launches only 64 workgroups and ran at ~30 GB/s.

### Aggregate decode throughput, 4096-token context

| slots | SP4 first measurement | after decode-tile fix | after lm_head batching | after gate + crossover |
|---|---|---|---|---|
| 1 | 43.19 | 43.92 | 42.33 | **53.34** |
| 2 | 34.95 | 58.17 | 59.98 | **75.47** |
| 3 | 48.57 | 74.59 | 77.24 | **93.95** |
| 4 | 58.31 | 83.65 | 89.32 | **110.42** |

**1.89x at 4 concurrent agents over the session's starting point**, and scaling
across slots is 2.07x.

### Follow-ups

- The tile path's profitability crossover between `WMMA_M_TILE` and full
  prefill chunks is unmeasured; only the two endpoints are known.
- `gemv_hfq4g256_xbatch` caps at B=4 (`HIPFIRE_HFQ4G256_XBATCH_MAX`, which
  sizes its accumulator arrays). Beyond 4 slots the path falls back to the
  per-slot loop rather than chunking.
- Q8_0 lm_heads (the dense 4B) have no batched path; only MQ4G256 does.
- The B=1 case is a 0.96× loss, so it is gated off. Whether a batched kernel
  could also win at B=1 with the tuned kernel's buffer-load strategy is
  untested.
- Contexts beyond 4096 and slot counts beyond 4 are unmeasured.
- **Launch gaps are now the third-largest term: 5.08 ms/step, 12.6% of wall,
  across 1393 dispatches per step.** hipGraph capture of the decode step is the
  obvious next target and is the largest remaining non-MoE item.
- MoE grouped MMQ is 35.7% of the step at ~120 GB/s effective. It is the floor
  for this model — four slots picking top-8 of 256 experts overlap only ~5% —
  but the bandwidth gap suggests some headroom in the kernel itself.
- The gfx1151 crossover is measured at 4096 tokens and a wash near 1086; the
  region between is unmeasured, and no other arch was measured at all.
