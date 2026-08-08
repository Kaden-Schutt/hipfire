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

### Follow-ups

- The tile path's profitability crossover between `WMMA_M_TILE` and full
  prefill chunks is unmeasured; only the two endpoints are known.
- Per-slot lm_head is a separate `Step::Gemv` per slot over a 248320-row
  vocabulary — roughly the whole ~8.3 ms marginal per-slot cost. Batching it
  into one GEMM over `n_slots` rows is the obvious next win and would flatten
  the marginal term.
- Contexts beyond 4096 and slot counts beyond 4 are unmeasured.
