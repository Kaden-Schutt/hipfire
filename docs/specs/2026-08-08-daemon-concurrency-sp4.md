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
- **The end-to-end win is unmeasured.** SP1 measured only the attention term
  (~1.36× at 8 slots). Aggregate throughput across a full forward — where weight
  reads amortise and MoE expert reads mostly do not — is still projected, not
  measured. SP4 is where it finally becomes measurable.
