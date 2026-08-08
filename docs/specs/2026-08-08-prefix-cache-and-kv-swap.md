# Session prefix cache and KV swap (SP5 + SP6)

- **Date:** 2026-08-08
- **Base:** `feat/batched-attn-impl` (SP1–SP4)
- **Status:** design

## 1. Goal

Two capabilities that together make a session cheap to leave and cheap to come
back to:

- **SP5 — prefix cache.** A session's second turn reuses the KV it already
  built, prefilling only the tokens that diverge. This is SP4 success criterion
  #3, which SP4 shipped without.
- **SP6 — KV swap.** An idle session's state survives losing its slot, so the
  daemon can hold far more sessions than the four that fit in VRAM.

They are one design because SP6 exists to serve SP5: without swap, an idle
session that loses its slot loses its cache and re-prefills from scratch.

Target: **30+ concurrent sessions against 4 resident slots**, tiered
VRAM → host RAM → disk.

## 2. Prior art

`daemon.rs` already does this for one session. On a new turn it computes the
longest common prefix (LCP) over `conversation_tokens`, rewinds the KV cursor to
that point, and re-prefills only the divergent suffix (`ep_serve_minimax` and
`generate_minimax` carry the rationale). SP5 is that mechanism generalised from
one session to N slots, not a new invention.

`SlotPool` gives each slot **one flat contiguous slab**. `PAGE_TOKENS = 128`
only rounds capacity so "a future page size divides them" — paging was
anticipated and never built. Same-session reuse works with this layout as-is;
cross-session *sharing* would not (see §10).

## 3. Governing invariant

**The tokens are authoritative; KV is only a cache.**

Every session owns a `Vec<u32>` of its conversation tokens. KV and DeltaNet
state are derived data, always recomputable by re-prefilling. Every failure path
— bad write, missing file, stamp mismatch, torn read, disk full — degrades to
"re-prefill from tokens". Slow, never wrong.

There must be no path that yields silently incorrect output. That is the only
failure class that matters here: a wrong logit surfaces as a subtly worse agent,
not as an error.

**The swap unit is `{KV slab, DeltaNet state, seq_len, tokens}` — all four or
none.** 35B-A3B has 30 linear-attention layers whose recurrent state lives
outside the KV arena. A session restored with correct KV and stale DN state
produces wrong output with nothing raised.

**Residency is a property of the session, not the slot.** A restored session may
land in a different slot than it left. The descriptor table already makes KV
addressing slot-agnostic; DN state is copied into the new slot's buffer. Nothing
may assume slot affinity.

## 4. Sizing (measured)

35B-A3B, from the SP3/SP4 preflight accounting:

| quantity | value |
|---|---|
| KV per token per slot | 10 FA layers × 2 × 544 B = **10.88 KB** |
| KV at 4096 tokens | 44.6 MB |
| KV at 128K (slab reservation) | 1.39 GB |
| DeltaNet state per slot | **48.3 MB, fixed** |
| 27B KV per token | 34 KB → 3.26 GB at 96K |

Two consequences:

- **Transfer only the live prefix.** A session 4K tokens into a 128K slab is
  44.6 MB of KV, not 1.39 GB. Swap cost scales with conversation length, not
  with the reservation — this is what makes a 30-session pool practical.
- **DN state is fixed and dominates early.** Below ~4.4K tokens the 48.3 MB DN
  state is the larger half of the swap. Short sessions are not free to park.

Against this, re-prefilling 4096 tokens costs **3.9 s** (measured, 1 slot,
2026-08-08). Restoring the same session moves ~93 MB. Even at a pessimistic
1 GB/s that is ~0.1 s — an order of magnitude better than recomputation, and the
gap widens with context length because prefill grows superlinearly while
transfer is linear.

## 5. Components

### 5.1 `SessionStore` (extends `SessionTable`)

```
Session { id, tokens: Vec<u32>, residency, slot: Option<SlotId>, last_used }
Residency = Resident | Host | Disk | Cold
```

Owns LCP. **Design note for §10:** LCP is computed against *a token sequence*,
not against "this session's tokens", so a future shared-prefix table becomes a
second lookup source rather than a rewrite.

### 5.2 `SlotSnapshot`

The serialised swap unit:

- **Header/stamp:** model hash, KV dtype, `per_pos_bytes`, `n_fa_layers`, DN
  layout version, `cap`, `seq_len`, token count, payload checksum.
- **Payload:** per FA layer, `seq_len × per_pos_bytes` of K then V; then the DN
  state.

The stamp carries everything persistence would need (§10), so enabling it later
is a policy change, not a format change.

### 5.3 `SwapManager`

Tier placement (host until a byte budget, then disk), LRU over **idle sessions
only** — a session mid-generation is never evicted — and a write-back worker
thread.

Host budget defaults to **16 GiB** and is overridable. It is a budget, not a
reservation: snapshots are allocated on demand and freed on session close. The
figure comes from the sizing above — 30 sessions averaging 8K tokens is ~4 GB,
so 16 GiB absorbs a realistic pool while leaving a 125 GiB box room for the
model (~18 GiB resident) and everything else. Beyond it, snapshots go to disk.

### 5.4 `AdmissionController` (extended)

Today it gates VRAM. It gains a host-tier byte budget and refuses a session when
neither VRAM nor host+disk can hold it. Admission *is* the production memory
gate, because the control group does not contain amdgpu GTT. Host-tier buffers
are ordinary allocations, so the existing cgroup gate does cover them — unlike
the GTT case that took down the desktop during SP1.

## 6. A turn

1. `lcp = LCP(session.tokens, prompt)`
2. Ensure resident: free slot, else evict LRU idle (enqueue async write-back,
   mark non-resident immediately), then restore blocking.
   **If no slot is free and none is idle** — all four mid-generation — the
   request queues until one frees, bounded by a timeout, then is rejected with a
   reason. It never preempts a generating session, and it never thrashes: the
   caller is told the daemon is saturated rather than being served slowly.
   Restore only `min(lcp, snapshot.seq_len)` tokens — a turn that diverges early
   transfers less.
3. `pool.set_seq_len(slot, lcp)`; truncate `session.tokens` to `lcp`. **The
   rewind is free** — no data movement, because KV past `lcp` is overwritten by
   the next prefill.
4. Prefill `prompt[lcp..]` through the existing chunked scheduler.
5. Decode, appending to `session.tokens`.

## 7. Failure handling

**One rule: validate in host memory, then commit to VRAM in a single pass.**

- Stamp and length checked before any payload is read.
- Payload checksum verified in the host buffer, before the H2D copy.
- A live slot is therefore never left half-written.
- Any failure marks the session `Cold`; the next turn re-prefills from tokens.
- Write-back still in flight when the session is requested again: serve from the
  host buffer still held, rather than waiting on the file.
- Disk full or IO error: drop the snapshot, mark `Cold`, log. Never fatal.

## 8. Testing and gates

1. **Round-trip, both tiers.** Capture → restore → KV bytes and DN bytes
   **bitwise identical**. Negative control: flip one payload byte, restore must
   be *refused*, not silently accepted.
2. **Stamp mismatch refused.** Wrong model hash / KV dtype / layout version.
3. **Swap equivalence.** N sessions > slots, forcing eviction and restore;
   greedy output token-for-token identical to a no-swap control.
4. **Prefix-cache equivalence.** A two-turn conversation with reuse produces
   identical tokens to a cold full prefill.
5. **Memory.** Every allocation path through `preflight_alloc`; harnesses run
   under `scripts/run-bounded.sh`.

Existing gates must stay green: `test_forward_slots_golden` at 0.000×,
`kernel_resource_gate.sh` and `attn_legacy_baseline.sh` bitwise identical to
beta.

## 9. Measurement plan, and one hazard

Measure: resume latency host vs disk vs re-prefill, at 4K / 16K / 64K tokens;
eviction cost; steady-state throughput with 4 active of 30 sessions.

**Hazard: Strix Halo has unified memory. Host-tier transfer costs measured on
this box will not transfer to the R9700**, where the same copy crosses PCIe.
This is the hazard that invalidated the expert-spill idea. Host-tier latency
claims stay explicitly provisional until R9700 access. Disk-tier numbers roughly
do transfer — NVMe is NVMe.

## 10. Non-goals

- **Cross-session prefix sharing.** Four agents sharing one system prompt would
  need KV paging (a rewrite of `SlotPool` and every `kv_offset` call site) or
  copying the shared prefix into each slab. §5.1's LCP interface leaves room for
  it; nothing else here does.
- **Persistence across daemon restart.** Spill files are scratch: deleted on
  session close, directory cleared at startup. The stamp is written anyway so
  this can become a policy change later.
- No changes to the forward pass, kernels, or scheduler.
- No compression of swapped KV.

## 11. Phasing

Two implementation plans, in order:

- **Phase 1 (SP5).** `SessionStore` + LCP + rewind. Useful on its own at
  sessions ≤ slots, and independently testable via gate #4.
- **Phase 2 (SP6).** `SlotSnapshot`, `SwapManager`, tiering, admission
  extension. Depends on Phase 1's residency model.

## 12. Risks

- **DN state is easy to forget.** It is not in the KV arena and it is fixed-size,
  so a KV-only implementation would pass a short smoke test and fail subtly on
  long conversations. Gate #1 covers DN bytes explicitly for this reason.
- **Async write-back is the one concurrency addition.** A session evicted and
  immediately re-requested is the race; §7 resolves it by serving from the
  retained host buffer.
- **LRU over idle sessions can thrash** if more than four sessions are
  continuously active. Out of scope to solve; admission should reject rather
  than thrash, and the measurement plan should show whether it happens.
