# MTP K=5 cycle anatomy on canonical 27B-3.5 LRU bench (gfx1100)

> Empirical breakdown after asym3 KV swap landed (commit 65412bf3).
> **The dispatch-fusion optimization opportunity I projected is a red
> herring. The real lever is the verify+replay round-trip.**

## Measurement (3-run mean, K-sweep delta)

| K | tok/s | tau   | cycle wall  | per-K delta |
|---|-------|-------|-------------|-------------|
| 1 | 26.29 | 1.95  | 74.8 ms     | —           |
| 5 | 45.38 | 3.61  | 80.1 ms     | **1.3 ms / extra MTP block** |

Each additional MTP block forward (block_only at hidden=5120, head_dim=256,
n_ff=17408, all MQ4G256) costs only **1.3 ms**. MTP block work at K=5 is
~6.5 ms total — **only 8% of the cycle wall.**

## Where the other 73 ms goes

Inferred from K=1 base wall = 74.8 ms (with only 1 MTP block, ~1.3 ms):

| Component                                           | Estimate |
|-----------------------------------------------------|----------|
| Trunk batched verify forward (n=K+1, all DN layers) | ~30 ms   |
| Trunk batched lm_head GEMM at n=K+1 + argmax        | ~15 ms   |
| Trunk DN snapshot + restore                         | ~5 ms    |
| **Trunk KV rollback REPLAY (forward_prefill_batch on advance tokens)** | **~15-20 ms** |
| MTP K=1 forward (block + compressed lm_head)        | ~3 ms    |
| Argmax + D2H + bookkeeping                          | ~3 ms    |
| **Total**                                           | **~73 ms** ✓ |

## What this means for "reduce MTP FFN dispatch cost (75% of cycle)"

It's wrong. The 75% figure was based on my mistaken estimate of ~16 ms
per MTP block. Actual per-block is 1.3 ms. **MTP-side dispatch fusion
(QKV fuse, gate+up fuse, etc.) tops out at ~3 ms savings = +4% perf.**
Not worth the load-time + format-change complexity.

## The real lever: verify+replay redundancy

The current spec-step pattern (mtp_spec.rs lines 916-950 in
`spec_step_mtp_compressed_serial`):

```rust
// 5. capture prev_hidden from verify_hidden[advance-1]
state.capture_prev_hidden_from_verify_row(gpu, advance - 1, dim)?;

// 6. UNCONDITIONAL restore + replay
state.trunk_snap.restore_to(&mut target.dn_state, gpu)?;
if advance >= 2 {
    let replay = &verify_tokens[..advance];
    qwen35::forward_prefill_batch(..., replay, cur_pos, ...)?;
} else {
    qwen35::forward_scratch(..., verify_tokens[0], cur_pos, ...)?;
}
```

**Always restores DN snapshot + replays accepted tokens through trunk
forward, even when `advance == K+1` (full chain accepted)**. The verify
pass already advanced trunk DN/KV correctly to position `cur_pos +
K+1` in the full-accept case — restore+replay is pure waste.

Probability of full-accept at K=5 with tau=3.61: rough estimate
~20-30% of cycles hit `advance == K+1`. Saving ~15-20 ms on those
cycles = ~5 ms average savings = **+6-8% perf** at K=5.

## Proposed optimization (gated on careful state-machine review)

```rust
if advance < max_n + 1 {
    state.trunk_snap.restore_to(&mut target.dn_state, gpu)?;
    if advance >= 2 {
        let replay = &verify_tokens[..advance];
        qwen35::forward_prefill_batch(..., replay, cur_pos, ...)?;
    } else {
        qwen35::forward_scratch(..., verify_tokens[0], cur_pos, ...)?;
    }
}
// else: verify already left trunk at correct state for next cycle
```

Requires verifying that:
1. After verify forward at n=K+1, trunk KV cache is correct at all K+1
   positions (= the verify_tokens we just ran)
2. After verify, trunk DN state is at `cur_pos + K+1` (= `cur_pos +
   advance` when advance == K+1) — ready for next cycle's verify
3. `state.prev_hidden` was captured from verify_hidden BEFORE any
   restore — already true at line 916 (capture happens before restore)

If those hold, skipping the restore+replay on full-accept is sound.

## Why this hasn't been done

The same pattern lives in `spec_step_mtp` and `spec_step_mtp_compressed`
(the lossy-chain variants). Probably reflects copy-paste from the
DFlash spec-decode pattern, where the rollback is needed because tree
verify writes per-tree-slot K/V that must be rolled back to the
accepted prefix. MTP's serial K-step doesn't have that constraint;
verify writes K/V positionally and the accepted prefix coincides with
the FIRST `advance` positions written.

## Other observations

* asym3 KV swap (commit 65412bf3) is correct + neutral on canonical
  short-ctx perf (within DPM noise). Saves ~6.4x KV memory and brings
  MTP attention into parity with trunk's path. Long-ctx win deferred
  pending the MTP attractor bug at high position (separate issue —
  AR-baseline coherent on same long prompt, MTP path goes degenerate).
* The K=5 sweet spot (49 tok/s with v1 sidecar earlier this session,
  45 tok/s post-asym3) is set by tau saturation at K=5 (~3.6-3.8) AND
  the verify+replay floor, not by MTP block work.
* Path to the 60 tok/s gate from current ~45-49 tok/s with code-tuned
  sidecar: skip-replay-on-full-accept (+5-8%) + EAGLE-style head
  retrain (+30-50% via tau lift). Still not enough on its own; the
  DFlash drafter-replacement composition (docs/plans/dflash-mtp-
  composition-orthogonal.md) becomes the next architectural step.

## What to do next

Pri 1: implement skip-replay-on-full-accept (1-2 hr including coherence
gate). Apply to all three spec_step_mtp* variants for consistency.

Pri 2: re-investigate the long-ctx MTP `!!!!!` attractor — likely a
position-handling bug in the asym3 swap OR a pre-existing MTP head
issue. AR-baseline at long ctx is coherent (43.66 tok/s on 8.1K-token
Python stdlib prompt), so it's MTP-path-specific.

Pri 3: defer FFN dispatch fusion entirely. The lever is too small.
