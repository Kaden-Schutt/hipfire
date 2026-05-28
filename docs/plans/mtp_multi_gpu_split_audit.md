# MTP hetero split — per-call audit + swimlane (2026-05-28)

Audit of `spec_step_mtp_compressed_serial` (the path mtp_only_demo
drives) to classify every `gpu.X` call as **TRUNK** (must run on
target_gpu where the trunk model lives) or **DRAFTER** (could run on
sibling drafter_gpu where the MTP head would live).

## Critical constraint discovered

The MTP chain depends on trunk weight tensors per step:

| weight             | shape                            | MQ4 size | where used                  |
| ---                | ---                              | ---      | ---                         |
| `trunk.token_embd` | `[vocab, n_embd]` (248320×5120)  | ~640 MB  | every chain step (embed)    |
| `trunk.output`     | `[vocab, n_embd]` (248320×5120)  | ~640 MB  | every chain step in `use_full_vocab` mode |

Both are read-only during decode. **Resolution:** mirror onto
drafter_gpu at session init.

**Picked compressed-sidecar path (user decision 2026-05-28):** the
sidecar `.mtp` file (`qwen3.6-27b-cvs16384.mtp`,
`compressed_vocab_size=16384`) makes the chain's lm_head dispatch hit
`head.weights.lm_head_draft` (drafter-local, ~80 MB), eliminating the
`trunk.output` mirror requirement. Only `trunk.token_embd` (~640 MB)
needs mirroring.

gfx1031 budget after mirror:
12 GB - (MTP head ~800 MB) - (token_embd mirror ~640 MB) = ~10.5 GB free.

The bundled `.mq4-mtp` path (use_full_vocab=true, requires BOTH
mirrors) is supported by the audit but not the v1 target.

## Swimlane (one MTP spec cycle, K=4)

```
   target_gpu (gfx906)                            drafter_gpu (gfx1031)
   ================                               ====================
                                                  prev_hidden: F32[dim]
                                                  mtp_head weights (~800 MB)
                                                  token_embd MIRROR (~640 MB) [init-time]
                                                  output MIRROR (~640 MB)     [init-time, use_full_vocab only]
                                                  mtp_scratch + mtp_kv
                                                  mtp_t_outs / mtp_lm_logits

   ── cycle entry ──
   (verify_hidden row dim*4 B
    holds the per-cycle prev_hidden source)
   record verify_done_evt ────────────────────→  wait verify_done_evt
                                                 [first cycle only: prev_hidden seeded
                                                  from prefill — same problem]
                                                 peer_copy_async prev_hidden 20 KB
                                                 sync scatter_stream

   ── K-step draft chain (DRAFTER ONLY) ──
                                                 for k in 0..K:
                                                   mtp_head_forward_block_only(next_tok,
                                                     prev_row, cur_pos+k, MIRRORED trunk_weights)
                                                   rmsnorm + weight_gemv(MIRRORED output)
                                                   argmax → candidates[k] (16 B D2H to host)
                                                   memcpy_dtod t_mtp_out → mtp_t_outs[k]

   ── candidates ride back via host (4*4 = 16 B, free) ──
   candidates: Vec<u32> on host          ← (D2H already in chain loop)

   ── trunk verify (TRUNK ONLY) ──
   trunk_snap.save
   forward_prefill_batch_with_logits(verify_tokens)
     → verify_hidden, verify_logits
   argmax verify_logits → predicted
   compare predicted vs candidates → advance N
   trunk_snap.restore + replay tape advance positions

   ── cycle exit: ship next prev_hidden source ──
   advance := number accepted (1..=K+1)
   verify_hidden_row = verify_hidden[advance-1] (dim*4 = 20 KB)
   record next_verify_done_evt ──────────────→  wait next_verify_done_evt
                                                peer_copy_async row 20 KB into prev_hidden
                                                  (overwrites for next cycle)
```

## Classification table (every gpu call in spec_step_mtp_compressed_serial)

| line | call                                       | lane    | notes                              |
| ---  | ---                                        | ---     | ---                                |
| 1347 | `stream_create` (active_stream init)      | both    | both gpus need their own           |
| 1412 | `mtp_head_forward_block_only`             | DRAFTER | needs token_embd mirror            |
| 1418 | `mtp_head_forward_block_only`             | DRAFTER | "                                  |
| 1427 | `rmsnorm_f32(t_mtp_out, shared_head_norm)` | DRAFTER | shared_head_norm is part of head   |
| 1433 | `weight_gemv(trunk.output, tmp, logits)`  | DRAFTER | NEEDS output mirror                |
| 1441 | `mtp_head_forward_compressed`             | DRAFTER | needs token_embd mirror (k=0)      |
| 1447 | `mtp_head_forward_compressed`             | DRAFTER | "                                  |
| 1472 | `sample_top_p`                            | DRAFTER | local scratches                    |
| 1495 | `memcpy_htod(idx)` + `softmax_prob_gather` | DRAFTER | local                              |
| 1509 | `memcpy_dtoh(p_draft)` (4 B)              | DRAFTER → host | tiny, free                  |
| 1524 | `topk_logsumexp_batched_f32`              | DRAFTER | local (p_min path)                 |
| 1534 | `memcpy_dtoh(idx_host)` (8 B)             | DRAFTER → host | tiny                       |
| 1538 | `memcpy_dtoh(logp_host)` (8 B)            | DRAFTER → host | tiny                       |
| 1557 | `argmax_f32_batched`                       | DRAFTER | local (greedy path)                |
| 1565 | `memcpy_dtoh(argmax)` (4 B)               | DRAFTER → host | tiny                       |
| 1578 | `memcpy_dtod_at(mtp_t_outs ← t_mtp_out)`  | DRAFTER | local D2D                          |
| 1593 | `trunk_snap.save_from(target.dn_state)`   | TRUNK   | trunk only                         |
| 1606 | `forward_prefill_batch_with_logits`       | TRUNK   | trunk verify                       |
| 1620 | (verify lm_head call inside above)        | TRUNK   | trunk                              |
| 1627 | `weight_gemv(trunk.output, ...)` (if any) | TRUNK   | trunk lm_head, real one            |
| 1860 | `forward_one_token` (replay)              | TRUNK   | trunk only                         |
| 1874 | `forward_prefill_batch_with_logits` (replay) | TRUNK | trunk only                        |
| 1880 | `forward_one_token` (replay)              | TRUNK   | trunk only                         |

## Peer-copy points (per cycle)

- **0:** prev_hidden seed at cycle entry — 20 KB, 906→1031, gated by trunk's "verify_done from previous cycle" event. On cycle 1 (no previous verify), seeded from prefill's last hidden via a one-shot setup peer-copy.
- **1:** verify_hidden row exit — 20 KB, 906→1031, gated by trunk's "verify done this cycle" event. Becomes next cycle's prev_hidden source.

Candidate tokens (max_n × u32 = 16 B) flow drafter→host inside the
chain loop (already a D2H today); host pushes them back to trunk as
the verify input. No peer-DMA needed for tokens.

## Implementation order (revised)

1. **Init-time mirror of trunk weights to drafter_gpu** — token_embd
   always, output only when `use_full_vocab`. Make Qwen35Weights
   aware of "borrowed-on-this-device" tensors that point into mirrored
   memory. Or: thread a separate "drafter trunk weight view" struct
   through mtp_head_forward_* calls.
2. **Split MtpSpecState** into trunk/drafter halves with split alloc.
3. **Add 2 peer-copies + 2 events per cycle** to spec_step.
4. **Wire mtp_only_demo** with --mtp-device flag.
5. **Validate** coherence + perf.

## Estimated effort

Higher than initial estimate due to the trunk-weight-mirror requirement.

- Trunk weight mirror plumbing: ~150 LOC (depends on whether
  Qwen35Weights can take a "borrow this tensor instead of owning" or
  we add a parallel struct).
- MtpSpecState split: ~100 LOC.
- spec_step threading: ~150 LOC (most lines unchanged; just `gpu` →
  `drafter_gpu` substitutions in the chain, peer-copies + events
  added at boundaries).
- Demo wiring: ~50 LOC.

**Total: ~450 LOC**, 1-2 days of focused work.
