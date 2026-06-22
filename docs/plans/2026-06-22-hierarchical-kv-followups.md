# Deferred-hierarchical KV — follow-up tasks

Status: **active** — feature merged to `chaingun` 2026-06-22 (merge `e374a6aa`,
from branch `kv-compression-explore`). Owner: chaingun.

## What shipped

A flag-gated (`HIPFIRE_KV_HIERARCHICAL=1`, default off → byte-identical baseline)
two-tier KV cache for the KVarN decode path:

- **HOT tier** — most-recent `hot_budget` tokens kept exact (raw f32) in a per-layer
  ring `[n_kv_heads × hot_budget × head_dim]`, read by the slot-major mode of
  `attention_cold_slots` (emits flash partials `(m,l)` directly).
- **COLD tier** — older tokens compacted by `compact_cold_kv` (importance-weighted
  `m:1` merge + KVarN quant) into 4-bit-or-2-bit segments that stay resident on GPU,
  dequantized on-the-fly (`kvarn_dequant_tile` → f16) and read by the channel-major
  mode of `attention_cold_slots`.
- The two tiers fold via `flash_tier_merge` (online softmax). Migration (hot→cold)
  runs either on the overflow fallback (`migrate_n(migrate_batch)`) or, preferred,
  via `idle_compact` in the between-turns idle gap (`qwen35_prefill_active_session`).

Code: `crates/hipfire-kvquant/` (leaf codec + `compact_cold_kv`),
`crates/hipfire-runtime/src/kv_hier.rs` (`HierKvState`), kernels
`attention_cold_slots` / `flash_tier_merge` / `flash_partials_ml` (+ bits-aware
`kvarn_dequant_tile`), dispatch hook in `kv_cache_attention_dispatch`
(`crates/hipfire-arch-qwen35/src/qwen35.rs`), serve hook + guard in
`crates/hipfire-serving-core/src/qwen35_prefill.rs`.

Parity oracles: `rdna-compute/examples/parity_{attention_cold_slots,
flash_tier_merge,flash_partials_ml,two_tier_e2e,cold_4bit_read}` and
`hipfire-runtime/examples/parity_kv_hier`. Env knobs registered in `env_docs`.

### Quality landed (qwen3.5-0.8b-mq4, KLD/PPL vs gold BF16, hot=64, 2 chunks)

| Config | PPL | Note |
| --- | --- | --- |
| baseline kvarn (all 4-bit) | 30.81 | reference |
| hier fold_m=1 (no merge) | 26.13 | **beats baseline** — machinery + cold read lossless |
| hier fold=4 uniform importance | 40.84 | the merge is the whole cost |
| hier fold=4 **vnorm** (default) | 34.84 | −15% vs uniform |
| hier fold=4 vnorm + **position-local** | 34.00 | shipped default; +10% over baseline, WITH compression |
| hier fold=4 **2-bit** cold | 34.56 | +1.6% vs 4-bit; ~2× cold-code storage cut |

Key established facts (the "why" for the follow-ups):

1. **The merge is the only quality cost.** Quant is free even at 2-bit and even
   with no rotation — the KVarN per-channel Sinkhorn variance-norm already does the
   incoherence job a rotation (FWHT/ConQuR) would. So **ConQuR / `rotate=true` were
   probed and rejected** as non-bottleneck work.
2. **`vnorm` (‖V‖) importance beats the "principled" attention-mass signal.**
   Accumulated while-hot attention mass (`HIPFIRE_KV_IMPORTANCE=attn`) measured
   *worse* than vnorm — it reflects only recent (while-hot) queries, not the future
   long-range retrieval the cold tier serves. Documented negative result; vnorm is
   default.
3. **Hierarchical KV is inherently per-token attention** — it lives in
   `kv_cache_attention_dispatch`. The batched session-batch prefill bypasses that
   and is **guarded off** (errors like CASK-eviction / PFlash).

## Follow-up tasks (rough priority order)

### 1. Multi-session batched-prefill support — **DONE (routing), commit db6f87bf**
Implemented approach (a) at the backend-selection layer rather than splitting rows:
the multi-session **batch protocol** (`generate_batch_prefill` /
`generate_batch_decode_step`) now forces the `SerialReference` backend when
`HIPFIRE_KV_HIERARCHICAL=1`, for BOTH prefill (`qwen35_prefill_suffix_batch`) and
decode (`run_generate_batch_decode_step_qwen35`). SerialReference activates each
session and runs per-token `forward_scratch` through `kv_cache_attention_dispatch`
(per-session isolated) — the path where the hot-ring/two-tier-read/idle hook live.
Correct by extension of the `infer_qwen35` proof (forward_scratch+kvarn+hier coherent).
Replaced the prior hard guard. Slower than fused batch; hier is a memory feature.

### 1b. kvarn/hier in the daemon `generate` path — **DONE, commit a5140860**
The pre-existing gap (kvarn garbage in the daemon serve path, q8 fine) had TWO causes
in `hipfire_serving_core::generate::generate()` (not the daemon main; the actual AR
entry), both bypassing the per-token `kv_cache_attention_dispatch` kvarn requires:
1. Models with a bundled MTP head (e.g. qwen3.5-0.8b-mq4, `mtp_mode=auto`) routed to
   `generate_mtp` spec-decode (batched prefill) and returned before the AR block.
2. The AR block itself prefilled via batched `forward_prefill_batch`.
Fix (3 changes in `generate()`): a `kvarn_active` guard skips the DFlash + MTP spec
paths for kvarn caches → AR path; the AR prefill uses per-token `forward_scratch` for
kvarn; and a between-turns `idle_compact` drain fires on continued turns (seq_pos>0).
Validated on `hipfire-daemon`: plain kvarn went from `!!!!` garbage to a coherent
semaphore definition (**also fixes plain-kvarn daemon serve**, the quick win); hier
multi-turn (hot=64, 2-bit cold) is coherent across both turns, turn 2 firing the idle
drain. Debugging note: cargo can recompile `hipfire-serving-core` without relinking
`hipfire-daemon` — `touch crates/hipfire-daemon/src/main.rs` to force the relink when
testing daemon changes (cost ~1hr of stale-binary confusion).

### 1c. (Optional) batched two-tier attention
Only if hier ever needs batched-prefill *throughput*: a session-batch attention
kernel doing the hot-ring + cold-segment two-tier read per row. Large; the
per-token serial route (#1) is correct and sufficient for a memory feature.

### 2. Segment defragmentation
`idle_compact` folds each turn's drain into ONE cold segment, so segments accumulate
~1 per turn. The two-tier read does an `attention_cold_slots` + `flash_tier_merge`
per segment, so read cost grows linearly with turn count. Add an idle-time **defrag**:
when segment count exceeds a threshold, dequant N old segments → re-`compact_cold_kv`
their union → one bigger segment. Cheap to fold into `idle_compact`; runs off the
critical path. Also improves compression (bigger tiles amortize the per-channel scale
overhead — see #4).

### 3. Per-channel scale-overhead reduction
A cold tile record is `code_bytes + r_dim*2*2 (scale_abs+zp_abs fp16) + c_dim*2`. For
head_dim=256 the fixed per-channel block is **1024 B/tile**, which dominates small
(narrow) tiles and caps the 2-bit storage win at ~1.7–1.9× instead of ~2×. Options:
group-scale (one scale per channel-group instead of per-channel), or a coarser scale
dtype. Measure the SQNR/PPL cost before shrinking. Most impactful combined with #2
(fewer, wider tiles).

### 4. 1-bit cold probe
The codec already supports `bits=1` (QMAX=1 path in `quantize_tile_qmax`; pack/dequant
handle `8/bits` codes/byte generically). Run the `HIPFIRE_KV_COLD_BITS=1` KLD sweep
(fold=1 to isolate quant, then fold=4 at the operating point). If 1-bit holds up like
2-bit did, it's another free-ish storage halving on the cold codes. Cheap — same probe
harness as the 2-bit one (`eval_hipfire --kv-mode kvarn`).

### 5. Tighter quality numbers + a perf benchmark
- **Multi-chunk KLD.** All current numbers are 2-chunk per-token `eval_hipfire`; the
  top-K KLD is noisy at that size (trust NLL/PPL). Re-run the headline configs at
  ≥16 chunks for publishable deltas, and on a larger model (9B/27B) where long-context
  compression matters more.
- **Decode-perf A/B.** Quantify the two-tier read's per-token cost vs single-tier
  KVarN (cold dequant + per-segment attend + merge add work). Confirm `idle_compact`
  actually removes the mid-generation migration latency spikes (the whole point).
  Use `scripts/probe_commits.sh` + a fixed byte-identical prompt per the perf rules.

### 6. (Deferred / likely unnecessary) rotation + ConQuR on cold tiles
Recorded for completeness: a per-tile rotation (FWHT `rotate=true`, or ConQuR
Procrustes-to-corners) would only help if cold quant became the bottleneck. The 2-bit
probe showed it is **not** the bottleneck (Sinkhorn variance-norm already handles
incoherence). Revisit ONLY if a future regime makes quant dominate (e.g. 1-bit cold
turns out lossy in #4, or a much wider tile). The deferred/idle compaction budget is
the one place ConQuR's runtime cost would be affordable — but it is not justified now.

## Notes / gotchas for implementers

- **gfx1103 LDS hazard**: all the cold kernels are zero-LDS (register + `__shfl_xor`)
  on purpose. Keep any new cold-path kernels LDS-free or they wedge the dev box.
- **GpuTensor has no pool-return Drop**: `idle_compact` segment buffers are allocated
  per migration and cleared on `reset()` (session start). Per-token allocation leaks —
  keep allocations on the migration/idle path, never per token.
- **`reset()` fires at `pos==0 && layer==0`** (sequence start) — mid-session decode
  never resets, so continued-turn history is preserved. The idle drain is the
  `seq_pos>0` prefill hook, distinct from reset.
- **Memory note**: full design history + negative results in the auto-memory
  `project_kv_compression_explore` (HoloKV dead, attn-mass null, ConQuR killed, etc.).
