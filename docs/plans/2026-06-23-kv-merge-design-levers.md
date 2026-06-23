# Plan 2: KV-compression merge floor — design-time levers (not FT)

Date: 2026-06-23. Branch: chaingun. Follows the KVarN+CASK recovery probe, which
established — on a leak-free harness (commit 571dc64f) — that the CASK merge noise
is **NOT recoverable by fine-tuning** (held-out ~27% recovery, overfits; the
RoPE-phase merge blur is genuine information loss, not a learnable bias; see
[[project_qat_recovery_probes]]). Therefore the merge floor must be lowered at
**design/encode time**, not trained away. This plan enumerates those levers.

## Established facts (the "why")

From [[project_kv_compression_explore]] + docs/plans/2026-06-22-hierarchical-kv-followups.md
(qwen3.5-0.8b, KLD/PPL vs gold BF16, hot=64):
- **The merge is the ONLY quality cost.** `fold_m=1` (no merge, pure 4-bit cold)
  PPL 26.13 BEATS the all-4-bit baseline 30.81 — quant is ~free even at 2-bit.
- The merge cost is **RoPE-phase blur** from averaging differently-positioned K
  (~+7 PPL per fold doubling). vnorm importance −15% (34.84 vs 40.84 uniform);
  position-local merge only −2%; attention-mass importance was WORSE than vnorm.
- Best compressed point today: hot=64 fold=4 vnorm+poslocal **PPL 34.0** (+10% over
  baseline, WITH real cold compression). Residual +3 PPL = the merge floor.

## Goal

Beat the current hot=64/fold=4/vnorm Pareto point (PPL 34, ~18× KV compression)
— either lower PPL at equal bytes, or equal PPL at fewer bytes — by attacking the
RoPE-phase-blur root cause and spending the deferred/idle compute budget.

## Levers (priority order; the first two are the novel high-value ones)

### 1. RoPE-dephased merge (NEW — directly attacks the root cause)
The merge blur IS RoPE-phase cancellation: averaging K vectors at different
positions destroys their distinct rotary phases. Fix: before averaging a cold
group, **de-rotate each K by its position phase** to a common reference, average in
the dephased frame, store `(mean_dephased, ref_position)`, and **re-apply the
reference phase at read**. This removes the dominant blur term (the QAT probe
proved it's otherwise irrecoverable). Position-local merge (−2%) is the coarse
proxy; explicit dephasing is the principled version. Implement in `compact_cold_kv`
(`crates/hipfire-kvquant/src/kv_compact.rs`) behind a flag; the inverse rotate
folds into `kvarn_dequant_tile`'s read. Validate KLD/PPL fold=4 vs current vnorm.
RISK: if the value being merged genuinely differs (not just phase), dephasing helps
only the phase component — measure the ceiling first on captured FA K (the
`HIPFIRE_DUMP_HIDDEN_ALL` Q/K/V dumps already exist from the explore work).

### 2. Low-rank cold residual (NEW — competitive long-ctx alt already flagged)
Store the merged group mean PLUS a rank-r correction for the group's residual
(`K_group ≈ mean + U_r Σ_r Vᵀ_r`). The explore work measured rank-64 SVD KV at
cos 0.991 @256B — competitive, and the basis amortizes across a wide tile. Spend
the **idle/between-turns budget** (the deferred thesis) to compute the SVD off the
critical path. Gate on KLD beating the flat-mean merge at equal bytes.

### 3. Hot-budget / fold operating-point retune
hot is the main dial (64→512 PPL 40.8→33 at uniform). With #1/#2 lowering the
per-fold cost, re-find the knee. Likely a smaller hot + cheaper-but-better cold
beats today's hot=64/fold=4. Cheap sweep via `HIPFIRE_KV_{HOT_BUDGET,FOLD_M}`.

### 4. Importance refinement
vnorm (‖V‖) is the current best (cheap, intrinsic salience). Probe a hybrid:
vnorm × recency-decay, or vnorm gated by a low-rank "is this token retrieved later"
proxy. attention-mass was a documented negative (recent-window bias) — don't repeat.

### 5. Carry-over follow-ups (from the hierarchical-kv doc, still open)
- **Segment defrag** (#2): idle-time re-compaction of accumulated per-turn segments
  → fewer/wider tiles (also amortizes scale overhead, enables #2's basis sharing).
- **Per-channel scale-overhead** (#3): group-scale instead of per-channel (1024 B/tile
  dominates narrow tiles, caps 2-bit win at ~1.7-1.9× vs ~2×).
- **1-bit cold probe** (#4): codec supports bits=1; run the KLD sweep — another
  free-ish storage halving if it holds like 2-bit did.

## Eval methodology

- `eval_hipfire --kv-mode kvarn` KLD/PPL vs gold BF16, **≥16 chunks** (2-chunk top-K
  KLD is noisy — trust NLL/PPL). Headline configs + a larger model (9B/27B) where
  long-context compression actually matters.
- **Coherence**: `scripts/coherence-gate-dflash.sh` (KV changes touch the attractor
  surface). The kv-compression memory notes occasional mangled multibyte emoji on
  the hier path — watch output integrity.
- **Decode-perf A/B**: two-tier read cost vs single-tier (`scripts/probe_commits.sh`,
  byte-identical prompt) — confirm `idle_compact` removes mid-gen migration spikes.
- gfx1103 LDS hazard: keep all new cold kernels zero-LDS (register + `__shfl_xor`).

## Decision this produces

A new quality/byte Pareto point for compressed KV (target: beat PPL 34 @ ~18×, or
hold PPL 34 at >23× via #1/#2 + 2-bit/1-bit cold), with the RoPE-dephased merge as
the lead candidate since it targets exactly the floor that FT could not recover.

## Cross-cutting note

This is **encode/design-time** work on the KV codec + compaction — the correct
response to "the merge floor is real information loss" (FT can't recover it, proven
this session). The deferred/idle compaction budget is what makes the heavier levers
(#1 dephasing, #2 low-rank SVD) affordable without hot-path decode cost.
