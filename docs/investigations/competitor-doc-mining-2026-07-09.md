> **CLEAN-ROOM BOUND:** techniques/mechanisms only, re-expressed on hipfire HIP+mqN. NO competitor code, NO ggml/Q4_K/Vulkan import.
> **Competitive context (verified, hashed):** hipfire WINS prefill (a3b: 2239 vs ZINC 759 tok/s, 2.95x); the entire deficit is DECODE (158 vs 166, ~5%). Every lever below targets decode.
> **Sources (read-only doc mining):** zinc@5ab624c, ROCmFPX@a6a9376, rocmfp4-llama@4795079. 132-agent workflow: 50 docs -> 483 findings -> 30 top-tier -> 1 hard-verified survivor (rest ranked with caveats).

# Competitor RDNA4 doc-mining — gfx1201 decode levers

Scope: hipfire gfx1201 a3b mq4r decode is **158 tok/s vs ZINC 166** (~4.8% gap). Prior fleet workflow (project_zinc_decode_gap_workflow_2026_07_08) already established only **~2.5% of the ~10% gross gap is single-kernel**; the rest is fusion / hipGraph-sched / occupancy / config / noise, and the one regime-changer is **MTP batch-K**. This report ranks what remains *liftable* after mining the ZINC + ROCmFPX corpus, filtered against hipfire's own memory + code.

---

## 1. Top liftable levers (ranked)

Ranking weight: value × novelty × gfx1201-**decode**-fit × orthogonality. System/config levers rank high because they are cheap, code-free, and stack additively with every kernel win; the decode-kernel space is largely tapped.

### Rank 1 — GDDR6 GECC/RAS disable (`amdgpu.ras_enable=0`) — VERIFIED survivor
- **Source:** ZINC `docs/RDNA4_TUNING.md` + `AMD_GPU_REFERENCE.md` (two independent citations, R9700/gfx1201, Qwen3.6-35B-A3B Q4_K).
- **Claim + their numbers:** GDDR6 inline-ECC consumes ~10% memory BW by default; disabling via GRUB boot param moved decode **101 → 110 tok/s (+9%)** on the *identical card + model class + quant + decode regime*.
- **Mechanism:** GRUB_CMDLINE_LINUX_DEFAULT += `amdgpu.ras_enable=0` + `update-grub` + reboot. Trades bit-flip resilience for a per-transaction memory-controller tax removal.
- **our_status:** novel-liftable (never toggled on our fleet; no code surface). Prior surfaced as CANDIDATE/UNVERIFIED in `project_rdna4_gecc_gddr6_ecc_lever_2026_07_09.md`.
- **Concrete hipfire action:** Fleet/GRUB change on hiptrx (dedicated box, no downside). **Mandatory protocol before any claim:** (a) resolve premise first — `rocm-smi --showrasinfo` says RAS disabled but sysfs `ras/features=0x101` is live/inconclusive; if GECC already off the toggle is a definitional no-op; (b) cross-boot A/B, byte-identical prompt + md5, warm cache/DPM, **AUTO clock** (NOT perf_level=high — underclocks gfx1201 ~13%), median-of-5, treat >5% as signal; (c) re-take rocprofv3 PMC at profile_standard — if DEP_WAIT/memory-stall cycles drop, the tax is per-transaction-latency (helps our latency-bound path); if only aggregate BW moves, expect near-noise. Ceiling ≤9% (ZINC's number is RADV/Vulkan; our hipGraph path may already hide part of the tax). **Never silently default on** — documented fleet ECC decision.

### Rank 2 — Kernel 6.17 SMU driver-interface clock cap
- **Source:** ZINC `docs/RDNA4_TUNING.md` (R9700/gfx1201).
- **Claim:** Kernel 6.17 ships SMU driver-IF v0x2e while RDNA4 firmware expects v0x32; mismatch silently pins max clock at 2200 MHz instead of 2350 MHz, no error emitted.
- **Mechanism:** amdgpu KMD negotiates a downgraded clock/power table on version mismatch.
- **our_status:** novel-liftable (distinct root cause from the known `perf_level=high` underclock in `feedback_rdna4_perf_level_high_underclocks.md`). hipfire has an identical *diagnostic playbook* for the SMU-shadowing class (`feedback_firmware_shadowing_perf_trap.md`, dmesg "SMU driver if version not matched", caused a 27B DFlash τ 7.7→4.1) but never checked this specific kernel-version threshold.
- **Concrete action:** On hiptrx: `uname -r`; `dmesg | grep -i "smu.*version"`; capture sustained `pp_dpm_sclk` active `*` level under a warm a3b decode load. If pinned ~2200 vs advertised ~2350 (or auto's 2838-3260), test kernel 6.14. Distinguish from `perf_level=high` before attributing any unexplained fleet variance. Cheap, orthogonal, compounds with everything.

### Rank 3 — PCIe ASPM `policy=performance`
- **Source:** ZINC `2026-07-01-...measured-sweep.md` (R9700/gfx1201, RADV).
- **Claim + numbers:** Forcing PCIe link into L0 gives **+10.8% dense Qwen3.5-27B decode** (29.30 → 32.46 tok/s), +1.3% AMDVLK, **0% on MoE decode** (fewer dispatches/token structurally sidesteps the doorbell L1-exit tax).
- **Mechanism:** each decode-step doorbell is an MMIO write; if the link idles between dispatches ASPM drops L0→L1 and the next doorbell pays ~4-16µs renegotiation. `echo performance | sudo tee /sys/module/pcie_aspm/parameters/policy`.
- **our_status:** novel-liftable — zero ASPM/doorbell references anywhere in the repo (not in `chip-profile-sweep.sh`, `perf-benchmarking.md`, or CLAUDE.md env-state list).
- **gfx1201-decode-fit CAVEAT:** **a3b is MoE → ZINC measured 0% there.** Additionally hipfire's whole-forward-pass hipGraph collapses per-token dispatch to ~1 hipGraphLaunch, further shrinking the surface. Highest value on **dense 27B/9B** and on non-graph archs (gfx1010/gfx1030/CDNA) + prefill/MoE-fallback bursts. Still worth the one-line A/B (zero code) on hiptrx: even 1 doorbell/token at decode cadence can cross an idle-link threshold.
- **Concrete action:** sysfs flip + `probe_commits.sh`-style before/after, dense 27B first, a3b second (expect near-noise, confirm).

### Rank 4 — Wide `NUM_ROWS` M-gated DMMV for the LM-head / vocab-scale GEMV
- **Source:** ZINC `2026-04-22-...32-column-dmmv.md` (R9700/gfx1201).
- **Claim + numbers:** Adding a `NUM_ROWS=8` DMMV variant dispatched only when output-row count **M ≥ 100000** (LM-head vs 262144 vocab) cut that kernel **44.89ms → 1.64ms (27×)** and lifted whole-decode-step **3.87 → 5.62 tok/s (+45%)** on gemma4-31b-q4k-m. Default `NUM_ROWS=2` spawns 131072 workgroups each re-reading the same hidden vector, thrashing L1 — the win is amortizing the X-read across 8 rows.
- **our_status:** novel-liftable *infrastructure-present*. hipfire has `HIPFIRE_GEMV_ROWS∈{1,2,4,8}` (`crates/rdna-compute/src/gemv.rs:4494`) but row-count is a **static per-arch default** (`arch_caps.gemv_rows_default()`), only M-gated by a single `m>=64` on/off. lm_head reuses the generic decode GEMV path (`llama.rs:4853`), so vocab-scale M (150k-262k) gets a mid-FFN-tuned default. The qkvza/gate_up "R=2/3/4 DEAD" verdict (gfx1201.md:56) was measured at M~thousands — a **different regime** (workgroup-count/L1-thrash dominates at huge M; VGPR/occupancy dominates at small M). R=8 at LM-head scale is untested territory, NOT a re-litigation of the dead lever.
- **gfx1201-decode-fit CAVEAT:** hipfire's own `docs/gfx1201-native-surface.md` census measures **lm_head at ~100% of the 611.8 GB/s DRAM roofline (0 headroom)** on a3b — so on *our* current model the LM-head is already saturated. This lever is high-value on **large-vocab dense models with an unsaturated LM-head**, lower on a3b. Worth wiring as an additive M≥100k dispatch axis regardless (helps qwen3.5 dense, embeddings).
- **Concrete action:** Add an M-gated branch in `gemv.rs` row-count selection: `if m >= 100_000 → gemv_hfq4g256_multirow_r8`. Re-sweep R∈{4,8} at vocab scale per-arch (gemv occ probe). Do NOT touch the qkvza/gate_up path (disjoint regime, dead there).

### Rank 5 — MTP-for-MoE batch-K enablement: MMVQ↔MMQ routing threshold + MoESD K* + cascade utility gate
- **Source:** ROCmFPX `ROCmFP4-DECODE-SPEED-EXPERIMENTS.md` (`GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH`); ZINC `2026-05-25-speculative-decoding-a3b-loses.md` (MoESD K* formula, cascade gate).
- **Claim:** (a) a tunable batch-size threshold decides GEMV-style MMVQ vs compute-tiled MMQ per MoE decode dispatch; (b) MoE verify weight-traffic scales as the *union of routed experts* `N·(1-(1-p)^K)`, p=top_k/num_experts — a per-model saturation threshold K* tells you whether a draft width is in loss or near-free regime; (c) a live per-step utility gate (token-gain vs verify-cost) beats a static on/off.
- **our_status:** the regime-changer flagged repeatedly (`project_dp4a_moe_decode_bw_bound_falsified` ENDPOINT: "only regime-changer=MTP batch-K"). hipfire's MoE decode dispatch is a **hard binary split** (batch==1 → GEMV; any n>1 → grouped-GEMM), no swept small-batch crossover (`qwen35.rs:12978` vs `7326`). The K* formula and cascade gate are genuinely absent: p_min (`mtp_spec.rs`) is drafter-*confidence*-gated and keyed by **GPU arch**, not by MoE routing cost or per-request verify economics. `mtp_head.rs` already loads num_experts/top_k at init, so K* is pure arithmetic.
- **OPEN DISCREPANCY to resolve (flag, don't assume):** ZINC's model says K~3-32 is deep in the pre-saturation loss region for low-top_k/high-expert routers, yet hipfire's *own* A3B MTP measured a large win (101→259 tok/s gfx1201, `project_hiptrx_moe_mtp_a3b_review`). Either MTP's lightweight-head verify economics differ from full-drafter DFlash, or our A3B trunk's num_experts ≪ ZINC's 256-expert reference → much lower K*. Re-measure against the MoESD formula before campaigning.
- **Concrete action:** (1) build the K* = f(top_k, num_experts) calc into `arch_caps`/`mtp_spec.rs` as a draft-width cap; (2) add a swept small-batch GEMV↔grouped-GEMM crossover (`ArchCaps::should_use_mmq`-style but for MoE decode, using the already-batched `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hip` grid.z=N machinery); (3) prototype the cascade utility gate as a second signal alongside p_min. Highest-ceiling item; largest lift.

### Rank 6 — wave64-for-DECODE re-test (see §2 — top RE-TEST item)
Ranked here because it is a *liftable candidate*, not just a premise check: ZINC ships wave64 decode on the same gfx1201 and RDNA4's I8/FP16 WMMA table gives wave64 a 2× data/slot amortization the "neutral" verdict never probed at the actual small-M shape. Full detail in §2.

### Rank 7 — Asymmetric K/V precision (protect K, compress V) + first/last-layer boundary protection
- **Source:** ROCmFPX `README.md` (`-ctk q8_0 -ctv turbo4`, `LLAMA_KV_TURBO_BOUNDARY_LAYERS=2`).
- **Claim:** K drives QK^T score fidelity (sensitive), V is linearly combined (tolerant) → keep K high, compress V aggressively; separately protect first/last N layers' K at Q8 while middle layers go aggressive.
- **our_status:** partial already-have with a **wiring gap on the correct asymmetry direction**. Our static `quant_asym2/3/4` modes compress **K** (Givens-rotated) with V pinned Q8 — the *opposite* direction — and `feedback_kv_mode_fwht_over_asym` already flags that combo as poor quality (independently hitting the failure mode). The newer `kv_adaptive.rs` (`balanced_steps`) correctly descends **V first** ("biggest byte win up front, keep K protected longer") — the same direction ZINC recommends. `VMode` is stored independent of the K rotation mode. Boundary protection is **scaffolded but inert**: `KvCache.boundary_layers` (default 2), `layer_is_boundary: Vec<bool>`, `is_boundary()` accessor all exist; `kv_tier.rs:160` says "Inert until the boundary-layer producer populates `layer_is_boundary`" — populated as `vec![]` everywhere.
- **gfx1201-decode-fit:** primarily long-context capacity + coherency, modest decode-BW (KV overtakes weight bytes only past the ~16k crossover — Rank 12). Directly targets the **A3B eviction blocker** (`feedback_a3b_r_not_acceptable`: R̄≈0.39, eviction off until R̄ improves).
- **Concrete action:** (a) wire a static K=Q8 / V=aggressive default into the KV-mode selection path (config-only, reuses existing independent K/V plumbing); (b) implement the `layer_is_boundary` producer from `boundary_layers` at `KvCache` construction (small, self-contained). Both gated + coherence-tested.

### Rank 8 — QJL residual-sign unbiased-inner-product correction for low-bit K
- **Source:** ZINC `TURBOQUANT_SPEC.md` (99.5% attention cosine sim at 3-bit on Qwen2.5-3B — *validated accuracy*, note the doc's perf numbers are self-labeled estimates).
- **Claim:** add a 1-bit signed random-projection residual term making the K-quant inner-product estimator provably unbiased independent of bit-width, preserving softmax fidelity at very low bits.
- **our_status:** novel-liftable — the one genuinely missing piece of hipfire's otherwise-complete FWHT+Lloyd-Max K pipeline. Every "residual" in the repo is transformer-residual naming; K-write kernels (`kv_cache_write_asym_k_fwht3.hip`) store only `[cnorm][packed MSE indices]` = plain biased round-to-nearest.
- **gfx1201-decode-fit:** quality lever, not tok/s — but it directly targets hipfire's own documented **MQ2/fwht2 low-bit collapse** (docs/QUANTIZATION.md), potentially rescuing the 2-bit K tier without a codebook redesign, which *unlocks* more aggressive KV compression → decode/capacity win at long ctx.
- **Concrete action:** extend the fwht3/asym3 K byte layout with a sign-bit field + residual norm; add the correction term to `attention_flash_fwht3_tile.hip`'s score path; coherence-gate before ship.

### Rank 9 — Reconstruction-MSE block-scale search for low-bit quant
- **Source:** ROCmFPX `README.md` (CPU-side encoder).
- **Claim:** pick per-block scale by minimizing reconstruction MSE (grid search) rather than absmax range-mapping — measurable coherency gain at the same bit budget.
- **our_status:** novel-liftable, precedent-in-repo. Mainline mq4/HFQ4G256/HFQ4G128 use absmax (`scale = max_abs/7.0`, `range/15.0`). The MSE-sweep pattern already exists but only for the experimental E8-lattice mfp2/mfp3 format (`e8.rs:802-874`). Porting it to the production group-scale computation is untried, directly relevant to the A3B mq4 KLD/lobotomy history.
- **Concrete action:** copy the `e8.rs` MSE-sweep into `quantize_hfq4g256`/`quantize_mq4g256` group-scale; CPU-quant-time only, zero decode-kernel risk; re-eval KLD vs llama.cpp bf16 oracle. Pairs with a `partition_point` binary-search for the E4M3 scale-roundup (`e8_gptq.rs` currently does a 0..0x7E linear scan — exact, monotonic, log-searchable; pure quantize-time speedup, no tok/s).

### Rank 10 — hipGraph rebuild-vs-kernel timing instrumentation
- **Source:** ROCmFPX `ROCmFP4-DECODE-SPEED-EXPERIMENTS.md` (`LLAMA_GRAPH_BUILD_TIMING=1`).
- **Claim:** perf-footer separates graph-rebuild count/time from eval time as the first triage step — if rebuild is small it's kernel-side, if large it's graph-reuse-side.
- **our_status:** novel-liftable diagnostics gap. No rebuild-count/time counter in `graph.rs`. Our two hipGraph regression postmortems (`project_gfx12_hipgraph_late_host_alloc_clobber`, `project_gemv_graph_cache_pr3`) were diagnosed by more expensive means; neither used a cheap rebuild/eval split.
- **Concrete action:** add rebuild-count + rebuild-µs counters to `graph.rs` capture/replay, surface in the daemon perf footer. Pure tooling, no kernel interaction, fills a documented diagnostic gap; feeds the certify loop.

### Rank 11 — MoE MMQ routing-threshold validation gap (dense-tuned cutoff never re-checked on MoE)
- **Source:** ROCmFPX (routing threshold regressed MoE guard 8-17% when ported from a dense-tie).
- **Claim:** an MMVQ↔MMQ crossover tuned on dense can regress MoE because per-expert batch distribution differs; validate on BOTH.
- **our_status:** novel-liftable *validation task*. `ArchCaps::should_use_mmq` sets MMQ min-batch=128 for RDNA4, with the code comment citing validation **only on a dense model** ("+118% prefill on qwen3.6-27b.mq4/gfx1151 by dropping 256→128") — no cited A3B/MoE-grouped re-validation, yet `gemm.rs` serves MoE grouped-prefill through the same cutoff.
- **Concrete action:** bench `should_use_mmq`'s 128 cutoff on A3B/MoE-grouped **prefill** specifically. Cheap, high-value, no code change unless a gap is confirmed. (Prefill-side, but protects the A3B path.)

### Rank 12 — Decode weight+KV bandwidth-crossover diagnostic (methodology → oracle)
- **Source:** ZINC `2026-05-16-...lmhead-costs.md` / KV-crossover posts (~16k-token crossover on R9700).
- **Claim:** decode bytes/token = constant active-weight term + linear KV term; KV overtakes active-weight past ~16k tokens; below → weight/routing kernels are the lever, above → KV/attention kernels.
- **our_status:** novel-liftable — no `crossover`/`bytes_per_token` additive model in repo; `docs/gfx1201-native-surface.md` is a single-context snapshot. Directly actionable for the **ACTIVE RDNA Kernel Oracle** (this branch) as a context-length-conditioned pre-screen ahead of the measured BoundClass. Also folds in ZINC's profiler-free `query_heads × batch vs CU_count` occupancy-starvation pre-screen.
- **Concrete action:** add the additive-term crossover + head×batch-vs-CU predicate to `roofline.rs`/oracle corpus as a zero-cost analytic pre-filter (no kernel change). Computes per-model (a3b/qwen3.5/cohere2moe) where to spend effort by prompt length.

---

## 2. RE-TEST queue (DO-NOT-RETRY findings competitor evidence + our roofline suggest re-opening)

### RE-TEST #1 (TOP) — wave64 for batch-1 DECODE GEMV on gfx1201
- **Our dead verdict:** `project_wave64_neutral_mq4_gemv_gfx1201_2026_07_08` ("DO-NOT-RETRY", −0.2..−0.8%, "wave32 empirically correct").
- **Why the premise changed:** that verdict's numbers were taken **exclusively in the DRAM-saturated regime** — M=16384/K=8192 @92% of 640 GB/s peak, M=32768/K=4096 @~90%. The note's own BONUS acknowledges the decode-representative shape is different ("small-M/decode-like = 252-308 GB/s vs 833 saturated → batch-1 UNDER-SATURATES, latency/utilization-bound") **but never ran a wave32-vs-wave64 A/B at that shape.** The 2026-07-09 roofline (`project_dp4a_moe_decode_bw_bound_falsified`, ROOFLINE CONFIRMATION) shows the REAL decode kernels (fused_qkvza, moe_gate_up_k8) at mem_busy 40.8-54.5%, occ 34.8-52.1% — genuinely **latency/occupancy-bound, NOT the DRAM-saturated regime the smoke test probed.** This is the textbook falsified-premise pattern: a "DRAM-bound" framing closing a wave-width lever on an arch the fresh instrumentation characterizes as latency-bound. ZINC additionally *ships wave64 decode on the same gfx1201* (their DMMV family default), and the RDNA4 I8→I32/FP16 WMMA table gives wave64 2× data/slot — a mechanism most relevant precisely at low occupancy.
- **Expected mechanism:** wider transactions + fewer per-wave fixed overheads hide VALU/latency when occupancy is the binding constraint (not when BW is saturated). Could net ahead where the saturated-shape test showed dead-flat.
- **Re-test protocol:** JIT wave64 variant (`-mwavefrontsize64` already compiles/runs correctly on gfx1201 per the same note) of the **actual production decode GEMV shape** (small-M, batch-1, ~16-52% occ), checksum-verified, interleaved A/B, median-5, AUTO clock. Note the corollary contradicting-ours findings all cite the same overturned premise — treat as one re-test class. **Caveat:** the same roofline shows gfx1201 *near* 92% BW at max occupancy, so wave-width may not be the binding constraint even if occupancy is; positive result still needs the per-kernel `-DWSZ` build investment across 356 wave32 kernels before it pays off. Low-cost to *test*, high-cost to *ship*.

### RE-TEST #2 — `k>=2048` shape-gated int8/dp4a MMVQ decode (narrow)
- **Our dead verdict:** dp4a MoE decode-GEMV FALSIFIED (BW/latency-bound, VALUBusy 44-45% ≪ MemUnitBusy, DEP_WAIT 83-94%).
- **Why re-open (weakly):** llama.cpp/ROCmFP4 only enable int8-dot MMVQ at **k≥2048**; hipfire never ran a dedicated k-threshold sweep isolating ≥2048 vs <2048. **BUT** the mechanism argument (VALU idle regardless of k, weights stay 4-bit-stored so no BW win) is shape-independent, AND the 2026-07-09 roofline *reconfirmed* (not overturned) the dp4a-dead verdict. **Expected value low** — list for completeness, run only if idle fleet capacity.

### RE-TEST #3 — single vkQueueSubmit-per-prompt analog / monolithic-graph on gfx1201 specifically
- **Our dead verdict:** `project_gemv_graph_cache_pr3` (monolithic/per-shape graph LOST −5..−18%) — **measured on gfx1010/RDNA1 under older ROCm**, never re-confirmed on gfx1201/RDNA4. Cheap confirmatory re-test worthwhile before fully closing, since the AR-forward whole-pass hipGraph *does win* on gfx12 (+2.4-2.7%) — the winning shape is "capture whole pass once, replay many," not "cache many per-shape graphs with boundaries." Low priority.

### RE-TEST #4 (NEGATIVE-KNOWLEDGE, don't build) — ZINC's own dead-ends to import as guards
Not re-tests of ours, but competitor falsifications to *not* re-derive: gate+up+SwiGLU FFN-input LDS-staging (−13.8%, operand already L1-resident — matches our row-tile-register-reuse design); split-K on gate+up GEMV (−1.8..−6.1%, matches our qkvza split-K −1.55%); producer/consumer warp specialization (no gain, wave scheduler already overlaps); register-headroom tuning when LDS caps occupancy (no gain). Fold into the occupancy-campaign do-not-retry list.

---

## 3. Orthogonal-compound map

**Independent / additively stacking (different layers — can all land + compound):**
- **System/config tier** (Rank 1 GECC, Rank 2 SMU, Rank 3 ASPM) — zero code, orthogonal to every kernel; stack additively with each other and with any kernel win. GECC + ASPM are both BW-tax removals so on a BW-saturated cell they'd partially overlap, but on our latency-bound a3b decode they hit different mechanisms (memory-controller per-transaction vs PCIe doorbell) → treat independent.
- **Rank 5 MTP batch-K** (dispatch/spec-decode layer) ⟂ all kernel-body work — reuses GEMV/GEMM kernels unchanged, just raises M. The *regime-changer*; compounds with everything below it.
- **Rank 4 wide-NUM_ROWS LM-head** ⟂ qkvza/gate_up decode kernels (disjoint M regime) — additive, does NOT reopen the dead multirow verdict.
- **Rank 8 QJL K-quant** ⟂ Rank 7 asymmetric K/V direction — QJL makes low-bit K *viable*, asymmetric policy *chooses* how low; they compound (QJL rescues the 2-bit K tier that the asymmetric policy then deploys).
- **Rank 9 MSE scale-search** + **e4m3 binary-search** — both quantize-time, orthogonal to all decode kernels, compound with each other (quality + encode-speed).
- **Rank 10 hipGraph timing** + **Rank 12 crossover diagnostic** — both pure tooling/oracle, compound as the certify-loop's triage layer.

**Overlapping / same-axis (pick one, or sequence — do NOT double-count):**
- Rank 4 wide-NUM_ROWS and the existing `HIPFIRE_GEMV_ROWS`/MOE_DOWN_FUSED row-tile work are the **same row-tile axis** at different M regimes — extend, don't fork.
- Rank 5's MMVQ↔MMQ crossover and the existing `should_use_mmq` batch gate are the **same dispatch-selection axis** — the MoE decode crossover is the missing rung on the existing ladder (BT-WMMA small-batch → MMQ ≥128), not a competing mechanism.
- Rank 7 boundary-layer protection and the `kv_adaptive.rs` V-first descent are the **same K/V precision-routing subsystem** — wire together.
- RE-TEST #1 (wave64) and Rank 5 (batch-K) both touch decode GEMV occupancy but from opposite directions (wave-width vs M-batching); if batch-K lands and turns decode into GEMM, the wave64-at-M=1 question is mooted → sequence batch-K first.

---

## 4. Already-have / tried-dead / refuted — DO NOT re-mine

**Already-have (competitor "wins" hipfire already ships, often ahead):**
- Batched/column-DMMV & layer-major prefill weight-reuse → `forward_prefill_batch` + WMMA batch-tile GEMM (default-on, +26-33% AR prefill, 6223db8f). ZINC's plain scalar DMMV is *behind* our WMMA.
- SGLang-style MoE scatter (histogram→offsets→permute→grouped-GEMM→combine) → `moe_scatter_*_k8.hip` + `gemm_*_moe_grouped_wmma*.hip`, default-on gfx11+/gfx12. `project_a3b_prefill_path1_path2` (1017→2966 tok/s).
- Block-resident SSM/GDN state (single-launch token loop) → `gated_delta_net_*_batch_seq.hip`. (Known narrow gap: DFlash tree-verify FP32 rollback still uses single-token `gated_delta_net_f32` — the 33 tok/s stuck-path — separate from prefill.)
- Fused kernels: rotate-query-not-key (fwht3 tile, default), fused_qkvza (4→1), fused_rmsnorm_mq_rotate, fused_silu_mul_mq_rotate, MoE-down fused-acc (explicitly ZINC-derived, +3.5% gfx1201), residual-into-GEMM epilogue family, magnitude/direction norm-split, compile-time Lloyd-Max centroids, quantize-at-KV-write.
- GPU-resident MoE top-k router (no D2H) → `moe_softmax_topk_k8.hip` (default hot path, k=8 indexable).
- hipGraph AR-forward decode capture (default-on gfx11/gfx12, +2.4-9.9%).
- Split-KV / flash-decoding, LDS bank-conflict padding, wave32-native attention, Q-in-LDS staging, register-ring P=4 KV prefetch (+6.95% gfx11), dead-tail LM-head skip, tied-embeddings-don't-save-decode-BW, MTP prefix-cache + DeltaNet checkpoint ring (default-on), DeltaNetTape/GdnTape O(1) rollback (+40% MTP gfx12), per-arch capability-gated dispatch ladder, F16-lm-head-gated, Q8-attn SKU split (attn+router protected, bulk FFN cheap), HFQ4G128/G256 granularity dial, peer-direct all-reduce (1637ms→1ms), p_min spec-decode confidence gate.

**Tried-dead / refuted (do NOT retry — cite):**
- int8/dp4a MoE decode-GEMV (`project_dp4a_moe_decode_bw_bound_falsified`, reconfirmed 2026-07-09).
- gfx12 MMQ HFQ4G256 prefill (`project_rdna4_mmq_falsified`, −11..−34.5% + attractor; RDNA4 keeps WMMA). Covers ZINC's DP4a-dense-FFN and Q8_1-activation-prefill proposals.
- int8-WMMA (iu8) HFQ4 GEMM (`project_wmmqa_int8_wmma_attempt`, −5-12% + coherence-fail). Covers "I8 WMMA 2× throughput" and "coopmat FA both matmuls."
- gfx12 iu4 K=32 (`project_gfx12_iu4_breakthrough`, coherence-fail on all prompts; symmetric-operand forces activation to 4-bit — kills quality). Covers ROCmFPX "iu4 is a software gap."
- FP8-WMMA HFP4G32 (`project_fp8_wmma_hfp4g32`, raw 1.87× but 0.7-1.26× e2e; per-block UE8M0 scale chain eats it). **Corrective:** ROCmFPX confirms **no native FP4 WMMA builtin on gfx12** — de-scope gfx1201.md's "fp4 WMMA" lever to FP8-native only.
- chunked parallel-scan GDN (`gated_delta_net_f32_chunked.hip`, built + 1.3e-15 parity but grid-underutilization, slower every shape; default-off).
- MQ4 128B cacheline-align (`project_mq4_128b_align_falsified`, −12.5% gfx1100). Covers "cache-line discipline on weight reads" + KV 128B-stride.
- rocBLAS gfx12 (`feedback_rocblas_gfx12_regresses`, 5.6× slower). Covers ZINC's cuBLAS+fp16-cache 3×-slower.
- fused softmax+topk+renorm MoE router (1-ULP divergence → attractor across 30+ layers, `qwen35-moe-precision` #152) — the exact "TOPK_MOE mega-fusion" ZINC claims as biggest win, we tried and reverted.
- Full-WMMA flash-attention P@V (`wmma-flash-attention-prefill.md`, fails NLL gate 6×; P stays scalar fp32).
- wave64 for prefill DP4a / MMQ subgroup-32 forcing (ZINC's own null results, plus our wave32-native toolchain default).

**Not-applicable (structurally absent surface):** all Vulkan/RADV-specific levers (dedicated-compute-queue stall, descriptor-buffer, timeline-semaphore pipelining, `rm_kq` Mesa patch, shaderc versions, barrier ledger, user-mode queues) — Rule 7, HIP/HSA already provides the target model. Paged-KV / radix-tree / multi-session / continuous-batching — single-session daemon, no surface. Gemma asymmetric-head-dim / M-RoPE-batch bug — no Gemma in this worktree (arch13 on unmerged branches), no M-RoPE rotation wired.

---

## 5. Provenance appendix

| Lever | Source doc / repo | Node |
|---|---|---|
| GECC/RAS disable (VERIFIED) | ZINC `docs/RDNA4_TUNING.md` + `docs/AMD_GPU_REFERENCE.md` | R9700/gfx1201, Qwen3.6-35B-A3B Q4_K |
| SMU kernel-6.17 clock cap | ZINC `docs/RDNA4_TUNING.md` | R9700/gfx1201 |
| PCIe ASPM performance | ZINC `site/.../2026-07-01-...measured-sweep.md` | R9700/gfx1201, RADV |
| Wide-NUM_ROWS LM-head DMMV | ZINC `site/.../2026-05-16-what-qwen3-151k-lmhead-costs-on-rdna4-decode.md` | R9700/gfx1201 |
| MMVQ↔MMQ batch threshold | ROCmFPX `docs/ROCmFP4-DECODE-SPEED-EXPERIMENTS.md` (`GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH`) | RDNA3.5 Strix, generic gfx |
| MoESD K* / cascade utility gate | ZINC `site/.../2026-05-25-speculative-decoding-on-qwen3-a3b-loses...md` | R9700 |
| wave64-decode (RE-TEST) | ZINC `2026-04-22-...32-column-dmmv.md` + our `project_wave64_neutral_mq4_gemv_gfx1201_2026_07_08` | R9700/gfx1201 |
| Asymmetric K/V + boundary layers | ROCmFPX `README.md` (`-ctk q8_0 -ctv turbo4`, `LLAMA_KV_TURBO_BOUNDARY_LAYERS`) | Strix Halo |
| QJL residual-sign K-quant | ZINC `docs/TURBOQUANT_SPEC.md` | generic RDNA3/4 (accuracy validated, perf est.) |
| Reconstruction-MSE scale search / e4m3 binary-search | ROCmFPX `README.md` + ZINC `TURBOQUANT_SPEC.md` | Strix Halo / generic (CPU encoder) |
| hipGraph rebuild-timing | ROCmFPX `docs/ROCmFP4-DECODE-SPEED-EXPERIMENTS.md` (`LLAMA_GRAPH_BUILD_TIMING`) | generic |
| MoE MMQ cutoff dense-vs-MoE validation | ROCmFPX `docs/ROCmFP4-DECODE-SPEED-EXPERIMENTS.md` | Strix Halo |
| Weight+KV crossover / head×batch-vs-CU pre-screen | ZINC `2026-05-16-...lmhead-costs.md`, `2026-05-06-...prefix-kv-reuse.md`, `2026-05-20-...kv-split.md` | R9700 |
| GPU_MAX_HW_QUEUES runtime A/B | ROCmFPX `docs/ROCmFP4-DECODE-SPEED-EXPERIMENTS.md` | generic |
| Tensor-split hipGraph-reuse guard | ROCmFPX `docs/ROCmFP4-DECODE-SPEED-EXPERIMENTS.md` | generic |
| FP8/FP4-WMMA "no native FP4 builtin" correction | ROCmFPX `docs/...` (rocWMMA 7.1.0 headers) | gfx12 |
| ZINC dead-end guards (LDS-stage / split-K / warp-spec) | ZINC `site/.../2026-06-05-...42-to-208-tok-s.md`, `MULTI_HOUR_EFFORT_15/17/19` | R9700/gfx1201 |
| Batched-prefill / MoE-scatter / GDN-state / fusion families (already-have) | ZINC `2026-04-25`, `2026-05-09`, `2026-06-05`, `MULTI_HOUR_EFFORT_15`; ROCmFPX `README.md` | R9700/gfx1201 |

Every §1 rank and §2 re-test is sized for a single fleet A/B (cross-arch guard protects kernel isolation; system levers are hiptrx-only reboot/sysfs A/Bs). The three cheapest, most-orthogonal, code-free items (Rank 1 GECC, Rank 2 SMU, Rank 3 ASPM) should enter the certify loop first; Rank 5 (MTP batch-K) is the highest-ceiling but largest lift; RE-TEST #1 (wave64) is the highest-value premise correction.
