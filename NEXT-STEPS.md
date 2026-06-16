# NEXT-STEPS — Phoenix APU inference plan (QTIP + KVarN + MTP)

Target box: Ryzen 7 7840HS (Phoenix), Radeon 780M (gfx1103, RDNA3),
XDNA1 NPU, 48 GB DDR5-5600 (32+16 SODIMM). **Unified memory** — CPU,
iGPU, NPU all share one ~90 GB/s (theoretical) DDR5 bus; there is no
discrete VRAM.

## Architectural premise (why this plan)

Decode is **memory-bandwidth-bound**, not compute-bound: at AI≈2 against a
~230 FLOP/byte roofline balance point, the 780M alone saturates the DDR5
bus and there is ~100× spare compute. Consequences that shaped the plan:

- **The NPU cannot raise peak decode tok/s.** It sits behind the same
  bus; adding it splits bandwidth, it does not add it. Its only genuine
  roles are low-power offload and a spec-decode draft (small handoff).
- **Cross-engine dequant splits are a net loss.** Dequant *expands* data
  4–8×; the GPU↔NPU (and CPU) handoff has to cross DRAM (no shared
  staging cache — the 16 MB L3 is CPU-attached; the 780M tops out at a
  private 2 MB GL2). Doorbells/fences are control-plane, not data-plane:
  they cannot avoid the bytes. **Dequant must stay fused on the matmul
  engine.**
- **Therefore: spend compute to save DRAM bytes.** Every byte removed
  from the per-token weight/KV stream is close to a linear tok/s win.

## The four levers

1. **QTIP weights** — trellis-coded 2-bit, fused dequant-matmul on GPU.
   Decode is a parallel sliding-window hash (Viterbi is *offline* encode
   only), computed codebook → ~zero LDS. Biggest bandwidth lever
   (weights dominate per-token traffic).
2. **KVarN KV** — KV cache compression (long-context bandwidth).
3. **DeltaNet state precision** — the recurrent state is the most
   precision-sensitive tensor (error compounds over the sequence). Keep
   it as the numerical anchor: FP16/FP32, never Q8 on small models.
4. **MTP draft** — Qwen3.5 ships co-trained MTP heads at *every* size
   (verified: 0.8B/2B carry the full 15-tensor head). Fixes the DFlash
   τ<1 regression on small models because the head is matched to the
   target by construction.

---

## Phase A — DeltaNet precision gating ✅ DONE + verified

Q8 DeltaNet state attractored on long decode for tiny models because the
recurrent state compounds quant error. Replaced the unconditional Q8
default with a gate keyed on **redundancy = `linear_key_head_dim ×
linear_num_value_heads`** (0.8B=2048, 9B=4096, 27B=6144) — a better signal
than parameter count. Threshold env `HIPFIRE_DN_STATE_FP32_BELOW` (default
`usize::MAX` ⇒ FP32 everywhere now). State is ~1–3% of bandwidth, so FP32
is nearly free.

- Impl: `qwen35::{deltanet_state_redundancy, deltanet_state_fp32_below,
  default_state_quant}`; daemon `resolve_tiny_model_state` rewired to the
  redundancy gate (param-count kept as config-parse fallback).
- Verified: unit test (`deltanet_state_gate_keys_on_redundancy`) + 0.8B
  long-decode coherence (uniq 0.46, no attractor) + daemon logs FP32.
- Follow-ups in TODO.md: real FP16 state kernel; FP32/FP16 **tree** replay
  (tree-mode is Q8-only today → MTP draft must stay non-tree, see Phase B).

## Phase B — MTP draft wiring ✅ DONE + verified

Wired the co-trained Qwen3.5 MTP head into the daemon as a spec-decode
drafter, fixing the DFlash τ<1 regression on small models.

- B1: `generate_mtp` in `main.rs` — routed under `mtp_mode` for qwen35
  (arch 5/6) with a bundled/sibling MTP head, no DFlash drafter, greedy.
  Uses non-tree `mtp_spec::spec_step_mtp` (FP32-state compatible; tree is
  Q8-only). Head lazy-loaded from `m.model_path`; rich `done` event with
  `mtp:true,tau,cycles`. `mtp_weights_present` detection extended to
  qwen35 bundled/sidecar heads.
- B2/B3: `qwen3.5-0.8b-mq4.mtp.hfq` (15 tensors, verify PASS) +
  `qwen3.5-0.8b-mq4+mtp.hfq` (bundled, verify PASS).
- B4: **τ on 0.8B** — K=2 τ=1.66 (best tok/s 13.0), K=3 τ=1.62, K=4 τ=1.62.
  **K=2 set as default `mtp_k`.** Output coherent (uniq 0.67, no attractor).

## Phase C — QTIP 2-bit weights (dominant bandwidth lever) — ACTIVE

2-bit is the biggest weight-bandwidth lever, but **scalar/Lloyd 2-bit is
quality-collapse as expected** — the `hipfire-quantize` guard refuses
dense MQ2-Lloyd (0.8B wikitext2 ppl≈19,651; 9B=2,163 vs MQ4=10, MQ3=42).
This is the known reason QTIP/trellis is the *only* viable 2-bit path. Do
**not** build kernels for lloyd-mq2 (or lloyd-mq3) — they're dead ends.

- **C1 — QTIP quantizer (offline).**
  - **C1a ✅ DONE** — trellis encoder core `crates/hipfire-quantize/src/qtip.rs`:
    computed Gaussian codebook (splitmix64 hash → Acklam inv-normal-CDF,
    zero-mean/unit-var), bitshift-trellis sliding-window state, Viterbi
    `encode_group` + reference `decode_group`. Consumes FWHT-rotated groups
    (`cpu_fwht_256` = the incoherence step). Unit-tested: beats uniform
    2-bit MSE by >15% on synthetic Gaussian.
  - **C1b ✅ DONE** — env-gated real-weights reconstruction gate
    (`HIPFIRE_QTIP_EVAL_ST`) + `optimal_scale` (closed-form per-group LS
    scale to store) + correct yardstick (the **2-bit rate-distortion bound**
    σ²/16, not uniform-3 — uniform-3 parity is information-theoretically
    impossible at 2 bits). Result on 0.8B weights: **QTIP-2/bound = 1.21**
    (within 21% of the optimal 2-bit floor), **QTIP-2/uniform-2 = 0.26**
    (uniform-2 ≈ 4.6× the bound — why MQ2-Lloyd collapses). QTIP makes 2-bit
    *near-optimal*.
  - **C1c — adopt the QTIP reference codebook (UNBLOCKED by ./Quantization).**
    My random-hash codebook is at 1.21× the bound; the paper's *computed*
    codebook is the real thing. Port `lib/codebook/bitshift.py` (1MAD/3INST
    hashes) + `lib/algo/ldlq.py` (Hessian-aware LDLQ encode, better than my
    MSE-only Viterbi). Re-run the C1b/whole-model gate to measure the new
    gap-to-bound.
  - **C1d — full-model wiring + PPL (the real gate).** New `--format qtip2`
    (per-group beam encoder across 2D weights) → QTIP `.hfq` + DType. Then
    full-model **PPL** — the usability verdict. Reference: `eval/eval_ppl.py`
    + `quantize_llama/` pipeline. PPL needs a forward → CPU dequant path or
    the C2 kernel (chicken-and-egg: C2 may land first).
  - **Gate:** full-model PPL via C1d (reconstruction settled: near bound).
- **C2 — fused QTIP decode GEMV.** Port the reference decode+matvec kernel
  `qtip-kernels/src` + `test_decompress_matvec.py` (CUDA → HIP) into the
  rdna-compute dispatch; variant of `gemv_mq2g256_lloyd.hip`. Friction is
  sub-byte bit-window unpack, not serialization (decode is parallel).
- **C2b — dense QTIP prefill GEMM** (mirror the mq3/mq4 `_residual_wmma`).
- **C3.** gfx1103 retune (`gfx-kernel-metadata` for occupancy/LDS/spill).
- **Gate:** coherence + fresh-process `scripts/probe_commits.sh`.

## Phase D — KVarN KV (long-context bandwidth)

Reference now in-tree: `./Quantization/KVarN…/` — paper + full repo (spec
`KVARN_MLA_BACKEND_SPEC.md`, Python refs `kvarn_mla_*`, Rust). Algorithm:
variance-normalized 4-bit KV — Sinkhorn per-channel scale/zp + per-token
row scale, GROUP=128 tile records, dequant-to-fp16-scratch then stock decode.

- **CAVEAT — reference is MLA-shaped (DeepSeek-style latent KV, R=512).**
  Qwen3.5 FullAttention is **GQA**, not MLA. The *core algorithm*
  (variance-normalize → 4-bit, keys tighter than values) ports; the
  MLA-latent tile layout does not. Two D-tracks:
  - **D1 (Qwen3.5 GQA KV):** adapt the variance-normalization to GQA K/V
    tensors + hipfire's KV cache (`kv_cache_write_q8`/asym paths). The
    DeltaNet recurrent state is a *separate* concern (Phase A anchor).
  - **D1 BUILD SPEC (scoped 2026-06-16) — it's a SUBSYSTEM, no sim shortcut.**
    Unlike qtip3 (offline tensor transform → existing bf16 forward), KV is
    GPU-fused (write + attention-read kernels), and asym4 quantizes K
    *per-token on write* while KVarN needs a GROUP=128-token *tile* for the
    Sinkhorn balance. So D1 = a staged block-flush KV subsystem (cf. the
    KVARN_MLA_BACKEND_SPEC), four components:
    1. **Staged write:** accumulate fp16 K/V per block_id until GROUP=128 fills,
       then flush. New KvCache mode `"kvarn"` + a staging buffer. (Mirror the
       spec's `do_kv_cache_update`; sink + in-progress tail stay fp16.)
    2. **GPU Sinkhorn** per tile (16 iters col/row std-norm) — new kernel; the
       CPU reference is `kvarn::variance_normalize` (already unit-tested).
    3. **4-bit pack tile** on GPU = `kvarn::quantize_tile` ported to a kernel
       (per-channel scale/zp + per-token s_col, 100 B/group-equivalent record).
    4. **Dequant-on-read:** `_dequant_tile` kernel → fp16 scratch → stock flash
       attention (the spec's "dequant → fp16 scratch → stock decode" — avoids a
       bespoke KVarN attention kernel). Tail/sink blocks copied fp16.
    - **Gate before kernels:** a CPU reconstruction gate on REAL Qwen3.5 KV
      activations (capture via a hook) comparing KVarN vs plain-4bit (asym4
      analog) — confirm Sinkhorn's win survives on real GQA KV before building
      the GPU Sinkhorn (mirrors the qtip2 PPL gate that killed 2-bit). The
      `kvarn::balancing_beats_naive_per_row_4bit` test shows the mechanism works
      when channel skew is present; the gate confirms real KV has that skew.
    - Effort ≈ the C2 effort (4 kernels + plumbing + coherence). Largest single
      remaining item in the plan.
  - **D1-MLA (DeepSeek V4):** the KVarN-MLA backend applies more directly —
    but DeepSeek runs on the bigger boxes, not this 780M.
- **Gate:** long-context coherence + τ stability under compressed KV.

## Phase E — MTP spec-decode optimization (moved from Phase B follow-up)

Phase B wired MTP and fixed DFlash's τ<1, but the **warmed A/B shows MTP is
net-negative on the 0.8B**: AR 57.7 tok/s vs MTP 13.1 tok/s (~4.4× slower),
τ=1.66. Cause (measured, not assumed): MTP does ~2.5× the GPU kernel work
and ~2× the bandwidth *per committed token*, plus heavy host overhead. On a
tiny, already-fast model spec-decode's machinery exceeds its savings; the
payoff is on the larger models (verify-dominated). Phase E is about making
the machinery cheap enough to be net-positive.

**Measured phase split** (`HIPFIRE_MTP_PHASE_TIMERS=1`, 0.8B, 46 cycles,
committed): verify 36.5 ms (46%), draft 23.7 ms (30%), accept+replay
18.9 ms (24%), total 79 ms/cycle. **~half of wall is host launch gaps**
(kernel ≈32 ms vs wall ≈79 ms) — graphing matters here (bandwidth-bound
780M), unlike the ~neutral gfx1201 (compute-bound, conclusions don't
transfer).

Three code-located fixes, ordered by effort↔payoff:
- **E1 — GDN-tape replay elimination (do first).** accept+replay (24%) runs
  a **second full trunk forward** to roll back DN state; MTP passes
  `gdn_tape: None` while DFlash uses a tape. Mirror DFlash's tape rollback →
  kill the redundant forward. Medium effort, known pattern in-repo.
- **E2 — compressed-vocab draft lm_head.** Draft (30%) reuses the full
  248K-vocab lm_head; `mtp_extract --vocab-sidecar` + the
  `spec_step_mtp_compressed` path (already exists) drops it (~1.8 GB/cycle
  BW per the code comment). Also makes the existing MTP **proposal graph**
  eligible (gated on `!use_full_vocab`).
- **E3 — verify graph capture (largest lift).** verify (46%) is mostly host
  launch gaps. Needs the MTP verify forward made graph-capture-ready
  (device-resident positions + `launch_kernel_blob`, mirroring DFlash's
  `verify_dflash_block_with_graph_policy`) for capture-once/replay-many.
- **E4 — tree MTP** (raises τ → more tokens per verify sweep): blocked on
  the FP32/FP16-state tree-replay kernel (TODO.md), since small models run
  FP32 DeltaNet state.
- **Instrumentation shipped:** `HIPFIRE_MTP_PHASE_TIMERS` (phase split);
  per-kernel via `HIPFIRE_PROFILE`; `rocprofv3` for host-gap timeline.
- **Gate:** warmed daemon A/B (decode tok/s, fresh process, byte-identical
  prompt) + `coherence-gate-dflash.sh`.

## Reference sources — `./Quantization/` (local, gitignored)

Papers + reference implementations vendored for Phases C/D:
- **QTIP** `[2406.11235v4]/qtip` — bitshift codebook (`lib/codebook/bitshift.py`),
  LDLQ encode (`lib/algo/ldlq.py`), decode+matvec CUDA kernel
  (`qtip-kernels/`), PPL eval, full `quantize_llama` pipeline. **Covers C1c,
  C1d, C2.** (CUDA → HIP port required for the kernel.)
- **KVarN** `[2606.03458]/KVarN` — variance-normalized 4-bit KV; spec
  `KVARN_MLA_BACKEND_SPEC.md`, Python refs, Rust. **Covers Phase D's
  algorithm/spec** (the gap that previously blocked D) — but MLA-shaped;
  GQA adaptation needed for Qwen3.5 (see Phase D caveat).
- Supporting (rotation / salient / mixed-precision / sparse-outlier context
  for C): ResQ, ROSAQ, CMPQ, SVD, SARQC, LIMPQ, SpQR; HoloKV (alt KV
  compression).

## Cross-cutting invariants

- **Commit as you go** to `origin` (= `xynexus/hipfire`) `chaingun`: one
  commit per meaningful state, push after each. Co-author trailer per repo
  convention. Pull/rebase onto `origin/chaingun` before large edits.
- Coherence-gate before claims (`coherence-gate.sh`; spec-decode →
  `coherence-gate-dflash.sh`); byte-identical prompts w/ recorded md5;
  gfx1103 warm-then-measure protocol.

## Decisions (resolved 2026-06-15)

1. **0.8B role:** standalone small model with **self-MTP draft**. This is
   a weak box — don't spend effort on large models here; they run on more
   powerful machines. So the 0.8B itself gets QTIP'd and self-drafts via
   its own MTP head.
2. **Ordering:** **Phase C (QTIP weights) before Phase D (KVarN).**
3. **Clean-room implementation (2026-06-16):** QTIP and KVarN are
   implemented as fresh Rust, reading `./Quantization/{QTIP,KVarN}` only as
   an algorithm *reference* — no direct code copy (also avoids their
   licenses). E (MTP perf) deferred.
4. **Active goal (2026-06-16): Phase C + D end-to-end on the 0.8B.**
   - QTIP: uniform 2-bit g256 first; if PPL unacceptable, fall back to
     3-bit QTIP (still a bandwidth win vs MQ4) rather than ship garbage.
   - KVarN: adapt the variance-normalization algorithm to Qwen3.5 GQA KV
     (keys tighter than values); the MLA reference is shape-only.
   - Done when both weights (QTIP) and KV (KVarN) are compressed with a
     real PPL + coherence verdict.

## Status

- **Phase A: ✅ DONE + verified** (committed).
- **Phase B: ✅ DONE** — MTP daemon wiring, τ=1.66 @ K=2, DFlash τ<1 fixed
  (committed). NOTE: warmed A/B shows MTP **net-negative on 0.8B** (13.1 vs
  AR 57.7 tok/s); optimization moved to **Phase E**.
- **Phase C: 3-BIT QTIP IS THE VERDICT (2026-06-16).** PPL on 0.8B, byte-identical
  calib-1m (ctx 2048, warmup 8, fresh process, LD_LIBRARY_PATH=/opt/rocm/lib):
  | format | PPL | vs MQ4 |
  |---|---|---|
  | MQ4 (4-bit baseline) | 14.03 | — |
  | **qtip3-sim (3-bit)** | **15.20** | **+8.3%** |
  | qtip2-sim (2-bit MSE) | 120.60 | unusable |
  | qtip2-ldlq (2-bit) | 53.6 | unusable |
  **3-bit QTIP = usable** (+8.3% PPL for 25% less weight bandwidth than MQ4).
  2-bit stays unusable on the 0.8B even with LDLQ. DECISION: ship 3-bit; the
  halo 2-bit finetune (C1h) is NOT worth multi-hours given 3-bit already lands
  at MQ4-class quality.
  - **Packed QTIP-3 format ✅ LANDED (2026-06-16):** `qtip.rs`
    `pack_qtip3_group`/`unpack_qtip3_group` + `QTIP3_BLOCK_BYTES=100`
    ([f32 scale][96 B 3-bit symbols], no zero-point — codebook is zero-mean).
    **0.391 B/weight** (26% < MQ4's 0.53). Round-trips bit-exactly; decode from
    the packed record == direct decode (kernel-faithful). 8×3-bit→3B packing
    matches MQ3's, so the kernel bit-window unpack is shared.
  - **C2 kernel ✅ LANDED (2026-06-16):** `kernels/src/gemv_qtip3g256.hip` —
    fused on-the-fly trellis decode + matvec. Recompute-per-lane computed
    codebook (1MAD hash + baked renorm affine MEAN/INV_STD → bit-identical to
    the PPL-validated sim), ZERO LDS. Parallel decode: state_i = last-4-symbol
    window, each lane reads 3 preceding symbols from the prev chunk. wave32,
    8 w/thread. Verified: compiles gfx1103+gfx1100, VGPR=39/SGPR=17/LDS=0/
    spills=0; test proves parallel-window decode == sequential trellis bit-exact.
  - **C2 END-TO-END ON GPU ✅ (2026-06-16):** real `--format qtip3` emit
    (QuantType::Qtip3G256) + full consumer wiring (DType, loader qt=31,
    FwhtG256 rotation plan, Prerotated/Residual/SwiGLU dispatch + registry +
    launch methods). **GPU parity: real kernel PPL 15.2117 vs sim 15.2040
    (0.05% — bit-faithful decode), at 37 tok/s** (vs bf16 sim 14.9, since it
    reads 0.39 B/w packed weights). Three dispatch bugs found+fixed via GPU
    iteration: gemv.unknown (post-rotation variant), not-registered
    (prerotated registry), residual/swiglu keys.
  - **Coherence ✅ (A/B 2026-06-16):** qtip3 greedy 2048-tok output loops into a
    block attractor on a reasoning prompt — but **MQ4 loops identically on the
    same prompt** (near-verbatim "Wait, re-reading…" repetition). So the
    attractor is a tiny-model long-greedy-decode artifact (documented Phase A
    behavior), NOT a qtip3 regression. qtip3 is numerically correct AND
    coherence-equivalent to MQ4. **PHASE C CORE DONE.**
  - **Mixed-quant ✅ (2026-06-16):** tied embed/lm_head → Q8F16 (gather-friendly;
    can't be trellis-qtip3 — no random access). Was the bandwidth blocker:
    bf16 lm_head [vocab×dim] read every token erased the win (qtip3 40.9 vs mq4
    57.4 tok/s). After Q8: **qtip3 480 MB < mq4 549 MB (13% smaller)**, PPL holds
    **15.20**, decode 40.9→53.3 tok/s. Fair same-2039-token-window: qtip3 ~37 ≈
    mq4 ~38 tok/s.
  - **PERF FINDING:** the 13% byte saving does NOT yet convert to a tok/s win —
    qtip3 is competitive-but-not-faster. The first-cut trellis kernel's extra
    ALU (per-lane 1MAD hash ×8) + untuned occupancy (39 VGPR, generic launch)
    offsets the smaller weight reads on the bandwidth-bound 780M. Converting
    bytes→tok/s needs **C3 gfx1103 retune** (occupancy, LDS-codebook vs
    recompute tradeoff, multi-row). C2b prefill GEMM also remains.
- **Phase C: (superseded) ACTIVE — LDLQ landed; pushing the rest of the QTIP stack.**
  C1e DONE end-to-end: clean-room `ldlq.rs` (inverse-Cholesky 1e-13, per-256
  Hessian FWHT rotation, block-trellis OBS) + `hessian_io` wired +
  `HIPFIRE_QTIP_HESSIAN` in qtip2-sim. **PPL: MSE-only 125.6 → LDLQ 53.6
  (2.3×), still ≫ MQ4 14.0.** Decision (2026-06-16): **push 2-bit** with the
  rest of the QTIP stack to chase usable:
  - **C1f — V=2 vector trellis** (2 weights/codebook-entry; the main RD lever).
    Restructures qtip.rs: K=4 bits/step, 128 steps/256-group, codebook[state]
    = 2-vector. Biggest quality gain.
  - **C1g — L=16** trellis state (richer codebook; pairs with V=2).
  - **C1h — finetune:** was "impractical (this box torch is CPU-only)" — but
    **halo (172.16.16.20) unblocks it** (2026-06-16): Strix Halo gfx1151,
    124 GB unified RAM, ROCm-7.13 torch 2.12 in `~/.venv` that runs on the
    8060S GPU (matmul verified, ~3 TFLOP/s fp32 — iGPU-class but real GPU, not
    CPU). `device_count()` reports 0 (gfx1151 enum quirk) but `device="cuda"`
    tensors execute. So GPU torch (Hessian collection AND end-to-end finetune)
    is now POSSIBLE — slow at 3 TFLOP/s, but feasible for the 0.8B over hours.
  - **qtip3-sim fallback ✅ WIRED (2026-06-16):** bit-rate-parametric trellis
    (`*_bits` fns) + `--format qtip3-sim`. Quantize + PPL pending.
  Realism: even the full stack reaches usable 2-bit mainly on 7B+; a 0.8B
  dense model may not hit MQ4-usable regardless. halo's 124 GB also lets us
  validate the QTIP-2 stack on a 7B+ model where the paper says it works.
- **Phase C (earlier verdict, superseded): 2-bit QTIP FAILS (MSE-only).**
  Built `--format qtip2-sim` (simulated QTIP-2 → bf16, kernel-free PPL via
  the normal forward) + 1MAD codebook + sort-based beam encoder. PPL on
  0.8B/calib-1m: **QTIP-2-sim 125.6 vs MQ4 14.0 vs bf16 ~10.9** — 2-bit
  unusable as implemented. Root cause: near-optimal *reconstruction* (1.21×
  RD bound) still collapses PPL because error accumulates across 186 weights
  AND the encode is MSE-optimal not **output-optimal** — missing **LDLQ**
  (Hessian/activation-aware) + V=2/L=16/finetune. FORK:
  - **C1e — LDLQ (rescue 2-bit) — CHOSEN, and tractable (infra exists).**
    hipfire already has: `collect_hessian` (native per-layer input-Hessian
    calibration), `gptq.rs` (Cholesky + inverse-Cholesky OBS error feedback,
    `fwht_similarity_per_256` = FWHT incoherence on the Hessian, AWQ
    rescaling), and `quantize_mq2g256_lloyd_gptq` (a working GPTQ-LDLQ for
    2-bit Lloyd). So QTIP-LDLQ = **swap that path's Lloyd scalar quantizer
    for the trellis `beam_encode_group` on the feedback-adjusted target.**
    Steps & exact design (scoped 2026-06-16):
    - **Hessians:** native `collect_hessian` is a SCAFFOLD (panics) — use the
      Tier-2 Python collector `scripts/collect_hessian.py` (torch CPU-only
      here, so slow; offline tooling, Rule-1-OK). HFHS read by `hessian_io.rs`.
    - **CORRECTION (2026-06-16): `gptq.rs` + `hessian_io.rs` are ORPHANED** —
      never declared as modules (no `mod gptq;`), and `gptq.rs` uses `faer`
      which isn't a dependency, so they don't compile. NOT reusable infra as-is.
      Reviving = add `faer` + revive ~1658 lines of possibly-API-drifted code.
      The in-`main.rs` `quantize_mq2g256_lloyd_gptq` is only DIAGONAL (imatrix).
      → **Clean-room path (per the goal): new compiled `ldlq.rs`** with own
      damped inverse-Cholesky-upper (unit-tested vs UᵀU≈(H+λI)⁻¹), own per-256
      Hessian FWHT rotation, and the block-sequential trellis OBS loop
      (algorithm written + the OBS-beats-no-feedback unit test designed).
      gptq.rs is kept only as an algorithm reference.
    - **New piece = `gptq_pipeline_qtip2`:** same prologue, but replace the
      per-element MQ4 quantizer in the OBS loop with a **block-sequential
      trellis**: process k-columns in 256-blocks, trellis-encode each block
      (`beam_encode_group` on the OBS-feedback-adjusted target), propagate
      the block residual to later columns via the inverse-Cholesky factor
      (cf. QTIP `ldlq.py` block loop). v1 keeps the 1D-group scalar trellis
      (not QTIP's 2D-tile V=2) — a deliberate simplification to validate the
      Hessian-feedback win first.
    - Re-quantize qtip2-ldlq-sim → PPL vs MQ4. Target: PPL ≪ 125 → toward
      MQ4's 14 / below. NOTE: existing `quantize_mq2g256_lloyd_gptq` is only
      a *diagonal* (imatrix) GPTQ, not full-Hessian — must use the
      gptq_pipeline (full H) path.
  - **3-bit QTIP fallback (plan default):** make bits configurable, quantize
    qtip3-sim, PPL. Quick; modest bandwidth win vs MQ4.
  Then C2 decode kernel (only worth building once a bit-rate passes PPL).
- **Phase D: clean-room CPU core LANDED (2026-06-16).** `hipfire-quantize/src/
  kvarn.rs` — log-domain Sinkhorn `variance_normalize` (imbalance 167→3.5,
  perfect=2.0) + per-channel 4-bit `quantize_tile`/`dequantize_tile`
  (`deq=(q*scale_abs+zp_abs)*s_col`, per-row Sinkhorn scale absorbed). Tests:
  Sinkhorn reduces imbalance, 4-bit cos-sim 0.9955 (un-rotated core; FWHT
  upstream lifts to the spec's 0.999 at the wiring layer), KVarN MSE 0.43× naive
  per-row 4-bit on a skewed tile. Operates on a generic `[R,C]` tile — the GQA
  adaptation. NEXT (D1): wire into the KV path (tile per `(layer,kv_head)` over
  head_dim, share engine FWHT) + long-context coherence gate.
- **Phase D (ref):** KVarN paper + repo in `./Quantization/` — algorithm ported
  clean-room; MLA tile layout (R=512 latent) intentionally NOT ported (GQA).
- **Phase E: scoped + instrumented** — MTP perf (phase split measured; E1
  GDN-tape replay, E2 compressed draft lm_head, E3 verify graph).
