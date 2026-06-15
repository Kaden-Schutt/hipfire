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
    scale to store). Result on 0.8B weights: **QTIP-2/uniform-2 = 0.26
    (~4× lower MSE), QTIP-2/uniform-3 = 1.41** (within 1.4× of 3-bit, at 2
    bits). Decisive iso-bpw win; not yet uniform-3 parity.
  - **C1c — better codebook (next, the gating blocker).** Reaching uniform-3
    parity is the quality bar before the kernel. Measured: scale refit and
    STATE_BITS=16 give little (1.41→1.34 for 16× cost). The lever is the
    QTIP paper's tuned hash (1MAD/3INST) / structured trellis — not brute
    force. This is the open C1 research item.
  - **C1d — full-model wiring.** New `--format qtip2` (per-group
    `encode_group` across 2D weights; beam-search encoder for throughput —
    per-group Viterbi is too slow at model scale) → QTIP `.hfq` + DType.
  - **Gate:** uniform-3-parity reconstruction (C1c) + `astrea` KLD/PPL on a
    wired model (C1d) BEFORE the decode kernel.
- **C2 — fused QTIP decode GEMV.** Variant of `gemv_mq2g256_lloyd.hip` with
  sliding-window trellis hash (computed codebook → ~zero LDS), reusing
  `rotate_x_mq_awq.hip`. Friction is sub-byte bit-window unpack (LDS stage
  + shift/mask), not serialization (decode is parallel; Viterbi is offline).
- **C2b — dense QTIP prefill GEMM** (mirror the mq3/mq4 `_residual_wmma`).
- **C3.** gfx1103 retune (`gfx-kernel-metadata` for occupancy/LDS/spill).
- **Gate:** coherence + fresh-process `scripts/probe_commits.sh`.

## Phase D — KVarN KV (long-context bandwidth)

- **D1.** KVarN KV compression, gated separately from weight quant;
  keys on a tighter bit budget than values (KV more sensitive than
  weights — `feedback_attention_precision`).
- **Gate:** long-context coherence + τ stability under compressed KV.

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

## Status

- **Phase A: ✅ DONE + verified** (committed).
- **Phase B: ✅ DONE + verified** — MTP daemon wiring, τ=1.66 @ K=2 on
  0.8B, DFlash τ<1 fixed (committed; artifacts in `~/.hipfire/models/`).
- **Phase C: ACTIVE** — QTIP quantizer (C1) is the next build, then the
  fused decode GEMV (C2) + dense prefill GEMM (C2b). Multi-day.
- **Phase D: pending** — KVarN KV compression.
