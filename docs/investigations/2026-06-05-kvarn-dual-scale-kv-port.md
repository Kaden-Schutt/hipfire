# Clean-room port plan — KVarN-style dual-scale 2-bit KV quantization

Status: **research plan / not started.** Author: Kaden Schutt. Date: 2026-06-05.

## Source & licensing posture

- **Paper:** *KVarN* — Muller, Bich, Boretti, Chang, Zhuang, Cavigelli (Huawei Computing
  Systems Lab, Zurich). arXiv:2606.03458.
- **Reference code:** `github.com/huawei-csl/KVarN` (Apache-2.0, Python/vLLM/Triton).
- **Posture: clean-room reimplementation FROM THE PAPER**, in hipfire's own Rust/HIP. We do
  **not** copy their code and do **not** track their repo (submodule PR #407 closed — it was a
  dead pointer that added Apache-2.0 attribution duties + a Huawei supply-chain entanglement in
  an AMD project for zero functional benefit). Algorithms/math are not copyrightable; we
  implement the published method and **cite the paper** as academic courtesy. Apache-2.0 is
  compatible with hipfire's MIT/Apache-2.0 dual license anyway, but the clean-room path avoids
  the code-license surface entirely.

## What we're porting (the technique)

1. **Hadamard rotation** (incoherence processing, QuaRot-style) on the K/V channel dim — **this
   is our existing FWHT**. They absorb 2 of 4 transforms into `W_V`/`W_O` (zero runtime); 2 are
   applied online after RoPE (O(N log N)).
2. **2-bit symmetric uniform quant** (round-to-nearest), per-head: **per-channel for K**,
   **per-token for V**, 128-token blocks.
3. **Dual-axis variance normalization** (the novel bit we lack): two FP8 scales per block —
   per-channel `s_c` (normalizes variance across tokens) and per-token `s_r` (across channels) —
   solved by **alternating log-domain Sinkhorn iteration, online, calibration-free**.
   Dequant: `K_dq = (K_q + z) ⊙ s_chan ⊙ s_tok`.
4. **Outlier guard:** first 128 (attention-sink) + last 128 (recent) tokens kept **FP16**.
5. **~2.3 effective bits/element.**

**Central thesis (why it matters to us):** KV-quant errors *accumulate across autoregressive
decode steps* (not just prefill), caused by **per-token scale misalignment** — the top 5% of
token-scale errors dominate end-to-end KL. The per-token scale `s_r` fixes per-token magnitude so
errors don't compound. This is exactly our recurring **KV-quant → attractor** failure (drift to
attractor within ~10 tokens under greedy decode; the thing coherence-gate.sh keeps catching, and
the motivation for the per-token requant work in #388 / DeltaNet Q8 state).

**Reported results (Qwen3-4B, 2.3 bits/elem):** AIME24 60.0 (KIVI 55.5, KVQuant@2.4b 40.0,
TurboQuant@4.6b 48.9), MATH500 79.2, HumanEval 88.4 — i.e. SOTA at the *lowest* bits, on
reasoning/code tasks = our agentic use case. (NB: the repo's "throughput above FP16" is
marketing; the paper only claims ≤1.4% dequant overhead vs KIVI. This is an **accuracy win at
2-bit**, not a speed win — but 2-bit KV still buys context + decode bandwidth.)

## Why hipfire is well-positioned

- **FWHT KV rotation already exists** (the hard half) — `kv-vquant-fwht-lloyd` work.
- asym3 / Lloyd / q8 KV + per-token requant infrastructure already in the KV path.
- **Native bf16/fp32 KLD eval harness (#398)** to measure the win against an oracle.
- **Coherence gates** (`coherence-gate.sh`, `coherence-gate-dflash.sh`) to *prove* the
  AR-decode attractor is killed — the actual payoff.

## Where it slots in

- **KV-write** (KV quantize path / `kv_cache_write` family): after FWHT, run the dual-axis
  Sinkhorn scale solve, then 2-bit pack. New format/mode (working name `kv2dual`). Store per
  128-block per head: 2-bit codes + FP16 zero-point + `s_chan[head_dim]` (FP8) + `s_tok[128]`
  (FP8). Keep first-128 + last-128 tokens FP16.
- **KV-read** (flash-attention decode kernel, `attention_flash*`): dequant arm applies both
  scales (`⊙ s_chan ⊙ s_tok`). **Hot path** — careful HIP + mandatory perf gate.
- FWHT already applied on the write path; reuse it (verify our layout vs their QuaRot
  W_V/W_O absorption — see Risks).

## Phased plan

- **P0 — CPU reference (Rust, no GPU):** implement dual-scale Sinkhorn + 2-bit quant/dequant;
  validate the math on captured K/V tensors; KLD vs bf16 oracle on a fixed prompt. Cheap, fast,
  de-risks the algorithm before any kernel work.
- **P1 — HIP KV-write kernel:** Sinkhorn scale solve + 2-bit pack + format write.
- **P2 — HIP flash-attention dequant arm:** dual-scale dequant in the decode kernel;
  `probe_commits.sh` perf A/B on gfx1100 + gfx1201 (overhead must hold within a few % vs asym3).
- **P3 — correctness:** `eval_hipfire` KLD vs bf16 oracle (must beat asym3 at lower bits);
  `coherence-gate.sh` + `coherence-gate-dflash.sh` (the attractor-kill is the thesis);
  long-context (>8K) adaptive-KV eval.
- **P4 — ship:** wire as `--kv-mode kv2dual`, default-OFF opt-in until validated; document.

## Mandatory validation gates

- **KLD vs native bf16 oracle** (`eval_hipfire`) — must beat asym3 at *lower* bits, else no point.
- **coherence-gate.sh + coherence-gate-dflash.sh** — the AR-attractor kill is the whole thesis.
- **probe_commits.sh ±1–3%** on gfx1100 + gfx1201 — dequant on the hot path must not regress decode.

## Risks / open questions

- **Hot-path kernel:** the flash-attention dequant adds work to the hottest kernel; the Sinkhorn
  solve adds write-time compute (cheap per paper — verify on RDNA).
- **2-bit is aggressive;** the dual-scale is what makes it hold. If our FWHT layout differs from
  their QuaRot Hadamard absorption (W_V/W_O), accuracy may differ — reconcile the rotation layout.
- **Hybrid/DeltaNet models:** KVarN tested dense Qwen3 attention. Validate the AR-accumulation
  claim + the fix hold on our DeltaNet / hybrid-attention models, not just dense.
- Adaptive KV downshift interaction: our KV tier downshifts mid-sequence; confirm the per-block
  online scales compose with that (they're recomputed per block, so likely fine).

## Expected payoff

KV at ~2.3 bits with FP16-level accuracy → **more context + less decode bandwidth** (decode is
BW-bound on our boxes) **without the KV-quant attractor** — the recurring coherence problem.
