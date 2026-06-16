# RoughQuant — autonomous goal prompt

Self-contained prompt for unattended work on RoughQuant (see
`docs/roughquant-spec.md`). Re-feed this verbatim to resume.

---

GOAL: Validate and (if it holds up) integrate RoughQuant into hipfire.

Spec: docs/roughquant-spec.md — read it first, in full. It is the source of
truth for the math, the lineage (ResQ 2412.14363), and the de-risk order.
RoughQuant = the inverse of SmoothQuant: concentrate weight energy into a tiny
fp32-protected low-rank subspace (PCA eigenbasis of C=XᵀX) so the residual bulk
crushes to 1–2 bits. Multi-tier partition by eigenvalue, within-tier Hadamard to
Gaussianize (also validates the QTIP codebook per tier).

THE GATE (the whole project lives or dies on this — do not lose sight of it):
  Does ~2.5 avg-bits with an fp32-protected dominant-energy subspace reach
  4-bit-uniform PPL on a real model+corpus? If yes → proceed to integration.
  If no → document exactly why, where the frontier actually lands, and STOP.
  A null result, fully documented, is a success. Do not force a win.

OPERATING PRINCIPLES (from CLAUDE.md / AGENTS.md — non-negotiable):
- Sim before kernels. Prove numerics on CPU before writing a single HIP kernel.
- Commit every meaningful state — successes AND failures — with structured
  results. The git history is the research. Document WHY things fail.
- Every PPL/perf claim must be reproducible: same model class, same corpus,
  same offset/ctx/warmup; record the exact invocation. For any token-gen perf
  claim, byte-identical prompts (benchmarks/prompts/*.txt) + recorded md5,
  fresh process, warm the cache first. Treat any Δ≥5% as real signal.
- No Python in the inference hot path. Python is fine for OFFLINE tooling
  (Hessian collection, sim glue, analysis); the eventual engine path is Rust-only.
- Portability: every design choice must answer "does this hold on RDNA1→RDNA4?"

FIXTURES — you may (and should) generate what you need; don't be blocked on
missing inputs:
- Model menu: /srv/huggingface holds many bf16 safetensors models
  (models--<org>--<name>/snapshots/<hash>/). Check it BEFORE downloading
  anything. Available includes Qwen3 0.6B/1.7B/4B/8B/14B/30B-A3B,
  Qwen2.5 0.5B/3B/7B/14B, LFM2.5, etc.
- Existing Hessian: ~/.hipfire/hessians/qwen3.5-0.8b.hessian.bin. A 0.8B model
  is fine for the NUMERICS gate (Phases 1–2) — PPL deltas at sub-4-bit are
  visible at this size — but it is NOT a valid vehicle for tok/s figures.
  IMPORTANT: a Hessian is tied to the exact model it was collected from (tensor
  names/shapes must match). Locate the qwen3.5-0.8b safetensors that this
  sidecar corresponds to before running PPL against it; if it can't be found,
  regenerate the sidecar against a model you DO have (next bullet).
- Generating a new Hessian sidecar: scripts/collect_hessian.py
  (--model <hf-dir> --output <out.hessian.bin> --n-sequences 128 --ctx-len 2048
  --corpus wikitext). NOTE: the Rust crates/.../bin/collect_hessian.rs is still
  unimplemented!() — use the Python script (offline tooling, allowed). It writes
  the HFHS-v1 format that hessian_io.rs reads unchanged.
- Generating an .hfq to run PPL: hipfire-quantize --input <model_dir> --output
  <out.hfq> [--format mq4]. Use the existing formats as the 4-bit-uniform and
  QTIP baselines to compare RoughQuant against.
- Two-model strategy: do all numerics/frontier work (Phases 1–2) on the 0.8B
  to iterate fast and cheap. For Phase 3 perf, pick a perf-class model that
  fits the perf box (gfx1100/24GB → quantized 7B–14B; e.g. Qwen3-8B or
  Qwen2.5-7B), collect ITS Hessian, and produce real tok/s there.

WHERE THE REUSE LIVES (verified to exist — start here, don't reinvent):
- C = XᵀX Hessian I/O: crates/hipfire-quantize/src/hessian_io.rs
  (HessianSidecar::open/get, HessianRef::iter_f64/at, check_symmetry/
  positive_diagonal). One artifact → P (eigenvectors=rotation),
  eigenvalues (importance bins), and LDLQ feedback.
- LDLQ / GPTQ: crates/hipfire-quantize/src/{ldlq.rs,gptq.rs}
- QTIP trellis (low-tier format): crates/hipfire-quantize/src/qtip.rs
- FWHT (within-tier Hadamard, and the runtime down_proj rotation later):
  cpu_fwht_256 + FWHT GEMV machinery (grep fwht across crates/).
- QuantLevel enum to generalize per-tensor → per-column-group:
  crates/hipfire-quantize/src/main.rs (~line 3264).
- PPL harness: crates/hipfire-runtime/examples/perplexity.rs
  (model.hfq + corpus.txt --ctx --warmup --offset; 2K tokens sees sub-4-bit
  deltas, 8K+ for stable second decimal). KLD refs: build_kld_ref*.rs.
- Coherence gates: tests/coherence-gate.sh, tests/coherence-gate-dflash.sh.
- Fresh-probe perf: scripts/probe_commits.sh.
- Eval batteries: crates/hipfire-eval (model/runtime evidence belongs here per
  AGENTS.md, not in ad-hoc scripts).

PLAN (map to spec §"De-risk order"; gate at each step before spending the next):

Phase 1 — CPU sim, NO rotation, on the 0.8B. Rank channels by diag(C) proxy,
  protect top-k at fp32/bf16, quantize the rest, dequant, measure PPL via
  perplexity.rs. Sweep k.
  GATE: does protecting a tiny top-k actually move PPL the way the super-weight
  thesis predicts? If protecting <2% of columns doesn't matter, the premise is
  wrong — stop and report.

Phase 2 — Add PCA rotation, on the 0.8B. Eigendecompose C, rotate, bin by
  eigenvalue, within-tier Hadamard, quantize per tier (top=fp32/bf16 dense,
  bulk=QTIP-2/3), dequant, PPL. Sweep: tier count, tier boundaries,
  fp32-vs-bf16 top bin, and the roughquant concentration strength (rank/size
  AND per-tier scale forms). Produce the avg-bits/PPL frontier as a committed
  table + plot data.
  GATE (THE GATE above): ~2.5 avg-bit ≈ 4-bit-uniform PPL? Cross-check the ResQ
  anchor (d/8 @ 8-bit ≈ 4.5 avg-bit) reproduces as a sanity floor.

Phase 3 — ONLY if the frontier wins. Move to a perf-class model (collect its
  Hessian first). Build the offline fold (U_A into adjacent weights:
  o_proj/down_proj right-mult, q/k/v/gate/up left-mult, embed/head), the
  runtime down_proj Hadamard (reuse FWHT kernels), and the per-tier GEMV
  (multi-launch or fused mixed-bit). Generalize QuantLevel to per-column-group.
  Validate: coherence gate MUST pass (tests/coherence-gate.sh; +dflash if the
  spec path is touched), THEN fresh-probe tok/s on byte-identical prompts. A
  correct-but-slower path is an acceptable first landing — correctness first.

OPEN QUESTIONS to resolve empirically in the sim (spec §"Open questions"):
- fp32 vs bf16 top bin — does fp32 buy enough residual-flattening to drop the
  bulk a further bit? (roughquant's core claim)
- tier count: launch cost on small models vs amortization on big ones.
- per-tensor refinement (different bins per weight sharing one rotation) — needed?
- super-weights as a sparse scalar exception bin (SpQR-style) vs full fp32 column.

DELIVERABLES (commit as you go, don't batch to the end):
- findings/roughquant-phase1.md, -phase2.md, -phase3.md — structured results,
  including the avg-bits/PPL frontier table and every swept config, plus the
  exact commands and fixture paths used.
- A clear VERDICT line at the top of each phase doc (PROCEED / STOP + why).
- If integrating: code under crates/hipfire-quantize + crates/hipfire-eval
  battery, behind a flag, with the coherence-gate report attached.
- Update docs/roughquant-spec.md "Status" and resolve the open questions inline
  as they're answered.
- A NEXT-STEPS section at the end of the final phase doc.
- Any generated fixtures (Hessians, .hfq) go to ~/.hipfire/ (NOT committed to
  the repo); record their provenance (source model, corpus, args) in the docs.

STOP CONDITIONS (any of these → stop, write the verdict, do not push further):
- Phase 1 premise falsified (top-k protection doesn't move PPL).
- Phase 2 frontier loses (no avg-bits < 4 reaches 4-bit-uniform PPL).
- A needed input can't be produced cheaply (e.g. Hessian collection OOMs or
  the matching model is absent) — report what's missing and what you tried.
- Any "win" that fails the coherence gate is NOT a win — treat as a regression.

Do not push to remote or open a PR unattended; leave the branch with committed
work and a verdict for human review.
