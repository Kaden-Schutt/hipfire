# DS4 MI300X/gfx942 overnight campaign — authoritative goal

This file is the authoritative scope, dependency graph, evidence contract, and exit
criteria for the swarm. `.omp/KICKOFF.md` is only the one-time launch instruction.
If they conflict, this file wins.

## 1. Objective

On the MI300X (`gfx942`), complete these stages in order:

1. **A — current preview MQ4R:** maximize coherent product-path prefill throughput
   and ordinary batch-1 autoregressive decode throughput for the existing DeepSeek V4
   Flash preview MQ4R artifact, without changing a single model byte or serving
   semantic.
2. **B — 0731 MQ4R:** only after A is frozen, quantize
   `deepseek-ai/DeepSeek-V4-Flash-0731` into a faithful, runtime-compatible MQ4R
   product artifact. Then quality-max the weights within that exact format using
   streamed Hessians/GPTQ and promote the best candidate using same-model KLD/PPL,
   runtime correctness, coherent output, and product performance evidence.
3. **C — long-context KV:** only after A and B are frozen, wire and evaluate the
   DeepSeek-specific long-context KV route, with Q8 as control and FWHT3-K/Q8-V as the
   first compressed candidate. Prove actual context capacity, quality, state parity,
   memory use, prefill, and decode behavior before any default changes.

The goal is not complete until A, B, and C each have an exit row in the campaign
ledger and all mandatory artifacts exist. If work is still progressing, keep the goal
active and continue or wait; do not convert incomplete work into a success summary.

## 2. Authority, identities, and protected state

- Work only from
  `/home/kaden/ClaudeCode/autorocm/hipfire/.claude/worktrees/ds4-mi300x-agentmaxx`
  and its corresponding checkout on the MI300X.
- The expected local branch is `codex/ds4-mi300x-agentmaxx`, initially based on
  `d4ab7434a9dad15d0bf6456c8f3c12779ac0edb5`. Record the actual commit and dirty-diff
  hash at every promotion; do not assume the base remains HEAD.
- Do not create another Git branch, worktree, or clone for experiments. Keep rejected
  candidates as explicit patches/evidence and restore only files owned by that
  experiment. Never use a destructive reset.
- `/home/kaden/ClaudeCode/autorocm/hipfire` is a protected dirty source worktree with
  valuable WIP. It is read-only to this campaign. Never build, edit, format, stash,
  reset, clean, delete, or commit there.
- A read-only census may produce a file-by-file import manifest for useful gfx942 WIP
  in the protected source. The conductor may selectively reproduce/import only those
  named files into this isolated worktree after reviewing hashes and diffs. Never bulk
  copy the dirty tree.
- Use `ssh mi300x` only after confirming the endpoint is the intended host. At the
  start of every GPU command, record hostname, ROCm version, visible devices, and the
  line proving the selected device is `gfx942`. Any other device is a hard stop.
- Discover the actual 5 TB scratch mount with read-only filesystem checks. Select one
  campaign root there, record its path and filesystem/free-space evidence, and use it
  for source shards, Hessians, artifacts, builds, and remote evidence. Do not use
  `/tmp`, the root filesystem, or HBM as an accidental backing store for canonical
  data.

## 3. Hard scope and non-goals

- A and B are ordinary batch-1 AR. Force speculation off: no DSpark, DFlash, MTP,
  batched/tree verify, draft, or sidecar. Logs must affirm the ordinary AR route.
- Q8 KV is pinned for every A/B comparison. Expert count, sampling, temperature,
  prompt bytes, chat template, attention mode, and context/output lengths are pinned
  within a comparison.
- MFP3 is off the table. Do not rebake it, evaluate it, or use it to claim speed.
- MQ2/MQ2R is not this campaign. Do not dispatch the MQ2-only agents unless artifact
  inspection proves the user-supplied preview was mislabeled; a filename or old note
  is not proof.
- A cannot change weights, tensor dtypes, packing, quant recipe, bit budget, expert
  count, KV mode, or arithmetic. Kernel/dispatch/fusion changes must be output-exact.
- B must reuse the actual preview MQ4R tensor-class/format contract. Inspect metadata,
  tensor names, dtypes, packing, scales, and rotations; never infer the recipe from
  `.mq4r`. Hessian/GPTQ may choose better quantized values within that contract, but
  may not add a new dtype or change the bit budget.
- No 0.8B/Qwen tuning. Qwen can be used only as read-only prior art or a strict
  neighboring-architecture regression control. No DeepSeek or gfx942 semantic may be
  placed inside a Qwen-owned body.
- No NPU work, no public upload, no push, no registry release, no package/system
  mutation, and no destructive deletion of models, shards, Hessians, or evidence.
- Retained PM4/Redline is optional A-stretch work only after normal HIP parity and a
  tuned HIP product baseline. It is decode-only, not a prefill solution, and it may
  not delay B. No PM4 claim is valid without the current `docs/REDLINE.md` route and
  timed-arm proof.
- No emulator is acceptance evidence. Compilation and ISA inspection may run away
  from the MI300X; correctness and performance promotion require real `gfx942`.

## 4. Orchestration and cost control

- The Opus main session is the only conductor. Every delegated task names one
  explicit `ds4-*` agent. Child agents never delegate.
- Up to 20 read-only research tasks may run concurrently. Exactly one Composer writer
  owns a bounded file slice at a time unless the conductor proves disjoint ownership.
- `ds4-mi300x-operator` is the only agent allowed to run remote builds, tests,
  profiling, benchmarks, calibration, or quantization. It is an operator, not the
  acceptance judge. `ds4-validation` freezes commands and gates first.
- Keep the GPU single-tenant. Before every GPU job, check for competing processes and
  HBM headroom. Never launch a job expected to exceed 85% of usable HBM without an
  explicit streamed/offloaded design and measured allocation budget.
- I/O-only preparation for B may overlap A: resume/download the 0731 source shards,
  verify completeness, hash them, inventory configs, and prepare scratch. No 0731
  calibration, Hessian collection, quantization, KLD/PPL evaluation, or runtime bench
  begins before the A exit row is written.
- Run CPU/source/ISA analysis during model transfer, compilation, or GPU-bound waits.
  Poll long jobs; do not start duplicate downloads/builds because a task is quiet.
- Cache builds by commit, dirty-diff hash, ROCm/clang version, target arch, flags, and
  source hash. Record the loaded HSACO/code-object hashes so stale-cache results cannot
  be promoted.
- Profile before tuning. One experiment changes one hypothesis/lever. A micro win is
  only a screen; product throughput and correctness decide promotion.
- Before editing any Rust/HIP symbol, run the repo-required GitNexus upstream impact
  analysis and surface HIGH/CRITICAL risk to the conductor. Before any local commit,
  run `detect_changes` against the expected base. Use `scripts/fmt-changed.sh`, never
  bare `cargo fmt`.
- The conductor may make local checkpoint commits only after promotion evidence and a
  reviewed scope. Never push during the overnight campaign.

## 5. Evidence and ledger

Create and append to:

`<worktree>/.codeinsight+research/ds4-mi300x-agentmaxx/ledger.jsonl`

Store remote artifacts beneath:

`<selected-5TB-scratch>/hipfire-evidence/ds4-mi300x-agentmaxx/<gate-or-lever>/`

Every ledger row must contain at least:

- UTC timestamp, gate/lever ID, status, verdict, parent/baseline ID;
- local and remote git commit, dirty-diff SHA-256, build/binary SHA-256;
- GPU/ROCm/clang identity and loaded code-object/kernel-bundle hashes;
- model path, source revision, artifact SHA-256, size, recipe/tensor-map identity;
- prompt path and MD5, tokenizer/chat-template identity, prompt/output token counts;
- batch, temperature, expert count, KV/attention/speculation modes, route/fallback
  evidence;
- raw samples, median, dispersion and confidence interval, plus the exact stopping
  decision;
- decoded-output identity/coherence result and the validation artifacts used;
- profile attribution, ISA/resource facts, projected Amdahl gain, measured product
  delta, rejection reason, and what was skipped.

Preserve full machine-readable JSON and logs, not only stdout summaries. Canonical
prompts and quality corpora live in the worktree, dataset cache, or campaign scratch,
never `/tmp`. Append a ledger row before beginning the next lever.

## 6. Measurement contract

### 6.1 Product path

- `hipfire bench` on the ordinary product load/forward path is the authoritative
  prefill+decode measurement. Do not headline a kernel demo or a special example.
- The primary A product fixture is deterministic prose/chat, exactly 2,048 prompt
  tokens and 510 generated tokens, batch 1, temperature 0, Q8 KV, six experts if the
  loaded config says six, and no speculation. Persist the exact prompt and record its
  MD5. If the current artifact legitimately specifies another expert count, record it
  and keep it unchanged; do not silently force a different model semantic.
- Also measure prefill-only path parity at 256, 1,024, and 2,048 prompt tokens with
  the production-equivalent DS4 prefill harness. These are A diagnostics. Contexts
  beyond 2,048 belong to C.
- Report prefill tok/s, decode tok/s, time-to-first-token when available, and total
  wall time separately. Do not blend prefill into AR decode throughput.

### 6.2 Statistical stopping

Do not impose an arbitrary 20-row minimum. Use paired, fresh-process baseline versus
candidate samples in ABBA/interleaved order and sequential confidence stopping:

- Cheap screening may use channel tests, in-model shape micros, and at most two
  exploratory product pairs; exploratory numbers are never claims.
- Promotion starts with five valid paired samples. After each additional pair, update
  a bootstrap 95% CI for the relative product delta. Stop early when either:
  1. the CI excludes zero and its lower bound is at least +0.5%; or
  2. the CI upper bound is below +0.5%, so the candidate cannot clear the checkpoint;
  3. fifteen valid pairs are reached.
- A promotion requires a positive CI and at least +0.5% on the relevant product
  metric, unless several individually micro-positive changes are deliberately bundled
  and the measured bundle clears +0.5%.
- Any invalid sample, thermal/clock fault, fallback, output mismatch, competing GPU
  process, model/cache identity mismatch, or non-stationary arm is rejected and
  repeated only after root cause is recorded.
- Absolute A and B exit numbers use at least five valid fresh processes and include
  the CI/spread. Never infer absolute throughput from an isolated microbenchmark.

### 6.3 Correctness

- Kernel candidates first pass deterministic host/scalar channel tests at the actual
  in-model shapes, including tails, odd shapes, scale/exponent extremes, rotation
  convention, and raw-bit comparisons where arithmetic is supposed to be exact.
- Compile concrete `gfx942`, inspect emitted ISA, and record VGPR/SGPR/AGPR, LDS,
  spills, wave size, matrix ops, and code size before product testing.
- For every kernel/dispatch/fusion candidate, prove the first observable boundary:
  projection/rotate output, logits, KV cache, compressor/recurrent state as applicable.
- A candidate must produce byte-identical greedy decoded output to its unmodified
  same-artifact baseline on the 510-token fixture. B candidates need semantic quality
  comparison because weights differ, but runtime paths for a fixed B artifact must be
  byte-identical across baseline/candidate kernels.
- Before gate exit, run `serve_harness.py` battery/session coverage on two fresh
  processes. Require finite logits, non-empty responses, normal stop behavior, zero
  runaways/attractors, multi-turn state integrity, and human-readable coherent output.
- Retired coherence scripts are not promotion evidence. Follow current
  `docs/VALIDATION.md` for the changed route and preserve its channel/path/state
  artifacts.

## 7. Gate G0 — preflight and truth map

1. Verify local and remote repo/branch/commit/diff identities without mutating the
   protected source worktree.
2. Verify MI300X hostname, exactly selected `gfx942`, ROCm/clang, HBM, host RAM,
   scratch mount/capacity/free space, GPU ownership, and clock/thermal observability.
3. Locate the existing preview MQ4R artifact. Record path, SHA-256, byte size,
   metadata, config, tokenizer, tensor count, complete dtype/class map, recipe markers,
   expert count, and whether `.mq4r` is merely a suffix or a real recipe identity.
4. Locate the 0731 source snapshot/download. Pin the Hugging Face revision and verify
   every expected shard/config/tokenizer file by manifest/size/hash. Resume one
   download if incomplete; do not start duplicates. If the preview artifact is absent
   remotely, locate the known copy on authorized hosts and resume one hash-verified
   transfer into scratch.
5. Run the protected-source read-only census and review any exact gfx942 WIP import
   manifest. Import only individually approved files into this worktree, preserving
   source hashes and recording the resulting patch.
6. Build the committed baseline for concrete gfx942, record the code-object bundle,
   run cheap channel/load smoke, then produce the first `hipfire bench` and prefill
   baseline with an explicitly ordinary AR route.

**G0 exit:** hardware/storage/repo/model identities are pinned; both model datasets
are complete or one verified resumable download is active; preview loads without
hidden fallback; the baseline is coherent; the primary prompt is persisted; and the
ledger/evidence roots exist. A missing identity is a blocker, not an invitation to
guess.

## 8. Gate A — maximize the current preview MQ4R HIP route

### A1. Profile and account for the product workload

- Profile the actual 2,048/510 `hipfire bench` route and the three prefill diagnostic
  shapes. Attribute at least 95% of GPU time by symbol, in-model shape, occurrence,
  bytes, launch structure, and phase. Separate prefill, AR decode, sampling/host, KV,
  recurrent/compressor, and load/JIT costs.
- For the hot set, inspect source and actual gfx942 ISA. Build rooflines from measured
  bytes/operations and device behavior, not peak-spec marketing arithmetic.
- Confirm which tensor formats really dominate the preview. Candidate priority is:
  rotate/FWHT, the actual hot MQ4R expert/dense projection families, MFP4E8 only where
  present, wave64 MFMA/staging, prefill GEMM, AR GEMV, attention/recurrent state, then
  launch/fusion overhead. Do not optimize a format because an old campaign used it.
- Rank candidates by conservative end-to-end Amdahl gain and implementation risk.
  Unknown value is unmeasured, not zero; size it before skipping it.

### A2. Optimization loop

For each ranked lever:

1. Query the existing research/attempt corpus for this exact symbol, shape, arch, and
   lever. Do not repeat a known-negative recipe without new evidence.
2. Freeze files, interface/arithmetic contract, gfx942/DS4 gate, cheapest kill test,
   channel/path/state tests, product fixture, and conservative product projection.
3. Run required impact analysis, then give one bounded slice to the appropriate
   Composer. No broad refactor and no neighboring-arch default change.
4. Compile for concrete gfx942, inspect ISA/resources/spills, and run the channel and
   first-divergence gates. Reject before product load if the intended ISA did not emit,
   occupancy collapsed without a compensating model, or parity fails.
5. Micro-screen across the actual in-model shape distribution. A single favorable
   shape cannot promote a route used by many shapes.
6. Product-screen using the statistical contract. Promote only measured product wins;
   preserve rejected patches, numbers, and explanation in evidence.
7. Reprofile checkpoint bundles because composition is not additive.

Useful levers include legal gfx942 MFMA operand staging, wave64-native reductions,
vectorized aligned loads, pointer iteration, prefetch/double buffering with proven
occupancy, eliminating redundant dequant/rotation, producer-consumer fusion that
removes real materialization, and DS4-scoped launch reduction. These are hypotheses,
not instructions to force a favorite technique. FP8/FNUZ/FP4 builtins count only if
the compiled ISA is legal for gfx942 and preserves the preview format's exact numeric
contract.

### A3. Optional retained replay screen

Only after the best normal HIP route passes A correctness, give
`ds4-redline-later` at most 60 minutes to determine whether gfx942 retained replay is
already technically ready and has a conservative product upside of at least 2%. If
not, ledger the concrete missing prerequisite and proceed to B. If yes, it remains a
separate opt-in experiment and must satisfy `docs/REDLINE.md`; it cannot replace the
HIP baseline or hold B hostage.

### A exit

A is frozen only when all are true:

- preview artifact SHA-256 is unchanged and ordinary AR is proven;
- coherent baseline and best-candidate results exist for the 2,048/510 product
  fixture plus 256/1,024/2,048 prefill diagnostics;
- channel, first-divergence, byte-identical 510-token output, and two-process serve
  batteries pass for the promoted bundle;
- the profile accounts for at least 95% of GPU time and every remaining credible
  independent lever with conservative product upside of at least 1% has been tested,
  killed by evidence, or bounded by a named prerequisite;
- an adjudicated roofline/Amdahl report bounds the combined remaining credible HIP
  upside below 2%, or explicitly proves the hardware-limited gap. Two null experiments
  alone are not a ceiling without this bound;
- best prefill/decode absolute numbers, raw samples, confidence intervals, code-object
  identities, promoted patch/commit, and all negative results are in the ledger.

Write the A exit row and freeze an A binary/kernel bundle before any B computation.

## 9. Gate B — quantize and quality-max DeepSeek-V4-Flash-0731

### B1. Model delta and reference design

- Compare preview versus 0731 config, tokenizer, architecture fields, layer/expert
  counts, tensor names/shapes/classes, RoPE/attention/KV semantics, and source
  revisions. Update only DS4-owned adapters required by verified deltas.
- Fingerprint the quantization/evaluation engine, tokenizer, RoPE convention, dataset,
  chunking, source revision, tensor map, and all artifacts using Astrea conventions.
- Build a higher-quality reference from the same 0731 checkpoint. Preferred order:
  1. same-model Q8 reference if it fits with measured HBM/host headroom;
  2. streamed/offloaded same-model reference-logit generation persisted per corpus
     chunk;
  3. another same-model higher-quality runnable reference with documented error floor.
  Never compare 0731 KLD to preview logits, never omit the reference identity, and
  never require the full BF16 checkpoint to reside in 192 GB HBM.
- Freeze disjoint calibration and evaluation corpora covering prose/chat, code, and
  reasoning/instruction behavior before evaluating candidates.

### B2. Faithful MQ4R control

- Derive the exact preview MQ4R tensor-class policy from the artifact itself: formats,
  group sizes, scales/exponents, FWHT/rotation, protected routers/embeddings/norms,
  shared versus routed experts, and metadata.
- Produce a deterministic, resumable 0731 faithful-control recipe and artifact with
  no Hessian/GPTQ refinement. Validate tensor coverage, alignment, metadata, size,
  checksums, and round-trip load before using it as the same-format control.
- If 0731 contains new/renamed tensor classes, stop that class and make the mapping
  explicit; do not fall through to a default dtype silently.

### B3. Streamed Hessian/GPTQ search

- Collect activations/Hessians layerwise or shardwise with resume manifests and
  checksums. Record coverage for every quantized tensor. Keep source shards and
  Hessians on the selected scratch volume and cap HBM use.
- Start with representative early/middle/late layers and each tensor class to choose
  damping, ordering, block size, calibration count, and any imatrix/AWQ interaction.
  Promote hyperparameters to a full run only when layer screens improve the same-model
  reference error without instability.
- Quantize the full candidate deterministically. Never reuse preview Hessians and
  never mix Hessian shards from another model/source/engine/RoPE fingerprint.
- Preserve faithful-control, each candidate, recipes, manifests, source revision, and
  checksums. A failed full run must be safely resumable; do not discard expensive
  Hessians.

### B4. Quality and runtime promotion

- Use finite same-model reference and candidate logits to compute per-domain KLD and
  PPL on the frozen eval corpus. Record chunk distributions, not only one aggregate.
- Treat non-finite logits as failure. Treat exact KLD zero as an integrity bug until
  distinct artifact hashes, finite logits, and reference/candidate separation are
  proven.
- The GPTQ artifact promotes over the faithful MQ4R control only if aggregate quality
  improves and no hard domain regresses beyond the predeclared measurement noise.
  If no GPTQ candidate clears that bar, the faithful control wins; do not force GPTQ
  merely because it was expensive.
- Load the winner through the production DS4 gfx942 path. Run channel/first-divergence
  coverage, finite logits, two fresh-process serve batteries, multi-turn state, and
  human coherence. Verify no missing-kernel or silent generic fallback.
- Measure the winner with the same A 2,048/510 and prefill fixtures. Report it as 0731
  performance, not as a preview regression/win unless model semantics are proven
  identical.

### B exit

B is frozen only when all are true:

- source revision and every source shard are pinned and complete;
- a loadable faithful MQ4R control and the promoted winner have artifact hashes,
  tensor-map/recipe identity, byte size, and complete quant coverage;
- Hessian provenance/coverage and resume manifests are complete and preserved;
- same-model higher-quality reference KLD/PPL exists for every frozen eval domain,
  with finite-logit and chunk-integrity evidence;
- the promotion verdict between faithful and GPTQ candidates is reproducible;
- the winning artifact is coherent and usable on gfx942 with no hidden fallback, and
  product prefill/AR numbers are recorded under the A fixture;
- a B exit ledger row names the artifact path and all evidence needed to reproduce it.

## 10. Gate C — DS4 long-context Q8 and FWHT3 KV

C may not begin until completed A and B exit rows exist.

1. Audit the winning 0731 model's actual head dimension, RoPE convention, attention
   windows, recurrent/compressor state, KV layout, maximum configured sequence, and
   every DS4 KV write/read/transcode/flash-attention call path.
2. Establish Q8 KV controls at real depths: 2K, 8K, 16K, 32K, and the highest safe
   supported depth allowed by the model and measured memory budget. Record allocated
   versus resident memory, prefill, AR decode, and coherent output.
3. Implement a DS4-and-gfx942-scoped FWHT3-K/Q8-V path only after proving the exact
   FWHT-256/scale/layout contract. Cover single and batched writes, attention reads,
   transcode if required, tails, and every production call path. Do not modify a
   Qwen-owned function body or broaden existing arch defaults.
4. Validate raw channel parity, attention output/logits, KV/recurrent state, finite
   values, two-process serve batteries, multi-turn recall, deterministic retrieval or
   needle tests, and quality/KLD at multiple depths against Q8.
5. Measure memory-capacity gain, prefill, and AR decode at matched actual KV depth.
   Configured max context is not evidence of populated depth.
6. Keep FWHT3 opt-in unless it is coherent, preserves the predeclared long-context
   quality budget, materially increases usable context or reduces memory, and does not
   impose an unacceptable prefill/decode regression. Never hide a speed regression
   behind the capacity number.

**C exit:** the DS4 gfx942 FWHT3-K/Q8-V path is wired, coherent, runtime-selectable,
and fully characterized versus Q8, with an explicit promotion/default verdict. A
compile-only stub, a Qwen-only route, or a rejection report is not a completed C gate.
If a verified architectural impossibility or external blocker prevents this, leave the
goal active/blocked with exact evidence rather than claiming completion.

## 11. Goal completion and unattended behavior

- At startup, create one persistent OMP goal using the exact objective in
  `.omp/KICKOFF.md`. Keep it active through compaction and continuation.
- Report to the user only at A exit, B exit, C exit, a promotion that changes the
  best measured product result, or a hard blocker that truly requires new authority.
- Do not ask the sleeping user to choose between ordinary technical alternatives.
  Use the frozen gates, adjudicator, conservative assumptions, and reversible
  experiments. Record assumptions in the ledger.
- A download/build/quant/benchmark that is progressing is not a blocker. Poll it and
  run orthogonal eligible work. If a task fails, preserve logs, diagnose, and retry
  only with a changed hypothesis.
- If credentials, host access, missing source data, hardware failure, or a necessary
  destructive action blocks the critical path, leave the goal active, write a blocker
  ledger row with exact evidence and the smallest requested intervention, then stop
  unsafe work. Never mark the goal complete to end the session.
- Mark the OMP goal complete only after A, B, and C exit rows exist and the final
  report links the best preview binary, promoted 0731 MQ4R artifact, quality bundle,
  long-context verdict, raw performance samples, negative-results ledger, and exact
  reproduction commands.
