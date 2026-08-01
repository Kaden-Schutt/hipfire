# DS4 gfx942 / MI300X invariants

Read this file before acting.

## Objective and order

The authoritative sequence is strict:

1. **A:** maximize coherent prefill and ordinary batch-1 AR decode for the existing
   DeepSeek V4 Flash preview **MQ4R** artifact on gfx942.
2. **B:** only after A is frozen, quantize `deepseek-ai/DeepSeek-V4-Flash-0731` to a
   faithful MQ4R product artifact, then quality-max it with streamed Hessian/GPTQ
   work and same-model KLD/PPL evidence until it is coherent and usable.
3. **C:** only after A and B are frozen, wire and evaluate DS4 long-context KV work,
   including the FWHT3-K/Q8-V route.

The old MQ2R/MFP3 campaign is not this campaign. Do not dispatch MQ2-only agents or
reuse its tensor map unless the conductor first proves that the preview artifact was
misidentified. MFP3, DSpark, MTP, DFlash, tree verification, and other speculation
are out of scope. Retained PM4/Redline is an optional A-stretch only after the normal
HIP route is coherent and measured; it may not delay B.

## Data preservation

- This worktree began from committed `d4ab7434a9dad15d0bf6456c8f3c12779ac0edb5`.
- The source worktree `/home/kaden/ClaudeCode/autorocm/hipfire` contains a large,
  valuable dirty state and untracked gfx942 prototypes that are not present here.
- Never edit, clean, reset, stash, format, build in, or delete from that source
  worktree. Reading it for inventory is allowed only when the assignment says so.
- Importing any source-worktree WIP requires an explicit conductor-owned file list
  and a reviewable patch. Never bulk-copy it.

## Correctness and architecture

- In A, preserve the preview artifact byte-for-byte: weights, tensor formats, expert
  count, sampling, KV mode, and model arithmetic cannot change to manufacture speed.
- In B, clone the preview MQ4R tensor-class policy only after inspecting its actual
  metadata and tensor dtypes. Do not infer the recipe from the filename and do not
  introduce MQ2, MFP3, or a new dtype/bit budget.
- Hessian/GPTQ work must remain within that frozen product format. It may improve
  weight values, calibration, and rounding, but not change the serving contract.
- Treat the rotate/FWHT contract, quant packing, scales, exponent handling, logits,
  KV cache, and recurrent state as correctness boundaries.
- No DeepSeek/gfx942 behavior may bleed into Qwen, gfx11, gfx12, or gfx1100 routes.
  Device/architecture dispatch must fail closed.
- A fast incoherent result is a failure. Locate the first numerical divergence before
  optimizing downstream symptoms.
- Never claim a performance win from a microbenchmark alone. Record the exact model,
  prompt, token counts, route, binary/kernel identities, correctness result, samples,
  and median/spread.
- Do not start Redline/PM4 work until the HIP route is coherent and kernel parity is
  established.
- Do not compare 0731 quality to preview logits. KLD/PPL requires a higher-quality
  reference from the same 0731 checkpoint, with finite-logit checks and a recorded
  tokenizer/model fingerprint.
- Do not require the full BF16 checkpoint to fit HBM. Use layerwise/streamed Hessian
  collection and a same-model Q8 or streamed/offloaded reference-logit path with
  explicit memory headroom.
- Q8 KV is the A/B control. FWHT3 work is C-only and begins as opt-in; no default is
  changed without long-context correctness, quality, memory, and performance proof.

## Swarm mechanics

- The main session is the only coordinator. Every task call names an explicit agent.
- No child agent may spawn another child. Kimi `max` is forbidden.
- Research agents are read-only. Composer agents receive one frozen, bounded file
  slice. Only one writer runs at a time unless the conductor proves disjoint ownership.
- `ds4-mi300x-operator` is the only remote GPU command executor. It executes approved
  commands and returns artifacts; it does not set acceptance criteria.
- `ds4-validation` owns measurement and correctness contracts. `ds4-adjudicator`
  resolves technical disputes before implementation.
- Preserve evidence under a conductor-selected durable directory, never `/tmp` for
  canonical fixtures or promotion artifacts.
- `tools.approvalMode=yolo` exists solely so the authorized overnight campaign does
  not pause. Treat the scope as narrower, not broader: no destructive deletion, no
  package/system mutation, no upload/push, no secret access, and no writes outside
  this worktree, the selected MI300X repo checkout, and the selected scratch/evidence
  roots.

## Required report shape

Return a direct verdict, source or artifact evidence, protected invariants, unknowns,
recommended next action, and an explicit abandon/kill criterion. Mark inference as
inference. Do not silently treat unknown value as zero.
