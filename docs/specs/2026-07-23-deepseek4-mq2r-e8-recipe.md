<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DeepSeek V4 MQ2R: immutable MQ2-Lloyd experts with an E8 dense route

| Field | Value |
|---|---|
| State | **active campaign** |
| Date | 2026-07-23 |
| Target | DeepSeek V4 Flash, first measured on `gfx1151` |
| Baseline artifact | `deepseek-v4-flash.mq2lloyd` (`arch_id=9`) |
| Candidate artifact | `deepseek-v4-flash.mq2r` (`arch_id=9`, new digest) |
| Format owners | [`docs/QUANTIZATION.md`](../QUANTIZATION.md), [`docs/QUANTIZE.md`](../QUANTIZE.md) |
| Validation owner | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Related research | [`2026-07-19-mq2-level-set-rescue.md`](2026-07-19-mq2-level-set-rescue.md) |

## 1. Decision

Do not mutate, replace, or silently repackage the current
`deepseek-v4-flash.mq2lloyd` artifact. It remains the compatibility and
quality baseline with its current registry identity, sampling profile, MTP
sidecar, and DSpark sidecar.

Build **MQ2R** as a separate mixed-format performance SKU:

- routed experts remain `MQ2G256Lloyd` (`qt=19`) and must be byte-identical
  to the pinned MQ2-Lloyd baseline;
- hot non-expert matrix weights use `MFP4G32E8SOA` (`qt=35`);
- one-dimensional control state, norms, biases, and non-matrix payloads keep
  their existing representation;
- the final artifact is a standalone HFQ, not a required runtime overlay.

`MQ2R` is a **recipe/SKU name**, parallel to MQ4R. It is not a new 2-bit wire
type and it does not overload `qt=19`. The fixed-level MQ2 rescue proposed in
the July 19 spec remains a separate experiment. If that format later wins,
it can replace the routed-expert tier only in a newly named candidate.

## 2. Why this split

DeepSeek V4's routed experts dominate artifact size, while the dense
attention, shared-expert, compressor, indexer, router, and output matrices
are read on every token. The current artifact spends the routed bulk at
2.25 bpw (`MQ2G256Lloyd`) but keeps hot dense weights at Q8 or F16.

E8 is the intended middle tier:

```text
quality / weight:  Q8  >  MFP4E8  >  MQ4
traffic:           Q8  >  MFP4E8 ~= MQ4
```

On `gfx1151`, MFP4E8 is software-decoded; it is not native FP4 ISA. The bet
is therefore bandwidth reduction with an acceptable lattice-decode cost,
not an FP4 matrix-core claim. The current E8 SoA layout is preferred because
it preserves the E8 reconstruction exactly while making codeword reads
coalesced.

The candidate must be built from the original DeepSeek source tensors (FP8
plus their published scales where applicable), not by quantizing the Q8
records in the current HFQ. Q8-to-E8 overlays are permitted as fast
performance screens only; their compounded error is not MQ2R quality
evidence.

## 3. Frozen tensor recipe

The tensor map is part of the model identity. A candidate that changes this
map is not the same MQ2R candidate.

| Tensor class | MQ2R representation | Rule |
|---|---|---|
| Routed experts `layers.L.ffn.experts.E.{w1,w2,w3}.weight` | `MQ2G256Lloyd` (`qt=19`) | Copy or reproduce byte-identically from the pinned MQ2-Lloyd artifact |
| Attention `wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b` | `MFP4G32E8SOA` (`qt=35`) | Quantize directly from the higher-precision source |
| Shared expert `w1`, `w2`, `w3` | `MFP4G32E8SOA` | Quantize directly from source |
| Main compressor `wkv`, `wgate` | `MFP4G32E8SOA` | Precision-sensitive bucket; must pass isolated quality A/B |
| Indexer `wq_b`, `weights_proj` | `MFP4G32E8SOA` | Precision-sensitive bucket; must pass isolated quality A/B |
| Indexer compressor `wkv`, `wgate` | `MFP4G32E8SOA` | Precision-sensitive bucket; must pass isolated quality A/B |
| Router `ffn.gate.weight` | `MFP4G32E8SOA` | Isolated routing-agreement and end-to-end quality A/B required |
| `head.weight` | `MFP4G32E8SOA` | Isolated logit/KLD A/B required |
| `embed.weight` | Existing Q8 representation | No full-matrix decode read; retain the higher-quality lookup tier |
| Norms, biases, HC matrices/scales, sinks, APE, lookup/control tensors | Existing F16/raw representation | No blanket E8 conversion |

“E8 dense” means every two-dimensional, per-token matmul weight outside the
routed experts, subject to the explicit embedding and control-state
exceptions above. It does not mean converting arbitrary F16 metadata or
one-dimensional parameters.

### 3.1 Staged buckets

The writer and evaluator must expose three cumulative buckets:

1. **P1 — projection core:** attention plus shared-expert matrices.
2. **P2 — compression path:** P1 plus main compressor and indexer matrices.
3. **P3 — full MQ2R:** P2 plus router and output head.

P1 and P2 are diagnostic candidates. Only P3 may be named
`deepseek-v4-flash.mq2r`. If P2 or P3 fails quality, either repair the E8
recipe or assign a different, explicit variant name; do not quietly ship a
partial tensor map under the MQ2R name.

### 3.2 Frozen P3 tensor identity and versioned routes

The concrete MQ2R candidate is the exact standalone bake recorded in
[`registry/deepseek4-mq2r-gfx1151-v2.json`](../../registry/deepseek4-mq2r-gfx1151-v2.json).
That manifest separates two identities:

- **MQ2R tensor identity:** the base, P3 overlay, REAP plan, exact 554-tensor
  map, standalone artifact digest, and captured quantizer source.
- **`gfx1151_v2` route identity:** accepted kernel-source hashes, 32-symbol
  decode tape, runtime selectors, kernel-cache bundle, PM4 policy, and the
  dated reference measurement.

The `.mq2r` model identity is intentionally strict: exactly 554 tensors must
be `MFP4G32E8SOA` with the P3 name map, every routed-expert projection remains
`MQ2G256Lloyd`, embedding retains Q8, and runtime overlays cannot mutate the
frozen map. It is not architecture-locked. The artifact loads on every
supported architecture; `gfx1151_v2` is selected by default only on gfx1151,
while other hardware uses its ordinary portable dispatch.

The route pins the accepted U4/grouped-E8, scoregrid, wide-RoPE, fused-HC,
RMS-no-X, and K8-all selectors as its v1 defaults. Later kernel, tape, cache,
or PM4 improvements mint `gfx1151_v2` (or later) without rebaking or renaming
the 82 GB model. P1, P2, alternate P3 heads, and alternate tensor maps still
require a different artifact/recipe identity.

The historical 30.21 tok/s result belonged to `gfx1151_v1`, not to the MQ2R
model identity, and is not current shipping evidence. `gfx1151_v2` replaces
the long-context serial indexer with the exact parallel kernel while preserving
the 554-tensor map and artifact digest. It does not create automatic Redline
admission: retained PM4 remains explicit until the route completes the Redline
certification ladder.

## 4. Artifact construction

### W0 — Pin the immutable baseline

Record before building:

- baseline path, byte size, SHA-256, and tensor-index digest;
- source checkpoint revision and every source shard digest;
- quantizer commit and engine fingerprint;
- MTP and DSpark sidecar identities;
- the current registry sampling profile.

The build must fail if the baseline identity differs from the pin.

### W1 — Assemble the mixed artifact

1. Read routed-expert records from the pinned MQ2-Lloyd artifact, or reproduce
   them from the same source and prove every expert payload is byte-identical.
2. Decode the original DeepSeek FP8-plus-scale source tensors to F32 only at
   the quantizer boundary.
3. Apply the canonical signed FWHT and encode the mapped dense matrices as
   `MFP4G32E8SOA`.
4. Preserve the existing non-matrix F16/raw policy.
5. Write one new `arch_id=9` HFQ with a fresh model digest and a machine-readable
   tensor-policy manifest.
6. Verify that no base tensor was sourced from a Q8-to-E8 round trip.

The manifest must include, for every tensor: source digest, source dtype,
output qt, shape, group size, byte count, and recipe bucket.

### W2 — Sidecars

Initial autoregressive evaluation disables MTP and DSpark.

Package dedicated sidecars:

- `deepseek-v4-flash-mtp.mq2r`
- `deepseek-v4-flash-dspark.mq2r`

The `.mq2r` names are distinct model identities, not compatibility aliases for
the MQ2-Lloyd sidecars. The MTP and DSpark artifacts must be calibrated and
validated against the frozen P3 MQ2R trunk. A byte-identical MQ2-Lloyd sidecar
may be used only as a diagnostic control; it must not be published or
registered as the MQ2R sidecar after target/draft misalignment has been
observed. The loader's existing `<stem>-mtp.<ext>` and
`<stem>-dspark.<ext>` rules then resolve the dedicated artifacts without a
special-case runtime path.

The first diagnostic DSpark artifact carried fail-closed metadata, but its
2,376 tensor payloads were byte-identical to the MQ2-Lloyd source sidecar.
Measurement rejected that restamp (`7.92 tok/s` versus `30.21 tok/s` AR).
It remains pinned in the manifest only as a rejected diagnostic control and
is not registered as an MQ2R sidecar. A promotable sidecar must be built and
calibrated against the frozen P3 target. The optional Q8 draft-head shadow is
also diagnostic-only.

## 5. Runtime contract

The product path must satisfy all of the following before model evaluation:

1. HFQ `qt=35` loads as `DType::MFP4G32E8SOA`; it must never fall through to
   `Raw`/MQ4 interpretation.
2. DeepSeek's architecture-owned FWHT scratch is passed to the E8
   **prerotated** GEMV. A generic `run_auto` call must not rotate it a second
   time.
3. Attention O-LoRA grouped views use the E8 SoA row stride for the exact
   logical group shape.
4. Rotations are reused when projections consume the same activation. MQ2R
   must not pay one redundant FWHT per weight matrix.
5. Long-prompt prefill uses a real dense E8 batched path. A row-at-a-time GEMV
   fallback is acceptable for correctness bring-up, not for the MQ2R SKU.
6. The output-head and router paths have dtype-specific numeric oracles.
7. Unknown shapes or missing kernels fail closed; they do not reinterpret E8
   bytes as MQ4, Q8, or raw F16.

The first performance target is ordinary HIP dispatch on `gfx1151`.
Retained replay is a later transport certification step, not part of proving
the quant recipe.

## 6. Quality program

No quality statement may be made from a text smoke or weight MSE alone.
Use the DeepSeek path's teacher-forced evaluator and retain full identities:

```text
deepseek4_perplexity <model> <corpus>
  --ctx <fixed> --warmup <fixed> --offset <fixed>
  --dump-logits <identity-bound output>
```

Compare the untouched MQ2-Lloyd baseline and P1/P2/P3 with byte-identical
corpora, offsets, KV policy, engine fingerprint, and RoPE convention.
At minimum, use one general-language corpus and one code/agentic corpus.

Required evidence:

- finite logits at every scored position;
- PPL and paired KLD for MQ2-Lloyd, P1, P2, and P3;
- routing-agreement statistics for the P3 router change;
- isolated head-only KLD before accepting the P3 output head;
- sampled user-facing runs at the registry's recommended temperature
  (`1.0`), without changing sampling to hide a regression.

Research promotion thresholds, predeclared for this candidate:

- P3 PPL no more than 3% above the MQ2-Lloyd baseline on either corpus;
- paired `KL(MQ2-Lloyd || P3)` no more than `0.20` nats and greedy top-1
  agreement at least `85%` on each corpus;
- no non-finite row, token-attractor failure, or material task regression;
- P2 and P3 must each show that their added tensor bucket accounts for the
  observed quality delta.

The paired KLD rule intentionally uses an absolute ceiling: the untouched
baseline compared with itself has KLD zero, so a percentage "above baseline"
is undefined. This correction was frozen after P1 and before any P2/P3
measurement; retain that timing in the campaign manifest.

These are candidate decision rules, not model admission. Product admission
still requires the machine-recorded route selected by
[`docs/VALIDATION.md`](../VALIDATION.md).

## 7. Performance program

Follow [`docs/methodology/perf-benchmarking.md`](../methodology/perf-benchmarking.md).
All comparisons use fresh processes, byte-identical committed prompts, and
recorded prompt/model/binary digests.

Primary `gfx1151` goals:

| Metric | MQ2R target |
|---|---|
| Ordinary-AR decode, code fixture | median **at least 30 tok/s** |
| Effective decode bandwidth on the measured resident tensor mix | **200–220 GB/s** |
| Ordinary-AR decode, prose transfer fixture | no more than 1% below the code-independent MQ2R expectation and faster than MQ2-Lloyd |
| Long-prompt prefill | no more than 10% below the MQ2-Lloyd baseline |
| VRAM / artifact size | no larger than the MQ2-Lloyd baseline |

Measure P1, P2, and P3 separately so the compressor/indexer and head/router
traffic savings are attributable. A short warm-cache overlay run may choose
which bucket to build next, but it is not promotion evidence.

After quality passes, collect phase-aware Kernel Atlas rows for the exact
MQ2R tensor mix. Atlas evidence explains kernel fit; it does not prove runtime
correctness or admission.

## 8. Speculative decode and Redline

MTP and DSpark are excluded from the initial 20 tok/s target. After the AR
candidate passes:

1. use the pinned MQ2-Lloyd MTP and DSpark sidecars as diagnostic controls and
   record target/draft agreement, target verification time, and end-to-end
   tok/s separately;
2. build dedicated `.mq2r` sidecars against the frozen P3 trunk;
3. require each dedicated sidecar to recover coherent output and useful
   acceptance before it can be attached to the MQ2R registry entry.

The existing automatic MQ4R retained-route predicate is Qwen/`arch_id=6` and
does not admit DeepSeek `arch_id=9` or a `.mq2r` extension. MQ2R therefore
stays on ordinary HIP unless a new exact-model Redline predicate completes
the certification ladder in [`docs/REDLINE.md`](../REDLINE.md). A successful
manual capture or a fast serial-HIP result is not retained-route proof.

## 9. Acceptance and rejection

Accept the MQ2R recipe for packaging only if:

1. the MQ2-Lloyd routed-expert payload is proven unchanged;
2. the P3 tensor map exactly matches this spec;
3. CPU/GPU format parity and dtype/stride/rotation oracles pass;
4. P1, P2, and P3 quality evidence passes Section 6;
5. P3 reaches the Section 7 `gfx1151` target without a prefill regression;
6. the standalone artifact and sidecars load through the normal model path;
7. any Redline claim independently completes Redline certification.

Reject or rename the candidate if a protected E8 bucket must be reverted.
Do not modify the existing MQ2-Lloyd artifact, registry entry, aliases, or
sampling profile as part of a failed MQ2R experiment.

## 10. Non-goals

- No change to the `MQ2G256Lloyd` wire format or kernel math.
- No implementation of the fixed-level MQ2 rescue inside this SKU.
- No in-place rewrite of a user's MQ2-Lloyd model.
- No claim that E8 is native FP4 on `gfx1151`.
- No weight paging requirement; MQ2R must first work as a fully resident model.
- No automatic Redline admission from the `.mq2r` suffix.
