# Redline Contributor Guide — Documentation Design

**Goal:** Preserve Redline as a reproducible engineering lever by documenting its active product architecture, model-integration recipe, certification gates, performance methodology, and post-Redline kernel-optimization loop in one canonical guide that future maintainers and agents can reliably discover.

**Canonical guide:** `docs/REDLINE.md`

**Discovery hook:** `docs/skills/redline-retained-replay.md`, indexed from `CLAUDE.md`

**Supporting links:** `README.md`, `CONTRIBUTING.md`, and `crates/redline-dispatch/HIPFIRE-GRAFT.md`

---

## Problem

The current Redline knowledge is fragmented:

- `README.md` provides only a short product description.
- `CONTRIBUTING.md` names the validation harnesses without explaining how to construct or certify a retained route.
- `crates/redline-dispatch/HIPFIRE-GRAFT.md` mixes provenance, dated measurements, implementation details, and lifecycle claims.
- `CLAUDE.md` names only the experimental direct-KMD `redline` crate as not wired into serving. It does not identify the active `redline-dispatch` and `redline-rocr` product path.
- Dated performance checkpoints preserve individual experiments but do not define a reusable integration and optimization procedure.

This fragmentation makes three failures likely: treating launch fusion or HipGraph as Redline, modifying shared kernels without preserving the retained-tape contract, and claiming a speedup without proving that PM4 was actually routed.

## Audience

The guide serves:

1. Contributors adding retained replay to a new model, architecture, or decode path.
2. Kernel engineers optimizing an already-certified retained route.
3. Reviewers deciding whether a Redline result is correct, reproducible, and promotion-worthy.
4. Agents that need an explicit discovery path and a fail-closed checklist before changing replay-sensitive code.

The guide assumes familiarity with HIP kernels and hipfire's model architecture crates. It must explain the Redline-specific state, data, ownership, and evidence contracts without requiring the reader to reverse-engineer the implementation.

## Non-goals

The documentation will not:

- turn the experimental direct-KMD `crates/redline` crate into the serving transport;
- prescribe a new runtime state machine or change current product admission;
- make manual shadow validation an automatic runtime gate when the implementation does not enforce that transition;
- promise a universal percentage speedup;
- establish stale benchmark numbers as permanent acceptance floors;
- document unsupported speculative, MTP, prefill, or mutable multi-token routes as certified ordinary-AR replay;
- duplicate low-level API documentation already maintained beside `redline-dispatch` or `redline-rocr`.

## Documentation architecture

### Canonical source: `docs/REDLINE.md`

This file owns the architecture, porting recipe, certification ladder, performance method, optimization loop, failure atlas, and worked cases. Other files link to it rather than restating the procedure.

### Agent hook: `docs/skills/redline-retained-replay.md`

This is a short discovery index, not a second manual. It contains only:

- triggers for when the canonical guide applies;
- examples that are not Redline;
- the distinction among the three Redline crates;
- a mandatory read-first link and direct links to the relevant guide sections;
- one tooling-gap warning that blocks promotion when timed-arm route proof is unavailable.

`CLAUDE.md` will index the hook in its existing `docs/skills/` section.

### Supporting surfaces

- `README.md`: retain the short user-facing overview and link to `docs/REDLINE.md` for architecture and contributor procedure.
- `CONTRIBUTING.md`: retain the mandatory harness commands and link to the guide's certification ladder and evidence contract.
- `crates/redline-dispatch/HIPFIRE-GRAFT.md`: remain the graft/provenance and dated evidence record; link to the canonical guide and remove or clarify claims that conflict with current runtime behavior.
- `CLAUDE.md`: identify all three crates and their distinct roles, then add the skill-index entry.

### Source-of-truth precedence

1. Runtime source and tests define current executable behavior, admission predicates, state transitions, and failure handling.
2. `docs/REDLINE.md` defines contributor workflow and pre-promotion certification policy.
3. Dated performance checkpoints and raw reports preserve observations for exact fixtures; they do not define current defaults or timeless floors.
4. `HIPFIRE-GRAFT.md` and `redline-rocr/PROVENANCE.md` preserve graft and ABI provenance; they do not own the current contributor procedure.

The guide must visibly distinguish runtime fact, certification policy, and dated case evidence. Runtime `Ready` is not proof that a new route has passed repository certification.

## Terminology contract

The guide must use these terms consistently:

- **`redline`**: experimental direct-KMD/bare-libdrm compute crate. It is not the product serving route.
- **`redline-dispatch`**: dispatch-DAG validation and compilation, retained-plan scheduling, dependency policy, and retained AQL/PM4 submission.
- **`redline-rocr`**: public ROCr/HSA ABI and lifetime ownership for agents, queues, memory, packets, signals, doorbells, and completion; it owns no model scheduling or backend policy.
- **Recorder-aware HIP launch**: an ordinary HIP launch that also contributes exact launch metadata to the retained tape. Recorder awareness must not change direct-HIP semantics.
- **Retained tape**: the stable ordered launch sequence plus exact artifacts, padded kernargs, geometry, resource contracts, and dynamic bindings required for replay.
- **PM4 command body**: the architecture-specific indirect-buffer command stream produced from the certified tape.
- **PM4-IB AQL packet**: the public ROCr/HSA vendor packet that submits the retained command body.
- **Runtime admission**: the current code path that decides whether capture, preparation, or replay is eligible.
- **Certification gate**: evidence required before promotion or a performance claim. A certification gate may be stricter than runtime enforcement.

The guide must not call the active product route bare-libdrm/direct-KMD dispatch, and it must not describe HipGraph, per-dispatch AQL publication, launch fusion, or a stable partial recorder fingerprint as retained PM4 replay.

## Canonical guide section contract

### 1. Mental model and architecture

Explain the complete active path:

```text
ordinary recorder-aware HIP launches
    -> stable typed retained tape
    -> architecture-specific PM4 lowering
    -> retained command memory
    -> one PM4-IB AQL packet
    -> public ROCr/HSA queue, doorbell, signal, completion
```

Name the ownership boundary of each crate. Contrast the active route with ordinary serial HIP, HipGraph, per-dispatch AQL, and the experimental direct-KMD crate. State that Redline retains the real model forward; it does not inherently replace the kernels.

### 2. Runtime lifecycle and model boundary

Document the current model-scoped controller lifecycle and the exact sequential-continuation boundary:

- successful model load resets prepared queues, command buffers, tape, and fallback state;
- explicit environment selections override model defaults;
- only an eligible ordinary single-token continuation may arm or consume the plain-AR tape;
- prefill, speculative verification, MTP reseed/proposal work, graph capture, and other non-sequential calls must not contaminate or consume it;
- token embedding and dynamic position preparation may remain outside the retained body where the adapter contract requires it;
- model allocations and all captured pointers must retain identity and lifetime while a prepared plan exists.

The automatic product path transitions `Armed → RecordingWarmup → Captured → Ready` after an eligible HIP warmup completes and preparation succeeds. It does not traverse `ShadowValidated`. The separate manual controller API requires two accepted shadow observations before `install_prepared_plan`, but the automatic Qwen adapter does not use that admission path. Manual multi-position shadow validation, ABI/guard checks, automatic-clock timing, and independent speed samples are mandatory pre-promotion certification evidence, not automatically enforced product admission.

Define fail-closed behavior by phase:

- **Preparation failure:** the ordinary HIP warmup forward has already completed successfully; the controller is poisoned, retained replay is disabled, and later calls use the eligible HIP-side route, which may include HipGraph.
- **Replay-execution failure:** the controller is poisoned and the current forward returns an error; there is no same-forward ordinary-HIP retry. Later retained replay remains disabled.
- **Successful model reconfiguration or swap:** recorded launches, prepared queues, command buffers, and sticky fallback state are reset; admission begins from scratch.

### 3. Reproducible model/architecture porting recipe

The recipe must be ordered and checkpointed:

1. Freeze the exact workload: model digest, architecture, quantization, topology, KV mode, continuation API, binary digest, prompt bytes, and baseline route.
2. Define a fail-closed eligibility predicate for one ordinary sequential generated-token forward. Preserve explicit negative gates for prefill, speculative, graph, capture, batched, and model-swap paths.
3. Count the complete compute body with rocprof or equivalent. State which launches intentionally remain outside the tape.
4. Migrate every in-body raw launch to the typed recorder while preserving the ordinary HIP call. The expected retained count must equal compute launches minus explicitly external launches.
5. Require a stable launch count, unique-kernel set, ordered sequence hash, and owning artifact identity across positions and fresh processes.
6. Bind the exact loaded HSACO, kernel symbol, padded kernarg bytes, launch geometry, shared memory, resource reads/writes, and dynamic fields. Aliases from runtime-specialized names to owning artifacts must fail closed.
7. Make geometry replay-stable. Dynamic grids may use a certified patch surface; dynamic block dimensions or shared-memory assumptions require a fixed/tiled redesign rather than replaying capture-time values.
8. Add model reset, prime, logits, KV, recurrent-state, and any model-specific mutable-state snapshot/restore support required by the shadow oracle.
9. Lower conservatively to a single queue with dependency waits and architecture-correct acquire policy. Treat wait removal, multi-queue phases, CU masks, and register-policy changes as later optimizations.
10. Pass the complete certification ladder before enabling product admission or claiming performance.

Every step must state its observable exit criterion and the failure that blocks advancement.

### 4. Tape and ABI contract

Document the load-bearing invariants:

- the recorder captures exact padded kernarg bytes rather than retaining caller stack pointers;
- the owning HSA-loaded code object must match the captured launch symbol and metadata;
- unsupported scratch, implicit SGPR, loader-kernarg, geometry, or artifact contracts fail before route installation;
- resource dependencies are explicit and conservative until proven otherwise;
- dynamic values are patched only through named, bounded bindings;
- model swap and allocation teardown invalidate every retained pointer and prepared plan;
- incomplete recorder coverage is not a smaller valid tape unless the omitted region is an explicitly external adapter boundary.

### 5. PM4 lowering and hazard policy

Explain that gfx10/gfx11 and gfx12 have separate register encodings and acquire/fence behavior. The initial route must preserve dependencies conservatively. Any wait/acquire elision requires both an exact resource-independence argument and multi-position parity evidence. Multi-queue or phase parallelism additionally requires an explicit fan-in point and must not share a dynamic patch key that assumes one global capture order.

The guide should teach the reasoning pattern, not freeze one architecture's packet recipe as universal.

### 6. Certification and route-proof ladder

Required evidence, in order:

1. **Baseline correctness:** ordinary HIP produces coherent, stable output on the exact fixture.
2. **Capture completeness:** compute count, external-launch count, retained dispatch count, unique kernels, and stable sequence hash reconcile.
3. **ABI/artifact validation:** every launch resolves its exact loaded artifact and loader metadata.
4. **Multi-position shadow parity:** ordinary HIP, retained PM4, and exact HIP-kernarg-blob oracle agree for logits, KV, recurrent/model-specific state, guards, and captured blobs. Bit-exactness is required where the route is intended to preserve execution exactly; any numerical tolerance must be explicitly justified by a changed kernel contract.
5. **Route proof:** report backend request, transport, controller state, dispatch count, packet count, queue identity/count, PM4 dwords, sequence hash, fallback reason, and evidence that the timed arm did not silently use HIP or HipGraph.
6. **Production serve validation:** exercise the user-facing path with the exact model/settings and inspect decoded output, finish state, repetition/attractor health, and model-specific response framing.
7. **Stationary matched performance:** compare the certified baseline and retained route under the methodology below.
8. **Long-context/state-lifecycle validation:** prove dynamic position, KV growth, recurrent state, request reset, and model swap remain correct across the supported operating range.

The guide must state that harness success without route proof is insufficient and a performance ratio from two arms that both executed HIP is invalid evidence.

Manual capture that reports only a stable fingerprint is discovery evidence. It neither installs a plan nor proves that a user-facing forward selected retained AQL or PM4.

### 7. Performance methodology and claim language

Every retained-replay performance claim must record:

- UTC date, branch, source commit, clean/dirty state, and exact daemon/binary digest;
- GPU product and PCI identity, architecture, device topology, ROCm/runtime/driver identity, and clock/governor policy;
- model path, architecture id, quantization, artifact digest, sidecars, and sidecar digests;
- harness path/revision, full command, prompt or deterministic token-stream path and byte digest;
- sampler and seed, KV mode, context, prefill/generated-token counts, parallel topology, graph/spec settings, and all route-affecting `HIPFIRE_*` variables;
- warmup policy, fresh-process or resident-daemon policy, run ordering, run count, minimum/median/maximum, and raw report path;
- capture fingerprint: dispatch count, unique-kernel count, sequence hash, owning artifacts, and dynamic bindings;
- route-proof fields for every arm: backend request, transport, successful preparation, `Ready`, observed replay at multiple positions, packet/queue/dword identity, replay-fault absence, and fallback reason;
- correctness report and decoded-output health.

Compare both tok/s and milliseconds per token. Use matched fixtures; numbers from different harnesses are not an A/B. Automatic clocks are the default unless the report explicitly studies clock policy. Dated case-study numbers are evidence for those exact fixtures, not permanent minimum gains.

The guide must explain two kinds of value:

1. **Direct:** one retained submission removes variable host launch/publication overhead.
2. **Enabling:** after the host floor is reduced, device-side kernel, fence, overlap, and composite transformations can be measured and composed with less host variance.

A historical progression across different kernel stacks must not be presented as a pure transport A/B.

### 8. Post-Redline optimization loop

After conservative retained PM4 is certified:

1. establish a stationary PM4 baseline;
2. profile the new device-bound timeline;
3. choose one device-side lever;
4. compare base PM4 against PM4 plus that single overlay;
5. retain only a correctness-preserving, reproducible win;
6. compose independently proven overlays one at a time;
7. rerun parity, route proof, serve, stationary speed, and long-context gates after every hazard-policy or state-boundary change.

Radiowave transformations and composite kernels belong here. Launch-count reduction alone is not proof of a wall-time win, and under retained PM4 the value of a fusion must come from reduced device work, traffic, or synchronization rather than avoided host API calls.

### 9. Failure atlas

Include diagnosis and required response for:

- capture beginning at model load or prefill rather than the first eligible continuation;
- raw launches missing from the recorder;
- dangling stack-backed kernargs;
- wrong or missing owning HSACO;
- launch-name aliases bound to the wrong artifact;
- capture-time block/shared-memory geometry replayed at another context length;
- mutable token, KV, recurrent, or convolution state omitted from the oracle;
- model swap retaining pointer-keyed state;
- unsupported scratch or implicit-SGPR contracts;
- unsafe wait/acquire removal or incorrect sibling-independence claims;
- timed `auto` and `hip` arms both using ordinary HIP;
- replay failure described as same-token HIP fallback;
- stable partial capture misreported as a complete route;
- a structurally smaller launch graph promoted despite failing wall-time gates;
- cross-harness or cross-binary results compared as one experiment.

### 10. Worked cases

Every case must use the same schema: intent, baseline route, candidate route, architecture/model/quant/topology, immutable retained contract, validation evidence, route proof, matched performance comparison, disposition, and reusable lesson. Rejected cases require evidence-bounded falsification: state exactly which gate the available evidence fails and do not promote a non-diagnostic artifact into a contract violation. A neutral result alone is insufficient. Dated raw evidence remains in its performance checkpoint or report and is linked rather than duplicated wholesale.

Any architecture/model matrix must separate **implementation capability**, **model-route evidence**, **explicit-opt-in availability**, **retained-route certification**, and **automatic-default admission**. An architecture register map, successful model benchmark, or README performance row does not by itself prove that Redline routed.

Required cases:

1. **Qwen3.5 0.8B dense on gfx1201:** compact known-good opt-in capture/parity example. Clearly state that this `.mq4`, `arch_id=5` model requires explicit replay selection and is not the product-default `.mq4r` admission route.
2. **Qwen3.6 35B-A3B MQ4R on gfx1100, gfx1151, and gfx1201:** primary production-model example. Separate the cross-architecture MQ4R model route and its evidence from the narrower current automatic Redline predicate: gfx12, `arch_id=6`, single GPU, `pp=tp=1`, and `.mq4r`. Each architecture row must cite positive route proof before it is labeled retained-replay certified.
3. **Rejected gfx1030 MQ2 lowering:** cite the exact branch/commit and preserve the rejection on the matched product regression and incomplete timed-arm route proof. The off/on reports were separate processes, the daemon hashed raw padded kernarg bytes, the first three pointer fields shifted uniformly by `0x9ef98000000`, and the scalar tails matched. Treat the resulting 942/942 hash drift as non-diagnostic pointer relocation. Require pointer-normalized field comparison or a within-session same-allocation comparison before concluding that an unexpected kernarg mutation violated the immutable tape.
4. **Rejected LFM Stage A:** distinguish serial-HIP activation-preparation fusion from Redline. State that LFM was never **admitted** to a retained route — no Redline admission predicate covers `arch_id==11` (the automatic gate is `arch_id==6` only; manual capture bypasses admission and installs no plan, so its fingerprint is discovery-only, not Redline stage progress). Record the reconciled launch accounting and failed wall-time gates, then show that the recorder→prepared-plan→`Ready`→observed-replay chain was absent and that retained-route admission, recorder completeness, model state, replay-stable attention geometry, shadow support, and PM4 routing remain prerequisites.

### 11. New-route checklist

End with a copyable checklist covering:

- exact admission and negative gates;
- complete recorder coverage;
- artifact/kernarg/resource/dynamic-binding identity;
- stable sequence fingerprint;
- reset and mutable-state oracle;
- architecture-correct conservative PM4 lowering;
- multi-position parity;
- route proof;
- serve and long-context behavior;
- stationary matched performance;
- model-swap and failure-path behavior;
- dated raw evidence location.

## Drift prevention

The documentation update must eliminate contradictory source-of-truth claims:

- `CLAUDE.md` must name all three crates and link the agent hook.
- `HIPFIRE-GRAFT.md` must not claim that automatic product routing enforces manual shadow validation if the controller directly enters `Ready` after preparation. It may state shadow validation as a mandatory certification policy.
- Supporting docs must link to sections in `docs/REDLINE.md` rather than copying procedural text.
- Dated performance checkpoints remain immutable evidence records; the canonical guide explains how to interpret them.
- Environment variables are referenced from their existing canonical environment documentation where available. The guide does not create a second exhaustive env-var table.
- `README.md` must qualify RDNA1-through-RDNA4 statements as implementation capability rather than automatic product admission.
- `HIPFIRE-GRAFT.md` must not present the controller's hard-coded `1.03` field as a durable configured contributor threshold; promotion uses the matched campaign and repository policy defined by the canonical guide.

## Acceptance criteria

The documentation work is complete only when:

1. `docs/REDLINE.md` contains every section and invariant defined above with no TODO, TBD, placeholder, or unsupported claim.
2. A reader can distinguish serial HIP, HipGraph, per-dispatch AQL, retained PM4 through `redline-dispatch`/`redline-rocr`, and experimental direct-KMD `redline` without reading source code.
3. The porting recipe reconciles compute launches, external launches, and retained dispatches and provides a binary exit criterion for each stage.
4. Current automatic runtime behavior and stricter pre-promotion certification policy are explicitly separated.
5. The evidence ladder requires multi-position state parity and positive route proof before any performance claim.
6. Performance methodology prohibits cross-harness, cross-binary, cross-prompt, or silent-fallback comparisons.
7. The post-Redline loop requires PM4-versus-PM4-plus-overlay experiments for kernel and hazard-policy improvements.
8. The four required worked cases are included and source-grounded. The gfx1030 MQ2 case uses evidence-bounded falsification, preserves the proven performance and route-proof failures, and requires pointer-normalized or within-session comparison before attributing raw cross-process hash drift to tape mutation.
9. `docs/skills/redline-retained-replay.md` is concise, links to the canonical guide, and contains no competing procedure.
10. `CLAUDE.md` identifies `redline`, `redline-dispatch`, and `redline-rocr` accurately and indexes the skill hook.
11. `README.md`, `CONTRIBUTING.md`, and `HIPFIRE-GRAFT.md` link to the canonical guide and contain no contradictory lifecycle or transport claims.
12. All added or changed markdown links resolve, the repository's documentation checks pass, and no source code or runtime behavior changes are included.
