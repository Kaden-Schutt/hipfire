# Spike 0 — AR replay-graph safety for mid-stream KV-mode switch

**Question:** Does switching the KV tier mid-generation corrupt subsequent
tokens via stale captured-graph state?

**Answer: provably safe today on the linear `generate` path; defensive
invalidation wired for future graph-on correctness.**

## Findings (verified by code read)

1. **AR forward graph is hard-disabled.** `let use_graph = false;` at
   `crates/hipfire-arch-qwen35/src/qwen35.rs:4324`. The whole capture/replay
   branch for the single-token AR forward (`gpu.graph_exec`, `captured_graph`,
   `ar_forward_replay_enabled`) is dead. So the FA attention forward — where
   `kv.v_mode_bits()` flows into `launch_asym_flash_batched` — runs **direct**
   every token, reading `v_mode_bits()` **live**. A mid-stream `v_mode` change is
   reflected on the very next forward.

2. **The `replay_graph_cache` is a different, irrelevant path.** It is the
   DeltaNet GDN **tape** replay (`crates/rdna-compute/src/dispatch.rs:427`,
   keyed by `n_steps`), used only under spec-decode and gated by env
   `HIPFIRE_REPLAY_GRAPH=1` (`speculative.rs:685`). It does **not** capture the
   FA attention forward, so it does **not** bake `v_mode_bits` or the reduce
   selection. Irrelevant to the linear path; defensively cleared anyway.

3. **The latent hazard (future graph-on).** Under `capture_mode`, the V-mode
   kernarg is pushed into a retained blob (`dispatch.rs:24357`) and the
   lloyd-vs-asym reduce kernel is chosen at capture (`dispatch.rs:24281`) — both
   would freeze at capture time. The **already-shipping eviction path** has the
   same latent hazard (it mutates `physical_cap` + buffer contents mid-stream
   with zero graph invalidation — `triattn.rs:833`, `cask.rs:77`) and ships
   today only because `use_graph == false`.

## Action taken

Added `Gpu::invalidate_for_kv_mode_switch()` (`dispatch.rs`, after
`graph_destroy`): calls `graph_destroy()` (drops the AR forward graph + sets
`ar_forward_kernel_dirty` / disables replay) and `replay_graph_destroy_all()`
(drains the GDN tape cache). `transcode_v_step` / `transcode_k_step` call it once
per downshift. Near-no-op today; correct-by-construction if the forward graph is
re-enabled. The end-to-end coherence proof (a real mid-stream switch producing
fluent continuation) lands with the first V transcode (Task 4/5), since a clean
switch can only be demonstrated with a clean transcode.

**Conclusion: unblocked. Proceed to shared infra.**
