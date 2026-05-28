# Stabilize Before Extraction

Large runtime files such as `crates/hipfire-arch-qwen35/src/qwen35.rs`
should not be split until the no-GPU and hardware gates around the
current production paths are stable.

Extraction boundaries to revisit after the gates are green:

- MoE batched prefill admission and dispatch planning.
- DFlash/MTP target verification plumbing.
- Prefill batch scratch allocation and shape invariants.
- Format-specific grouped GEMM dispatch helpers.

Do not combine extraction with MQ3/MQ6/MTP admission changes. First land
the invariants, then move code with behavior-preserving tests.
