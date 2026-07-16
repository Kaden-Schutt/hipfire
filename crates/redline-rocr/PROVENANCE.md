# Provenance

`redline-rocr` is an independent Rust implementation against the public
ROCr/HSA ABI. The ROCm source tree is not vendored.

The numeric constants, handle types, function signatures, and C layouts in
`src/abi.rs` are transcribed from the MIT-licensed ROCm 7.2 public headers:

- `/opt/rocm/include/hsa/hsa.h`
- `/opt/rocm/include/hsa/hsa_ext_amd.h`

They were cross-checked against the corresponding files in
[`ROCm/rocm-systems`](https://github.com/ROCm/rocm-systems) at commit
`c0430a50286200ab0562f4733445cdee6e48d416`:

- `projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h`
- `projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h`

The packet encoders, ownership wrappers, finite polling, fault propagation,
and release-ordered queue publication in `src/packet.rs` and `src/runtime.rs`
were written for Redline against that public API. No ROCm implementation file
was copied into this crate.
