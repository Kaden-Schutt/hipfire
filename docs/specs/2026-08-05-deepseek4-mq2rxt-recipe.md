# DeepSeek V4 Flash 0731 MQ2RXT recipe

Status: frozen product recipe, version 1.

Recipe identity: `deepseek4-mq2rxt-mq4-p3-v1`.

## Intent

MQ2RXT is a distinct DeepSeek V4 Flash 0731 SKU. It preserves the released
MQ2R P3 routed-expert tier and replaces the P3 dense tier with MQ4G256.

It is not an in-place mutation of MQ2R and it is not an E8-to-MQ4
requantization. Every MQ4 tensor is encoded directly from the pinned native
0731 parent checkpoint.

## Frozen inputs

| Role | Identity |
|---|---|
| Parent checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Parent revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| MQ2R trunk byte source | SHA-256 `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce` |
| MQ2R DSpark byte source | SHA-256 `bc695a000643801d26e5ae96c9f4ac4c222a36d9db40566f4cc1de0e9d3d5d2e` |

The MQ2R files are byte-copy bases only. They supply the already-certified
routed MQ2-Lloyd experts and protected tensors; they are never used as a
floating-point quantization source.

## Tensor contract

### Trunk

- 554 P3 dense tensors: native parent to qt=13 `MQ4G256`, group size 256,
  standard MQ FWHT signs with seeds 42 and 1042.
- 33,024 routed-expert tensors: qt=19 `MQ2G256Lloyd`, copied byte-for-byte
  from the released 0731 MQ2R trunk.
- Embedding, norms, control tensors, and all other non-replaced payloads:
  copied byte-for-byte from that same base.
- No qt=35 `MFP4G32E8SOA` tensors may remain.

The 554 dense tensors are exactly:

- `head.weight`;
- all 43 layers' `attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight`;
- all 43 layers' `ffn.shared_experts.{w1,w2,w3}.weight`;
- all 43 layers' `ffn.gate.weight`;
- compressor `wkv` and `wgate` tensors on each configured compressed layer;
- indexer `wq_b`, `weights_proj`, compressor `wkv`, and compressor `wgate`
  tensors on each ratio-4 layer.

### DSpark sidecar

- 24 dense tensors: the eight attention/shared-expert classes above for each
  of the three `mtp` stages, directly quantized from the native parent to
  qt=13 `MQ4G256`.
- 2,304 routed-expert tensors: qt=19 `MQ2G256Lloyd`, copied byte-for-byte
  from the released accepted 0731 DSpark sidecar.
- Seven protected Q8 tensors and 41 F16 tensors: copied byte-for-byte.
- Total tensor count: 2,376. No qt=35 payload and no independent draft head.
- The sidecar reuses the trunk's MQ4G256 head and carries an explicit
  `mq2rxt_sidecar` metadata identity.

## Reproduction

Run the checked-in builder with four explicit paths:

```bash
scripts/quantize-deepseek4-mq2rxt.sh \
  /path/to/DeepSeek-V4-Flash-0731-parent \
  /path/to/deepseek-v4-flash-0731.mq2r \
  /path/to/deepseek-v4-flash-0731-dspark.mq2r \
  /path/to/output-directory
```

The builder verifies both base SHA-256 identities, refuses to overwrite any
output, builds exact 554- and 24-tensor parent-derived overlays, and bakes the
standalone `.mq2rxt` artifacts. The runtime validators reject partial maps,
stale E8 payloads, wrong expert tiers, and mismatched sidecar identities.

## Runtime and Redline policy

The first implementation reuses existing typed MQ4G256 kernels, including
the grouped O-LoRA kernel and the small-B batched GEMM. Kernel specialization
is profile-gated after a coherent artifact exists.

MQ2RXT is not automatically admitted to the MQ2R retained-PM4 tape. It must
produce and certify its own route because the symbol sequence and dispatch
geometry can differ when 554 trunk and 24 DSpark tensors change dtype.
