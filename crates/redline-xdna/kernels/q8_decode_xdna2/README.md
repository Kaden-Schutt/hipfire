# Q8_0 decoder proof for XDNA2

This diagnostic artifact consumes Hipfire/GGML Q8_0 blocks without repacking:
each 32-element block is a little-endian FP16 scale followed by 32 signed
bytes. It emits the decoded values as BF16.

The proof deliberately stops before matrix multiplication. Its purpose is to
certify the exact packed-weight ABI and the AIE-side FP16-scale conversion
needed by the persistent-B W8A16 projection artifact. It is tagged
`q8_decode_bf16_diagnostic`, so the production artifact loader rejects it.

Build inside an activated `mlir-aie`/Peano environment:

```sh
make K=2048
```

The generated PDI is
`build/q8_decode.mlir.prj/main.pdi`; `build/insts_q8_decode.bin` contains the
raw instruction stream.
