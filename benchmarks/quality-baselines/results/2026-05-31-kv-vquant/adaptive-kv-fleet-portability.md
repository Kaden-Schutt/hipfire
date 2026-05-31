# Adaptive-KV transcode kernels — fleet wave32 portability (2026-05-31)

The four new JIT-compiled transcode/re-rotation kernels
(`kv_transcode_v_q8_to_lloyd4`, `kv_transcode_v_lloyd_down`,
`kv_transcode_k_fwht4_to_fwht2`, `kv_transcode_k_fwht4_to_fwht3`) verified across
the RDNA fleet via the synthetic harness `adaptive_kv_check` (transcode≈direct,
per-case max diff = one quant-boundary step). **Zero code changes required on any
arch** — they share device primitives with the already-fleet-validated fwht K/V
write kernels.

| host | GPU | arch | build | synthetic harness | real-27B coherence |
|---|---|---|---|---|---|
| k9lin | RX 7900 XTX | gfx1100 (RDNA3) | ✓ | ALL PASS | presets + advanced fluent |
| hiptrx | R9700 | gfx1201 (RDNA4) | ✓ | ALL PASS | balanced 750-tok: 4 downshifts @192/463/463/632, last-128 unique 0.812, no attractor |
| hipx | Strix Halo | gfx1151 (RDNA3.5) | ✓ | ALL PASS (numbers identical) | — |
| hipx | RX 5700 XT | gfx1010 (RDNA1) | ✓ | ALL PASS (bonus) | — |

Synthetic per-case bounds (deterministic input, identical across archs):
q8→lloyd4 max 6.78e-1 (≤7.01e-1); lloyd4→lloyd3 8.60e-1 (≤8.99e-1);
lloyd3→lloyd2 1.234 (≤1.249); K fwht4→fwht2 1.593 (≤1.621); K fwht4→fwht3
(re-rotation 128→256) 9.23e-1. K mode-flip asserts pass on every arch.

**Verdict:** wave32-portable across RDNA1 → RDNA4. Verified, not assumed.
