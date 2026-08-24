# Amendment 4 — Q8 query-tiled verify candidate rejected

**Lifecycle:** historical
**Disposition:** rejected, no source/default/product change
**Amends:** [`2026-08-21-qwen38-mqv2-mtp-prefix-cache.md`](./2026-08-21-qwen38-mqv2-mtp-prefix-cache.md), [`AMENDMENT-2`](./2026-08-21-qwen38-mqv2-mtp-prefix-cache-AMENDMENT-2.md), and [`AMENDMENT-3`](./2026-08-21-qwen38-mqv2-mtp-prefix-cache-AMENDMENT-3.md)

## Reason for amendment

Amendments 2 and 3 established a compressed MTP draft-head lever and full-session prefix-cache durability on gfx1201. Separately, a default-off Q8 query-tiled attention verify candidate was screened against the production scalar tiled+reduce path for the Qwen3.8 production attention shape. This amendment records that screen as a closed rejection: the predeclared promotion gate failed, candidate source was removed, and no product path changed.

This is historical evidence only. It does **not** claim the candidate improved throughput, quality, or product behavior, and **no code ships**.

## Fixtures and host

| item | value |
|---|---|
| host | `hiptrx` |
| device | HIP device 3, Radeon AI PRO R9700 gfx1201, BDF `0000:13:00.0`, HIP 7.14 |
| base revision | guarded disposable checkout based on `673ea3ae0` |
| attention shape | Qwen3.8 production: NH24 / NKV4 / HD256 / B7 / CTX24576 |
| promotion gate (predeclared) | candidate ≥ **1.10×** scalar on paired B7/24K wall time |
| guard | exclusive device lock/guard used for all hiptrx runs |

## Designs under test

Two query-tile designs were measured; neither cleared the gate.

### 1. Initial 32-thread row-owned P@V

k9lin micro-screen (not the hiptrx promotion pair):

| batch | scalar | candidate | speedup (scalar/query) |
|---:|---:|---:|---:|
| B7 | 2.464 ms | 6.143 ms | 0.401× |
| B16 | 4.647 ms | 5.530 ms | 0.840× |

### 2. V-sharing design

Wave0 WMMA QK; block sized to `head_dim`; V reused across rows within the block. This was the design taken to hiptrx parity and the fresh B7/24K promotion pairs below.

Offline code-object resources (V-sharing candidate):

| resource | value |
|---|---|
| wavefront | wave32 |
| LDS | 24,960 B |
| VGPR | 55 |
| SGPR | 42 |
| spills / private | 0 / 0 |
| max workgroup | 1024 |

## Measured evidence (hiptrx)

### Parity

- Production shape B7 ragged: **PASS**, relL2 `5.366e-4`, cosine `0.999999940`.
- Shape sweep: B∈{2,4,8,16} × HD∈{128,256} × CTX∈{2048,8192,16384,24576} — **32 shapes, 0 failures**.
- Worst observed relL2 `1.050e-3`; minimum cosine `0.999999583`.

Parity establishes numerical agreement with the scalar reference under the harness thresholds. It is **not** a performance claim.

### Fresh guarded B7 / CTX24576 paired wall times

Three independent guarded paired runs (scalar ms, query-tile ms, scalar/query speedup):

| run | scalar | query-tile | speedup |
|---:|---:|---:|---:|
| 1 | 2.332 ms | 4.728 ms | 0.493× |
| 2 | 1.676 ms | 3.182 ms | 0.527× |
| 3 | 1.668 ms | 3.173 ms | 0.526× |

Median: scalar **1.676 ms** vs query-tile **3.182 ms** → **0.526× speedup**.

The predeclared ≥1.10× promotion gate **failed decisively** (candidate was slower by roughly 2×). Because the gate failed, the compressed-MTP context run that would have exercised the candidate under Amendment 2/3 fixtures was **intentionally not performed**.

### Device health after runs

After the guarded pairs, device3 remained healthy: `runtime_status=suspended`. Guard/lock were used for exclusive access; no recovery action was required.

## Closure

- Env route, query-tile kernel, and harness wiring were **removed** from the disposable checkout.
- Production path remains **scalar tiled+reduce**.
- Source for the six query-tile implementation/harness paths returns exactly to HEAD `673ea3ae0`; only this immutable checkpoint amendment remains.
- **No source, default, or product change** lands from this experiment.

## Non-claims

- Does not promote query-tiled Q8 verify.
- Does not attribute MTP, compressed-head, or session numbers from Amendments 2–3 to this candidate.
- Does not assert occupancy, resource counts, or parity as throughput benefits.
- Does not recommend a follow-up redesign; the measured gate result is the decision.
