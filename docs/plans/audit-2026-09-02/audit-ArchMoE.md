<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# Audit: ArchMoE

# ArchMoE audit — hipfire origin/master @ 8cd15a62b

Scope: hipfire-arch-deepseek4, minimax, cohere2moe, lfm2moe, lfm2-vl, ds4-parent + generate/loader EP wiring. Read-only. gh issue keyword scan MoE/EP/DSML: no matching open issues (exit 0). Arch IDs: DS4=9, MiniMax=10, LFM2-MoE=11, Cohere2-MoE=12; LFM2-VL vision for 11.

## Broken

### 1. EP Unknown fallthrough runs DeepSeek4 EP server (high)
**path:** crates/hipfire-generate/src/ar.rs — select_generation_route + generate_ep  
**verified:** yes

EP arms only 9→Deepseek4Ep, 10→MiniMaxEp; else Unknown (incl. Qwen 5|6 EP, future 11/12). generate_ep: 10→ep_serve_minimax, 5|6→ep_serve_qwen35_dense_tp, **_→ep_serve_ds4**. Any Unknown EP reaching generate_ep runs DS4 EP against wrong weights/protocol (#683 class). Loader today only EpArch::Ds4|Minimax limits blast radius; match still fail-open.

### 2. LFM continuous-batch admission always false (high for dense LFM)
**path:** crates/hipfire-generate/src/batch.rs:168-180  
**verified:** yes

After LfmAr, lfm2_decode_batch, is_dense(), batch_weight_formats_supported → unconditional `return false`. carriers.rs:1470 supports_continuous_batch:true; staging can build dense batch — unreachable. MoE LFM intentionally excluded (batch.rs~84-86); dense LFM was meant to pass. Dead path / half-migration.

### 3. Cohere2-MoE docs vs norm implementation
**path:** hipfire-arch-cohere2moe lib.rs/map vs forward RMSNorm  
**verified:** yes

Docs: mean-centered LayerNorm. Code: rmsnorm_batched / RMSNorm at rms_norm_eps. Code/docs disagree (stale docs vs wrong math).

### 4. DS4 routed MoE scale is permanent defect compensation
**path:** crates/hipfire-arch-deepseek4/src/config_cache.rs:450-510 resolve_route_scale  
**verified:** yes

Ignores cfg.routed_scaling_factor (checkpoint ~1.5). Defaults 2.2 non-mq2r / 1.8 mq2r; env HIPFIRE_DEEPSEEK4_ROUTE_SCALE wins. Restoring 1.5 cost ~51% PPL (16.31 vs 10.81 ctx2048); reference fine at 1.5 (PPL~4.7) — systematic hipfire MoE routed-branch shortfall. Parent path must keep 1.5.

### 5. MTP/SWA reject stale ring residual
**path:** crates/hipfire-arch-deepseek4/src/spec_decode.rs (+ spec_impl/MTP)  
**verified:** yes

Rejected drafts can leave position-indexed SWA/DSA rings inconsistent unless rollback mirrors dense partial-LCP cold-rebuild. Cross-turn reset pairs state.reset + zero_decode_caches; intra-step partial reject weaker. In-code production-hardening follow-up.

## Missing

### 1. LFM2-MoE and Cohere2-MoE EP serve
**path:** ar.rs EP arms; no EpArch 11|12  
**verified:** yes — only DS4/MiniMax EP; LFM supports_ep_batch false; forced EP → Unknown → DS4 fallthrough risk.

### 2. Heterogeneous DS4: no spec/tools until G6
**path:** carriers.rs:1127-1128; deepseek4 carrier ~76-77  
**verified:** yes — SpecTarget "direct-AR only until G6"; draft/DSpark refused until G6/G7. Dense DS4 keeps MTP/DSpark.

Ownership (healthy): hetero Drop/release frees prefill→state→weights.free_gpu(dense,routed) then both pools; RoutedWeights::free_gpu only free_routed_gpu; MiniMax dummy_gate_up owned on layer (no mem::forget leak).

### 3. LFM continuous batch product gap
MoE excluded; dense admission broken → no live LFM continuous batch despite caps.

### 4. LFM multi-turn prefix reuse
**path:** dense.rs:6105-6121; lfm2moe state.reset memsets conv_states  
**verified:** yes — every request cold-resets KV+conv; hybrid conv cannot cheaply rewind. Full history re-prefill each turn. LFM2-VL vision-only crate; text in lfm2moe.

### 5. Cohere block-parallel speculative verify
**path:** spec_impl.rs:16-38 — sequential decode_step only; windowed causal-within-block FOLLOW-UP.

### 6. Runtime expert paging
Load-time EP shard + hetero split only; no ExpertPager; experts process-lifetime resident.

### DSML / invented tokens — clean
**path:** dsml.rs:39-57; git a4630b753 reverts 2b86c6f6e  
a4630b753 restored HF DSML string markup after native-token experiment. String constants only; no hardcoded invented special-token ID table. Fail-closed Malformed on unclosed tool spans. Do not reintroduce numeric invented IDs.

### ds4-parent reachability
**path:** hipfire-ds4-parent; grep loader/generate/daemon clean  
Offline parent-checkpoint oracle/examples only. Banner: NOT production quant calibration ref (PPL~59 vs teacher~4.7). Must not drive serving route_scale or quant. Heterogeneous DS4 architecture-local; no parent hook in Qwen loaders.

## Would change (ranked)

1. Fail-closed generate_ep — explicit DS4 arm; Unknown→error; sync route table. **hours.** Highest priority.
2. Fix or delete LFM batch admission — remove return false OR caps false + delete staging. **hours.**
3. Loader refuse --ep for arch 11/12 (no EpArch). **hours.**
4. Cohere norm doc/code alignment (+ contract test). **hours.**
5. Root-cause DS4 routed MoE shortfall; keep compensation until fixed; parent stays 1.5. **week+.**
6. MTP reject → SWA/DSA scrub to accept boundary; parity vs AR. **days.**
7. LFM multi-turn cache only with conv snapshot proof — else document re-prefill permanent. **days** if pursued.
8. Quarantine ds4-parent in CI/docs (already out of serve). **hours.**

## Confidence

Did: crate maps; EP route/serve; DS4 ep+hetero free/Drop; MiniMax EP ownership; LFM reset/free + generate cold reset; LFM batch admission; Cohere spec_impl + RMSNorm vs docs; DSML strings + revert history; ds4-parent isolation; route_scale; carriers caps/G6; open issue scan empty.

Did not fully deep-read: every MTP accept/reject block; full DSpark rollback matrix; all MiniMax/LFM spec_impl EP interaction; load_model_ep full body (inferred EpArch sites); hardware (forbidden).

Suspicious not Broken: whether HTTP can set m.ep for non-9/10 today (likely no); whether Cohere RMSNorm matches shipped checkpoint.

Peer boundaries: Generate/Runtime/ConfigTopology/Other own overlapping surfaces; ar.rs/batch.rs/carriers.rs cited for MoE wiring only.

---
CONTRACT_JSON_FOR_PARENT: use summary field embedded JSON (slice/broken/missing/changes); full markdown is this architecture field. report path would have been local://audit-ArchMoE.md — parent persists.
