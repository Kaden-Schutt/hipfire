// SPDX-License-Identifier: Apache-2.0
//! Deferred-hierarchical KV cache (Phase 2b sub-task 4c, flag-gated).
//!
//! When `HIPFIRE_KV_HIERARCHICAL=1`, the KVarN decode path is replaced by a
//! two-tier cache:
//!   * HOT tier — the most recent `hot_budget` tokens, kept as a raw-f32 ring
//!     `[n_kv_heads × hot_budget × head_dim]` (slot-major). For a single decode
//!     query at the last position every hot token is causally visible, so it is
//!     read by `attention_cold_slots` (slot-major f32), which already emits the
//!     flash partials (m,l) — no `kvarn_attend`/`flash_partials_ml` plumbing.
//!   * COLD tier — older tokens, compacted by `compact_cold_kv` (KVarN 4-bit,
//!     importance-weighted m:1 merge) into segments that stay 4-bit-resident on
//!     GPU and are dequantized on-the-fly each step (`kvarn_dequant_tile` → f16)
//!     and read by the channel-major mode of `attention_cold_slots`.
//!
//! The two tiers are folded by `flash_tier_merge` (online softmax). Migration
//! (hot → cold) fires when the hot ring fills; the heavy compaction is intended
//! to run in idle gaps between turns, but the overflow path keeps it correct
//! during long single-turn generation too. The hot tier being raw-f32 (not 4-bit)
//! costs `hot_budget × kv_dim × 4` B/layer — small; the storage win lives in the
//! 4-bit cold tier that holds the bulk of a long context.
//!
//! v1 uses `rotate=false` compaction (sidesteps the FWHT Q/K basis-cancel
//! question); rotation is a later quality lever. head_dim is fixed at 256 (the
//! kernels' CHD). Parity oracle: `ColdTier::two_tier_attend` + the GPU kernels
//! validated in rdna-compute/examples/parity_{cold_slots,flash_tier_merge,
//! flash_partials_ml,two_tier_e2e,cold_4bit_read}.

use hipfire_kvquant::kv_compact::compact_cold_kv;
use hipfire_kvquant::kvarn::{kvarn_record_bytes, pack_kvarn_tile};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

const HD: usize = 256; // head_dim (kernel CHD)

/// One compacted cold segment, 4-bit-resident on GPU (per kv-head record tiles).
pub struct ColdSegmentGpu {
    pub k_recs: GpuTensor, // [n_kv_heads × rec_bytes] as f32-view (bytes/4 elems)
    pub v_recs: GpuTensor,
    pub n_valid: usize, // real slots attended
    pub n_slots: usize, // padded tile width (= slot_stride for the cold read)
    pub rec_bytes: usize,
}

/// Reusable read scratch (lazily sized to the largest cold segment seen).
struct HierScratch {
    acc_m: GpuTensor, // [n_heads] accumulator flash max
    acc_l: GpuTensor, // [n_heads] accumulator flash denom
    out_c: GpuTensor,
    m_c: GpuTensor,
    l_c: GpuTensor,
    deq_k: GpuTensor, // f16 [n_kv_heads × HD × max_slots]
    deq_v: GpuTensor,
    max_slots: usize,
}

pub struct HierKvState {
    pub enabled: bool,
    pub hot_budget: usize,
    pub migrate_batch: usize,
    pub core_frac: f32,
    pub fold_m: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub hot_k: Vec<GpuTensor>, // [n_layers] slot-major [nkv × hot_budget × HD] f32
    pub hot_v: Vec<GpuTensor>,
    pub hot_count: Vec<usize>, // live hot tokens per layer
    pub migrated: Vec<usize>,  // tokens already moved to cold per layer
    pub cold: Vec<Vec<ColdSegmentGpu>>, // [n_layers][segments]
    scr: Option<HierScratch>,
}

impl HierKvState {
    /// Read `HIPFIRE_KV_HIERARCHICAL` / `HIPFIRE_KV_HOT_BUDGET` /
    /// `HIPFIRE_KV_MIGRATE_BATCH`. Returns a disabled state when the flag is off.
    pub fn from_env(
        gpu: &mut Gpu,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<Self> {
        let enabled = std::env::var("HIPFIRE_KV_HIERARCHICAL").ok().as_deref() == Some("1")
            && head_dim == HD;
        let hot_budget = std::env::var("HIPFIRE_KV_HOT_BUDGET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256usize);
        let migrate_batch = std::env::var("HIPFIRE_KV_MIGRATE_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128usize)
            .min(hot_budget / 2)
            .max(1);
        // Cold-tier compaction knobs. fold_m=1 disables the m:1 merge (cold = pure
        // 4-bit KVarN, no token reduction, no RoPE-phase blur); higher = more
        // compression but more blur. core_frac keeps the top fraction exact (1 slot).
        let fold_m = std::env::var("HIPFIRE_KV_FOLD_M")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4usize)
            .max(1);
        let core_frac = std::env::var("HIPFIRE_KV_CORE_FRAC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.125f32);
        let mut hot_k = Vec::with_capacity(n_layers);
        let mut hot_v = Vec::with_capacity(n_layers);
        if enabled {
            for _ in 0..n_layers {
                hot_k.push(gpu.zeros(&[n_kv_heads * hot_budget * HD], DType::F32)?);
                hot_v.push(gpu.zeros(&[n_kv_heads * hot_budget * HD], DType::F32)?);
            }
        }
        Ok(Self {
            enabled,
            hot_budget,
            migrate_batch,
            core_frac,
            fold_m,
            n_heads,
            n_kv_heads,
            hot_k,
            hot_v,
            hot_count: vec![0; n_layers],
            migrated: vec![0; n_layers],
            cold: (0..n_layers).map(|_| Vec::new()).collect(),
            scr: None,
        })
    }

    fn kv_dim(&self) -> usize {
        self.n_kv_heads * HD
    }

    /// Reset all per-layer tier state for a new sequence (pos==0). Hot ring buffers
    /// are kept (overwritten by `append_token`); cold segments are dropped. Call
    /// once at sequence start. NB: dropped segment GpuTensors are not pool-returned
    /// — a minor VRAM churn at the rare session boundary, not per-token.
    pub fn reset(&mut self) {
        for c in self.hot_count.iter_mut() {
            *c = 0;
        }
        for m in self.migrated.iter_mut() {
            *m = 0;
        }
        for segs in self.cold.iter_mut() {
            segs.clear();
        }
    }

    /// Append one token's K/V (`fa_k`/`fa_v` = [kv_dim] head-major) into the hot
    /// ring at the current tail slot. Migrates the oldest `migrate_batch` tokens
    /// to a cold segment first if the ring is full.
    pub fn append_token(
        &mut self,
        gpu: &mut Gpu,
        layer: usize,
        fa_k: &GpuTensor,
        fa_v: &GpuTensor,
    ) -> HipResult<()> {
        if self.hot_count[layer] >= self.hot_budget {
            self.migrate(gpu, layer)?;
        }
        let slot = self.hot_count[layer];
        let hb = self.hot_budget;
        // fa_k is [nkv × HD] (kv*HD + d); place head kv at hot slot (kv*hb+slot)*HD.
        for kv in 0..self.n_kv_heads {
            let dst = ((kv * hb + slot) * HD) * 4;
            let src = (kv * HD) * 4;
            gpu.memcpy_dtod_at_auto(&self.hot_k[layer].buf, dst, &fa_k.buf, src, HD * 4)?;
            gpu.memcpy_dtod_at_auto(&self.hot_v[layer].buf, dst, &fa_v.buf, src, HD * 4)?;
        }
        self.hot_count[layer] += 1;
        Ok(())
    }

    /// Migrate the oldest `migrate_batch` hot tokens into a new cold segment,
    /// then shift the remaining hot tokens down to the front of the ring.
    fn migrate(&mut self, gpu: &mut Gpu, layer: usize) -> HipResult<()> {
        let mb = self.migrate_batch.min(self.hot_count[layer]);
        if mb == 0 {
            return Ok(());
        }
        let hb = self.hot_budget;
        let nkv = self.n_kv_heads;
        let kv_dim = self.kv_dim();
        // Download hot rings, assemble the oldest `mb` tokens as token-major
        // [mb × kv_dim] for compact_cold_kv.
        let hk = gpu.download_f32(&self.hot_k[layer])?;
        let hv = gpu.download_f32(&self.hot_v[layer])?;
        let mut ck = vec![0.0f32; mb * kv_dim];
        let mut cv = vec![0.0f32; mb * kv_dim];
        for t in 0..mb {
            for kv in 0..nkv {
                let src = (kv * hb + t) * HD;
                let dst = t * kv_dim + kv * HD;
                ck[dst..dst + HD].copy_from_slice(&hk[src..src + HD]);
                cv[dst..dst + HD].copy_from_slice(&hv[src..src + HD]);
            }
        }
        // Uniform importance (the evicted batch is the oldest, ~equal age) → average merge.
        let importance = vec![1.0f32; mb];
        let cold = compact_cold_kv(&ck, &cv, mb, nkv, HD, &importance, self.core_frac, self.fold_m, false);
        let n_slots = cold.n_slots;
        let rec_bytes = kvarn_record_bytes(HD, n_slots);
        let mut krecs = Vec::with_capacity(nkv * rec_bytes);
        let mut vrecs = Vec::with_capacity(nkv * rec_bytes);
        for h in 0..nkv {
            krecs.extend_from_slice(&pack_kvarn_tile(&cold.k_tiles[h]));
            vrecs.extend_from_slice(&pack_kvarn_tile(&cold.v_tiles[h]));
        }
        let k_recs = gpu.upload_raw(&krecs, &[nkv * rec_bytes / 4])?;
        let v_recs = gpu.upload_raw(&vrecs, &[nkv * rec_bytes / 4])?;
        self.cold[layer].push(ColdSegmentGpu {
            k_recs,
            v_recs,
            n_valid: cold.n_valid,
            n_slots,
            rec_bytes,
        });
        self.migrated[layer] += mb;

        // Shift the remaining (hot_count - mb) tokens down to slots [0, ...).
        let rem = self.hot_count[layer] - mb;
        if rem > 0 {
            for kv in 0..nkv {
                let dst = ((kv * hb) * HD) * 4;
                let src = ((kv * hb + mb) * HD) * 4;
                gpu.memcpy_dtod_at_auto(&self.hot_k[layer].buf, dst, &self.hot_k[layer].buf, src, rem * HD * 4)?;
                gpu.memcpy_dtod_at_auto(&self.hot_v[layer].buf, dst, &self.hot_v[layer].buf, src, rem * HD * 4)?;
            }
        }
        self.hot_count[layer] = rem;
        Ok(())
    }

    fn ensure_scratch(&mut self, gpu: &mut Gpu, need_slots: usize) -> HipResult<()> {
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let realloc = match &self.scr {
            None => true,
            Some(s) => need_slots > s.max_slots,
        };
        if realloc {
            let slots = need_slots.max(self.migrate_batch).max(1);
            self.scr = Some(HierScratch {
                acc_m: gpu.zeros(&[nh], DType::F32)?,
                acc_l: gpu.zeros(&[nh], DType::F32)?,
                out_c: gpu.zeros(&[nh * HD], DType::F32)?,
                m_c: gpu.zeros(&[nh], DType::F32)?,
                l_c: gpu.zeros(&[nh], DType::F32)?,
                // f16 dequant scratch: 2 bytes/elem.
                deq_k: gpu.upload_raw(&vec![0u8; nkv * HD * slots * 2], &[nkv * HD * slots])?,
                deq_v: gpu.upload_raw(&vec![0u8; nkv * HD * slots * 2], &[nkv * HD * slots])?,
                max_slots: slots,
            });
        }
        Ok(())
    }

    /// Two-tier decode read for one layer: hot (raw f32) ⊕ all cold segments, all
    /// folded by online-softmax merge into `out` ([n_heads × HD]). `q` = post-RoPE
    /// fa_q ([n_heads × HD]). The flash (m,l) accumulator is internal scratch.
    pub fn two_tier_read(
        &mut self,
        gpu: &mut Gpu,
        layer: usize,
        q: &GpuTensor,
        out: &GpuTensor,
    ) -> HipResult<()> {
        let scale = 1.0f32 / (HD as f32).sqrt();
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let max_seg = self.cold[layer].iter().map(|s| s.n_slots).max().unwrap_or(0);
        self.ensure_scratch(gpu, max_seg)?;
        // Take the scratch out to satisfy the borrow checker, then restore.
        let scr = self.scr.take().unwrap();

        // Hot tier → accumulator (out/acc_m/acc_l). Slot-major f32, stride =
        // hot_budget so the live count reads from a fixed-width ring.
        gpu.attention_cold_slots(
            q,
            &self.hot_k[layer],
            &self.hot_v[layer],
            out,
            &scr.acc_m,
            &scr.acc_l,
            nh,
            nkv,
            self.hot_count[layer],
            scale,
            0,
            self.hot_budget,
        )?;

        // Fold each cold segment: dequant 4-bit → f16, channel-major attend, merge.
        for seg in &self.cold[layer] {
            gpu.kvarn_dequant_tile(&seg.k_recs, &scr.deq_k, nkv, HD, seg.n_slots, seg.rec_bytes)?;
            gpu.kvarn_dequant_tile(&seg.v_recs, &scr.deq_v, nkv, HD, seg.n_slots, seg.rec_bytes)?;
            gpu.attention_cold_slots(
                q,
                &scr.deq_k,
                &scr.deq_v,
                &scr.out_c,
                &scr.m_c,
                &scr.l_c,
                nh,
                nkv,
                seg.n_valid,
                scale,
                1,
                seg.n_slots,
            )?;
            // Merge cold segment into the accumulator (in place — safe, see kernel).
            gpu.flash_tier_merge(
                out, &scr.acc_m, &scr.acc_l, &scr.out_c, &scr.m_c, &scr.l_c, out, &scr.acc_m,
                &scr.acc_l, nh,
            )?;
        }
        self.scr = Some(scr);
        Ok(())
    }
}
