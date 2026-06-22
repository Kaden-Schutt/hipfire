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
//! The two tiers are folded by `flash_tier_merge` (online softmax). The hot tier
//! being raw-f32 (not 4-bit) costs `hot_budget × kv_dim × 4` B/layer — small; the
//! storage win lives in the compacted cold tier that holds the bulk of a long
//! context. head_dim is fixed at 256 (the kernels' CHD).
//!
//! Migration (hot → cold) has two paths: an overflow fallback on the critical path
//! (`migrate_n(migrate_batch)` when the ring fills), and `idle_compact` — the
//! deferred drain run between turns (off the latency path; see
//! `qwen35_prefill_active_session`). Both fold a token range into ONE cold segment
//! via `compact_cold_kv`.
//!
//! Cold compaction is tunable (defaults shown):
//!   * importance (`HIPFIRE_KV_IMPORTANCE`): vnorm (best) | uniform | knorm |
//!     kvnorm | attn — ranks which cold tokens stay exact (core) and weights the
//!     merge average; vnorm beats the others (attn underperforms — see commit log).
//!   * merge (`HIPFIRE_KV_FOLD_M`=4, `HIPFIRE_KV_CORE_FRAC`=0.125,
//!     `HIPFIRE_KV_POS_LOCAL`=on): m:1 importance-weighted average of the non-core
//!     tail, grouped by adjacent position to limit RoPE-phase blur (the dominant
//!     merge cost). fold_m=1 = no merge (lossless, no compression).
//!   * precision (`HIPFIRE_KV_COLD_BITS`=4): 2 halves cold-code storage at ~+1.6%
//!     PPL — quant is cheap even at 2-bit (Sinkhorn variance-norm does the
//!     incoherence job a rotation would, so `rotate=false` and no ConQuR needed).
//!
//! Window/drain knobs: `HIPFIRE_KV_HOT_BUDGET`(256), `HIPFIRE_KV_MIGRATE_BATCH`(128),
//! `HIPFIRE_KV_IDLE_KEEP`(0 = full between-turns drain).
//!
//! Constraint: this is an inherently per-token-attention feature (it lives in
//! `kv_cache_attention_dispatch`); the batched session-batch prefill bypasses that
//! and is guarded against hier. Parity oracle: `ColdTier::two_tier_attend` + the
//! GPU kernels validated in rdna-compute/examples/parity_{attention_cold_slots,
//! flash_tier_merge,flash_partials_ml,two_tier_e2e,cold_4bit_read} and
//! hipfire-runtime/examples/parity_kv_hier.

use hipfire_kvquant::kv_compact::compact_cold_kv;
use hipfire_kvquant::kvarn::{kvarn_record_bytes_bits, pack_kvarn_tile_bits};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

const HD: usize = 256; // head_dim (kernel CHD)

/// Per-token importance proxy used to rank/weight cold compaction.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImportanceMode {
    Uniform,
    VNorm,
    KNorm,
    KvNorm,
    /// Real accumulated attention mass (CASK): Σ over q-heads & decode steps of the
    /// normalized attention weight each token received while in the hot window.
    Attn,
}

impl ImportanceMode {
    fn from_str(s: &str) -> Self {
        match s {
            "uniform" => ImportanceMode::Uniform,
            "knorm" => ImportanceMode::KNorm,
            "kvnorm" => ImportanceMode::KvNorm,
            "attn" => ImportanceMode::Attn,
            _ => ImportanceMode::VNorm, // default
        }
    }
}

/// One compacted cold segment, 4-bit-resident on GPU (per kv-head record tiles).
pub struct ColdSegmentGpu {
    pub k_recs: GpuTensor, // [n_kv_heads × rec_bytes] as f32-view (bytes/4 elems)
    pub v_recs: GpuTensor,
    pub n_valid: usize, // real slots attended
    pub n_slots: usize, // padded tile width (= slot_stride for the cold read)
    pub rec_bytes: usize,
    pub bits: usize, // quant bits per code (4 or 2) — for the dequant unpack
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
    /// Per-token importance signal for cold compaction (core selection + merge
    /// weighting). "uniform" (meaningless, average merge), "vnorm" (‖V_t‖),
    /// "knorm" (‖K_t‖), "kvnorm" (‖K_t‖·‖V_t‖). A real attention-mass signal
    /// (CASK) would need per-key accumulation in the hot read; norms are the
    /// zero-tracking proxy.
    pub importance_mode: ImportanceMode,
    /// Group merged (non-core) cold tokens by adjacent position (similar RoPE
    /// phase → less merge blur) rather than importance rank. Default on.
    pub position_local: bool,
    /// Max quant code for cold tiles (15=4-bit default, 3=2-bit probe).
    pub cold_qmax: f32,
    /// Bits per cold code (4 or 2) — drives real sub-nibble packing + dequant.
    pub cold_bits: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub hot_k: Vec<GpuTensor>, // [n_layers] slot-major [nkv × hot_budget × HD] f32
    pub hot_v: Vec<GpuTensor>,
    /// Per-layer per-hot-slot accumulated attention mass [hot_budget] f32 (CASK
    /// importance). Filled by the hot read's mass pass; only used when
    /// importance_mode == Attn. Zeroed at reset, shifted on migrate.
    pub attn_mass: Vec<GpuTensor>,
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
        let importance_mode = ImportanceMode::from_str(
            &std::env::var("HIPFIRE_KV_IMPORTANCE").unwrap_or_else(|_| "vnorm".to_string()),
        );
        let position_local = std::env::var("HIPFIRE_KV_POS_LOCAL").ok().as_deref() != Some("0");
        // Cold-tile quant precision probe: code max = 2^bits - 1 (4-bit=15 default,
        // 2-bit=3). Same nibble storage; this measures lower-precision quant QUALITY.
        let cold_bits: u32 = std::env::var("HIPFIRE_KV_COLD_BITS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4)
            .clamp(1, 4);
        let cold_qmax = ((1u32 << cold_bits) - 1) as f32;
        let mut hot_k = Vec::with_capacity(n_layers);
        let mut hot_v = Vec::with_capacity(n_layers);
        let mut attn_mass = Vec::with_capacity(n_layers);
        if enabled {
            for _ in 0..n_layers {
                hot_k.push(gpu.zeros(&[n_kv_heads * hot_budget * HD], DType::F32)?);
                hot_v.push(gpu.zeros(&[n_kv_heads * hot_budget * HD], DType::F32)?);
                attn_mass.push(gpu.zeros(&[hot_budget], DType::F32)?);
            }
        }
        Ok(Self {
            enabled,
            hot_budget,
            migrate_batch,
            core_frac,
            fold_m,
            importance_mode,
            position_local,
            cold_qmax,
            cold_bits: cold_bits as usize,
            n_heads,
            n_kv_heads,
            hot_k,
            hot_v,
            attn_mass,
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
    /// are kept (overwritten by `append_token`); cold segments are dropped; the
    /// attention-mass accumulators are zeroed. Call once at sequence start. NB:
    /// dropped segment GpuTensors are not pool-returned — a minor VRAM churn at the
    /// rare session boundary, not per-token.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        for c in self.hot_count.iter_mut() {
            *c = 0;
        }
        for m in self.migrated.iter_mut() {
            *m = 0;
        }
        for segs in self.cold.iter_mut() {
            segs.clear();
        }
        if self.importance_mode == ImportanceMode::Attn {
            for mass in self.attn_mass.iter() {
                gpu.fill_f32(mass, 0.0)?;
            }
        }
        Ok(())
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
            // Overflow fallback (on the critical path): evict the oldest batch. The
            // idle/between-turns path (idle_compact) keeps this from firing often.
            self.migrate_n(gpu, layer, self.migrate_batch)?;
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

    /// Migrate the oldest `n_req` hot tokens into ONE new cold segment, then shift
    /// the remaining hot tokens down to the front of the ring. Used both by the
    /// overflow fallback (n_req = migrate_batch) and idle_compact (n_req = drain).
    fn migrate_n(&mut self, gpu: &mut Gpu, layer: usize, n_req: usize) -> HipResult<()> {
        let mb = n_req.min(self.hot_count[layer]);
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
        // Per-token importance for core selection + merge weighting. Norm proxies
        // pull the merged K toward the dominant token's RoPE phase (less blur) and
        // keep high-norm tokens exact; Attn = real accumulated attention mass
        // (CASK); uniform = the old average merge.
        let mass = if self.importance_mode == ImportanceMode::Attn {
            gpu.download_f32(&self.attn_mass[layer])?
        } else {
            Vec::new()
        };
        let importance: Vec<f32> = (0..mb)
            .map(|t| {
                let base = t * kv_dim;
                let kn = || (0..kv_dim).map(|d| ck[base + d] * ck[base + d]).sum::<f32>().sqrt();
                let vn = || (0..kv_dim).map(|d| cv[base + d] * cv[base + d]).sum::<f32>().sqrt();
                match self.importance_mode {
                    ImportanceMode::Uniform => 1.0,
                    ImportanceMode::VNorm => vn(),
                    ImportanceMode::KNorm => kn(),
                    ImportanceMode::KvNorm => kn() * vn(),
                    // Small floor so an unattended token still sorts/weights sanely.
                    ImportanceMode::Attn => mass[t] + 1e-6,
                }
            })
            .collect();
        let cold = compact_cold_kv(
            &ck, &cv, mb, nkv, HD, &importance, self.core_frac, self.fold_m, false,
            self.position_local, self.cold_qmax,
        );
        let n_slots = cold.n_slots;
        let bits = self.cold_bits;
        let rec_bytes = kvarn_record_bytes_bits(HD, n_slots, bits);
        // rec_bytes must be a multiple of 4 to upload as an f32-view buffer; pad up.
        let rec_words = rec_bytes.div_ceil(4);
        let padded = rec_words * 4;
        let mut krecs = vec![0u8; nkv * padded];
        let mut vrecs = vec![0u8; nkv * padded];
        for h in 0..nkv {
            let kp = pack_kvarn_tile_bits(&cold.k_tiles[h], bits);
            let vp = pack_kvarn_tile_bits(&cold.v_tiles[h], bits);
            krecs[h * padded..h * padded + kp.len()].copy_from_slice(&kp);
            vrecs[h * padded..h * padded + vp.len()].copy_from_slice(&vp);
        }
        let k_recs = gpu.upload_raw(&krecs, &[nkv * rec_words])?;
        let v_recs = gpu.upload_raw(&vrecs, &[nkv * rec_words])?;
        self.cold[layer].push(ColdSegmentGpu {
            k_recs,
            v_recs,
            n_valid: cold.n_valid,
            n_slots,
            rec_bytes: padded,
            bits,
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
        // Mirror the shift for the attention-mass ring (slot s holds token s's mass),
        // then zero the vacated tail [rem, hot_budget) so reused slots start at 0.
        if self.importance_mode == ImportanceMode::Attn {
            if rem > 0 {
                gpu.memcpy_dtod_at_auto(
                    &self.attn_mass[layer].buf,
                    0,
                    &self.attn_mass[layer].buf,
                    mb * 4,
                    rem * 4,
                )?;
            }
            // Zero the vacated tail [rem, hot_budget) so reused slots start at 0
            // (the shift above already moved the surviving prefix down).
            let tail = hb - rem;
            if tail > 0 {
                let tail_view = self.attn_mass[layer].sub_offset(rem, tail);
                gpu.fill_f32(&tail_view, 0.0)?;
            }
        }
        self.hot_count[layer] = rem;
        Ok(())
    }

    /// Deferred between-turns compaction (the "deferred-hierarchical" thesis). Run
    /// in the idle gap after a turn ends, off the latency-critical path: drain each
    /// layer's hot ring down to `keep_recent` tokens, folding everything older into
    /// ONE cold segment per layer (big tile → better merge + amortized scale
    /// overhead). The next turn then starts with a near-empty hot ring but the full
    /// history present, compressed, in cold. No-op when a layer is already at/below
    /// `keep_recent`. Heavy compaction is justified here precisely because the user
    /// isn't waiting (single-user chat). Safe to call repeatedly.
    pub fn idle_compact(&mut self, gpu: &mut Gpu, keep_recent: usize) -> HipResult<()> {
        if !self.enabled {
            return Ok(());
        }
        let n_layers = self.hot_count.len();
        for layer in 0..n_layers {
            let hc = self.hot_count[layer];
            if hc > keep_recent {
                self.migrate_n(gpu, layer, hc - keep_recent)?;
            }
        }
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
        // hot_budget so the live count reads from a fixed-width ring. When using
        // attention-mass importance, accumulate this query's per-token weight.
        let mass = if self.importance_mode == ImportanceMode::Attn {
            Some(&self.attn_mass[layer])
        } else {
            None
        };
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
            mass,
        )?;

        // Fold each cold segment: dequant 4-bit → f16, channel-major attend, merge.
        for seg in &self.cold[layer] {
            gpu.kvarn_dequant_tile(&seg.k_recs, &scr.deq_k, nkv, HD, seg.n_slots, seg.rec_bytes, seg.bits)?;
            gpu.kvarn_dequant_tile(&seg.v_recs, &scr.deq_v, nkv, HD, seg.n_slots, seg.rec_bytes, seg.bits)?;
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
                None, // cold tier: no mass accumulation
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
