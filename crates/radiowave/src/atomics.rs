// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Per-architecture atomic capability matrix and fail-closed recipe emission gate.
//!
//! Hardware support is taken from the AMD GPU atomics operation support docs
//! (ROCm 7.14), corrected by header-proven overload presence in
//! `hip/amd_detail/amd_hip_atomic.h`, `amd_hip_fp16.h`, and `amd_hip_bf16.h`.
//! Header declaration alone is not HW support: double atomics are declared
//! unconditionally but are unsupported on RDNA (gfx10/11/12).
//!
//! Proven header surface (7.14):
//! - `atomicAdd`: int32/uint32, uint64, float, double, `__half`, `__half2`,
//!   `__hip_bfloat16`, `__hip_bfloat162`
//! - `atomicMin`/`atomicMax`: int32/64, uint32/64, float, double — **not**
//!   half/bf16 (no overloads)
//! - Sub-word: only `atomicCAS(unsigned short)` — no int8/int16 RMW
//! - No fp8 atomic overloads

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Architectures covered by the harness and RDNA4 dual-ISA targets.
///
/// Distinct from [`crate::arch::ArchProfile`], which is an exact certified
/// profile for occupancy geometry. Atomic capability is family-level with a
/// few named parts used by tests and recipes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtomicArch {
    Gfx1010,
    Gfx1030,
    Gfx1100,
    Gfx1151,
    Gfx1200,
    Gfx1201,
}

impl AtomicArch {
    /// Parse an exact offload-arch name (no feature suffixes).
    pub fn from_arch(arch: &str) -> Option<Self> {
        match arch {
            "gfx1010" => Some(Self::Gfx1010),
            "gfx1030" => Some(Self::Gfx1030),
            "gfx1100" => Some(Self::Gfx1100),
            "gfx1151" => Some(Self::Gfx1151),
            "gfx1200" => Some(Self::Gfx1200),
            "gfx1201" => Some(Self::Gfx1201),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gfx1010 => "gfx1010",
            Self::Gfx1030 => "gfx1030",
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1151 => "gfx1151",
            Self::Gfx1200 => "gfx1200",
            Self::Gfx1201 => "gfx1201",
        }
    }

    pub const fn family(self) -> AtomicArchFamily {
        match self {
            Self::Gfx1010 | Self::Gfx1030 => AtomicArchFamily::Rdna2,
            Self::Gfx1100 | Self::Gfx1151 => AtomicArchFamily::Rdna3,
            Self::Gfx1200 | Self::Gfx1201 => AtomicArchFamily::Rdna4,
        }
    }
}

/// RDNA generation for matrix lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtomicArchFamily {
    /// gfx10xx — RDNA2
    Rdna2,
    /// gfx11xx — RDNA3
    Rdna3,
    /// gfx12xx — RDNA4
    Rdna4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtomicOp {
    Add,
    Min,
    Max,
    Exchange,
    Cas,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtomicType {
    Int8,
    Int16,
    /// Signed/unsigned 32-bit integer family (header: `int` / `unsigned int`).
    Int32,
    /// Signed 64-bit (`long long`). Min/Max have header overloads; Add/Exchange
    /// are CAS-loop only (no signed `atomicAdd`/`atomicExch` overload).
    Int64,
    /// Unsigned 64-bit (`unsigned long` / `unsigned long long`) — native
    /// Add/Exchange/CAS/Min/Max overloads in `amd_hip_atomic.h`.
    UInt64,
    Float32,
    Float64,
    /// Packed `2×fp16` (`__half2`).
    PackedF16x2,
    /// Packed `2×bf16` (`__hip_bfloat162`).
    PackedBF16x2,
    /// OCP FP8 E4M3 — no atomic overloads.
    Fp8E4m3,
    /// OCP FP8 E5m2 — no atomic overloads.
    Fp8E5m2,
    /// Scalar `__half` (not packed).
    Float16,
    /// Scalar `__hip_bfloat16` (not packed).
    BFloat16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Agent/device scope (`__HIP_MEMORY_SCOPE_AGENT`).
    Device,
    /// System scope (`__HIP_MEMORY_SCOPE_SYSTEM`) — may leave L2 for fabric.
    System,
}

/// Memory coherence granularity (gfx12 model).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Granularity {
    /// Coarse-grained (typically cached; device-local).
    Coarse,
    /// Fine-grained. On gfx12: fine DEVICE is write-uncached (atomics→fabric);
    /// fine SYSTEM is cacheable (device-scope in L2; system-scope→fabric).
    Fine,
}

/// Where the atomic address lives.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtomicMemory {
    /// Device HBM / local VRAM.
    Device,
    /// Host memory reached over PCIe — unsupported HW atomics become
    /// load-op-store; system scope may be downgraded toward device.
    Host,
}

/// Result of looking up a single matrix cell.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Support {
    /// Hardware atomic RMW (or true HW CAS) with no extra compiler flags.
    Native,
    /// Hardware path exists only when recipe emission supplies a specific
    /// compiler lowering (define / flag). `lowering` is the flag text, e.g.
    /// `-munsafe-fp-atomics`.
    NativeRequires { lowering: String },
    /// Software CAS loop (header provides safe path; high contention cost).
    EmulatedCas,
    /// No usable path for recipe emission.
    Unsupported,
    /// Hardware path exists only after relaxing scope/memory semantics.
    Downgraded(String),
}

impl Support {
    pub const fn is_native(&self) -> bool {
        matches!(self, Self::Native)
    }

    /// True when the cell is HW-native, with or without an extra lowering flag.
    pub const fn is_native_family(&self) -> bool {
        matches!(self, Self::Native | Self::NativeRequires { .. })
    }
}

/// One atomic use a recipe wants to emit.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AtomicRequirement {
    pub op: AtomicOp,
    pub ty: AtomicType,
    pub arch: AtomicArch,
    pub scope: Scope,
    pub granularity: Granularity,
    #[serde(default = "default_memory_device")]
    pub memory: AtomicMemory,
}

fn default_memory_device() -> AtomicMemory {
    AtomicMemory::Device
}

impl AtomicRequirement {
    pub const fn new(
        op: AtomicOp,
        ty: AtomicType,
        arch: AtomicArch,
        scope: Scope,
        granularity: Granularity,
    ) -> Self {
        Self {
            op,
            ty,
            arch,
            scope,
            granularity,
            memory: AtomicMemory::Device,
        }
    }

    pub const fn with_memory(mut self, memory: AtomicMemory) -> Self {
        self.memory = memory;
        self
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AtomicError {
    #[error(
        "atomic {op:?} on {ty:?} is unsupported for {arch} (scope={scope:?}, gran={granularity:?}): {detail}"
    )]
    Unsupported {
        op: AtomicOp,
        ty: AtomicType,
        arch: String,
        scope: Scope,
        granularity: Granularity,
        detail: String,
    },
    #[error(
        "atomic {op:?} on {ty:?} for {arch} would use CAS emulation (rejected; high negative performance impact under contention): {detail}"
    )]
    Emulated {
        op: AtomicOp,
        ty: AtomicType,
        arch: String,
        detail: String,
    },
    #[error(
        "atomic {op:?} on {ty:?} for {arch} is only available with downgraded semantics (rejected fail-closed): {reason}"
    )]
    Downgraded {
        op: AtomicOp,
        ty: AtomicType,
        arch: String,
        reason: String,
    },
}

/// Fail-closed gate for recipe emission.
///
/// On success returns the optional compiler lowering required for a true HW
/// path (e.g. [`Some("-munsafe-fp-atomics")`] from [`Support::NativeRequires`]).
/// Callers **must** thread a `Some(flag)` into the compile command; discarding
/// it would silently fall back to the safe CAS-loop path.
///
/// [`Support::Native`] yields `Ok(None)`. Emulated, unsupported, and downgraded
/// cells are rejected.
pub fn check_atomic_requirement(
    requirement: &AtomicRequirement,
) -> Result<Option<String>, AtomicError> {
    match native_support_with_memory(
        requirement.op,
        requirement.ty,
        requirement.arch,
        requirement.scope,
        requirement.granularity,
        requirement.memory,
    ) {
        Support::Native => Ok(None),
        Support::NativeRequires { lowering } => Ok(Some(lowering)),
        Support::EmulatedCas => Err(AtomicError::Emulated {
            op: requirement.op,
            ty: requirement.ty,
            arch: requirement.arch.as_str().to_owned(),
            detail: "software CAS loop".to_owned(),
        }),
        Support::Unsupported => Err(AtomicError::Unsupported {
            op: requirement.op,
            ty: requirement.ty,
            arch: requirement.arch.as_str().to_owned(),
            scope: requirement.scope,
            granularity: requirement.granularity,
            detail: "no native hardware atomic".to_owned(),
        }),
        Support::Downgraded(reason) => Err(AtomicError::Downgraded {
            op: requirement.op,
            ty: requirement.ty,
            arch: requirement.arch.as_str().to_owned(),
            reason,
        }),
    }
}

/// Query HW/native support for one matrix cell.
///
/// The five-arg form defaults memory to device HBM.
pub fn native_support(
    op: AtomicOp,
    ty: AtomicType,
    arch: AtomicArch,
    scope: Scope,
    granularity: Granularity,
) -> Support {
    native_support_with_memory(op, ty, arch, scope, granularity, AtomicMemory::Device)
}

/// Full matrix lookup including host-memory (PCIe) semantics.
pub fn native_support_with_memory(
    op: AtomicOp,
    ty: AtomicType,
    arch: AtomicArch,
    scope: Scope,
    granularity: Granularity,
    memory: AtomicMemory,
) -> Support {
    // Host memory over PCIe: atomics the device cannot issue as true atomics
    // become load-op-store (CPU observes non-atomic; waves stall). System scope
    // is further degraded toward device visibility.
    if memory == AtomicMemory::Host {
        return host_memory_support(op, ty, arch, scope, granularity);
    }

    let base = match ty {
        AtomicType::Int8 => Support::Unsupported,
        AtomicType::Int16 => int16_support(op),
        AtomicType::Fp8E4m3 | AtomicType::Fp8E5m2 => Support::Unsupported,
        AtomicType::Float64 => float64_support(op, arch, scope),
        AtomicType::PackedF16x2 => packed_f16x2_support(op, arch, scope),
        AtomicType::PackedBF16x2 => packed_bf16x2_support(op, arch, scope),
        AtomicType::Float16 => scalar_f16_support(op, arch, scope),
        AtomicType::BFloat16 => scalar_bf16_support(op, arch, scope),
        AtomicType::Int32 => int32_support(op, arch, scope),
        AtomicType::Int64 => int64_signed_support(op, arch, scope),
        AtomicType::UInt64 => uint64_support(op, arch, scope),
        AtomicType::Float32 => float32_support(op, arch, scope),
    };

    apply_device_granularity(base, ty, arch, scope, granularity)
}

/// gfx12 memory model: coarse-grained device memory keeps system-scope
/// atomics from reaching the fabric as true system atomics for device-local
/// integer/float types — report Downgraded rather than Native.
fn apply_device_granularity(
    base: Support,
    ty: AtomicType,
    arch: AtomicArch,
    scope: Scope,
    granularity: Granularity,
) -> Support {
    // Only RDNA4 documents the coarse/fine system-scope distinction we encode.
    if arch.family() != AtomicArchFamily::Rdna4 {
        return base;
    }
    if scope != Scope::System || granularity != Granularity::Coarse {
        return base;
    }
    // Packed/scalar fp16/bf16 already carry NativeRequires / EmulatedCas —
    // still mark the coarse+system downgrade when the base looked native-ish.
    let device_local = matches!(
        ty,
        AtomicType::Int8
            | AtomicType::Int16
            | AtomicType::Int32
            | AtomicType::Int64
            | AtomicType::UInt64
            | AtomicType::Float32
            | AtomicType::Float64
            | AtomicType::Float16
            | AtomicType::BFloat16
            | AtomicType::PackedF16x2
            | AtomicType::PackedBF16x2
            | AtomicType::Fp8E4m3
            | AtomicType::Fp8E5m2
    );
    if !device_local {
        return base;
    }
    match base {
        Support::Native | Support::NativeRequires { .. } => Support::Downgraded(
            "gfx12 coarse-grained device memory: system-scope atomics do not \
             retain full system visibility (would downgrade toward device-scope \
             / L2 semantics)"
                .to_owned(),
        ),
        Support::Downgraded(reason) => Support::Downgraded(format!(
            "{reason}; coarse+system further restricts system visibility on gfx12"
        )),
        other => other,
    }
}

fn int16_support(op: AtomicOp) -> Support {
    // Header surface is only `atomicCAS(unsigned short int*, ...)` — signed
    // `short` / Int16 has no native CAS overload. Recipes that need 16-bit
    // CAS must widen to the unsigned short path (emulation / bitcast loop).
    match op {
        AtomicOp::Cas => Support::EmulatedCas,
        _ => Support::Unsupported,
    }
}

fn int32_support(op: AtomicOp, _arch: AtomicArch, _scope: Scope) -> Support {
    // int32/uint32 Add/Min/Max/Exch/CAS native on all covered RDNA families.
    match op {
        AtomicOp::Add
        | AtomicOp::Min
        | AtomicOp::Max
        | AtomicOp::Exchange
        | AtomicOp::Cas => Support::Native,
    }
}

fn uint64_support(op: AtomicOp, _arch: AtomicArch, _scope: Scope) -> Support {
    // Header overloads: atomicAdd/Exch/CAS/Min/Max on unsigned long long.
    match op {
        AtomicOp::Add
        | AtomicOp::Min
        | AtomicOp::Max
        | AtomicOp::Exchange
        | AtomicOp::Cas => Support::Native,
    }
}

fn int64_signed_support(op: AtomicOp, _arch: AtomicArch, _scope: Scope) -> Support {
    // Signed long long: Min/Max have overloads; Add/Exchange do not (CAS-loop).
    // CAS is available via the unsigned long long path with bitcast.
    match op {
        AtomicOp::Min | AtomicOp::Max | AtomicOp::Cas => Support::Native,
        AtomicOp::Add | AtomicOp::Exchange => Support::EmulatedCas,
    }
}

fn float32_support(op: AtomicOp, arch: AtomicArch, scope: Scope) -> Support {
    match op {
        AtomicOp::Cas | AtomicOp::Exchange => Support::Native,
        AtomicOp::Add => float32_add_minmax(arch, scope, "add"),
        AtomicOp::Min => float32_add_minmax(arch, scope, "min"),
        AtomicOp::Max => float32_add_minmax(arch, scope, "max"),
    }
}

/// fp32 add/min/max: L2 vs fabric differs by generation.
fn float32_add_minmax(arch: AtomicArch, scope: Scope, op_name: &str) -> Support {
    match arch.family() {
        // RDNA4: L2 AND Infinity Fabric — device and system native.
        AtomicArchFamily::Rdna4 => Support::Native,
        // RDNA3: L2 ONLY, not fabric. System scope leaves L2 → fabric.
        AtomicArchFamily::Rdna3 => match scope {
            Scope::Device => Support::Native,
            Scope::System => Support::Downgraded(format!(
                "gfx11 float32 atomic{op_name} is L2-only; system scope requires fabric \
                 (unsupported) — would downgrade toward device-scope/L2 semantics"
            )),
        },
        // RDNA2: add native; min/max native at L2, unsupported by fabric.
        AtomicArchFamily::Rdna2 => match (op_name, scope) {
            ("add", _) => Support::Native,
            (_, Scope::Device) => Support::Native,
            (_, Scope::System) => Support::Downgraded(format!(
                "gfx10 float32 atomic{op_name} unsupported by fabric; system scope downgrades"
            )),
        },
    }
}

fn float64_support(op: AtomicOp, arch: AtomicArch, _scope: Scope) -> Support {
    // Docs: float64 atomics unsupported on all RDNA (gfx10/11/12). Headers still
    // declare double atomicAdd/Min/Max — fail-closed on HW matrix, not decls.
    let _ = op;
    match arch.family() {
        AtomicArchFamily::Rdna2 | AtomicArchFamily::Rdna3 | AtomicArchFamily::Rdna4 => {
            Support::Unsupported
        }
    }
}

/// HIP lowering flag required for true HW packed/scalar fp16/bf16 atomics
/// (`unsafeAtomicAdd` path / `-munsafe-fp-atomics`; see amd_hip_unsafe_atomics.h).
const UNSAFE_FP_ATOMICS_LOWERING: &str = "-munsafe-fp-atomics";

fn packed_f16x2_support(op: AtomicOp, arch: AtomicArch, _scope: Scope) -> Support {
    // No atomicMin/Max for half2 in headers (dead_end confirmed).
    // Safe `atomicAdd(__half2*)` is a u32 CAS loop; HW path is
    // `unsafeAtomicAdd` → flat_atomic_fadd_v2f16 under -munsafe-fp-atomics.
    match op {
        AtomicOp::Min | AtomicOp::Max => Support::Unsupported,
        AtomicOp::Exchange | AtomicOp::Cas => Support::EmulatedCas,
        AtomicOp::Add => match arch.family() {
            AtomicArchFamily::Rdna4 => Support::NativeRequires {
                lowering: UNSAFE_FP_ATOMICS_LOWERING.to_owned(),
            },
            AtomicArchFamily::Rdna2 | AtomicArchFamily::Rdna3 => Support::EmulatedCas,
        },
    }
}

fn packed_bf16x2_support(op: AtomicOp, arch: AtomicArch, _scope: Scope) -> Support {
    // No atomicMin/Max for bfloat162 in headers.
    match op {
        AtomicOp::Min | AtomicOp::Max => Support::Unsupported,
        AtomicOp::Exchange | AtomicOp::Cas => Support::EmulatedCas,
        AtomicOp::Add => match arch.family() {
            AtomicArchFamily::Rdna4 => Support::NativeRequires {
                lowering: UNSAFE_FP_ATOMICS_LOWERING.to_owned(),
            },
            AtomicArchFamily::Rdna2 | AtomicArchFamily::Rdna3 => Support::EmulatedCas,
        },
    }
}

fn scalar_f16_support(op: AtomicOp, arch: AtomicArch, _scope: Scope) -> Support {
    // Header has atomicAdd(__half*) via scoped fetch_add; no min/max.
    // HW-native flat path for true RMW still wants unsafe-fp-atomics on gfx12.
    match op {
        AtomicOp::Add => match arch.family() {
            AtomicArchFamily::Rdna4 => Support::NativeRequires {
                lowering: UNSAFE_FP_ATOMICS_LOWERING.to_owned(),
            },
            // Other RDNA: treat scoped fetch_add as Native (no gfx12 packed ISA).
            AtomicArchFamily::Rdna2 | AtomicArchFamily::Rdna3 => Support::Native,
        },
        AtomicOp::Min | AtomicOp::Max => Support::Unsupported,
        AtomicOp::Exchange | AtomicOp::Cas => Support::EmulatedCas,
    }
}

fn scalar_bf16_support(op: AtomicOp, arch: AtomicArch, _scope: Scope) -> Support {
    match op {
        AtomicOp::Add => match arch.family() {
            AtomicArchFamily::Rdna4 => Support::NativeRequires {
                lowering: UNSAFE_FP_ATOMICS_LOWERING.to_owned(),
            },
            AtomicArchFamily::Rdna2 | AtomicArchFamily::Rdna3 => Support::Native,
        },
        AtomicOp::Min | AtomicOp::Max => Support::Unsupported,
        AtomicOp::Exchange | AtomicOp::Cas => Support::EmulatedCas,
    }
}

fn host_memory_support(
    op: AtomicOp,
    ty: AtomicType,
    arch: AtomicArch,
    scope: Scope,
    granularity: Granularity,
) -> Support {
    // Classify as if the address were device-local first.
    let device = native_support_with_memory(
        op,
        ty,
        arch,
        scope,
        granularity,
        AtomicMemory::Device,
    );
    match device {
        Support::Unsupported => Support::Unsupported,
        other => {
            // ROCm 7.14 gfx12 tables: host coarse cells and host fine+device
            // cells retain native HW atomics for types that are native on
            // device. Other host paths (pre-gfx12, or fine+system) degrade
            // over PCIe to load-op-store / scope downgrade.
            if arch.family() == AtomicArchFamily::Rdna4 {
                match (granularity, scope) {
                    (Granularity::Coarse, _) | (Granularity::Fine, Scope::Device) => other,
                    (Granularity::Fine, Scope::System) => Support::Downgraded(
                        "host fine-grained system-scope atomic over PCIe: \
                         system visibility is not guaranteed (would downgrade \
                         toward device-scope / load-op-store)"
                            .to_owned(),
                    ),
                }
            } else {
                match other {
                    Support::EmulatedCas => Support::Downgraded(
                        "host-memory atomic over PCIe: CAS emulation becomes \
                         load-op-store (waves stall; CPU sees non-atomic updates)"
                            .to_owned(),
                    ),
                    Support::Downgraded(reason) => Support::Downgraded(format!(
                        "host-memory PCIe path compounds prior downgrade: {reason}"
                    )),
                    Support::Native | Support::NativeRequires { .. } => match scope {
                        Scope::System => Support::Downgraded(
                            "host-memory atomic over PCIe: system scope downgrades \
                             to device-visible load-op-store (CPU observes non-atomic)"
                                .to_owned(),
                        ),
                        Scope::Device => Support::Downgraded(
                            "host-memory atomic over PCIe: device-scope ops degrade \
                             to load-op-store against host pages (waves stall)"
                                .to_owned(),
                        ),
                    },
                    Support::Unsupported => Support::Unsupported,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdna4_packed_bf16_add_requires_unsafe_fp_atomics() {
        for arch in [AtomicArch::Gfx1200, AtomicArch::Gfx1201] {
            for ty in [AtomicType::PackedBF16x2, AtomicType::PackedF16x2] {
                let support = native_support(
                    AtomicOp::Add,
                    ty,
                    arch,
                    Scope::Device,
                    Granularity::Fine,
                );
                assert!(
                    matches!(
                        &support,
                        Support::NativeRequires { lowering }
                            if lowering == "-munsafe-fp-atomics"
                    ),
                    "{arch:?} {ty:?} => {support:?}",
                );
            }
        }
    }

    #[test]
    fn gfx1201_int32_add_coarse_system_downgrades() {
        // Regression: coarse + system must not report Native for device-local
        // types (system-scope atomics on coarse memory downgrade).
        let support = native_support(
            AtomicOp::Add,
            AtomicType::Int32,
            AtomicArch::Gfx1201,
            Scope::System,
            Granularity::Coarse,
        );
        assert!(
            matches!(&support, Support::Downgraded(r) if r.contains("coarse")),
            "expected Downgraded for coarse+system Int32 Add, got {support:?}",
        );
        // Fine + system remains Native on RDNA4.
        assert_eq!(
            native_support(
                AtomicOp::Add,
                AtomicType::Int32,
                AtomicArch::Gfx1201,
                Scope::System,
                Granularity::Fine,
            ),
            Support::Native,
        );
        // Coarse + device remains Native.
        assert_eq!(
            native_support(
                AtomicOp::Add,
                AtomicType::Int32,
                AtomicArch::Gfx1201,
                Scope::Device,
                Granularity::Coarse,
            ),
            Support::Native,
        );
    }

    #[test]
    fn fp64_unsupported_on_gfx1201_and_all_rdna() {
        for arch in [
            AtomicArch::Gfx1010,
            AtomicArch::Gfx1030,
            AtomicArch::Gfx1100,
            AtomicArch::Gfx1151,
            AtomicArch::Gfx1200,
            AtomicArch::Gfx1201,
        ] {
            for op in [
                AtomicOp::Add,
                AtomicOp::Min,
                AtomicOp::Max,
                AtomicOp::Exchange,
                AtomicOp::Cas,
            ] {
                // Exchange/CAS on f64 may be bitcast CAS in ISA; matrix marks
                // fp64 atomics unsupported on RDNA for RMW. Exchange/CAS on
                // integer-sized slots still go through float64_support → Unsupported
                // for the Float64 type cell (recipes must not claim f64 RMW).
                let support = native_support(
                    op,
                    AtomicType::Float64,
                    arch,
                    Scope::Device,
                    Granularity::Fine,
                );
                assert_eq!(
                    support,
                    Support::Unsupported,
                    "{arch:?} {op:?}",
                );
            }
        }
    }

    #[test]
    fn int8_unsupported_all_ops_all_archs() {
        for arch in [
            AtomicArch::Gfx1010,
            AtomicArch::Gfx1100,
            AtomicArch::Gfx1201,
        ] {
            for op in [
                AtomicOp::Add,
                AtomicOp::Min,
                AtomicOp::Max,
                AtomicOp::Exchange,
                AtomicOp::Cas,
            ] {
                assert_eq!(
                    native_support(
                        op,
                        AtomicType::Int8,
                        arch,
                        Scope::Device,
                        Granularity::Fine,
                    ),
                    Support::Unsupported,
                    "{arch:?} {op:?}",
                );
            }
        }
    }

    #[test]
    fn gfx11_system_scope_fp32_add_minmax_downgrades() {
        for arch in [AtomicArch::Gfx1100, AtomicArch::Gfx1151] {
            for op in [AtomicOp::Add, AtomicOp::Min, AtomicOp::Max] {
                let support = native_support(
                    op,
                    AtomicType::Float32,
                    arch,
                    Scope::System,
                    Granularity::Fine,
                );
                assert!(
                    matches!(support, Support::Downgraded(_)),
                    "{arch:?} {op:?} => {support:?}",
                );
                // Device scope remains native (L2).
                assert_eq!(
                    native_support(
                        op,
                        AtomicType::Float32,
                        arch,
                        Scope::Device,
                        Granularity::Fine,
                    ),
                    Support::Native,
                    "{arch:?} {op:?}",
                );
            }
        }
    }

    #[test]
    fn int16_cas_is_emulated_not_native() {
        // Regression: headers allow CAS only for *unsigned* short. Signed
        // Int16 must not be reported as Native.
        for arch in [
            AtomicArch::Gfx1010,
            AtomicArch::Gfx1100,
            AtomicArch::Gfx1201,
        ] {
            assert_eq!(
                native_support(
                    AtomicOp::Cas,
                    AtomicType::Int16,
                    arch,
                    Scope::Device,
                    Granularity::Fine,
                ),
                Support::EmulatedCas,
                "{arch:?}: signed Int16 CAS has no header overload",
            );
            // Non-CAS int16 remains unsupported (no header RMW).
            assert_eq!(
                native_support(
                    AtomicOp::Add,
                    AtomicType::Int16,
                    arch,
                    Scope::Device,
                    Granularity::Fine,
                ),
                Support::Unsupported,
                "{arch:?}",
            );
        }
    }

    #[test]
    fn gfx12_fabric_supports_fp32_add_minmax_at_system_scope() {
        for arch in [AtomicArch::Gfx1200, AtomicArch::Gfx1201] {
            for op in [AtomicOp::Add, AtomicOp::Min, AtomicOp::Max] {
                assert_eq!(
                    native_support(
                        op,
                        AtomicType::Float32,
                        arch,
                        Scope::System,
                        Granularity::Fine,
                    ),
                    Support::Native,
                    "{arch:?} {op:?}",
                );
            }
        }
    }

    #[test]
    fn half_bf16_min_max_unsupported_header_proven() {
        for arch in [AtomicArch::Gfx1201, AtomicArch::Gfx1100] {
            for ty in [
                AtomicType::Float16,
                AtomicType::BFloat16,
                AtomicType::PackedF16x2,
                AtomicType::PackedBF16x2,
            ] {
                for op in [AtomicOp::Min, AtomicOp::Max] {
                    assert_eq!(
                        native_support(op, ty, arch, Scope::Device, Granularity::Fine),
                        Support::Unsupported,
                        "{arch:?} {ty:?} {op:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn packed_add_emulated_on_gfx11() {
        assert_eq!(
            native_support(
                AtomicOp::Add,
                AtomicType::PackedF16x2,
                AtomicArch::Gfx1100,
                Scope::Device,
                Granularity::Fine,
            ),
            Support::EmulatedCas,
        );
        assert_eq!(
            native_support(
                AtomicOp::Add,
                AtomicType::PackedBF16x2,
                AtomicArch::Gfx1151,
                Scope::Device,
                Granularity::Fine,
            ),
            Support::EmulatedCas,
        );
    }

    #[test]
    fn fp8_atomics_unsupported() {
        for ty in [AtomicType::Fp8E4m3, AtomicType::Fp8E5m2] {
            assert_eq!(
                native_support(
                    AtomicOp::Add,
                    ty,
                    AtomicArch::Gfx1201,
                    Scope::Device,
                    Granularity::Fine,
                ),
                Support::Unsupported,
            );
        }
    }

    #[test]
    fn host_memory_pcie_downgrades_native_ops() {
        let support = native_support_with_memory(
            AtomicOp::Add,
            AtomicType::Int32,
            AtomicArch::Gfx1201,
            Scope::System,
            Granularity::Fine,
            AtomicMemory::Host,
        );
        assert!(
            matches!(&support, Support::Downgraded(r) if r.contains("PCIe")),
            "{support:?}",
        );
    }

    #[test]
    fn check_atomic_requirement_fail_closed() {
        // NativeRequires (packed bf16 add on gfx12) passes and yields the flag.
        let ok = AtomicRequirement::new(
            AtomicOp::Add,
            AtomicType::PackedBF16x2,
            AtomicArch::Gfx1201,
            Scope::Device,
            Granularity::Fine,
        );
        assert_eq!(
            check_atomic_requirement(&ok).expect("packed bf16 should pass"),
            Some("-munsafe-fp-atomics".to_owned()),
        );

        // Packed f16x2 add on gfx12 likewise propagates the lowering.
        let f16 = AtomicRequirement::new(
            AtomicOp::Add,
            AtomicType::PackedF16x2,
            AtomicArch::Gfx1200,
            Scope::Device,
            Granularity::Fine,
        );
        assert_eq!(
            check_atomic_requirement(&f16).expect("packed f16 should pass"),
            Some("-munsafe-fp-atomics".to_owned()),
        );

        // Plain Native (int32 add) yields Ok(None) — no extra flag.
        let plain = AtomicRequirement::new(
            AtomicOp::Add,
            AtomicType::Int32,
            AtomicArch::Gfx1201,
            Scope::Device,
            Granularity::Fine,
        );
        assert_eq!(
            check_atomic_requirement(&plain).expect("int32 native"),
            None,
        );

        // Unsupported fails.
        let bad_f64 = AtomicRequirement::new(
            AtomicOp::Add,
            AtomicType::Float64,
            AtomicArch::Gfx1201,
            Scope::Device,
            Granularity::Fine,
        );
        assert!(matches!(
            check_atomic_requirement(&bad_f64),
            Err(AtomicError::Unsupported { .. })
        ));

        // Emulated CAS rejected.
        let emu = AtomicRequirement::new(
            AtomicOp::Add,
            AtomicType::PackedF16x2,
            AtomicArch::Gfx1100,
            Scope::Device,
            Granularity::Fine,
        );
        assert!(matches!(
            check_atomic_requirement(&emu),
            Err(AtomicError::Emulated { .. })
        ));

        // Downgraded rejected.
        let down = AtomicRequirement::new(
            AtomicOp::Add,
            AtomicType::Float32,
            AtomicArch::Gfx1100,
            Scope::System,
            Granularity::Fine,
        );
        assert!(matches!(
            check_atomic_requirement(&down),
            Err(AtomicError::Downgraded { .. })
        ));

        // Signed Int16 CAS is emulated (not native ushort) → rejected.
        let cas16 = AtomicRequirement::new(
            AtomicOp::Cas,
            AtomicType::Int16,
            AtomicArch::Gfx1201,
            Scope::Device,
            Granularity::Coarse,
        );
        assert!(matches!(
            check_atomic_requirement(&cas16),
            Err(AtomicError::Emulated { .. })
        ));
    }

    #[test]
    fn uint64_native_int64_add_emulated() {
        for arch in [
            AtomicArch::Gfx1010,
            AtomicArch::Gfx1100,
            AtomicArch::Gfx1201,
        ] {
            for op in [
                AtomicOp::Add,
                AtomicOp::Min,
                AtomicOp::Max,
                AtomicOp::Exchange,
                AtomicOp::Cas,
            ] {
                assert_eq!(
                    native_support(op, AtomicType::UInt64, arch, Scope::Device, Granularity::Fine),
                    Support::Native,
                    "{arch:?} UInt64 {op:?}",
                );
            }
            // Signed Int64: Min/Max/CAS native; Add/Exchange CAS-loop.
            assert_eq!(
                native_support(
                    AtomicOp::Min,
                    AtomicType::Int64,
                    arch,
                    Scope::Device,
                    Granularity::Fine,
                ),
                Support::Native,
                "{arch:?}",
            );
            assert_eq!(
                native_support(
                    AtomicOp::Max,
                    AtomicType::Int64,
                    arch,
                    Scope::Device,
                    Granularity::Fine,
                ),
                Support::Native,
                "{arch:?}",
            );
            assert_eq!(
                native_support(
                    AtomicOp::Cas,
                    AtomicType::Int64,
                    arch,
                    Scope::Device,
                    Granularity::Fine,
                ),
                Support::Native,
                "{arch:?}",
            );
            assert_eq!(
                native_support(
                    AtomicOp::Add,
                    AtomicType::Int64,
                    arch,
                    Scope::Device,
                    Granularity::Fine,
                ),
                Support::EmulatedCas,
                "{arch:?}",
            );
            assert_eq!(
                native_support(
                    AtomicOp::Exchange,
                    AtomicType::Int64,
                    arch,
                    Scope::Device,
                    Granularity::Fine,
                ),
                Support::EmulatedCas,
                "{arch:?}",
            );
            // Int32 remains fully native.
            for op in [
                AtomicOp::Add,
                AtomicOp::Min,
                AtomicOp::Max,
                AtomicOp::Exchange,
                AtomicOp::Cas,
            ] {
                assert_eq!(
                    native_support(op, AtomicType::Int32, arch, Scope::Device, Granularity::Fine),
                    Support::Native,
                    "{arch:?} Int32 {op:?}",
                );
            }
        }
    }

    #[test]
    fn atomic_arch_parse_exact() {
        assert_eq!(AtomicArch::from_arch("gfx1201"), Some(AtomicArch::Gfx1201));
        assert_eq!(AtomicArch::from_arch("gfx1151"), Some(AtomicArch::Gfx1151));
        assert_eq!(AtomicArch::from_arch("gfx12"), None);
        assert_eq!(AtomicArch::from_arch("gfx1201:xnack-"), None);
        assert_eq!(AtomicArch::Gfx1201.family(), AtomicArchFamily::Rdna4);
        assert_eq!(AtomicArch::Gfx1151.family(), AtomicArchFamily::Rdna3);
        assert_eq!(AtomicArch::Gfx1010.family(), AtomicArchFamily::Rdna2);
    }

    #[test]
    fn gfx10_system_scope_fp32_minmax_downgrades_add_native() {
        assert_eq!(
            native_support(
                AtomicOp::Add,
                AtomicType::Float32,
                AtomicArch::Gfx1030,
                Scope::System,
                Granularity::Fine,
            ),
            Support::Native,
        );
        let min = native_support(
            AtomicOp::Min,
            AtomicType::Float32,
            AtomicArch::Gfx1030,
            Scope::System,
            Granularity::Fine,
        );
        assert!(matches!(min, Support::Downgraded(_)), "{min:?}");
    }
}
