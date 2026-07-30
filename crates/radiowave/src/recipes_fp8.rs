// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Experimental OCP FP8 recipe types for RDNA3/RDNA4 (Wave 2 — types only).
//!
//! Verified against ROCm 7.14 headers:
//! - `hip/hip_fp8.h` → `hip/amd_detail/amd_hip_fp8.h`
//! - OCP `__hip_fp8_e4m3` / `__hip_fp8_e5m2` (+ packed x2/x4) device-gated
//!   `__gfx1200__` / `__gfx1201__` via `HIP_FP8_TYPE_OCP` (no bare `__gfx12__`)
//! - convert: `__hip_cvt_float_to_fp8` / `__hip_cvt_float2_to_fp8x2`
//! - gfx12 native WMMA:
//!   `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
//! - gfx11 fallback: software OCP decode into FP16 register fragments followed
//!   by `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
//!
//! FNUZ (`__hip_fp8_*_fnuz`) is CDNA3/gfx942-only and is excluded here.
//! gfx11 does not expose native OCP FP8 device arithmetic; its recipes are
//! explicitly marked as software-decode lowerings rather than native FP8.
//! Catalog emission is Wave 3 — this module only exposes recipe types and
//! experimental gating.

use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

/// Compile-time experimental gate via Cargo feature `experimental-fp8`.
///
/// Wave-3 catalog wiring stays off until this feature is enabled **and**
/// [`set_experimental_fp8_enabled`] is set at runtime.
#[cfg(feature = "experimental-fp8")]
pub const EXPERIMENTAL_FP8: bool = true;
#[cfg(not(feature = "experimental-fp8"))]
pub const EXPERIMENTAL_FP8: bool = false;

static RUNTIME_ENABLE: AtomicBool = AtomicBool::new(false);

/// OCP FP8 encodings accepted by the recipe catalog.
///
/// Header evidence (`amd_hip_fp8.h`):
/// - `struct __hip_fp8_e4m3` with `__default_interpret = __HIP_E4M3`
/// - `struct __hip_fp8_e5m2` with `__default_interpret = __HIP_E5M2`
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Fp8Format {
    /// OCP E4M3 (`__hip_fp8_e4m3` / `rocwmma::float8_t`).
    E4M3Ocp,
    /// OCP E5M2 (`__hip_fp8_e5m2` / `rocwmma::bfloat8_t`).
    E5M2Ocp,
}

impl Fp8Format {
    pub const fn hip_type_name(self) -> &'static str {
        match self {
            Self::E4M3Ocp => "__hip_fp8_e4m3",
            Self::E5M2Ocp => "__hip_fp8_e5m2",
        }
    }

    pub const fn hip_interpretation(self) -> &'static str {
        match self {
            Self::E4M3Ocp => "__HIP_E4M3",
            Self::E5M2Ocp => "__HIP_E5M2",
        }
    }

    pub const fn rocwmma_type_name(self) -> &'static str {
        match self {
            Self::E4M3Ocp => "rocwmma::float8_t",
            Self::E5M2Ocp => "rocwmma::bfloat8_t",
        }
    }
}

/// How an FP8 recipe is lowered on the target architecture.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Fp8Lowering {
    /// Native OCP conversion/type support (gfx12).
    NativeConvert,
    /// Native OCP FP8 WMMA (gfx12).
    NativeWmma,
    /// Integer/bitwise OCP decode to FP16 (gfx11).
    SoftwareDecode,
    /// Software OCP decode to FP16 followed by gfx11 FP16 WMMA.
    SoftwareDecodeF16Wmma,
}

/// One experimental FP8 source-lowering recipe.
///
/// `source_variant` is a HIP source fragment template that exercises the
/// verified path. Catalog emission is intentionally absent until Wave 3.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fp8Recipe {
    pub format: Fp8Format,
    /// True only for a native gfx12 FP8 WMMA recipe.
    pub wmma: bool,
    pub source_variant: String,
}

/// An architecture-explicit recipe that may use native or software lowering.
///
/// This is separate from [`Fp8Recipe`] so its historical gfx12-native contract
/// remains intact for downstream callers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fp8LoweringRecipe {
    pub format: Fp8Format,
    pub lowering: Fp8Lowering,
    pub source_variant: String,
}

impl Fp8LoweringRecipe {
    pub const fn uses_wmma(&self) -> bool {
        matches!(
            self.lowering,
            Fp8Lowering::NativeWmma | Fp8Lowering::SoftwareDecodeF16Wmma
        )
    }
}

/// Runtime override for the experimental catalog (still requires the
/// `experimental-fp8` Cargo feature and arch gate).
pub fn set_experimental_fp8_enabled(enabled: bool) {
    RUNTIME_ENABLE.store(enabled, Ordering::SeqCst);
}

/// Whether experimental FP8 recipes may be offered (feature ∧ runtime).
pub fn experimental_fp8_enabled() -> bool {
    EXPERIMENTAL_FP8 && RUNTIME_ENABLE.load(Ordering::SeqCst)
}

/// True only for concrete gfx12 targets with native OCP FP8 support.
///
/// This preserves the original native-capability contract. Use
/// [`lowering_available`] when software-decoded gfx11 recipes are acceptable.
pub fn available(arch: &str) -> bool {
    is_gfx12(arch)
}

/// True for concrete gfx11 software-lowering or gfx12 native targets.
///
/// Family aliases (`gfx11`, `gfx12`) are rejected because compiler builtins and
/// source gates require a concrete offload architecture.
pub fn lowering_available(arch: &str) -> bool {
    is_gfx11(arch) || is_gfx12(arch)
}

/// Native experimental FP8 recipe candidates for `arch`.
///
/// Returns empty unless `available(arch)` and `experimental_fp8_enabled()`.
pub fn candidates(arch: &str) -> Vec<Fp8Recipe> {
    if !available(arch) || !experimental_fp8_enabled() {
        return Vec::new();
    }
    build_gfx12_ocp_recipes()
}

/// Architecture-explicit native or software-lowered FP8 candidates.
pub fn lowering_candidates(arch: &str) -> Vec<Fp8LoweringRecipe> {
    if !lowering_available(arch) || !experimental_fp8_enabled() {
        return Vec::new();
    }
    if is_gfx12(arch) {
        build_gfx12_lowering_recipes()
    } else {
        build_gfx11_lowering_recipes()
    }
}

fn normalize_arch(arch: &str) -> &str {
    let arch = arch.strip_prefix("amdgcn-amd-amdhsa--").unwrap_or(arch);
    arch.split(':').next().unwrap_or(arch)
}

fn is_gfx11(arch: &str) -> bool {
    matches!(
        normalize_arch(arch),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151" | "gfx1152"
    )
}

fn is_gfx12(arch: &str) -> bool {
    matches!(normalize_arch(arch), "gfx1200" | "gfx1201")
}

fn build_gfx12_ocp_recipes() -> Vec<Fp8Recipe> {
    vec![
        Fp8Recipe {
            format: Fp8Format::E4M3Ocp,
            wmma: false,
            source_variant: source_cvt_e4m3().into(),
        },
        Fp8Recipe {
            format: Fp8Format::E5M2Ocp,
            wmma: false,
            source_variant: source_cvt_e5m2().into(),
        },
        Fp8Recipe {
            format: Fp8Format::E4M3Ocp,
            wmma: true,
            source_variant: source_wmma_fp8_fp8().into(),
        },
        Fp8Recipe {
            format: Fp8Format::E5M2Ocp,
            wmma: true,
            source_variant: source_wmma_bf8_bf8().into(),
        },
    ]
}

fn build_gfx12_lowering_recipes() -> Vec<Fp8LoweringRecipe> {
    vec![
        Fp8LoweringRecipe {
            format: Fp8Format::E4M3Ocp,
            lowering: Fp8Lowering::NativeConvert,
            source_variant: source_cvt_e4m3().into(),
        },
        Fp8LoweringRecipe {
            format: Fp8Format::E5M2Ocp,
            lowering: Fp8Lowering::NativeConvert,
            source_variant: source_cvt_e5m2().into(),
        },
        Fp8LoweringRecipe {
            format: Fp8Format::E4M3Ocp,
            lowering: Fp8Lowering::NativeWmma,
            source_variant: source_wmma_fp8_fp8().into(),
        },
        Fp8LoweringRecipe {
            format: Fp8Format::E5M2Ocp,
            lowering: Fp8Lowering::NativeWmma,
            source_variant: source_wmma_bf8_bf8().into(),
        },
    ]
}

fn build_gfx11_lowering_recipes() -> Vec<Fp8LoweringRecipe> {
    vec![
        Fp8LoweringRecipe {
            format: Fp8Format::E4M3Ocp,
            lowering: Fp8Lowering::SoftwareDecode,
            source_variant: source_gfx11_decode_e4m3().into(),
        },
        Fp8LoweringRecipe {
            format: Fp8Format::E5M2Ocp,
            lowering: Fp8Lowering::SoftwareDecode,
            source_variant: source_gfx11_decode_e5m2().into(),
        },
        Fp8LoweringRecipe {
            format: Fp8Format::E4M3Ocp,
            lowering: Fp8Lowering::SoftwareDecodeF16Wmma,
            source_variant: source_gfx11_wmma_e4m3().into(),
        },
        Fp8LoweringRecipe {
            format: Fp8Format::E5M2Ocp,
            lowering: Fp8Lowering::SoftwareDecodeF16Wmma,
            source_variant: source_gfx11_wmma_e5m2().into(),
        },
    ]
}

/// HIP fragment: OCP e4m3 scalar + packed x2 via verified cvt intrinsics.
///
/// Headers: `struct __hip_fp8_e4m3`, `__hip_cvt_float_to_fp8`,
/// `__hip_cvt_float2_to_fp8x2`, `__hip_fp8x2_e4m3`.
fn source_cvt_e4m3() -> &'static str {
    concat!(
        "#include <hip/hip_runtime.h>\n",
        "#include <hip/hip_fp8.h>\n",
        "\n",
        "// OCP E4M3 on gfx1200/gfx1201 (HIP_FP8_TYPE_OCP). CDNA3-only encodings omitted.\n",
        "__device__ __hip_fp8_storage_t radiowave_fp8_e4m3_from_float(float x) {\n",
        "    return __hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3);\n",
        "}\n",
        "\n",
        "__device__ __hip_fp8x2_storage_t radiowave_fp8x2_e4m3_from_float2(float2 v) {\n",
        "    return __hip_cvt_float2_to_fp8x2(v, __HIP_SATFINITE, __HIP_E4M3);\n",
        "}\n",
        "\n",
        "__device__ float radiowave_fp8_e4m3_roundtrip(float x) {\n",
        "    __hip_fp8_e4m3 a(x);\n",
        "    __hip_fp8x2_e4m3 p(float2(x, x));\n",
        "    (void)p;\n",
        "    return static_cast<float>(a);\n",
        "}\n",
        "\n",
        "extern \"C\" __global__ void radiowave_ocp_e4m3_native_convert_gfx12(\n",
        "    const float* in, unsigned char* encoded, float* roundtrip, int n) {\n",
        "    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n",
        "    if (i < n) {\n",
        "        encoded[i] = radiowave_fp8_e4m3_from_float(in[i]);\n",
        "        roundtrip[i] = radiowave_fp8_e4m3_roundtrip(in[i]);\n",
        "    }\n",
        "}\n",
    )
}

/// HIP fragment: OCP e5m2 scalar + packed x2 via verified cvt intrinsics.
///
/// Headers: `struct __hip_fp8_e5m2`, `__hip_cvt_float_to_fp8` with `__HIP_E5M2`,
/// `__hip_fp8x2_e5m2`.
fn source_cvt_e5m2() -> &'static str {
    concat!(
        "#include <hip/hip_runtime.h>\n",
        "#include <hip/hip_fp8.h>\n",
        "\n",
        "// OCP E5M2 on gfx1200/gfx1201 (HIP_FP8_TYPE_OCP). CDNA3-only encodings omitted.\n",
        "__device__ __hip_fp8_storage_t radiowave_fp8_e5m2_from_float(float x) {\n",
        "    return __hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E5M2);\n",
        "}\n",
        "\n",
        "__device__ __hip_fp8x2_storage_t radiowave_fp8x2_e5m2_from_float2(float2 v) {\n",
        "    return __hip_cvt_float2_to_fp8x2(v, __HIP_SATFINITE, __HIP_E5M2);\n",
        "}\n",
        "\n",
        "__device__ float radiowave_fp8_e5m2_roundtrip(float x) {\n",
        "    __hip_fp8_e5m2 a(x);\n",
        "    __hip_fp8x2_e5m2 p(float2(x, x));\n",
        "    (void)p;\n",
        "    return static_cast<float>(a);\n",
        "}\n",
        "\n",
        "extern \"C\" __global__ void radiowave_ocp_e5m2_native_convert_gfx12(\n",
        "    const float* in, unsigned char* encoded, float* roundtrip, int n) {\n",
        "    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n",
        "    if (i < n) {\n",
        "        encoded[i] = radiowave_fp8_e5m2_from_float(in[i]);\n",
        "        roundtrip[i] = radiowave_fp8_e5m2_roundtrip(in[i]);\n",
        "    }\n",
        "}\n",
    )
}

/// HIP fragment: software OCP E4M3 decode to FP16 on gfx11.
///
/// E4M3 finite normals map directly into FP16 exponent/mantissa fields.
/// E4M3 subnormals are normalized into FP16; `0x7f`/`0xff` map to NaN.
fn source_gfx11_decode_e4m3() -> &'static str {
    concat!(
        "#include <hip/hip_runtime.h>\n",
        "#include <hip/hip_fp16.h>\n",
        "\n",
        "#if defined(__HIP_DEVICE_COMPILE__) && \\\n",
        "    !(defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \\\n",
        "      defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \\\n",
        "      defined(__gfx1152__))\n",
        "#error \"radiowave software OCP E4M3 lowering requires a concrete gfx11 target\"\n",
        "#endif\n",
        "__device__ __forceinline__ _Float16 radiowave_ocp_e4m3_to_f16(unsigned char b) {\n",
        "    const unsigned int sign = ((unsigned int)b & 0x80u) << 8;\n",
        "    const unsigned int mag = (unsigned int)b & 0x7fu;\n",
        "    unsigned int bits;\n",
        "    if (mag == 0u) {\n",
        "        bits = sign;\n",
        "    } else if (mag == 0x7fu) {\n",
        "        bits = sign | 0x7e00u;\n",
        "    } else {\n",
        "        const unsigned int exp = mag >> 3;\n",
        "        const unsigned int mant = mag & 7u;\n",
        "        if (exp != 0u) {\n",
        "            bits = sign | ((exp + 8u) << 10) | (mant << 7);\n",
        "        } else {\n",
        "            const unsigned int leading = mant >= 4u ? 2u : (mant >= 2u ? 1u : 0u);\n",
        "            bits = sign | ((leading + 6u) << 10)\n",
        "                | ((mant - (1u << leading)) << (10u - leading));\n",
        "        }\n",
        "    }\n",
        "    return __builtin_bit_cast(_Float16, (unsigned short)bits);\n",
        "}\n",
        "\n",
        "extern \"C\" __global__ void radiowave_ocp_e4m3_decode_gfx11(\n",
        "    const unsigned char* in, _Float16* out, int n) {\n",
        "    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n",
        "    if (i < n) out[i] = radiowave_ocp_e4m3_to_f16(in[i]);\n",
        "}\n",
    )
}

/// HIP fragment: software OCP E5M2 decode to FP16 on gfx11.
///
/// OCP E5M2 and FP16 share exponent width/bias; shifting the byte into the
/// high half bits preserves zero, finite, infinity, and NaN encodings.
fn source_gfx11_decode_e5m2() -> &'static str {
    concat!(
        "#include <hip/hip_runtime.h>\n",
        "#include <hip/hip_fp16.h>\n",
        "\n",
        "#if defined(__HIP_DEVICE_COMPILE__) && \\\n",
        "    !(defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \\\n",
        "      defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \\\n",
        "      defined(__gfx1152__))\n",
        "#error \"radiowave software OCP E5M2 lowering requires a concrete gfx11 target\"\n",
        "#endif\n",
        "__device__ __forceinline__ _Float16 radiowave_ocp_e5m2_to_f16(unsigned char b) {\n",
        "    const unsigned short bits = (unsigned short)b << 8;\n",
        "    return __builtin_bit_cast(_Float16, bits);\n",
        "}\n",
        "\n",
        "extern \"C\" __global__ void radiowave_ocp_e5m2_decode_gfx11(\n",
        "    const unsigned char* in, _Float16* out, int n) {\n",
        "    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n",
        "    if (i < n) out[i] = radiowave_ocp_e5m2_to_f16(in[i]);\n",
        "}\n",
    )
}

macro_rules! source_gfx11_wmma_prefix_e4m3 {
    () => {
        concat!(
            "#include <hip/hip_runtime.h>\n",
            "#include <hip/hip_fp16.h>\n",
            "\n",
            "#if defined(__HIP_DEVICE_COMPILE__) && \\\n",
            "    !(defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \\\n",
            "      defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \\\n",
            "      defined(__gfx1152__))\n",
            "#error \"radiowave staged OCP E4M3 WMMA requires a concrete gfx11 target\"\n",
            "#endif\n",
            "typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;\n",
            "typedef float __attribute__((ext_vector_type(8))) float8_t;\n",
            "__device__ __forceinline__ _Float16 radiowave_ocp_e4m3_to_f16(unsigned char b) {\n",
            "    const unsigned int sign = ((unsigned int)b & 0x80u) << 8;\n",
            "    const unsigned int mag = (unsigned int)b & 0x7fu;\n",
            "    unsigned int bits;\n",
            "    if (mag == 0u) bits = sign;\n",
            "    else if (mag == 0x7fu) bits = sign | 0x7e00u;\n",
            "    else {\n",
            "        const unsigned int exp = mag >> 3;\n",
            "        const unsigned int mant = mag & 7u;\n",
            "        if (exp != 0u) bits = sign | ((exp + 8u) << 10) | (mant << 7);\n",
            "        else {\n",
            "            const unsigned int leading = mant >= 4u ? 2u : (mant >= 2u ? 1u : 0u);\n",
            "            bits = sign | ((leading + 6u) << 10)\n",
            "                | ((mant - (1u << leading)) << (10u - leading));\n",
            "        }\n",
            "    }\n",
            "    return __builtin_bit_cast(_Float16, (unsigned short)bits);\n",
            "}\n",
        )
    };
}

macro_rules! source_gfx11_wmma_prefix_e5m2 {
    () => {
        concat!(
            "#include <hip/hip_runtime.h>\n",
            "#include <hip/hip_fp16.h>\n",
            "\n",
            "#if defined(__HIP_DEVICE_COMPILE__) && \\\n",
            "    !(defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \\\n",
            "      defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \\\n",
            "      defined(__gfx1152__))\n",
            "#error \"radiowave staged OCP E5M2 WMMA requires a concrete gfx11 target\"\n",
            "#endif\n",
            "typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;\n",
            "typedef float __attribute__((ext_vector_type(8))) float8_t;\n",
            "__device__ __forceinline__ _Float16 radiowave_ocp_e5m2_to_f16(unsigned char b) {\n",
            "    const unsigned short bits = (unsigned short)b << 8;\n",
            "    return __builtin_bit_cast(_Float16, bits);\n",
            "}\n",
        )
    };
}

/// HIP fragment: software E4M3 decode followed by gfx11 FP16 WMMA.
fn source_gfx11_wmma_e4m3() -> &'static str {
    concat!(
        source_gfx11_wmma_prefix_e4m3!(),
        "extern \"C\" __launch_bounds__(32, 8) __global__ void radiowave_ocp_e4m3_f16_wmma_gfx11(\n",
        "    const unsigned char* a, const unsigned char* b, float* out, int tiles) {\n",
        "    const int tile = (int)blockIdx.x;\n",
        "    const int tid = (int)threadIdx.x;\n",
        "    if (tile >= tiles) return;\n",
        "    const long long base = ((long long)tile * 32 + tid) * 16;\n",
        "    half16_t av;\n",
        "    half16_t bv;\n",
        "    #pragma unroll\n",
        "    for (int i = 0; i < 16; ++i) {\n",
        "        av[i] = radiowave_ocp_e4m3_to_f16(a[base + i]);\n",
        "        bv[i] = radiowave_ocp_e4m3_to_f16(b[base + i]);\n",
        "    }\n",
        "    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};\n",
        "    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av, bv, acc);\n",
        "    #pragma unroll\n",
        "    for (int i = 0; i < 8; ++i) out[((long long)tile * 32 + tid) * 8 + i] = acc[i];\n",
        "}\n",
    )
}

/// HIP fragment: software E5M2 decode followed by gfx11 FP16 WMMA.
fn source_gfx11_wmma_e5m2() -> &'static str {
    concat!(
        source_gfx11_wmma_prefix_e5m2!(),
        "extern \"C\" __launch_bounds__(32, 8) __global__ void radiowave_ocp_e5m2_f16_wmma_gfx11(\n",
        "    const unsigned char* a, const unsigned char* b, float* out, int tiles) {\n",
        "    const int tile = (int)blockIdx.x;\n",
        "    const int tid = (int)threadIdx.x;\n",
        "    if (tile >= tiles) return;\n",
        "    const long long base = ((long long)tile * 32 + tid) * 16;\n",
        "    half16_t av;\n",
        "    half16_t bv;\n",
        "    #pragma unroll\n",
        "    for (int i = 0; i < 16; ++i) {\n",
        "        av[i] = radiowave_ocp_e5m2_to_f16(a[base + i]);\n",
        "        bv[i] = radiowave_ocp_e5m2_to_f16(b[base + i]);\n",
        "    }\n",
        "    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};\n",
        "    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av, bv, acc);\n",
        "    #pragma unroll\n",
        "    for (int i = 0; i < 8; ++i) out[((long long)tile * 32 + tid) * 8 + i] = acc[i];\n",
        "}\n",
    )
}

/// HIP fragment: native OCP E4M3 + gfx12 WMMA fp8×fp8→f32 builtin.
///
/// Builtin: `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` (wmma_impl.hpp).
/// Register layout from rocWMMA: A/B = `VecT<int,2>`, C/D = `AccRegF32x8`.
fn source_wmma_fp8_fp8() -> &'static str {
    concat!(
        "#include <hip/hip_runtime.h>\n",
        "\n",
        "// gfx12 native OCP E4M3 WMMA fp8×fp8 → f32 (wave32).\n",
        "#if defined(__HIP_DEVICE_COMPILE__) && \\\n",
        "    !(defined(__gfx1200__) || defined(__gfx1201__))\n",
        "#error \"radiowave FP8 WMMA fp8×fp8 requires __gfx1200__ or __gfx1201__\"\n",
        "#endif\n",
        "extern \"C\" __launch_bounds__(32, 8) __global__ void radiowave_wmma_fp8_fp8_probe(\n",
        "    const int* a_raw, const int* b_raw, float* out) {\n",
        "#if defined(__gfx1200__) || defined(__gfx1201__)\n",
        "    using AVec = __attribute__((__vector_size__(2 * sizeof(int)))) int;\n",
        "    using BVec = __attribute__((__vector_size__(2 * sizeof(int)))) int;\n",
        "    using CVec = __attribute__((__vector_size__(8 * sizeof(float)))) float;\n",
        "    const long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n",
        "    AVec a = *(const AVec*)(a_raw + lane * 2);\n",
        "    BVec b = *(const BVec*)(b_raw + lane * 2);\n",
        "    CVec c = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};\n",
        "    CVec d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);\n",
        "    #pragma unroll\n",
        "    for (int i = 0; i < 8; ++i) out[lane * 8 + i] = d[i];\n",
        "#else\n",
        "    (void)a_raw; (void)b_raw; (void)out;\n",
        "#endif\n",
        "}\n",
    )
}

/// HIP fragment: native OCP E5M2 + gfx12 WMMA bf8×bf8→f32 builtin.
///
/// Builtin: `__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12`.
fn source_wmma_bf8_bf8() -> &'static str {
    concat!(
        "#include <hip/hip_runtime.h>\n",
        "\n",
        "// gfx12 native OCP E5M2 WMMA bf8×bf8 → f32 (wave32).\n",
        "#if defined(__HIP_DEVICE_COMPILE__) && \\\n",
        "    !(defined(__gfx1200__) || defined(__gfx1201__))\n",
        "#error \"radiowave FP8 WMMA bf8×bf8 requires __gfx1200__ or __gfx1201__\"\n",
        "#endif\n",
        "extern \"C\" __launch_bounds__(32, 8) __global__ void radiowave_wmma_bf8_bf8_probe(\n",
        "    const int* a_raw, const int* b_raw, float* out) {\n",
        "#if defined(__gfx1200__) || defined(__gfx1201__)\n",
        "    using AVec = __attribute__((__vector_size__(2 * sizeof(int)))) int;\n",
        "    using BVec = __attribute__((__vector_size__(2 * sizeof(int)))) int;\n",
        "    using CVec = __attribute__((__vector_size__(8 * sizeof(float)))) float;\n",
        "    const long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n",
        "    AVec a = *(const AVec*)(a_raw + lane * 2);\n",
        "    BVec b = *(const BVec*)(b_raw + lane * 2);\n",
        "    CVec c = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};\n",
        "    CVec d = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a, b, c);\n",
        "    #pragma unroll\n",
        "    for (int i = 0; i < 8; ++i) out[lane * 8 + i] = d[i];\n",
        "#else\n",
        "    (void)a_raw; (void)b_raw; (void)out;\n",
        "#endif\n",
        "}\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "experimental-fp8")]
    use std::io::Write;
    #[cfg(feature = "experimental-fp8")]
    use std::process::Command;

    /// Process-global `RUNTIME_ENABLE` is shared across tests. Hold this for
    /// the whole body of any test that reads or toggles it so cargo's default
    /// parallelism cannot interleave enable/disable (flake under `cargo test`).
    static FP8_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock_fp8_tests() -> MutexGuard<'static, ()> {
        FP8_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    struct EnableGuard {
        prev: bool,
        /// Held for the guard's lifetime so no other fp8 test can race the flag.
        _lock: MutexGuard<'static, ()>,
    }

    impl EnableGuard {
        fn set(enabled: bool) -> Self {
            let lock = lock_fp8_tests();
            let prev = RUNTIME_ENABLE.swap(enabled, Ordering::SeqCst);
            Self { prev, _lock: lock }
        }
    }

    impl Drop for EnableGuard {
        fn drop(&mut self) {
            RUNTIME_ENABLE.store(self.prev, Ordering::SeqCst);
        }
    }

    #[test]
    fn available_preserves_native_gfx12_contract() {
        for arch in ["gfx1200", "gfx1201"] {
            assert!(available(arch), "{arch}");
        }
        assert!(available("amdgcn-amd-amdhsa--gfx1201"));
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152",
        ] {
            assert!(!available(arch), "{arch}");
        }
        assert!(!available("amdgcn-amd-amdhsa--gfx1100:sramecc+:xnack-"));
        assert!(!available("gfx11"));
        assert!(!available("gfx12"));
        assert!(!available("gfx120"));
        assert!(!available("gfx942"));
        assert!(!available("gfx950"));
        assert!(!available(""));
    }

    #[test]
    fn lowering_available_for_concrete_gfx11_and_gfx12_targets() {
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200",
            "gfx1201",
        ] {
            assert!(lowering_available(arch), "{arch}");
        }
        assert!(lowering_available("amdgcn-amd-amdhsa--gfx1201"));
        assert!(lowering_available(
            "amdgcn-amd-amdhsa--gfx1100:sramecc+:xnack-"
        ));
        assert!(!lowering_available("gfx11"));
        assert!(!lowering_available("gfx12"));
        assert!(!lowering_available("gfx120"));
        assert!(!lowering_available("gfx942"));
        assert!(!lowering_available("gfx950"));
        assert!(!lowering_available(""));
    }

    #[test]
    fn experimental_const_defaults_match_feature() {
        let _lock = lock_fp8_tests();
        #[cfg(feature = "experimental-fp8")]
        assert!(EXPERIMENTAL_FP8);
        #[cfg(not(feature = "experimental-fp8"))]
        assert!(!EXPERIMENTAL_FP8);
        // Runtime defaults off regardless of feature (when no other test holds it).
        assert!(!RUNTIME_ENABLE.load(Ordering::SeqCst));
    }

    #[test]
    fn candidates_empty_when_runtime_disabled() {
        let _g = EnableGuard::set(false);
        assert!(candidates("gfx1100").is_empty());
        assert!(candidates("gfx1201").is_empty());
        assert!(candidates("gfx1200").is_empty());
        assert!(lowering_candidates("gfx1100").is_empty());
        assert!(lowering_candidates("gfx1201").is_empty());
    }

    #[test]
    fn candidates_empty_on_unsupported_arch_even_if_runtime_flag_set() {
        let _g = EnableGuard::set(true);
        assert!(candidates("gfx942").is_empty());
        assert!(candidates("gfx11").is_empty());
        assert!(candidates("gfx12").is_empty());
        assert!(lowering_candidates("gfx942").is_empty());
        assert!(lowering_candidates("gfx11").is_empty());
        assert!(lowering_candidates("gfx12").is_empty());
    }

    #[test]
    #[cfg(feature = "experimental-fp8")]
    fn candidates_select_arch_specific_lowerings() {
        let _g = EnableGuard::set(true);
        assert!(
            candidates("gfx1100").is_empty(),
            "the historical candidates API remains native-only"
        );

        let native_gfx12 = candidates("gfx1201");
        assert_eq!(
            native_gfx12.len(),
            4,
            "two formats x (native cvt + native wmma)"
        );
        assert_eq!(native_gfx12.iter().filter(|r| r.wmma).count(), 2);

        let gfx11 = lowering_candidates("gfx1100");
        assert_eq!(gfx11.len(), 4, "two formats × (decode + staged wmma)");
        assert!(gfx11.iter().all(|r| matches!(
            r.lowering,
            Fp8Lowering::SoftwareDecode | Fp8Lowering::SoftwareDecodeF16Wmma
        )));
        assert!(
            gfx11
                .iter()
                .any(|r| r.format == Fp8Format::E4M3Ocp && !r.uses_wmma())
        );
        assert!(
            gfx11
                .iter()
                .any(|r| r.format == Fp8Format::E5M2Ocp && r.uses_wmma())
        );

        let gfx12 = lowering_candidates("gfx1201");
        assert_eq!(gfx12.len(), 4, "two formats × (native cvt + native wmma)");
        assert!(gfx12.iter().all(|r| matches!(
            r.lowering,
            Fp8Lowering::NativeConvert | Fp8Lowering::NativeWmma
        )));
    }

    #[test]
    #[cfg(not(feature = "experimental-fp8"))]
    fn candidates_empty_without_feature_even_if_runtime() {
        let _g = EnableGuard::set(true);
        assert!(candidates("gfx1100").is_empty());
        assert!(candidates("gfx1201").is_empty());
        assert!(candidates("gfx1200").is_empty());
        assert!(lowering_candidates("gfx1100").is_empty());
        assert!(lowering_candidates("gfx1201").is_empty());
        assert!(lowering_candidates("gfx1200").is_empty());
    }

    #[test]
    fn wmma_fragments_fail_closed_without_gfx12_gate() {
        let fp8 = source_wmma_fp8_fp8();
        let bf8 = source_wmma_bf8_bf8();
        assert!(fp8.contains("#error"), "must not silently return 0.f");
        assert!(bf8.contains("#error"), "must not silently return 0.f");
        assert!(!fp8.contains("return 0.f;"));
        assert!(!bf8.contains("return 0.f;"));
    }

    #[test]
    fn format_names_match_header_types() {
        assert_eq!(Fp8Format::E4M3Ocp.hip_type_name(), "__hip_fp8_e4m3");
        assert_eq!(Fp8Format::E5M2Ocp.hip_type_name(), "__hip_fp8_e5m2");
        assert_eq!(Fp8Format::E4M3Ocp.hip_interpretation(), "__HIP_E4M3");
        assert_eq!(Fp8Format::E5M2Ocp.hip_interpretation(), "__HIP_E5M2");
        assert_eq!(Fp8Format::E4M3Ocp.rocwmma_type_name(), "rocwmma::float8_t");
        assert_eq!(Fp8Format::E5M2Ocp.rocwmma_type_name(), "rocwmma::bfloat8_t");
    }

    #[test]
    fn source_fragments_reference_verified_intrinsics() {
        let e4 = source_cvt_e4m3();
        assert!(e4.contains("__hip_cvt_float_to_fp8"));
        assert!(e4.contains("__hip_cvt_float2_to_fp8x2"));
        assert!(e4.contains("__hip_fp8_e4m3"));
        assert!(e4.contains("__HIP_E4M3"));
        assert!(!e4.contains("__hip_fp8_e4m3_fnuz"));
        assert!(!e4.contains("__HIP_E4M3_FNUZ"));

        let e5 = source_cvt_e5m2();
        assert!(e5.contains("__hip_fp8_e5m2"));
        assert!(e5.contains("__HIP_E5M2"));
        assert!(!e5.contains("__hip_fp8_e5m2_fnuz"));
        assert!(!e5.contains("__HIP_E5M2_FNUZ"));

        let w = source_wmma_fp8_fp8();
        assert!(w.contains("__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12"));
        assert!(w.contains("__gfx1200__") && w.contains("__gfx1201__"));
        assert!(!w.contains("rocwmma/rocwmma.hpp"));
        // Bare __gfx12__ must not appear as a gate (only concrete 1200/1201).
        assert!(!w.contains("defined(__gfx12__)"));

        let b = source_wmma_bf8_bf8();
        assert!(b.contains("__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12"));

        let gfx11_e4 = source_gfx11_decode_e4m3();
        assert!(gfx11_e4.contains("radiowave_ocp_e4m3_to_f16"));
        assert!(gfx11_e4.contains("mag == 0x7fu"));
        assert!(!gfx11_e4.contains("__hip_fp8_e4m3"));

        let gfx11_e5 = source_gfx11_decode_e5m2();
        assert!(gfx11_e5.contains("(unsigned short)b << 8"));
        assert!(!gfx11_e5.contains("__hip_fp8_e5m2"));

        for staged in [source_gfx11_wmma_e4m3(), source_gfx11_wmma_e5m2()] {
            assert!(staged.contains("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32"));
            assert!(!staged.contains("_fp8_fp8_"));
            assert!(!staged.contains("_bf8_bf8_"));
        }
    }

    #[test]
    fn build_arch_recipes_cover_decode_and_wmma() {
        let native_gfx12 = build_gfx12_ocp_recipes();
        assert_eq!(native_gfx12.len(), 4);
        assert_eq!(native_gfx12.iter().filter(|r| r.wmma).count(), 2);
        let literal = Fp8Recipe {
            format: Fp8Format::E4M3Ocp,
            wmma: false,
            source_variant: String::new(),
        };
        assert!(!literal.wmma, "the historical struct literal remains valid");

        let gfx12 = build_gfx12_lowering_recipes();
        assert!(
            gfx12
                .iter()
                .any(|r| r.lowering == Fp8Lowering::NativeConvert && !r.uses_wmma())
        );
        assert!(
            gfx12
                .iter()
                .any(|r| r.lowering == Fp8Lowering::NativeWmma && r.uses_wmma())
        );

        let gfx11 = build_gfx11_lowering_recipes();
        assert_eq!(gfx11.len(), 4);
        assert!(
            gfx11
                .iter()
                .any(|r| r.lowering == Fp8Lowering::SoftwareDecode && !r.uses_wmma())
        );
        assert!(
            gfx11
                .iter()
                .any(|r| r.lowering == Fp8Lowering::SoftwareDecodeF16Wmma && r.uses_wmma())
        );
        for r in gfx11.iter().chain(gfx12.iter()) {
            assert!(!r.source_variant.is_empty());
        }
        for r in native_gfx12 {
            assert!(!r.source_variant.is_empty());
        }
    }

    #[test]
    fn ocp_decode_semantics_cover_special_values_and_signs() {
        assert_eq!(decode_e4m3(0x00).to_bits(), 0.0f32.to_bits());
        assert_eq!(decode_e4m3(0x80).to_bits(), (-0.0f32).to_bits());
        assert_eq!(decode_e4m3(0x01), 2.0f32.powi(-9));
        assert_eq!(decode_e4m3(0x07), 7.0 * 2.0f32.powi(-9));
        assert_eq!(decode_e4m3(0x08), 2.0f32.powi(-6));
        assert_eq!(decode_e4m3(0x38), 1.0);
        assert_eq!(decode_e4m3(0x7e), 448.0);
        assert!(decode_e4m3(0x7f).is_nan());

        assert_eq!(decode_e5m2(0x00).to_bits(), 0.0f32.to_bits());
        assert_eq!(decode_e5m2(0x80).to_bits(), (-0.0f32).to_bits());
        assert_eq!(decode_e5m2(0x01), 2.0f32.powi(-16));
        assert_eq!(decode_e5m2(0x04), 2.0f32.powi(-14));
        assert_eq!(decode_e5m2(0x3c), 1.0);
        assert_eq!(decode_e5m2(0x7b), 57_344.0);
        assert_eq!(decode_e5m2(0x7c), f32::INFINITY);
        assert!(decode_e5m2(0x7d).is_nan());

        for byte in 0u8..=0x7f {
            let pos_e4 = decode_e4m3(byte);
            let neg_e4 = decode_e4m3(byte | 0x80);
            if pos_e4.is_nan() {
                assert!(neg_e4.is_nan());
            } else {
                assert_eq!(neg_e4.to_bits(), pos_e4.to_bits() | 0x8000_0000);
            }

            let pos_e5 = decode_e5m2(byte);
            let neg_e5 = decode_e5m2(byte | 0x80);
            if pos_e5.is_nan() {
                assert!(neg_e5.is_nan());
            } else {
                assert_eq!(neg_e5.to_bits(), pos_e5.to_bits() | 0x8000_0000);
            }
        }
    }

    /// Compile each HIP source fragment against real ROCm headers and target
    /// builtins. `-c` is intentional: it validates device code generation, not
    /// only host-side syntax.
    /// Skips gracefully when hipcc is absent (CI without ROCm toolchain).
    #[test]
    #[cfg(feature = "experimental-fp8")]
    fn hipcc_compiles_source_fragments_for_gfx11_and_gfx12() {
        let hipcc = match find_hipcc() {
            Some(p) => p,
            None => {
                eprintln!("skipping hipcc syntax check: hipcc not found on PATH or ROCM_PATH");
                return;
            }
        };

        let fragments = [
            ("gfx1201", "cvt_e4m3", source_cvt_e4m3()),
            ("gfx1201", "cvt_e5m2", source_cvt_e5m2()),
            ("gfx1201", "wmma_fp8", source_wmma_fp8_fp8()),
            ("gfx1201", "wmma_bf8", source_wmma_bf8_bf8()),
            ("gfx1100", "decode_e4m3", source_gfx11_decode_e4m3()),
            ("gfx1100", "decode_e5m2", source_gfx11_decode_e5m2()),
            ("gfx1100", "wmma_e4m3_f16", source_gfx11_wmma_e4m3()),
            ("gfx1100", "wmma_e5m2_f16", source_gfx11_wmma_e5m2()),
            ("gfx1101", "wmma_e4m3_f16", source_gfx11_wmma_e4m3()),
            ("gfx1101", "wmma_e5m2_f16", source_gfx11_wmma_e5m2()),
            ("gfx1102", "wmma_e4m3_f16", source_gfx11_wmma_e4m3()),
            ("gfx1102", "wmma_e5m2_f16", source_gfx11_wmma_e5m2()),
            ("gfx1103", "wmma_e4m3_f16", source_gfx11_wmma_e4m3()),
            ("gfx1103", "wmma_e5m2_f16", source_gfx11_wmma_e5m2()),
            ("gfx1150", "wmma_e4m3_f16", source_gfx11_wmma_e4m3()),
            ("gfx1150", "wmma_e5m2_f16", source_gfx11_wmma_e5m2()),
            ("gfx1151", "wmma_e4m3_f16", source_gfx11_wmma_e4m3()),
            ("gfx1151", "wmma_e5m2_f16", source_gfx11_wmma_e5m2()),
            ("gfx1152", "wmma_e4m3_f16", source_gfx11_wmma_e4m3()),
            ("gfx1152", "wmma_e5m2_f16", source_gfx11_wmma_e5m2()),
        ];

        let dir = std::env::temp_dir().join(format!("radiowave-fp8-syntax-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm/core".into());
        let include = format!("{rocm}/include");

        for (arch, name, src) in fragments {
            let path = dir.join(format!("{arch}-{name}.hip"));
            let object = dir.join(format!("{arch}-{name}.o"));
            {
                let mut f = std::fs::File::create(&path).expect("create hip fragment");
                f.write_all(src.as_bytes()).expect("write hip fragment");
            }

            let output = Command::new(&hipcc)
                .args([
                    &format!("--offload-arch={arch}"),
                    "-c",
                    "-x",
                    "hip",
                    "-I",
                    &include,
                    path.to_str().expect("utf8 path"),
                    "-o",
                    object.to_str().expect("utf8 object path"),
                ])
                .env(
                    "PATH",
                    format!(
                        "{}:{rocm}/bin:{rocm}/lib/llvm/bin",
                        std::env::var("PATH").unwrap_or_default(),
                    ),
                )
                .env("ROCM_PATH", &rocm)
                .output();

            match output {
                Ok(out) if out.status.success() => {}
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    panic!(
                        "hipcc compile failed for {arch}/{name} (status {})\nstdout:\n{stdout}\nstderr:\n{stderr}",
                        out.status
                    );
                }
                Err(err) => {
                    eprintln!("skipping hipcc syntax check: failed to spawn hipcc: {err}");
                    return;
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "experimental-fp8")]
    fn hipcc_rejects_recipe_on_wrong_architecture() {
        let hipcc = match find_hipcc() {
            Some(p) => p,
            None => {
                eprintln!("skipping hipcc gate check: hipcc not found on PATH or ROCM_PATH");
                return;
            }
        };
        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm/core".into());
        let include = format!("{rocm}/include");
        let dir =
            std::env::temp_dir().join(format!("radiowave-fp8-negative-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create negative compile directory");

        let cases = [
            (
                "gfx1100",
                "native-on-gfx11",
                source_wmma_fp8_fp8(),
                "requires __gfx1200__ or __gfx1201__",
            ),
            (
                "gfx1201",
                "software-on-gfx12",
                source_gfx11_wmma_e4m3(),
                "requires a concrete gfx11 target",
            ),
        ];
        for (arch, name, src, expected_error) in cases {
            let path = dir.join(format!("{name}.hip"));
            let object = dir.join(format!("{name}.o"));
            std::fs::write(&path, src).expect("write negative HIP fragment");
            let output = Command::new(&hipcc)
                .args([
                    &format!("--offload-arch={arch}"),
                    "-c",
                    "-x",
                    "hip",
                    "-I",
                    &include,
                    path.to_str().expect("utf8 path"),
                    "-o",
                    object.to_str().expect("utf8 object path"),
                ])
                .env(
                    "PATH",
                    format!(
                        "{}:{rocm}/bin:{rocm}/lib/llvm/bin",
                        std::env::var("PATH").unwrap_or_default(),
                    ),
                )
                .env("ROCM_PATH", &rocm)
                .output()
                .expect("spawn validated hipcc");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !output.status.success(),
                "{name} unexpectedly compiled for {arch}"
            );
            assert!(
                stderr.contains(expected_error),
                "{name} failed for an unexpected reason on {arch}:\n{stderr}"
            );
        }
    }

    fn decode_e4m3(byte: u8) -> f32 {
        let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
        let mag = byte & 0x7f;
        if mag == 0 {
            return f32::from_bits((byte as u32 & 0x80) << 24);
        }
        if mag == 0x7f {
            return f32::NAN;
        }
        let exponent = (mag >> 3) as i32;
        let mantissa = (mag & 7) as f32;
        if exponent == 0 {
            sign * mantissa * 2.0f32.powi(-9)
        } else {
            sign * 2.0f32.powi(exponent - 7) * (1.0 + mantissa / 8.0)
        }
    }

    fn decode_e5m2(byte: u8) -> f32 {
        let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
        let magnitude = byte & 0x7f;
        if magnitude == 0 {
            return f32::from_bits((byte as u32 & 0x80) << 24);
        }
        let exponent = (magnitude >> 2) as i32;
        let mantissa = (magnitude & 3) as f32;
        if exponent == 0x1f {
            return if mantissa == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }
        if exponent == 0 {
            sign * mantissa * 2.0f32.powi(-16)
        } else {
            sign * 2.0f32.powi(exponent - 15) * (1.0 + mantissa / 4.0)
        }
    }

    #[cfg(feature = "experimental-fp8")]
    fn find_hipcc() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("HIPCC") {
            let pb = std::path::PathBuf::from(&p);
            if pb.is_file() {
                return Some(pb);
            }
        }
        for cand in ["/opt/rocm/core/bin/hipcc", "/opt/rocm/bin/hipcc", "hipcc"] {
            if cand.contains('/') {
                let pb = std::path::PathBuf::from(cand);
                if pb.is_file() {
                    return Some(pb);
                }
            } else if Command::new(cand)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
            {
                return Some(std::path::PathBuf::from(cand));
            }
        }
        None
    }
}
