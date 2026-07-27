// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Experimental gfx12 OCP FP8 recipe types (Wave 2 — types only).
//!
//! Verified against ROCm 7.14 headers:
//! - `hip/hip_fp8.h` → `hip/amd_detail/amd_hip_fp8.h`
//! - OCP `__hip_fp8_e4m3` / `__hip_fp8_e5m2` (+ packed x2/x4) device-gated
//!   `__gfx1200__` / `__gfx1201__` via `HIP_FP8_TYPE_OCP` (no bare `__gfx12__`)
//! - convert: `__hip_cvt_float_to_fp8` / `__hip_cvt_float2_to_fp8x2`
//! - rocWMMA: `rocwmma::float8_t` / `bfloat8_t` +
//!   `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
//!
//! FNUZ (`__hip_fp8_*_fnuz`) is CDNA3/gfx942-only and is excluded here.
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

/// OCP FP8 encodings available on gfx1200/gfx1201 (RDNA4).
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

/// One experimental FP8 source-lowering recipe for gfx12.
///
/// `source_variant` is a HIP source fragment template that exercises the
/// verified intrinsics/types. Catalog emission is intentionally absent until
/// Wave 3; fragments fail closed on non-gfx1200/1201 device compiles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fp8Recipe {
    pub format: Fp8Format,
    /// When true, fragment uses rocWMMA + gfx12 WMMA builtins.
    pub wmma: bool,
    pub source_variant: String,
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

/// True only for concrete RDNA4 gfx12 targets that define OCP device types.
///
/// Matches header gates: `__gfx1200__` / `__gfx1201__` only — bare `gfx12` is
/// rejected (HIP never defines bare `__gfx12__` for FP8).
pub fn available(arch: &str) -> bool {
    matches!(normalize_arch(arch), "gfx1200" | "gfx1201")
}

/// Experimental FP8 recipe candidates for `arch`.
///
/// Returns empty unless `available(arch)` and `experimental_fp8_enabled()`.
pub fn candidates(arch: &str) -> Vec<Fp8Recipe> {
    if !available(arch) || !experimental_fp8_enabled() {
        return Vec::new();
    }
    build_ocp_recipes()
}

fn normalize_arch(arch: &str) -> &str {
    arch.strip_prefix("amdgcn-amd-amdhsa--").unwrap_or(arch)
}

fn build_ocp_recipes() -> Vec<Fp8Recipe> {
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
    )
}

/// HIP fragment: rocWMMA OCP float8_t + gfx12 WMMA fp8×fp8→f32 builtin.
///
/// Builtin: `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` (wmma_impl.hpp).
/// Register layout from rocWMMA: A/B = `VecT<int,2>`, C/D = `AccRegF32x8`.
fn source_wmma_fp8_fp8() -> &'static str {
    concat!(
        "#include <hip/hip_runtime.h>\n",
        "#include <hip/hip_fp8.h>\n",
        "#include <rocwmma/rocwmma.hpp>\n",
        "\n",
        "// gfx12 WMMA OCP fp8×fp8 → f32 (wave32). Types: rocwmma::float8_t = __hip_fp8_e4m3.\n",
        "__device__ float radiowave_wmma_fp8_fp8_probe(\n",
        "    const rocwmma::float8_t* /*a*/,\n",
        "    const rocwmma::float8_t* /*b*/) {\n",
        "#if defined(__gfx1200__) || defined(__gfx1201__)\n",
        "    using AVec = __attribute__((__vector_size__(2 * sizeof(int)))) int;\n",
        "    using BVec = __attribute__((__vector_size__(2 * sizeof(int)))) int;\n",
        "    using CVec = __attribute__((__vector_size__(8 * sizeof(float)))) float;\n",
        "    AVec a = {0, 0};\n",
        "    BVec b = {0, 0};\n",
        "    CVec c = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};\n",
        "    CVec d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);\n",
        "    return d[0];\n",
        "#else\n",
        "    #error \"radiowave FP8 WMMA fp8×fp8 requires __gfx1200__ or __gfx1201__\"\n",
        "#endif\n",
        "}\n",
    )
}

/// HIP fragment: rocWMMA OCP bfloat8_t + gfx12 WMMA bf8×bf8→f32 builtin.
///
/// Builtin: `__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12`.
fn source_wmma_bf8_bf8() -> &'static str {
    concat!(
        "#include <hip/hip_runtime.h>\n",
        "#include <hip/hip_fp8.h>\n",
        "#include <rocwmma/rocwmma.hpp>\n",
        "\n",
        "// gfx12 WMMA OCP bf8×bf8 → f32 (wave32). Types: rocwmma::bfloat8_t = __hip_fp8_e5m2.\n",
        "__device__ float radiowave_wmma_bf8_bf8_probe(\n",
        "    const rocwmma::bfloat8_t* /*a*/,\n",
        "    const rocwmma::bfloat8_t* /*b*/) {\n",
        "#if defined(__gfx1200__) || defined(__gfx1201__)\n",
        "    using AVec = __attribute__((__vector_size__(2 * sizeof(int)))) int;\n",
        "    using BVec = __attribute__((__vector_size__(2 * sizeof(int)))) int;\n",
        "    using CVec = __attribute__((__vector_size__(8 * sizeof(float)))) float;\n",
        "    AVec a = {0, 0};\n",
        "    BVec b = {0, 0};\n",
        "    CVec c = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};\n",
        "    CVec d = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a, b, c);\n",
        "    return d[0];\n",
        "#else\n",
        "    #error \"radiowave FP8 WMMA bf8×bf8 requires __gfx1200__ or __gfx1201__\"\n",
        "#endif\n",
        "}\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
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
    fn available_only_gfx1200_and_gfx1201() {
        assert!(available("gfx1200"));
        assert!(available("gfx1201"));
        assert!(available("amdgcn-amd-amdhsa--gfx1201"));
        assert!(!available("gfx12"));
        assert!(!available("gfx120"));
        assert!(!available("gfx1100"));
        assert!(!available("gfx942"));
        assert!(!available("gfx950"));
        assert!(!available(""));
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
        assert!(candidates("gfx1201").is_empty());
        assert!(candidates("gfx1200").is_empty());
    }

    #[test]
    fn candidates_empty_on_non_gfx12_even_if_runtime_flag_set() {
        let _g = EnableGuard::set(true);
        assert!(candidates("gfx1100").is_empty());
        assert!(candidates("gfx942").is_empty());
        assert!(candidates("gfx12").is_empty());
    }

    #[test]
    #[cfg(feature = "experimental-fp8")]
    fn candidates_populated_when_feature_and_runtime_and_arch() {
        let _g = EnableGuard::set(true);
        let c = candidates("gfx1201");
        assert_eq!(c.len(), 4, "two formats × (cvt + wmma)");
        assert!(c.iter().any(|r| r.format == Fp8Format::E4M3Ocp && !r.wmma));
        assert!(c.iter().any(|r| r.format == Fp8Format::E5M2Ocp && r.wmma));
        for r in &c {
            assert!(
                r.source_variant.contains("#error")
                    || r.source_variant.contains("__hip_cvt")
                    || r.source_variant.contains("wmma"),
                "fragment should be non-empty"
            );
        }
    }

    #[test]
    #[cfg(not(feature = "experimental-fp8"))]
    fn candidates_empty_without_feature_even_if_runtime() {
        let _g = EnableGuard::set(true);
        assert!(candidates("gfx1201").is_empty());
        assert!(candidates("gfx1200").is_empty());
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
        assert!(w.contains("rocwmma::float8_t"));
        assert!(w.contains("__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12"));
        assert!(w.contains("__gfx1200__") && w.contains("__gfx1201__"));
        // Bare __gfx12__ must not appear as a gate (only concrete 1200/1201).
        assert!(!w.contains("defined(__gfx12__)"));

        let b = source_wmma_bf8_bf8();
        assert!(b.contains("rocwmma::bfloat8_t"));
        assert!(b.contains("__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12"));
    }

    #[test]
    fn build_ocp_recipes_cover_cvt_and_wmma() {
        let recipes = build_ocp_recipes();
        assert_eq!(recipes.len(), 4);
        assert!(recipes
            .iter()
            .any(|r| r.format == Fp8Format::E4M3Ocp && !r.wmma));
        assert!(recipes
            .iter()
            .any(|r| r.format == Fp8Format::E5M2Ocp && !r.wmma));
        assert!(recipes
            .iter()
            .any(|r| r.format == Fp8Format::E4M3Ocp && r.wmma));
        assert!(recipes
            .iter()
            .any(|r| r.format == Fp8Format::E5M2Ocp && r.wmma));
        for r in &recipes {
            assert!(!r.source_variant.is_empty());
        }
    }

    /// Compile-validate each HIP source fragment against real 7.14 headers.
    /// Skips gracefully when hipcc is absent (CI without ROCm toolchain).
    #[test]
    fn hipcc_syntax_only_validates_source_fragments() {
        let hipcc = match find_hipcc() {
            Some(p) => p,
            None => {
                eprintln!("skipping hipcc syntax check: hipcc not found on PATH or ROCM_PATH");
                return;
            }
        };

        let fragments = [
            ("cvt_e4m3", source_cvt_e4m3()),
            ("cvt_e5m2", source_cvt_e5m2()),
            ("wmma_fp8", source_wmma_fp8_fp8()),
            ("wmma_bf8", source_wmma_bf8_bf8()),
        ];

        let dir = std::env::temp_dir().join("radiowave-fp8-syntax");
        let _ = std::fs::create_dir_all(&dir);

        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm/core".into());
        let include = format!("{rocm}/include");

        for (name, src) in fragments {
            let path = dir.join(format!("{name}.hip"));
            {
                let mut f = std::fs::File::create(&path).expect("create hip fragment");
                f.write_all(src.as_bytes()).expect("write hip fragment");
            }

            let output = Command::new(&hipcc)
                .args([
                    "--offload-arch=gfx1201",
                    "-fsyntax-only",
                    "-x",
                    "hip",
                    "-I",
                    &include,
                    path.to_str().expect("utf8 path"),
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
                    // Incomplete toolchain only: missing binary/header, not source errors.
                    let incomplete = stderr.contains("failed to execute")
                        || stderr.contains("No such file or directory")
                        || (stderr.contains("fatal error:")
                            && stderr.contains("file not found")
                            && !stderr.contains("/tmp/radiowave-fp8-syntax"));
                    if incomplete {
                        eprintln!(
                            "skipping hipcc syntax check for {name}: toolchain incomplete\n{stderr}"
                        );
                        return;
                    }
                    panic!(
                        "hipcc -fsyntax-only failed for {name} (status {})\nstdout:\n{stdout}\nstderr:\n{stderr}",
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

    fn find_hipcc() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("HIPCC") {
            let pb = std::path::PathBuf::from(&p);
            if pb.is_file() {
                return Some(pb);
            }
        }
        for cand in [
            "/opt/rocm/core/bin/hipcc",
            "/opt/rocm/bin/hipcc",
            "hipcc",
        ] {
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
