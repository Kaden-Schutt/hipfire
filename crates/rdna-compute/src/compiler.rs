// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compile HIP kernels to code objects (.hsaco) via hipcc.
//! Supports pre-compiled .hsaco blobs for deployment without ROCm SDK.

use hip_bridge::HipResult;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

/// Copy .hsaco and .hash files from the persistent install location (cold)
/// into the tmpfs hot path. Used once at KernelCompiler startup to seed the
/// hot path after reboot (when /tmp gets cleared) without forcing a full
/// recompile. Returns on first IO failure without rolling back — the caller
/// falls back to reading from the cold dir directly.
///
/// Skip rule: if the hot dir already has BOTH a .hsaco AND a matching .hash
/// for this kernel, that pair was JIT-validated against the current source
/// (the .hash file is only written after a successful compile()), so it must
/// NOT be overwritten by a potentially-stale cold blob. Without this guard,
/// a cold blob whose size differs from the hot one (e.g. checked-in
/// kernels/compiled/<arch>/foo.hsaco produced by an older ROCm or a stale
/// source revision) silently downgrades the freshly-JIT'd hot blob on every
/// process startup. We saw this on gfx906 wave64 FP16 hybrid kernels: same
/// source, same hipcc, but the cold blob ran ~2× slower than the hot one.
fn seed_hot_from_cold(cold: &Path, hot: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(hot)?;
    for entry in std::fs::read_dir(cold)? {
        let entry = entry?;
        let src = entry.path();
        let ext = src.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext != "hsaco" && ext != "hash" {
            continue;
        }
        let name = match src.file_name() {
            Some(n) => n,
            None => continue,
        };
        let dst = hot.join(name);

        // Don't clobber a JIT-validated hot pair. A .hash is only written by
        // a successful KernelCompiler::compile() against the current source,
        // so if both .hsaco AND .hash exist in hot, that pair is the source
        // of truth — keep it regardless of size.
        let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if !stem.is_empty() {
            let hot_hsaco = hot.join(format!("{stem}.hsaco"));
            let hot_hash = hot.join(format!("{stem}.hash"));
            if hot_hsaco.exists() && hot_hash.exists() {
                continue;
            }
        }

        // Otherwise: skip if destination already exists with the same size.
        // We don't compare mtime because std::fs::copy doesn't preserve it —
        // the destination mtime is the copy time, which is always later than
        // the src mtime after an update. `hipfire update` wipes both dirs
        // before re-copy, so a same-size dst without a paired .hash is a
        // fresh seed from this install. Different size means an install
        // pulled in an updated cold blob and we should refresh hot to match.
        if let (Ok(s_meta), Ok(d_meta)) = (std::fs::metadata(&src), std::fs::metadata(&dst)) {
            if s_meta.len() == d_meta.len() {
                continue;
            }
        }
        std::fs::copy(&src, &dst)?;
    }
    Ok(())
}

/// Cache-key version. Bump when the kernel ABI or hipcc invocation changes in a
/// way that makes previously-cached `.hsaco` blobs incompatible, to force a clean
/// recompile instead of loading a stale "invalid device image".
const KERNEL_CACHE_ABI: u32 = 2;

/// Compiles HIP kernel sources to code objects, with caching.
/// Tries pre-compiled blobs first (kernels/compiled/{arch}/), falls back to hipcc.
pub struct KernelCompiler {
    cache_dir: PathBuf,
    arch: String,
    compiled: HashMap<String, PathBuf>,
    precompiled_dir: Option<PathBuf>,
    has_hipcc: bool,
    pub extra_flags: String,
    /// Exact gfx1151 JIT module names compiled in CU mode. This is an
    /// experiment boundary for per-code-object admission; other arches and
    /// unlisted modules retain their existing WGP-mode code objects.
    gfx1151_cumode_modules: HashSet<String>,
    /// Toolchain fingerprint (hipcc --version first line). Folded into the cache
    /// hash so blobs built by a different compiler/ROCm don't get reused across
    /// builds sharing one `.hipfire_kernels` dir (the "invalid device image" trap).
    toolchain_id: String,
    /// Resolved device-compiler binary. Bare "hipcc" when PATH provides it;
    /// otherwise an absolute path discovered under a resolved ROCm root.
    hipcc_bin: std::path::PathBuf,
    /// ROCM_PATH handed to the spawned compiler so it can find its own LLVM.
    /// None when the environment already sets it.
    rocm_env_root: Option<std::path::PathBuf>,
}

impl KernelCompiler {
    pub fn new(arch: &str, extra_flags: String) -> HipResult<Self> {
        // Cache (hot path) defaults to $CWD/.hipfire_kernels so parallel
        // worktrees/agents on the same machine don't clobber each other's
        // JIT'd .hsaco blobs. /tmp was shared state: two daemons from
        // different git states wrote the same {name}.hsaco path and
        // thrashed each other's hash sidecars. $CWD isolation fixes that.
        // End-user / CI can pin the old location back via
        // HIPFIRE_KERNEL_CACHE=/tmp/hipfire_kernels if tmpfs speed matters.
        // Per-arch keying matters for hetero (gfx906 + gfx1031 in one process):
        // without it, both arches would race for the same `{name}.hsaco` path,
        // surviving correctness via the source+arch hash check but thrashing
        // recompiles every cross-arch interleaving. Path layout matches the
        // pre-compiled `kernels/compiled/{arch}/` install dir + the already-
        // documented `.hipfire_kernels/{arch}/{name}.hsaco` shape in
        // docs/perf-checkpoints/2026-05-04-gfx906-mmq-junroll.md.
        let cache_root = std::env::var_os("HIPFIRE_KERNEL_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".hipfire_kernels"));
        let cache_dir = cache_root.join(arch);
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("failed to create cache dir: {e}"))
        })?;

        // Probe for pre-compiled kernels: exe-relative → CWD-relative → ~/.hipfire/bin/
        let precompiled_dir = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
            .map(|dir| dir.join("kernels").join("compiled").join(arch))
            .filter(|p| p.is_dir())
            .or_else(|| {
                let cwd_path = PathBuf::from("kernels/compiled").join(arch);
                if cwd_path.is_dir() {
                    Some(cwd_path)
                } else {
                    None
                }
            })
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| {
                        PathBuf::from(h)
                            .join(".hipfire/bin/kernels/compiled")
                            .join(arch)
                    })
                    .filter(|p| p.is_dir())
            });

        // Seed the tmpfs hot path from the persistent install location. /tmp
        // dies on reboot but the install blobs don't, so first-daemon-after-
        // boot copies them in. Subsequent daemons see a warm /tmp and skip
        // this. Copy is incremental — only copies files not already present
        // (or with stale hash) to avoid churn when both locations agree.
        // `hipfire update` wipes BOTH /tmp and the install dir, so after an
        // update + restart we get a fully-fresh re-seed.
        // cache_dir is already arch-keyed; the hot dir IS the cache dir.
        let hot_dir = cache_dir.clone();
        if let Some(ref cold) = precompiled_dir {
            if let Err(e) = seed_hot_from_cold(cold, &hot_dir) {
                eprintln!(
                    "  hot-path seed failed at {} ({e}) — falling back to install dir reads",
                    hot_dir.display()
                );
            }
        }
        // Prefer the hot-path (tmpfs) dir when it exists and has contents.
        // This is what the `compile()` lookup uses from here on.
        let effective_precompiled = if hot_dir.is_dir()
            && std::fs::read_dir(&hot_dir)
                .map(|mut it| {
                    it.any(|e| {
                        e.map(|e| e.path().extension().map(|x| x == "hsaco").unwrap_or(false))
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        {
            Some(hot_dir.clone())
        } else {
            precompiled_dir.clone()
        };

        if let Some(ref dir) = effective_precompiled {
            eprintln!("  pre-compiled kernels: {}", dir.display());
        }
        let precompiled_dir = effective_precompiled;

        // Probe for hipcc once at init, not per-kernel. Capture its version line
        // as a toolchain fingerprint for the cache hash (Fix #1).
        //
        // PATH stays authoritative so an explicit `module load` / PATH choice
        // still wins. The resolved-root fallback is strictly additive: it only
        // engages when PATH has no hipcc at all, which is the common
        // "ROCm installed under /opt/rocm/core-<ver> but not on PATH" case that
        // previously degraded silently to has_hipcc=false.
        let hipcc_bin: std::path::PathBuf = if Command::new("hipcc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            "hipcc".into()
        } else {
            hipfire_config::rocm::tool("hipcc").unwrap_or_else(|| "hipcc".into())
        };
        // hipcc resolves its LLVM as $ROCM_PATH/lib/llvm/bin/clang++ and
        // ROCM_PATH defaults to /opt/rocm. On a root elsewhere the probe below
        // still succeeds while every real compile fails, so hand the child the
        // selected compiler's installation root unless the operator already
        // set one that matches.
        let rocm_env_root = hipfire_config::rocm::compiler_env_root(&hipcc_bin);

        // HIPFIRE_NO_DEVICE_COMPILER=1 makes the engine behave as if no device
        // compiler exists, so pre-compiled blobs are used verbatim instead of
        // being recompiled locally.
        //
        // This has to be an explicit switch. It used to be achievable by
        // removing the compiler from PATH, because the probe was a bare
        // `Command::new("hipcc")`. Resolved-root discovery (added so a ROCm
        // rooted outside /opt/rocm still works) finds the compiler regardless
        // of PATH, which silently defeated PATH-shadowing and made
        // "pin these exact code objects" quietly recompile instead. Verified:
        // a pinned cross-machine run reported 0 recompiles while actually
        // executing locally-built blobs.
        let compiler_disabled = hipfire_config::developer_var_os("HIPFIRE_NO_DEVICE_COMPILER")
            .is_some_and(|v| v != "0" && !v.is_empty());

        let hipcc_out = if compiler_disabled {
            eprintln!(
                "  HIPFIRE_NO_DEVICE_COMPILER set — treating the device compiler as absent; \
                 pre-compiled blobs will be used verbatim"
            );
            None
        } else {
            let mut probe = Command::new(&hipcc_bin);
            probe.arg("--version");
            if let Some(ref root) = rocm_env_root {
                probe.env("ROCM_PATH", root);
            }
            probe.output().ok()
        };
        let has_hipcc = hipcc_out
            .as_ref()
            .map(|o| o.status.success())
            .unwrap_or(false);
        let toolchain_id = hipcc_out
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string()
            })
            .unwrap_or_default();

        let gfx1151_cumode_modules = if arch == "gfx1151" {
            hipfire_config::developer_var("HIPFIRE_GFX1151_CUMODE_MODULES")
                .unwrap_or_default()
                .split(|character: char| {
                    character == ',' || character == ';' || character.is_whitespace()
                })
                .filter(|module| !module.is_empty())
                .map(str::to_owned)
                .collect::<HashSet<_>>()
        } else {
            HashSet::new()
        };
        if !gfx1151_cumode_modules.is_empty() {
            let mut modules = gfx1151_cumode_modules.iter().cloned().collect::<Vec<_>>();
            modules.sort();
            eprintln!("  gfx1151 CU-mode modules: {}", modules.join(","));
        }

        Ok(Self {
            cache_dir,
            arch: arch.to_string(),
            compiled: HashMap::new(),
            precompiled_dir,
            has_hipcc,
            hipcc_bin,
            rocm_env_root,
            extra_flags,
            gfx1151_cumode_modules,
            toolchain_id,
        })
    }

    /// Returns a reference to all compiled kernel paths (name → .hsaco path).
    pub fn compiled_kernels(&self) -> &HashMap<String, PathBuf> {
        &self.compiled
    }

    /// Register `func_name` as an additional key pointing at an already-compiled
    /// artifact. `compiled_kernels()` is keyed by MODULE name, but the retained-PM4
    /// capture resolves a launched FUNCTION name; for arch-variant kernels (e.g.
    /// `gemv_hfq4g256_residual`, whose module is `gemv_hfq4g256_residual_rdna3` on
    /// RDNA3) the two differ, so without this alias the capture cannot find the
    /// owning `.hsaco` and the retained replay route fails closed. Additive and
    /// idempotent: never overwrites an existing key, so module-name lookups and
    /// the default-arch (module == func) case are unchanged.
    pub fn register_func_artifact(&mut self, func_name: &str, path: PathBuf) {
        self.compiled.entry(func_name.to_string()).or_insert(path);
    }

    fn module_flags(&self, name: &str) -> Vec<String> {
        if self.arch == "gfx1151" && self.gfx1151_cumode_modules.contains(name) {
            vec!["-mcumode".to_owned()]
        } else {
            Vec::new()
        }
    }

    fn cache_hash(&self, name: &str, source: &str) -> String {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        self.arch.hash(&mut hasher);
        self.extra_flags.hash(&mut hasher);
        let module_flags = self.module_flags(name);
        if !module_flags.is_empty() {
            "module-flags-v1".hash(&mut hasher);
            module_flags.hash(&mut hasher);
        }
        self.toolchain_id.hash(&mut hasher);
        KERNEL_CACHE_ABI.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Compile a HIP kernel source string. Returns path to .hsaco file.
    /// Tries pre-compiled blob first (with hash validation), falls back to hipcc.
    pub fn compile(&mut self, name: &str, source: &str) -> HipResult<&Path> {
        if self.compiled.contains_key(name) {
            return Ok(&self.compiled[name]);
        }

        // Hash source + arch + flags + toolchain + ABI for cache validation (used by
        // both pre-compiled and runtime paths). Flags and toolchain matter: identical
        // source compiled with different hipcc flags / ROCm versions yields a different
        // .hsaco, and reusing the wrong one surfaces as "device kernel image is invalid".
        let module_flags = self.module_flags(name);
        let src_hash = self.cache_hash(name, source);

        // Try pre-compiled .hsaco first, validating with a .hash sidecar file.
        // If hash is missing/mismatched AND hipcc is available, prefer recompilation.
        // If hipcc is unavailable (packaged install), use the blob as-is.
        // See: https://github.com/warpfront/hipfire/issues/2
        if let Some(ref dir) = self.precompiled_dir {
            let precompiled = dir.join(format!("{name}.hsaco"));
            let hash_file = dir.join(format!("{name}.hash"));
            if precompiled.exists() {
                let hash_ok = hash_file.exists() && {
                    let stored = std::fs::read_to_string(&hash_file).unwrap_or_default();
                    stored.trim() == src_hash
                };
                if hash_ok {
                    self.compiled.insert(name.to_string(), precompiled);
                    return Ok(&self.compiled[name]);
                }
                // No valid hash — only reject if hipcc can recompile
                if !self.has_hipcc {
                    eprintln!(
                        "  WARNING: {name}: using UNVALIDATED pre-compiled blob (hipcc unavailable)"
                    );
                    eprintln!(
                        "           Output may be incorrect. Install ROCm SDK or rebuild blobs with matching hashes."
                    );
                    self.compiled.insert(name.to_string(), precompiled);
                    return Ok(&self.compiled[name]);
                }
                eprintln!("  {name}: pre-compiled blob has no hash file, recompiling");
            }
        }

        // Fall back to runtime compilation via hipcc
        let src_path = self.cache_dir.join(format!("{name}.hip"));
        let obj_path = self.cache_dir.join(format!("{name}.hsaco"));
        let hash_path = self.cache_dir.join(format!("{name}.hash"));

        let cache_valid = obj_path.exists()
            && hash_path.exists()
            && std::fs::read_to_string(&hash_path).unwrap_or_default() == src_hash;

        if !cache_valid {
            Self::hipcc_compile(
                &self.hipcc_bin,
                self.rocm_env_root.as_deref(),
                &self.arch,
                &src_path,
                &obj_path,
                name,
                source,
                &self.extra_flags,
                &module_flags,
            )?;
            let _ = std::fs::write(&hash_path, &src_hash);
        }

        // Ensure precompiled dir has valid hash + blob (writeback from cache or fresh compile)
        if let Some(ref dir) = self.precompiled_dir {
            let pre_hash = dir.join(format!("{name}.hash"));
            let pre_valid = pre_hash.exists() && {
                let stored = std::fs::read_to_string(&pre_hash).unwrap_or_default();
                stored.trim() == src_hash
            };
            if !pre_valid {
                let pre_hsaco = dir.join(format!("{name}.hsaco"));
                let _ = std::fs::copy(&obj_path, &pre_hsaco);
                let _ = std::fs::write(&pre_hash, &src_hash);
            }
        }

        self.compiled.insert(name.to_string(), obj_path);
        Ok(&self.compiled[name])
    }

    /// Force a fresh hipcc recompile, evicting any cached / pre-compiled / seeded
    /// blob for `name` first. Self-heals a `.hsaco` the driver rejects as an invalid
    /// device image (a stale cross-build or cross-toolchain blob sitting in a shared
    /// `.hipfire_kernels` cache). Returns the path to the freshly built object.
    pub(crate) fn recompile(&mut self, name: &str, source: &str) -> HipResult<PathBuf> {
        self.compiled.remove(name);
        let _ = std::fs::remove_file(self.cache_dir.join(format!("{name}.hsaco")));
        let _ = std::fs::remove_file(self.cache_dir.join(format!("{name}.hash")));
        if let Some(ref dir) = self.precompiled_dir {
            let _ = std::fs::remove_file(dir.join(format!("{name}.hsaco")));
            let _ = std::fs::remove_file(dir.join(format!("{name}.hash")));
        }
        if !self.has_hipcc {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("{name}: cached kernel image invalid and hipcc unavailable to recompile"),
            ));
        }
        // Cache + blob are now gone → compile() takes the fresh hipcc path.
        self.compile(name, source)?;
        Ok(self.compiled[name].clone())
    }

    /// Extract per-kernel hipcc flags from magic comments in the source.
    /// The marker must be the dominant content of a comment line — i.e. a
    /// line whose non-whitespace starts with `//` followed (possibly after
    /// more whitespace) by `HIPFIRE_COMPILER_FLAGS:`. Flags after the colon
    /// are split on whitespace and appended to the hipcc invocation.
    /// Lines that merely *mention* the tag in prose (e.g. in a docstring
    /// explaining how to use it) are ignored, so we don't accidentally turn
    /// documentation into command-line arguments.
    fn per_kernel_flags(source: &str) -> Vec<String> {
        const TAG: &str = "HIPFIRE_COMPILER_FLAGS:";
        let mut out = Vec::new();
        for line in source.lines() {
            let trimmed = line.trim_start();
            let after_slashes = match trimmed.strip_prefix("//") {
                Some(rest) => rest.trim_start(),
                None => continue,
            };
            if let Some(rest) = after_slashes.strip_prefix(TAG) {
                for tok in rest.split_whitespace() {
                    out.push(tok.to_string());
                }
            }
        }
        out
    }

    /// On Windows, convert a path containing spaces to its 8.3 short-path
    /// form (e.g. `C:\Program Files\AMD\ROCm\6.4\include` to
    /// `C:\PROGRA~1\AMD\ROCm\6.4\include`) so it can be embedded as a single
    /// argv element to hipcc.bat without being split by the inner clang.exe
    /// re-tokenisation. Falls back to the original path on any error or on
    /// non-Windows hosts. Reported as #82.
    #[cfg(target_os = "windows")]
    fn win_short_path_if_needed(p: &str) -> String {
        if !p.contains(' ') {
            return p.to_string();
        }
        // Use cmd.exe's `for %A in (LONG) do echo %~sA` to ask the OS for the
        // 8.3 alias. Subprocess approach avoids pulling in a winapi crate dep
        // for this single call site.
        let out = Command::new("cmd")
            .raw_arg("/c")
            .raw_arg(&format!("for %A in (\"{}\") do @echo %~sA", p))
            .output();
        match out {
            Ok(o) if o.status.success() => {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if !s.is_empty() && !s.contains(' ') {
                    s
                } else {
                    p.to_string()
                }
            }
            _ => p.to_string(),
        }
    }

    /// No-op on non-Windows: POSIX argv handling preserves embedded spaces
    /// and ROCm's standard `/opt/rocm/include` has no spaces anyway.
    #[cfg(not(target_os = "windows"))]
    fn win_short_path_if_needed(p: &str) -> String {
        p.to_string()
    }

    /// Core hipcc argv: genco/arch/O3, then passthrough, then -o out src.
    fn direct_hipcc_args(
        arch: &str,
        src_path: &Path,
        obj_path: &Path,
        passthrough: Vec<String>,
    ) -> Vec<String> {
        let mut args: Vec<String> = vec![
            "--genco".into(),
            format!("--offload-arch={arch}"),
            "-O3".into(),
        ];
        args.extend(passthrough);
        args.push("-o".into());
        args.push(obj_path.to_str().unwrap().into());
        args.push(src_path.to_str().unwrap().into());
        args
    }

    /// Run hipcc for a single kernel. Shared by compile() and compile_batch().
    fn hipcc_compile(
        hipcc_bin: &Path,
        rocm_env_root: Option<&Path>,
        arch: &str,
        src_path: &Path,
        obj_path: &Path,
        name: &str,
        source: &str,
        extra_flags: &str,
        module_flags: &[String],
    ) -> HipResult<()> {
        std::fs::write(src_path, source).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("failed to write kernel source: {e}"))
        })?;
        let _ = std::fs::remove_file(obj_path);

        let extra = extra_flags;
        let per_kernel = Self::per_kernel_flags(source);
        let mut passthrough: Vec<String> = Vec::new();
        // Some hipcc installs (notably V620's CachyOS build of ROCm 7.2) do not
        // auto-inject the HIP include path, so `#include <hip/hip_runtime.h>`
        // fails with "file not found". Add well-known candidates as -I flags;
        // existence-checked so wrong paths on other distros don't leak in.
        let hip_path = std::env::var("HIP_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        for candidate in [
            format!("{hip_path}/include"),
            "/opt/rocm/include".to_string(),
        ] {
            if Path::new(&candidate).join("hip/hip_runtime.h").exists() {
                // Windows hipcc (hipcc.bat) re-tokenises its argv on the inner
                // clang.exe command line WITHOUT preserving quoting around
                // embedded spaces, so an include path inside `Program Files`
                // gets split at the space and clang sees the half before the
                // split. Convert to the 8.3 short-path form (e.g.
                // C:\PROGRA~1\AMD\ROCm\6.4\include) which contains no spaces.
                // Reported in #82.
                let resolved = Self::win_short_path_if_needed(&candidate);
                passthrough.push(format!("-I{resolved}"));
                break;
            }
        }
        for flag in extra.split_whitespace() {
            passthrough.push(flag.to_string());
        }
        for flag in module_flags {
            passthrough.push(flag.clone());
        }
        for flag in &per_kernel {
            passthrough.push(flag.clone());
        }
        if !module_flags.is_empty() || !per_kernel.is_empty() {
            let flags = module_flags
                .iter()
                .chain(per_kernel.iter())
                .cloned()
                .collect::<Vec<_>>();
            eprintln!("  {name}: per-kernel flags: {}", flags.join(" "));
        }

        let args = Self::direct_hipcc_args(arch, src_path, obj_path, passthrough);

        let mut cmd = Command::new(hipcc_bin);
        cmd.args(&args);
        if let Some(root) = rocm_env_root {
            cmd.env("ROCM_PATH", root);
        }
        let output = cmd
            .output()
            .map_err(|e| hip_bridge::HipError::new(0, &format!("failed to run hipcc: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(hip_bridge::HipError::new(
                0,
                &format!("hipcc compilation failed for {name}:\n{stderr}"),
            ));
        }
        Ok(())
    }

    /// Compile multiple kernels in parallel. Returns paths to .hsaco files.
    /// Kernels already compiled or cached are skipped.
    pub fn compile_batch(&mut self, kernels: &[(&str, &str)]) -> HipResult<()> {
        // Partition into already-done vs needs-work
        let mut to_compile: Vec<(
            String,
            String,
            String,
            PathBuf,
            PathBuf,
            PathBuf,
            Vec<String>,
        )> = Vec::new();

        for &(name, source) in kernels {
            if self.compiled.contains_key(name) {
                continue;
            }

            let module_flags = self.module_flags(name);
            let src_hash = self.cache_hash(name, source);

            // Check precompiled with valid hash
            if let Some(ref dir) = self.precompiled_dir {
                let precompiled = dir.join(format!("{name}.hsaco"));
                let hash_file = dir.join(format!("{name}.hash"));
                if precompiled.exists() {
                    let hash_ok = hash_file.exists() && {
                        let stored = std::fs::read_to_string(&hash_file).unwrap_or_default();
                        stored.trim() == src_hash
                    };
                    if hash_ok {
                        self.compiled.insert(name.to_string(), precompiled);
                        continue;
                    }
                    if !self.has_hipcc {
                        self.compiled.insert(name.to_string(), precompiled);
                        continue;
                    }
                }
            }

            // Check temp cache
            let obj_path = self.cache_dir.join(format!("{name}.hsaco"));
            let hash_path = self.cache_dir.join(format!("{name}.hash"));
            let src_path = self.cache_dir.join(format!("{name}.hip"));

            let cache_valid = obj_path.exists()
                && hash_path.exists()
                && std::fs::read_to_string(&hash_path).unwrap_or_default() == src_hash;

            if cache_valid {
                // Writeback to precompiled dir if missing
                if let Some(ref dir) = self.precompiled_dir {
                    let pre_hash = dir.join(format!("{name}.hash"));
                    let pre_valid = pre_hash.exists() && {
                        let stored = std::fs::read_to_string(&pre_hash).unwrap_or_default();
                        stored.trim() == src_hash
                    };
                    if !pre_valid {
                        let pre_hsaco = dir.join(format!("{name}.hsaco"));
                        let _ = std::fs::copy(&obj_path, &pre_hsaco);
                        let _ = std::fs::write(&pre_hash, &src_hash);
                    }
                }
                self.compiled.insert(name.to_string(), obj_path);
                continue;
            }

            to_compile.push((
                name.to_string(),
                source.to_string(),
                src_hash,
                src_path,
                obj_path,
                hash_path,
                module_flags,
            ));
        }

        if to_compile.is_empty() {
            return Ok(());
        }

        let n = to_compile.len();
        eprintln!("  compiling {n} kernels in parallel...");
        let arch = self.arch.clone();
        let precompiled_dir = self.precompiled_dir.clone();
        let hipcc_bin = self.hipcc_bin.clone();
        let rocm_env_root = self.rocm_env_root.clone();

        // Shared counter so parallel threads can report "[i/N] name" as each one
        // completes. Ordering follows completion (not launch) — matches the pace
        // of hipcc finishing.
        let done = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Spawn hipcc in parallel threads
        let results: Vec<_> = to_compile
            .into_iter()
            .map(
                |(name, source, src_hash, src_path, obj_path, hash_path, module_flags)| {
                    let arch = arch.clone();
                    let precompiled_dir = precompiled_dir.clone();
                    let hipcc_bin = hipcc_bin.clone();
                    let rocm_env_root = rocm_env_root.clone();
                    let extra_flags = self.extra_flags.clone();
                    let done = std::sync::Arc::clone(&done);
                    let handle = thread::spawn(move || {
                        let result = Self::hipcc_compile(
                            &hipcc_bin,
                            rocm_env_root.as_deref(),
                            &arch,
                            &src_path,
                            &obj_path,
                            &name,
                            &source,
                            &extra_flags,
                            &module_flags,
                        );
                        if result.is_ok() {
                            let _ = std::fs::write(&hash_path, &src_hash);
                            // Write back to precompiled dir
                            if let Some(ref dir) = precompiled_dir {
                                let pre_hash = dir.join(format!("{name}.hash"));
                                let pre_hsaco = dir.join(format!("{name}.hsaco"));
                                let _ = std::fs::copy(&obj_path, &pre_hsaco);
                                let _ = std::fs::write(&pre_hash, &src_hash);
                            }
                        }
                        let i = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        let marker = if result.is_ok() { "✓" } else { "✗" };
                        eprintln!("  [{i:>3}/{n}] {marker} {name}");
                        (name, obj_path, result)
                    });
                    handle
                },
            )
            .collect();

        let mut errors = Vec::new();
        for handle in results {
            let (name, obj_path, result) = handle.join().unwrap();
            match result {
                Ok(()) => {
                    self.compiled.insert(name, obj_path);
                }
                Err(e) => errors.push(e),
            }
        }
        eprintln!("  done ({n} kernels).");

        if let Some(e) = errors.into_iter().next() {
            return Err(e);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_compiler(extra_flags: &str, toolchain_id: &str) -> KernelCompiler {
        KernelCompiler {
            cache_dir: PathBuf::from(".test-cache"),
            arch: "gfx1151".to_string(),
            compiled: HashMap::new(),
            precompiled_dir: None,
            has_hipcc: false,
            extra_flags: extra_flags.to_string(),
            gfx1151_cumode_modules: HashSet::new(),
            toolchain_id: toolchain_id.to_string(),
            hipcc_bin: PathBuf::from("hipcc"),
            rocm_env_root: None,
        }
    }

    #[test]
    fn cache_abi_invalidates_pre_radiowave_compiler_entries() {
        assert_eq!(KERNEL_CACHE_ABI, 2);
    }

    #[test]
    fn cache_hash_includes_flags_and_toolchain() {
        let source = "__global__ void kernel() {}";
        let base = test_compiler("", "hipcc 7.2").cache_hash("kernel", source);
        let flags_changed = test_compiler("-mllvm -amdgpu-enable-flat-scratch=false", "hipcc 7.2")
            .cache_hash("kernel", source);
        let toolchain_changed = test_compiler("", "hipcc 7.3").cache_hash("kernel", source);

        assert_ne!(
            base, flags_changed,
            "cache key must change when hipcc flags change"
        );
        assert_ne!(
            base, toolchain_changed,
            "cache key must change when hipcc toolchain changes"
        );
    }

    #[test]
    fn direct_hipcc_args_keep_rocm_as_the_default_compiler() {
        let args = KernelCompiler::direct_hipcc_args(
            "gfx1100",
            Path::new("kernel.hip"),
            Path::new("kernel.hsaco"),
            vec!["-I/opt/rocm/include".to_owned()],
        );

        assert_eq!(
            args,
            vec![
                "--genco",
                "--offload-arch=gfx1100",
                "-O3",
                "-I/opt/rocm/include",
                "-o",
                "kernel.hsaco",
                "kernel.hip",
            ]
        );
        assert!(
            args.iter().all(|arg| !arg.contains("RADIOWAVE")),
            "the product compiler must not inject Radiowave"
        );
    }

    #[test]
    fn gfx1151_cumode_is_module_exact_and_cache_keyed() {
        let source = "__global__ void kernel() {}";
        let control = test_compiler("", "hipcc 7.2");
        let mut candidate = test_compiler("", "hipcc 7.2");
        candidate
            .gfx1151_cumode_modules
            .insert("selected".to_owned());

        assert_eq!(candidate.module_flags("selected"), vec!["-mcumode"]);
        assert!(candidate.module_flags("other").is_empty());
        assert_ne!(
            control.cache_hash("selected", source),
            candidate.cache_hash("selected", source)
        );
        assert_eq!(
            control.cache_hash("other", source),
            candidate.cache_hash("other", source)
        );

        candidate.arch = "gfx1100".to_owned();
        assert!(candidate.module_flags("selected").is_empty());
    }
}
