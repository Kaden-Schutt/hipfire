// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compile HIP kernels to code objects (.hsaco) via hipcc.
//! Supports pre-compiled .hsaco blobs for deployment without ROCm SDK.

use hip_bridge::HipResult;
use radiowave::{CodeObjectCertification, ExistingCodeObjectRequest, SchedulerProfile};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

/// True when `a` and `b` name the same filesystem object. Lexical equality is
/// checked first; if both paths exist, canonicalization also equates aliases
/// (symlinks, `.` / `..` segments) so two-directory logic never treats one
/// underlying dir as two targets.
fn same_path(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => false,
    }
}

/// True when `path` is an existing nonempty regular file (usable HSACO blob).
fn nonempty_blob(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
}

/// Unique same-dir staging token (pid + counter + nanos) for parallel-safe temps.
fn unique_token() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}_{n}_{nanos}", std::process::id())
}

/// Validated pair: nonempty regular HSACO **and** matching hash sidecar.
/// A hash alone never certifies a missing/empty/directory blob. A blob alone
/// is never treated as hash-validated.
fn pair_valid(hsaco: &Path, hash: &Path, src_hash: &str) -> bool {
    if !nonempty_blob(hsaco) {
        return false;
    }
    hash.exists() && {
        let stored = std::fs::read_to_string(hash).unwrap_or_default();
        stored.trim() == src_hash
    }
}

/// Transactionally publish a blob+hash pair into `dest_dir`.
///
/// Staging uses same-directory temps so `rename` is atomic on one volume.
/// Order: stage blob → verify nonempty → stage hash → **invalidate old hash**
/// → publish blob → publish hash. A crash mid-publish never leaves a hash that
/// certifies a missing/wrong-generation blob. `force` replaces even when the
/// destination pair already matches `src_hash`.
///
/// When `src_blob` is already the destination blob path, only the hash sidecar
/// is repaired (still via stage+rename). Never copies a path onto itself.
fn publish_pair(
    dest_dir: &Path,
    name: &str,
    src_blob: &Path,
    src_hash: &str,
    force: bool,
) -> Result<(), String> {
    let dest_hsaco = dest_dir.join(format!("{name}.hsaco"));
    let dest_hash = dest_dir.join(format!("{name}.hash"));

    if !force && pair_valid(&dest_hsaco, &dest_hash, src_hash) {
        return Ok(());
    }

    if !nonempty_blob(src_blob) {
        return Err(format!("{name}: source blob missing or empty"));
    }

    let token = unique_token();
    let tmp_hsaco = dest_dir.join(format!(".{name}.{token}.hsaco.tmp"));
    let tmp_hash = dest_dir.join(format!(".{name}.{token}.hash.tmp"));
    let cleanup = || {
        let _ = std::fs::remove_file(&tmp_hsaco);
        let _ = std::fs::remove_file(&tmp_hash);
    };

    let same_blob = same_path(src_blob, &dest_hsaco);

    if !same_blob {
        if let Err(e) = std::fs::copy(src_blob, &tmp_hsaco) {
            cleanup();
            return Err(format!("{name}: blob stage failed ({e})"));
        }
        if !nonempty_blob(&tmp_hsaco) {
            cleanup();
            return Err(format!("{name}: staged blob empty"));
        }
    }

    if let Err(e) = std::fs::write(&tmp_hash, src_hash) {
        cleanup();
        return Err(format!("{name}: hash stage failed ({e})"));
    }

    // Invalidate old hash before publishing a replacement blob so an old
    // sidecar can never certify a new generation.
    let _ = std::fs::remove_file(&dest_hash);

    if !same_blob {
        if let Err(e) = std::fs::rename(&tmp_hsaco, &dest_hsaco) {
            cleanup();
            return Err(format!("{name}: blob publish failed ({e})"));
        }
    }

    if let Err(e) = std::fs::rename(&tmp_hash, &dest_hash) {
        let _ = std::fs::remove_file(&tmp_hash);
        return Err(format!("{name}: hash publish failed ({e})"));
    }

    Ok(())
}

/// Publish only a nonempty blob into `dest_dir`, removing any orphan hash so
/// it cannot certify the copied content. Used for hashless cold seed fallback.
fn publish_blob_only(dest_dir: &Path, name: &str, src_blob: &Path) -> Result<(), String> {
    let dest_hsaco = dest_dir.join(format!("{name}.hsaco"));
    let dest_hash = dest_dir.join(format!("{name}.hash"));

    if !nonempty_blob(src_blob) {
        return Err(format!("{name}: source blob missing or empty"));
    }
    if same_path(src_blob, &dest_hsaco) {
        // Drop orphan hash that would certify this blob under the wrong key.
        let _ = std::fs::remove_file(&dest_hash);
        return Ok(());
    }

    let token = unique_token();
    let tmp_hsaco = dest_dir.join(format!(".{name}.{token}.hsaco.tmp"));
    if let Err(e) = std::fs::copy(src_blob, &tmp_hsaco) {
        let _ = std::fs::remove_file(&tmp_hsaco);
        return Err(format!("{name}: blob stage failed ({e})"));
    }
    if !nonempty_blob(&tmp_hsaco) {
        let _ = std::fs::remove_file(&tmp_hsaco);
        return Err(format!("{name}: staged blob empty"));
    }

    // Remove hash before blob publish so it cannot certify the new content.
    let _ = std::fs::remove_file(&dest_hash);

    if let Err(e) = std::fs::rename(&tmp_hsaco, &dest_hsaco) {
        let _ = std::fs::remove_file(&tmp_hsaco);
        return Err(format!("{name}: blob publish failed ({e})"));
    }
    Ok(())
}

/// Copy a validated hot `.hsaco` into the persistent cold install dir, then
/// publish the matching `.hash` only after the blob is published.
///
/// When `force` is false, a cold pair that is already valid for `src_hash` is
/// left alone. When `force` is true (post-recompile after a driver-rejected
/// image), the cold pair is replaced even if its hash already matched.
///
/// Optional: never fails the caller. Never copies a path onto itself.
/// Unwritable cold dirs only warn — hot runtime remains valid.
fn writeback_cold(name: &str, obj_path: &Path, src_hash: &str, cold: &Path, force: bool) {
    if let Err(e) = publish_pair(cold, name, obj_path, src_hash, force) {
        eprintln!(
            "  WARNING: {name}: cold writeback failed ({e}); \
             runtime blob remains valid in the hot cache"
        );
    }
}

/// Seed hot from cold using **complete kernel pairs**, never independent
/// entry-by-entry copies.
///
/// - Orphan cold hashes and empty/directory blobs are ignored.
/// - Skip only when hot is already a nonempty pair with the **same** cold hash.
/// - Pair replacement publishes blob then hash as a unit (via `publish_pair`).
/// - Hashless nonempty cold may seed only when hot has no validated pair, and
///   any orphan hot hash is removed before the blob is published.
fn seed_hot_from_cold(cold: &Path, hot: &Path) -> std::io::Result<()> {
    if same_path(cold, hot) {
        return Ok(());
    }
    std::fs::create_dir_all(hot)?;

    for entry in std::fs::read_dir(cold)? {
        let entry = entry?;
        let src = entry.path();
        let ext = src.extension().and_then(|s| s.to_str()).unwrap_or("");
        // Blob-centric: never iterate orphan .hash files as seed units.
        if ext != "hsaco" {
            continue;
        }
        if !nonempty_blob(&src) {
            continue;
        }
        let stem = match src.file_stem().and_then(|s| s.to_str()) {
            Some(s) if !s.is_empty() && !s.starts_with('.') => s,
            _ => continue,
        };

        let cold_hash_path = cold.join(format!("{stem}.hash"));
        let hot_hsaco = hot.join(format!("{stem}.hsaco"));
        let hot_hash = hot.join(format!("{stem}.hash"));

        let cold_hash = std::fs::read_to_string(&cold_hash_path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        match &cold_hash {
            Some(h) => {
                // Complete cold pair: skip only when hot already matches that hash.
                if pair_valid(&hot_hsaco, &hot_hash, h) {
                    continue;
                }
                if let Err(e) = publish_pair(hot, stem, &src, h, true) {
                    return Err(std::io::Error::other(e));
                }
            }
            None => {
                // Hashless cold fallback: only when hot has no validated pair
                // (nonempty blob + any hash sidecar written by a prior compile).
                let hot_has_validated_pair = nonempty_blob(&hot_hsaco) && hot_hash.exists();
                if hot_has_validated_pair {
                    continue;
                }
                if let Err(e) = publish_blob_only(hot, stem, &src) {
                    return Err(std::io::Error::other(e));
                }
            }
        }
    }
    Ok(())
}

/// Cache-key version. Bump when the kernel ABI or hipcc invocation changes in a
/// way that makes previously-cached `.hsaco` blobs incompatible, to force a clean
/// recompile instead of loading a stale "invalid device image".
const KERNEL_CACHE_ABI: u32 = 3;

/// Compiles HIP kernel sources to code objects, with caching.
///
/// Two-directory contract:
/// - **hot** (`cache_dir`): per-process JIT workspace (`.hipfire_kernels/{arch}`
///   by default). Preferred for lookup after seeding.
/// - **cold** (`cold_dir`): persistent install location
///   (`kernels/compiled/{arch}` under the installed binary). Writeback target
///   so install-time / post-update recompiles refresh the blobs that seed the
///   next process. Distinct from hot; must not be collapsed into the lookup dir.
///
/// `precompiled_dir` is the lookup view: hot when it has blobs, otherwise cold.
/// Tries that first (with hash validation), falls back to hipcc into hot, then
/// writeback to cold.
pub struct KernelCompiler {
    cache_dir: PathBuf,
    arch: String,
    compiled: HashMap<String, PathBuf>,
    /// Lookup directory for pre-compiled blobs (hot when seeded, else cold).
    precompiled_dir: Option<PathBuf>,
    /// Persistent install dir (`kernels/compiled/{arch}`). Writeback target;
    /// kept even when lookup prefers the hot path.
    cold_dir: Option<PathBuf>,
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
    /// Override scheduler profile applied to all kernels when set via
    /// `HIPFIRE_SCHED_PROFILE`. `None` selects the per-kernel table.
    sched_profile_override: Option<SchedulerProfile>,
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

        // Seed the hot path from the persistent install location. Cold survives
        // reboots / process restarts; hot may not (or may be worktree-local).
        // Copy is incremental — only copies files not already present as a
        // JIT-validated hot pair (see seed_hot_from_cold skip rule).
        // Stale pairs are not wiped by update: hash mismatch in compile()
        // triggers hipcc and writeback refreshes cold in place.
        // cache_dir is already arch-keyed; the hot dir IS the cache dir.
        let cold_dir = precompiled_dir;
        let hot_dir = cache_dir.clone();
        if let Some(cold) = &cold_dir {
            if let Err(e) = seed_hot_from_cold(cold, &hot_dir) {
                eprintln!(
                    "  hot-path seed failed at {} ({e}) — falling back to install dir reads",
                    hot_dir.display()
                );
            }
        }
        // Prefer the hot-path dir when it exists and has contents.
        // This is what the `compile()` lookup uses from here on. Cold stays
        // on `cold_dir` for writeback even when lookup points at hot.
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
            cold_dir.clone()
        };

        if let Some(ref dir) = effective_precompiled {
            eprintln!("  pre-compiled kernels: {}", dir.display());
        }
        let precompiled_dir = effective_precompiled;

        // Probe for hipcc once at init, not per-kernel. Capture its version line
        // as a toolchain fingerprint for the cache hash (Fix #1).
        //
        // Resolve hipcc through the same selected ROCm root as the runtime and
        // headers. Probing a bare PATH hipcc first could pair a configured
        // ROCM_PATH from one version with another version's compiler.
        let resolved_hipcc = hipfire_config::rocm::tool("hipcc");
        let hipcc_bin = resolved_hipcc
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from("hipcc"));
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
        } else if resolved_hipcc.is_none() {
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

        let sched_profile_override = hipfire_config::developer_var("HIPFIRE_SCHED_PROFILE")
            .ok()
            .and_then(|value| SchedulerProfile::parse(&value));

        Ok(Self {
            cache_dir,
            arch: arch.to_string(),
            compiled: HashMap::new(),
            precompiled_dir,
            cold_dir,
            has_hipcc,
            hipcc_bin,
            rocm_env_root,
            extra_flags,
            gfx1151_cumode_modules,
            toolchain_id,
            sched_profile_override,
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

    fn module_flags_for(
        arch: &str,
        name: &str,
        gfx1151_cumode_modules: &HashSet<String>,
    ) -> Vec<String> {
        // Radiowave-selected spill-free RM2/BV6 schedule.
        if arch == "gfx1100" && name == "gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_bt" {
            vec!["-mllvm".to_owned(), "-misched=gcn-iterative-ilp".to_owned()]
        } else if arch == "gfx1151" && gfx1151_cumode_modules.contains(name) {
            vec!["-mcumode".to_owned()]
        } else {
            Vec::new()
        }
    }

    fn module_flags(&self, name: &str) -> Vec<String> {
        let mut flags = Self::module_flags_for(&self.arch, name, &self.gfx1151_cumode_modules);
        let profile = self.scheduler_profile_for(name);
        flags.extend(profile.llvm_args().iter().map(|flag| flag.to_string()));
        flags
    }

    /// Select the scheduler profile for a kernel. The per-kernel table is the
    /// primary source; `HIPFIRE_SCHED_PROFILE` overrides everything when set.
    ///
    /// **The table is deliberately empty.** A kernel was
    /// briefly mapped to `MemoryClause` on the strength of static counters
    /// (r4 VGPR 120 -> 109, max consecutive VMEM 10 -> 16, zero spills, and
    /// strictly better than every ILP profile). Measured device-side with HIP
    /// events on gfx1201 that produced **no benefit** — multirow medians over
    /// three runs were 22.32 / 22.88 / 23.56 us under `memory-clause` against
    /// 22.32 / 22.56 / 22.56 us under `default`, overlapping at the fast end
    /// and differing by less than the run-to-run spread.
    ///
    /// Better static counters are not a performance result. A non-default
    /// compile policy with no measured win is carried complexity, so the
    /// selection was withdrawn and the boring baseline kept. The mechanism
    /// stays, and is worth keeping: the manifest now reports the real profile
    /// instead of a hardcoded `Default`, the profile participates in the cache
    /// hash so changing it cannot silently reuse a stale object, and
    /// `HIPFIRE_SCHED_PROFILE` makes a sweep cheap.
    ///
    /// If you add an entry here, gate it on a device-side measurement, not on
    /// VGPR or clause counts. Note also that `IterativeIlp` spills 138 VGPRs on
    /// the GL multirow kernel and must never be selected for it.
    fn scheduler_profile_for(&self, name: &str) -> SchedulerProfile {
        if let Some(profile) = self.sched_profile_override {
            return profile;
        }
        let _ = name;
        SchedulerProfile::Default
    }

    /// Single hashing sequence for all kernel cache keys. Every caller must
    /// go through this so KERNEL_CACHE_ABI or field order changes cannot
    /// silently diverge between JIT and packaging paths.
    fn hash_parts(
        source: &str,
        arch: &str,
        extra_flags: &str,
        module_flags: &[String],
        toolchain_id: &str,
        scheduler_profile: SchedulerProfile,
    ) -> String {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        arch.hash(&mut hasher);
        extra_flags.hash(&mut hasher);
        if !module_flags.is_empty() {
            "module-flags-v1".hash(&mut hasher);
            module_flags.hash(&mut hasher);
        }
        toolchain_id.hash(&mut hasher);
        scheduler_profile.as_str().hash(&mut hasher);
        KERNEL_CACHE_ABI.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    fn cache_hash(&self, name: &str, source: &str) -> String {
        Self::hash_parts(
            source,
            &self.arch,
            &self.extra_flags,
            &self.module_flags(name),
            &self.toolchain_id,
            self.scheduler_profile_for(name),
        )
    }

    /// Hash a kernel as a **packaged** blob will be validated on a
    /// compiler-free runtime. Identical to `cache_hash` except
    /// `toolchain_id` is empty — the value `KernelCompiler::new` computes
    /// when `hipcc --version` cannot be run (`unwrap_or_default()`).
    ///
    /// Dropping `toolchain_id` for shipped blobs is sound: that field exists
    /// to stop different builds sharing one mutable `.hipfire_kernels` dir from
    /// reusing each other's blobs (the "invalid device image" trap). A baked,
    /// read-only, shipped-as-a-unit `kernels/compiled/{arch}` directory has no
    /// such sharing, and the key still binds source + arch + flags + ABI.
    pub fn packaging_hash(&self, name: &str, source: &str) -> String {
        Self::hash_parts(
            source,
            &self.arch,
            &self.extra_flags,
            &self.module_flags(name),
            "",
            self.scheduler_profile_for(name),
        )
    }

    /// Static packaging hash for the `hipfire-kernel-hash` binary: same inputs
    /// as `packaging_hash` but without needing a `KernelCompiler` instance.
    /// Resolves `gfx1151` CU-mode modules from the environment so the key
    /// matches what a runtime constructed with the same env would compute.
    pub fn packaging_hash_for(arch: &str, name: &str, source: &str, extra_flags: &str) -> String {
        let gfx1151_cumode_modules = if arch == "gfx1151" {
            hipfire_config::developer_var("HIPFIRE_GFX1151_CUMODE_MODULES")
                .unwrap_or_default()
                .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
                .filter(|m| !m.is_empty())
                .map(str::to_owned)
                .collect::<HashSet<_>>()
        } else {
            HashSet::new()
        };
        let module_flags = Self::module_flags_for(arch, name, &gfx1151_cumode_modules);
        let sched_profile_override = hipfire_config::developer_var("HIPFIRE_SCHED_PROFILE")
            .ok()
            .and_then(|value| SchedulerProfile::parse(&value));
        let scheduler_profile = sched_profile_override.unwrap_or(SchedulerProfile::Default);
        Self::hash_parts(
            source,
            arch,
            extra_flags,
            &module_flags,
            "",
            scheduler_profile,
        )
    }

    /// Persistent install dir for writeback. None when no cold location was
    /// probed, or when it coincides with the hot cache (nothing extra to sync).
    fn writeback_dir(&self) -> Option<&Path> {
        self.cold_dir
            .as_deref()
            .filter(|d| !same_path(d, self.cache_dir.as_path()))
    }

    fn compiler_unavailable_error(&self, name: &str, action: &str) -> hip_bridge::HipError {
        let tried = vec![self.hipcc_bin.display().to_string()];
        let guidance =
            hipfire_config::rocm::resolution_failure("the ROCm HIP compiler (hipcc)", &tried);
        hip_bridge::HipError::new(
            0,
            &format!("{name}: {action}, but hipcc is unavailable.\n{guidance}"),
        )
    }

    /// Attach a hash-bound Radiowave inspection to an already validated
    /// Hipfire kernel-cache artifact. Retained PM4 consumes this adjacent
    /// manifest for cache-scope and argument-effect proofs; without it the
    /// replay path correctly fails closed to broad acquires and unknown
    /// resource effects even when the code object itself is already safe.
    ///
    /// Certification never recompiles or replaces the HSACO. It is best-effort
    /// because packaged installs may intentionally have no ROCm inspection
    /// tools; those installs retain the conservative replay policy.
    fn ensure_radiowave_certification(&self, name: &str, artifact: &Path) {
        if !self.has_hipcc {
            return;
        }
        let manifest = artifact.with_extension("radiowave.json");
        let already_valid = std::fs::read(artifact)
            .ok()
            .zip(std::fs::read_to_string(&manifest).ok())
            .is_some_and(|(code, encoded)| {
                CodeObjectCertification::from_json(&code, &encoded).is_ok()
            });
        if already_valid {
            return;
        }

        let source = artifact.with_extension("hip");
        if !source.is_file() {
            return;
        }
        let request = ExistingCodeObjectRequest::new(&source, artifact, &self.arch)
            .hipcc(&self.hipcc_bin)
            .command(vec![
                self.hipcc_bin.display().to_string(),
                "hipfire-kernel-cache".to_owned(),
                name.to_owned(),
            ])
            .manifest(&manifest)
            .scheduler_profile(self.scheduler_profile_for(name));
        if let Err(error) = radiowave::Compiler.certify_existing(&request) {
            eprintln!(
                "  WARNING: {name}: Radiowave could not certify existing cache artifact: {error}"
            );
        }
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

        // Try pre-compiled .hsaco first. A hit requires a nonempty regular blob;
        // matching hash certifies it. Hashless/mismatched nonempty blobs may be
        // used only when hipcc is unavailable (packaged install), with warning.
        // See: https://github.com/warpfront/hipfire/issues/2
        if let Some(dir) = &self.precompiled_dir {
            let precompiled = dir.join(format!("{name}.hsaco"));
            let hash_file = dir.join(format!("{name}.hash"));
            if pair_valid(&precompiled, &hash_file, &src_hash) {
                // Validated pair still refreshes a distinct cold install dir when
                // that pair is missing, empty, or stale.
                if let Some(cold) = self.writeback_dir().map(Path::to_path_buf) {
                    writeback_cold(name, &precompiled, &src_hash, &cold, false);
                }
                self.ensure_radiowave_certification(name, &precompiled);
                self.compiled.insert(name.to_string(), precompiled);
                return Ok(&self.compiled[name]);
            }
            if nonempty_blob(&precompiled) {
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

        // Fall back to runtime compilation via hipcc into the hot cache.
        let obj_path = self.cache_dir.join(format!("{name}.hsaco"));
        let hash_path = self.cache_dir.join(format!("{name}.hash"));

        if pair_valid(&obj_path, &hash_path, &src_hash) {
            if let Some(dir) = self.writeback_dir() {
                writeback_cold(name, &obj_path, &src_hash, dir, false);
            }
            self.ensure_radiowave_certification(name, &obj_path);
            self.compiled.insert(name.to_string(), obj_path);
            return Ok(&self.compiled[name]);
        }

        if !self.has_hipcc {
            // Nonempty unvalidated hot blob may still be usable on packaged installs.
            if nonempty_blob(&obj_path) {
                eprintln!(
                    "  WARNING: {name}: using UNVALIDATED pre-compiled blob (hipcc unavailable)"
                );
                eprintln!(
                    "           Output may be incorrect. Install ROCm SDK or rebuild blobs with matching hashes."
                );
                self.compiled.insert(name.to_string(), obj_path);
                return Ok(&self.compiled[name]);
            }
            return Err(
                self.compiler_unavailable_error(name, "no usable cached kernel image exists")
            );
        }

        Self::hipcc_compile_publish(
            &self.hipcc_bin,
            self.rocm_env_root.as_deref(),
            &self.arch,
            &self.cache_dir,
            name,
            source,
            &src_hash,
            &self.extra_flags,
            &module_flags,
        )?;
        self.ensure_radiowave_certification(name, &obj_path);

        // Ensure cold install dir has valid hash + blob (writeback from hot).
        // Must target cold_dir, not the lookup view: when hot is preferred for
        // lookup, precompiled_dir == cache_dir and writing there would leave
        // the persistent install stale. Failures warn only (see writeback_cold).
        if let Some(dir) = self.writeback_dir() {
            writeback_cold(name, &obj_path, &src_hash, dir, false);
        }

        self.compiled.insert(name.to_string(), obj_path);
        Ok(&self.compiled[name])
    }

    /// Force a fresh hipcc recompile after the driver rejects a loaded image.
    ///
    /// Does **not** pre-delete hot/cold artifacts (safe when hot==cold or the
    /// dirs are aliased). Bypasses all lookup and invokes transactional hipcc
    /// publication directly. Cold is force-synced only after success. A failed
    /// hipcc leaves prior durable pairs intact.
    pub(crate) fn recompile(&mut self, name: &str, source: &str) -> HipResult<PathBuf> {
        if !self.has_hipcc {
            return Err(self.compiler_unavailable_error(
                name,
                "the cached kernel image is invalid and must be recompiled",
            ));
        }

        self.compiled.remove(name);

        let module_flags = self.module_flags(name);
        let src_hash = self.cache_hash(name, source);
        let obj_path = self.cache_dir.join(format!("{name}.hsaco"));

        Self::hipcc_compile_publish(
            &self.hipcc_bin,
            self.rocm_env_root.as_deref(),
            &self.arch,
            &self.cache_dir,
            name,
            source,
            &src_hash,
            &self.extra_flags,
            &module_flags,
        )?;
        self.ensure_radiowave_certification(name, &obj_path);

        // Force-sync distinct cold only after a successful fresh hot compile.
        if let Some(dir) = self.writeback_dir().map(Path::to_path_buf) {
            writeback_cold(name, &obj_path, &src_hash, &dir, true);
        }

        self.compiled.insert(name.to_string(), obj_path.clone());
        Ok(obj_path)
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
        // clang ≥19 compresses offload bundles by default. HIP's loader accepts the
        // compressed container; the public HSA reader Redline uses does not, so a
        // CCOB blob demotes every retained route to plain HIP.
        let mut args: Vec<String> = vec![
            "--genco".into(),
            format!("--offload-arch={arch}"),
            "-O3".into(),
            "--no-offload-compress".into(),
        ];
        args.extend(passthrough);
        args.push("-o".into());
        args.push(obj_path.to_str().unwrap().into());
        args.push(src_path.to_str().unwrap().into());
        args
    }

    /// Build hipcc passthrough flags (include paths, extras, module, per-kernel).
    fn hipcc_passthrough(
        name: &str,
        source: &str,
        extra_flags: &str,
        module_flags: &[String],
    ) -> Vec<String> {
        let per_kernel = Self::per_kernel_flags(source);
        let mut passthrough: Vec<String> = Vec::new();
        // Some hipcc installs (notably V620's CachyOS build of ROCm 7.2) do not
        // auto-inject the HIP include path, so `#include <hip/hip_runtime.h>`
        // fails with "file not found". Add well-known candidates as -I flags;
        // existence-checked so wrong paths on other distros don't leak in.
        //
        // The include path comes from the same selected root as hipcc and the
        // runtime. Do not separately prepend ambient HIP_PATH or /opt/rocm:
        // either could belong to another side-by-side ROCm version.
        //
        // Tell the device compiler where ROCm is rather than relying on its
        // self-relative probe. That probe needs marker files (`bin/.hipVersion`
        // and friends) which some installs omit; without them clang reports
        // "cannot find ROCm device library" and, once headers are supplied by
        // -I alone, still resolves device math like `rsqrtf` against the host
        // `/usr/include/math.h` instead of the HIP runtime wrapper. Only a
        // complete root is passed, so a shim directory can never be handed to
        // clang as authoritative; on a conventional install this resolves to
        // the same path clang would have found by itself.
        let selected_root = hipfire_config::rocm::root();
        if let Some(root) = selected_root.as_ref() {
            if hipfire_config::rocm::is_complete_root(&root) {
                let root = root.to_string_lossy();
                passthrough.push(format!("--rocm-path={root}"));
                passthrough.push(format!("--hip-path={root}"));
            }
        }
        if let Some(candidate) =
            selected_root.map(|root| root.join("include").to_string_lossy().into_owned())
        {
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
            }
        }
        for flag in extra_flags.split_whitespace() {
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
        passthrough
    }

    /// Low-level hipcc invocation writing the object to `obj_path` (may be a temp).
    /// Does not mutate hash sidecars. Shared by the transactional publisher.
    fn hipcc_compile_to(
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

        let passthrough = Self::hipcc_passthrough(name, source, extra_flags, module_flags);
        let args = Self::direct_hipcc_args(arch, src_path, obj_path, passthrough);

        let mut cmd = Command::new(hipcc_bin);
        cmd.args(&args);
        if let Some(root) = rocm_env_root {
            cmd.env("ROCM_PATH", root);
        }
        let output = cmd.output().map_err(|e| {
            let tried = vec![hipcc_bin.display().to_string()];
            let guidance =
                hipfire_config::rocm::resolution_failure("the ROCm HIP compiler (hipcc)", &tried);
            hip_bridge::HipError::new(
                0,
                &format!("failed to run {}: {e}\n{guidance}", hipcc_bin.display()),
            )
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // A missing hip_runtime.h is a missing PACKAGE, not a missing flag.
            // clang reports it as a bare "file not found" that names neither the
            // ROCm root it searched nor the component that would supply it, so
            // spell both out rather than leaving the user to guess.
            let hint = if stderr.contains("hip/hip_runtime.h") {
                let root = hipfire_config::rocm::root();
                let guidance = hipfire_config::rocm::install_guidance()
                    .into_iter()
                    .map(|line| format!("\n  {line}"))
                    .collect::<String>();
                format!(
                    "\n\nThe HIP headers are not installed under the ROCm root ({}).\n\
                     A working hipcc does not imply them — they ship in a separate\n\
                     package.{guidance}\n\
                     Run `hipfire diag` for the per-root component inventory.",
                    root.map(|r| r.display().to_string())
                        .unwrap_or_else(|| "none resolved".into()),
                )
            } else {
                String::new()
            };
            return Err(hip_bridge::HipError::new(
                0,
                &format!("hipcc compilation failed for {name}:\n{stderr}{hint}"),
            ));
        }
        Ok(())
    }

    /// Compile via hipcc into a unique same-dir temp object, then transactionally
    /// publish the final hot blob+hash. Never writes final hashes directly.
    /// On any failure, temps are cleaned and prior hot pairs remain intact.
    /// Hash publication errors are **not** ignored.
    fn hipcc_compile_publish(
        hipcc_bin: &Path,
        rocm_env_root: Option<&Path>,
        arch: &str,
        cache_dir: &Path,
        name: &str,
        source: &str,
        src_hash: &str,
        extra_flags: &str,
        module_flags: &[String],
    ) -> HipResult<()> {
        let src_path = cache_dir.join(format!("{name}.hip"));
        let final_hsaco = cache_dir.join(format!("{name}.hsaco"));
        let token = unique_token();
        let tmp_obj = cache_dir.join(format!(".{name}.{token}.hsaco.tmp"));

        let cleanup_tmp = || {
            let _ = std::fs::remove_file(&tmp_obj);
        };

        let compile_result = Self::hipcc_compile_to(
            hipcc_bin,
            rocm_env_root,
            arch,
            &src_path,
            &tmp_obj,
            name,
            source,
            extra_flags,
            module_flags,
        );

        if let Err(e) = compile_result {
            cleanup_tmp();
            return Err(e);
        }

        if !nonempty_blob(&tmp_obj) {
            cleanup_tmp();
            return Err(hip_bridge::HipError::new(
                0,
                &format!("hipcc produced empty object for {name}"),
            ));
        }

        // Publish blob then hash via the shared transactional helper.
        // On failure, leave prior durable pair intact (temps cleaned inside).
        if let Err(e) = publish_pair(cache_dir, name, &tmp_obj, src_hash, true) {
            cleanup_tmp();
            // If publish moved the temp already, final may exist without hash —
            // that is the safe invalid state (blob without certifying hash).
            let _ = std::fs::remove_file(&tmp_obj);
            return Err(hip_bridge::HipError::new(
                0,
                &format!("failed to publish compiled kernel {name}: {e}"),
            ));
        }

        // publish_pair copies then leaves the source temp behind when src != dest.
        cleanup_tmp();

        // Defensive: final pair must be valid after publication.
        let final_hash = cache_dir.join(format!("{name}.hash"));
        if !pair_valid(&final_hsaco, &final_hash, src_hash) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("post-publish pair invalid for {name}"),
            ));
        }

        Ok(())
    }

    /// Compile multiple kernels in parallel. Returns paths to .hsaco files.
    /// Kernels already compiled or cached are skipped.
    pub fn compile_batch(&mut self, kernels: &[(&str, &str)]) -> HipResult<()> {
        // Partition into already-done vs needs-work
        let mut to_compile: Vec<(String, String, String, Vec<String>)> = Vec::new();

        for &(name, source) in kernels {
            if self.compiled.contains_key(name) {
                continue;
            }

            let module_flags = self.module_flags(name);
            let src_hash = self.cache_hash(name, source);

            // Check precompiled with valid pair (nonempty blob + matching hash).
            if let Some(dir) = &self.precompiled_dir {
                let precompiled = dir.join(format!("{name}.hsaco"));
                let hash_file = dir.join(format!("{name}.hash"));
                if pair_valid(&precompiled, &hash_file, &src_hash) {
                    // Same opportunistic cold sync as compile(): a hot hit must
                    // not skip writeback when the install dir is still stale.
                    if let Some(cold) = self.writeback_dir().map(Path::to_path_buf) {
                        writeback_cold(name, &precompiled, &src_hash, &cold, false);
                    }
                    self.compiled.insert(name.to_string(), precompiled);
                    continue;
                }
                if nonempty_blob(&precompiled) && !self.has_hipcc {
                    // Mirror compile()'s two-line warning so a later compile()
                    // short-circuit via `compiled` cannot suppress it.
                    eprintln!(
                        "  WARNING: {name}: using UNVALIDATED pre-compiled blob (hipcc unavailable)"
                    );
                    eprintln!(
                        "           Output may be incorrect. Install ROCm SDK or rebuild blobs with matching hashes."
                    );
                    self.compiled.insert(name.to_string(), precompiled);
                    continue;
                }
            }

            // Check temp cache with the same pair-valid helper.
            let obj_path = self.cache_dir.join(format!("{name}.hsaco"));
            let hash_path = self.cache_dir.join(format!("{name}.hash"));

            if pair_valid(&obj_path, &hash_path, &src_hash) {
                if let Some(dir) = self.writeback_dir() {
                    writeback_cold(name, &obj_path, &src_hash, dir, false);
                }
                self.compiled.insert(name.to_string(), obj_path);
                continue;
            }

            if !self.has_hipcc {
                if nonempty_blob(&obj_path) {
                    eprintln!(
                        "  WARNING: {name}: using UNVALIDATED pre-compiled blob (hipcc unavailable)"
                    );
                    eprintln!(
                        "           Output may be incorrect. Install ROCm SDK or rebuild blobs with matching hashes."
                    );
                    self.compiled.insert(name.to_string(), obj_path);
                    continue;
                }
                return Err(
                    self.compiler_unavailable_error(name, "no usable cached kernel image exists")
                );
            }

            to_compile.push((name.to_string(), source.to_string(), src_hash, module_flags));
        }

        if to_compile.is_empty() {
            return Ok(());
        }

        let n = to_compile.len();
        eprintln!("  compiling {n} kernels in parallel...");
        let arch = self.arch.clone();
        let cache_dir = self.cache_dir.clone();
        let writeback_dir = self.writeback_dir().map(|p| p.to_path_buf());
        let hipcc_bin = self.hipcc_bin.clone();
        let rocm_env_root = self.rocm_env_root.clone();

        // Shared counter so parallel threads can report "[i/N] name" as each one
        // completes. Ordering follows completion (not launch) — matches the pace
        // of hipcc finishing.
        let done = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Spawn hipcc in parallel threads — each uses transactional publication.
        let results: Vec<_> = to_compile
            .into_iter()
            .map(|(name, source, src_hash, module_flags)| {
                let arch = arch.clone();
                let cache_dir = cache_dir.clone();
                let writeback_dir = writeback_dir.clone();
                let hipcc_bin = hipcc_bin.clone();
                let rocm_env_root = rocm_env_root.clone();
                let extra_flags = self.extra_flags.clone();
                let done = std::sync::Arc::clone(&done);
                let handle = thread::spawn(move || {
                    let result = Self::hipcc_compile_publish(
                        &hipcc_bin,
                        rocm_env_root.as_deref(),
                        &arch,
                        &cache_dir,
                        &name,
                        &source,
                        &src_hash,
                        &extra_flags,
                        &module_flags,
                    );
                    if result.is_ok() {
                        let obj_path = cache_dir.join(format!("{name}.hsaco"));
                        // Write back to cold install dir (blob before hash).
                        if let Some(dir) = &writeback_dir {
                            writeback_cold(&name, &obj_path, &src_hash, dir, false);
                        }
                    }
                    let i = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    let marker = if result.is_ok() { "✓" } else { "✗" };
                    eprintln!("  [{i:>3}/{n}] {marker} {name}");
                    let obj_path = cache_dir.join(format!("{name}.hsaco"));
                    (name, obj_path, result)
                });
                handle
            })
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
            cold_dir: None,
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
        assert_eq!(KERNEL_CACHE_ABI, 3);
    }

    #[test]
    fn two_dir_writeback_stays_distinct() {
        // Lookup may prefer hot while cold remains the install writeback target.
        // Defends the directory-selection invariant without invoking hipcc.
        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = PathBuf::from("/tmp/hipfire_hot/gfx1201");
        c.precompiled_dir = Some(PathBuf::from("/tmp/hipfire_hot/gfx1201"));
        c.cold_dir = Some(PathBuf::from(
            "/home/user/.hipfire/bin/kernels/compiled/gfx1201",
        ));

        assert_eq!(
            c.writeback_dir(),
            Some(Path::new(
                "/home/user/.hipfire/bin/kernels/compiled/gfx1201"
            )),
            "writeback must target cold, not the hot lookup view"
        );

        // When cold coincides with hot there is nothing extra to sync.
        c.cold_dir = Some(c.cache_dir.clone());
        assert_eq!(c.writeback_dir(), None);
    }

    #[test]
    fn same_path_equates_lexical_identity() {
        let p = Path::new("/tmp/hipfire_hot/gfx1201");
        assert!(same_path(p, p));
        assert!(!same_path(
            Path::new("/tmp/hipfire_hot/gfx1201"),
            Path::new("/home/user/.hipfire/bin/kernels/compiled/gfx1201"),
        ));
    }

    fn temp_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "hipfire_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn pair_valid_requires_nonempty_blob() {
        let root = temp_root("pair_valid");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let hsaco = root.join("k.hsaco");
        let hash = root.join("k.hash");
        std::fs::write(&hash, "deadbeef").unwrap();

        // Missing blob.
        assert!(!pair_valid(&hsaco, &hash, "deadbeef"));

        // Empty blob.
        std::fs::write(&hsaco, b"").unwrap();
        assert!(!nonempty_blob(&hsaco));
        assert!(!pair_valid(&hsaco, &hash, "deadbeef"));

        // Nonempty blob + matching hash.
        std::fs::write(&hsaco, b"BLOB").unwrap();
        assert!(pair_valid(&hsaco, &hash, "deadbeef"));

        // Hash mismatch.
        assert!(!pair_valid(&hsaco, &hash, "other"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn writeback_cold_skips_self_copy_and_missing_src() {
        // No real dirs required: missing src → warn path, no panic.
        // Self-path equality short-circuits before copy.
        let missing = Path::new("/nonexistent/hipfire_writeback_test/kernel.hsaco");
        let cold = Path::new("/nonexistent/hipfire_writeback_test_cold");
        writeback_cold("kernel", missing, "deadbeef", cold, false);

        // Same path → no-op (must not attempt copy-onto-self).
        writeback_cold(
            "kernel",
            missing,
            "deadbeef",
            Path::new("/nonexistent"),
            false,
        );
        // The join cold/kernel.hsaco differs from missing above; exercise the
        // explicit same_path guard with identical paths:
        writeback_cold(
            "kernel",
            Path::new("/nonexistent/kernel.hsaco"),
            "deadbeef",
            Path::new("/nonexistent"),
            false,
        );
    }

    #[test]
    fn writeback_cold_restores_missing_or_empty_blob() {
        // Verified RED: matching cold hash alone must not skip writeback when
        // the HSACO is missing or empty.
        let root = temp_root("wb_missing_blob");
        let hot = root.join("hot");
        let cold = root.join("cold");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&hot).unwrap();
        std::fs::create_dir_all(&cold).unwrap();

        let name = "add_inplace";
        let src_hash = "abc123hash";
        let hot_hsaco = hot.join(format!("{name}.hsaco"));
        std::fs::write(&hot_hsaco, b"HOT_VALID_BLOB").unwrap();

        // Case A: hash present, blob missing.
        std::fs::write(cold.join(format!("{name}.hash")), src_hash).unwrap();
        assert!(!cold.join(format!("{name}.hsaco")).exists());
        writeback_cold(name, &hot_hsaco, src_hash, &cold, false);
        assert_eq!(
            std::fs::read(cold.join(format!("{name}.hsaco"))).unwrap(),
            b"HOT_VALID_BLOB"
        );
        assert_eq!(
            std::fs::read_to_string(cold.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            src_hash
        );

        // Case B: hash present, blob empty — still invalid, must restore.
        std::fs::write(cold.join(format!("{name}.hsaco")), b"").unwrap();
        writeback_cold(name, &hot_hsaco, src_hash, &cold, false);
        assert_eq!(
            std::fs::read(cold.join(format!("{name}.hsaco"))).unwrap(),
            b"HOT_VALID_BLOB"
        );

        // No leftover stage temps in cold.
        let leftovers: Vec<_> = std::fs::read_dir(&cold)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "stage temps must be cleaned up: {leftovers:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn writeback_cold_force_replaces_hash_matching_pair() {
        // After a driver-rejected image, force must overwrite cold even when
        // the source hash already matches.
        let root = temp_root("wb_force");
        let hot = root.join("hot");
        let cold = root.join("cold");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&hot).unwrap();
        std::fs::create_dir_all(&cold).unwrap();

        let name = "add_inplace";
        let src_hash = "samehash";
        let hot_hsaco = hot.join(format!("{name}.hsaco"));
        std::fs::write(&hot_hsaco, b"FRESH_HOT").unwrap();
        std::fs::write(cold.join(format!("{name}.hsaco")), b"OLD_REJECTED").unwrap();
        std::fs::write(cold.join(format!("{name}.hash")), src_hash).unwrap();

        // Non-force leaves a valid matching pair alone.
        writeback_cold(name, &hot_hsaco, src_hash, &cold, false);
        assert_eq!(
            std::fs::read(cold.join(format!("{name}.hsaco"))).unwrap(),
            b"OLD_REJECTED"
        );

        // Force replaces it.
        writeback_cold(name, &hot_hsaco, src_hash, &cold, true);
        assert_eq!(
            std::fs::read(cold.join(format!("{name}.hsaco"))).unwrap(),
            b"FRESH_HOT"
        );
        assert_eq!(
            std::fs::read_to_string(cold.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            src_hash
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn writeback_cold_failed_stage_preserves_prior_pair() {
        // Missing/empty hot source must not clobber an existing valid cold pair.
        let root = temp_root("wb_preserve");
        let cold = root.join("cold");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&cold).unwrap();

        let name = "add_inplace";
        let src_hash = "keepme";
        std::fs::write(cold.join(format!("{name}.hsaco")), b"PRIOR_COLD").unwrap();
        std::fs::write(cold.join(format!("{name}.hash")), src_hash).unwrap();

        let missing_hot = root.join("nope.hsaco");
        writeback_cold(name, &missing_hot, "newhash", &cold, true);

        assert_eq!(
            std::fs::read(cold.join(format!("{name}.hsaco"))).unwrap(),
            b"PRIOR_COLD",
            "failed force writeback must leave prior cold blob untouched"
        );
        assert_eq!(
            std::fs::read_to_string(cold.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            src_hash,
            "failed force writeback must leave prior cold hash untouched"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn validated_hot_lookup_writeback_refreshes_stale_cold() {
        // End-to-end: hot lookup is hash-valid so compile returns without hipcc,
        // but a distinct cold install pair is stale and must still be refreshed.
        let root = temp_root("hot_lookup_wb");
        let hot = root.join("hot");
        let cold = root.join("cold");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&hot).unwrap();
        std::fs::create_dir_all(&cold).unwrap();

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = hot.clone();
        c.precompiled_dir = Some(hot.clone());
        c.cold_dir = Some(cold.clone());
        c.has_hipcc = false;

        let name = "add_inplace";
        let source = "__global__ void add_inplace() {}";
        let src_hash = c.cache_hash(name, source);

        std::fs::write(hot.join(format!("{name}.hsaco")), b"HOT_BLOB_V1").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), &src_hash).unwrap();
        std::fs::write(cold.join(format!("{name}.hsaco")), b"OLD_COLD_BLOB").unwrap();
        std::fs::write(cold.join(format!("{name}.hash")), "stale").unwrap();

        let path = c
            .compile(name, source)
            .expect("validated hot lookup must succeed without hipcc");
        assert_eq!(path, &hot.join(format!("{name}.hsaco")));

        let cold_hash = std::fs::read_to_string(cold.join(format!("{name}.hash"))).unwrap();
        assert_eq!(
            cold_hash.trim(),
            src_hash,
            "stale cold hash must be refreshed from validated hot lookup"
        );
        assert_eq!(
            std::fs::read(cold.join(format!("{name}.hsaco"))).unwrap(),
            b"HOT_BLOB_V1",
            "cold blob must be copied from the validated hot pair"
        );
        // Hot pair must remain the source of truth (no self-clobber).
        assert_eq!(
            std::fs::read(hot.join(format!("{name}.hsaco"))).unwrap(),
            b"HOT_BLOB_V1"
        );
        assert_eq!(
            std::fs::read_to_string(hot.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            src_hash
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn validated_hot_lookup_restores_missing_cold_blob() {
        // End-to-end RED: hot hit + cold hash present + cold hsaco missing
        // must restore the cold blob (install daemon --precompile path).
        let root = temp_root("hot_lookup_missing_cold");
        let hot = root.join("hot");
        let cold = root.join("cold");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&hot).unwrap();
        std::fs::create_dir_all(&cold).unwrap();

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = hot.clone();
        c.precompiled_dir = Some(hot.clone());
        c.cold_dir = Some(cold.clone());
        c.has_hipcc = false;

        let name = "add_inplace";
        let source = "__global__ void add_inplace() {}";
        let src_hash = c.cache_hash(name, source);

        std::fs::write(hot.join(format!("{name}.hsaco")), b"HOT_BLOB_V1").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), &src_hash).unwrap();
        // Cold hash matches, but blob is gone (the verified RED fixture).
        std::fs::write(cold.join(format!("{name}.hash")), &src_hash).unwrap();

        let path = c
            .compile(name, source)
            .expect("validated hot lookup must succeed without hipcc");
        assert_eq!(path, &hot.join(format!("{name}.hsaco")));

        assert!(
            cold.join(format!("{name}.hsaco")).exists(),
            "missing cold hsaco must be restored from validated hot"
        );
        assert_eq!(
            std::fs::read(cold.join(format!("{name}.hsaco"))).unwrap(),
            b"HOT_BLOB_V1"
        );
        assert_eq!(
            std::fs::read_to_string(cold.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            src_hash
        );

        let _ = std::fs::remove_dir_all(&root);
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
                "--no-offload-compress",
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

    #[test]
    fn gfx1100_muse_rm_bt_uses_iterative_ilp_and_is_cache_keyed() {
        let source = "__global__ void kernel() {}";
        let module = "gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_bt";
        let mut gfx1100 = test_compiler("", "hipcc 7.2");
        gfx1100.arch = "gfx1100".to_owned();
        let control = test_compiler("", "hipcc 7.2");

        assert_eq!(
            gfx1100.module_flags(module),
            vec!["-mllvm".to_owned(), "-misched=gcn-iterative-ilp".to_owned(),]
        );
        assert!(gfx1100.module_flags("other").is_empty());
        assert!(control.module_flags(module).is_empty());
        assert_ne!(
            control.cache_hash(module, source),
            gfx1100.cache_hash(module, source)
        );
    }

    /// Write an executable fake hipcc that copies a canned blob to `-o` path,
    /// or exits nonzero when `fail` is true.
    fn install_fake_hipcc(bin_dir: &Path, canned_blob: &[u8], fail: bool) -> PathBuf {
        std::fs::create_dir_all(bin_dir).unwrap();
        let blob_path = bin_dir.join("canned.hsaco");
        std::fs::write(&blob_path, canned_blob).unwrap();
        let hipcc = bin_dir.join("fake_hipcc");
        let script = if fail {
            "#!/bin/sh\necho 'fake hipcc failure' >&2\nexit 1\n".to_string()
        } else {
            format!(
                "#!/bin/sh\nout=\"\"\nwhile [ $# -gt 0 ]; do\n  if [ \"$1\" = \"-o\" ]; then out=\"$2\"; shift 2; continue; fi\n  shift\ndone\ncp \"{}\" \"$out\"\n",
                blob_path.display()
            )
        };
        std::fs::write(&hipcc, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&hipcc).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&hipcc, perms).unwrap();
        }
        hipcc
    }

    fn leftover_temps(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| n.contains(".tmp"))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn seed_orphan_cold_hash_cannot_certify_stale_hot_blob() {
        // CRITICAL: orphan cold .hash must never be copied beside a stale hot
        // blob and make compile() accept it as current.
        let root = temp_root("seed_orphan_hash");
        let _ = std::fs::remove_dir_all(&root);
        let cold = root.join("cold");
        let hot = root.join("hot");
        std::fs::create_dir_all(&cold).unwrap();
        std::fs::create_dir_all(&hot).unwrap();

        let name = "k";
        let current = "current_hash";
        // Cold has only the current hash (missing blob — repair case).
        std::fs::write(cold.join(format!("{name}.hash")), current).unwrap();
        // Hot has a stale-but-loadable blob with no hash.
        std::fs::write(hot.join(format!("{name}.hsaco")), b"STALE_BLOB").unwrap();

        seed_hot_from_cold(&cold, &hot).unwrap();

        assert!(
            !hot.join(format!("{name}.hash")).exists(),
            "orphan cold hash must not be seeded beside a stale hot blob"
        );
        assert_eq!(
            std::fs::read(hot.join(format!("{name}.hsaco"))).unwrap(),
            b"STALE_BLOB"
        );
        // pair_valid must still reject (no matching hash on hot).
        assert!(!pair_valid(
            &hot.join(format!("{name}.hsaco")),
            &hot.join(format!("{name}.hash")),
            current
        ));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn seed_pair_replaces_mismatched_hot_pair_as_unit() {
        let root = temp_root("seed_pair_replace");
        let _ = std::fs::remove_dir_all(&root);
        let cold = root.join("cold");
        let hot = root.join("hot");
        std::fs::create_dir_all(&cold).unwrap();
        std::fs::create_dir_all(&hot).unwrap();

        let name = "k";
        std::fs::write(cold.join(format!("{name}.hsaco")), b"COLD_BLOB").unwrap();
        std::fs::write(cold.join(format!("{name}.hash")), "cold_hash").unwrap();
        std::fs::write(hot.join(format!("{name}.hsaco")), b"HOT_OLD").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), "hot_old_hash").unwrap();

        seed_hot_from_cold(&cold, &hot).unwrap();

        assert_eq!(
            std::fs::read(hot.join(format!("{name}.hsaco"))).unwrap(),
            b"COLD_BLOB"
        );
        assert_eq!(
            std::fs::read_to_string(hot.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            "cold_hash"
        );
        assert!(pair_valid(
            &hot.join(format!("{name}.hsaco")),
            &hot.join(format!("{name}.hash")),
            "cold_hash"
        ));
        assert!(
            leftover_temps(&hot).is_empty(),
            "no temp residue after pair seed: {:?}",
            leftover_temps(&hot)
        );

        // Matching hot pair must not be overwritten.
        std::fs::write(hot.join(format!("{name}.hsaco")), b"HOT_MATCH").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), "cold_hash").unwrap();
        seed_hot_from_cold(&cold, &hot).unwrap();
        assert_eq!(
            std::fs::read(hot.join(format!("{name}.hsaco"))).unwrap(),
            b"HOT_MATCH",
            "matching hot pair must be preserved"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn empty_or_directory_blob_rejected_without_hipcc_single_and_batch() {
        let root = temp_root("empty_dir_blob");
        let _ = std::fs::remove_dir_all(&root);
        let hot = root.join("hot");
        std::fs::create_dir_all(&hot).unwrap();

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = hot.clone();
        c.precompiled_dir = Some(hot.clone());
        c.has_hipcc = false;

        let name = "add_inplace";
        let source = "__global__ void add_inplace() {}";
        let src_hash = c.cache_hash(name, source);

        // Empty file with matching hash.
        std::fs::write(hot.join(format!("{name}.hsaco")), b"").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), &src_hash).unwrap();

        let err = c.compile(name, source).unwrap_err();
        assert!(
            err.to_string().contains("hipcc unavailable")
                || err.to_string().contains("no usable cached"),
            "empty blob must error without hipcc: {err}"
        );
        assert!(!c.compiled.contains_key(name));

        // Directory named like a blob with matching hash.
        c.compiled.clear();
        let _ = std::fs::remove_file(hot.join(format!("{name}.hsaco")));
        std::fs::create_dir(hot.join(format!("{name}.hsaco"))).unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), &src_hash).unwrap();

        let err = c.compile(name, source).unwrap_err();
        assert!(
            err.to_string().contains("hipcc unavailable")
                || err.to_string().contains("no usable cached"),
            "directory blob must error without hipcc: {err}"
        );

        // Batch path must agree.
        c.compiled.clear();
        let err = c.compile_batch(&[(name, source)]).unwrap_err();
        assert!(
            err.to_string().contains("hipcc unavailable")
                || err.to_string().contains("no usable cached"),
            "batch empty/dir blob must error without hipcc: {err}"
        );
        assert!(!c.compiled.contains_key(name));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_recompile_preserves_pair_when_hot_equals_cold() {
        let root = temp_root("recompile_hot_eq_cold");
        let _ = std::fs::remove_dir_all(&root);
        let shared = root.join("shared");
        std::fs::create_dir_all(&shared).unwrap();
        let hipcc = install_fake_hipcc(&root.join("bin"), b"NEW", true);

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = shared.clone();
        c.precompiled_dir = Some(shared.clone());
        c.cold_dir = Some(shared.clone());
        c.has_hipcc = true;
        c.hipcc_bin = hipcc;

        let name = "k";
        let source = "__global__ void k() {}";
        let src_hash = c.cache_hash(name, source);
        std::fs::write(shared.join(format!("{name}.hsaco")), b"PRIOR_PAIR").unwrap();
        std::fs::write(shared.join(format!("{name}.hash")), &src_hash).unwrap();

        let err = c.recompile(name, source).unwrap_err();
        assert!(err.to_string().contains("hipcc"), "{err}");

        assert_eq!(
            std::fs::read(shared.join(format!("{name}.hsaco"))).unwrap(),
            b"PRIOR_PAIR",
            "failed recompile must not delete hot==cold blob"
        );
        assert_eq!(
            std::fs::read_to_string(shared.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            src_hash,
            "failed recompile must not delete hot==cold hash"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_recompile_preserves_pair_when_dirs_aliased() {
        let root = temp_root("recompile_alias");
        let _ = std::fs::remove_dir_all(&root);
        let real = root.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let alias = root.join("alias");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real, &alias).unwrap();
        #[cfg(not(unix))]
        {
            let _ = std::fs::remove_dir_all(&root);
            return;
        }
        let hipcc = install_fake_hipcc(&root.join("bin"), b"NEW", true);

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = real.clone();
        c.precompiled_dir = Some(alias.clone());
        c.cold_dir = Some(alias.clone());
        c.has_hipcc = true;
        c.hipcc_bin = hipcc;

        let name = "k";
        let source = "__global__ void k() {}";
        let src_hash = c.cache_hash(name, source);
        std::fs::write(real.join(format!("{name}.hsaco")), b"ALIAS_PRIOR").unwrap();
        std::fs::write(real.join(format!("{name}.hash")), &src_hash).unwrap();

        let _ = c.recompile(name, source).unwrap_err();

        assert_eq!(
            std::fs::read(real.join(format!("{name}.hsaco"))).unwrap(),
            b"ALIAS_PRIOR"
        );
        assert_eq!(
            std::fs::read_to_string(real.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            src_hash
        );
        // writeback_dir must treat alias as same path → no distinct cold.
        assert_eq!(c.writeback_dir(), None);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_transactional_hipcc_leaves_old_hot_pair_intact() {
        let root = temp_root("tx_hipcc_fail");
        let _ = std::fs::remove_dir_all(&root);
        let hot = root.join("hot");
        std::fs::create_dir_all(&hot).unwrap();
        let hipcc = install_fake_hipcc(&root.join("bin"), b"NEW", true);

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = hot.clone();
        c.precompiled_dir = None;
        c.cold_dir = None;
        c.has_hipcc = true;
        c.hipcc_bin = hipcc;

        let name = "k";
        let source = "__global__ void k() {}";
        // Prior pair with a different hash so compile must attempt hipcc.
        std::fs::write(hot.join(format!("{name}.hsaco")), b"OLD_HOT").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), "old_hash").unwrap();

        let err = c.compile(name, source).unwrap_err();
        assert!(err.to_string().contains("hipcc"), "{err}");

        assert_eq!(
            std::fs::read(hot.join(format!("{name}.hsaco"))).unwrap(),
            b"OLD_HOT"
        );
        assert_eq!(
            std::fs::read_to_string(hot.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            "old_hash"
        );
        assert!(
            leftover_temps(&hot).is_empty(),
            "failed hipcc must clean temps: {:?}",
            leftover_temps(&hot)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn successful_publish_pair_has_no_temp_residue() {
        let root = temp_root("publish_ok");
        let _ = std::fs::remove_dir_all(&root);
        let dest = root.join("dest");
        let src_dir = root.join("src");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::create_dir_all(&src_dir).unwrap();

        let name = "k";
        let src_blob = src_dir.join("blob.hsaco");
        std::fs::write(&src_blob, b"FRESH_BLOB").unwrap();
        // Prior pair to replace.
        std::fs::write(dest.join(format!("{name}.hsaco")), b"OLD").unwrap();
        std::fs::write(dest.join(format!("{name}.hash")), "old").unwrap();

        publish_pair(&dest, name, &src_blob, "new_hash", true).unwrap();

        assert_eq!(
            std::fs::read(dest.join(format!("{name}.hsaco"))).unwrap(),
            b"FRESH_BLOB"
        );
        assert_eq!(
            std::fs::read_to_string(dest.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            "new_hash"
        );
        assert!(pair_valid(
            &dest.join(format!("{name}.hsaco")),
            &dest.join(format!("{name}.hash")),
            "new_hash"
        ));
        assert!(
            leftover_temps(&dest).is_empty(),
            "successful publish must leave no temps: {:?}",
            leftover_temps(&dest)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn successful_hipcc_publish_yields_matching_pair_no_temps() {
        let root = temp_root("hipcc_publish_ok");
        let _ = std::fs::remove_dir_all(&root);
        let hot = root.join("hot");
        std::fs::create_dir_all(&hot).unwrap();
        let hipcc = install_fake_hipcc(&root.join("bin"), b"COMPILED_OK", false);

        let mut c = test_compiler("", "hipcc 7.2");
        c.cache_dir = hot.clone();
        c.precompiled_dir = None;
        c.cold_dir = None;
        c.has_hipcc = true;
        c.hipcc_bin = hipcc;

        let name = "k";
        let source = "__global__ void k() {}";
        let src_hash = c.cache_hash(name, source);

        // Stale prior pair forces recompile path.
        std::fs::write(hot.join(format!("{name}.hsaco")), b"STALE").unwrap();
        std::fs::write(hot.join(format!("{name}.hash")), "stale").unwrap();

        let path = c.compile(name, source).expect("fake hipcc must succeed");
        assert_eq!(path, &hot.join(format!("{name}.hsaco")));
        assert_eq!(std::fs::read(path).unwrap(), b"COMPILED_OK");
        assert_eq!(
            std::fs::read_to_string(hot.join(format!("{name}.hash")))
                .unwrap()
                .trim(),
            src_hash
        );
        assert!(pair_valid(
            &hot.join(format!("{name}.hsaco")),
            &hot.join(format!("{name}.hash")),
            &src_hash
        ));
        assert!(
            leftover_temps(&hot).is_empty(),
            "successful hipcc publish must leave no temps: {:?}",
            leftover_temps(&hot)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn packaging_hash_validates_on_compiler_free_runtime() {
        // Proves that a blob+hash pair emitted by compile-kernels.sh in
        // packaging mode (toolchain_id = "") satisfies pair_valid for a
        // compiler-free runtime (has_hipcc = false, empty toolchain_id) so
        // compile() uses the silent validated path, not the UNVALIDATED
        // fallback.
        let root = temp_root("packaging_hash_validates");
        let _ = std::fs::remove_dir_all(&root);
        let precompiled = root.join("precompiled");
        std::fs::create_dir_all(&precompiled).unwrap();

        let name = "packaged_kernel";
        let source = "__global__ void packaged_kernel() {}";
        let arch = "gfx1201";

        // Builder has a real toolchain, but packaging must use empty id.
        let mut builder = test_compiler("", "hipcc 7.2");
        builder.arch = arch.to_string();
        let packaging_via_instance = builder.packaging_hash(name, source);
        let packaging_via_static =
            KernelCompiler::packaging_hash_for(arch, name, source, "");
        assert_eq!(
            packaging_via_instance, packaging_via_static,
            "instance and static packaging hashes must agree (same hash_parts)"
        );

        // Compiler-free runtime computes with empty toolchain_id.
        let mut runtime = test_compiler("", "");
        runtime.arch = arch.to_string();
        let runtime_hash = runtime.cache_hash(name, source);
        assert_eq!(
            packaging_via_instance, runtime_hash,
            "packaging hash must equal what a hipcc-free runtime computes"
        );
        // And must differ from a builder's normal cache_hash (which folds in toolchain).
        let builder_hash = builder.cache_hash(name, source);
        assert_ne!(
            packaging_via_instance, builder_hash,
            "packaging (empty toolchain) must differ from normal cache hash"
        );

        // Bake blob+hash as the script now does.
        let hsaco = precompiled.join(format!("{name}.hsaco"));
        let hash_file = precompiled.join(format!("{name}.hash"));
        std::fs::write(&hsaco, b"FAKE_HSACO_BLOB").unwrap();
        std::fs::write(&hash_file, &packaging_via_instance).unwrap();

        // Direct pair_valid check: what compile() checks first.
        assert!(
            pair_valid(&hsaco, &hash_file, &runtime_hash),
            "packaged pair must be pair_valid for compiler-free hash"
        );

        // End-to-end via KernelCompiler::compile with has_hipcc = false:
        // must take the validated branch (no UNVALIDATED warning path) and
        // return the precompiled blob.
        let mut c = test_compiler("", "");
        c.arch = arch.to_string();
        c.has_hipcc = false;
        c.cache_dir = root.join("hot");
        std::fs::create_dir_all(&c.cache_dir).unwrap();
        c.precompiled_dir = Some(precompiled.clone());
        c.cold_dir = Some(precompiled.clone());

        let path = c
            .compile(name, source)
            .expect("packaged blob must validate without hipcc");
        assert_eq!(path, &hsaco);
        assert!(c.compiled.contains_key(name));

        let _ = std::fs::remove_dir_all(&root);
    }
}
