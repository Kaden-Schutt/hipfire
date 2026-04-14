//! Compile HIP kernels to code objects (.hsaco) via hipcc.
//! Supports pre-compiled .hsaco blobs for deployment without ROCm SDK.

use hip_bridge::HipResult;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;

/// Copy .hsaco and .hash files from the persistent install location (cold)
/// into the tmpfs hot path. Used once at KernelCompiler startup to seed the
/// hot path after reboot (when /tmp gets cleared) without forcing a full
/// recompile. Per-file: only copy if destination is missing or stale (size
/// mismatch or older mtime). Returns on first IO failure without rolling
/// back — the caller falls back to reading from the cold dir directly.
fn seed_hot_from_cold(cold: &Path, hot: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(hot)?;
    for entry in std::fs::read_dir(cold)? {
        let entry = entry?;
        let src = entry.path();
        let ext = src.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext != "hsaco" && ext != "hash" { continue; }
        let name = match src.file_name() { Some(n) => n, None => continue };
        let dst = hot.join(name);
        // Skip if destination already exists with same size. We don't compare
        // mtime because std::fs::copy doesn't preserve it — the destination
        // mtime is the copy time, which is always later than the src mtime
        // after an update. `hipfire update` wipes both dirs before re-copy,
        // so a same-size dst is fresh-from-this-session's seed. Mid-session
        // a kernel source edit + install rebuild would change the size (.hsaco
        // is opaque binary) in realistic cases; size equality + fresh wipe is
        // sufficient for the deployment workflow.
        if let (Ok(s_meta), Ok(d_meta)) = (std::fs::metadata(&src), std::fs::metadata(&dst)) {
            if s_meta.len() == d_meta.len() {
                continue;
            }
        }
        std::fs::copy(&src, &dst)?;
    }
    Ok(())
}

/// Compiles HIP kernel sources to code objects, with caching.
/// Tries pre-compiled blobs first (kernels/compiled/{arch}/), falls back to hipcc.
pub struct KernelCompiler {
    cache_dir: PathBuf,
    arch: String,
    compiled: HashMap<String, PathBuf>,
    precompiled_dir: Option<PathBuf>,
    has_hipcc: bool,
}

impl KernelCompiler {
    pub fn new(arch: &str) -> HipResult<Self> {
        let cache_dir = std::env::temp_dir().join("hipfire_kernels");
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("failed to create cache dir: {e}"))
        })?;

        // Probe for pre-compiled kernels: exe-relative → CWD-relative → ~/.hipfire/bin/
        let precompiled_dir = std::env::current_exe().ok()
            .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
            .map(|dir| dir.join("kernels").join("compiled").join(arch))
            .filter(|p| p.is_dir())
            .or_else(|| {
                let cwd_path = PathBuf::from("kernels/compiled").join(arch);
                if cwd_path.is_dir() { Some(cwd_path) } else { None }
            })
            .or_else(|| {
                std::env::var("HOME").ok()
                    .map(|h| PathBuf::from(h).join(".hipfire/bin/kernels/compiled").join(arch))
                    .filter(|p| p.is_dir())
            });

        // Seed the tmpfs hot path from the persistent install location. /tmp
        // dies on reboot but the install blobs don't, so first-daemon-after-
        // boot copies them in. Subsequent daemons see a warm /tmp and skip
        // this. Copy is incremental — only copies files not already present
        // (or with stale hash) to avoid churn when both locations agree.
        // `hipfire update` wipes BOTH /tmp and the install dir, so after an
        // update + restart we get a fully-fresh re-seed.
        let hot_dir = cache_dir.join(arch);
        if let Some(ref cold) = precompiled_dir {
            if let Err(e) = seed_hot_from_cold(cold, &hot_dir) {
                eprintln!("  hot-path seed failed ({e}) — falling back to install dir reads");
            }
        }
        // Prefer the hot-path (tmpfs) dir when it exists and has contents.
        // This is what the `compile()` lookup uses from here on.
        let effective_precompiled = if hot_dir.is_dir()
            && std::fs::read_dir(&hot_dir).map(|mut it| it.any(|e| e.map(|e| e.path().extension().map(|x| x == "hsaco").unwrap_or(false)).unwrap_or(false))).unwrap_or(false)
        {
            Some(hot_dir.clone())
        } else {
            precompiled_dir.clone()
        };

        if let Some(ref dir) = effective_precompiled {
            eprintln!("  pre-compiled kernels: {}", dir.display());
        }
        let precompiled_dir = effective_precompiled;

        // Probe for hipcc once at init, not per-kernel
        let has_hipcc = Command::new("hipcc").arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        Ok(Self {
            cache_dir,
            arch: arch.to_string(),
            compiled: HashMap::new(),
            precompiled_dir,
            has_hipcc,
        })
    }

    /// Returns a reference to all compiled kernel paths (name → .hsaco path).
    pub fn compiled_kernels(&self) -> &HashMap<String, PathBuf> {
        &self.compiled
    }

    /// Compile a HIP kernel source string. Returns path to .hsaco file.
    /// Tries pre-compiled blob first (with hash validation), falls back to hipcc.
    pub fn compile(&mut self, name: &str, source: &str) -> HipResult<&Path> {
        if self.compiled.contains_key(name) {
            return Ok(&self.compiled[name]);
        }

        // Hash source + arch for cache validation (used by both pre-compiled and runtime paths)
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        self.arch.hash(&mut hasher);
        let src_hash = format!("{:016x}", hasher.finish());

        // Try pre-compiled .hsaco first, validating with a .hash sidecar file.
        // If hash is missing/mismatched AND hipcc is available, prefer recompilation.
        // If hipcc is unavailable (packaged install), use the blob as-is.
        // See: https://github.com/Kaden-Schutt/hipfire/issues/2
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
                    eprintln!("  WARNING: {name}: using UNVALIDATED pre-compiled blob (hipcc unavailable)");
                    eprintln!("           Output may be incorrect. Install ROCm SDK or rebuild blobs with matching hashes.");
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

        let cache_valid = obj_path.exists() && hash_path.exists()
            && std::fs::read_to_string(&hash_path).unwrap_or_default() == src_hash;

        if !cache_valid {
            Self::hipcc_compile(&self.arch, &src_path, &obj_path, name, source)?;
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

    /// Run hipcc for a single kernel. Shared by compile() and compile_batch().
    fn hipcc_compile(arch: &str, src_path: &Path, obj_path: &Path, name: &str, source: &str) -> HipResult<()> {
        std::fs::write(src_path, source).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("failed to write kernel source: {e}"))
        })?;
        let _ = std::fs::remove_file(obj_path);

        // Optional extra hipcc flags via HIPFIRE_HIPCC_EXTRA_FLAGS. Used for
        // one-off experiments like `-mcumode` vs `-mno-cumode` on RDNA1
        // without having to rebuild every call site.
        let extra = std::env::var("HIPFIRE_HIPCC_EXTRA_FLAGS").unwrap_or_default();
        let mut args: Vec<String> = vec![
            "--genco".into(),
            format!("--offload-arch={arch}"),
            "-O3".into(),
        ];
        // Some hipcc installs (notably V620's CachyOS build of ROCm 7.2) do not
        // auto-inject the HIP include path, so `#include <hip/hip_runtime.h>`
        // fails with "file not found". Add well-known candidates as -I flags —
        // existence-checked so wrong paths on other distros don't leak in.
        let hip_path = std::env::var("HIP_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        for candidate in [
            format!("{hip_path}/include"),
            "/opt/rocm/include".to_string(),
        ] {
            if Path::new(&candidate).join("hip/hip_runtime.h").exists() {
                args.push(format!("-I{candidate}"));
                break;
            }
        }
        for flag in extra.split_whitespace() {
            args.push(flag.to_string());
        }
        args.push("-o".into());
        args.push(obj_path.to_str().unwrap().into());
        args.push(src_path.to_str().unwrap().into());

        let output = Command::new("hipcc")
            .args(&args)
            .output()
            .map_err(|e| {
                hip_bridge::HipError::new(0, &format!("failed to run hipcc: {e}"))
            })?;

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
        let mut to_compile: Vec<(String, String, String, PathBuf, PathBuf, PathBuf)> = Vec::new();

        for &(name, source) in kernels {
            if self.compiled.contains_key(name) {
                continue;
            }

            let mut hasher = DefaultHasher::new();
            source.hash(&mut hasher);
            self.arch.hash(&mut hasher);
            let src_hash = format!("{:016x}", hasher.finish());

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

            let cache_valid = obj_path.exists() && hash_path.exists()
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
                name.to_string(), source.to_string(), src_hash,
                src_path, obj_path, hash_path,
            ));
        }

        if to_compile.is_empty() {
            return Ok(());
        }

        let n = to_compile.len();
        eprintln!("  compiling {n} kernels in parallel...");
        let arch = self.arch.clone();
        let precompiled_dir = self.precompiled_dir.clone();

        // Shared counter so parallel threads can report "[i/N] name" as each one
        // completes. Ordering follows completion (not launch) — matches the pace
        // of hipcc finishing.
        let done = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Spawn hipcc in parallel threads
        let results: Vec<_> = to_compile.into_iter().map(|(name, source, src_hash, src_path, obj_path, hash_path)| {
            let arch = arch.clone();
            let precompiled_dir = precompiled_dir.clone();
            let done = std::sync::Arc::clone(&done);
            let handle = thread::spawn(move || {
                let result = Self::hipcc_compile(&arch, &src_path, &obj_path, &name, &source);
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
        }).collect();

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
