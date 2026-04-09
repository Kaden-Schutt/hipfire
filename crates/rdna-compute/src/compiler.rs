//! Compile HIP kernels to code objects (.hsaco) via hipcc.
//! Supports pre-compiled .hsaco blobs for deployment without ROCm SDK.

use hip_bridge::HipResult;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;

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

        if let Some(ref dir) = precompiled_dir {
            eprintln!("  pre-compiled kernels: {}", dir.display());
        }

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

        // Spawn hipcc in parallel threads
        let results: Vec<_> = to_compile.into_iter().map(|(name, source, src_hash, src_path, obj_path, hash_path)| {
            let arch = arch.clone();
            let precompiled_dir = precompiled_dir.clone();
            let handle = thread::spawn(move || {
                eprint!("    {name}...");
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
        eprintln!(" done.");

        if let Some(e) = errors.into_iter().next() {
            return Err(e);
        }
        Ok(())
    }
}
