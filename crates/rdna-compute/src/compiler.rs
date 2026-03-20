//! Compile HIP kernels to code objects (.hsaco) via hipcc.

use hip_bridge::HipResult;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Compiles HIP kernel sources to code objects, with caching.
pub struct KernelCompiler {
    cache_dir: PathBuf,
    arch: String,
    compiled: HashMap<String, PathBuf>,
}

impl KernelCompiler {
    pub fn new(arch: &str) -> HipResult<Self> {
        let cache_dir = std::env::temp_dir().join("hipfire_kernels");
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("failed to create cache dir: {e}"))
        })?;
        Ok(Self {
            cache_dir,
            arch: arch.to_string(),
            compiled: HashMap::new(),
        })
    }

    /// Compile a HIP kernel source string. Returns path to .hsaco file.
    /// Caches by kernel name — won't recompile if already cached.
    pub fn compile(&mut self, name: &str, source: &str) -> HipResult<&Path> {
        if self.compiled.contains_key(name) {
            return Ok(&self.compiled[name]);
        }

        let src_path = self.cache_dir.join(format!("{name}.hip"));
        let obj_path = self.cache_dir.join(format!("{name}.hsaco"));

        // Check if .hsaco already exists on disk (persists across runs)
        if !obj_path.exists() {
            std::fs::write(&src_path, source).map_err(|e| {
                hip_bridge::HipError::new(0, &format!("failed to write kernel source: {e}"))
            })?;

            let output = Command::new("hipcc")
                .args([
                    "--genco",
                    &format!("--offload-arch={}", self.arch),
                    "-O3",
                    "-o",
                    obj_path.to_str().unwrap(),
                    src_path.to_str().unwrap(),
                ])
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
        }

        self.compiled.insert(name.to_string(), obj_path);
        Ok(&self.compiled[name])
    }
}
