//! Kernel efficiency profiler for RDNA GPUs.
//! Measures how close each compiled kernel is to theoretical hardware limits.

use std::collections::HashMap;

/// Hardware capabilities for an RDNA GPU.
#[derive(Debug, Clone)]
pub struct GpuCapability {
    pub arch: String,
    pub generation: &'static str,      // "RDNA1", "RDNA2", etc.
    pub cu_count: u32,
    pub simds_per_cu: u32,
    pub max_waves_per_simd: u32,
    pub vgprs_per_simd: u32,
    pub lds_per_cu_bytes: u32,
    pub l2_cache_mb: f32,
    pub infinity_cache_mb: f32,         // 0 for RDNA1
    pub peak_bw_gbs: f32,              // theoretical peak memory BW
    pub boost_clock_mhz: u32,
    pub mem_clock_mhz: u32,
    pub mem_bus_width_bits: u32,
    pub vram_mb: u32,
}

/// Per-arch hardware specs that can't be queried at runtime.
struct ArchSpec {
    generation: &'static str,
    simds_per_cu: u32,
    max_waves_per_simd: u32,
    vgprs_per_simd: u32,
    lds_per_cu: u32,
    l2_cache_mb: f32,
    infinity_cache_mb: f32,
    default_bus_width: u32,
}

fn arch_spec(arch: &str) -> ArchSpec {
    match arch {
        // Vega 20 / GCN5
        "gfx906" => ArchSpec {
            generation: "GCN5", simds_per_cu: 4, max_waves_per_simd: 10,
            vgprs_per_simd: 1024, lds_per_cu: 65536,
            l2_cache_mb: 4.0, infinity_cache_mb: 0.0, default_bus_width: 4096,
        },
        // RDNA1
        "gfx1010" | "gfx1011" | "gfx1012" => ArchSpec {
            generation: "RDNA1", simds_per_cu: 2, max_waves_per_simd: 20,
            vgprs_per_simd: 1024, lds_per_cu: 65536,
            l2_cache_mb: 4.0, infinity_cache_mb: 0.0, default_bus_width: 256,
        },
        // RDNA2
        "gfx1030" | "gfx1031" | "gfx1032" | "gfx1033" | "gfx1034" | "gfx1035" | "gfx1036" => ArchSpec {
            generation: "RDNA2", simds_per_cu: 2, max_waves_per_simd: 20,
            vgprs_per_simd: 1024, lds_per_cu: 65536,
            l2_cache_mb: 4.0, infinity_cache_mb: 128.0, default_bus_width: 256,
        },
        // RDNA3
        "gfx1100" | "gfx1101" | "gfx1102" => ArchSpec {
            generation: "RDNA3", simds_per_cu: 2, max_waves_per_simd: 16,
            vgprs_per_simd: 1536, lds_per_cu: 65536,
            l2_cache_mb: 6.0, infinity_cache_mb: 96.0, default_bus_width: 384,
        },
        // RDNA4
        "gfx1200" | "gfx1201" => ArchSpec {
            generation: "RDNA4", simds_per_cu: 2, max_waves_per_simd: 16,
            vgprs_per_simd: 1536, lds_per_cu: 65536,
            l2_cache_mb: 4.0, infinity_cache_mb: 64.0, default_bus_width: 256,
        },
        // Unknown — conservative RDNA1 defaults
        _ => ArchSpec {
            generation: "unknown", simds_per_cu: 2, max_waves_per_simd: 20,
            vgprs_per_simd: 1024, lds_per_cu: 65536,
            l2_cache_mb: 4.0, infinity_cache_mb: 0.0, default_bus_width: 256,
        },
    }
}

impl GpuCapability {
    /// Build from arch string + runtime queries.
    pub fn detect(arch: &str, vram_bytes: u64) -> Self {
        let spec = arch_spec(arch);

        // CU count from sysfs (try multiple paths)
        let cu_count = read_sysfs_cu_count().unwrap_or_else(|| {
            // Fallback: common defaults per arch
            match arch {
                "gfx906" => 60,   // Vega 20 / Radeon VII / MI50 class
                "gfx1010" => 40,   // RX 5700 XT
                "gfx1030" => 60,   // RX 6800
                "gfx1100" => 48,   // RX 7800 XT
                "gfx1200" => 32,   // RX 9070
                "gfx1201" => 32,   // RX 9070 XT
                _ => 40,
            }
        });

        // Clock speeds from sysfs
        let (boost_mhz, mem_mhz) = read_sysfs_clocks().unwrap_or((1800, 875));

        // Detect bus width from VRAM size heuristic when sysfs unavailable
        let bus_width = read_sysfs_bus_width().unwrap_or(spec.default_bus_width);

        // GDDR6 effective rate: sysfs reports interface clock (e.g., 875 MHz).
        // GDDR6 data rate = clock * 2 (DDR) * 8 (prefetch) = 16x multiplier.
        // Peak BW = mem_clock * 16 * bus_width / 8 (bits→bytes) / 1000 (MHz→GHz)
        let gddr_multiplier: f32 = match spec.generation {
            "GCN5" => 2.0,                         // HBM2 DDR
            "RDNA1" | "RDNA2" | "RDNA3" => 16.0, // GDDR6
            "RDNA4" => 16.0,                       // GDDR6 (9070 series)
            _ => 16.0,
        };
        let peak_bw = mem_mhz as f32 * gddr_multiplier * bus_width as f32 / 8.0 / 1000.0;

        Self {
            arch: arch.to_string(),
            generation: spec.generation,
            cu_count,
            simds_per_cu: spec.simds_per_cu,
            max_waves_per_simd: spec.max_waves_per_simd,
            vgprs_per_simd: spec.vgprs_per_simd,
            lds_per_cu_bytes: spec.lds_per_cu,
            l2_cache_mb: spec.l2_cache_mb,
            infinity_cache_mb: spec.infinity_cache_mb,
            peak_bw_gbs: peak_bw,
            boost_clock_mhz: boost_mhz,
            mem_clock_mhz: mem_mhz,
            mem_bus_width_bits: bus_width,
            vram_mb: (vram_bytes / (1024 * 1024)) as u32,
        }
    }

    /// Ridge point: peak FLOPS / peak BW = FLOP/byte threshold for compute-bound
    pub fn ridge_point_flop_per_byte(&self) -> f32 {
        // Peak FP32 FLOPS: CUs * SIMDs/CU * 32 lanes * 2 (FMA) * boost_clock
        let peak_flops = self.cu_count as f64
            * self.simds_per_cu as f64
            * 32.0 * 2.0
            * self.boost_clock_mhz as f64
            * 1e6;
        let peak_bw = self.peak_bw_gbs as f64 * 1e9;
        if peak_bw > 0.0 { (peak_flops / peak_bw) as f32 } else { 0.0 }
    }

    pub fn total_simds(&self) -> u32 { self.cu_count * self.simds_per_cu }
    pub fn max_total_waves(&self) -> u32 { self.total_simds() * self.max_waves_per_simd }
}

/// Parsed kernel ISA metadata from an .hsaco file.
#[derive(Debug, Clone)]
pub struct KernelProfile {
    pub name: String,
    pub vgprs: u32,
    pub sgprs: u32,
    pub lds_bytes: u32,
    pub scratch_bytes: u32,
    pub kernarg_bytes: u64,
    pub occupancy_waves: u32,
    pub max_waves: u32,
    pub occupancy_limiter: &'static str,
}

impl KernelProfile {
    pub fn occupancy_pct(&self) -> f32 {
        if self.max_waves > 0 { self.occupancy_waves as f32 / self.max_waves as f32 * 100.0 } else { 0.0 }
    }
}

/// Profile all compiled kernels. Returns (GpuCapability, Vec<KernelProfile>).
pub fn profile_kernels(
    arch: &str,
    vram_bytes: u64,
    compiled_kernels: &HashMap<String, std::path::PathBuf>,
) -> (GpuCapability, Vec<KernelProfile>) {
    let cap = GpuCapability::detect(arch, vram_bytes);
    let mut profiles = Vec::new();

    for (name, path) in compiled_kernels {
        if let Ok(data) = std::fs::read(path) {
            if let Some(profile) = profile_hsaco(name, &data, &cap) {
                profiles.push(profile);
            }
        }
    }

    profiles.sort_by(|a, b| a.name.cmp(&b.name));
    (cap, profiles)
}

/// Profile a single .hsaco file (may contain multiple kernels).
fn profile_hsaco(module_name: &str, data: &[u8], cap: &GpuCapability) -> Option<KernelProfile> {
    // Skip offload bundle wrapper if present
    let elf = if data.len() > 24 && &data[0..24] == b"__CLANG_OFFLOAD_BUNDLE__" {
        data.windows(4).position(|w| w == &[0x7f, b'E', b'L', b'F'])
            .map(|pos| &data[pos..])
    } else {
        Some(data)
    }?;

    if elf.len() < 64 || elf[0..4] != [0x7f, b'E', b'L', b'F'] {
        return None;
    }

    // Find kernel descriptor via symbol table (same as redline/hsaco.rs)
    let phoff = u64_le(elf, 32) as usize;
    let phentsize = u16_le(elf, 54) as usize;
    let phnum = u16_le(elf, 56) as usize;

    // Build VA → file offset mapping
    let mut segments = Vec::new();
    for i in 0..phnum {
        let base = phoff + i * phentsize;
        if base + phentsize > elf.len() { break; }
        if u32_le(elf, base) == 1 { // PT_LOAD
            segments.push((u64_le(elf, base + 16), u64_le(elf, base + 8), u64_le(elf, base + 32)));
        }
    }

    let shoff = u64_le(elf, 40) as usize;
    let shentsize = u16_le(elf, 58) as usize;
    let shnum = u16_le(elf, 60) as usize;
    let shstrndx = u16_le(elf, 62) as usize;
    if shstrndx >= shnum { return None; }
    let _shstr_offset = u64_le(elf, shoff + shstrndx * shentsize + 24) as usize;

    let mut symtab_offset = 0usize;
    let mut symtab_size = 0usize;
    let mut symtab_entsize = 0usize;
    let mut symtab_link = 0usize;

    for i in 0..shnum {
        let base = shoff + i * shentsize;
        if base + 40 > elf.len() { break; }
        if u32_le(elf, base + 4) == 2 { // SHT_SYMTAB
            symtab_offset = u64_le(elf, base + 24) as usize;
            symtab_size = u64_le(elf, base + 32) as usize;
            symtab_entsize = u64_le(elf, base + 56) as usize;
            symtab_link = u32_le(elf, base + 40) as usize;
        }
    }

    if symtab_entsize == 0 { return None; }

    let strtab_offset = if symtab_link < shnum {
        u64_le(elf, shoff + symtab_link * shentsize + 24) as usize
    } else { return None; };

    // Find the first .kd symbol (most .hsaco have exactly one kernel)
    let num_syms = symtab_size / symtab_entsize;
    for i in 0..num_syms {
        let base = symtab_offset + i * symtab_entsize;
        if base + symtab_entsize > elf.len() { break; }
        let st_name = u32_le(elf, base) as usize;
        let st_value = u64_le(elf, base + 8);
        let sym_name = read_cstr(elf, strtab_offset + st_name);

        if sym_name.ends_with(".kd") {
            let kd_off = va_to_offset(&segments, st_value)? as usize;
            if kd_off + 64 > elf.len() { continue; }

            let meta = RawKernelMeta {
                pgm_rsrc1: u32_le(elf, kd_off + 48),
                pgm_rsrc2: u32_le(elf, kd_off + 52),
                group_segment_size: u32_le(elf, kd_off),
                private_segment_size: u32_le(elf, kd_off + 4),
                kernarg_size: u64_le(elf, kd_off + 8),
            };

            let vgprs = decode_vgprs(meta.pgm_rsrc1, cap);
            let sgprs = decode_sgprs(meta.pgm_rsrc1);

            let max_waves = cap.max_waves_per_simd;
            let vgpr_waves = if vgprs > 0 { cap.vgprs_per_simd / vgprs } else { max_waves };
            let lds_waves = if meta.group_segment_size > 0 {
                (cap.lds_per_cu_bytes / meta.group_segment_size) / cap.simds_per_cu
            } else {
                max_waves
            };

            let occupancy = max_waves.min(vgpr_waves).min(lds_waves);
            let limiter = if occupancy >= max_waves { "wave limit" }
                else if vgpr_waves <= lds_waves { "VGPRs" }
                else { "LDS" };

            return Some(KernelProfile {
                name: module_name.to_string(),
                vgprs, sgprs,
                lds_bytes: meta.group_segment_size,
                scratch_bytes: meta.private_segment_size,
                kernarg_bytes: meta.kernarg_size,
                occupancy_waves: occupancy,
                max_waves,
                occupancy_limiter: limiter,
            });
        }
    }
    None
}

struct RawKernelMeta {
    pgm_rsrc1: u32,
    pgm_rsrc2: u32,
    group_segment_size: u32,
    private_segment_size: u32,
    kernarg_size: u64,
}


/// Decode VGPR count from pgm_rsrc1. Granularity depends on arch.
fn decode_vgprs(pgm_rsrc1: u32, cap: &GpuCapability) -> u32 {
    let field = pgm_rsrc1 & 0x3F;
    // GFX10+: wave32 uses granularity 8, wave64 uses granularity 4
    // hipcc targets wave32 for RDNA, so granularity = 8
    let granularity = match cap.generation {
        "RDNA3" | "RDNA4" => 8, // confirmed wave32 granularity
        _ => 8,                  // RDNA1/2 also wave32
    };
    (field + 1) * granularity
}

fn decode_sgprs(pgm_rsrc1: u32) -> u32 {
    (((pgm_rsrc1 >> 6) & 0xF) + 1) * 8
}

fn va_to_offset(segments: &[(u64, u64, u64)], va: u64) -> Option<u64> {
    for &(vaddr, offset, filesz) in segments {
        if va >= vaddr && va < vaddr + filesz {
            return Some(va - vaddr + offset);
        }
    }
    None
}

// ── Sysfs readers ──────────────────────────────────────────

fn read_sysfs_cu_count() -> Option<u32> {
    // Try KFD topology (most reliable)
    for node in &["1", "0", "2"] {
        let path = format!("/sys/class/kfd/kfd/topology/nodes/{node}/properties");
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Some(line) = text.lines().find(|l| l.starts_with("simd_count")) {
                // simd_count = total SIMDs, CUs = simd_count / 2
                if let Some(val) = line.split_whitespace().last() {
                    if let Ok(simds) = val.parse::<u32>() {
                        if simds > 0 { return Some(simds / 2); }
                    }
                }
            }
        }
    }
    None
}

fn read_sysfs_clocks() -> Option<(u32, u32)> {
    let find_card = || -> Option<String> {
        for entry in std::fs::read_dir("/sys/class/drm/").ok()? {
            let name = entry.ok()?.file_name().into_string().ok()?;
            if name.starts_with("card") && !name.contains('-') {
                let vendor = std::fs::read_to_string(format!("/sys/class/drm/{name}/device/vendor")).ok()?;
                if vendor.trim() == "0x1002" { return Some(name); }
            }
        }
        None
    };
    let card = find_card()?;

    // GPU boost clock: last entry in pp_dpm_sclk (highest P-state)
    let sclk = std::fs::read_to_string(format!("/sys/class/drm/{card}/device/pp_dpm_sclk")).ok()?;
    let gpu_mhz = sclk.lines().last()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.trim_end_matches("Mhz").parse::<u32>().ok())?;

    // Memory clock: last entry in pp_dpm_mclk
    let mclk = std::fs::read_to_string(format!("/sys/class/drm/{card}/device/pp_dpm_mclk")).ok()?;
    let mem_mhz = mclk.lines().last()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.trim_end_matches("Mhz").parse::<u32>().ok())?;

    Some((gpu_mhz, mem_mhz))
}

fn read_sysfs_bus_width() -> Option<u32> {
    // Try KFD topology for width_x
    for node in &["1", "0"] {
        let path = format!("/sys/class/kfd/kfd/topology/nodes/{node}/properties");
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Some(line) = text.lines().find(|l| l.starts_with("width")) {
                if let Some(val) = line.split_whitespace().last() {
                    if let Ok(w) = val.parse::<u32>() {
                        if w > 0 { return Some(w); }
                    }
                }
            }
        }
    }
    None
}

// ── Minimal ELF parsing helpers ────────────────────────────

fn u16_le(d: &[u8], o: usize) -> u16 { u16::from_le_bytes([d[o], d[o+1]]) }
fn u32_le(d: &[u8], o: usize) -> u32 { u32::from_le_bytes([d[o], d[o+1], d[o+2], d[o+3]]) }
fn u64_le(d: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([d[o], d[o+1], d[o+2], d[o+3], d[o+4], d[o+5], d[o+6], d[o+7]])
}
fn read_cstr(d: &[u8], o: usize) -> String {
    let mut e = o;
    while e < d.len() && d[e] != 0 { e += 1; }
    String::from_utf8_lossy(&d[o..e]).into()
}

/// JSON serialization for daemon IPC.
impl GpuCapability {
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"arch":"{}","generation":"{}","cu_count":{},"simds_per_cu":{},"max_waves_per_simd":{},"vgprs_per_simd":{},"lds_per_cu":{},"l2_cache_mb":{},"infinity_cache_mb":{},"peak_bw_gbs":{:.1},"boost_clock_mhz":{},"mem_clock_mhz":{},"mem_bus_width":{},"vram_mb":{},"ridge_point":{:.1}}}"#,
            self.arch, self.generation, self.cu_count, self.simds_per_cu,
            self.max_waves_per_simd, self.vgprs_per_simd, self.lds_per_cu_bytes,
            self.l2_cache_mb, self.infinity_cache_mb, self.peak_bw_gbs,
            self.boost_clock_mhz, self.mem_clock_mhz, self.mem_bus_width_bits,
            self.vram_mb, self.ridge_point_flop_per_byte()
        )
    }
}

impl KernelProfile {
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"name":"{}","vgprs":{},"sgprs":{},"lds_bytes":{},"scratch_bytes":{},"occupancy":{{"waves":{},"max":{},"pct":{:.1},"limiter":"{}"}}}}"#,
            self.name, self.vgprs, self.sgprs, self.lds_bytes, self.scratch_bytes,
            self.occupancy_waves, self.max_waves, self.occupancy_pct(), self.occupancy_limiter
        )
    }
}
