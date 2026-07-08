//! dspark_parity: GPU-vs-CPU numeric parity for the DSpark **novel** head
//! kernels — `main_proj`+`main_norm` (target-hidden ingestion) and the Markov
//! head (`markov_w1` embed + `markov_w2` bias). These are the only DSpark code
//! paths with NO trunk reuse, hence the highest-risk surface for a silent
//! layout/transpose/dequant bug.
//!
//! This is the runnable slice of the plan's mandated numeric-parity spike: the
//! full fp8 `inference/model.py` reference cannot run on an RDNA box (fp8
//! kernels + 167 GB trunk), so we validate the novel linear heads against CPU
//! references derived from `model.py`, on the real quantized sidecar weights.
//! Reused MLA/MoE/HC kernels are covered by the trunk gates.
//!
//! Loads ONLY the ~5.6 GB sidecar (not the 83 GB trunk) and runs no inference.
//!
//! Usage: dspark_parity [path-to-dspark.mq2lloyd]
//! Exit 0 = all checks pass; exit 1 = a check failed (a real port bug).

use hipfire_arch_deepseek4::forward::dspark_head_parity;
use hipfire_arch_deepseek4::DeepseekV4;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::Path;

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/models/deepseek-v4-flash-dspark.mq2lloyd",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    eprintln!("opening {path}");
    let mut hfq = HfqFile::open(Path::new(&path)).map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    hfq.drop_mmap();
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;

    let dspark = DeepseekV4::load_dspark(&hfq, &mut gpu, &cfg)?
        .ok_or_else(|| "load_dspark returned None — no dspark_* config in metadata".to_string())?;

    let report = dspark_head_parity(&cfg, &dspark, &mut gpu)?;

    println!("\nDSpark novel-head numeric parity (GPU primitive vs CPU reference):");
    println!(
        "  {:<32} {:>8} {:>12} {:>12} {:>10}  {}",
        "check", "n", "max_abs", "rel_max", "cosine", "verdict"
    );
    for c in &report.checks {
        println!(
            "  {:<32} {:>8} {:>12.3e} {:>12.3e} {:>10.6}  {}",
            c.name,
            c.n,
            c.max_abs_diff,
            c.rel_max_abs,
            c.cosine,
            if c.pass { "PASS" } else { "FAIL" }
        );
    }

    dspark.free_gpu(&mut gpu);

    if report.all_pass() {
        println!("\nPARITY PASS — novel head kernels match CPU reference");
        Ok(())
    } else {
        Err("PARITY FAIL — a novel head kernel diverges from its CPU reference".into())
    }
}
