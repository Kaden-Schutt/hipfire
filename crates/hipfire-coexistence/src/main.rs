// SPDX-License-Identifier: Apache-2.0
//! hipfire-coexistence — offline import/export/conversion/interop tooling, kept
//! OUT of the inference binaries (daemon/server/runtime) per AGENTS.md.
//!
//! ```text
//! hipfire-coexistence lora export  --hfq <model.hfq> --data-dir <dir> \
//!     [--limit N] [--strength S] [--no-orthogonalize] [--max-seq N] --out <adapter.lora.{hfq,json}>
//! hipfire-coexistence lora merge   --hfq <base.hfq> --adapter <adapter.lora> --out <merged.hfq>
//! hipfire-coexistence lora convert --in <adapter.lora.{hfq,json}> --out <adapter.lora.{hfq,json}>
//! ```
//!
//! `export` derives a rank-1 abliteration adapter by capturing +/- residual means
//! through a `hipfire-daemon` (it needs the model forward); `merge` and `convert`
//! are pure offline file operations.

use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use hipfire_steer::driver::{load_prompts, ModelHarness, Prompt};
use hipfire_steer::{derive_directions, lora, SteerMode};
use hipfire_steer_harness::DaemonHarness;

const SYSTEM_PROMPT: &str = "You are a helpful assistant.";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let group = args.first().map(String::as_str);
    let op = args.get(1).map(String::as_str);
    match (group, op) {
        (Some("lora"), Some("export")) => lora_export(&args[2..]),
        (Some("lora"), Some("merge")) => lora_merge(&args[2..]),
        (Some("lora"), Some("convert")) => lora_convert(&args[2..]),
        _ => {
            usage();
            std::process::exit(2);
        }
    }
}

fn usage() {
    eprintln!(
        "usage: hipfire-coexistence <group> <op> [flags]\n\
         \n\
         lora export  --hfq <model.hfq> --data-dir <dir> [--limit N] [--strength S] \
         [--no-orthogonalize] [--max-seq N] --out <adapter.lora.{{hfq,json}}>\n\
         lora merge   --hfq <base.hfq> --adapter <adapter.lora> --out <merged.hfq>\n\
         lora convert --in <adapter.lora.{{hfq,json}}> --out <adapter.lora.{{hfq,json}}>"
    );
}

/// Minimal `--key value` flag bag. `--no-orthogonalize` is a presence-only flag.
struct Flags(HashMap<String, String>);

impl Flags {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut m = HashMap::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let key = a
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected arg {a}"))?;
            if key == "no-orthogonalize" {
                m.insert(key.to_string(), "1".to_string());
                continue;
            }
            let v = it.next().ok_or_else(|| format!("{a} needs a value"))?;
            m.insert(key.to_string(), v.clone());
        }
        Ok(Flags(m))
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(String::as_str)
    }
    fn req(&self, k: &str) -> Result<&str, String> {
        self.get(k).ok_or_else(|| format!("--{k} is required"))
    }
    fn has(&self, k: &str) -> bool {
        self.0.contains_key(k)
    }
}

/// Write an adapter to the binary container for `*.hfq` paths, JSON otherwise.
fn write_adapter_dispatch(out: &Path, adapter: &lora::LoraAdapter) -> Result<(), Box<dyn Error>> {
    if out.extension().and_then(|e| e.to_str()) == Some("hfq") {
        hipfire_lora_hfq::write_lora_hfq(out, adapter)?;
    } else {
        lora::write_adapter(out, adapter)?;
    }
    Ok(())
}

fn load_set(dir: &Path, name: &str, limit: usize) -> Result<Vec<Prompt>, Box<dyn Error>> {
    let path = dir.join(name);
    let mut prompts = load_prompts(&path, SYSTEM_PROMPT)
        .map_err(|e| format!("loading {}: {e}", path.display()))?;
    prompts.truncate(limit);
    Ok(prompts)
}

/// Capture +/- residual means through a daemon, derive per-block directions, and
/// write a rank-1 ablate adapter. `--strength` seeds the adapter's default scale.
fn lora_export(args: &[String]) -> Result<(), Box<dyn Error>> {
    let f = Flags::parse(args)?;
    let hfq = f.req("hfq")?;
    let data_dir = PathBuf::from(f.req("data-dir")?);
    let limit: usize = f
        .get("limit")
        .unwrap_or("16")
        .parse()
        .map_err(|_| "bad --limit")?;
    let strength: f32 = f
        .get("strength")
        .unwrap_or("0.2")
        .parse()
        .map_err(|_| "bad --strength")?;
    let max_seq: usize = f
        .get("max-seq")
        .unwrap_or("2048")
        .parse()
        .map_err(|_| "bad --max-seq")?;
    let orthogonalize = !f.has("no-orthogonalize");
    let out = PathBuf::from(f.req("out")?);

    let good = load_set(&data_dir, "good_prompts.txt", limit)?;
    let bad = load_set(&data_dir, "bad_prompts.txt", limit)?;

    let daemon_bin = hipfire_daemon_adapter::find_daemon_bin_or_error()?;
    let tmp = std::env::temp_dir().join(format!("hipfire-coex-{}", std::process::id()));
    let mut h = DaemonHarness::connect(
        &daemon_bin,
        Path::new(hfq),
        max_seq,
        64,
        SYSTEM_PROMPT.to_string(),
        tmp,
    )?;

    eprintln!(
        "capturing directions ({} good / {} bad, {} layers) ...",
        good.len(),
        bad.len(),
        h.num_layers()
    );
    h.begin_capture()
        .map_err(|e| format!("begin_capture: {e}"))?;
    h.capture(&good).map_err(|e| format!("capture good: {e}"))?;
    let good_means = h
        .finish_capture()
        .map_err(|e| format!("finish good: {e}"))?;
    h.begin_capture()
        .map_err(|e| format!("begin_capture: {e}"))?;
    h.capture(&bad).map_err(|e| format!("capture bad: {e}"))?;
    let bad_means = h.finish_capture().map_err(|e| format!("finish bad: {e}"))?;

    let directions = derive_directions(&good_means, &bad_means, orthogonalize);
    let layers = h.num_layers();
    let adapter = lora::abliteration_adapter(
        "abliterate",
        &directions,
        SteerMode::Ablate,
        strength,
        0..layers,
    )?;
    write_adapter_dispatch(&out, &adapter)?;
    eprintln!(
        "wrote LoRA adapter: {} rank-1 deltas, default scale {strength:.2} → {}",
        adapter.deltas.len(),
        out.display()
    );
    Ok(())
}

/// Bundle an adapter into a base model `.hfq` → a self-contained model the daemon
/// auto-applies on load. Pure offline file op.
fn lora_merge(args: &[String]) -> Result<(), Box<dyn Error>> {
    let f = Flags::parse(args)?;
    let base = PathBuf::from(f.req("hfq")?);
    let adapter_path = PathBuf::from(f.req("adapter")?);
    let out = PathBuf::from(f.req("out")?);
    let adapter = hipfire_lora_hfq::read_lora_any(&adapter_path)?;
    hipfire_lora_hfq::merge_lora_into_model(&base, &adapter, &out)?;
    eprintln!(
        "merged {} + {} ({} deltas, scale {:.2}) → {}",
        base.display(),
        adapter_path.display(),
        adapter.deltas.len(),
        adapter.scale,
        out.display()
    );
    Ok(())
}

/// Convert an adapter container between the binary (`.hfq`) and JSON forms
/// (dispatched by the output extension). Pure offline file op.
fn lora_convert(args: &[String]) -> Result<(), Box<dyn Error>> {
    let f = Flags::parse(args)?;
    let inp = PathBuf::from(f.req("in")?);
    let out = PathBuf::from(f.req("out")?);
    let adapter = hipfire_lora_hfq::read_lora_any(&inp)?;
    write_adapter_dispatch(&out, &adapter)?;
    eprintln!(
        "converted {} → {} ({} deltas)",
        inp.display(),
        out.display(),
        adapter.deltas.len()
    );
    Ok(())
}
