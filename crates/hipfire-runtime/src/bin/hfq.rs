// SPDX-License-Identifier: Apache-2.0
// hipfire — `hfq`: HFQ container manipulation tool.
//
//! Inspect and edit HFQ containers (models, calibration-artifact sidecars,
//! `.mtp`/`.dflash`/`.triattn` sidecars). With these, "bundle everything vs keep
//! separate" is a runtime choice — split a Hessian out of a bundle, merge
//! sidecars, or embed a chat template into a model — not an upfront commitment.
//!
//! Subcommands:
//!   hfq list   <file>                          — arch, metadata keys, tensor table
//!   hfq extract <in> <out> --tensor <pat> ...  — copy matching tensors to a new HFQ
//!                                                 (pat: exact, `prefix*`, `*suffix`, `*sub*`)
//!   hfq meta-set <in> <out> --key <k> (--value <v> | --value-file <f>)
//!                                              — set a metadata JSON key (all tensors
//!                                                copied), e.g. embed a jinja2 template
//!   hfq meta-get <file> [--key <k>]            — dump metadata JSON (or one key)
//!   hfq rearch <in> <out> --arch-id <id>       — rewrite HFQM header arch_id and
//!                                                numeric metadata.arch_id
//!
//! Examples:
//!   hfq extract model.calib.hfq just.hessian.hfq --tensor '*.hessian'
//!   hfq meta-set model.hfq model+tmpl.hfq --key chat_template --value-file tmpl.jinja

use hipfire_runtime::hfq::HfqFile;
use std::io::Write;
use std::path::Path;

const HFQM_MAGIC: &[u8; 4] = b"HFQM";
const HFQM_VERSION: u32 = 1;

/// Owned tensor (name, quant_type, shape, group_size, data) for writing.
struct Tensor {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
}

fn write_hfq(
    path: &str,
    arch: u32,
    metadata_json: &str,
    tensors: &[Tensor],
) -> std::io::Result<()> {
    let meta = metadata_json.as_bytes();
    let metadata_offset = 32u64;
    let index_offset = metadata_offset + meta.len() as u64;
    let mut index = Vec::new();
    index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        let nb = t.name.as_bytes();
        index.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        index.extend_from_slice(nb);
        index.push(t.quant_type);
        index.push(t.shape.len() as u8);
        for &d in &t.shape {
            index.extend_from_slice(&d.to_le_bytes());
        }
        index.extend_from_slice(&t.group_size.to_le_bytes());
        index.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
    }
    let data_start = index_offset + index.len() as u64;
    let data_offset = (data_start + 4095) & !4095;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(HFQM_MAGIC)?;
    f.write_all(&HFQM_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;
    f.write_all(meta)?;
    f.write_all(&index)?;
    f.write_all(&vec![0u8; (data_offset - data_start) as usize])?;
    for t in tensors {
        f.write_all(&t.data)?;
    }
    f.flush()
}

/// Simple glob: exact, `prefix*`, `*suffix`, `*sub*`, or `*` (all).
fn matches(pat: &str, name: &str) -> bool {
    if pat == "*" {
        return true;
    }
    match (pat.starts_with('*'), pat.ends_with('*')) {
        (true, true) => name.contains(pat.trim_matches('*')),
        (true, false) => name.ends_with(pat.trim_start_matches('*')),
        (false, true) => name.starts_with(pat.trim_end_matches('*')),
        (false, false) => name == pat,
    }
}

fn flag<'a>(args: &'a [String], f: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == f)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}
fn flags_all(args: &[String], f: &str) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..args.len() {
        if args[i] == f {
            if let Some(v) = args.get(i + 1) {
                out.push(v.clone());
            }
        }
    }
    out
}

fn load_all(path: &str) -> (u32, String, Vec<Tensor>) {
    let hfq = HfqFile::open(Path::new(path)).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let arch = hfq.arch_id;
    let meta = hfq.metadata_json.clone();
    let names: Vec<(String, u8, Vec<u32>, u32)> = hfq
        .tensors()
        .iter()
        .map(|t| (t.name.clone(), t.quant_type, t.shape.clone(), t.group_size))
        .collect();
    let tensors = names
        .into_iter()
        .map(|(name, qt, shape, gs)| {
            let (_, data) = hfq.tensor_data(&name).expect("tensor data");
            Tensor {
                name,
                quant_type: qt,
                shape,
                group_size: gs,
                data: data.to_vec(),
            }
        })
        .collect();
    (arch, meta, tensors)
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let cmd = argv.get(1).map(|s| s.as_str()).unwrap_or("");
    match cmd {
        "list" => {
            let path = argv.get(2).expect("usage: hfq list <file>");
            let hfq = HfqFile::open(Path::new(path)).expect("open");
            println!("arch_id: {}", hfq.arch_id);
            let meta: serde_json::Value =
                serde_json::from_str(&hfq.metadata_json).unwrap_or(serde_json::json!({}));
            if let Some(o) = meta.as_object() {
                let mut keys: Vec<&String> = o.keys().collect();
                keys.sort();
                println!(
                    "metadata keys: {}",
                    keys.iter()
                        .map(|k| k.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            let ts = hfq.tensors();
            println!("tensors: {}", ts.len());
            let mut total = 0u64;
            for t in ts {
                total += t.data_size as u64;
                println!(
                    "  {:60} qt={:<2} shape={:?} g={} {:.2} MB",
                    t.name,
                    t.quant_type,
                    t.shape,
                    t.group_size,
                    t.data_size as f64 / 1e6
                );
            }
            println!("total tensor bytes: {:.2} MB", total as f64 / 1e6);
        }
        "extract" => {
            let inp = argv
                .get(2)
                .expect("usage: hfq extract <in> <out> --tensor <pat>...");
            let out = argv.get(3).expect("out path");
            let pats = flags_all(&argv, "--tensor");
            assert!(!pats.is_empty(), "need at least one --tensor <pattern>");
            let (arch, meta, tensors) = load_all(inp);
            let kept: Vec<Tensor> = tensors
                .into_iter()
                .filter(|t| pats.iter().any(|p| matches(p, &t.name)))
                .collect();
            assert!(!kept.is_empty(), "no tensors matched {pats:?}");
            eprintln!("extracting {} tensor(s) → {out}", kept.len());
            for t in &kept {
                eprintln!("  {}", t.name);
            }
            write_hfq(out, arch, &meta, &kept).expect("write");
        }
        "meta-set" => {
            let inp = argv
                .get(2)
                .expect("usage: hfq meta-set <in> <out> --key <k> --value[-file] <v>");
            let out = argv.get(3).expect("out path");
            let key = flag(&argv, "--key").expect("--key required");
            let value: String = if let Some(vf) = flag(&argv, "--value-file") {
                std::fs::read_to_string(vf).expect("read value-file")
            } else {
                flag(&argv, "--value")
                    .expect("--value or --value-file required")
                    .to_string()
            };
            let (arch, meta_json, tensors) = load_all(inp);
            let mut meta: serde_json::Value =
                serde_json::from_str(&meta_json).unwrap_or(serde_json::json!({}));
            meta.as_object_mut()
                .expect("metadata not an object")
                .insert(key.to_string(), serde_json::Value::String(value));
            eprintln!(
                "set metadata[{key}] ({} tensors copied) → {out}",
                tensors.len()
            );
            write_hfq(out, arch, &serde_json::to_string(&meta).unwrap(), &tensors).expect("write");
        }
        "meta-get" => {
            let path = argv.get(2).expect("usage: hfq meta-get <file> [--key <k>]");
            let hfq = HfqFile::open(Path::new(path)).expect("open");
            if let Some(k) = flag(&argv, "--key") {
                let meta: serde_json::Value =
                    serde_json::from_str(&hfq.metadata_json).unwrap_or(serde_json::json!({}));
                match meta.get(k) {
                    Some(v) => println!("{}", serde_json::to_string_pretty(v).unwrap()),
                    None => {
                        eprintln!("key {k} not found");
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", hfq.metadata_json);
            }
        }
        "rearch" => {
            let inp = argv
                .get(2)
                .expect("usage: hfq rearch <in> <out> --arch-id <id>");
            let out = argv.get(3).expect("out path");
            let arch: u32 = flag(&argv, "--arch-id")
                .expect("--arch-id required")
                .parse()
                .expect("--arch-id must be a u32");
            let (_old_arch, meta_json, tensors) = load_all(inp);
            let mut meta: serde_json::Value =
                serde_json::from_str(&meta_json).unwrap_or(serde_json::json!({}));
            let obj = meta.as_object_mut().expect("metadata not an object");
            obj.insert(
                "arch_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(arch)),
            );
            obj.remove("arch_id_semantics");
            eprintln!(
                "set header arch_id and metadata.arch_id to {arch} ({} tensors copied) → {out}",
                tensors.len()
            );
            write_hfq(out, arch, &serde_json::to_string(&meta).unwrap(), &tensors).expect("write");
        }
        _ => {
            eprintln!(
                "hfq — HFQ container tool\n\
                 usage:\n  hfq list <file>\n  hfq extract <in> <out> --tensor <pat>...\n\
                 \x20 hfq meta-set <in> <out> --key <k> (--value <v> | --value-file <f>)\n\
                 \x20 hfq meta-get <file> [--key <k>]\n\
                 \x20 hfq rearch <in> <out> --arch-id <id>"
            );
            std::process::exit(1);
        }
    }
}
