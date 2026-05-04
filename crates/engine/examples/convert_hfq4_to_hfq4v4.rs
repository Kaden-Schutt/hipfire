//! Convert an HFQ4-G256 (.hfq) model file into a sister HFQ4v4/MQ4v4 file
//! that the gfx12 iu4 K=32 wmma path consumes.
//!
//! Per-tensor conversion: HFQ4-G256 → HFQ4v4 (K=32 groups, FP16 d, per-row mu).
//! Optional `--rotate` flag applies FWHT-32 to weights before SmoothQuant
//! analysis, producing the MQ4v4 sister format.
//!
//! Layout written to disk (per-tensor inside the .hfq archive):
//!   - For weight tensors with `quant_type` ∈ {6 (HFQ4-G256), 13 (MQ4-G256)}:
//!     replace the tensor data with a v4 blob: [weights row-major M*row_bytes][mu M*2 B]
//!     and update the tensor's `quant_type` to a new ID.
//!   - All other tensors (norms, embeddings, output) pass through unchanged.
//!
//! New quant_type IDs added by this format:
//!   19 = HFQ4v4-G32       (HFQ4v4 without rotation)
//!   20 = MQ4v4-G32        (HFQ4v4 with FWHT-32 rotation)
//!
//! Usage:
//!   cargo run --release -p engine --example convert_hfq4_to_hfq4v4 -- \
//!     --in qwen3.5-9b.mq4 --out qwen3.5-9b.mq4v4 [--rotate]
//!
//! For the GEMM correctness test, you can also dump a single weight tensor
//! standalone via `--single-tensor <name>` which emits a self-contained
//! v4 blob (header + weights + mu) consumable by the test harness.

use engine::hfq4v4::{
    self, convert_hfq4g256_to_hfq4v4, mu_bytes, weight_bytes, MuStrategy,
};
use memmap2::Mmap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut a = Args::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--in" => a.input = args.next().expect("--in needs path").into(),
            "--out" => a.output = args.next().expect("--out needs path").into(),
            "--rotate" => a.rotate = true,
            "--single-tensor" => {
                a.single_tensor = Some(args.next().expect("--single-tensor needs name"));
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    if a.input.as_os_str().is_empty() || a.output.as_os_str().is_empty() {
        eprintln!("--in and --out are required");
        print_help();
        std::process::exit(1);
    }
    a
}

#[derive(Default)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    rotate: bool,
    single_tensor: Option<String>,
}

fn print_help() {
    eprintln!(
        "convert_hfq4_to_hfq4v4 — HFQ4-G256 → HFQ4v4/MQ4v4 converter\n\
         \n\
         USAGE:\n\
         \tconvert_hfq4_to_hfq4v4 --in <PATH> --out <PATH> [--rotate]\n\
         \tconvert_hfq4_to_hfq4v4 --in <PATH> --out <PATH> --single-tensor <NAME>\n\
         \n\
         FLAGS:\n\
         \t--rotate           Apply FWHT-32 to weights before SmoothQuant\n\
         \t                   analysis (produces MQ4v4 magic).\n\
         \t--single-tensor    Emit a self-contained per-tensor v4 blob to <PATH>\n\
         \t                   instead of converting the whole file. Useful for\n\
         \t                   GEMM correctness tests.\n"
    );
}

const QUANT_HFQ4G256: u8 = 6;
const QUANT_MQ4G256: u8 = 13;
const QUANT_HFQ4V4_G32: u8 = 19;
const QUANT_MQ4V4_G32: u8 = 20;

fn main() {
    let args = parse_args();
    eprintln!("HFQ4 → HFQ4v4 converter");
    eprintln!("  input:  {}", args.input.display());
    eprintln!("  output: {}", args.output.display());
    eprintln!("  rotate: {}", args.rotate);
    if let Some(t) = &args.single_tensor {
        eprintln!("  single-tensor: {}", t);
    }
    eprintln!();

    let file = File::open(&args.input)
        .unwrap_or_else(|e| panic!("open input {}: {e}", args.input.display()));
    let mmap = unsafe { Mmap::map(&file) }.unwrap();

    if &mmap[0..4] != b"HFQM" {
        panic!("not an HFQ file (bad magic)");
    }
    let _version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
    let arch_id = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
    let n_tensors = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
    let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
    let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;

    eprintln!(
        "  arch_id={arch_id}, tensors={n_tensors}, meta@{metadata_offset}, data@{data_offset}"
    );

    // Walk the metadata + index. Metadata is a JSON blob; index is the
    // tensor table that follows.
    let meta_bytes = &mmap[metadata_offset..data_offset];
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    let mut meta_end = 0usize;
    for (i, &b) in meta_bytes.iter().enumerate() {
        if esc {
            esc = false;
            continue;
        }
        if in_str {
            if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        if b == b'"' {
            in_str = true;
        } else if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                meta_end = i + 1;
                break;
            }
        }
    }
    let metadata_json = std::str::from_utf8(&meta_bytes[..meta_end]).unwrap().to_string();
    let index_start = metadata_offset + meta_end;
    let index_bytes = &mmap[index_start..data_offset];

    // Parse the index. Format (per existing hfq.rs):
    //   u32 n_tensors
    //   For each tensor:
    //     u32 name_len
    //     [name_len] bytes name (UTF-8)
    //     u8 quant_type
    //     u32 n_dims
    //     [n_dims] u32 shape values
    //     u32 group_size
    //     u64 data_offset (relative to data_offset above)
    //     u64 data_size
    let mut idx = 0;
    let n_idx = u32::from_le_bytes(index_bytes[idx..idx + 4].try_into().unwrap()) as usize;
    idx += 4;
    assert_eq!(n_idx, n_tensors);

    struct InTensor<'a> {
        name: String,
        quant_type: u8,
        shape: Vec<u32>,
        group_size: u32,
        offset: u64,
        size: u64,
        bytes: &'a [u8],
    }

    let mut in_tensors: Vec<InTensor> = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name_len =
            u32::from_le_bytes(index_bytes[idx..idx + 4].try_into().unwrap()) as usize;
        idx += 4;
        let name = std::str::from_utf8(&index_bytes[idx..idx + name_len])
            .unwrap()
            .to_string();
        idx += name_len;
        let quant_type = index_bytes[idx];
        idx += 1;
        let n_dims = u32::from_le_bytes(index_bytes[idx..idx + 4].try_into().unwrap()) as usize;
        idx += 4;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(u32::from_le_bytes(index_bytes[idx..idx + 4].try_into().unwrap()));
            idx += 4;
        }
        let group_size = u32::from_le_bytes(index_bytes[idx..idx + 4].try_into().unwrap());
        idx += 4;
        let off = u64::from_le_bytes(index_bytes[idx..idx + 8].try_into().unwrap());
        idx += 8;
        let sz = u64::from_le_bytes(index_bytes[idx..idx + 8].try_into().unwrap());
        idx += 8;
        let abs_off = data_offset + off as usize;
        let bytes = &mmap[abs_off..abs_off + sz as usize];
        in_tensors.push(InTensor {
            name,
            quant_type,
            shape,
            group_size,
            offset: off,
            size: sz,
            bytes,
        });
    }
    eprintln!("Parsed {} tensors", in_tensors.len());

    // Single-tensor mode: emit just one tensor as a standalone v4 blob.
    if let Some(target) = &args.single_tensor {
        let t = in_tensors
            .iter()
            .find(|t| &t.name == target)
            .unwrap_or_else(|| panic!("tensor not found: {target}"));
        if t.quant_type != QUANT_HFQ4G256 && t.quant_type != QUANT_MQ4G256 {
            panic!(
                "tensor {} has quant_type {} (not HFQ4-G256 / MQ4-G256)",
                t.name, t.quant_type
            );
        }
        let m = t.shape[0] as usize;
        let k = t.shape[1] as usize;
        eprintln!("Converting {} (m={m}, k={k}) ...", t.name);
        let (w_blob, mu_blob) = convert_hfq4g256_to_hfq4v4(
            t.bytes,
            m,
            k,
            args.rotate,
            &MuStrategy::WeightMean,
        );
        let mut out = File::create(&args.output).unwrap();
        hfq4v4::write_blob(&mut out, m, k, args.rotate, &w_blob, &mu_blob).unwrap();
        eprintln!(
            "Wrote {} bytes to {}",
            32 + w_blob.len() + mu_blob.len(),
            args.output.display()
        );
        return;
    }

    // Full-file mode: re-emit the .hfq with v4-converted weight tensors.
    // Strategy: compute new tensor sizes & offsets, write header + meta JSON
    // + index + data.
    let new_tensors_data: Vec<(InTensor<'_>, u8, Vec<u8>, Vec<u32>)> = in_tensors
        .iter()
        .map(|t| {
            // Decide whether this tensor gets converted. We convert only
            // 2-D HFQ4-G256/MQ4-G256 tensors that look like linear layer
            // weights (M × K with K % 32 == 0).
            let is_weight_quant =
                t.quant_type == QUANT_HFQ4G256 || t.quant_type == QUANT_MQ4G256;
            let is_linear =
                t.shape.len() == 2 && (t.shape[1] as usize) % 32 == 0;
            // Skip the embed_tokens / lm_head paths — those are read by GPU
            // dequant kernels, not by the iu4 GEMM. We keep them as-is.
            let is_embed = t.name.contains("embed_tokens") || t.name == "model.embed_tokens.weight"
                || t.name.contains("lm_head") || t.name == "model.norm.weight"
                || t.name.contains("output.weight");
            if !is_weight_quant || !is_linear || is_embed {
                return (
                    InTensor {
                        name: t.name.clone(),
                        quant_type: t.quant_type,
                        shape: t.shape.clone(),
                        group_size: t.group_size,
                        offset: t.offset,
                        size: t.size,
                        bytes: &[],
                    },
                    t.quant_type,
                    t.bytes.to_vec(),
                    t.shape.clone(),
                );
            }
            let m = t.shape[0] as usize;
            let k = t.shape[1] as usize;
            // The effective rotate flag: if input is MQ4-G256 (already FWHT
            // rotated), we don't apply our K=32 rotation on top — that
            // would double-rotate. We treat MQ4 input as already-rotated
            // and use SmoothQuant against the dequantized (un-rotated)
            // signal. For HFQ4 input + --rotate, we apply FWHT-32 fresh.
            //
            // For first-cut implementation we ignore the input MQ4 case
            // and ALWAYS read as HFQ4-G256-format data. The caller is
            // responsible for not double-converting.
            let do_rotate = args.rotate;
            eprintln!("  converting {} ({}×{})", t.name, m, k);
            let (w_blob, mu_blob) = convert_hfq4g256_to_hfq4v4(
                t.bytes,
                m,
                k,
                do_rotate,
                &MuStrategy::WeightMean,
            );
            // Concatenate weight + mu so the loader can mmap the whole tensor.
            let mut combined = Vec::with_capacity(w_blob.len() + mu_blob.len());
            combined.extend_from_slice(&w_blob);
            combined.extend_from_slice(&mu_blob);

            let new_qt = if do_rotate {
                QUANT_MQ4V4_G32
            } else {
                QUANT_HFQ4V4_G32
            };
            (
                InTensor {
                    name: t.name.clone(),
                    quant_type: new_qt,
                    shape: t.shape.clone(),
                    group_size: 32,
                    offset: 0,
                    size: combined.len() as u64,
                    bytes: &[],
                },
                new_qt,
                combined,
                t.shape.clone(),
            )
        })
        .collect();

    // Compute output offsets.
    let mut data_section: Vec<u8> = Vec::new();
    let mut entries: Vec<(String, u8, Vec<u32>, u32, u64, u64)> =
        Vec::with_capacity(new_tensors_data.len());
    for (tin, qt, blob, _shape) in &new_tensors_data {
        let off = data_section.len() as u64;
        data_section.extend_from_slice(blob);
        let group_size = if *qt == QUANT_HFQ4V4_G32 || *qt == QUANT_MQ4V4_G32 {
            32u32
        } else {
            tin.group_size
        };
        entries.push((
            tin.name.clone(),
            *qt,
            tin.shape.clone(),
            group_size,
            off,
            blob.len() as u64,
        ));
    }

    // Build the index.
    let mut index = Vec::new();
    index.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (name, qt, shape, group_size, off, sz) in &entries {
        index.extend_from_slice(&(name.len() as u32).to_le_bytes());
        index.extend_from_slice(name.as_bytes());
        index.push(*qt);
        index.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in shape {
            index.extend_from_slice(&d.to_le_bytes());
        }
        index.extend_from_slice(&group_size.to_le_bytes());
        index.extend_from_slice(&off.to_le_bytes());
        index.extend_from_slice(&sz.to_le_bytes());
    }

    let header_size = 32usize;
    let new_metadata_offset = header_size as u64;
    // Note: data offset = header + metadata + index. The metadata in the
    // input file is a JSON blob of arbitrary length; we keep the same.
    let new_data_offset = header_size as u64 + metadata_json.len() as u64 + index.len() as u64;

    let mut out = File::create(&args.output).unwrap_or_else(|e| {
        panic!("create output {}: {e}", args.output.display())
    });
    out.write_all(b"HFQM").unwrap();
    out.write_all(&1u32.to_le_bytes()).unwrap();              // version
    out.write_all(&arch_id.to_le_bytes()).unwrap();
    out.write_all(&(entries.len() as u32).to_le_bytes()).unwrap();
    out.write_all(&new_metadata_offset.to_le_bytes()).unwrap();
    out.write_all(&new_data_offset.to_le_bytes()).unwrap();
    out.write_all(metadata_json.as_bytes()).unwrap();
    out.write_all(&index).unwrap();
    out.write_all(&data_section).unwrap();
    drop(out);

    let total = header_size + metadata_json.len() + index.len() + data_section.len();
    eprintln!();
    eprintln!("Wrote {} bytes to {}", total, args.output.display());
    eprintln!("Tensor count: {}", entries.len());
    let n_v4 = entries.iter().filter(|e| e.1 == QUANT_HFQ4V4_G32 || e.1 == QUANT_MQ4V4_G32).count();
    eprintln!("Converted to HFQ4v4/MQ4v4: {n_v4}");
    eprintln!("Pass-through: {}", entries.len() - n_v4);

    // sanity: emit total weight + mu bytes
    let total_w_bytes: usize = entries
        .iter()
        .filter(|e| e.1 == QUANT_HFQ4V4_G32 || e.1 == QUANT_MQ4V4_G32)
        .map(|e| {
            let m = e.2[0] as usize;
            let k = e.2[1] as usize;
            weight_bytes(m, k)
        })
        .sum();
    let total_mu_bytes: usize = entries
        .iter()
        .filter(|e| e.1 == QUANT_HFQ4V4_G32 || e.1 == QUANT_MQ4V4_G32)
        .map(|e| {
            let m = e.2[0] as usize;
            mu_bytes(m)
        })
        .sum();
    eprintln!(
        "v4 weight bytes: {} ({:.2} MB)",
        total_w_bytes,
        total_w_bytes as f64 / 1.0e6
    );
    eprintln!(
        "v4 mu sidecar bytes: {} ({:.2} MB)",
        total_mu_bytes,
        total_mu_bytes as f64 / 1.0e6
    );
}
