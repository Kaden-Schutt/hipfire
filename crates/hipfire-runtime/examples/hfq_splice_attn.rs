//! Splice the attention projections of a DONOR .hfq into a BASE .hfq, producing
//! a new file. Used to build `mq4r` (redline): BASE=mq4p (Q8 attn + graded
//! experts) with its attention-projection tensors REPLACED by DONOR=mq4-fresh
//! (uniform MQ4 attention). Result = MQ4 attention + graded experts, same AWQ
//! basis (both quantized from the same imatrix/alpha), so the spliced block is
//! self-consistent. Everything non-attention (experts, conv1d, norms, router,
//! embed, lm_head) is kept from BASE verbatim.
//!
//! Swaps tensors whose name matches an attention-PROJECTION prefix (both the
//! `.weight` and any `.awq_scale` sidecar). Deliberately EXCLUDES
//! `linear_attn.conv1d` (Q8 is load-bearing for the gated-delta path), norms,
//! A_log, dt_bias — those stay BASE.
//!
//! .hfq format: see crates/hipfire-runtime/examples/hfq_split.rs.
//!
//!   hfq_splice_attn --base mq4p --donor mq4-fresh --out mq4r [--dry-run]

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::process::ExitCode;

struct TensorEntry {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data_offset_src: u64,
    data_size: u64,
}
struct ParsedHfq {
    version: u32,
    arch_id: u32,
    metadata_json_bytes: Vec<u8>,
    tensors: Vec<TensorEntry>,
}

/// Tensors to pull from the (uniform-MQ4) donor for the `mq4r` redline build:
/// attention projections AND the MoE gate-side (router + shared expert). Taking
/// the router + shared expert as MQ4 makes `gate_side_mq4` true, so the graded
/// MoE decode uses the FUSED gate path (fused_qkvza_hfq4g256 on one rotated x)
/// instead of separate per-component gemvs+rotates. The donor's MQ4 router is
/// FWHT-rotated (coherent — it's the shipping uniform-MQ4 SKU); "redline" trades
/// the Q8 router's precision for speed. NOT swapped: routed experts
/// (`mlp.experts.*`, kept graded from base), conv1d, norms, A_log, dt_bias,
/// embed/lm_head.
fn is_attn_proj(name: &str) -> bool {
    name.contains(".linear_attn.in_proj_")     // qkv, z, b, a (DeltaNet)
        || name.contains(".linear_attn.out_proj")
        || (name.contains(".self_attn.") && name.contains("_proj")) // full-attn q/k/v/o
        // MoE gate-side (NOT routed experts) → MQ4 to enable the fused gate path
        || name.ends_with(".mlp.gate.weight")              // router
        || name.ends_with(".mlp.shared_expert_gate.weight") // shared-expert scalar gate
        || name.contains(".mlp.shared_expert.")            // shared expert gate/up/down
}

fn parse_hfq(path: &PathBuf) -> std::io::Result<(File, ParsedHfq)> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 32];
    file.read_exact(&mut header)?;
    if &header[0..4] != b"HFQM" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not an .hfq file"));
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let arch_id = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let n_tensors = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let metadata_offset = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let data_offset = u64::from_le_bytes(header[24..32].try_into().unwrap());

    let meta_region_len = (data_offset - metadata_offset) as usize;
    let mut meta_region = vec![0u8; meta_region_len];
    file.seek(SeekFrom::Start(metadata_offset))?;
    file.read_exact(&mut meta_region)?;

    let (mut depth, mut in_str, mut esc, mut json_end) = (0i32, false, false, 0usize);
    for (i, &b) in meta_region.iter().enumerate() {
        if esc { esc = false; continue; }
        if b == b'\\' && in_str { esc = true; continue; }
        if b == b'"' { in_str = !in_str; continue; }
        if !in_str {
            if b == b'{' { depth += 1; }
            if b == b'}' { depth -= 1; if depth == 0 { json_end = i + 1; break; } }
        }
    }
    let metadata_json_bytes = meta_region[..json_end].to_vec();
    let idx_buf = meta_region[json_end..].to_vec();
    let idx_n = u32::from_le_bytes(idx_buf[0..4].try_into().unwrap()) as usize;
    assert_eq!(idx_n, n_tensors, "tensor count mismatch");

    let mut pos = 4usize;
    let mut tensors = Vec::with_capacity(n_tensors);
    let mut cumulative = data_offset;
    for _ in 0..n_tensors {
        let name_len = u16::from_le_bytes(idx_buf[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&idx_buf[pos..pos + name_len]).to_string();
        pos += name_len;
        let quant_type = idx_buf[pos]; pos += 1;
        let n_dims = idx_buf[pos] as usize; pos += 1;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims { shape.push(u32::from_le_bytes(idx_buf[pos..pos + 4].try_into().unwrap())); pos += 4; }
        let group_size = u32::from_le_bytes(idx_buf[pos..pos + 4].try_into().unwrap()); pos += 4;
        let data_size = u64::from_le_bytes(idx_buf[pos..pos + 8].try_into().unwrap()); pos += 8;
        tensors.push(TensorEntry { name, quant_type, shape, group_size, data_offset_src: cumulative, data_size });
        cumulative += data_size;
    }
    Ok((file, ParsedHfq { version, arch_id, metadata_json_bytes, tensors }))
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let (mut base, mut donor, mut out, mut dry) = (None, None, None, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--base" => base = args.next().map(PathBuf::from),
            "--donor" => donor = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--dry-run" => dry = true,
            s => { eprintln!("unknown arg {s}"); return ExitCode::from(1); }
        }
    }
    let (base, donor, out) = match (base, donor, out) {
        (Some(b), Some(d), Some(o)) => (b, d, o),
        _ => { eprintln!("usage: hfq_splice_attn --base <mq4p> --donor <mq4-fresh> --out <mq4r> [--dry-run]"); return ExitCode::from(1); }
    };

    let (mut base_f, base_p) = parse_hfq(&base).expect("parse base");
    let (mut donor_f, donor_p) = parse_hfq(&donor).expect("parse donor");
    eprintln!("base  : {} tensors, arch_id={}", base_p.tensors.len(), base_p.arch_id);
    eprintln!("donor : {} tensors, arch_id={}", donor_p.tensors.len(), donor_p.arch_id);

    // Build output plan: for each base tensor, swap to donor if attn-proj + donor has it (matching shape).
    use std::collections::HashMap;
    let donor_by_name: HashMap<&str, &TensorEntry> = donor_p.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    enum Src { Base(usize), Donor(usize) }
    let mut plan: Vec<Src> = Vec::with_capacity(base_p.tensors.len());
    let mut swapped = 0usize;
    let (mut swap_q8, mut swap_other) = (0usize, 0usize);
    // HIPFIRE_DEMOTE_MQ6=1: also pull the MQ6 (qt=15) ROUTED experts from the
    // donor (uniform MQ4) — demotes the hot tier MQ6->MQ4 to remove the MQ6-hot
    // decode BW floor. Produces a lighter "speed" redline (MQ4 hot/mid + MQ3-
    // Lloyd cold) at a hot-expert quality cost. Off by default.
    let demote_mq6 = std::env::var("HIPFIRE_DEMOTE_MQ6").is_ok();
    // HIPFIRE_UNIFORM_GATE_UP=1: pull ALL routed-expert gate_up_proj from the
    // donor (uniform MQ4), leaving down_proj graded. → uniform MQ4 gate_up GEMV
    // (fast) + merged kernel only for down. Removes the merged kernel from the
    // dominant gate_up step at a hot-expert gate_up quality cost.
    let uniform_gate_up = std::env::var("HIPFIRE_UNIFORM_GATE_UP").is_ok();
    for (i, bt) in base_p.tensors.iter().enumerate() {
        let demote = demote_mq6 && bt.quant_type == 15 && bt.name.contains(".mlp.experts.");
        let ug = uniform_gate_up && bt.name.contains(".mlp.experts.") && bt.name.contains("gate_up_proj");
        if is_attn_proj(&bt.name) || demote || ug {
            if let Some(dt) = donor_by_name.get(bt.name.as_str()) {
                if dt.shape == bt.shape {
                    if bt.quant_type == 3 { swap_q8 += 1; } else { swap_other += 1; }
                    if swapped < 8 {
                        eprintln!("  SWAP {} : base_qt={} -> donor_qt={} shape={:?}", bt.name, bt.quant_type, dt.quant_type, bt.shape);
                    }
                    swapped += 1;
                    let di = donor_p.tensors.iter().position(|t| t.name == dt.name).unwrap();
                    plan.push(Src::Donor(di));
                    continue;
                } else {
                    eprintln!("  WARN shape mismatch {} base={:?} donor={:?} — keeping base", bt.name, bt.shape, dt.shape);
                }
            } else {
                eprintln!("  WARN attn-proj {} not in donor — keeping base", bt.name);
            }
        }
        plan.push(Src::Base(i));
    }
    eprintln!("\nswapped {swapped} attn-proj tensors ({swap_q8} were Q8/qt=3, {swap_other} other); {} kept from base", base_p.tensors.len() - swapped);

    // resolve each plan entry to (name, quant_type, shape, group_size, data_size, src_file_offset, from_donor)
    let resolved: Vec<(&TensorEntry, bool)> = plan.iter().map(|s| match s {
        Src::Base(i) => (&base_p.tensors[*i], false),
        Src::Donor(i) => (&donor_p.tensors[*i], true),
    }).collect();

    if dry {
        let total: u64 = resolved.iter().map(|(t, _)| t.data_size).sum();
        eprintln!("dry-run: out would be {} tensors / {:.2} GB", resolved.len(), total as f64 / 1e9);
        return ExitCode::from(0);
    }

    // Write merged .hfq (metadata + index from base; data from base/donor per plan).
    let mut index_len = 4usize;
    for (t, _) in &resolved { index_len += 2 + t.name.len() + 1 + 1 + t.shape.len() * 4 + 4 + 8; }
    let metadata_offset = 32u64;
    let data_offset = metadata_offset + base_p.metadata_json_bytes.len() as u64 + index_len as u64;

    let mut of = OpenOptions::new().write(true).create(true).truncate(true).open(&out).expect("open out");
    let mut header = [0u8; 32];
    header[0..4].copy_from_slice(b"HFQM");
    header[4..8].copy_from_slice(&base_p.version.to_le_bytes());
    header[8..12].copy_from_slice(&base_p.arch_id.to_le_bytes());
    header[12..16].copy_from_slice(&(resolved.len() as u32).to_le_bytes());
    header[16..24].copy_from_slice(&metadata_offset.to_le_bytes());
    header[24..32].copy_from_slice(&data_offset.to_le_bytes());
    of.write_all(&header).unwrap();
    of.write_all(&base_p.metadata_json_bytes).unwrap();

    let mut idx = Vec::with_capacity(index_len);
    idx.extend_from_slice(&(resolved.len() as u32).to_le_bytes());
    for (t, _) in &resolved {
        idx.extend_from_slice(&(t.name.len() as u16).to_le_bytes());
        idx.extend_from_slice(t.name.as_bytes());
        idx.push(t.quant_type);
        idx.push(t.shape.len() as u8);
        for &d in &t.shape { idx.extend_from_slice(&d.to_le_bytes()); }
        idx.extend_from_slice(&t.group_size.to_le_bytes());
        idx.extend_from_slice(&t.data_size.to_le_bytes());
    }
    assert_eq!(idx.len(), index_len);
    of.write_all(&idx).unwrap();

    let mut buf = vec![0u8; 16 * 1024 * 1024];
    let mut total: u64 = 0;
    for (i, (t, from_donor)) in resolved.iter().enumerate() {
        let src = if *from_donor { &mut donor_f } else { &mut base_f };
        src.seek(SeekFrom::Start(t.data_offset_src)).unwrap();
        let mut rem = t.data_size;
        while rem > 0 {
            let want = std::cmp::min(rem as usize, buf.len());
            src.read_exact(&mut buf[..want]).unwrap();
            of.write_all(&buf[..want]).unwrap();
            rem -= want as u64;
            total += want as u64;
        }
        if (i + 1) % 2000 == 0 || i + 1 == resolved.len() {
            eprintln!("  wrote {}/{} tensors ({:.2} GB)", i + 1, resolved.len(), total as f64 / 1e9);
        }
    }
    of.sync_data().unwrap();
    eprintln!("done -> {}", out.display());
    ExitCode::from(0)
}
