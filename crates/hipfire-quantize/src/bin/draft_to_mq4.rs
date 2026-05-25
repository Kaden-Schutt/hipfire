// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use memmap2::Mmap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign << 31); }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
    }).collect()
}

fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 { x[i] *= signs1[i]; }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625;
    for i in 0..256 { x[i] *= scale * signs2[i]; }
}

fn quantize_mq4g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];
    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let mut group = [0.0f32; 256];
        group[..end - start].copy_from_slice(&f32_data[start..end]);
        cpu_fwht_256(&mut group, signs1, signs2);
        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };
        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());
        for i in 0..128 {
            let lo_q = ((group[2 * i] - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((group[2 * i + 1] - min_val) * inv_scale + 0.5) as u8;
            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }
    output
}

struct HfqInTensor {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
}

fn read_hfq(path: &Path) -> (u32, String, Vec<HfqInTensor>) {
    let file = File::open(path).expect("open input");
    let mmap = unsafe { Mmap::map(&file).expect("mmap") };
    assert_eq!(&mmap[0..4], HFQ_MAGIC, "bad magic");
    let _version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
    let arch_id = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
    let n_tensors = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
    let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
    let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;

    let meta_bytes = &mmap[metadata_offset..data_offset];
    let mut brace = 0i32;
    let mut in_str = false;
    let mut escape = false;
    let mut json_end = 0;
    for (i, &b) in meta_bytes.iter().enumerate() {
        if escape { escape = false; continue; }
        if b == b'\\' && in_str { escape = true; continue; }
        if b == b'"' { in_str = !in_str; continue; }
        if !in_str {
            if b == b'{' { brace += 1; }
            if b == b'}' { brace -= 1; if brace == 0 { json_end = i + 1; break; } }
        }
    }
    let metadata_json = String::from_utf8_lossy(&meta_bytes[..json_end]).to_string();

    let mut pos = metadata_offset + json_end;
    let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
    assert_eq!(idx_n, n_tensors);
    pos += 4;

    let mut out = Vec::with_capacity(n_tensors);
    let mut cumulative = data_offset;
    for _ in 0..n_tensors {
        let name_len = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).to_string();
        pos += name_len;
        let qt = mmap[pos]; pos += 1;
        let n_dims = mmap[pos] as usize; pos += 1;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        let group_size = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()); pos += 4;
        let data_size = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let data = mmap[cumulative..cumulative + data_size].to_vec();
        cumulative += data_size;
        out.push(HfqInTensor { name, quant_type: qt, shape, group_size, data });
    }
    (arch_id, metadata_json, out)
}

fn write_hfq(path: &Path, arch: u32, metadata_json: &str, tensors: &[HfqInTensor]) {
    let mut f = File::create(path).expect("create output");
    let meta = metadata_json.as_bytes();
    let metadata_offset = 32u64;
    let mut index = Vec::new();
    index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        let nb = t.name.as_bytes();
        index.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        index.extend_from_slice(nb);
        index.push(t.quant_type);
        index.push(t.shape.len() as u8);
        for &d in &t.shape { index.extend_from_slice(&d.to_le_bytes()); }
        index.extend_from_slice(&t.group_size.to_le_bytes());
        index.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
    }
    let data_start_unaligned = metadata_offset + meta.len() as u64 + index.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    f.write_all(HFQ_MAGIC).unwrap();
    f.write_all(&HFQ_VERSION.to_le_bytes()).unwrap();
    f.write_all(&arch.to_le_bytes()).unwrap();
    f.write_all(&(tensors.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&metadata_offset.to_le_bytes()).unwrap();
    f.write_all(&data_offset.to_le_bytes()).unwrap();
    f.write_all(meta).unwrap();
    f.write_all(&index).unwrap();
    let pad = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad]).unwrap();
    for t in tensors { f.write_all(&t.data).unwrap(); }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: draft_to_mq4 <input.hfq> <output.mq4>");
        std::process::exit(2);
    }
    let inp = Path::new(&args[1]);
    let outp = Path::new(&args[2]);

    eprintln!("reading {}...", inp.display());
    let (arch, meta, mut tensors) = read_hfq(inp);
    eprintln!("  arch_id={arch}  tensors={}", tensors.len());

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    let mut converted = 0usize;
    let mut skipped = 0usize;
    for t in &mut tensors {
        if t.quant_type != 1 {
            skipped += 1;
            continue;
        }
        if t.shape.len() != 2 {
            skipped += 1;
            continue;
        }
        let k = t.shape[1] as usize;
        if k % 256 != 0 {
            eprintln!("  SKIP {}: K={} not multiple of 256", t.name, k);
            skipped += 1;
            continue;
        }
        let m = t.shape[0] as usize;
        let numel = m * k;
        assert_eq!(t.data.len(), numel * 2, "{}: F16 byte-size mismatch", t.name);
        let mut f32_data = Vec::with_capacity(numel);
        for c in t.data.chunks_exact(2) {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32_data.push(f16_to_f32(bits));
        }
        let mq4 = quantize_mq4g256(&f32_data, &signs1, &signs2);
        eprintln!("  MQ4 {}: F16 {} MiB -> MQ4 {} MiB  ({}x{})",
            t.name, t.data.len()/(1024*1024), mq4.len()/(1024*1024), m, k);
        t.quant_type = 13;
        t.group_size = 256;
        t.data = mq4;
        converted += 1;
    }
    eprintln!("converted {converted} tensors, skipped {skipped}");

    eprintln!("writing {}...", outp.display());
    write_hfq(outp, arch, &meta, &tensors);
    eprintln!("done. output size: {} MiB",
        std::fs::metadata(outp).unwrap().len() / (1024*1024));
}
