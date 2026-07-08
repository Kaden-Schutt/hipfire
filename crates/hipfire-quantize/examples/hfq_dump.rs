//! hfq_dump: print the arch_id and tensor index of a hipfire `.hfq`/`.mq*`
//! container. Read-only inspector — no GPU, no quantization.
//!
//! Usage:
//!     hfq_dump <file.hfq> [name-substring-filter]
//!
//! Added to inspect DeepSeek-V4 MTP/DSpark sidecars: confirms exactly which
//! `mtp.0.*` tensors a container holds (e.g. whether `main_proj` / `markov_*` /
//! `confidence_head` DSpark tensors are present).

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: hfq_dump <file.hfq> [name-substring-filter]");
        std::process::exit(2);
    }
    let path = Path::new(&args[1]);
    let filter = args.get(2).map(|s| s.as_str());

    // open_at_offset(.,0) is the plain reader: no REAP env-overlay attach.
    let hfq = match HfqFile::open_at_offset(path, 0) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("failed to open {}: {e}", path.display());
            std::process::exit(1);
        }
    };

    println!("file      : {}", path.display());
    println!("arch_id   : {}", hfq.arch_id);
    let all = hfq.tensors();
    println!("n_tensors : {}", all.len());
    if !hfq.metadata_json.is_empty() {
        println!("metadata  : {}", hfq.metadata_json);
    }
    println!();

    let mut shown = 0usize;
    let mut total_bytes = 0u64;
    for t in all {
        total_bytes += t.data_size as u64;
        if let Some(f) = filter {
            if !t.name.contains(f) {
                continue;
            }
        }
        let mb = t.data_size as f64 / (1024.0 * 1024.0);
        println!(
            "{:<48} qt={:<3} g={:<4} shape={:?} {:.2} MiB",
            t.name, t.quant_type, t.group_size, t.shape, mb
        );
        shown += 1;
    }
    println!();
    println!(
        "shown {} / {} tensors, total container tensor bytes {:.2} GiB",
        shown,
        all.len(),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
}
