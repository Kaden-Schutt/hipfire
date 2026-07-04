//! W3a validation: parse a real mlir-aie xclbin and dump its section directory.
//! Run: `cargo run -p hipfire-xdna --example xclbin_dump -- <path/to/final.xclbin>`

use hipfire_xdna::xclbin::{self, Axlf};

fn kind_name(k: u32) -> &'static str {
    match k {
        xclbin::KIND_EMBEDDED_METADATA => "EMBEDDED_METADATA",
        xclbin::KIND_PDI => "PDI",
        xclbin::KIND_PARTITION_METADATA => "PARTITION_METADATA",
        xclbin::KIND_AIE_METADATA => "AIE_METADATA",
        xclbin::KIND_AIE_RESOURCES => "AIE_RESOURCES",
        xclbin::KIND_AIE_PARTITION => "AIE_PARTITION",
        _ => "?",
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: xclbin_dump <path/to/final.xclbin>");
        std::process::exit(2);
    });
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("read {path}: {e}");
        std::process::exit(1);
    });
    let axlf = Axlf::parse(&bytes).unwrap_or_else(|e| {
        eprintln!("parse {path}: {e}");
        std::process::exit(1);
    });
    println!(
        "{path}: {} bytes, {} sections",
        bytes.len(),
        axlf.sections.len()
    );
    for s in &axlf.sections {
        println!(
            "  kind={:>2} {:<20} name={:<12} offset={:>7} size={:>7}",
            s.kind,
            kind_name(s.kind),
            format!("\"{}\"", s.name),
            s.offset,
            s.size
        );
    }
    let have_pdi = axlf.section(xclbin::KIND_PDI).is_some();
    let have_part = axlf.section(xclbin::KIND_AIE_PARTITION).is_some();
    println!("PDI present: {have_pdi} | AIE_PARTITION present: {have_part}");
}
