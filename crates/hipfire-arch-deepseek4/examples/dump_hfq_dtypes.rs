use hipfire_runtime::hfq::HfqFile;
use std::collections::BTreeMap;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: dump_hfq_dtypes <path.hfq>")?;
    let hfq = HfqFile::open(Path::new(&path))?;
    let mut by_qt: BTreeMap<u8, (usize, u64)> = BTreeMap::new();
    for t in hfq.tensors() {
        let n: u64 = t.shape.iter().map(|&s| s as u64).product();
        let e = by_qt.entry(t.quant_type).or_insert((0, 0));
        e.0 += 1;
        e.1 += n;
    }
    println!("== quant_type summary (per HFQ header) ==");
    for (qt, (c, els)) in &by_qt {
        println!("  qt={qt}: {c} tensors, {els} total elems");
    }
    println!("\n== qt=1 (F16) tensors in layer 0 + non-layer scope ==");
    for t in hfq.tensors() {
        if t.quant_type == 1 && (t.name.contains("layers.0.") || !t.name.contains("layers.")) {
            let n: u64 = t.shape.iter().map(|&s| s as u64).product();
            println!("  {:<46} shape={:?} elems={}", t.name, t.shape, n);
        }
    }
    Ok(())
}
