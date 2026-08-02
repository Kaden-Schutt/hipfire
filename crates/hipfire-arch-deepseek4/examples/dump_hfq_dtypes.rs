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
    // Optional second arg: substring filter, printing every matching tensor's
    // quant_type. Without it, fall back to the qt=1 layer-0 listing. The
    // filter is what answers "which tier is the shared expert in?", which
    // decides whether a routed-vs-shared gain constant still applies after a
    // re-quant moved one of those tiers.
    match std::env::args().nth(2) {
        Some(filter) => {
            println!("\n== tensors matching {filter:?} ==");
            for t in hfq.tensors() {
                if t.name.contains(&filter) {
                    println!("  qt={:<3} {:<50} shape={:?}", t.quant_type, t.name, t.shape);
                }
            }
        }
        None => {
            println!("\n== qt=1 (F16) tensors in layer 0 + non-layer scope ==");
            for t in hfq.tensors() {
                if t.quant_type == 1
                    && (t.name.contains("layers.0.") || !t.name.contains("layers."))
                {
                    let n: u64 = t.shape.iter().map(|&s| s as u64).product();
                    println!("  {:<46} shape={:?} elems={}", t.name, t.shape, n);
                }
            }
        }
    }
    Ok(())
}
