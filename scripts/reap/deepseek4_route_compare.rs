use std::collections::BTreeMap;
use std::convert::TryInto;
use std::path::Path;

#[derive(Clone)]
struct Record {
    layer: u32,
    ids: Vec<i32>,
    weights: Vec<f32>,
}

#[derive(Default)]
struct Stats {
    records: u64,
    ordered_exact: u64,
    set_exact: u64,
    top1_exact: u64,
    member_hits: u64,
    members: u64,
    weight_l1: f64,
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
    let end = *offset + 4;
    let raw: [u8; 4] = bytes
        .get(*offset..end)
        .ok_or_else(|| "truncated u32".to_string())?
        .try_into()
        .unwrap();
    *offset = end;
    Ok(u32::from_le_bytes(raw))
}

fn read(path: &Path) -> Result<Vec<Record>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.get(..8) != Some(b"DS4RTR01") {
        return Err(format!("{}: bad DS4 route magic", path.display()));
    }
    let mut offset = 8usize;
    let mut records = Vec::new();
    while offset < bytes.len() {
        let layer = read_u32(&bytes, &mut offset)?;
        let k = read_u32(&bytes, &mut offset)? as usize;
        let mut ids = Vec::with_capacity(k);
        for _ in 0..k {
            ids.push(read_u32(&bytes, &mut offset)? as i32);
        }
        let mut weights = Vec::with_capacity(k);
        for _ in 0..k {
            weights.push(f32::from_bits(read_u32(&bytes, &mut offset)?));
        }
        records.push(Record {
            layer,
            ids,
            weights,
        });
    }
    Ok(records)
}

fn accumulate(stats: &mut Stats, base: &Record, candidate: &Record) -> Result<(), String> {
    if base.layer != candidate.layer || base.ids.len() != candidate.ids.len() {
        return Err(format!(
            "record mismatch: base layer {} k {}, candidate layer {} k {}",
            base.layer,
            base.ids.len(),
            candidate.layer,
            candidate.ids.len()
        ));
    }
    stats.records += 1;
    stats.members += base.ids.len() as u64;
    if base.ids == candidate.ids {
        stats.ordered_exact += 1;
    }
    if base.ids.first() == candidate.ids.first() {
        stats.top1_exact += 1;
    }
    let mut base_set = base.ids.clone();
    let mut candidate_set = candidate.ids.clone();
    base_set.sort_unstable();
    candidate_set.sort_unstable();
    if base_set == candidate_set {
        stats.set_exact += 1;
    }
    stats.member_hits += base_set
        .iter()
        .filter(|id| candidate_set.binary_search(id).is_ok())
        .count() as u64;

    for (&id, &weight) in base.ids.iter().zip(&base.weights) {
        let candidate_weight = candidate
            .ids
            .iter()
            .position(|candidate_id| *candidate_id == id)
            .map(|index| candidate.weights[index])
            .unwrap_or(0.0);
        stats.weight_l1 += (weight as f64 - candidate_weight as f64).abs();
    }
    Ok(())
}

fn print_stats(label: &str, stats: &Stats) {
    let pct = |n: u64, d: u64| 100.0 * n as f64 / d.max(1) as f64;
    println!(
        "{label}: records={} ordered_exact={:.2}% set_exact={:.2}% \
         top1_exact={:.2}% member_recall={:.2}% mean_weight_l1={:.6}",
        stats.records,
        pct(stats.ordered_exact, stats.records),
        pct(stats.set_exact, stats.records),
        pct(stats.top1_exact, stats.records),
        pct(stats.member_hits, stats.members),
        stats.weight_l1 / stats.records.max(1) as f64,
    );
}

fn print_first_differences(base: &[Record], candidate: &[Record]) {
    if let Some((record, (base, candidate))) = base
        .iter()
        .zip(candidate)
        .enumerate()
        .find(|(_, (base, candidate))| base.ids != candidate.ids)
    {
        println!(
            "first-id-difference: record={record} layer={} base={:?} candidate={:?}",
            base.layer, base.ids, candidate.ids
        );
    } else {
        println!("first-id-difference: none");
    }

    if let Some((record, (base, candidate))) =
        base.iter()
            .zip(candidate)
            .enumerate()
            .find(|(_, (base, candidate))| {
                base.weights
                    .iter()
                    .map(|weight| weight.to_bits())
                    .ne(candidate.weights.iter().map(|weight| weight.to_bits()))
            })
    {
        println!(
            "first-weight-difference: record={record} layer={} base={:?} candidate={:?}",
            base.layer, base.weights, candidate.weights
        );
    } else {
        println!("first-weight-difference: none");
    }
}

fn main() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let first = args
        .next()
        .ok_or("usage: route_compare BASE CANDIDATE | --halves COMBINED")?;
    let (base, candidate) = if first == "--halves" {
        let combined_path = args
            .next()
            .ok_or("usage: route_compare --halves COMBINED")?;
        if args.next().is_some() {
            return Err("usage: route_compare --halves COMBINED".to_string());
        }
        let combined = read(Path::new(&combined_path))?;
        if !combined.len().is_multiple_of(2) {
            return Err(format!(
                "combined route record count {} is not even",
                combined.len()
            ));
        }
        let half = combined.len() / 2;
        (combined[..half].to_vec(), combined[half..].to_vec())
    } else {
        let candidate_path = args.next().ok_or("usage: route_compare BASE CANDIDATE")?;
        if args.next().is_some() {
            return Err("usage: route_compare BASE CANDIDATE".to_string());
        }
        (read(Path::new(&first))?, read(Path::new(&candidate_path))?)
    };
    if base.len() != candidate.len() {
        return Err(format!(
            "record count mismatch: base {} candidate {}",
            base.len(),
            candidate.len()
        ));
    }

    print_first_differences(&base, &candidate);

    let mut all = Stats::default();
    let mut hash = Stats::default();
    let mut score = Stats::default();
    let mut layers: BTreeMap<u32, Stats> = BTreeMap::new();
    for (base_record, candidate_record) in base.iter().zip(&candidate) {
        accumulate(&mut all, base_record, candidate_record)?;
        accumulate(
            if base_record.layer < 3 {
                &mut hash
            } else {
                &mut score
            },
            base_record,
            candidate_record,
        )?;
        accumulate(
            layers.entry(base_record.layer).or_default(),
            base_record,
            candidate_record,
        )?;
    }

    print_stats("all", &all);
    print_stats("hash-routed", &hash);
    print_stats("score-routed", &score);
    for (layer, stats) in layers {
        print_stats(&format!("layer-{layer:02}"), &stats);
    }
    Ok(())
}
