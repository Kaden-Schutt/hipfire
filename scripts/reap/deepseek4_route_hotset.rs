// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Rank DeepSeek4 `(layer, expert)` residency slots by observed route count.
//! This is a CPU-only capacity planner for preserved `DS4RTR01` dumps.

use std::collections::BTreeSet;
use std::convert::{TryFrom, TryInto};
use std::path::PathBuf;

const LAYERS: usize = 43;
const EXPERTS: usize = 256;

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "route dump offset overflow".to_owned())?;
    let raw: [u8; 4] = bytes
        .get(*offset..end)
        .ok_or_else(|| "truncated route dump u32".to_owned())?
        .try_into()
        .unwrap();
    *offset = end;
    Ok(u32::from_le_bytes(raw))
}

fn parse_counts_range(
    bytes: &[u8],
    start_record: usize,
    end_record: Option<usize>,
) -> Result<(Vec<u64>, u64, usize, usize), String> {
    if bytes.get(..8) != Some(b"DS4RTR01") {
        return Err("bad DS4 route magic".to_owned());
    }
    let mut counts = vec![0_u64; LAYERS * EXPERTS];
    let mut offset = 8_usize;
    let mut members = 0_u64;
    let mut records = 0_usize;
    while offset < bytes.len() {
        let layer = read_u32(bytes, &mut offset)? as usize;
        let k = read_u32(bytes, &mut offset)? as usize;
        if layer >= LAYERS {
            return Err(format!("route layer {layer} is outside 0..{LAYERS}"));
        }
        let mut ids = Vec::with_capacity(k);
        for _ in 0..k {
            ids.push(read_u32(bytes, &mut offset)? as usize);
        }
        let selected = records >= start_record && end_record.is_none_or(|end| records < end);
        for expert in ids {
            if expert >= EXPERTS {
                return Err(format!("route expert {expert} is outside 0..{EXPERTS}"));
            }
            if selected {
                counts[layer * EXPERTS + expert] += 1;
                members += 1;
            }
        }
        for _ in 0..k {
            let _ = read_u32(bytes, &mut offset)?;
        }
        records += 1;
    }
    let selected_records = records
        .min(end_record.unwrap_or(records))
        .saturating_sub(start_record);
    Ok((counts, members, selected_records, records))
}

fn parse_counts(bytes: &[u8]) -> Result<(Vec<u64>, u64, usize), String> {
    let (counts, members, selected_records, _) = parse_counts_range(bytes, 0, None)?;
    Ok((counts, members, selected_records))
}

fn hot_count_histogram_range(
    bytes: &[u8],
    start_record: usize,
    end_record: usize,
    selected: &BTreeSet<usize>,
) -> Result<Vec<u64>, String> {
    if bytes.get(..8) != Some(b"DS4RTR01") {
        return Err("bad DS4 route magic".to_owned());
    }
    let mut histogram = Vec::<u64>::new();
    let mut offset = 8_usize;
    let mut record = 0_usize;
    while offset < bytes.len() {
        let layer = read_u32(bytes, &mut offset)? as usize;
        let k = read_u32(bytes, &mut offset)? as usize;
        if layer >= LAYERS {
            return Err(format!("route layer {layer} is outside 0..{LAYERS}"));
        }
        let mut hot = 0_usize;
        for _ in 0..k {
            let expert = read_u32(bytes, &mut offset)? as usize;
            if expert >= EXPERTS {
                return Err(format!("route expert {expert} is outside 0..{EXPERTS}"));
            }
            hot += usize::from(selected.contains(&(layer * EXPERTS + expert)));
        }
        for _ in 0..k {
            let _ = read_u32(bytes, &mut offset)?;
        }
        if record >= start_record && record < end_record {
            if histogram.len() <= hot {
                histogram.resize(hot + 1, 0);
            }
            histogram[hot] += 1;
        }
        record += 1;
    }
    if end_record > record {
        return Err(format!(
            "histogram range ends at record {end_record}, dump has {record}"
        ));
    }
    Ok(histogram)
}

fn print_hot_count_histogram(label: &str, histogram: &[u64]) {
    let records = histogram.iter().sum::<u64>();
    let members = histogram
        .iter()
        .enumerate()
        .map(|(hot, count)| hot as u64 * count)
        .sum::<u64>();
    let encoded = histogram
        .iter()
        .enumerate()
        .map(|(hot, count)| format!("{hot}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{label}_hot_count_histogram={encoded} records={records} mean_hot_slots={:.6}",
        members as f64 / records.max(1) as f64
    );
}

fn rank(counts: Vec<u64>) -> Vec<(u64, usize, usize)> {
    let mut ranked = counts
        .into_iter()
        .enumerate()
        .map(|(slot, count)| (count, slot / EXPERTS, slot % EXPERTS))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    ranked
}

fn coverage(ranked: &[(u64, usize, usize)], slots: usize, members: u64) -> f64 {
    let covered = ranked[..slots]
        .iter()
        .map(|(count, _, _)| *count)
        .sum::<u64>();
    100.0 * covered as f64 / members.max(1) as f64
}

fn main() -> Result<(), String> {
    let mut path = None;
    let mut halves = false;
    let mut slot_bytes = None;
    let mut budget_bytes = None;
    let mut train_tokens = None;
    let mut eval_tokens = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--halves" => halves = true,
            "--slot-bytes" => {
                slot_bytes = Some(
                    arguments
                        .next()
                        .ok_or("--slot-bytes requires N")?
                        .parse::<u64>()
                        .map_err(|error| format!("invalid --slot-bytes: {error}"))?,
                )
            }
            "--budget-bytes" => {
                budget_bytes = Some(
                    arguments
                        .next()
                        .ok_or("--budget-bytes requires N")?
                        .parse::<u64>()
                        .map_err(|error| format!("invalid --budget-bytes: {error}"))?,
                )
            }
            "--train-tokens" => {
                train_tokens = Some(
                    arguments
                        .next()
                        .ok_or("--train-tokens requires N")?
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --train-tokens: {error}"))?,
                )
            }
            "--eval-tokens" => {
                eval_tokens = Some(
                    arguments
                        .next()
                        .ok_or("--eval-tokens requires N")?
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --eval-tokens: {error}"))?,
                )
            }
            value if path.is_none() => path = Some(PathBuf::from(value)),
            value => return Err(format!("unexpected argument {value:?}")),
        }
    }
    let path = path.ok_or(
        "usage: deepseek4_route_hotset DUMP [--halves] --slot-bytes N --budget-bytes N [--train-tokens N --eval-tokens N]",
    )?;
    let slot_bytes = slot_bytes.ok_or("--slot-bytes is required")?;
    let budget_bytes = budget_bytes.ok_or("--budget-bytes is required")?;
    if slot_bytes == 0 {
        return Err("--slot-bytes must be nonzero".to_owned());
    }
    let bytes =
        std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let (_, _, records) = parse_counts(&bytes)?;
    let sample = if halves {
        if !records.is_multiple_of(2) {
            return Err(format!("combined dump has odd record count {records}"));
        }
        let half_records = records / 2;
        let bytes_per_record = (bytes.len() - 8) / records;
        let split = 8 + half_records * bytes_per_record;
        let mut first = Vec::with_capacity(split);
        first.extend_from_slice(&bytes[..split]);
        let (first_counts, first_members, first_records) = parse_counts(&first)?;
        let mut second = Vec::with_capacity(bytes.len() - split + 8);
        second.extend_from_slice(b"DS4RTR01");
        second.extend_from_slice(&bytes[split..]);
        let (second_counts, second_members, second_records) = parse_counts(&second)?;
        if first_counts != second_counts || first_members != second_members {
            return Err("route dump halves do not have identical residency counts".to_owned());
        }
        if first_records != second_records {
            return Err("route dump halves have different record counts".to_owned());
        }
        first
    } else {
        bytes
    };
    let (all_counts, all_members, records) = parse_counts(&sample)?;
    let slots = usize::try_from(budget_bytes / slot_bytes)
        .unwrap_or(usize::MAX)
        .min(all_counts.len());
    let (ranked, members, label) = match (train_tokens, eval_tokens) {
        (None, None) => (rank(all_counts), all_members, "all"),
        (Some(train), Some(eval)) => {
            let train_records = train
                .checked_mul(LAYERS)
                .ok_or("train record count overflow")?;
            let eval_records = eval
                .checked_mul(LAYERS)
                .ok_or("eval record count overflow")?;
            if train_records + eval_records != records {
                return Err(format!(
                    "train/eval tokens imply {} records, dump has {records}",
                    train_records + eval_records
                ));
            }
            let (train_counts, train_members, _, _) =
                parse_counts_range(&sample, 0, Some(train_records))?;
            let (eval_counts, eval_members, _, _) =
                parse_counts_range(&sample, train_records, Some(records))?;
            let ranked = rank(train_counts);
            let selected = ranked[..slots]
                .iter()
                .map(|(_, layer, expert)| layer * EXPERTS + expert)
                .collect::<BTreeSet<_>>();
            let eval_covered = eval_counts
                .iter()
                .enumerate()
                .filter(|(slot, _)| selected.contains(slot))
                .map(|(_, count)| *count)
                .sum::<u64>();
            let oracle_eval = coverage(&rank(eval_counts), slots, eval_members);
            println!(
                "train_tokens={train} eval_tokens={eval} train_coverage={:.6}% eval_coverage={:.6}% eval_oracle_coverage={oracle_eval:.6}%",
                coverage(&ranked, slots, train_members),
                100.0 * eval_covered as f64 / eval_members.max(1) as f64,
            );
            let train_histogram = hot_count_histogram_range(&sample, 0, train_records, &selected)?;
            let eval_histogram =
                hot_count_histogram_range(&sample, train_records, records, &selected)?;
            print_hot_count_histogram("train", &train_histogram);
            print_hot_count_histogram("eval", &eval_histogram);
            (ranked, train_members, "train")
        }
        _ => return Err("--train-tokens and --eval-tokens must be supplied together".to_owned()),
    };
    println!(
        "records={records} {label}_members={members} slot_bytes={slot_bytes} budget_bytes={budget_bytes} slots={slots} {label}_coverage={:.6}%",
        coverage(&ranked, slots, members)
    );
    for (rank, &(count, layer, expert)) in ranked.iter().take(32).enumerate() {
        println!(
            "rank={:02} layer={layer:02} expert={expert:03} count={count} share={:.6}%",
            rank + 1,
            100.0 * count as f64 / members.max(1) as f64
        );
    }
    Ok(())
}
