// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Rank DeepSeek4 `(layer, expert)` residency slots by observed route count.
//! This is a CPU-only capacity planner for preserved `DS4RTR01` dumps.

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

fn parse_counts(bytes: &[u8]) -> Result<(Vec<u64>, u64, usize), String> {
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
        for expert in ids {
            if expert >= EXPERTS {
                return Err(format!("route expert {expert} is outside 0..{EXPERTS}"));
            }
            counts[layer * EXPERTS + expert] += 1;
            members += 1;
        }
        for _ in 0..k {
            let _ = read_u32(bytes, &mut offset)?;
        }
        records += 1;
    }
    Ok((counts, members, records))
}

fn main() -> Result<(), String> {
    let mut path = None;
    let mut halves = false;
    let mut slot_bytes = None;
    let mut budget_bytes = None;
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
            value if path.is_none() => path = Some(PathBuf::from(value)),
            value => return Err(format!("unexpected argument {value:?}")),
        }
    }
    let path = path.ok_or(
        "usage: deepseek4_route_hotset DUMP [--halves] --slot-bytes N --budget-bytes N",
    )?;
    let slot_bytes = slot_bytes.ok_or("--slot-bytes is required")?;
    let budget_bytes = budget_bytes.ok_or("--budget-bytes is required")?;
    if slot_bytes == 0 {
        return Err("--slot-bytes must be nonzero".to_owned());
    }
    let bytes = std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let (counts, members, records) = parse_counts(&bytes)?;
    let (counts, members, records) = if halves {
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
        (first_counts, first_members, first_records)
    } else {
        (counts, members, records)
    };
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
    let slots = usize::try_from(budget_bytes / slot_bytes)
        .unwrap_or(usize::MAX)
        .min(ranked.len());
    let covered = ranked[..slots]
        .iter()
        .map(|(count, _, _)| *count)
        .sum::<u64>();
    println!(
        "records={records} members={members} slot_bytes={slot_bytes} budget_bytes={budget_bytes} slots={slots} coverage={:.6}%",
        100.0 * covered as f64 / members.max(1) as f64
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
