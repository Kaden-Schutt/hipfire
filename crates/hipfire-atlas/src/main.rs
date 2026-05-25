// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire-atlas` CLI — corpus inspection, legacy stdout parsing,
//! render-fit, suggestion, and task-bundle generation.
//!
//! Collection itself happens inside bench/inference binaries via
//! `--emit-atlas <path>`. This binary covers everything *around* that
//! collection: reading corpora, parsing legacy bench captures into rows,
//! ranking, rendering, and generating task bundles.
//!
//! Status: transitional. `origin/master` currently carries this Rust
//! Atlas crate, but the project direction is Python-first for the Atlas
//! CLI/analyzer. Keep this binary as a compatibility bridge while the
//! useful schema/task/eval behavior is moved back to Python; do not grow
//! the Rust port as the long-term agent or user surface.
//!
//! For ad-hoc analysis at scale, use `scripts/kernel_atlas.py` on the
//! HIPa branch instead — pandas/notebook iteration is faster there for
//! large rankings.

use hipfire_atlas::{
    eval::eval_task_file,
    load_rows,
    parse::{bench_rows_from_output, dflash_row_from_output},
    render::render_fit_view,
    schema::{load_row, AtlasRow},
    suggest::{suggestions_for_row, suggestions_markdown},
    task::{pytorch_task, task_from_row},
};
use std::fs;

fn print_usage() {
    eprintln!("hipfire-atlas — kernel atlas corpus tool");
    eprintln!();
    eprintln!("Corpus inspection:");
    eprintln!("  hipfire-atlas read <path.jsonl>             show row count + first row pretty");
    eprintln!("  hipfire-atlas count <path.jsonl>            print number of rows");
    eprintln!("  hipfire-atlas head <path.jsonl> [N]         print first N rows (default 1)");
    eprintln!();
    eprintln!("Legacy stdout parsing (migration bridge):");
    eprintln!("  hipfire-atlas parse-bench <bench_stdout.txt> <out.jsonl>");
    eprintln!("                                              parse a captured bench_qwen35_mq4");
    eprintln!("                                              stdout into prefill + decode_ar rows");
    eprintln!("  hipfire-atlas parse-dflash <dflash_stdout.txt> <out.jsonl>");
    eprintln!("                                              parse a captured dflash_spec_demo");
    eprintln!("                                              stdout into a decode_dflash row");
    eprintln!();
    eprintln!("Analysis:");
    eprintln!("  hipfire-atlas render-fit <path.jsonl> [INDEX]  ASCII fit view of one row");
    eprintln!("  hipfire-atlas suggest <path.jsonl> [INDEX] [MAX]  ranked tuning suggestions");
    eprintln!("  hipfire-atlas task <path.jsonl> [INDEX]     emit TaskBundle JSON for a row");
    eprintln!("  hipfire-atlas task-pytorch <name> <op> <shape> <dtype> <eval_cmd>");
    eprintln!("                                              emit a TaskBundle for a PyTorch shape");
    eprintln!();
    eprintln!("Eval:");
    eprintln!("  hipfire-atlas eval <task.json> [CWD]        run task's correctness+eval commands");
    eprintln!();
    eprintln!("For collection itself: bench binaries' --emit-atlas <path> flag.");
    eprintln!("For large-scale analysis: scripts/kernel_atlas.py on the HIPa branch.");
}

fn cmd_count(path: &str) -> Result<(), String> {
    let rows = load_rows(path)?;
    println!("{}", rows.len());
    Ok(())
}

fn cmd_head(path: &str, n: usize) -> Result<(), String> {
    let rows = load_rows(path)?;
    for row in rows.iter().take(n) {
        let pretty = serde_json::to_string_pretty(row)
            .map_err(|e| format!("serialize row: {e}"))?;
        println!("{pretty}");
    }
    Ok(())
}

fn cmd_read(path: &str) -> Result<(), String> {
    let rows = load_rows(path)?;
    println!("rows: {}", rows.len());
    if let Some(first) = rows.first() {
        let pretty = serde_json::to_string_pretty(first)
            .map_err(|e| format!("serialize first row: {e}"))?;
        println!("first row:");
        println!("{pretty}");
    }
    Ok(())
}

fn cmd_parse_bench(stdout_path: &str, out_path: &str) -> Result<(), String> {
    let text = fs::read_to_string(stdout_path)
        .map_err(|e| format!("read {stdout_path}: {e}"))?;
    let rows = bench_rows_from_output(&text)?;
    write_rows_jsonl(&rows, out_path)?;
    println!("wrote {} rows to {out_path}", rows.len());
    Ok(())
}

fn cmd_parse_dflash(stdout_path: &str, out_path: &str) -> Result<(), String> {
    let text = fs::read_to_string(stdout_path)
        .map_err(|e| format!("read {stdout_path}: {e}"))?;
    let row = dflash_row_from_output(&text)?;
    write_rows_jsonl(&[row], out_path)?;
    println!("wrote 1 row to {out_path}");
    Ok(())
}

fn cmd_render_fit(path: &str, idx: usize) -> Result<(), String> {
    let row = load_row(path, idx)?;
    println!("{}", render_fit_view(&row));
    Ok(())
}

fn cmd_suggest(path: &str, idx: usize, max: usize) -> Result<(), String> {
    let row = load_row(path, idx)?;
    let suggestions = suggestions_for_row(&row, max);
    if suggestions.is_empty() {
        println!("(no suggestions for row {idx})");
    } else {
        println!("{}", suggestions_markdown(&suggestions));
    }
    Ok(())
}

fn cmd_task(path: &str, idx: usize) -> Result<(), String> {
    let row = load_row(path, idx)?;
    let bundle = task_from_row(&row, None, Vec::new(), Vec::new());
    let pretty = serde_json::to_string_pretty(&bundle)
        .map_err(|e| format!("serialize task: {e}"))?;
    println!("{pretty}");
    Ok(())
}

fn cmd_task_pytorch(
    name: &str,
    op: &str,
    shape: &str,
    dtype: &str,
    eval_cmd: &str,
) -> Result<(), String> {
    let shapes = shape
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let bundle = pytorch_task(
        name.to_string(),
        op.to_string(),
        shapes,
        dtype.to_string(),
        eval_cmd.to_string(),
        None,
        Vec::new(),
    );
    let pretty = serde_json::to_string_pretty(&bundle)
        .map_err(|e| format!("serialize task: {e}"))?;
    println!("{pretty}");
    Ok(())
}

fn cmd_eval(task_path: &str, cwd: Option<&str>) -> Result<(), String> {
    let result = eval_task_file(task_path, cwd)?;
    let pretty = serde_json::to_string_pretty(&result)
        .map_err(|e| format!("serialize eval result: {e}"))?;
    println!("{pretty}");
    if result.status != "pass" {
        return Err(format!("task {} failed", result.task_id));
    }
    Ok(())
}

fn write_rows_jsonl(rows: &[AtlasRow], path: &str) -> Result<(), String> {
    let lines: Vec<String> = rows
        .iter()
        .map(|row| serde_json::to_string(row))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("serialize row: {e}"))?;
    let body = lines.join("\n") + "\n";
    fs::write(path, body).map_err(|e| format!("write {path}: {e}"))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return Err("missing subcommand".into());
    }
    match args[1].as_str() {
        "read" if args.len() == 3 => cmd_read(&args[2]),
        "count" if args.len() == 3 => cmd_count(&args[2]),
        "head" if args.len() == 3 => cmd_head(&args[2], 1),
        "head" if args.len() == 4 => {
            let n: usize = args[3].parse().map_err(|e| format!("invalid N: {e}"))?;
            cmd_head(&args[2], n)
        }
        "parse-bench" if args.len() == 4 => cmd_parse_bench(&args[2], &args[3]),
        "parse-dflash" if args.len() == 4 => cmd_parse_dflash(&args[2], &args[3]),
        "render-fit" if args.len() >= 3 && args.len() <= 4 => {
            let idx: usize = if args.len() == 4 {
                args[3].parse().map_err(|e| format!("invalid INDEX: {e}"))?
            } else {
                0
            };
            cmd_render_fit(&args[2], idx)
        }
        "suggest" if args.len() >= 3 && args.len() <= 5 => {
            let idx: usize = if args.len() >= 4 {
                args[3].parse().map_err(|e| format!("invalid INDEX: {e}"))?
            } else {
                0
            };
            let max: usize = if args.len() == 5 {
                args[4].parse().map_err(|e| format!("invalid MAX: {e}"))?
            } else {
                4
            };
            cmd_suggest(&args[2], idx, max)
        }
        "task" if args.len() >= 3 && args.len() <= 4 => {
            let idx: usize = if args.len() == 4 {
                args[3].parse().map_err(|e| format!("invalid INDEX: {e}"))?
            } else {
                0
            };
            cmd_task(&args[2], idx)
        }
        "task-pytorch" if args.len() == 7 => {
            cmd_task_pytorch(&args[2], &args[3], &args[4], &args[5], &args[6])
        }
        "eval" if args.len() == 3 => cmd_eval(&args[2], None),
        "eval" if args.len() == 4 => cmd_eval(&args[2], Some(&args[3])),
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        other => {
            print_usage();
            Err(format!("unknown or malformed subcommand: {other}"))
        }
    }
}

fn main() {
    if let Err(msg) = run() {
        eprintln!("error: {msg}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use hipfire_atlas::AtlasRow;
    use serde_json::json;

    #[test]
    fn row_roundtrip() {
        let mut row = AtlasRow::new("ar", "qwen3.5-9b");
        row.set_metric_f64("prefill_tok_s", 1432.5)
            .set_metric_f64("prefill_kernel_ms", 88.4)
            .set_extra("git_sha", json!("46632a35"));
        let text = serde_json::to_string(&row).unwrap();
        let back: AtlasRow = serde_json::from_str(&text).unwrap();
        assert_eq!(back.workload_kind, "qwen3.5-9b");
        assert_eq!(back.metric_f64("prefill_tok_s"), Some(1432.5));
        assert_eq!(back.extra.get("git_sha"), Some(&json!("46632a35")));
    }
}
