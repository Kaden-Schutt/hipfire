#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

//! Print a one-shot memory snapshot the way a UI would render it.
//!
//! `cargo run -p hipfire-sysinfo --example mem`
//!
//! Demonstrates the intended consumer flow: collect with `snapshot`, then lean
//! on the wasm-safe derived helpers (`primary_pool`, `MemPool::percent`,
//! `fmt_bytes`) — the exact calls the webUI and TUI make to render.

use hipfire_admin_types::fmt_bytes;
use hipfire_sysinfo::snapshot;

fn bar(pct: f64) -> String {
    let filled = ((pct / 100.0) * 20.0).round() as usize;
    format!("[{}{}]", "#".repeat(filled), "-".repeat(20 - filled))
}

fn main() {
    let snap = snapshot(0);

    if snap.gpus.is_empty() {
        println!("no AMD GPUs visible under /sys/class/drm");
    }
    for g in &snap.gpus {
        let class = if g.is_integrated() { "APU/UMA" } else { "dGPU" };
        println!("{} ({class})", g.card);
        if let Some(p) = g.primary_pool() {
            println!(
                "  {:<16} {} {:>5.1}%  {} / {}",
                p.label,
                bar(p.percent()),
                p.percent(),
                fmt_bytes(p.used_bytes),
                fmt_bytes(p.total_bytes),
            );
        }
        // Always show the secondary pool too so dGPU GTT spillover is visible.
        let secondary = if g.is_integrated() {
            g.vram_pool()
        } else {
            g.gtt_pool()
        };
        if let Some(p) = secondary {
            println!(
                "  {:<16} {} {:>5.1}%  {} / {}",
                p.label,
                bar(p.percent()),
                p.percent(),
                fmt_bytes(p.used_bytes),
                fmt_bytes(p.total_bytes),
            );
        }
        if let (Some(t), Some(w)) = (g.temp_c, g.power_w) {
            println!("  {t:.0}°C  {w:.1} W");
        }
        if let Some(m) = &g.metrics {
            let mut extras = Vec::new();
            if let Some(p) = m.socket_power_w {
                extras.push(format!("socket {p:.1}W"));
            }
            if let Some(t) = m.soc_temp_c {
                extras.push(format!("soc {t:.0}°C"));
            }
            if let Some(t) = m.gfx_temp_c {
                extras.push(format!("gfx {t:.0}°C"));
            }
            extras.push(if m.throttling() {
                "THROTTLING".to_string()
            } else {
                "no throttle".to_string()
            });
            if let (Some(r), Some(w)) = (m.dram_read_mbps, m.dram_write_mbps) {
                extras.push(format!("dram r/w {r:.0}/{w:.0} MB/s"));
            }
            println!(
                "  gpu_metrics v{}.{}: {}",
                m.version.0,
                m.version.1,
                extras.join("  ")
            );
        }
    }

    if !snap.npus.is_empty() {
        println!("NPU");
        for n in &snap.npus {
            let pw = n
                .power_w
                .map(|p| format!("{p:.2}W"))
                .unwrap_or_else(|| "—".into());
            println!(
                "  {} util {:.0}%  pwr {pw}  TOPS {}/{}  tasks {}/{}  clk {}MHz",
                n.node,
                n.mean_util_pct,
                n.tops_current,
                n.tops_max,
                n.tasks_current,
                n.tasks_max,
                n.mp_npu_mhz,
            );
        }
    }

    if !snap.clients.is_empty() {
        println!("GPU clients (by VRAM):");
        for c in snap.clients.iter().take(6) {
            println!(
                "  {:>7} {:<16} {:<6} vram {:<10} gtt {}",
                c.pid,
                c.comm,
                c.card.as_deref().unwrap_or("?"),
                fmt_bytes(c.vram_bytes),
                fmt_bytes(c.gtt_bytes),
            );
        }
    }

    if let Some(h) = &snap.host {
        let p = h.as_pool();
        println!(
            "{:<18} {} {:>5.1}%  {} / {}",
            p.label,
            bar(p.percent()),
            p.percent(),
            fmt_bytes(p.used_bytes),
            fmt_bytes(p.total_bytes),
        );
    }
}
