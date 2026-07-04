//! hipfire admin console — Leptos CSR app.
//!
//! Live compute/memory panel backed by `GET /admin/stats`, polled on an
//! interval into a rolling in-memory history that drives per-metric usage
//! sparklines. The layout is device-generic on purpose: a host can carry many
//! heterogeneous accelerators (e.g. medusa's 12× MI50 + 1× W7800) plus the
//! CPU/host, and NPUs later. So the page renders *sections of device cards*
//! over the GPU `Vec` and the host, and every memory figure flows through the
//! wasm-safe helpers in `hipfire-admin-types` (`primary_pool`, `MemPool`,
//! `fmt_bytes`). Usage history is client-side only — no server storage — so it
//! shows the live trailing window with zero backend changes.

use std::collections::VecDeque;

use hipfire_admin_types::{
    fmt_bytes, AdminStats, ClientUsage, GpuTelemetry, HostMemory, MemPool, NpuTelemetry,
};
use leptos::prelude::*;

/// Poll cadence and how many samples the trailing window keeps (~3 min @ 2s).
const POLL_MS: u32 = 2000;
const HISTORY: usize = 90;

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(App);
}

/// One polled sample, reduced to the scalars the sparklines plot. Keyed series
/// are rebuilt per render by matching `card` names against the live GPU list,
/// so cards appearing/disappearing between polls degrade gracefully.
#[derive(Clone)]
struct Sample {
    gpus: Vec<CardSample>,
    host_pct: Option<f64>,
}

#[derive(Clone)]
struct CardSample {
    card: String,
    util: f64,
    mem_pct: f64,
}

fn sample_from(stats: &AdminStats) -> Sample {
    Sample {
        gpus: stats
            .gpus
            .iter()
            .map(|g| CardSample {
                card: g.card.clone(),
                util: g.busy_percent.unwrap_or(0) as f64,
                mem_pct: g.primary_pool().map(|p| p.percent()).unwrap_or(0.0),
            })
            .collect(),
        host_pct: stats.host.as_ref().map(|h| h.percent()),
    }
}

#[component]
fn App() -> impl IntoView {
    // Latest snapshot (or error), plus the rolling history that backs charts.
    let (stats, set_stats) = signal(None::<Result<AdminStats, String>>);
    let (history, set_history) = signal(VecDeque::<Sample>::new());

    // Poll forever. Errors surface but never clear the chart history, so a
    // transient blip shows "reconnecting…" over the last-known trend instead
    // of wiping the page.
    leptos::task::spawn_local(async move {
        loop {
            let result = fetch_stats().await;
            if let Ok(s) = &result {
                set_history.update(|h| {
                    h.push_back(sample_from(s));
                    while h.len() > HISTORY {
                        h.pop_front();
                    }
                });
            }
            set_stats.set(Some(result));
            gloo_timers::future::TimeoutFuture::new(POLL_MS).await;
        }
    });

    view! {
        <div class="wrap">
            <h1>"hipfire admin console"</h1>
            <p class="sub">"compute & memory — /admin/stats (live)"</p>
            {move || {
                let history = history.get();
                match stats.get() {
                    None => view! { <p class="sub">"loading…"</p> }.into_any(),
                    // Fresh success: render the rich snapshot with real bytes.
                    Some(Ok(s)) => {
                        view! { <Dashboard stats=Some(s) history=history stale=false/> }.into_any()
                    }
                    // Errored with no data yet: show the error outright.
                    Some(Err(e)) if history.is_empty() => {
                        view! { <p class="err">{e}</p> }.into_any()
                    }
                    // Transient error with prior data: keep drawing, flag stale.
                    Some(Err(_)) => {
                        view! { <Dashboard stats=None history=history stale=true/> }.into_any()
                    }
                }
            }}
        </div>
    }
}

/// Top-level layout: one section per device class. GPUs and the host today;
/// NPU/other-accelerator sections drop in here as their telemetry lands.
#[component]
fn Dashboard(
    /// Latest rich snapshot, when the most recent poll succeeded.
    stats: Option<AdminStats>,
    /// Rolling sample window for the usage sparklines.
    history: VecDeque<Sample>,
    /// True when the newest poll failed but we are still showing prior data.
    stale: bool,
) -> impl IntoView {
    // On the stale path `stats` is None; fall back to the newest sample's shape
    // by synthesizing minimal cards from history so trends keep rendering.
    let stats = stats.unwrap_or_else(|| synth_stats(&history));
    let AdminStats {
        gpus,
        host,
        clients,
        npus,
        ..
    } = stats;

    let gpu_section = if gpus.is_empty() {
        view! { <p class="sub">"No AMD GPUs detected."</p> }.into_any()
    } else {
        let count = gpus.len();
        let cards = gpus
            .into_iter()
            .map(|gpu| {
                let util = series(&history, |s| {
                    s.gpus.iter().find(|c| c.card == gpu.card).map(|c| c.util)
                });
                let mem = series(&history, |s| {
                    s.gpus
                        .iter()
                        .find(|c| c.card == gpu.card)
                        .map(|c| c.mem_pct)
                });
                view! { <DeviceCard gpu=gpu util_series=util mem_series=mem/> }
            })
            .collect::<Vec<_>>();
        view! {
            <div class="section">{format!("GPUs · {count}")}</div>
            <div class="grid">{cards}</div>
        }
        .into_any()
    };

    let host_section = host
        .map(|h| {
            let ram = series(&history, |s| s.host_pct);
            view! {
                <div class="section">"Host"</div>
                <div class="grid"><HostCard host=h ram_series=ram/></div>
            }
            .into_any()
        })
        .unwrap_or_else(|| ().into_any());

    let npu_section = if npus.is_empty() {
        ().into_any()
    } else {
        let cards = npus
            .into_iter()
            .map(|n| view! { <NpuCard npu=n/> })
            .collect::<Vec<_>>();
        view! {
            <div class="section">"NPU"</div>
            <div class="grid">{cards}</div>
        }
        .into_any()
    };

    let clients_section = if clients.is_empty() {
        ().into_any()
    } else {
        view! {
            <div class="section">{format!("GPU processes · {}", clients.len())}</div>
            <ClientsTable clients=clients/>
        }
        .into_any()
    };

    let banner = stale
        .then(|| view! { <span class="stale">"reconnecting…"</span> }.into_any())
        .unwrap_or_else(|| ().into_any());

    view! { <div>{banner}{gpu_section}{npu_section}{host_section}{clients_section}</div> }
}

/// One NPU (AMD XDNA). Leads with mean column utilization, then power/TOPS/
/// clock chips. Fields that the kernel left unpopulated are simply omitted.
#[component]
fn NpuCard(npu: NpuTelemetry) -> impl IntoView {
    let util = npu.mean_util_pct;
    let mut chips = Vec::new();
    if let Some(p) = npu.power_w {
        chips.push(format!("{p:.2} W"));
    }
    if npu.tops_max > 0 {
        chips.push(format!("{}/{} TOPS", npu.tops_current, npu.tops_max));
    }
    if npu.tasks_max > 0 {
        chips.push(format!("{}/{} tasks", npu.tasks_current, npu.tasks_max));
    }
    if npu.mp_npu_mhz > 0 {
        chips.push(format!("{} MHz", npu.mp_npu_mhz));
    }
    let chip_views = chips
        .into_iter()
        .map(|c| view! { <span class="chip">{c}</span> })
        .collect::<Vec<_>>();
    view! {
        <div class="card">
            <div class="head">
                <h2>{npu.node}</h2>
                <span class="badge apu">"XDNA"</span>
            </div>
            <div class="memrow">
                <div class="memhead">
                    <span class="k">"Utilization"</span>
                    <span class="v">{format!("{util:.0}%")}</span>
                </div>
                <div class="bar" data-level=fill_level(util)>
                    <span style=format!("width:{:.1}%", util.min(100.0))></span>
                </div>
            </div>
            <div class="chips">{chip_views}</div>
        </div>
    }
}

/// Per-process GPU memory, as a compact table. Most-VRAM first (collector sorts).
#[component]
fn ClientsTable(clients: Vec<ClientUsage>) -> impl IntoView {
    let rows = clients
        .into_iter()
        .map(|c| {
            let card = c.card.unwrap_or_else(|| "?".to_string());
            view! {
                <tr>
                    <td class="mono">{c.pid}</td>
                    <td>{c.comm}</td>
                    <td class="mono">{card}</td>
                    <td class="mono num">{fmt_bytes(c.vram_bytes)}</td>
                    <td class="mono num">{fmt_bytes(c.gtt_bytes)}</td>
                </tr>
            }
        })
        .collect::<Vec<_>>();
    view! {
        <table class="clients">
            <thead>
                <tr><th>"PID"</th><th>"Process"</th><th>"Card"</th><th class="num">"VRAM"</th><th class="num">"GTT"</th></tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    }
}

/// Project one scalar series out of the history window, dropping samples that
/// lack the metric (e.g. a card that wasn't present yet).
fn series(history: &VecDeque<Sample>, pick: impl Fn(&Sample) -> Option<f64>) -> Vec<f64> {
    history.iter().filter_map(pick).collect()
}

/// Build a minimal `AdminStats` from the newest sample for the stale-render
/// path. Only the fields the cards display are populated; primary-pool numbers
/// come back as percentages already, so we encode them as a synthetic pool.
fn synth_stats(history: &VecDeque<Sample>) -> AdminStats {
    let Some(last) = history.back() else {
        return AdminStats::default();
    };
    AdminStats {
        generated_unix: 0,
        gpus: last
            .gpus
            .iter()
            .map(|c| GpuTelemetry {
                card: c.card.clone(),
                busy_percent: Some(c.util as u32),
                // Encode the known percentage as a 0..100 pool so the bar/label
                // stay meaningful without the absolute byte counts.
                vram_used_bytes: Some(c.mem_pct.round() as u64),
                vram_total_bytes: Some(100),
                ..Default::default()
            })
            .collect(),
        host: last.host_pct.map(|p| HostMemory {
            total_bytes: 100,
            available_bytes: (100.0 - p).round().max(0.0) as u64,
        }),
        // Stale-render path trends only; instantaneous lists aren't replayed.
        clients: Vec::new(),
        npus: Vec::new(),
    }
}

/// One GPU. Leads with the OOM-governing pool (GTT on APUs, VRAM on dGPUs),
/// shows the other pool muted, then live util/mem usage sparklines.
#[component]
fn DeviceCard(gpu: GpuTelemetry, util_series: Vec<f64>, mem_series: Vec<f64>) -> impl IntoView {
    let integrated = gpu.is_integrated();
    let (badge_class, badge_text) = if integrated {
        ("badge apu", "APU · UMA")
    } else {
        ("badge", "dGPU")
    };
    let busy = gpu.busy_percent.unwrap_or(0);

    let primary = gpu.primary_pool();
    let secondary = if integrated {
        gpu.vram_pool()
    } else {
        gpu.gtt_pool().map(|mut p| {
            if p.used_bytes > 0 {
                p.label = "GTT (spill)".to_string();
            }
            p
        })
    };

    let primary_view = match primary {
        Some(p) => view! { <MemBar pool=p emphasis="primary"/> }.into_any(),
        None => view! { <p class="sub">"memory unavailable"</p> }.into_any(),
    };
    let secondary_view = secondary
        .map(|p| view! { <MemBar pool=p emphasis="muted"/> }.into_any())
        .unwrap_or_else(|| ().into_any());

    view! {
        <div class="card">
            <div class="head">
                <h2>{gpu.card.clone()}</h2>
                <span class=badge_class>{badge_text}</span>
            </div>
            <div class="memrow">
                <div class="memhead">
                    <span class="k">"Utilization"</span>
                    <span class="v">{format!("{busy}%")}</span>
                </div>
                <div class="bar" data-level=fill_level(busy as f64)>
                    <span style=format!("width:{busy}%")></span>
                </div>
            </div>
            {primary_view}
            {secondary_view}
            <div class="trend">
                <span class="tlabel">"util"</span>
                <Sparkline series=util_series max=100.0/>
            </div>
            <div class="trend">
                <span class="tlabel">"mem"</span>
                <Sparkline series=mem_series max=100.0/>
            </div>
            <div class="chips">
                {chip(gpu.temp_c, "", "°C")}
                {chip(gpu.power_w, "", " W")}
                {gpu.sclk_mhz
                    .map(|m| view! { <span class="chip">{format!("{m} MHz")}</span> }.into_any())
                    .unwrap_or_else(|| ().into_any())}
                {metrics_chips(&gpu.metrics)}
            </div>
        </div>
    }
}

/// Firmware `gpu_metrics` extras as chips: whole-package socket power, SoC
/// temperature, DRAM bandwidth (v3), and a red throttle badge when throttling.
fn metrics_chips(metrics: &Option<hipfire_admin_types::GpuMetrics>) -> AnyView {
    let Some(m) = metrics else {
        return ().into_any();
    };
    let mut chips: Vec<AnyView> = Vec::new();
    if let Some(p) = m.socket_power_w {
        chips.push(view! { <span class="chip">{format!("socket {p:.1} W")}</span> }.into_any());
    }
    if let Some(t) = m.soc_temp_c {
        chips.push(view! { <span class="chip">{format!("soc {t:.0}°C")}</span> }.into_any());
    }
    if let (Some(r), Some(w)) = (m.dram_read_mbps, m.dram_write_mbps) {
        chips.push(
            view! { <span class="chip">{format!("dram {r:.0}/{w:.0} MB/s")}</span> }.into_any(),
        );
    }
    if m.throttling() {
        chips.push(view! { <span class="badge crit">"THROTTLING"</span> }.into_any());
    }
    chips.into_any()
}

/// CPU/host system memory + its usage trend. On UMA APUs this is the real
/// ceiling — GPU GTT allocations are ordinary system pages counted here.
#[component]
fn HostCard(host: HostMemory, ram_series: Vec<f64>) -> impl IntoView {
    let avail = fmt_bytes(host.available_bytes);
    view! {
        <div class="card">
            <div class="head">
                <h2>"System RAM"</h2>
                <span class="badge">"CPU · host"</span>
            </div>
            <MemBar pool=host.as_pool() emphasis="primary"/>
            <div class="trend">
                <span class="tlabel">"ram"</span>
                <Sparkline series=ram_series max=100.0/>
            </div>
            <div class="chips">
                <span class="chip">{format!("avail {avail}")}</span>
            </div>
        </div>
    }
}

/// One memory pool rendered as a labeled, color-graded fill bar. The single
/// place a pool becomes pixels — reused for GPU VRAM, GPU GTT, and host RAM.
#[component]
fn MemBar(pool: MemPool, emphasis: &'static str) -> impl IntoView {
    let pct = pool.percent();
    let level = fill_level(pct);
    let figures = format!(
        "{} / {}",
        fmt_bytes(pool.used_bytes),
        fmt_bytes(pool.total_bytes)
    );
    view! {
        <div class=format!("memrow {emphasis}")>
            <div class="memhead">
                <span class="k">{pool.label}</span>
                <span class="v">{figures}<span class="pct">{format!("{pct:.0}%")}</span></span>
            </div>
            <div class="bar" data-level=level>
                <span style=format!("width:{:.1}%", pct.min(100.0))></span>
            </div>
        </div>
    }
}

/// A trailing-window sparkline: filled area under a line, plotted in a unit
/// `0..100 × 0..28` viewBox stretched to fit. `max` sets the full-scale value
/// (100 for percentages). Colored by the latest value's threshold.
#[component]
fn Sparkline(series: Vec<f64>, max: f64) -> impl IntoView {
    let max = if max <= 0.0 { 1.0 } else { max };
    let n = series.len();
    let level = series.last().map(|v| fill_level(*v)).unwrap_or("ok");
    let class = format!("spark {level}");
    if n < 2 {
        // Not enough points to draw a trend yet.
        return view! {
            <svg class=class viewBox="0 0 100 28" preserveAspectRatio="none"></svg>
        }
        .into_any();
    }
    let coords: Vec<(f64, f64)> = series
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = i as f64 / (n - 1) as f64 * 100.0;
            let y = 28.0 - (v.clamp(0.0, max) / max) * 28.0;
            (x, y)
        })
        .collect();
    let mut line = String::new();
    for (i, (x, y)) in coords.iter().enumerate() {
        line.push_str(&format!("{}{x:.2},{y:.2} ", if i == 0 { "M" } else { "L" }));
    }
    let mut area = String::from("M0,28 ");
    for (x, y) in &coords {
        area.push_str(&format!("L{x:.2},{y:.2} "));
    }
    area.push_str("L100,28 Z");
    view! {
        <svg class=class viewBox="0 0 100 28" preserveAspectRatio="none">
            <path class="area" d=area/>
            <path class="line" d=line/>
        </svg>
    }
    .into_any()
}

/// Bar/spark color thresholds, shared so warn/crit means the same everywhere.
fn fill_level(pct: f64) -> &'static str {
    if pct >= 90.0 {
        "crit"
    } else if pct >= 75.0 {
        "warn"
    } else {
        "ok"
    }
}

/// A `°C`/`W`-style chip from an optional metric, omitted entirely when absent.
fn chip(value: Option<f64>, prefix: &str, unit: &str) -> AnyView {
    match value {
        Some(v) => view! { <span class="chip">{format!("{prefix}{v:.1}{unit}")}</span> }.into_any(),
        None => ().into_any(),
    }
}

async fn fetch_stats() -> Result<AdminStats, String> {
    hipfire_web_ui::get_json::<AdminStats>("/admin/stats").await
}
