//! hipfire admin console — Leptos CSR app.
//!
//! Phase 3 milestone: a live GPU panel backed by `GET /admin/stats`. Panels
//! for logs/alerts and model controls land in later phases. Built with
//! `trunk build` and embedded into `hipfire-server`.

use hipfire_admin_types::{AdminStats, GpuTelemetry};
use leptos::prelude::*;

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(App);
}

#[component]
fn App() -> impl IntoView {
    let (stats, set_stats) = signal(None::<Result<AdminStats, String>>);

    // Fetch once on mount. (Live SSE polling lands in a later phase.)
    leptos::task::spawn_local(async move {
        set_stats.set(Some(fetch_stats().await));
    });

    view! {
        <div class="wrap">
            <h1>"hipfire admin console"</h1>
            <p class="sub">"GPU telemetry — /admin/stats"</p>
            {move || match stats.get() {
                None => view! { <p class="sub">"loading…"</p> }.into_any(),
                Some(Err(e)) => view! { <p class="err">{e}</p> }.into_any(),
                Some(Ok(s)) => view! { <GpuGrid stats=s/> }.into_any(),
            }}
        </div>
    }
}

#[component]
fn GpuGrid(stats: AdminStats) -> impl IntoView {
    if stats.gpus.is_empty() {
        return view! { <p class="sub">"No AMD GPUs detected."</p> }.into_any();
    }
    let cards = stats
        .gpus
        .into_iter()
        .map(|gpu| view! { <GpuCard gpu=gpu/> })
        .collect::<Vec<_>>();
    view! { <div class="grid">{cards}</div> }.into_any()
}

#[component]
fn GpuCard(gpu: GpuTelemetry) -> impl IntoView {
    let busy = gpu.busy_percent.unwrap_or(0);
    let vram = match (gpu.vram_used_bytes, gpu.vram_total_bytes) {
        (Some(u), Some(t)) if t > 0 => format!(
            "{:.0} / {:.0} MB ({:.0}%)",
            u as f64 / 1.0e6,
            t as f64 / 1.0e6,
            (u as f64 / t as f64) * 100.0
        ),
        _ => "—".to_string(),
    };
    view! {
        <div class="card">
            <h2>{gpu.card.clone()}</h2>
            <div class="metric"><span class="k">"Utilization"</span><span class="v">{format!("{busy}%")}</span></div>
            <div class="bar"><span style=move || format!("width:{busy}%")></span></div>
            <div class="metric"><span class="k">"VRAM (carveout)"</span><span class="v">{vram}</span></div>
            <div class="metric"><span class="k">"Temp"</span><span class="v">{opt_f(gpu.temp_c, "°C")}</span></div>
            <div class="metric"><span class="k">"Power"</span><span class="v">{opt_f(gpu.power_w, " W")}</span></div>
            <div class="metric"><span class="k">"Clock"</span><span class="v">{gpu.sclk_mhz.map(|m| format!("{m} MHz")).unwrap_or_else(|| "—".into())}</span></div>
        </div>
    }
}

fn opt_f(value: Option<f64>, unit: &str) -> String {
    match value {
        Some(v) => format!("{v:.1}{unit}"),
        None => "—".to_string(),
    }
}

async fn fetch_stats() -> Result<AdminStats, String> {
    let resp = gloo_net::http::Request::get("/admin/stats")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.status() == 401 {
        return Err("authentication required — sign in at /admin".to_string());
    }
    if !resp.ok() {
        return Err(format!("/admin/stats returned HTTP {}", resp.status()));
    }
    resp.json::<AdminStats>().await.map_err(|e| e.to_string())
}
