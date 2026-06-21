// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Paragraph, Row, Table, Tabs, Wrap},
    Frame,
};

use crate::{
    app::{App, ControlAction, Tab},
    hipfire::registry::ModelListItem,
};

const BG: Color = Color::Rgb(7, 7, 9);
const PANEL: Color = Color::Rgb(18, 16, 18);
const PANEL_2: Color = Color::Rgb(40, 24, 27);
const TEXT: Color = Color::Rgb(222, 226, 232);
const MUTED: Color = Color::Rgb(142, 150, 163);
const ACCENT: Color = Color::Rgb(237, 45, 57);
const GREEN: Color = Color::Rgb(102, 217, 139);
const YELLOW: Color = Color::Rgb(238, 190, 95);
const RED: Color = Color::Rgb(255, 95, 104);
/// Darker green for the "online" host indicator in the title bar.
const DARK_GREEN: Color = Color::Rgb(56, 142, 84);
/// Spinner glyph color — distinct from BLUE so it doesn't read as faked.
const SPINNER: Color = Color::Rgb(120, 200, 220);
/// Bright blue marks statically faked data that has no backend source yet.
const BLUE: Color = Color::Rgb(80, 170, 255);

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    frame.render_widget(Clear, area);
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(10),
            Constraint::Length(2),
        ])
        .split(area);

    draw_header(frame, app, root[0]);
    match app.tab {
        Tab::Home => draw_home(frame, app, root[1]),

        Tab::Chat => draw_chat(frame, app, root[1]),
        Tab::Models => draw_models(frame, app, root[1]),
        Tab::Runtime => draw_runtime(frame, app, root[1]),
        Tab::Logs => draw_logs(frame, app, root[1]),
        Tab::Training => draw_training(frame, app, root[1]),
        Tab::Settings => draw_settings(frame, app, root[1]),
        Tab::System => draw_system(frame, app, root[1]),
    }
    draw_footer(frame, app, root[2]);
}

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Length(3)])
        .split(area);

    // Title row: [spinner | hipfire | host status] .... [GPU load bar].
    let bar = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(20), Constraint::Length(26)])
        .split(chunks[0]);

    let online = app.status.serve_http_ok;
    let host_status = if online {
        Span::styled(
            "online",
            Style::default().fg(DARK_GREEN).add_modifier(Modifier::BOLD),
        )
    } else {
        // Matches the offline color of the Status pane "Serve" line.
        Span::styled(
            "offline",
            Style::default().fg(RED).add_modifier(Modifier::BOLD),
        )
    };
    let title = Line::from(vec![
        Span::styled(
            format!("{} ", app.spinner_frame()),
            Style::default().fg(SPINNER),
        ),
        Span::styled(
            "hipfire",
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ),
        Span::raw("    "),
        Span::styled(
            format!("host: {} ", app.status.hostname),
            Style::default().fg(MUTED),
        ),
        host_status,
    ]);
    frame.render_widget(
        Paragraph::new(title)
            .style(Style::default().bg(BG))
            .alignment(Alignment::Left),
        bar[0],
    );

    draw_gpu_bar(frame, bar[1]);

    let titles = Tab::ALL
        .iter()
        .map(|tab| Line::from(Span::raw(tab.title())))
        .collect::<Vec<_>>();
    let selected = Tab::ALL.iter().position(|t| *t == app.tab).unwrap_or(0);
    let tabs = Tabs::new(titles)
        .select(selected)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(PANEL_2)),
        )
        .style(Style::default().fg(MUTED).bg(BG))
        .highlight_style(Style::default().fg(YELLOW).add_modifier(Modifier::BOLD))
        .divider(Span::styled(" | ", Style::default().fg(PANEL_2)));
    frame.render_widget(tabs, chunks[1]);
}

/// Title-bar GPU load gauge. FAKED (bright blue) — no GPU-utilization source is
/// wired yet; this is a placeholder bar at a fixed percentage.
fn draw_gpu_bar(frame: &mut Frame, area: Rect) {
    const FAKE_GPU_PCT: u16 = 62;
    // Reserve "GPU " prefix (4) + " 100%" suffix (5) from the cell width.
    let track = (area.width as usize).saturating_sub(9).max(1);
    let filled = track * FAKE_GPU_PCT as usize / 100;
    let bar: String = "█".repeat(filled) + &"░".repeat(track.saturating_sub(filled));
    let line = Line::from(vec![
        Span::styled("GPU ", Style::default().fg(MUTED)),
        Span::styled(bar, Style::default().fg(BLUE)),
        Span::styled(format!(" {FAKE_GPU_PCT}%"), Style::default().fg(BLUE)),
    ]);
    frame.render_widget(
        Paragraph::new(line)
            .style(Style::default().bg(BG))
            .alignment(Alignment::Left),
        area,
    );
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    let help = match app.tab {
        Tab::Chat => {
            "Tab switch  Enter send/start serve  Ctrl+O newline  Up/Down scroll  Esc blur/quit"
        }
        Tab::Models => {
            "Tab switch  Up/Down select  Enter expand/select  Left/Right fold  r refresh  q quit"
        }
        Tab::Training => "Tab switch  Up/Down select run  r refresh  q quit",
        Tab::Runtime => "Tab switch  r refresh health/kernels/locks  q quit",
        Tab::Logs => "Tab switch  r refresh log tails  q quit",
        Tab::Settings => "Tab switch  e easy  a advanced  Up/Down select  r refresh  q quit",
        Tab::Home => {
            "Tab switch  s/x/t serve start/stop/restart  c chat  d admin  click controls  r refresh  q quit"
        }
        _ => "Tab switch  r refresh  q quit",
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(help, Style::default().fg(MUTED)),
            Span::styled(
                format!("    {}", app.last_reload),
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .style(Style::default().bg(BG)),
        area,
    );
}

fn draw_home(frame: &mut Frame, app: &mut App, area: Rect) {
    app.control_buttons.clear();

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(pad(area, 1, 0));
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(11), Constraint::Min(6)])
        .split(cols[0]);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(6)])
        .split(cols[1]);

    draw_home_status(frame, app, left[0]);
    draw_home_models(frame, app, left[1]);
    draw_home_control(frame, app, right[0]);
    draw_home_connections(frame, right[1]);
}

/// Status pane: serve health, full endpoint URLs, and live serve/daemon PIDs.
fn draw_home_status(frame: &mut Frame, app: &App, area: Rect) {
    let serve_color = if app.status.serve_http_ok {
        GREEN
    } else if app.status.serve_pid_alive || app.status.serve_pid.is_some() {
        YELLOW
    } else {
        RED
    };

    let pid_list = |pids: &[u32]| {
        if pids.is_empty() {
            "-".to_string()
        } else {
            pids.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        }
    };

    let mut status = vec![Line::from(vec![
        Span::raw("Serve       "),
        Span::styled(app.status.serve_label(), Style::default().fg(serve_color)),
    ])];
    for (idx, url) in app.status.endpoints.iter().enumerate() {
        let label = if idx == 0 {
            "Endpoint    "
        } else {
            "            "
        };
        status.push(Line::from(vec![
            Span::raw(label),
            Span::styled(url.clone(), Style::default().fg(TEXT)),
        ]));
    }
    status.push(Line::from(format!(
        "serve PID   {}",
        pid_list(&app.status.serve_pids)
    )));
    status.push(Line::from(format!(
        "daemon PID  {}",
        pid_list(&app.status.daemon_pids)
    )));
    if let Some(warning) = &app.config.warning {
        status.push(Line::from(vec![
            Span::styled("Config      ", Style::default().fg(YELLOW)),
            Span::styled(warning.clone(), Style::default().fg(YELLOW)),
        ]));
    }
    if let Some(warning) = &app.registry.warning {
        status.push(Line::from(vec![
            Span::styled("Registry    ", Style::default().fg(YELLOW)),
            Span::styled(warning.clone(), Style::default().fg(YELLOW)),
        ]));
    }
    frame.render_widget(card("Status", status), area);
}

/// Models pane: each loaded model with its resident size and override flag.
/// The active model name is real; resident size and the override badge are
/// FAKED (bright blue) until the daemon exposes a loaded-model inventory.
fn draw_home_models(frame: &mut Frame, app: &App, area: Rect) {
    // (name, resident, total, override, name_is_real) — the first row is the
    // real active model; the rest are illustrative placeholders (blue) for the
    // multi-model layout.
    let models: Vec<(String, &str, &str, bool, bool)> = vec![
        (
            app.active_model.clone(),
            "7.8",
            "8.0",
            app.config.per_model_count > 0,
            true,
        ),
        ("gemma3-vl-4b".into(), "3.1", "4.2", false, false),
    ];

    let mut lines = Vec::new();
    for (name, resident, total, has_override, name_is_real) in &models {
        let name_color = if *name_is_real { TEXT } else { BLUE };
        let mut spans = vec![
            Span::styled(
                name.clone(),
                Style::default().fg(name_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  ({resident}/{total} GB in RAM)"),
                Style::default().fg(BLUE),
            ),
        ];
        if *has_override {
            spans.push(Span::styled(
                "  [config override]",
                Style::default().fg(BLUE),
            ));
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "resident size + override flags are placeholder data (blue)",
        Style::default().fg(MUTED),
    )));

    frame.render_widget(card("Models", lines), area);
}

/// Control pane: start/stop/restart serve plus chat/admin endpoint toggles.
/// Each row is a clickable button (mouse) and has a keyboard hotkey. Serve
/// actions are real; the endpoint toggles are FAKED state (blue).
fn draw_home_control(frame: &mut Frame, app: &mut App, area: Rect) {
    frame.render_widget(block("Control"), area);
    let inner = Rect {
        x: area.x.saturating_add(2),
        y: area.y.saturating_add(1),
        width: area.width.saturating_sub(4),
        height: area.height.saturating_sub(2),
    };

    let on_off = |on: bool| if on { "on" } else { "off" };
    let buttons: [(ControlAction, Span, Span); 5] = [
        (
            ControlAction::StartServe,
            Span::styled("[s] Start serve", Style::default().fg(TEXT)),
            Span::styled("", Style::default()),
        ),
        (
            ControlAction::StopServe,
            Span::styled("[x] Stop serve", Style::default().fg(TEXT)),
            Span::styled("", Style::default()),
        ),
        (
            ControlAction::RestartServe,
            Span::styled("[t] Restart serve", Style::default().fg(TEXT)),
            Span::styled("", Style::default()),
        ),
        (
            ControlAction::ToggleChatEndpoint,
            Span::styled("[c] Chat endpoint", Style::default().fg(TEXT)),
            Span::styled(
                on_off(app.chat_endpoint_on).to_string(),
                Style::default().fg(BLUE),
            ),
        ),
        (
            ControlAction::ToggleAdminConsole,
            Span::styled("[d] Admin console", Style::default().fg(TEXT)),
            Span::styled(
                on_off(app.admin_console_on).to_string(),
                Style::default().fg(BLUE),
            ),
        ),
    ];

    for (idx, (action, label, state)) in buttons.into_iter().enumerate() {
        let y = inner.y.saturating_add(idx as u16);
        if y >= inner.y.saturating_add(inner.height) {
            break;
        }
        let row = Rect {
            x: inner.x,
            y,
            width: inner.width,
            height: 1,
        };
        let mut spans = vec![label];
        if !state.content.is_empty() {
            spans.push(Span::raw("  "));
            spans.push(state);
        }
        frame.render_widget(
            Paragraph::new(Line::from(spans)).style(Style::default().bg(PANEL)),
            row,
        );
        app.control_buttons.push((row, action));
    }
}

/// Connections pane: who is connected to the server. FAKED (bright blue) — the
/// auth/account layer is not finished, so client IPs and accounts are stubbed.
fn draw_home_connections(frame: &mut Frame, area: Rect) {
    let rows = [
        ("192.168.0.10", "alice", "2"),
        ("192.168.0.42", "bob", "1"),
        ("127.0.0.1", "local-cli", "1"),
    ];
    let total: u32 = rows
        .iter()
        .filter_map(|(_, _, n)| n.parse::<u32>().ok())
        .sum();

    let table_rows = rows.iter().map(|(ip, account, conns)| {
        Row::new([ip.to_string(), account.to_string(), conns.to_string()])
            .style(Style::default().fg(BLUE))
    });
    frame.render_widget(
        Table::new(
            table_rows,
            [
                Constraint::Length(18),
                Constraint::Min(12),
                Constraint::Length(6),
            ],
        )
        .header(Row::new(["Client IP", "Account", "Conns"]).style(Style::default().fg(MUTED)))
        .block(block(&format!(
            "Connections  ({total} active, placeholder)"
        )))
        .style(Style::default().bg(PANEL)),
        area,
    );
}

fn draw_chat(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(5)])
        .split(pad(area, 1, 0));

    let mut lines = Vec::new();
    if app.chat.messages.is_empty() {
        lines.push(Line::from(Span::styled(
            "No messages yet. Type below and press Enter.",
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(
            "Prototype 1 uses the existing hipfire serve OpenAI endpoint.",
        ));
    } else {
        for msg in &app.chat.messages {
            let color = if msg.role == "user" { ACCENT } else { GREEN };
            lines.push(Line::from(Span::styled(
                format!("{}:", msg.role),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )));
            for line in msg.content.lines() {
                lines.push(Line::from(line.to_string()));
            }
            lines.push(Line::from(""));
        }
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(block("Chat shell"))
            .scroll((app.chat.scroll, 0))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        chunks[0],
    );

    let input_title = if app.chat.sending {
        format!("Input - {}", app.chat.status)
    } else {
        format!("Input - {} - model {}", app.chat.status, app.active_model)
    };
    let input = Paragraph::new(app.chat.input.as_str())
        .block(block(&input_title))
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(TEXT).bg(PANEL_2));
    frame.render_widget(input, chunks[1]);
}

fn draw_models(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(8)])
        .split(pad(area, 1, 0));
    let summary = format!(
        "active: {}    {} downloaded / {} available    registry: {}    aliases: {}",
        app.active_model,
        app.registry.downloaded_count(),
        app.registry.models.len(),
        app.registry
            .loaded_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "missing".into()),
        app.registry.aliases.len(),
    );
    frame.render_widget(
        Paragraph::new(summary)
            .block(block("Model hub"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        chunks[0],
    );

    let visible_items = app.registry.visible_items();
    let rows = visible_items
        .iter()
        .enumerate()
        .skip(scroll_start(app.registry.selected, chunks[1].height, 3))
        .take(visible_rows(chunks[1].height, 3))
        .map(|(idx, item)| {
            let selected = idx == app.registry.selected;
            let row = match item {
                ModelListItem::Group {
                    name,
                    count,
                    downloaded,
                    expanded,
                } => {
                    let marker = if *expanded { "v" } else { ">" };
                    Row::new([
                        format!("{marker} {name}"),
                        format!("{downloaded}/{count} local"),
                        String::new(),
                        String::new(),
                        String::new(),
                        "Enter/Right to expand, Left to collapse".into(),
                    ])
                }
                ModelListItem::Model { model_index } => {
                    let row = &app.registry.models[*model_index];
                    let status = if row.tag == app.active_model {
                        "active"
                    } else if row.downloaded {
                        "local"
                    } else if row.entry.repo.is_empty() {
                        "local-only"
                    } else {
                        "remote"
                    };
                    let mut extras = Vec::new();
                    if row.has_triattn {
                        extras.push("triattn");
                    }
                    if row.has_mtp {
                        extras.push("mtp");
                    }
                    if row.has_draft {
                        extras.push("draft");
                    }
                    if row.has_template {
                        extras.push("template");
                    }
                    Row::new([
                        format!("  {}", row.tag),
                        status.into(),
                        format!("{:.1} GB", row.entry.size_gb),
                        format!("{:.0} GB", row.entry.min_vram_gb),
                        extras.join(" "),
                        if row.entry.repo.is_empty() {
                            format!("{} (no remote repo)", row.entry.desc)
                        } else {
                            row.entry.desc.clone()
                        },
                    ])
                }
            };
            row.style(if selected {
                Style::default().fg(ACCENT).bg(PANEL_2)
            } else {
                match item {
                    ModelListItem::Group { .. } => Style::default()
                        .fg(YELLOW)
                        .bg(PANEL)
                        .add_modifier(Modifier::BOLD),
                    ModelListItem::Model { model_index } => {
                        if app.registry.models[*model_index].tag == app.active_model {
                            Style::default().fg(GREEN).bg(PANEL)
                        } else {
                            Style::default().fg(TEXT).bg(PANEL)
                        }
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    let table = Table::new(
        rows,
        [
            Constraint::Length(24),
            Constraint::Length(8),
            Constraint::Length(9),
            Constraint::Length(8),
            Constraint::Length(12),
            Constraint::Min(20),
        ],
    )
    .header(
        Row::new(["Tag", "Have", "Size", "VRAM", "Sidecars", "Notes"])
            .style(Style::default().fg(MUTED)),
    )
    .block(block("Registry browser"))
    .style(Style::default().bg(PANEL));
    frame.render_widget(table, chunks[1]);
}

fn draw_runtime(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(44), Constraint::Percentage(56)])
        .split(pad(area, 1, 0));
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Min(6),
        ])
        .split(cols[0]);

    let serve_color = if app.status.serve_http_ok {
        GREEN
    } else if app.status.serve_pid_alive || app.status.serve_pid.is_some() {
        YELLOW
    } else {
        RED
    };
    let runtime = vec![
        Line::from(vec![
            Span::raw("Serve      "),
            Span::styled(app.status.serve_label(), Style::default().fg(serve_color)),
        ]),
        Line::from(format!(
            "Endpoint   {}:{}",
            app.config.probe_host(),
            app.config.port
        )),
        Line::from(format!(
            "PID        {}",
            app.status
                .serve_pid
                .map(|p| p.to_string())
                .unwrap_or_else(|| "-".into())
        )),
        Line::from(format!("Active     {}", app.active_model)),
        Line::from(format!("Default    {}", app.config.default_model)),
        Line::from(format!(
            "Profiles   {} overlays",
            app.config.per_model_count
        )),
    ];
    frame.render_widget(card("Daemon", runtime), left[0]);

    let kernel_lines = app
        .status
        .kernel_lines
        .iter()
        .map(|line| Line::from(line.clone()))
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(kernel_lines)
            .block(block("Kernel cache"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        left[1],
    );

    let lock_lines = app
        .status
        .lock_lines
        .iter()
        .map(|line| Line::from(line.clone()))
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(lock_lines)
            .block(block("Resource locks"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        left[2],
    );

    let mut health_lines = vec![Line::from(Span::styled(
        "Health response",
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    ))];
    health_lines.extend(
        app.status
            .health_text
            .lines()
            .map(|line| Line::from(line.to_string())),
    );
    if app.status.health_text.is_empty() {
        health_lines.push(Line::from("No health body returned."));
    }
    frame.render_widget(
        Paragraph::new(Text::from(health_lines))
            .block(block("Raw /health"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        cols[1],
    );
}

fn draw_logs(frame: &mut Frame, app: &App, area: Rect) {
    let lines = app
        .status
        .log_lines
        .iter()
        .map(|line| {
            if line.starts_with("== ") && line.ends_with(" ==") {
                Line::from(Span::styled(
                    line.clone(),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ))
            } else {
                Line::from(line.clone())
            }
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block("Log tails"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        pad(area, 1, 0),
    );
}

fn draw_settings(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(8)])
        .split(pad(area, 1, 0));
    let mode = if app.settings_easy {
        "Easy settings"
    } else {
        "Advanced settings"
    };
    let note = if app.settings_easy {
        "Read-only prototype. Press a for advanced."
    } else {
        "Raw config view. Press e for easy."
    };
    frame.render_widget(
        Paragraph::new(format!("{mode}    {note}"))
            .block(block("Settings"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        chunks[0],
    );

    if app.settings_easy {
        let rows_all = app
            .config
            .easy_rows()
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>();
        let start = scroll_start(app.settings_selected, chunks[1].height, 3);
        let rows = rows_all
            .into_iter()
            .skip(start)
            .take(visible_rows(chunks[1].height, 3))
            .map(|(idx, (label, value, desc))| {
                Row::new([label.to_string(), value, desc.to_string()]).style(
                    if idx == app.settings_selected {
                        Style::default().fg(ACCENT).bg(PANEL_2)
                    } else {
                        Style::default().fg(TEXT).bg(PANEL)
                    },
                )
            })
            .collect::<Vec<_>>();
        frame.render_widget(
            Table::new(
                rows,
                [
                    Constraint::Length(16),
                    Constraint::Length(24),
                    Constraint::Min(30),
                ],
            )
            .header(Row::new(["Setting", "Value", "Meaning"]).style(Style::default().fg(MUTED)))
            .block(block("User-safe controls"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
            chunks[1],
        );
    } else {
        let rows_all = app.config.values.iter().enumerate().collect::<Vec<_>>();
        let start = scroll_start(app.settings_selected, chunks[1].height, 3);
        let rows = rows_all
            .into_iter()
            .skip(start)
            .take(visible_rows(chunks[1].height, 3))
            .map(|(idx, (k, v))| {
                Row::new([k.clone(), v.clone()]).style(if idx == app.settings_selected {
                    Style::default().fg(ACCENT).bg(PANEL_2)
                } else {
                    Style::default().fg(TEXT).bg(PANEL)
                })
            })
            .collect::<Vec<_>>();
        frame.render_widget(
            Table::new(rows, [Constraint::Length(28), Constraint::Min(20)])
                .header(Row::new(["Key", "Value"]).style(Style::default().fg(MUTED)))
                .block(block("Advanced config.json view"))
                .style(Style::default().fg(TEXT).bg(PANEL)),
            chunks[1],
        );
    }
}

fn draw_training(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(56), Constraint::Percentage(44)])
        .split(pad(area, 1, 0));
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(8)])
        .split(cols[0]);

    let mut summary = vec![
        Line::from(format!(
            "Source     {}    runs: {}    active: {}    stale: {}",
            app.training.source,
            app.training.list.runs.len(),
            app.training.active_count(),
            app.training.stale_count()
        )),
        Line::from(format!("Directory  {}", app.training.list.runs_dir)),
    ];
    if let Some(warning) = &app.training.warning {
        summary.push(Line::from(vec![
            Span::styled("Fallback   ", Style::default().fg(YELLOW)),
            Span::styled(warning.clone(), Style::default().fg(YELLOW)),
        ]));
    }
    if !app.training.list.errors.is_empty() {
        summary.push(Line::from(vec![
            Span::styled("Reader     ", Style::default().fg(YELLOW)),
            Span::styled(
                app.training.list.errors.join("; "),
                Style::default().fg(YELLOW),
            ),
        ]));
    }
    frame.render_widget(card("Training monitor", summary), left[0]);

    let rows = app
        .training
        .list
        .runs
        .iter()
        .enumerate()
        .skip(scroll_start(app.training.selected, left[1].height, 3))
        .take(visible_rows(left[1].height, 3))
        .map(|(idx, run)| {
            let status = if run.stale {
                format!("{} stale", run.status_label())
            } else {
                run.status_label().to_string()
            };
            Row::new([
                run.id.clone(),
                status,
                run.phase_label().to_string(),
                run.progress_label(),
                run.best_metric_label(),
                run.artifact.clone().unwrap_or_else(|| "-".into()),
            ])
            .style(if idx == app.training.selected {
                Style::default().fg(ACCENT).bg(PANEL_2)
            } else if run.last_error.is_some() || run.read_error.is_some() {
                Style::default().fg(YELLOW).bg(PANEL)
            } else if run.is_active() {
                Style::default().fg(GREEN).bg(PANEL)
            } else {
                Style::default().fg(TEXT).bg(PANEL)
            })
        })
        .collect::<Vec<_>>();
    let rows = if rows.is_empty() {
        vec![Row::new([
            "No runs".to_string(),
            "unknown".to_string(),
            "-".to_string(),
            "-".to_string(),
            "-".to_string(),
            "Create ~/.hipfire/training/runs/<id>/status.json".to_string(),
        ])
        .style(Style::default().fg(MUTED).bg(PANEL))]
    } else {
        rows
    };
    frame.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(18),
                Constraint::Length(14),
                Constraint::Length(14),
                Constraint::Length(10),
                Constraint::Length(9),
                Constraint::Min(20),
            ],
        )
        .header(
            Row::new(["Run", "Status", "Phase", "Progress", "Best", "Artifact"])
                .style(Style::default().fg(MUTED)),
        )
        .block(block("Runs"))
        .style(Style::default().fg(TEXT).bg(PANEL)),
        left[1],
    );

    let detail_lines = training_detail_lines(app);
    frame.render_widget(
        Paragraph::new(Text::from(detail_lines))
            .block(block("Selected run"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        cols[1],
    );
}

fn training_detail_lines(app: &App) -> Vec<Line<'static>> {
    let Some(detail) = &app.training.detail else {
        return vec![Line::from(Span::styled(
            "No selected training run.",
            Style::default().fg(MUTED),
        ))];
    };
    let run = &detail.summary;
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Run        ", Style::default().fg(MUTED)),
            Span::styled(run.id.clone(), Style::default().fg(ACCENT)),
        ]),
        Line::from(format!(
            "Target     {}",
            run.target_model.clone().unwrap_or_else(|| "-".into())
        )),
        Line::from(format!(
            "Artifact   {}",
            run.artifact
                .clone()
                .or_else(|| run.handoff.as_ref().and_then(|h| h.artifact.clone()))
                .unwrap_or_else(|| "-".into())
        )),
        Line::from(format!(
            "Checkpoint {}",
            run.checkpoint
                .as_ref()
                .and_then(|c| c.path.clone().or_else(|| c.state.clone()))
                .unwrap_or_else(|| "-".into())
        )),
        Line::from(format!(
            "Admission  {}",
            run.handoff
                .as_ref()
                .and_then(|h| h
                    .admission_verdict
                    .clone()
                    .or_else(|| h.admission_status.clone()))
                .unwrap_or_else(|| "-".into())
        )),
    ];
    if let Some(issue) = &run.last_error {
        lines.push(Line::from(vec![
            Span::styled("Issue      ", Style::default().fg(YELLOW)),
            Span::styled(issue.message.clone(), Style::default().fg(YELLOW)),
        ]));
    }
    if let Some(err) = &run.read_error {
        lines.push(Line::from(vec![
            Span::styled("Read       ", Style::default().fg(YELLOW)),
            Span::styled(err.clone(), Style::default().fg(YELLOW)),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Recent events",
        Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
    )));
    if detail.recent_events.is_empty() {
        lines.push(Line::from("No structured events recorded."));
    } else {
        for record in detail.recent_events.iter().rev().take(10) {
            let message = record
                .event
                .message()
                .map(str::to_string)
                .unwrap_or_else(|| {
                    serde_json::to_string(&record.event.payload)
                        .unwrap_or_else(|_| "{}".into())
                        .chars()
                        .take(96)
                        .collect::<String>()
                });
            lines.push(Line::from(format!(
                "{:>4}  {:<18} {}",
                record.line,
                record.event.label(),
                message
            )));
        }
    }
    for err in detail.event_errors.iter().take(4) {
        lines.push(Line::from(vec![
            Span::styled(
                format!("line {:>4}  malformed  ", err.line),
                Style::default().fg(YELLOW),
            ),
            Span::styled(err.message.clone(), Style::default().fg(YELLOW)),
        ]));
    }
    lines
}

fn draw_system(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(pad(area, 1, 0));
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(8)])
        .split(cols[0]);

    let gpu_lines = app
        .status
        .gpu_lines
        .iter()
        .map(|line| Line::from(line.clone()))
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(gpu_lines)
            .block(block("Hardware glimpse"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        left[0],
    );

    let paths = app
        .status
        .paths_ok
        .iter()
        .map(|(label, ok)| {
            Row::new([
                label.clone(),
                if *ok {
                    "present".into()
                } else {
                    "missing".into()
                },
            ])
            .style(if *ok {
                Style::default().fg(GREEN)
            } else {
                Style::default().fg(YELLOW)
            })
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Table::new(paths, [Constraint::Length(24), Constraint::Length(12)])
            .header(Row::new(["Path", "Status"]).style(Style::default().fg(MUTED)))
            .block(block("Files"))
            .style(Style::default().bg(PANEL)),
        left[1],
    );

    let mut diagnostic_lines = vec![
        Line::from(Span::styled(
            "Diagnostics roadmap",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from("Prototype 1 is intentionally read-only here."),
        Line::from("Next slice should wrap hipfire diag, kernel cache status, ROCm version,"),
        Line::from("serve logs, model checksums, and first-run setup checks."),
        Line::from(""),
        Line::from(Span::styled(
            "Local model files:",
            Style::default().fg(MUTED),
        )),
    ];
    if app.registry.local_files.is_empty() {
        diagnostic_lines.push(Line::from("No local models under ~/.hipfire/models."));
    } else {
        diagnostic_lines.extend(
            app.registry
                .local_files
                .iter()
                .take(7)
                .map(|m| Line::from(format!("{}  {}", m.size, m.file))),
        );
    }
    diagnostic_lines.extend([
        Line::from(""),
        Line::from(Span::styled(
            "Current health response:",
            Style::default().fg(MUTED),
        )),
        Line::from(app.status.health_text.chars().take(500).collect::<String>()),
    ]);
    frame.render_widget(
        Paragraph::new(Text::from(diagnostic_lines))
            .block(block("System"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        cols[1],
    );
}

fn card(title: &str, lines: Vec<Line<'static>>) -> Paragraph<'static> {
    Paragraph::new(lines)
        .block(block(title))
        .style(Style::default().fg(TEXT).bg(PANEL))
        .wrap(Wrap { trim: false })
}

fn block(title: &str) -> Block<'static> {
    Block::default()
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(PANEL_2))
        .style(Style::default().bg(PANEL))
}

fn pad(area: Rect, x: u16, y: u16) -> Rect {
    Rect {
        x: area.x.saturating_add(x),
        y: area.y.saturating_add(y),
        width: area.width.saturating_sub(x * 2),
        height: area.height.saturating_sub(y * 2),
    }
}

fn visible_rows(height: u16, chrome: u16) -> usize {
    height.saturating_sub(chrome).max(1) as usize
}

fn scroll_start(selected: usize, height: u16, chrome: u16) -> usize {
    let visible = visible_rows(height, chrome);
    selected.saturating_sub(visible.saturating_sub(1))
}
