// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Row, Table, Tabs, Wrap},
    Frame,
};

use crate::{
    app::{App, Tab},
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

    let title = Line::from(vec![
        Span::styled(
            "hipfire",
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" / wick spike", Style::default().fg(ACCENT)),
        Span::styled(
            format!(
                "    serve: {}    model: {}",
                app.status.serve_label(),
                app.active_model
            ),
            Style::default().fg(MUTED),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(title)
            .style(Style::default().bg(BG))
            .alignment(Alignment::Center),
        chunks[0],
    );

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
        .highlight_style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
        .divider(Span::styled(" | ", Style::default().fg(PANEL_2)));
    frame.render_widget(tabs, chunks[1]);
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    let help = match app.tab {
        Tab::Chat => {
            "Tab switch  Enter send/start serve  Ctrl+O newline  Up/Down scroll  Esc blur/quit"
        }
        Tab::Models => {
            "Tab switch  Up/Down select  Enter expand/select  Left/Right fold  r refresh  q quit"
        }
        Tab::Settings => "Tab switch  e easy  a advanced  Up/Down select  r refresh  q quit",
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

fn draw_home(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(pad(area, 1, 0));
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(8)])
        .split(cols[0]);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(12), Constraint::Min(6)])
        .split(cols[1]);

    let serve_color = if app.status.serve_http_ok {
        GREEN
    } else if app.status.serve_pid_alive || app.status.serve_pid.is_some() {
        YELLOW
    } else {
        RED
    };
    let mut status = vec![
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
        Line::from(format!(
            "Config     {} ({})",
            app.config.default_model,
            if app.config.loaded_from_disk {
                "custom"
            } else {
                "defaults"
            }
        )),
        Line::from(format!(
            "Overrides  {} model overlays",
            app.config.per_model_count
        )),
        Line::from(format!(
            "Models     {} local / {} registry",
            app.registry.local_files.len(),
            app.registry.models.len()
        )),
    ];
    if let Some(warning) = &app.config.warning {
        status.push(Line::from(vec![
            Span::styled("Config     ", Style::default().fg(YELLOW)),
            Span::styled(warning.clone(), Style::default().fg(YELLOW)),
        ]));
    }
    if let Some(warning) = &app.registry.warning {
        status.push(Line::from(vec![
            Span::styled("Registry   ", Style::default().fg(YELLOW)),
            Span::styled(warning.clone(), Style::default().fg(YELLOW)),
        ]));
    }
    frame.render_widget(card("Runtime", status), left[0]);

    let actions = vec![
        ListItem::new("Chat: use the Chat tab; it streams through existing hipfire serve."),
        ListItem::new("Models: browse registry and local downloads."),
        ListItem::new("Settings: easy/advanced split, read-only in prototype 1."),
        ListItem::new("System: hardware and path checks."),
    ];
    frame.render_widget(
        List::new(actions)
            .block(block("Prototype map"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        left[1],
    );

    let philosophy = Text::from(vec![
        Line::from(Span::styled(
            "Spike leash",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from("This is Wick-shaped, but still hipfire-owned."),
        Line::from("No command replacement, no plugin runtime, no generated skills yet."),
        Line::from("Every active surface reads real local state or talks to the existing daemon."),
        Line::from("The disabled tool cards below mark the next vertical slice."),
    ]);
    frame.render_widget(
        Paragraph::new(philosophy)
            .block(block("Intent"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        right[0],
    );

    let tools = vec![
        Row::new([
            "Quantizer",
            "planned",
            "front-end existing hipfire quantize flow",
        ]),
        Row::new(["AWQ import", "planned", "model conversion/install workflow"]),
        Row::new([
            "TriAttention",
            "planned",
            "sidecar generation/validation wizard",
        ]),
        Row::new([
            "Agent profiles",
            "later",
            "/default, /agent, /code profile system",
        ]),
    ];
    frame.render_widget(
        Table::new(
            tools,
            [
                Constraint::Length(16),
                Constraint::Length(12),
                Constraint::Min(20),
            ],
        )
        .header(Row::new(["Tooling", "Status", "Reason"]).style(Style::default().fg(MUTED)))
        .block(block("Tooling runway"))
        .style(Style::default().fg(TEXT).bg(PANEL))
        .row_highlight_style(Style::default().bg(PANEL_2)),
        right[1],
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
                    let extras = match (row.has_triattn, row.has_mtp) {
                        (true, true) => "triattn mtp",
                        (true, false) => "triattn",
                        (false, true) => "mtp",
                        _ => "",
                    };
                    Row::new([
                        format!("  {}", row.tag),
                        status.into(),
                        format!("{:.1} GB", row.entry.size_gb),
                        format!("{:.0} GB", row.entry.min_vram_gb),
                        extras.into(),
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
