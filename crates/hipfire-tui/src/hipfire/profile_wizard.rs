// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::io;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use hipfire_config::{
    apply_config_profile, create_config_profile, detect_config_profile, fields,
    list_config_profiles, load_global, write_global_toml, ConfigField, ConfigLayer, ConfigPaths,
    ConfigProfileEntry, ConfigProfileKind, DefaultValue,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};

use super::HipfirePaths;

const BG: Color = Color::Rgb(7, 7, 9);
const PANEL: Color = Color::Rgb(18, 16, 18);
const PANEL_2: Color = Color::Rgb(40, 24, 27);
const TEXT: Color = Color::Rgb(222, 226, 232);
const MUTED: Color = Color::Rgb(142, 150, 163);
const ACCENT: Color = Color::Rgb(237, 45, 57);
const GREEN: Color = Color::Rgb(102, 217, 139);
const RED: Color = Color::Rgb(255, 95, 104);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Focus {
    Profiles,
    Variables,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputMode {
    Browse,
    Search,
    Create,
}

pub struct ProfileWizard {
    paths: ConfigPaths,
    layer: ConfigLayer,
    profiles: Vec<ConfigProfileEntry>,
    active_profile: Option<String>,
    profile_index: usize,
    field_index: usize,
    filtered_fields: Vec<&'static ConfigField>,
    focus: Focus,
    mode: InputMode,
    search: String,
    create_name: String,
    status: String,
    status_error: bool,
}

impl ProfileWizard {
    fn load() -> Result<Self> {
        let root = HipfirePaths::discover().root;
        Self::load_from_paths(ConfigPaths::under(&root))
    }

    fn load_from_paths(paths: ConfigPaths) -> Result<Self> {
        let loaded = load_global(&paths)?;
        let profiles = list_config_profiles(&paths)?;
        let active_profile = detect_config_profile(&paths, &loaded.layer);
        let profile_index = active_profile
            .as_deref()
            .and_then(|active| profiles.iter().position(|profile| profile.name == active))
            .unwrap_or(0);
        Ok(Self {
            paths,
            layer: loaded.layer,
            profiles,
            active_profile,
            profile_index,
            field_index: 0,
            filtered_fields: fields().iter().collect(),
            focus: Focus::Profiles,
            mode: InputMode::Browse,
            search: String::new(),
            create_name: String::new(),
            status: "Select a profile or browse the complete configuration schema.".to_owned(),
            status_error: false,
        })
    }

    fn refresh_profiles(&mut self) -> Result<()> {
        self.profiles = list_config_profiles(&self.paths)?;
        self.active_profile = detect_config_profile(&self.paths, &self.layer);
        self.profile_index = self
            .profile_index
            .min(self.profiles.len().saturating_sub(1));
        Ok(())
    }

    fn apply_selected_profile(&mut self) {
        let Some(name) = self
            .profiles
            .get(self.profile_index)
            .map(|profile| profile.name.clone())
        else {
            return;
        };
        match self.apply_named_profile(&name) {
            Ok(()) => self.set_status(format!("Applied profile '{name}'."), false),
            Err(error) => self.set_status(format!("Could not apply '{name}': {error}"), true),
        }
    }

    fn apply_named_profile(&mut self, name: &str) -> Result<()> {
        let mut next = self.layer.clone();
        apply_config_profile(&mut next, &self.paths, name)?;
        write_global_toml(&self.paths, &next)?;
        self.layer = next;
        self.refresh_profiles()?;
        Ok(())
    }

    fn create_profile(&mut self) {
        let name = self.create_name.trim().to_owned();
        if name.is_empty() {
            self.set_status("Profile name must not be empty.".to_owned(), true);
            return;
        }
        match create_config_profile(&self.paths, &name, &self.layer) {
            Ok(path) => {
                if let Err(error) = self.refresh_profiles() {
                    self.set_status(
                        format!("Profile was saved, but the list could not refresh: {error}"),
                        true,
                    );
                    return;
                }
                if let Some(index) = self
                    .profiles
                    .iter()
                    .position(|profile| profile.name == name)
                {
                    self.profile_index = index;
                }
                self.mode = InputMode::Browse;
                self.create_name.clear();
                self.set_status(
                    format!("Created profile '{name}' at {}.", path.display()),
                    false,
                );
            }
            Err(error) => {
                self.set_status(format!("Could not create '{name}': {error}"), true);
            }
        }
    }

    fn set_status(&mut self, status: String, error: bool) {
        self.status = status;
        self.status_error = error;
    }

    fn update_filter(&mut self) {
        self.filtered_fields = fields()
            .iter()
            .filter(|field| field_matches(field, &self.search))
            .collect();
        self.field_index = self
            .field_index
            .min(self.filtered_fields.len().saturating_sub(1));
    }

    fn move_selection(&mut self, delta: isize) {
        let (index, len) = match self.focus {
            Focus::Profiles => (&mut self.profile_index, self.profiles.len()),
            Focus::Variables => (&mut self.field_index, self.filtered_fields.len()),
        };
        if len == 0 {
            *index = 0;
            return;
        }
        if delta < 0 {
            *index = index.saturating_sub(delta.unsigned_abs());
        } else {
            *index = (*index + delta as usize).min(len - 1);
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        match self.mode {
            InputMode::Search => match key.code {
                KeyCode::Esc => {
                    self.search.clear();
                    self.update_filter();
                    self.mode = InputMode::Browse;
                }
                KeyCode::Enter => self.mode = InputMode::Browse,
                KeyCode::Backspace => {
                    self.search.pop();
                    self.update_filter();
                }
                KeyCode::Char(character) => {
                    self.search.push(character);
                    self.update_filter();
                }
                _ => {}
            },
            InputMode::Create => match key.code {
                KeyCode::Esc => {
                    self.create_name.clear();
                    self.mode = InputMode::Browse;
                }
                KeyCode::Enter => self.create_profile(),
                KeyCode::Backspace => {
                    self.create_name.pop();
                }
                KeyCode::Char(character) => self.create_name.push(character),
                _ => {}
            },
            InputMode::Browse => match key.code {
                KeyCode::Char('q') => return true,
                KeyCode::Tab | KeyCode::BackTab => {
                    self.focus = match self.focus {
                        Focus::Profiles => Focus::Variables,
                        Focus::Variables => Focus::Profiles,
                    };
                }
                KeyCode::Up => self.move_selection(-1),
                KeyCode::Down => self.move_selection(1),
                KeyCode::Enter if self.focus == Focus::Profiles => self.apply_selected_profile(),
                KeyCode::Char('n') => {
                    self.focus = Focus::Profiles;
                    self.create_name.clear();
                    self.mode = InputMode::Create;
                }
                KeyCode::Char('/') => {
                    self.focus = Focus::Variables;
                    self.mode = InputMode::Search;
                }
                KeyCode::Esc if !self.search.is_empty() => {
                    self.search.clear();
                    self.update_filter();
                }
                _ => {}
            },
        }
        false
    }
}

pub fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut wizard = ProfileWizard::load()?;
    loop {
        terminal.draw(|frame| draw(frame, &wizard))?;
        if event::poll(std::time::Duration::from_millis(80))? {
            match event::read()? {
                Event::Key(key) if wizard.handle_key(key) => break,
                Event::Key(_) | Event::Mouse(_) | Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }
    Ok(())
}

fn field_matches(field: &ConfigField, query: &str) -> bool {
    if query.is_empty() {
        return true;
    }
    let query = query.to_ascii_lowercase();
    let env = field.env_compat.unwrap_or_default();
    let registry = if field.registry_allowed {
        "allowed"
    } else {
        "not allowed"
    };
    let experimental = if field.experimental {
        "experimental"
    } else {
        "stable"
    };
    [
        field.key.to_owned(),
        field.legacy_key.to_owned(),
        field.help.to_owned(),
        env.to_owned(),
        format!("{:?}", field.category),
        format!("{:?}", field.scope),
        format!("{:?}", field.rule),
        registry.to_owned(),
        experimental.to_owned(),
    ]
    .iter()
    .any(|candidate| {
        let candidate = candidate.to_ascii_lowercase();
        if !candidate.contains(&query) {
            return false;
        }
        // Details-pane "not allowed" must not match a bare "allowed" query.
        if query == "allowed" && candidate == "not allowed" {
            return false;
        }
        true
    })
}

fn draw(frame: &mut Frame, wizard: &ProfileWizard) {
    let area = frame.area();
    frame.render_widget(Clear, area);
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(10),
            Constraint::Length(2),
        ])
        .split(area);

    let status_color = if wizard.status_error { RED } else { MUTED };
    let header = Text::from(vec![
        Line::from(vec![
            Span::styled(
                "HIPFIRE  ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Configuration profiles",
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::styled(
            wizard.status.clone(),
            Style::default().fg(status_color),
        )),
    ]);
    frame.render_widget(
        Paragraph::new(header)
            .block(
                Block::default()
                    .borders(Borders::BOTTOM)
                    .border_style(Style::default().fg(PANEL_2)),
            )
            .style(Style::default().fg(TEXT).bg(BG)),
        root[0],
    );

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(root[1]);
    draw_profiles(frame, wizard, body[0]);
    draw_variables(frame, wizard, body[1]);

    let footer = match wizard.mode {
        InputMode::Browse => {
            "↑/↓ move · Enter apply · n new · / search variables · Tab/Shift+Tab switch · q quit"
                .to_owned()
        }
        InputMode::Search => format!(
            "Search: {}_  · Enter keep filter · Esc clear",
            wizard.search
        ),
        InputMode::Create => format!(
            "New profile: {}_  · Enter create snapshot · Esc cancel",
            wizard.create_name
        ),
    };
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().fg(MUTED).bg(PANEL)),
        root[2],
    );
}

fn draw_profiles(frame: &mut Frame, wizard: &ProfileWizard, area: Rect) {
    let focused = wizard.focus == Focus::Profiles;
    let items: Vec<ListItem> = wizard
        .profiles
        .iter()
        .map(|profile| {
            let active = wizard.active_profile.as_deref() == Some(profile.name.as_str());
            let kind = match profile.kind {
                ConfigProfileKind::Builtin => "built-in",
                ConfigProfileKind::Custom => "custom",
            };
            ListItem::new(Line::from(vec![
                Span::styled(if active { "● " } else { "  " }, Style::default().fg(GREEN)),
                Span::styled(profile.name.clone(), Style::default().fg(TEXT)),
                Span::styled(format!("  {kind}"), Style::default().fg(MUTED)),
            ]))
        })
        .collect();
    let mut state = ListState::default();
    if !items.is_empty() {
        state.select(Some(wizard.profile_index));
    }
    let border = if focused { ACCENT } else { PANEL_2 };
    let title = if wizard.mode == InputMode::Create {
        format!(" Profiles · new: {}_ ", wizard.create_name)
    } else {
        " Profiles ".to_owned()
    };
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border))
                .title(title),
        )
        .highlight_style(
            Style::default()
                .bg(PANEL_2)
                .fg(TEXT)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ")
        .style(Style::default().fg(TEXT).bg(PANEL));
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_variables(frame: &mut Frame, wizard: &ProfileWizard, area: Rect) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(area);
    let focused = wizard.focus == Focus::Variables;
    let border = if focused { ACCENT } else { PANEL_2 };
    let title = if wizard.search.is_empty() {
        format!(" Variables · all {} ", fields().len())
    } else {
        format!(
            " Variables · /{} · {} matches ",
            wizard.search,
            wizard.filtered_fields.len()
        )
    };
    let items: Vec<ListItem> = wizard
        .filtered_fields
        .iter()
        .map(|field| {
            ListItem::new(Line::from(vec![
                Span::styled(field.key, Style::default().fg(TEXT)),
                Span::styled(
                    format!("  ({})", field.legacy_key),
                    Style::default().fg(MUTED),
                ),
            ]))
        })
        .collect();
    let mut state = ListState::default();
    if !items.is_empty() {
        state.select(Some(wizard.field_index));
    }
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border))
                .title(title),
        )
        .highlight_style(
            Style::default()
                .bg(PANEL_2)
                .fg(TEXT)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ")
        .style(Style::default().fg(TEXT).bg(PANEL));
    frame.render_stateful_widget(list, columns[0], &mut state);

    let detail = wizard
        .filtered_fields
        .get(wizard.field_index)
        .map(|field| field_detail(wizard, field))
        .unwrap_or_else(|| Text::from("No configuration variables match this search."));
    frame.render_widget(
        Paragraph::new(detail)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(PANEL_2))
                    .title(" Variable reference "),
            )
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        columns[1],
    );
}

fn field_detail(wizard: &ProfileWizard, field: &ConfigField) -> Text<'static> {
    let current = wizard
        .layer
        .get(field.key)
        .map(|value| format!("{value:?} (profile override)"))
        .unwrap_or_else(|| format!("{} (inherited default)", default_text(field.default)));
    let env = field.env_compat.unwrap_or("none");
    Text::from(vec![
        Line::from(Span::styled(
            field.key.to_owned(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        detail_line("Legacy key", field.legacy_key),
        detail_line("Category", &format!("{:?}", field.category)),
        detail_line("Scope", &format!("{:?}", field.scope)),
        detail_line("Type / rule", &format!("{:?}", field.rule)),
        detail_line("Default", &default_text(field.default)),
        detail_line("Current", &current),
        detail_line("Environment", env),
        detail_line(
            "Registry",
            if field.registry_allowed {
                "allowed"
            } else {
                "not allowed"
            },
        ),
        detail_line(
            "Experimental",
            if field.experimental { "yes" } else { "no" },
        ),
        Line::from(""),
        Line::from(Span::styled(
            "What it controls",
            Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
        )),
        Line::from(field.help.to_owned()),
    ])
}

fn detail_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<13}"), Style::default().fg(MUTED)),
        Span::styled(value.to_owned(), Style::default().fg(TEXT)),
    ])
}

fn default_text(value: DefaultValue) -> String {
    match value {
        DefaultValue::Bool(value) => value.to_string(),
        DefaultValue::Integer(value) => value.to_string(),
        DefaultValue::Float(value) => value.to_string(),
        DefaultValue::String(value) => format!("\"{value}\""),
        DefaultValue::Null => "null".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use hipfire_config::{load_global, ConfigValue};

    use super::*;

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn test_paths(label: &str) -> ConfigPaths {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "hipfire-profile-wizard-{label}-{}-{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        ConfigPaths::under(&root)
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn search_matches_help_legacy_and_environment_metadata() {
        let help_field = fields()
            .iter()
            .find(|field| !field.help.is_empty())
            .unwrap();
        assert!(field_matches(
            help_field,
            help_field.help.split_whitespace().next().unwrap()
        ));
        assert!(field_matches(help_field, help_field.legacy_key));
        let env_field = fields()
            .iter()
            .find(|field| field.env_compat.is_some())
            .unwrap();
        assert!(field_matches(env_field, env_field.env_compat.unwrap()));
    }

    #[test]
    fn search_matches_registry_and_experimental_metadata() {
        let allowed = fields()
            .iter()
            .find(|field| {
                field.registry_allowed
                    && !field_text_contains(field, "not allowed")
                    && !field_text_contains(field, "stable")
                    && !field_text_contains(field, "experimental")
            })
            .expect("registry-allowed field");
        let not_allowed = fields()
            .iter()
            .find(|field| {
                !field.registry_allowed
                    && !field_text_contains(field, "allowed")
                    && !field_text_contains(field, "stable")
                    && !field_text_contains(field, "experimental")
            })
            .expect("registry-not-allowed field");
        assert!(field_matches(allowed, "allowed"));
        assert!(!field_matches(allowed, "not allowed"));
        assert!(field_matches(not_allowed, "not allowed"));
        assert!(!field_matches(not_allowed, "allowed"));

        let experimental = fields()
            .iter()
            .find(|field| {
                field.experimental
                    && !field_text_contains(field, "stable")
                    && !field_text_contains(field, "allowed")
                    && !field_text_contains(field, "not allowed")
            })
            .expect("experimental field");
        let stable = fields()
            .iter()
            .find(|field| {
                !field.experimental
                    && !field_text_contains(field, "experimental")
                    && !field_text_contains(field, "allowed")
                    && !field_text_contains(field, "not allowed")
            })
            .expect("stable field");
        assert!(field_matches(experimental, "experimental"));
        assert!(!field_matches(experimental, "stable"));
        assert!(field_matches(stable, "stable"));
        assert!(!field_matches(stable, "experimental"));
    }

    fn field_text_contains(field: &ConfigField, needle: &str) -> bool {
        let needle = needle.to_ascii_lowercase();
        [
            field.key.to_owned(),
            field.legacy_key.to_owned(),
            field.help.to_owned(),
            field.env_compat.unwrap_or_default().to_owned(),
            format!("{:?}", field.category),
            format!("{:?}", field.scope),
            format!("{:?}", field.rule),
        ]
        .iter()
        .any(|candidate| candidate.to_ascii_lowercase().contains(&needle))
    }

    #[test]
    fn selecting_profile_replaces_the_complete_sparse_layer() {
        let paths = test_paths("select");
        let mut wizard = ProfileWizard::load_from_paths(paths.clone()).unwrap();
        wizard
            .layer
            .set("generation.temperature", ConfigValue::Float(0.42))
            .unwrap();
        write_global_toml(&paths, &wizard.layer).unwrap();
        wizard.apply_named_profile("redline").unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.layer, wizard.layer);
        assert_eq!(
            detect_config_profile(&paths, &loaded.layer).as_deref(),
            Some("redline")
        );
        assert!(loaded.layer.get("generation.temperature").is_none());
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn selecting_hip_profile_persists_redline_opt_out() {
        let paths = test_paths("hip-opt-out");
        let mut wizard = ProfileWizard::load_from_paths(paths.clone()).unwrap();
        wizard.apply_named_profile("hip").unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(
            loaded.layer.get("replay.backend"),
            Some(&ConfigValue::String("hip".to_owned()))
        );
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn creating_custom_profile_snapshots_current_layer() {
        let paths = test_paths("create");
        let mut wizard = ProfileWizard::load_from_paths(paths.clone()).unwrap();
        wizard
            .layer
            .set("generation.temperature", ConfigValue::Float(0.42))
            .unwrap();
        wizard.mode = InputMode::Create;
        wizard.create_name = "lab".to_owned();
        wizard.create_profile();
        assert_eq!(wizard.mode, InputMode::Browse);
        let saved = hipfire_config::load_config_profile(&paths, "lab").unwrap();
        assert_eq!(saved, wizard.layer);
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn invalid_name_keeps_create_mode_for_correction() {
        let paths = test_paths("invalid");
        let mut wizard = ProfileWizard::load_from_paths(paths.clone()).unwrap();
        wizard.mode = InputMode::Create;
        wizard.create_name = "../escape".to_owned();
        wizard.handle_key(key(KeyCode::Enter));
        assert_eq!(wizard.mode, InputMode::Create);
        assert!(wizard.status_error);
        assert!(wizard.status.contains("invalid profile name"));
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn key_modes_switch_and_search_filters_live() {
        let paths = test_paths("keys");
        let mut wizard = ProfileWizard::load_from_paths(paths.clone()).unwrap();
        wizard.handle_key(key(KeyCode::Char('/')));
        assert_eq!(wizard.mode, InputMode::Search);
        assert_eq!(wizard.focus, Focus::Variables);
        for character in "replay".chars() {
            wizard.handle_key(key(KeyCode::Char(character)));
        }
        assert!(!wizard.filtered_fields.is_empty());
        assert!(wizard.filtered_fields.len() < fields().len());
        wizard.handle_key(key(KeyCode::Esc));
        assert_eq!(wizard.mode, InputMode::Browse);
        assert_eq!(wizard.filtered_fields.len(), fields().len());
        wizard.handle_key(key(KeyCode::Char('n')));
        assert_eq!(wizard.mode, InputMode::Create);
        wizard.handle_key(key(KeyCode::Esc));
        assert_eq!(wizard.mode, InputMode::Browse);
        let _ = fs::remove_dir_all(paths.root);
    }
}
