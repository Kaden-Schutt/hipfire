// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
        Arc, Mutex,
    },
    thread,
};

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::hipfire::{
    chat::{stream_chat, ChatEvent, ChatMessage},
    cli,
    config::ConfigState,
    registry::{RegistryAction, RegistryState},
    status::{spawn_health_poller, start_background_serve, HealthUpdate, ProbeTarget, StatusState},
    HipfirePaths,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    Home,
    Chat,
    Models,
    Settings,
    System,
}

impl Tab {
    pub const ALL: [Tab; 5] = [
        Tab::Home,
        Tab::Chat,
        Tab::Models,
        Tab::Settings,
        Tab::System,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Home => "Home",
            Tab::Chat => "Chat",
            Tab::Models => "Models",
            Tab::Settings => "Settings",
            Tab::System => "System",
        }
    }
}

pub struct App {
    pub paths: HipfirePaths,
    pub config: ConfigState,
    pub registry: RegistryState,
    pub status: StatusState,
    pub active_model: String,
    pub tab: Tab,
    pub settings_easy: bool,
    pub settings_selected: usize,
    pub settings_edit: Option<SettingsEdit>,
    /// When true, Enter-applied settings target the active model
    /// (`hipfire config <tag> set ...`) instead of the global config.
    pub settings_scope_model: bool,
    pub chat: ChatState,
    pub last_reload: String,
    /// Endpoint shared with the background health poller; updated on reload
    /// so config host/port changes are picked up without a restart.
    probe_target: ProbeTarget,
    health_rx: Option<Receiver<HealthUpdate>>,
}

/// An in-progress settings edit; applied through the hipfire CLI so the CLI
/// remains the single validation source of truth.
#[derive(Clone, Debug)]
pub struct SettingsEdit {
    pub key: String,
    pub buffer: String,
}

impl App {
    pub fn load() -> Result<Self> {
        let paths = HipfirePaths::discover();
        let config = ConfigState::load(&paths);
        let registry = RegistryState::load(&paths);
        let status = StatusState::load(&paths, &config);
        let active_model = config.default_model.clone();
        let probe_target: ProbeTarget = Arc::new(Mutex::new((config.probe_host(), config.port)));
        let health_rx = Some(spawn_health_poller(
            paths.serve_pid.clone(),
            Arc::clone(&probe_target),
        ));
        Ok(Self {
            paths,
            config,
            registry,
            status,
            active_model,
            tab: Tab::Home,
            settings_easy: true,
            settings_selected: 0,
            settings_edit: None,
            settings_scope_model: false,
            chat: ChatState::default(),
            last_reload: "loaded hipfire state".into(),
            probe_target,
            health_rx,
        })
    }

    pub fn reload(&mut self) {
        self.config = ConfigState::load(&self.paths);
        self.registry = RegistryState::load(&self.paths);
        self.status = StatusState::load(&self.paths, &self.config);
        // Repoint the background poller at the (possibly changed) endpoint.
        {
            let mut target = self
                .probe_target
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            *target = (self.config.probe_host(), self.config.port);
        }
        self.last_reload = "reloaded config, registry, models, and serve status".into();
    }

    /// Apply the freshest snapshot from the 2s background health poller.
    pub fn drain_health_events(&mut self) {
        let Some(rx) = &self.health_rx else {
            return;
        };
        let mut latest = None;
        while let Ok(update) = rx.try_recv() {
            latest = Some(update);
        }
        if let Some(update) = latest {
            self.status.serve_http_ok = update.serve_http_ok;
            self.status.health_text = update.health_text;
            self.status.serve_pid = update.serve_pid;
            self.status.serve_pid_alive = update.serve_pid_alive;
        }
    }

    pub fn next_tab(&mut self) {
        let idx = Tab::ALL.iter().position(|t| *t == self.tab).unwrap_or(0);
        self.tab = Tab::ALL[(idx + 1) % Tab::ALL.len()];
    }

    pub fn prev_tab(&mut self) {
        let idx = Tab::ALL.iter().position(|t| *t == self.tab).unwrap_or(0);
        self.tab = Tab::ALL[(idx + Tab::ALL.len() - 1) % Tab::ALL.len()];
    }

    /// True only while a text-entry surface is actually capturing keystrokes.
    /// Focus is per-tab: the chat input only swallows keys while the Chat tab
    /// is active (and not mid-stream), so `q`/`r` work everywhere else from
    /// launch without needing Esc first.
    pub fn text_input_active(&self) -> bool {
        match self.tab {
            Tab::Chat => self.chat.is_input_focused() && !self.chat.sending,
            Tab::Settings => self.settings_edit.is_some(),
            _ => false,
        }
    }

    /// Esc on an active text surface: blur the chat input or cancel the
    /// settings edit without applying it.
    pub fn dismiss_text_input(&mut self) {
        match self.tab {
            Tab::Chat => self.chat.blur_input(),
            Tab::Settings => {
                if self.settings_edit.take().is_some() {
                    self.last_reload = "edit cancelled; nothing applied".into();
                }
            }
            _ => {}
        }
    }

    pub fn handle_tab_key(&mut self, key: KeyEvent) {
        match self.tab {
            Tab::Chat => self.handle_chat_key(key),
            Tab::Models => self.handle_models_key(key),
            Tab::Settings => self.handle_settings_key(key),
            _ => {}
        }
    }

    fn handle_models_key(&mut self, key: KeyEvent) {
        let len = self.registry.visible_len().max(1);
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.registry.selected = (self.registry.selected + 1).min(len - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.registry.selected = self.registry.selected.saturating_sub(1);
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if let Some(action) = self.registry.activate_selected() {
                    match action {
                        RegistryAction::ToggledGroup { name, expanded } => {
                            self.last_reload = format!(
                                "{} {name}",
                                if expanded { "expanded" } else { "collapsed" }
                            );
                        }
                        RegistryAction::SelectedModel { tag } => {
                            self.active_model = tag.clone();
                            self.chat.status = format!("model selected: {tag}");
                            self.last_reload =
                                "selected model for this TUI session; config unchanged".into();
                        }
                    }
                }
            }
            KeyCode::Right => {
                if let Some(name) = self.registry.expand_selected_group() {
                    self.last_reload = format!("expanded {name}");
                }
            }
            KeyCode::Left => {
                if let Some(name) = self.registry.collapse_selected_group() {
                    self.last_reload = format!("collapsed {name}");
                }
            }
            _ => {}
        }
    }

    fn handle_chat_key(&mut self, key: KeyEvent) {
        if self.chat.sending {
            self.chat.status = "generation in progress".into();
            return;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
            self.chat.input.push('\n');
            self.chat.focus_input();
            return;
        }

        if !self.chat.is_input_focused() {
            match key.code {
                KeyCode::Enter | KeyCode::Char('i') => self.chat.focus_input(),
                KeyCode::Up => self.chat.scroll = self.chat.scroll.saturating_add(1),
                KeyCode::Down => self.chat.scroll = self.chat.scroll.saturating_sub(1),
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Enter => {
                let prompt = self.chat.input.trim().to_string();
                if prompt.is_empty() {
                    return;
                }
                if !self.status.serve_http_ok {
                    self.start_serve_for_chat();
                    return;
                }
                self.chat.input.clear();
                self.chat.messages.push(ChatMessage {
                    role: "user".into(),
                    content: prompt.clone(),
                    reasoning: String::new(),
                });
                self.chat.messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: String::new(),
                    reasoning: String::new(),
                });
                self.chat.sending = true;
                self.chat.status = "streaming from hipfire serve".into();

                let (tx, rx) = mpsc::channel();
                self.chat.rx = Some(rx);
                self.chat.abort = Arc::new(AtomicBool::new(false));
                let abort = Arc::clone(&self.chat.abort);
                let host = self.config.probe_host();
                let port = self.config.port;
                let model = self.active_model.clone();
                let mut messages = self.chat.messages.clone();
                if let Some(last) = messages.last_mut() {
                    if last.role == "assistant" && last.content.is_empty() {
                        messages.pop();
                    }
                }
                // Belt and braces: serde skip_serializing already keeps
                // reasoning out of the request body.
                thread::spawn(move || {
                    let _ = stream_chat(&host, port, &model, &messages, tx, abort);
                });
            }
            KeyCode::Backspace => {
                self.chat.input.pop();
            }
            KeyCode::Char(c) => {
                self.chat.input.push(c);
            }
            KeyCode::Up => {
                self.chat.scroll = self.chat.scroll.saturating_add(1);
            }
            KeyCode::Down => {
                self.chat.scroll = self.chat.scroll.saturating_sub(1);
            }
            _ => {}
        }
    }

    fn start_serve_for_chat(&mut self) {
        if self.status.serve_pid_alive {
            self.chat.status =
                "serve process exists; health auto-refreshes every 2s, retry shortly".into();
            return;
        }

        match start_background_serve() {
            Ok(()) => {
                self.chat.status =
                    "starting serve -d; health auto-refreshes, retry once it shows online".into();
                self.last_reload = "requested background serve start".into();
                self.status = StatusState::load(&self.paths, &self.config);
            }
            Err(err) => {
                self.chat.status = format!("{err}");
            }
        }
    }

    fn handle_settings_key(&mut self, key: KeyEvent) {
        if self.settings_edit.is_some() {
            match key.code {
                KeyCode::Enter => self.apply_settings_edit(),
                KeyCode::Backspace => {
                    if let Some(edit) = &mut self.settings_edit {
                        edit.buffer.pop();
                    }
                }
                KeyCode::Char(c) => {
                    if let Some(edit) = &mut self.settings_edit {
                        edit.buffer.push(c);
                    }
                }
                _ => {}
            }
            return;
        }

        let len = if self.settings_easy {
            self.config.easy_rows().len()
        } else {
            self.config.values.len()
        }
        .max(1);
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.settings_selected = (self.settings_selected + 1).min(len - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.settings_selected = self.settings_selected.saturating_sub(1);
            }
            KeyCode::Enter => self.begin_settings_edit(),
            KeyCode::Char('m') => {
                self.settings_scope_model = !self.settings_scope_model;
                self.last_reload = if self.settings_scope_model {
                    format!(
                        "settings scope: per-model ({}) — applies via `config {} set`",
                        self.active_model, self.active_model
                    )
                } else {
                    "settings scope: global — applies via `config set`".into()
                };
            }
            _ => {}
        }
    }

    /// The config key + current value behind the selected settings row.
    pub fn selected_settings_entry(&self) -> Option<(String, String)> {
        if self.settings_easy {
            let rows = self.config.easy_rows();
            let row = rows.get(self.settings_selected)?;
            Some(((*row.key.as_ref()?).to_string(), row.value.clone()))
        } else {
            self.config
                .values
                .iter()
                .nth(self.settings_selected)
                .map(|(k, v)| (k.clone(), v.clone()))
        }
    }

    fn begin_settings_edit(&mut self) {
        match self.selected_settings_entry() {
            Some((key, value)) => {
                self.settings_edit = Some(SettingsEdit { key, buffer: value });
                self.last_reload = "editing: Enter applies via hipfire CLI, Esc cancels".into();
            }
            None => {
                self.last_reload =
                    "row is informational; edit host/port in advanced view (a)".into();
            }
        }
    }

    fn apply_settings_edit(&mut self) {
        let Some(edit) = self.settings_edit.clone() else {
            return;
        };
        let value = edit.buffer.trim().to_string();
        if value.is_empty() {
            self.last_reload = "value is empty; type a value or Esc to cancel".into();
            return;
        }
        let scope = if self.settings_scope_model {
            Some(self.active_model.clone())
        } else {
            None
        };
        let args = cli::config_set_args(scope.as_deref(), &edit.key, &value);
        let Some(cli_inv) = cli::resolve_cli() else {
            self.last_reload =
                "hipfire CLI not found: install hipfire, run from a checkout, or set HIPFIRE_TUI_CLI"
                    .into();
            return;
        };
        match cli::run_cli(&cli_inv, &args) {
            Ok(out) => {
                self.settings_edit = None;
                self.reload();
                self.last_reload = format!(
                    "{}: {}",
                    cli_inv.label,
                    if out.is_empty() {
                        "applied".into()
                    } else {
                        out
                    }
                );
            }
            Err(err) => {
                // Keep the edit buffer so the user can correct it; the CLI's
                // stderr is the status line.
                self.last_reload = format!("rejected: {err}");
            }
        }
    }

    pub fn drain_chat_events(&mut self) {
        let mut finished = false;
        if let Some(rx) = self.chat.rx.take() {
            while let Ok(event) = rx.try_recv() {
                match event {
                    ChatEvent::Delta(text) => {
                        if let Some(last) = self.chat.messages.last_mut() {
                            last.content.push_str(&text);
                        }
                    }
                    ChatEvent::Reasoning(text) => {
                        if let Some(last) = self.chat.messages.last_mut() {
                            last.reasoning.push_str(&text);
                        }
                    }
                    ChatEvent::Status(status) => self.chat.status = status,
                    ChatEvent::Done => {
                        self.chat.status = "ready".into();
                        self.chat.sending = false;
                        finished = true;
                    }
                    ChatEvent::Aborted => {
                        self.chat.status = "stream aborted; partial reply kept".into();
                        self.chat.sending = false;
                        finished = true;
                    }
                    ChatEvent::Error(err) => {
                        self.chat.status = format!("error: {err}");
                        self.chat.sending = false;
                        finished = true;
                    }
                }
            }

            if !finished {
                self.chat.rx = Some(rx);
            }
        }
    }
}

#[cfg(test)]
impl App {
    /// Construct an App with empty state and no health/GPU probing, for
    /// keymap and focus tests.
    pub fn test_app() -> Self {
        use std::collections::{BTreeMap, BTreeSet};
        Self {
            paths: HipfirePaths::discover(),
            config: ConfigState {
                host: "127.0.0.1".into(),
                port: 11435,
                default_model: "qwen3.5:9b".into(),
                values: BTreeMap::new(),
                per_model_count: 0,
                loaded_from_disk: false,
                warning: None,
            },
            registry: RegistryState {
                models: Vec::new(),
                aliases: BTreeMap::new(),
                local_files: Vec::new(),
                selected: 0,
                expanded_groups: BTreeSet::new(),
                loaded_path: None,
                warning: None,
            },
            status: StatusState {
                serve_pid: None,
                serve_pid_alive: false,
                serve_http_ok: false,
                health_text: String::new(),
                gpu_lines: Vec::new(),
                paths_ok: Vec::new(),
            },
            active_model: "qwen3.5:9b".into(),
            tab: Tab::Home,
            settings_easy: true,
            settings_selected: 0,
            settings_edit: None,
            settings_scope_model: false,
            chat: ChatState::default(),
            last_reload: String::new(),
            probe_target: Arc::new(Mutex::new(("127.0.0.1".into(), 11435))),
            health_rx: None,
        }
    }
}

pub struct ChatState {
    pub input: String,
    pub messages: Vec<ChatMessage>,
    pub status: String,
    pub sending: bool,
    /// Render reasoning blocks expanded (dim) or collapsed to one line.
    pub show_reasoning: bool,
    pub scroll: u16,
    rx: Option<Receiver<ChatEvent>>,
    abort: Arc<AtomicBool>,
    input_focused: bool,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            input: String::new(),
            messages: Vec::new(),
            status: "ready".into(),
            sending: false,
            show_reasoning: true,
            scroll: 0,
            rx: None,
            abort: Arc::new(AtomicBool::new(false)),
            input_focused: true,
        }
    }
}

impl ChatState {
    pub fn focus_input(&mut self) {
        self.input_focused = true;
    }

    pub fn blur_input(&mut self) {
        self.input_focused = false;
    }

    pub fn is_input_focused(&self) -> bool {
        self.input_focused
    }

    /// Ask the streaming thread to stop at the next SSE line. The UI flips
    /// to non-sending when the thread acknowledges with ChatEvent::Aborted.
    pub fn request_abort(&mut self) {
        if self.sending {
            self.abort.store(true, Ordering::Relaxed);
            self.status = "aborting stream...".into();
        }
    }

    #[cfg(test)]
    pub fn abort_requested(&self) -> bool {
        self.abort.load(Ordering::Relaxed)
    }
}
