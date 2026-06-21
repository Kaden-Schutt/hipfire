// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    sync::mpsc::{self, Receiver},
    thread,
};

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::Rect;

use crate::hipfire::{
    chat::{stream_chat, ChatEvent, ChatMessage},
    config::ConfigState,
    registry::{RegistryAction, RegistryState},
    status::{start_background_serve, stop_pids, StatusState},
    training::TrainingState,
    HipfirePaths,
};

/// Actions exposed by the Home → Control pane. Wired to keyboard hotkeys and
/// mouse clicks. Serve actions are real; the chat/admin endpoint toggles are
/// presentational placeholders until those services are separable (rendered in
/// bright blue to mark them as faked state).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlAction {
    StartServe,
    StopServe,
    RestartServe,
    ToggleChatEndpoint,
    ToggleAdminConsole,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    Home,
    Chat,
    Models,
    Runtime,
    Logs,
    Training,
    Settings,
    System,
}

impl Tab {
    pub const ALL: [Tab; 8] = [
        Tab::Home,
        Tab::Chat,
        Tab::Models,
        Tab::Runtime,
        Tab::Logs,
        Tab::Training,
        Tab::Settings,
        Tab::System,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Home => "Home",
            Tab::Chat => "Chat",
            Tab::Models => "Models",
            Tab::Runtime => "Runtime",
            Tab::Logs => "Logs",
            Tab::Training => "Training",
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
    pub training: TrainingState,
    pub active_model: String,
    pub tab: Tab,
    pub settings_easy: bool,
    pub settings_selected: usize,
    pub chat: ChatState,
    pub last_reload: String,
    /// Monotonic frame counter driving the title-bar spinner. Advances every
    /// event-loop tick so the spinner animates even when nothing else changes.
    pub tick: u64,
    /// Faked toggle state for the user chat endpoint (no separable backend yet).
    pub chat_endpoint_on: bool,
    /// Faked toggle state for the admin console endpoint.
    pub admin_console_on: bool,
    /// Clickable Control-pane button regions, captured during draw for mouse
    /// hit-testing.
    pub control_buttons: Vec<(Rect, ControlAction)>,
}

impl App {
    pub fn load() -> Result<Self> {
        let paths = HipfirePaths::discover();
        let config = ConfigState::load(&paths);
        let registry = RegistryState::load(&paths, &config);
        let status = StatusState::load(&paths, &config);
        let training = TrainingState::load(&paths, &config);
        let active_model = config.default_model.clone();
        Ok(Self {
            paths,
            config,
            registry,
            status,
            training,
            active_model,
            tab: Tab::Home,
            settings_easy: true,
            settings_selected: 0,
            chat: ChatState::default(),
            last_reload: "loaded hipfire state".into(),
            tick: 0,
            chat_endpoint_on: false,
            admin_console_on: false,
            control_buttons: Vec::new(),
        })
    }

    pub fn reload(&mut self) {
        self.config = ConfigState::load(&self.paths);
        self.registry = RegistryState::load(&self.paths, &self.config);
        self.status = StatusState::load(&self.paths, &self.config);
        self.training = TrainingState::load(&self.paths, &self.config);
        self.last_reload = "reloaded config, registry, models, and serve status".into();
    }

    pub fn next_tab(&mut self) {
        let idx = Tab::ALL.iter().position(|t| *t == self.tab).unwrap_or(0);
        self.tab = Tab::ALL[(idx + 1) % Tab::ALL.len()];
    }

    pub fn prev_tab(&mut self) {
        let idx = Tab::ALL.iter().position(|t| *t == self.tab).unwrap_or(0);
        self.tab = Tab::ALL[(idx + Tab::ALL.len() - 1) % Tab::ALL.len()];
    }

    pub fn handle_tab_key(&mut self, key: KeyEvent) {
        match self.tab {
            Tab::Home => self.handle_home_key(key),
            Tab::Chat => self.handle_chat_key(key),
            Tab::Models => self.handle_models_key(key),
            Tab::Training => self.handle_training_key(key),
            Tab::Settings => self.handle_settings_key(key),
            _ => {}
        }
    }

    fn handle_home_key(&mut self, key: KeyEvent) {
        let action = match key.code {
            KeyCode::Char('s') => Some(ControlAction::StartServe),
            KeyCode::Char('x') => Some(ControlAction::StopServe),
            KeyCode::Char('t') => Some(ControlAction::RestartServe),
            KeyCode::Char('c') => Some(ControlAction::ToggleChatEndpoint),
            KeyCode::Char('d') => Some(ControlAction::ToggleAdminConsole),
            _ => None,
        };
        if let Some(action) = action {
            self.exec_control(action);
        }
    }

    /// Run a Control-pane action. Serve start/stop/restart act on real PIDs; the
    /// chat/admin toggles only flip faked UI state.
    pub fn exec_control(&mut self, action: ControlAction) {
        match action {
            ControlAction::StartServe => {
                if self.status.serve_pid_alive || !self.status.serve_pids.is_empty() {
                    self.last_reload = "serve already running".into();
                } else {
                    match start_background_serve() {
                        Ok(()) => self.last_reload = "requested background serve start".into(),
                        Err(err) => self.last_reload = format!("{err}"),
                    }
                }
                self.status = StatusState::load(&self.paths, &self.config);
            }
            ControlAction::StopServe => {
                match stop_pids(&self.status.serve_pids) {
                    Ok(n) => self.last_reload = format!("sent SIGTERM to {n} serve process(es)"),
                    Err(err) => self.last_reload = format!("{err}"),
                }
                self.status = StatusState::load(&self.paths, &self.config);
            }
            ControlAction::RestartServe => {
                let _ = stop_pids(&self.status.serve_pids);
                std::thread::sleep(std::time::Duration::from_millis(300));
                match start_background_serve() {
                    Ok(()) => self.last_reload = "serve restart requested (stop + start)".into(),
                    Err(err) => self.last_reload = format!("restart: {err}"),
                }
                self.status = StatusState::load(&self.paths, &self.config);
            }
            ControlAction::ToggleChatEndpoint => {
                self.chat_endpoint_on = !self.chat_endpoint_on;
                self.last_reload = format!(
                    "[faked] chat endpoint {}",
                    if self.chat_endpoint_on {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }
            ControlAction::ToggleAdminConsole => {
                self.admin_console_on = !self.admin_console_on;
                self.last_reload = format!(
                    "[faked] admin console {}",
                    if self.admin_console_on {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }
        }
    }

    /// Dispatch a mouse click at terminal cell (col, row) to a Control button
    /// if it lands inside one of the regions captured during the last draw.
    pub fn handle_mouse_click(&mut self, col: u16, row: u16) {
        let hit = self
            .control_buttons
            .iter()
            .copied()
            .find(|(rect, _)| {
                col >= rect.x
                    && col < rect.x.saturating_add(rect.width)
                    && row >= rect.y
                    && row < rect.y.saturating_add(rect.height)
            })
            .map(|(_, action)| action);
        if let Some(action) = hit {
            self.exec_control(action);
        }
    }

    /// Current braille spinner glyph for the title bar, driven by `tick`.
    pub fn spinner_frame(&self) -> char {
        const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        FRAMES[(self.tick / 2) as usize % FRAMES.len()]
    }

    fn handle_training_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.training.select_delta(1, &self.paths, &self.config);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.training.select_delta(-1, &self.paths, &self.config);
            }
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

        match key.code {
            KeyCode::Enter => {
                let prompt = self.chat.input.trim().to_string();
                if prompt.is_empty() {
                    self.chat.focus_input();
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
                });
                self.chat.messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: String::new(),
                });
                self.chat.sending = true;
                self.chat.status = "streaming from hipfire serve".into();

                let (tx, rx) = mpsc::channel();
                self.chat.rx = Some(rx);
                let host = self.config.probe_host();
                let port = self.config.port;
                let model = self.active_model.clone();
                let mut messages = self.chat.messages.clone();
                if let Some(last) = messages.last_mut() {
                    if last.role == "assistant" && last.content.is_empty() {
                        messages.pop();
                    }
                }
                thread::spawn(move || {
                    let _ = stream_chat(&host, port, &model, &messages, tx);
                });
            }
            KeyCode::Backspace => {
                self.chat.input.pop();
                self.chat.focus_input();
            }
            KeyCode::Char(c) => {
                self.chat.input.push(c);
                self.chat.focus_input();
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
                "serve process exists; waiting for HTTP health, press r to refresh".into();
            return;
        }

        match start_background_serve() {
            Ok(()) => {
                self.chat.status =
                    "starting serve -d; keep your prompt and retry after health is online".into();
                self.last_reload = "requested background serve start".into();
                self.status = StatusState::load(&self.paths, &self.config);
            }
            Err(err) => {
                self.chat.status = format!("{err}");
            }
        }
    }

    fn handle_settings_key(&mut self, key: KeyEvent) {
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
            _ => {}
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
                    ChatEvent::Status(status) => self.chat.status = status,
                    ChatEvent::Done => {
                        self.chat.status = "ready".into();
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

pub struct ChatState {
    pub input: String,
    pub messages: Vec<ChatMessage>,
    pub status: String,
    pub sending: bool,
    pub scroll: u16,
    rx: Option<Receiver<ChatEvent>>,
    input_focused: bool,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            input: String::new(),
            messages: Vec::new(),
            status: "ready".into(),
            sending: false,
            scroll: 0,
            rx: None,
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
}
