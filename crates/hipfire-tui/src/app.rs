// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    sync::mpsc::{self, Receiver},
    thread,
};

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::hipfire::{
    chat::{stream_chat, ChatEvent, ChatMessage},
    config::ConfigState,
    registry::{RegistryAction, RegistryState},
    status::{start_background_serve, StatusState},
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
    pub chat: ChatState,
    pub last_reload: String,
}

impl App {
    pub fn load() -> Result<Self> {
        let paths = HipfirePaths::discover();
        let config = ConfigState::load(&paths);
        let registry = RegistryState::load(&paths);
        let status = StatusState::load(&paths, &config);
        let active_model = config.default_model.clone();
        Ok(Self {
            paths,
            config,
            registry,
            status,
            active_model,
            tab: Tab::Home,
            settings_easy: true,
            settings_selected: 0,
            chat: ChatState::default(),
            last_reload: "loaded hipfire state".into(),
        })
    }

    pub fn reload(&mut self) {
        self.config = ConfigState::load(&self.paths);
        self.registry = RegistryState::load(&self.paths);
        self.status = StatusState::load(&self.paths, &self.config);
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
