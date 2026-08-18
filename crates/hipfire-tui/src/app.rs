// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::hipfire::{
    chat::{stream_chat, ChatEvent, ChatMessage},
    chat_history::ChatHistory,
    config::ConfigState,
    dashboard::{Dashboard, DashboardWorker},
    doctor::{self, DoctorReport},
    log_tail::{LogSnapshot, LogTailer},
    model_actions::{self, PullEvent, RmOutcome},
    registry::{RegistryAction, RegistryState},
    serve_ctrl::{self, ServeAction, ServeOutcome},
    status::{start_background_serve, StatusState},
    ui_state::UiState,
    writer::{self, FieldKind},
    HipfirePaths,
};

/// In-progress numeric/string edit of a settings field (Enter to commit,
/// Esc to cancel). Enum fields use [`PendingEnum`] instead of this state.
pub struct EditState {
    pub key: String,
    pub buffer: String,
}

/// A previewed-but-uncommitted enum value (5b). Cycling Left/Right/Space stages
/// the next value here (dimmed in the UI) WITHOUT writing config.toml, so merely
/// browsing the options can't mutate config. Enter commits it; Esc or navigating
/// away discards it. Booleans don't use this — a binary flip is a decision, not
/// browsing, so they still toggle-on-key.
pub struct PendingEnum {
    pub key: String,
    pub value: String,
}

/// Severity of a transient toast, drives its color in the footer overlay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToastLevel {
    Info,
    Error,
}

/// A transient status line shown over the footer for a few seconds after an
/// action (a write failure, a serve-unreachable action, a save confirmation).
/// Expires on its own — the UI checks [`Toast::is_live`] each frame.
#[derive(Clone, Debug)]
pub struct Toast {
    pub text: String,
    pub level: ToastLevel,
    born: Instant,
    ttl: Duration,
}

impl Toast {
    pub fn is_live(&self) -> bool {
        self.born.elapsed() < self.ttl
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    Home,
    Dashboard,
    Chat,
    Models,
    Settings,
    System,
    Logs,
}

impl Tab {
    pub const ALL: [Tab; 7] = [
        Tab::Home,
        Tab::Dashboard,
        Tab::Chat,
        Tab::Models,
        Tab::Settings,
        Tab::System,
        Tab::Logs,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Home => "Home",
            Tab::Dashboard => "Dashboard",
            Tab::Chat => "Chat",
            Tab::Models => "Models",
            Tab::Settings => "Settings",
            Tab::System => "System",
            Tab::Logs => "Logs",
        }
    }

    /// Resolve a tab from its title (for restoring the persisted active tab).
    pub fn from_title(s: &str) -> Option<Tab> {
        Tab::ALL.into_iter().find(|t| t.title() == s)
    }
}

/// In-flight model download. `percent`/`line` are refreshed each frame from the
/// background pull worker's progress events.
pub struct PullJob {
    pub tag: String,
    rx: Receiver<PullEvent>,
    pub percent: Option<f64>,
    pub line: String,
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
    pub settings_edit: Option<EditState>,
    /// A staged-but-uncommitted enum value preview (5b). `Some` only while the
    /// user is cycling an enum and hasn't pressed Enter (commit) / Esc (cancel).
    pub settings_pending: Option<PendingEnum>,
    pub chat: ChatState,
    pub last_reload: String,
    /// Transient over-footer status toast (action failures + confirmations).
    /// `None` when nothing recent; the UI clears it once it expires.
    pub toast: Option<Toast>,
    /// Latest live-serve Dashboard snapshot mirrored from the background fetch
    /// thread each frame (None until the worker's first fetch completes). The
    /// UI thread ONLY reads this — it never calls fetch_dashboard / rocm-smi /
    /// HTTP synchronously, so a hung probe cannot block render or input.
    pub dashboard: Option<Dashboard>,
    /// Whether the `?` keybinding help overlay is currently open.
    pub show_help: bool,
    /// Per-frame tab-bar hit regions `[x_start, x_end)` -> tab, recomputed by
    /// the renderer each frame so a mouse-click column maps to a tab.
    pub tab_hitboxes: Vec<(u16, u16, Tab)>,
    /// Terminal row the tab labels render on (for click hit-testing).
    pub tab_row_y: u16,
    /// In-flight serve lifecycle command (start/stop/restart). Its single
    /// outcome arrives on this receiver; `drain_serve_command` consumes it each
    /// frame and turns it into a toast. `None` when no command is running.
    serve_cmd: Option<Receiver<ServeOutcome>>,
    /// Label ("start"/"stop"/"restart") of the in-flight serve command, for the
    /// status line; empty when idle.
    pub serve_cmd_label: String,
    /// In-flight model pull (Models tab). `None` when idle.
    pub pull: Option<PullJob>,
    /// In-flight `rm` command receiver (Models tab).
    rm_cmd: Option<Receiver<RmOutcome>>,
    /// Model tag awaiting a y/n delete confirmation; the Models tab shows a
    /// prompt while this is `Some`.
    pub confirm_delete: Option<String>,
    /// Settings tab: a "reset ALL config to defaults" y/n confirmation is armed.
    /// Single-key reset is reversible so it commits immediately; wiping the whole
    /// file is gated behind this prompt (mirrors the Models delete-confirm modal).
    pub confirm_reset_all: bool,
    /// Text staged by Chat `/copy` for the event loop to emit to the terminal
    /// clipboard (OSC52) after the next render; cleared once emitted.
    pub pending_clipboard: Option<String>,
    /// Recent finished chat requests (newest first), for the System-tab request
    /// inspector. Bounded to [`REQUEST_LOG_CAP`].
    pub request_log: Vec<RequestRecord>,
    /// In-flight `hipfire diag --json` doctor run (System tab `d`).
    doctor_cmd: Option<Receiver<DoctorReport>>,
    /// Latest doctor report (System tab); `None` until first run.
    pub doctor: Option<DoctorReport>,
    /// Latest serve.log tail (Logs tab), mirrored from the LogTailer worker.
    pub logs: LogSnapshot,
    /// Background serve.log tailer. Reads the file off the UI thread; dropped
    /// (joined) when the App is dropped.
    log_tailer: LogTailer,
    /// Background fetch thread. Owns the network + rocm-smi I/O off the UI
    /// thread. Dropped (joined) when the App is dropped.
    dashboard_worker: DashboardWorker,
}

impl App {
    pub fn load() -> Result<Self> {
        let paths = HipfirePaths::discover();
        let config = ConfigState::load(&paths);
        let mut registry = RegistryState::load(&paths);
        let status = StatusState::load_local(&paths);
        let active_model = config.default_model.clone();
        let dashboard_worker = DashboardWorker::spawn(config.clone());
        let log_tailer = LogTailer::spawn(paths.serve_log.clone());

        // Restore session UI state (active tab + expanded Models groups) so the
        // TUI reopens where the user left off. Missing/corrupt -> defaults. The
        // active *model* is restored separately via config.default_model, which
        // the Models tab already persists.
        let ui = UiState::load(&paths.ui_state);
        let tab = Tab::from_title(&ui.active_tab).unwrap_or(Tab::Home);
        registry.expanded_groups = ui.expanded_groups.into_iter().collect();

        Ok(Self {
            paths,
            config,
            registry,
            status,
            active_model,
            tab,
            settings_easy: true,
            settings_selected: 0,
            settings_edit: None,
            settings_pending: None,
            chat: ChatState::default(),
            last_reload: "loaded hipfire state".into(),
            toast: None,
            dashboard: None,
            show_help: false,
            tab_hitboxes: Vec::new(),
            tab_row_y: 0,
            serve_cmd: None,
            serve_cmd_label: String::new(),
            pull: None,
            rm_cmd: None,
            confirm_delete: None,
            confirm_reset_all: false,
            pending_clipboard: None,
            request_log: Vec::new(),
            doctor_cmd: None,
            doctor: None,
            logs: LogSnapshot::default(),
            log_tailer,
            dashboard_worker,
        })
    }

    /// Mirror the latest serve.log tail from the background tailer, and tell it
    /// whether the Logs tab is focused (so it only reads while visible). Called
    /// each frame — a cheap lock+clone, never a synchronous file read on the UI
    /// thread.
    pub fn sync_logs(&mut self) {
        self.log_tailer.set_active(self.tab == Tab::Logs);
        if self.tab == Tab::Logs {
            self.logs = self.log_tailer.snapshot();
        }
    }

    /// Start downloading `tag` on a background worker (Models tab `p`). No-op
    /// (with a toast) if a pull is already running or the model is already local.
    fn start_pull(&mut self, tag: String) {
        if self.pull.is_some() {
            self.toast_info("a pull is already running");
            return;
        }
        let rx = model_actions::pull(tag.clone());
        self.toast_info(format!("pulling {tag}\u{2026}"));
        self.pull = Some(PullJob {
            tag,
            rx,
            percent: None,
            line: String::new(),
        });
    }

    /// Consume pull progress/result events (called each frame). On completion it
    /// reloads the registry so the new local model appears.
    pub fn drain_pull(&mut self) {
        let mut finished: Option<Result<String, String>> = None;
        if let Some(job) = &mut self.pull {
            loop {
                match job.rx.try_recv() {
                    Ok(PullEvent::Progress { percent, line }) => {
                        job.percent = percent;
                        job.line = line;
                    }
                    Ok(PullEvent::Done) => {
                        finished = Some(Ok(job.tag.clone()));
                        break;
                    }
                    Ok(PullEvent::Failed(err)) => {
                        finished = Some(Err(format!("pull {}: {err}", job.tag)));
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        finished = Some(Err(format!("pull {}: worker exited", job.tag)));
                        break;
                    }
                }
            }
        }
        if let Some(result) = finished {
            self.pull = None;
            match result {
                Ok(tag) => {
                    self.toast_info(format!("pulled {tag}"));
                    self.reload(); // surface the new local model
                }
                Err(msg) => self.toast_error(msg),
            }
        }
    }

    /// Request deletion of the selected local model — arms the y/n confirm.
    fn request_delete(&mut self) {
        match self.registry.selected_model() {
            Some((tag, true)) => self.confirm_delete = Some(tag),
            Some((_, false)) => self.toast_info("that model isn't downloaded"),
            None => self.toast_info("select a model row to delete"),
        }
    }

    /// Confirm the armed delete: spawn `rm <tag> --yes` and disarm.
    fn confirm_delete_yes(&mut self) {
        if let Some(tag) = self.confirm_delete.take() {
            if self.rm_cmd.is_some() {
                self.toast_info("a delete is already running");
                return;
            }
            self.rm_cmd = Some(model_actions::remove(tag.clone()));
            self.toast_info(format!("deleting {tag}\u{2026}"));
        }
    }

    /// Cancel the armed delete.
    fn cancel_delete(&mut self) {
        if self.confirm_delete.take().is_some() {
            self.toast_info("delete cancelled");
        }
    }

    /// Consume the rm outcome if it has arrived (called each frame). Reloads the
    /// registry on success so the removed model drops out of the list.
    pub fn drain_rm(&mut self) {
        let outcome = match &self.rm_cmd {
            Some(rx) => match rx.try_recv() {
                Ok(o) => o,
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => {
                    RmOutcome::Failed("rm worker exited".into())
                }
            },
            None => return,
        };
        self.rm_cmd = None;
        match outcome {
            RmOutcome::Ok(msg) => {
                self.toast_info(msg);
                self.reload();
            }
            RmOutcome::Failed(msg) => self.toast_error(msg),
        }
    }

    /// True while a serve lifecycle command (start/stop/restart) is running.
    pub fn serve_cmd_running(&self) -> bool {
        self.serve_cmd.is_some()
    }

    /// Kick off a serve lifecycle command on a background thread. No-op (with a
    /// toast) if one is already in flight. The live serve up/down state updates
    /// separately via the DashboardWorker once the daemon responds.
    fn start_serve_command(&mut self, action: ServeAction) {
        if self.serve_cmd.is_some() {
            self.toast_info("a serve command is already running");
            return;
        }
        self.serve_cmd = Some(serve_ctrl::run(action));
        self.serve_cmd_label = action.label().to_string();
        self.toast_info(format!("serve {}\u{2026}", action.label()));
    }

    /// Consume the serve command outcome if it has arrived (called each frame).
    /// Surfaces a toast and kicks an immediate dashboard re-probe so the new
    /// serve state shows up promptly.
    pub fn drain_serve_command(&mut self) {
        let outcome = match &self.serve_cmd {
            Some(rx) => match rx.try_recv() {
                Ok(o) => Some(o),
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => Some(ServeOutcome::Failed(
                    "serve command thread exited unexpectedly".into(),
                )),
            },
            None => return,
        };
        self.serve_cmd = None;
        self.serve_cmd_label.clear();
        match outcome {
            Some(ServeOutcome::Ok(msg)) => self.toast_info(msg),
            Some(ServeOutcome::Failed(msg)) => self.toast_error(msg),
            None => {}
        }
        self.dashboard_worker.force_refresh();
    }

    /// Dashboard-tab keys: serve lifecycle controls.
    fn handle_dashboard_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('s') => self.start_serve_command(ServeAction::Start),
            KeyCode::Char('x') => self.start_serve_command(ServeAction::Stop),
            KeyCode::Char('R') => self.start_serve_command(ServeAction::Restart),
            _ => {}
        }
    }

    /// Test seam: inject an in-flight serve-command receiver without spawning a
    /// real `bun` process, so `drain_serve_command` can be exercised in unit tests.
    #[cfg(test)]
    pub fn inject_serve_cmd(&mut self, rx: Receiver<ServeOutcome>) {
        self.serve_cmd = Some(rx);
        self.serve_cmd_label = "test".into();
    }

    /// Test seam: inject an in-flight pull without spawning a real process.
    #[cfg(test)]
    pub fn inject_pull(&mut self, tag: String, rx: Receiver<PullEvent>) {
        self.pull = Some(PullJob {
            tag,
            rx,
            percent: None,
            line: String::new(),
        });
    }

    /// Persist the current session UI state (active tab + expanded Models
    /// groups) for the next launch. Best-effort — a failed write is ignored
    /// (UI state is a convenience, not load-bearing).
    pub fn save_ui_state(&self) {
        let ui = UiState {
            active_tab: self.tab.title().to_string(),
            expanded_groups: self.registry.expanded_groups.iter().cloned().collect(),
        };
        let _ = ui.save(&self.paths.ui_state);
    }

    /// Raise a transient error toast (3.5s) over the footer.
    pub fn toast_error(&mut self, text: impl Into<String>) {
        self.toast = Some(Toast {
            text: text.into(),
            level: ToastLevel::Error,
            born: Instant::now(),
            ttl: Duration::from_millis(3500),
        });
    }

    /// Raise a transient info/confirmation toast (2.5s) over the footer.
    pub fn toast_info(&mut self, text: impl Into<String>) {
        self.toast = Some(Toast {
            text: text.into(),
            level: ToastLevel::Info,
            born: Instant::now(),
            ttl: Duration::from_millis(2500),
        });
    }

    /// Drop the toast once it has expired. Called each frame before render.
    pub fn expire_toast(&mut self) {
        if let Some(t) = &self.toast {
            if !t.is_live() {
                self.toast = None;
            }
        }
    }

    /// Mirror the latest snapshot produced by the background fetch thread into
    /// `self.dashboard`. Called every render frame from the UI thread — this is
    /// a cheap lock + clone and performs NO network / rocm-smi I/O. It also
    /// tells the worker whether the Dashboard tab is currently focused so the
    /// worker only fetches while the tab is visible.
    pub fn sync_dashboard(&mut self) {
        // The worker feeds both the Dashboard tab (serve/VRAM telemetry) and
        // the System tab (live GPU/HIP/kernel-cache/loaded-model diagnostics),
        // so it must run while either is focused.
        self.dashboard_worker
            .set_active(self.tab == Tab::Dashboard || self.tab == Tab::System);
        if let Some(snap) = self.dashboard_worker.snapshot() {
            // Fold the worker's off-thread /health + rocm-smi results into the
            // live status fields (serve_http_ok / health_text / gpu_lines) so the
            // Home + System tabs render live data without any synchronous probe.
            self.status.overlay_live(&snap);
            self.dashboard = Some(snap);
        }
    }

    pub fn reload(&mut self) {
        // Reload ONLY the fast local data synchronously (config / registry /
        // local-models / serve.pid — all file I/O). The live serve health, VRAM
        // and system diagnostics come from the background DashboardWorker, so we
        // never run /health or lspci/rocm-smi on the UI thread here.
        self.config = ConfigState::load(&self.paths);
        // A reload re-reads config from disk, so drop any uncommitted enum preview
        // (5b) rather than leave it pointing at a now-stale value.
        self.settings_pending = None;
        // Preserve the user's expanded Models groups across an `r` refresh — a
        // fresh RegistryState::load() defaults them to empty, which would
        // collapse everything (and then a clean quit would persist that empty
        // set back to ui_state.json).
        let expanded = self.registry.expanded_groups.clone();
        self.registry = RegistryState::load(&self.paths);
        self.registry.expanded_groups = expanded;
        // Preserve the live overlay (serve_http_ok / health_text / gpu_lines)
        // already mirrored from the worker; only the local fields are refreshed.
        let mut status = StatusState::load_local(&self.paths);
        if let Some(snap) = self.dashboard.as_ref() {
            status.overlay_live(snap);
        }
        self.status = status;
        // Keep the worker pointed at the (possibly changed) host/port and kick
        // an immediate non-blocking re-fetch; the snapshot updates on a later frame.
        self.dashboard_worker.update_config(self.config.clone());
        self.dashboard_worker.force_refresh();
        self.last_reload = "reloaded config, registry, models; refreshing serve status".into();
    }

    pub fn next_tab(&mut self) {
        // Leaving Settings discards an uncommitted enum preview (5b).
        self.settings_pending = None;
        let idx = Tab::ALL.iter().position(|t| *t == self.tab).unwrap_or(0);
        self.tab = Tab::ALL[(idx + 1) % Tab::ALL.len()];
    }

    pub fn prev_tab(&mut self) {
        self.settings_pending = None;
        let idx = Tab::ALL.iter().position(|t| *t == self.tab).unwrap_or(0);
        self.tab = Tab::ALL[(idx + Tab::ALL.len() - 1) % Tab::ALL.len()];
    }

    /// Resolve the tab whose recorded header hit region contains `(col, row)`,
    /// or None when the position is not on the tab bar. The hit regions are
    /// refreshed every frame by the renderer (`ui::draw_header`).
    pub fn tab_at(&self, col: u16, row: u16) -> Option<Tab> {
        if row != self.tab_row_y {
            return None;
        }
        self.tab_hitboxes
            .iter()
            .find(|(start, end, _)| col >= *start && col < *end)
            .map(|(_, _, tab)| *tab)
    }

    pub fn handle_tab_key(&mut self, key: KeyEvent) {
        match self.tab {
            Tab::Dashboard => self.handle_dashboard_key(key),
            Tab::Chat => self.handle_chat_key(key),
            Tab::Models => self.handle_models_key(key),
            Tab::Settings => self.handle_settings_key(key),
            Tab::System => self.handle_system_key(key),
            _ => {}
        }
    }

    fn handle_models_key(&mut self, key: KeyEvent) {
        // A delete confirmation is modal: intercept y / n / Esc.
        if self.confirm_delete.is_some() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => self.confirm_delete_yes(),
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => self.cancel_delete(),
                _ => {}
            }
            return;
        }
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
                            // Persist default_model to ~/.hipfire/config.toml
                            // (read-modify-write, atomic). Was session-only.
                            match writer::write_raw_value(
                                &self.paths.config,
                                "default_model",
                                serde_json::Value::String(tag.clone()),
                            ) {
                                Ok(()) => {
                                    self.config.default_model = tag.clone();
                                    self.config
                                        .values
                                        .insert("default_model".into(), tag.clone());
                                    // Now explicitly on disk -> resettable override.
                                    self.config.overrides.insert("default_model".into());
                                    self.config.loaded_from_disk = true;
                                    self.last_reload = format!(
                                        "default_model = {tag} saved to ~/.hipfire/config.toml"
                                    );
                                    self.toast_info(format!("default model → {tag}"));
                                }
                                Err(err) => {
                                    self.last_reload = format!("default_model save failed: {err}");
                                    self.toast_error(format!("default_model save failed: {err}"));
                                }
                            }
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
            KeyCode::Char('p') => match self.registry.selected_model() {
                Some((tag, false)) => self.start_pull(tag),
                Some((_, true)) => self.toast_info("already downloaded"),
                None => self.toast_info("select a model row to pull"),
            },
            KeyCode::Char('d') => self.request_delete(),
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
                let input = self.chat.input.trim().to_string();
                if input.is_empty() {
                    self.chat.focus_input();
                    return;
                }
                // A leading '/' is a command — except `//…`, which escapes to a
                // literal message whose text starts with a single '/'.
                if let Some(rest) = input.strip_prefix('/') {
                    if let Some(literal) = rest.strip_prefix('/') {
                        if !self.status.serve_http_ok {
                            self.start_serve_for_chat();
                            return;
                        }
                        self.send_chat(format!("/{literal}"));
                        self.chat.focus_input();
                        return;
                    }
                    // Commands act locally (no serve needed) and never send.
                    self.chat.input.clear();
                    self.handle_chat_command(rest);
                    self.chat.focus_input();
                    return;
                }
                if !self.status.serve_http_ok {
                    self.start_serve_for_chat();
                    return;
                }
                self.send_chat(input);
                self.chat.focus_input();
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

    /// Push the user prompt, then spawn the stream worker (system prompt +
    /// sampling applied in spawn_stream).
    fn send_chat(&mut self, prompt: String) {
        self.chat.input.clear();
        self.chat.messages.push(ChatMessage {
            role: "user".into(),
            content: prompt,
            reasoning_content: None,
        });
        self.spawn_stream();
    }

    /// Push an empty assistant slot and spawn the stream worker for the current
    /// `messages` tail. Shared by send (after pushing the user turn) and
    /// regenerate (the user turn is already last). Applies the session system
    /// prompt + sampling overrides and resets per-generation stats.
    fn spawn_stream(&mut self) {
        self.chat.messages.push(ChatMessage {
            role: "assistant".into(),
            content: String::new(),
            reasoning_content: None,
        });
        self.chat.sending = true;
        self.chat.status = "streaming from hipfire serve".into();
        self.chat.gen_start = Some(Instant::now());
        self.chat.gen_model = self.active_model.clone();
        self.chat.gen_tokens = 0;
        self.chat.last_stats = None;

        let (tx, rx) = mpsc::channel();
        self.chat.rx = Some(rx);
        // Fresh cancel flag per generation; Esc flips it (request_abort).
        self.chat.abort = Arc::new(AtomicBool::new(false));
        let abort = self.chat.abort.clone();
        let host = self.config.probe_host();
        let port = self.config.port;
        let model = self.active_model.clone();
        let temp = self.chat.temp;
        let top_p = self.chat.top_p;
        let mut messages = self.chat.messages.clone();
        if let Some(last) = messages.last_mut() {
            let reasoning_empty = last
                .reasoning_content
                .as_deref()
                .map_or(true, |s| s.is_empty());
            if last.role == "assistant" && last.content.is_empty() && reasoning_empty {
                messages.pop();
            }
        }
        if !self.chat.system_prompt.is_empty() {
            messages.insert(
                0,
                ChatMessage {
                    role: "system".into(),
                    content: self.chat.system_prompt.clone(),
                    reasoning_content: None,
                },
            );
        }
        thread::spawn(move || {
            let _ = stream_chat(&host, port, &model, &messages, temp, top_p, tx, abort);
        });
    }

    /// Whether the conversation tail is `[.., user]` or `[.., user, assistant]`
    /// — the only shapes /regen and /edit can act on. Checked BEFORE any mutation
    /// so a malformed or freshly-loaded history is never corrupted.
    fn tail_is_redoable(&self) -> bool {
        let msgs = &self.chat.messages;
        let n = msgs.len();
        match msgs.last().map(|m| m.role.as_str()) {
            Some("assistant") => n >= 2 && msgs[n - 2].role == "user",
            Some("user") => true,
            _ => false,
        }
    }

    /// `/regen` — drop the last assistant reply and re-stream from the last user
    /// turn (no new user message).
    fn regenerate(&mut self) {
        if self.chat.sending {
            self.toast_info("generation in progress");
            return;
        }
        if !self.tail_is_redoable() {
            self.toast_error("nothing to regenerate");
            return;
        }
        // Check serve readiness BEFORE the destructive pop, so an offline serve
        // doesn't lose the existing reply.
        if !self.status.serve_http_ok {
            self.start_serve_for_chat();
            return;
        }
        if self.chat.messages.last().map(|m| m.role.as_str()) == Some("assistant") {
            self.chat.messages.pop();
        }
        self.spawn_stream();
    }

    /// `/edit` — pull the last user turn back into the input (dropping its reply)
    /// so it can be edited and re-sent.
    fn edit_last(&mut self) {
        if self.chat.sending {
            self.toast_info("generation in progress");
            return;
        }
        if !self.tail_is_redoable() {
            self.toast_error("no message to edit");
            return;
        }
        if self.chat.messages.last().map(|m| m.role.as_str()) == Some("assistant") {
            self.chat.messages.pop();
        }
        let last = self
            .chat
            .messages
            .pop()
            .expect("tail validated as user turn");
        self.chat.input = last.content;
        self.chat.reset_stats();
        self.chat.focus_input();
        self.chat.status = "editing last message — Enter to resend".into();
    }

    /// `/copy` — stage the last non-empty assistant reply for the OSC52 clipboard
    /// emit (performed by the event loop after render).
    fn copy_last(&mut self) {
        match self
            .chat
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "assistant" && !m.content.is_empty())
        {
            Some(m) => {
                // Cap the OSC52 payload — a very large escape can stall or be
                // rejected by the terminal.
                const MAX_COPY: usize = 100 * 1024;
                if m.content.len() > MAX_COPY {
                    self.toast_error(format!(
                        "reply too large to copy ({} KB > 100 KB)",
                        m.content.len() / 1024
                    ));
                } else {
                    self.pending_clipboard = Some(m.content.clone());
                    self.chat.status = "copied last reply to clipboard".into();
                }
            }
            None => self.toast_error("no reply to copy"),
        }
    }

    /// Record a finished request in the inspector ring (newest first, capped).
    fn log_request(&mut self, stats: GenStats, secs: f64) {
        self.request_log.insert(
            0,
            RequestRecord {
                model: self.chat.gen_model.clone(),
                tokens: stats.tokens,
                tps: stats.tps,
                secs,
            },
        );
        self.request_log.truncate(REQUEST_LOG_CAP);
    }

    /// Whether a doctor run is in flight (for the footer + re-trigger guard).
    pub fn doctor_running(&self) -> bool {
        self.doctor_cmd.is_some()
    }

    /// Start a `hipfire diag --json` doctor run on a background thread (System
    /// tab `d`). No-op if one is already running.
    fn run_doctor(&mut self) {
        if self.doctor_cmd.is_some() {
            self.toast_info("doctor already running");
            return;
        }
        self.doctor_cmd = Some(doctor::run());
        self.toast_info("running hipfire diag\u{2026}");
    }

    /// Consume the doctor report when it arrives (called each frame).
    pub fn drain_doctor(&mut self) {
        let report = match &self.doctor_cmd {
            Some(rx) => match rx.try_recv() {
                Ok(r) => r,
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => DoctorReport {
                    checks: Vec::new(),
                    error: Some("doctor worker exited".into()),
                },
            },
            None => return,
        };
        self.doctor_cmd = None;
        match &report.error {
            Some(err) => self.toast_error(format!("doctor: {err}")),
            None => {
                let fails = report.checks.iter().filter(|c| !c.ok).count();
                if fails == 0 {
                    self.toast_info("doctor: all checks passed");
                } else {
                    self.toast_error(format!("doctor: {fails} check(s) failed"));
                }
            }
        }
        self.doctor = Some(report);
    }

    /// System-tab keys: `d` runs the doctor.
    fn handle_system_key(&mut self, key: KeyEvent) {
        if let KeyCode::Char('d') = key.code {
            self.run_doctor();
        }
    }

    /// Dispatch a Chat slash command (the input after the leading '/').
    fn handle_chat_command(&mut self, cmd: &str) {
        let mut parts = cmd.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();
        match name {
            "model" | "m" => {
                if arg.is_empty() {
                    self.chat.status = format!("model: {} (use /model <tag>)", self.active_model);
                } else if self.registry.has_model(arg) {
                    self.active_model = arg.to_string();
                    self.chat.status = format!("model -> {arg}");
                } else {
                    self.toast_error(format!("unknown model: {arg}"));
                }
            }
            "system" | "sys" => {
                self.chat.system_prompt = arg.to_string();
                self.chat.status = if arg.is_empty() {
                    "system prompt cleared".into()
                } else {
                    format!("system prompt set ({} chars)", arg.chars().count())
                };
            }
            "temp" => self.set_sampling(true, arg),
            "top_p" => self.set_sampling(false, arg),
            "clear" => {
                self.chat.messages.clear();
                self.chat.scroll = 0;
                self.chat.reset_stats();
                self.chat.status = "conversation cleared".into();
            }
            "save" => self.chat_save(arg),
            "load" => self.chat_load(arg),
            "sessions" | "ls" => self.chat_sessions(),
            "delete" | "del" => self.chat_delete(arg),
            "regen" | "retry" => self.regenerate(),
            "edit" => self.edit_last(),
            "copy" => self.copy_last(),
            "help" | "" => {
                self.chat.status = "/model · /system · /temp · /top_p · /clear · /save · /load · /sessions · /delete · /regen · /edit · /copy".into();
            }
            other => self.toast_error(format!("unknown command: /{other} (try /help)")),
        }
    }

    /// `/save <name>` — persist the current conversation as a named session.
    fn chat_save(&mut self, name: &str) {
        if name.is_empty() {
            self.toast_error("usage: /save <name>");
            return;
        }
        if self.chat.messages.is_empty() {
            self.toast_info("nothing to save (empty conversation)");
            return;
        }
        let mut h = ChatHistory::load(&self.paths.chat_history);
        h.upsert(name, self.chat.messages.clone());
        match h.save(&self.paths.chat_history) {
            Ok(()) => self.chat.status = format!("saved session '{name}'"),
            Err(e) => self.toast_error(format!("save failed: {e}")),
        }
    }

    /// `/load <name>` — replace the conversation with a saved session.
    fn chat_load(&mut self, name: &str) {
        if name.is_empty() {
            self.toast_error("usage: /load <name> (see /sessions)");
            return;
        }
        let h = ChatHistory::load(&self.paths.chat_history);
        match h.get(name) {
            Some(msgs) => {
                self.chat.messages = msgs.to_vec();
                self.chat.scroll = 0;
                self.chat.reset_stats();
                self.chat.status =
                    format!("loaded '{name}' ({} messages)", self.chat.messages.len());
            }
            None => self.toast_error(format!("no session '{name}' (see /sessions)")),
        }
    }

    /// `/sessions` — list saved session names.
    fn chat_sessions(&mut self) {
        let h = ChatHistory::load(&self.paths.chat_history);
        let names = h.names();
        self.chat.status = if names.is_empty() {
            "no saved sessions (use /save <name>)".into()
        } else {
            format!("sessions: {}", names.join(", "))
        };
    }

    /// `/delete <name>` — remove a saved session.
    fn chat_delete(&mut self, name: &str) {
        if name.is_empty() {
            self.toast_error("usage: /delete <name>");
            return;
        }
        let mut h = ChatHistory::load(&self.paths.chat_history);
        if h.remove(name) {
            match h.save(&self.paths.chat_history) {
                Ok(()) => self.chat.status = format!("deleted session '{name}'"),
                Err(e) => self.toast_error(format!("delete failed: {e}")),
            }
        } else {
            self.toast_error(format!("no session '{name}'"));
        }
    }

    /// Set or clear a per-session sampling override. `is_temp` selects
    /// temperature (range 0..=2) vs top_p (range 0..=1). An empty/"off" arg
    /// clears the override (serve default).
    fn set_sampling(&mut self, is_temp: bool, arg: &str) {
        let (label, lo, hi) = if is_temp {
            ("temperature", 0.0, 2.0)
        } else {
            ("top_p", 0.0, 1.0)
        };
        if arg.is_empty() || arg == "off" {
            if is_temp {
                self.chat.temp = None;
            } else {
                self.chat.top_p = None;
            }
            self.chat.status = format!("{label}: serve default");
            return;
        }
        match arg.parse::<f64>() {
            Ok(v) if (lo..=hi).contains(&v) => {
                if is_temp {
                    self.chat.temp = Some(v);
                } else {
                    self.chat.top_p = Some(v);
                }
                self.chat.status = format!("{label}: {v}");
            }
            _ => self.toast_error(format!("{label} must be a number in {lo}..={hi}")),
        }
    }

    fn start_serve_for_chat(&mut self) {
        if self.status.serve_pid_alive {
            self.chat.status =
                "serve process exists; waiting for HTTP health, press r to refresh".into();
            self.toast_info("serve offline — process exists, waiting for HTTP; press r to refresh");
            return;
        }

        match start_background_serve() {
            Ok(()) => {
                self.chat.status =
                    "starting serve -d; keep your prompt and retry after health is online".into();
                self.last_reload = "requested background serve start".into();
                self.toast_info("serve offline — starting `hipfire serve -d`; retry when online");
                // Refresh only the fast local serve.pid status (the just-spawned
                // pid). Live HTTP health arrives via the worker's next fetch — no
                // synchronous /health probe on the UI thread here.
                let mut status = StatusState::load_local(&self.paths);
                if let Some(snap) = self.dashboard.as_ref() {
                    status.overlay_live(snap);
                }
                self.status = status;
                self.dashboard_worker.force_refresh();
            }
            Err(err) => {
                self.chat.status = format!("{err}");
                self.toast_error(format!("serve start failed: {err}"));
            }
        }
    }

    /// Number of selectable rows in the current Settings mode.
    fn settings_row_count(&self) -> usize {
        if self.settings_easy {
            self.config.easy_rows().len()
        } else {
            self.config.values.len()
        }
    }

    /// Switch the Settings tab between easy and advanced mode, clamping
    /// `settings_selected` into the NEW mode's visible row range so we never
    /// render an empty table with no editable row (the bug: from a high advanced
    /// row index, pressing `e` left the selection past the end of the short easy
    /// table). When the currently-selected key also exists in the target mode,
    /// the selection follows that key so the cursor stays on the same setting.
    pub fn set_settings_easy(&mut self, easy: bool) {
        if self.settings_easy == easy {
            return;
        }
        // Remember the key under the cursor BEFORE flipping mode.
        let prev_key = self.selected_setting_key();
        self.settings_easy = easy;
        // Cancel any in-progress edit / enum preview — their key may not exist in
        // the new mode (5b: switching views discards an uncommitted preview).
        self.settings_edit = None;
        self.settings_pending = None;

        // Try to keep the cursor on the same key in the new mode.
        if let Some(key) = prev_key {
            if let Some(idx) = self.settings_key_index(&key) {
                self.settings_selected = idx;
                return;
            }
        }
        // Otherwise clamp into the new mode's row range (empty → 0).
        let max_idx = self.settings_row_count().saturating_sub(1);
        if self.settings_selected > max_idx {
            self.settings_selected = max_idx;
        }
    }

    /// Find the row index for `key` in the current Settings mode, if it maps to
    /// a row there. Used to preserve the selection across an easy/advanced flip.
    fn settings_key_index(&self, key: &str) -> Option<usize> {
        if self.settings_easy {
            self.config
                .easy_keys()
                .iter()
                .position(|k| matches!(k, Some(s) if *s == key))
        } else {
            self.config.values.keys().position(|k| k == key)
        }
    }

    /// Resolve the currently-selected settings row to a config key, if any.
    /// In easy mode this maps through `ConfigState::easy_keys`; in advanced
    /// mode the row IS a `(key, value)` pair from the raw config map.
    pub(crate) fn selected_setting_key(&self) -> Option<String> {
        if self.settings_easy {
            self.config
                .easy_keys()
                .get(self.settings_selected)
                .and_then(|k| k.map(|s| s.to_string()))
        } else {
            self.config
                .values
                .iter()
                .nth(self.settings_selected)
                .map(|(k, _)| k.clone())
        }
    }

    fn current_setting_value(&self, key: &str) -> String {
        self.config.values.get(key).cloned().unwrap_or_default()
    }

    /// Persist a validated value for `key` and reflect it in the in-memory
    /// config so the UI updates immediately. Returns `(ok, status)` — `ok` is
    /// false on a validation/write error so callers (the edit-mode Enter
    /// handler) can KEEP the edit buffer for in-place correction.
    fn persist_setting(&mut self, key: &str, raw: &str) -> (bool, String) {
        match writer::write_value(&self.paths.config, key, raw) {
            Ok(value) => {
                let as_str = match &value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                if key == "default_model" {
                    self.config.default_model = as_str.clone();
                }
                self.config.values.insert(key.to_string(), as_str.clone());
                // The key is now explicitly on disk -> an override, so a
                // following Delete/Backspace can reset it this session (without
                // it, is_override() stays false until the next full reload and
                // reset wrongly reports "already at its default").
                self.config.overrides.insert(key.to_string());
                self.config.loaded_from_disk = true;
                // Honest pflash state: turning compression on without a drafter is
                // a no-op until a prefill_drafter (.hfq) is set.
                if key == "prefill_compression" && self.config.pflash_needs_drafter() {
                    self.toast_error(
                        "prefill_compression on, but prefill_drafter is unset — no-op until set",
                    );
                    return (
                        true,
                        format!(
                            "{key} = {as_str} saved; set prefill_drafter (a .hfq path) or it stays off"
                        ),
                    );
                }
                self.toast_info(format!("{key} saved"));
                (
                    true,
                    format!("{key} = {as_str} saved to ~/.hipfire/config.toml"),
                )
            }
            Err(err) => {
                // Surface config-write failures as a loud transient toast in
                // addition to the persistent footer line.
                self.toast_error(format!("save failed: {err}"));
                (false, format!("{err}"))
            }
        }
    }

    /// Reload only the config from disk after a reset. Keeps the cursor on the
    /// same KEY (advanced-mode row indices shift when disk-only overrides vanish),
    /// re-points the background dashboard worker (a reset can change host/port),
    /// and clamps as a fallback. Registry/local-models are untouched — a config
    /// reset can't change them.
    fn reload_config_after_reset(&mut self) {
        let prev_key = self.selected_setting_key();
        self.config = ConfigState::load(&self.paths);
        // Follow the previously-selected key to its new row, else clamp in range.
        let restored = prev_key.and_then(|k| self.settings_key_index(&k));
        if let Some(idx) = restored {
            self.settings_selected = idx;
        } else {
            let max_idx = self.settings_row_count().saturating_sub(1);
            if self.settings_selected > max_idx {
                self.settings_selected = max_idx;
            }
        }
        // host/port may have changed (reset-all clears them, advanced reset can
        // delete a host/port override): keep the worker on the live endpoint.
        self.dashboard_worker.update_config(self.config.clone());
        self.dashboard_worker.force_refresh();
    }

    /// Reset the currently-selected setting to its inherited/default value by
    /// deleting its key from config.toml. A no-op (with an explanatory toast) for
    /// rows that are already at their default or that have no resettable key
    /// (the composite Model / Serve easy rows). Single-key reset is reversible —
    /// just re-set the value — so it commits immediately without a confirm.
    fn reset_selected_setting(&mut self) {
        // A corrupt config.toml loads as all-defaults with an empty
        // override set, so per-key reset would wrongly say "already default" (and
        // the writer refuses to clobber it anyway). Point the user at reset-all,
        // which is the recovery hatch.
        if let Some(warn) = self.config.warning.clone() {
            self.last_reload = format!("config.toml unreadable ({warn}); press R to reset all");
            self.toast_error("config.toml unreadable — use R (reset-all) to recover");
            return;
        }
        let Some(k) = self.selected_setting_key() else {
            self.last_reload = "this row has no resettable setting".into();
            self.toast_info("nothing to reset on this row");
            return;
        };
        if !self.config.is_override(&k) {
            let cur = self.current_setting_value(&k);
            self.last_reload = format!("{k} is already at its default ({cur})");
            self.toast_info(format!("{k} already default"));
            return;
        }
        let was = self.current_setting_value(&k);
        match writer::delete_key(&self.paths.config, &k) {
            Ok(_) => {
                self.reload_config_after_reset();
                let now = self.current_setting_value(&k);
                self.last_reload = format!("{k} reset to default ({now}); was {was}");
                self.toast_info(format!("{k} reset to default"));
            }
            Err(err) => {
                self.last_reload = format!("{k} reset failed: {err}");
                self.toast_error(format!("reset failed: {err}"));
            }
        }
    }

    /// Arm the "reset ALL config to defaults" confirmation. Wiping the whole file
    /// is destructive (it also clears default_model / host / port), so it waits
    /// for an explicit y before [`confirm_reset_all_yes`] runs.
    fn arm_reset_all(&mut self) {
        self.confirm_reset_all = true;
        self.last_reload = "reset ALL settings to defaults? y to confirm, n / Esc to cancel".into();
    }

    fn cancel_reset_all(&mut self) {
        if self.confirm_reset_all {
            self.confirm_reset_all = false;
            self.last_reload = "reset-all cancelled".into();
            self.toast_info("reset-all cancelled");
        }
    }

    /// Commit the confirmed reset-all: overwrite config.toml with only its schema
    /// falls back to inherited/default), reload, and re-clamp the cursor.
    fn confirm_reset_all_yes(&mut self) {
        self.confirm_reset_all = false;
        match writer::reset_all(&self.paths.config) {
            Ok(()) => {
                self.reload_config_after_reset();
                self.last_reload = "all settings reset to defaults (config.toml cleared)".into();
                self.toast_info("all settings reset to defaults");
            }
            Err(err) => {
                self.last_reload = format!("reset-all failed: {err}");
                self.toast_error(format!("reset-all failed: {err}"));
            }
        }
    }

    fn handle_settings_key(&mut self, key: KeyEvent) {
        // A reset-all confirmation is modal: only y / n / Esc act while armed.
        if self.confirm_reset_all {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => self.confirm_reset_all_yes(),
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => self.cancel_reset_all(),
                _ => {}
            }
            return;
        }

        // If a numeric/string edit is in progress, keystrokes feed the buffer.
        if self.settings_edit.is_some() {
            self.handle_settings_edit_key(key);
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
                // Moving off the row discards an uncommitted enum preview (5b).
                self.discard_settings_preview();
                self.settings_selected = (self.settings_selected + 1).min(len - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.discard_settings_preview();
                self.settings_selected = self.settings_selected.saturating_sub(1);
            }
            // Esc cancels an in-progress enum preview (5b). With no preview this
            // arm isn't reached (Esc is handled globally and only routed here when
            // a preview/edit is active).
            KeyCode::Esc => {
                if self.settings_pending.is_some() {
                    self.discard_settings_preview();
                    self.last_reload = "preview cancelled".into();
                }
            }
            // Reset the selected key to its inherited/default value (5a). Delete
            // and Backspace both read as "remove this override". Any staged enum
            // preview is dropped first (reset is a different action).
            KeyCode::Delete | KeyCode::Backspace => {
                self.discard_settings_preview();
                self.reset_selected_setting();
            }
            // Arm the reset-ALL confirmation (5a). Shift+R, mirroring the
            // Dashboard's `R restart` convention; the lowercase `r` is the global
            // refresh.
            KeyCode::Char('R') => {
                self.discard_settings_preview();
                self.arm_reset_all();
            }
            // Cycle an enum field forward/back. Enums PREVIEW (staged, not written)
            // so browsing options can't mutate config (5b); booleans toggle-commit
            // (a binary flip is a decision, not browsing).
            KeyCode::Right | KeyCode::Char(' ') | KeyCode::Left => {
                let forward = !matches!(key.code, KeyCode::Left);
                let Some(k) = self.selected_setting_key() else {
                    self.last_reload = "this row is not inline-editable".into();
                    return;
                };
                match writer::field_spec(&k).map(|s| s.kind) {
                    Some(FieldKind::Enum(_)) => {
                        // Cycle from the current PREVIEW if we're mid-preview on
                        // this key, else from the committed value. Stage only.
                        let from = match &self.settings_pending {
                            Some(p) if p.key == k => p.value.clone(),
                            _ => self.current_setting_value(&k),
                        };
                        if let Some(next) = writer::cycle_enum(&k, &from, forward) {
                            self.settings_pending = Some(PendingEnum {
                                key: k.clone(),
                                value: next.clone(),
                            });
                            self.last_reload =
                                format!("{k} → {next} (preview) · Enter commit · Esc cancel");
                        }
                    }
                    Some(FieldKind::Bool) => {
                        // Binary toggle commits immediately (no browsing to guard).
                        let cur = self.current_setting_value(&k);
                        let next = if cur == "true" { "false" } else { "true" };
                        let (_, status) = self.persist_setting(&k, next);
                        self.last_reload = status;
                    }
                    Some(_) => {
                        self.last_reload = format!("{k} is numeric/text; press Enter to edit");
                    }
                    None => {
                        self.last_reload = format!("{k} is not editable from the TUI");
                    }
                }
            }
            // Enter commits a staged enum preview (5b), else begins editing a
            // numeric/free-string field.
            KeyCode::Enter => {
                // Commit a pending enum preview if one is staged for this row.
                if let Some(pending) = self.settings_pending.take() {
                    let cur_key = self.selected_setting_key();
                    if cur_key.as_deref() == Some(pending.key.as_str()) {
                        let (_, status) = self.persist_setting(&pending.key, &pending.value);
                        self.last_reload = status;
                    } else {
                        // Selection moved under us — discard rather than write a
                        // stale preview to the wrong key.
                        self.last_reload = "preview discarded".into();
                    }
                    return;
                }
                let Some(k) = self.selected_setting_key() else {
                    self.last_reload = "this row is not inline-editable".into();
                    return;
                };
                match writer::field_spec(&k).map(|s| s.kind) {
                    Some(FieldKind::Int { .. })
                    | Some(FieldKind::Float { .. })
                    | Some(FieldKind::FreeStr { .. }) => {
                        let buffer = self.current_setting_value(&k);
                        self.settings_edit = Some(EditState {
                            key: k.clone(),
                            buffer,
                        });
                        self.last_reload =
                            format!("editing {k}: type a value, Enter to save, Esc to cancel");
                    }
                    Some(FieldKind::Enum(_)) => {
                        self.last_reload =
                            format!("{k}: Left/Right/Space to preview values, Enter to commit");
                    }
                    Some(FieldKind::Bool) => {
                        self.last_reload = format!("{k} is on/off; Left/Right/Space toggles");
                    }
                    None => {
                        self.last_reload = format!("{k} is not editable from the TUI");
                    }
                }
            }
            _ => {}
        }
    }

    /// Drop any staged-but-uncommitted enum preview (5b). Called when the cursor
    /// leaves the row, the tab changes, or another action supersedes it. Silent —
    /// callers set their own status line.
    fn discard_settings_preview(&mut self) {
        self.settings_pending = None;
    }

    fn handle_settings_edit_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.settings_edit = None;
                self.last_reload = "edit cancelled".into();
            }
            KeyCode::Enter => {
                // F4: peek the buffer (do NOT take() before validation). Only
                // clear settings_edit on a successful save; on a validation /
                // write error keep the buffer intact so the user can correct
                // the value in place. The error is surfaced via the toast +
                // status line by persist_setting.
                if let Some(edit) = self.settings_edit.as_ref() {
                    let key = edit.key.clone();
                    let raw = edit.buffer.trim().to_string();
                    let (ok, status) = self.persist_setting(&key, &raw);
                    self.last_reload = status;
                    if ok {
                        self.settings_edit = None;
                    }
                }
            }
            KeyCode::Backspace => {
                if let Some(edit) = self.settings_edit.as_mut() {
                    edit.buffer.pop();
                }
            }
            KeyCode::Char(c) => {
                if let Some(edit) = self.settings_edit.as_mut() {
                    edit.buffer.push(c);
                }
            }
            _ => {}
        }
    }

    pub fn drain_chat_events(&mut self) {
        let mut finished = false;
        if let Some(rx) = self.chat.rx.take() {
            while let Ok(event) = rx.try_recv() {
                match event {
                    ChatEvent::Reasoning(text) => {
                        if let Some(last) = self.chat.messages.last_mut() {
                            let rc = last.reasoning_content.get_or_insert_with(String::new);
                            rc.push_str(&text);
                        }
                        self.chat.gen_tokens += 1;
                    }
                    ChatEvent::Content(text) => {
                        if let Some(last) = self.chat.messages.last_mut() {
                            last.content.push_str(&text);
                        }
                        self.chat.gen_tokens += 1;
                    }
                    ChatEvent::Done => {
                        // Normalize empty reasoning to None so serialization omits it.
                        if let Some(last) = self.chat.messages.last_mut() {
                            if last.reasoning_content.as_deref().is_some_and(|s| s.is_empty()) {
                                last.reasoning_content = None;
                            }
                        }
                        // Record throughput + drop a zero-delta empty assistant.
                        if let Some((stats, secs)) = self.chat.finalize_generation() {
                            self.log_request(stats, secs);
                        }
                        self.chat.status = "ready".into();
                        self.chat.sending = false;
                        finished = true;
                    }
                    ChatEvent::Error(err) => {
                        // Normalize empty reasoning as with Done.
                        if let Some(last) = self.chat.messages.last_mut() {
                            if last.reasoning_content.as_deref().is_some_and(|s| s.is_empty()) {
                                last.reasoning_content = None;
                            }
                        }
                        // finalize drops the trailing empty assistant so a failed
                        // request doesn't read as a blank reply; surface the cause.
                        // An errored partial is NOT logged to the inspector
                        // (incomplete + misleading tok/s, same rule as abort).
                        let _ = self.chat.finalize_generation();
                        self.toast_error(format!("chat error: {err}"));
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

/// Throughput of one completed generation (≈ tokens counted from streamed
/// deltas, over the wall-clock generation time).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenStats {
    pub tokens: usize,
    pub tps: f64,
}

/// One finished chat request, for the System-tab request inspector. All fields
/// are measured locally from the TUI's own generation — never fabricated.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestRecord {
    pub model: String,
    pub tokens: usize,
    pub tps: f64,
    pub secs: f64,
}

/// Cap on retained request records (newest kept).
const REQUEST_LOG_CAP: usize = 20;

pub struct ChatState {
    pub input: String,
    pub messages: Vec<ChatMessage>,
    pub status: String,
    pub sending: bool,
    pub scroll: u16,
    /// Session system prompt, prepended to the request when non-empty (/system).
    pub system_prompt: String,
    /// Per-session sampling overrides (/temp, /top_p); None = serve default.
    pub temp: Option<f64>,
    pub top_p: Option<f64>,
    /// Wall-clock start of the in-flight generation (for tok/s).
    gen_start: Option<Instant>,
    /// Model the in-flight generation was sent to (captured at send, since
    /// active_model can change mid-stream via /model).
    gen_model: String,
    /// Streamed deltas counted so far this generation (≈ tokens).
    gen_tokens: usize,
    /// Stats of the most recent completed generation (shown under the reply).
    pub last_stats: Option<GenStats>,
    rx: Option<Receiver<ChatEvent>>,
    input_focused: bool,
    // Set true to ask the in-flight stream thread to stop (checked per line).
    abort: Arc<AtomicBool>,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            input: String::new(),
            messages: Vec::new(),
            status: "ready".into(),
            sending: false,
            scroll: 0,
            system_prompt: String::new(),
            temp: None,
            top_p: None,
            gen_start: None,
            gen_model: String::new(),
            gen_tokens: 0,
            last_stats: None,
            rx: None,
            input_focused: true,
            abort: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ChatState {
    /// Ask an in-flight generation to stop, and optimistically free the tab. A
    /// hung read may never deliver Done, so we don't wait on the worker: clear
    /// `sending`, drop the receiver, and finalize. The detached thread exits when
    /// its socket errors (bounded by the read timeout); its late events land on a
    /// dropped receiver. The partial reply already streamed is kept.
    pub fn request_abort(&mut self) {
        if self.sending {
            self.abort.store(true, Ordering::Relaxed);
            self.sending = false;
            self.rx = None;
            let _ = self.finalize_generation(); // aborted = incomplete, not logged
            self.status = "stopped".into();
        }
    }

    /// End-of-generation cleanup, shared by Done / Error / abort: record tok/s
    /// (only if any tokens actually streamed) and drop a trailing empty assistant
    /// bubble left by a zero-delta finish (immediate [DONE], empty deltas, or
    /// abort-before-first-token) so it isn't rendered or sent in the next turn.
    fn finalize_generation(&mut self) -> Option<(GenStats, f64)> {
        let mut recorded = None;
        if let Some(start) = self.gen_start.take() {
            if self.gen_tokens > 0 {
                let secs = start.elapsed().as_secs_f64();
                let tps = if secs > 0.0 {
                    self.gen_tokens as f64 / secs
                } else {
                    0.0
                };
                let stats = GenStats {
                    tokens: self.gen_tokens,
                    tps,
                };
                self.last_stats = Some(stats);
                recorded = Some((stats, secs));
            }
        }
        if let Some(last) = self.messages.last() {
            let reasoning_empty = last
                .reasoning_content
                .as_deref()
                .map_or(true, |s| s.is_empty());
            if last.role == "assistant" && last.content.is_empty() && reasoning_empty {
                self.messages.pop();
            }
        }
        recorded
    }

    /// Clear any displayed throughput + in-flight counters. Called when the
    /// conversation the stats referred to is replaced (/clear, /load, /edit).
    fn reset_stats(&mut self) {
        self.last_stats = None;
        self.gen_start = None;
        self.gen_tokens = 0;
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// Build an App whose config writes target an isolated temp file, so the
    /// settings tests never touch the user's real ~/.hipfire/config.toml.
    fn test_app() -> (App, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-tui-app-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("config.toml");
        std::fs::write(&cfg, "schema_version = 1\n").unwrap();

        let mut app = App::load().expect("load app");
        app.paths.root = dir.clone();
        app.paths.config = cfg;
        app.paths.legacy_config = dir.join("config.json");
        app.config = ConfigState::load(&app.paths);
        app.tab = Tab::Settings;
        (app, dir)
    }

    #[test]
    fn mode_switch_clamps_selection_into_visible_rows() {
        // F3: from a high advanced-row index, pressing `e` (easy) must clamp the
        // selection into the short easy table so a row is always selectable.
        let (mut app, dir) = test_app();
        app.settings_easy = false;
        // Populate advanced rows so the index can sit far past the easy count.
        app.config.values.clear();
        for i in 0..20 {
            app.config.values.insert(format!("k{i:02}"), "v".into());
        }
        app.settings_selected = 19; // last advanced row

        app.set_settings_easy(true);

        let easy_rows = app.config.easy_rows().len();
        assert!(easy_rows > 0);
        assert!(
            app.settings_selected < easy_rows,
            "selection {} must be clamped below easy row count {}",
            app.settings_selected,
            easy_rows
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mode_switch_preserves_selected_key_when_present() {
        // F3 niceness: if the selected advanced key also appears in easy mode,
        // the cursor follows it (kv_cache is row 3 in easy_keys).
        let (mut app, dir) = test_app();
        app.settings_easy = false;
        app.config.values.clear();
        // Order matters: BTreeMap sorts keys. Put kv_cache somewhere mid-list.
        for k in ["aaa", "kv_cache", "zzz"] {
            app.config.values.insert(k.into(), "auto".into());
        }
        // Advanced index of "kv_cache" (sorted: aaa, kv_cache, zzz → idx 1).
        app.settings_selected = 1;

        app.set_settings_easy(true);
        // kv_cache is index 3 in easy_keys().
        let easy_idx = app
            .config
            .easy_keys()
            .iter()
            .position(|k| matches!(k, Some("kv_cache")))
            .unwrap();
        assert_eq!(app.settings_selected, easy_idx);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reset_selected_deletes_override_and_reverts_to_default() {
        // 5a: an explicit override is removed from disk and the in-memory value
        // falls back to the hardcoded default after a config reload.
        let (mut app, dir) = test_app();
        std::fs::write(
            &app.paths.config,
            "schema_version = 1\n[memory]\nkv_cache = \"q8\"\n",
        )
        .unwrap();
        app.config = ConfigState::load(&app.paths);
        app.settings_easy = false;
        assert!(app.config.is_override("kv_cache"));
        assert_eq!(
            app.config.values.get("kv_cache").map(String::as_str),
            Some("q8")
        );

        // Select kv_cache in the advanced list and reset it.
        app.settings_selected = app
            .config
            .values
            .keys()
            .position(|k| k == "kv_cache")
            .unwrap();
        app.handle_settings_key(key(KeyCode::Delete));

        // Gone from disk, no longer an override, reverted to the default literal.
        let loaded =
            hipfire_config::load_global(&hipfire_config::ConfigPaths::under(&app.paths.root))
                .unwrap();
        assert!(loaded.layer.get("kv_cache").is_none());
        assert!(!app.config.is_override("kv_cache"));
        assert_eq!(
            app.config.values.get("kv_cache").map(String::as_str),
            Some("auto")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reset_selected_is_noop_when_already_default() {
        // Resetting a key that has no override must not write the file.
        let (mut app, dir) = test_app();
        std::fs::write(
            &app.paths.config,
            "schema_version = 1\n[reasoning]\nmode = \"off\"\n",
        )
        .unwrap();
        app.config = ConfigState::load(&app.paths);
        app.settings_easy = false;
        // kv_cache is NOT overridden here.
        app.settings_selected = app
            .config
            .values
            .keys()
            .position(|k| k == "kv_cache")
            .unwrap();
        let before = std::fs::read_to_string(&app.paths.config).unwrap();
        app.handle_settings_key(key(KeyCode::Delete));
        let after = std::fs::read_to_string(&app.paths.config).unwrap();
        assert_eq!(before, after, "no-op reset must not rewrite the file");
        assert!(app.last_reload.contains("already at its default"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reset_all_requires_confirmation_then_clears_file() {
        // 5a reset-all: Shift+R arms a confirm; only `y` wipes the file.
        let (mut app, dir) = test_app();
        std::fs::write(
            &app.paths.config,
            "schema_version = 1\n[memory]\nkv_cache = \"q8\"\n[speculation]\ndflash = \"auto\"\n",
        )
        .unwrap();
        app.config = ConfigState::load(&app.paths);

        // Arm, then CANCEL — the file is untouched.
        app.handle_settings_key(key(KeyCode::Char('R')));
        assert!(app.confirm_reset_all, "Shift+R arms the confirm");
        app.handle_settings_key(key(KeyCode::Char('n')));
        assert!(!app.confirm_reset_all, "n cancels");
        let still =
            hipfire_config::load_global(&hipfire_config::ConfigPaths::under(&app.paths.root))
                .unwrap();
        assert_eq!(still.layer.values.len(), 2, "cancel keeps overrides");

        // Arm again and CONFIRM — config.toml retains only schema_version.
        app.handle_settings_key(key(KeyCode::Char('R')));
        app.handle_settings_key(key(KeyCode::Char('y')));
        assert!(!app.confirm_reset_all);
        let cleared =
            hipfire_config::load_global(&hipfire_config::ConfigPaths::under(&app.paths.root))
                .unwrap();
        assert!(cleared.layer.values.is_empty(), "confirm clears overrides");
        assert!(!app.config.is_override("kv_cache"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cycled_setting_is_resettable_same_session() {
        // 5a review #1: a setting changed this session (via cycle/persist) is
        // immediately an override, so Delete can reset it without a full reload.
        let (mut app, dir) = test_app();
        std::fs::write(&app.paths.config, "schema_version = 1\n").unwrap();
        app.config = ConfigState::load(&app.paths);
        app.settings_easy = false;
        // Cycle kv_cache off its default (auto -> q8): previews, then Enter commits
        // + marks override (5b: cycle no longer persists on its own).
        app.settings_selected = app
            .config
            .values
            .keys()
            .position(|k| k == "kv_cache")
            .unwrap();
        app.handle_settings_key(key(KeyCode::Right));
        app.handle_settings_key(key(KeyCode::Enter));
        assert!(
            app.config.is_override("kv_cache"),
            "commit marks the override"
        );
        assert_ne!(
            app.config.values.get("kv_cache").map(String::as_str),
            Some("auto")
        );

        // Now Delete must actually reset it (not report "already default").
        app.handle_settings_key(key(KeyCode::Delete));
        assert!(!app.config.is_override("kv_cache"));
        assert_eq!(
            app.config.values.get("kv_cache").map(String::as_str),
            Some("auto")
        );
        assert!(app.last_reload.contains("reset to default"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reset_all_keeps_cursor_on_surviving_key() {
        // Reset-all rebuilds from the typed schema. The cursor must follow the
        // selected key rather than relying on an old numeric position.
        let (mut app, dir) = test_app();
        std::fs::write(
            &app.paths.config,
            "schema_version = 1\n[memory]\nkv_cache = \"q8\"\n[attention]\nflash = \"always\"\n",
        )
        .unwrap();
        app.config = ConfigState::load(&app.paths);
        app.settings_easy = false;
        app.settings_selected = app
            .config
            .values
            .keys()
            .position(|k| k == "kv_cache")
            .unwrap();
        // Confirmed reset-all.
        app.handle_settings_key(key(KeyCode::Char('R')));
        app.handle_settings_key(key(KeyCode::Char('y')));

        // kv_cache survives as a schema default and remains selected.
        assert_eq!(
            app.config
                .values
                .keys()
                .nth(app.settings_selected)
                .map(String::as_str),
            Some("kv_cache"),
            "cursor follows kv_cache to its new row"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reset_on_corrupt_config_points_to_reset_all() {
        // 5a review #5: a corrupt config.toml is not "already default" — direct
        // the user to reset-all (the recovery hatch).
        let (mut app, dir) = test_app();
        std::fs::write(&app.paths.config, "{ broken not json").unwrap();
        app.config = ConfigState::load(&app.paths);
        assert!(
            app.config.warning.is_some(),
            "corrupt file loads with a warning"
        );
        app.settings_easy = false;
        app.settings_selected = 0;
        app.handle_settings_key(key(KeyCode::Delete));
        assert!(
            app.last_reload.contains("reset all") || app.last_reload.contains("unreadable"),
            "got: {}",
            app.last_reload
        );
        // The corrupt file was NOT clobbered by the single-key path.
        assert_eq!(
            std::fs::read_to_string(&app.paths.config).unwrap(),
            "{ broken not json"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reset_all_confirm_is_modal_against_quit_and_reload() {
        // While the reset-all confirm is armed, a stray key (not y/n/Esc) must be
        // swallowed — it neither cancels nor leaks to a global shortcut.
        let (mut app, dir) = test_app();
        app.handle_settings_key(key(KeyCode::Char('R')));
        assert!(app.confirm_reset_all);
        app.handle_settings_key(key(KeyCode::Char('j'))); // would normally move
        assert!(
            app.confirm_reset_all,
            "unrelated key is swallowed, stays armed"
        );
        app.handle_settings_key(key(KeyCode::Esc));
        assert!(!app.confirm_reset_all, "Esc cancels");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Select the advanced-mode row for `k` and return the on-disk config text.
    fn select_advanced(app: &mut App, k: &str) {
        app.settings_easy = false;
        app.settings_selected = app.config.values.keys().position(|x| x == k).unwrap();
    }

    #[test]
    fn enum_cycle_previews_without_writing() {
        // 5b core: cycling an enum stages a preview and does NOT touch config.toml
        // — merely browsing options can no longer mutate config.
        let (mut app, dir) = test_app();
        std::fs::write(&app.paths.config, "schema_version = 1\n").unwrap();
        app.config = ConfigState::load(&app.paths);
        select_advanced(&mut app, "dflash_mode");

        app.handle_settings_key(key(KeyCode::Right));
        // A preview is staged for dflash_mode...
        let pending = app.settings_pending.as_ref().expect("preview staged");
        assert_eq!(pending.key, "dflash_mode");
        assert_ne!(pending.value, "off", "cycled off the default");
        // ...but nothing was written and the committed value is unchanged.
        assert_eq!(
            std::fs::read_to_string(&app.paths.config).unwrap(),
            "schema_version = 1\n"
        );
        assert!(!app.config.is_override("dflash_mode"));
        assert_eq!(
            app.config.values.get("dflash_mode").map(String::as_str),
            Some("off")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn enum_preview_commits_on_enter() {
        // Enter writes the staged preview to disk and marks the override.
        let (mut app, dir) = test_app();
        std::fs::write(&app.paths.config, "schema_version = 1\n").unwrap();
        app.config = ConfigState::load(&app.paths);
        select_advanced(&mut app, "dflash_mode");

        app.handle_settings_key(key(KeyCode::Right)); // off -> on (preview)
        let staged = app.settings_pending.as_ref().unwrap().value.clone();
        app.handle_settings_key(key(KeyCode::Enter)); // commit

        assert!(
            app.settings_pending.is_none(),
            "preview cleared after commit"
        );
        assert!(app.config.is_override("dflash_mode"));
        assert_eq!(
            app.config.values.get("dflash_mode").cloned(),
            Some(staged.clone())
        );
        let loaded =
            hipfire_config::load_global(&hipfire_config::ConfigPaths::under(&app.paths.root))
                .unwrap();
        assert_eq!(
            loaded.layer.get("dflash_mode"),
            Some(&hipfire_config::ConfigValue::String(staged))
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn enum_preview_cancels_on_esc_and_on_navigation() {
        // Esc OR moving off the row discards the preview without writing.
        let (mut app, dir) = test_app();
        std::fs::write(&app.paths.config, "schema_version = 1\n").unwrap();
        app.config = ConfigState::load(&app.paths);
        select_advanced(&mut app, "dflash_mode");

        // Esc path.
        app.handle_settings_key(key(KeyCode::Right));
        assert!(app.settings_pending.is_some());
        app.handle_settings_key(key(KeyCode::Esc));
        assert!(app.settings_pending.is_none(), "Esc cancels preview");
        assert_eq!(
            std::fs::read_to_string(&app.paths.config).unwrap(),
            "schema_version = 1\n"
        );

        // Navigation path: Down discards.
        app.handle_settings_key(key(KeyCode::Right));
        assert!(app.settings_pending.is_some());
        app.handle_settings_key(key(KeyCode::Down));
        assert!(
            app.settings_pending.is_none(),
            "moving off the row discards preview"
        );
        assert_eq!(
            std::fs::read_to_string(&app.paths.config).unwrap(),
            "schema_version = 1\n"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn enum_preview_repeated_cycle_stages_from_preview() {
        // Two Rights advance through the allowlist from the PREVIEW, not the
        // committed value (so you can browse forward without committing).
        let (mut app, dir) = test_app();
        std::fs::write(&app.paths.config, "schema_version = 1\n").unwrap();
        app.config = ConfigState::load(&app.paths);
        select_advanced(&mut app, "dflash_mode"); // allowlist: on, off, auto; cur=off

        app.handle_settings_key(key(KeyCode::Right)); // off -> auto
        let first = app.settings_pending.as_ref().unwrap().value.clone();
        app.handle_settings_key(key(KeyCode::Right)); // auto -> on
        let second = app.settings_pending.as_ref().unwrap().value.clone();
        assert_ne!(first, second, "second cycle advances from the preview");
        // Still nothing written.
        assert_eq!(
            std::fs::read_to_string(&app.paths.config).unwrap(),
            "schema_version = 1\n"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bool_toggle_still_commits_immediately() {
        // 5b: booleans are a binary decision, not browsing — they keep
        // toggle-on-key (immediate write), no preview.
        let (mut app, dir) = test_app();
        std::fs::write(&app.paths.config, "schema_version = 1\n").unwrap();
        app.config = ConfigState::load(&app.paths);
        select_advanced(&mut app, "cask"); // default false

        app.handle_settings_key(key(KeyCode::Char(' ')));
        assert!(app.settings_pending.is_none(), "bool does not preview");
        assert!(app.config.is_override("cask"));
        let loaded =
            hipfire_config::load_global(&hipfire_config::ConfigPaths::under(&app.paths.root))
                .unwrap();
        assert_eq!(
            loaded.layer.get("cask"),
            Some(&hipfire_config::ConfigValue::Bool(true))
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tab_switch_discards_preview() {
        // Leaving the Settings tab drops an uncommitted preview.
        let (mut app, dir) = test_app();
        std::fs::write(&app.paths.config, "schema_version = 1\n").unwrap();
        app.config = ConfigState::load(&app.paths);
        select_advanced(&mut app, "dflash_mode");
        app.handle_settings_key(key(KeyCode::Right));
        assert!(app.settings_pending.is_some());
        app.next_tab();
        assert!(
            app.settings_pending.is_none(),
            "tab switch discards preview"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_save_keeps_edit_buffer() {
        // F4: a rejected value (out of range) on Enter must KEEP settings_edit
        // so the user can correct in place; a valid value commits + exits edit.
        let (mut app, dir) = test_app();
        // Edit `temperature` (Float 0.0..=2.0) with an out-of-range buffer.
        app.settings_edit = Some(EditState {
            key: "temperature".into(),
            buffer: "5.0".into(),
        });
        app.handle_settings_edit_key(key(KeyCode::Enter));
        assert!(
            app.settings_edit.is_some(),
            "rejected save must keep the edit buffer for correction"
        );
        assert_eq!(
            app.settings_edit.as_ref().unwrap().buffer,
            "5.0",
            "buffer contents must be preserved verbatim on failure"
        );
        // An error toast was surfaced.
        assert!(matches!(
            app.toast.as_ref().map(|t| t.level),
            Some(ToastLevel::Error)
        ));

        // Now correct it to a valid value: commits and exits edit mode.
        app.settings_edit.as_mut().unwrap().buffer = "0.7".into();
        app.handle_settings_edit_key(key(KeyCode::Enter));
        assert!(
            app.settings_edit.is_none(),
            "valid save must commit and exit edit mode"
        );
        assert_eq!(
            app.config.values.get("temperature").map(String::as_str),
            Some("0.7")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tab_at_maps_column_to_tab() {
        let (mut app, dir) = test_app();
        app.tab_row_y = 3;
        app.tab_hitboxes = vec![
            (0, 6, Tab::Home),
            (9, 20, Tab::Dashboard),
            (23, 29, Tab::Chat),
        ];
        assert_eq!(app.tab_at(3, 3), Some(Tab::Home));
        assert_eq!(app.tab_at(5, 3), Some(Tab::Home)); // end-exclusive: 5 < 6
        assert_eq!(app.tab_at(6, 3), None); // divider gap between cells
        assert_eq!(app.tab_at(12, 3), Some(Tab::Dashboard));
        assert_eq!(app.tab_at(3, 4), None); // wrong row
        assert_eq!(app.tab_at(200, 3), None); // past the last tab
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ui_state_persists_tab_and_expanded_groups() {
        let (mut app, dir) = test_app();
        app.paths.ui_state = dir.join("ui_state.json");
        app.tab = Tab::Models;
        app.registry.expanded_groups.insert("qwen".into());
        app.save_ui_state();
        let restored = crate::hipfire::ui_state::UiState::load(&app.paths.ui_state);
        assert_eq!(restored.active_tab, "Models");
        assert!(restored.expanded_groups.contains(&"qwen".to_string()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn drain_serve_command_idle_is_noop() {
        let (mut app, dir) = test_app();
        assert!(!app.serve_cmd_running());
        app.drain_serve_command();
        assert!(app.toast.is_none(), "no in-flight command -> no toast");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn drain_serve_command_consumes_outcome_and_toasts() {
        let (mut app, dir) = test_app();
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(ServeOutcome::Ok("serve start: done".into()))
            .unwrap();
        app.inject_serve_cmd(rx);
        assert!(app.serve_cmd_running());
        app.drain_serve_command();
        assert!(
            !app.serve_cmd_running(),
            "outcome clears the in-flight command"
        );
        assert!(app.toast.is_some(), "outcome raises a toast");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn delete_confirm_can_be_cancelled() {
        let (mut app, dir) = test_app();
        app.confirm_delete = Some("qwen3.5:9b".into());
        app.cancel_delete();
        assert!(
            app.confirm_delete.is_none(),
            "Esc/n clears the armed delete"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn drain_pull_and_rm_idle_are_noops() {
        let (mut app, dir) = test_app();
        app.drain_pull();
        app.drain_rm();
        assert!(app.toast.is_none(), "no in-flight pull/rm -> no toast");
        assert!(app.pull.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn drain_pull_updates_progress() {
        let (mut app, dir) = test_app();
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(PullEvent::Progress {
            percent: Some(42.0),
            line: "[bar] 42.0% 8 MB/s".into(),
        })
        .unwrap();
        app.inject_pull("qwen3.5:9b".into(), rx);
        app.drain_pull();
        let job = app
            .pull
            .as_ref()
            .expect("pull stays active during progress");
        assert_eq!(job.percent, Some(42.0));
        assert_eq!(job.line, "[bar] 42.0% 8 MB/s");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_slash_commands_set_session_state() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;

        // /system sets then clears the prompt.
        app.handle_chat_command("system You are a helpful pirate.");
        assert_eq!(app.chat.system_prompt, "You are a helpful pirate.");
        app.handle_chat_command("system");
        assert!(app.chat.system_prompt.is_empty());

        // /temp + /top_p validate range and accept off.
        app.handle_chat_command("temp 0.7");
        assert_eq!(app.chat.temp, Some(0.7));
        app.handle_chat_command("temp 9"); // out of 0..=2 -> rejected, unchanged
        assert_eq!(app.chat.temp, Some(0.7));
        app.handle_chat_command("temp off");
        assert_eq!(app.chat.temp, None);
        app.handle_chat_command("top_p 0.9");
        assert_eq!(app.chat.top_p, Some(0.9));
        app.handle_chat_command("top_p 5"); // out of 0..=1 -> rejected
        assert_eq!(app.chat.top_p, Some(0.9));

        // /clear empties the conversation.
        app.chat.messages.push(ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            reasoning_content: None,
        });
        app.handle_chat_command("clear");
        assert!(app.chat.messages.is_empty());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_unknown_command_toasts_does_not_panic() {
        let (mut app, dir) = test_app();
        app.handle_chat_command("definitely-not-a-command");
        assert!(
            app.toast.is_some(),
            "unknown command surfaces an error toast"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_save_load_sessions_round_trip() {
        let (mut app, dir) = test_app();
        app.paths.chat_history = dir.join("chat_history.json");
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "hello".into(),
                reasoning_content: None,
            },
        ];
        app.handle_chat_command("save debug");
        app.handle_chat_command("clear");
        assert!(app.chat.messages.is_empty());
        app.handle_chat_command("load debug");
        assert_eq!(app.chat.messages.len(), 2);
        assert_eq!(app.chat.messages[1].content, "hello");
        app.handle_chat_command("sessions");
        assert!(app.chat.status.contains("debug"));
        app.handle_chat_command("delete debug");
        app.handle_chat_command("load debug");
        assert!(
            app.toast.is_some(),
            "loading a deleted session surfaces an error"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_edit_last_pulls_user_message_into_input() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "first".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "reply".into(),
                reasoning_content: None,
            },
        ];
        app.handle_chat_command("edit");
        assert_eq!(app.chat.input, "first");
        assert!(
            app.chat.messages.is_empty(),
            "the edited turn and its reply are removed"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_copy_stages_last_reply_for_clipboard() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "q".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "the answer".into(),
                reasoning_content: None,
            },
        ];
        app.handle_chat_command("copy");
        assert_eq!(app.pending_clipboard.as_deref(), Some("the answer"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_regenerate_without_history_toasts() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.handle_chat_command("regen");
        assert!(
            app.toast.is_some(),
            "regenerate with no turn to redo toasts"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn regenerate_offline_preserves_the_reply() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.status.serve_http_ok = false; // serve down
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "q".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "a".into(),
                reasoning_content: None,
            },
        ];
        app.regenerate();
        // Serve-readiness is checked BEFORE the destructive pop, so the reply
        // survives an offline regen attempt.
        assert_eq!(
            app.chat.messages.len(),
            2,
            "reply preserved when serve offline"
        );
        assert_eq!(app.chat.messages[1].content, "a");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn zero_delta_done_drops_empty_assistant_bubble() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
            reasoning_content: None, // empty slot, no deltas streamed
            },
        ];
        let (tx, rx) = std::sync::mpsc::channel();
        app.chat.rx = Some(rx);
        app.chat.sending = true;
        app.chat.gen_start = Some(Instant::now());
        tx.send(ChatEvent::Done).unwrap();
        app.drain_chat_events();
        assert_eq!(app.chat.messages.len(), 1, "empty assistant bubble dropped");
        assert!(
            app.chat.last_stats.is_none(),
            "no stats when zero tokens streamed"
        );
        assert!(!app.chat.sending);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn clear_resets_stale_throughput_stats() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.last_stats = Some(GenStats {
            tokens: 42,
            tps: 99.0,
        });
        app.handle_chat_command("clear");
        assert!(
            app.chat.last_stats.is_none(),
            "tok/s cleared with the conversation"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn finished_request_is_logged_for_the_inspector() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                reasoning_content: None,
            },
        ];
        let (tx, rx) = std::sync::mpsc::channel();
        app.chat.rx = Some(rx);
        app.chat.sending = true;
        app.chat.gen_start = Some(Instant::now());
        app.chat.gen_model = "qwen3.5:9b".into();
        tx.send(ChatEvent::Content("hello".into())).unwrap();
        tx.send(ChatEvent::Content(" world".into())).unwrap();
        tx.send(ChatEvent::Done).unwrap();
        app.drain_chat_events();
        assert_eq!(app.request_log.len(), 1, "completed request is logged");
        assert_eq!(app.request_log[0].model, "qwen3.5:9b");
        assert_eq!(app.request_log[0].tokens, 2);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn errored_request_is_not_logged() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                reasoning_content: None,
            },
        ];
        let (tx, rx) = std::sync::mpsc::channel();
        app.chat.rx = Some(rx);
        app.chat.sending = true;
        app.chat.gen_start = Some(Instant::now());
        app.chat.gen_model = "qwen3.5:9b".into();
        tx.send(ChatEvent::Content("partial".into())).unwrap();
        tx.send(ChatEvent::Error("boom".into())).unwrap();
        app.drain_chat_events();
        assert!(
            app.request_log.is_empty(),
            "an errored partial is not logged as a request"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn reload_preserves_expanded_groups() {
        // An `r` refresh reloads the registry; it must NOT collapse the user's
        // expanded Models groups (a fresh RegistryState defaults them empty).
        let (mut app, dir) = test_app();
        app.registry.expanded_groups.insert("qwen".into());
        app.reload();
        assert!(
            app.registry.expanded_groups.contains("qwen"),
            "reload must carry expanded groups across the registry reload"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn drain_chat_events_separates_reasoning_and_content() {
        // SSE fold now routes Reasoning and Content to separate accumulators;
        // rendering keeps visual order but history must keep them distinct.
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                reasoning_content: None,
            },
        ];
        let (tx, rx) = std::sync::mpsc::channel();
        app.chat.rx = Some(rx);
        app.chat.sending = true;
        app.chat.gen_start = Some(Instant::now());
        app.chat.gen_model = "test".into();
        tx.send(ChatEvent::Reasoning("think ".into())).unwrap();
        tx.send(ChatEvent::Reasoning("step".into())).unwrap();
        tx.send(ChatEvent::Content("answer".into())).unwrap();
        tx.send(ChatEvent::Done).unwrap();
        app.drain_chat_events();
        let last = app.chat.messages.last().unwrap();
        assert_eq!(
            last.reasoning_content.as_deref(),
            Some("think step"),
            "reasoning deltas accumulated separately"
        );
        assert_eq!(last.content, "answer", "content deltas accumulated separately");
        // Serialized request keeps them as distinct keys for prefix-cache.
        let v = serde_json::to_value(&*app.chat.messages).unwrap();
        let asst = &v[1];
        assert_eq!(asst["content"], "answer");
        assert_eq!(asst["reasoning_content"], "think step");
        assert_eq!(app.chat.gen_tokens, 3);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_message_legacy_json_without_reasoning_deserializes() {
        // Old chat_history.json files lack the field; they must load as None.
        let msg: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "old"
        }))
        .unwrap();
        assert!(msg.reasoning_content.is_none());
        let v = serde_json::to_value(&msg).unwrap();
        assert!(v.get("reasoning_content").is_none());
    }

    #[test]
    fn reasoning_empty_normalizes_to_none_on_done() {
        let (mut app, dir) = test_app();
        app.tab = Tab::Chat;
        app.chat.messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                reasoning_content: None,
            },
        ];
        let (tx, rx) = std::sync::mpsc::channel();
        app.chat.rx = Some(rx);
        app.chat.sending = true;
        app.chat.gen_start = Some(Instant::now());
        // No reasoning deltas, but an empty Some("") was somehow set.
        app.chat.messages.last_mut().unwrap().reasoning_content = Some(String::new());
        tx.send(ChatEvent::Content("hi".into())).unwrap();
        tx.send(ChatEvent::Done).unwrap();
        app.drain_chat_events();
        assert!(app.chat.messages.last().unwrap().reasoning_content.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}
