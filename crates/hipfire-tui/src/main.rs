// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

mod app;
mod hipfire;
mod ui;
mod ui_chat;

use std::{io, panic};

use anyhow::Result;
use app::App;
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
        MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> Result<()> {
    // Non-interactive argument handling (headless-safe, no TTY required).
    // Only the first argument is inspected — every branch returns/exits, so
    // there is never a second iteration.
    let profile_wizard = if let Some(arg) = std::env::args().nth(1) {
        match arg.as_str() {
            "--version" | "-V" => {
                println!("hipfire-tui {VERSION}");
                return Ok(());
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--check" => {
                return run_check();
            }
            "--config-profile-wizard" => true,
            other => {
                eprintln!("hipfire-tui: unknown argument '{other}'");
                print_help();
                std::process::exit(2);
            }
        }
    } else {
        false
    };

    let mut terminal = setup_terminal()?;
    let result = if profile_wizard {
        hipfire::profile_wizard::run(&mut terminal)
    } else {
        run(&mut terminal)
    };
    restore_terminal(&mut terminal)?;
    result
}
fn print_help() {
    println!(
        "hipfire-tui {VERSION} - terminal UI for hipfire\n\
         \n\
         USAGE:\n    \
             hipfire-tui [FLAGS]\n\
         \n\
         FLAGS:\n    \
             -h, --help               Print this help and exit\n    \
             -V, --version            Print version and exit\n    \
                 --check              Load config/registry/models without entering the\n                         \
                                      render loop, then exit 0 on success (headless smoke)\n    \
                 --config-profile-wizard\n                         \
                                      Select/create profiles and browse config variables\n\
         \n\
         With no flags, hipfire-tui launches the interactive ratatui UI (requires a TTY).\n\
         Tabs: Home, Dashboard, Chat, Models, Settings, System. Tab/Shift+Tab to switch, q to quit."
    );
}

/// Construct the App state (config + registry + local models) WITHOUT entering
/// the ratatui render/event loop. Exits 0 on success, non-zero on init failure.
/// This is the headless smoke path for CI and TTY-less environments.
fn run_check() -> Result<()> {
    let app = App::load()?;
    println!("hipfire-tui --check OK");
    println!("  config: default_model = {:?}", app.config.default_model);
    println!(
        "  registry: {} models, {} aliases",
        app.registry.models.len(),
        app.registry.aliases.len()
    );
    println!("  local models: {}", app.registry.local_files.len());
    if let Some(warn) = &app.registry.warning {
        println!("  registry warning: {warn}");
    }
    Ok(())
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

    let hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
        hook(info);
    }));

    let backend = CrosstermBackend::new(stdout);
    match Terminal::new(backend) {
        Ok(t) => Ok(t),
        Err(e) => {
            // We enabled raw mode + alt screen + mouse capture above; undo them
            // (best-effort) before surfacing the failure so the terminal is left
            // usable.
            let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
            let _ = disable_raw_mode();
            Err(e.into())
        }
    }
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    // Attempt every step independently so an early failure can't leave mouse
    // capture, the alternate screen, or raw mode enabled. Return the first
    // error after all steps have been tried.
    let leave = execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    );
    let cursor = terminal.show_cursor();
    let raw = disable_raw_mode();
    leave?;
    cursor?;
    raw?;
    Ok(())
}

fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut app = App::load()?;
    let result = event_loop(terminal, &mut app);
    // Persist session UI state (active tab + expanded Models groups) on ANY exit
    // — clean quit OR a propagated error — so the next launch reopens where the
    // user left off. Best-effort (the save swallows its own errors); a panic
    // still skips this, which is acceptable for non-load-bearing UI state.
    app.save_ui_state();
    result
}

/// Copy `text` to the terminal's clipboard via OSC 52. Supported by many
/// terminals (kitty, iTerm2, wezterm, tmux with `set-clipboard on`); a harmless
/// no-op where unsupported — the escape is consumed without altering the screen.
fn emit_clipboard(text: &str) {
    use std::io::Write;
    let seq = osc52_sequence(text);
    let mut out = io::stdout();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}

/// The OSC 52 set-clipboard escape frame for `text` (extracted for testing).
fn osc52_sequence(text: &str) -> String {
    format!("\x1b]52;c;{}\x07", base64_encode(text.as_bytes()))
}

/// Minimal standard-alphabet base64 (OSC 52 payload); avoids a new dependency.
fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = (chunk[0] as u32) << 16
            | (*chunk.get(1).unwrap_or(&0) as u32) << 8
            | *chunk.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18 & 63) as usize] as char);
        out.push(T[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn event_loop(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> Result<()> {
    loop {
        // Live-serve Dashboard: the fetch (HTTP + rocm-smi) runs on a dedicated
        // background thread. Here on the UI thread we ONLY mirror its latest
        // snapshot (a cheap lock+clone) and tell it whether the Dashboard tab
        // is focused — never any synchronous network / rocm-smi call, so a hung
        // probe cannot block render or input.
        app.sync_dashboard();
        app.sync_logs();
        app.expire_toast();

        terminal.draw(|frame| ui::draw(frame, app))?;
        app.drain_chat_events();
        app.drain_serve_command();
        app.drain_pull();
        app.drain_rm();
        app.drain_doctor();
        if let Some(text) = app.pending_clipboard.take() {
            emit_clipboard(&text);
        }

        if event::poll(std::time::Duration::from_millis(80))? {
            match event::read()? {
                Event::Key(key) => {
                    if handle_key(app, key) {
                        break;
                    }
                }
                Event::Mouse(mouse) => handle_mouse(app, mouse),
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }
    Ok(())
}

fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        if app.chat.sending {
            app.chat.status = "generation in progress — press Esc to stop it".into();
            return false;
        }
        return true;
    }

    // The `?` help overlay is modal: while it is open, any key (other than the
    // Ctrl+C quit handled above) simply closes it.
    if app.show_help {
        app.show_help = false;
        return false;
    }

    // A Models delete-confirmation is modal: route every key to the tab handler
    // (which acts only on y / n / Esc) so a global `q` or `r` cannot quit or
    // reload while the confirm is armed.
    if app.tab == app::Tab::Models && app.confirm_delete.is_some() {
        app.handle_tab_key(key);
        return false;
    }

    // The Settings reset-all confirmation is likewise modal: route every key to
    // the tab handler (which acts only on y / n / Esc) so a global `q` or `r`
    // can't quit or reload while the destructive prompt is armed.
    if app.tab == app::Tab::Settings && app.confirm_reset_all {
        app.handle_tab_key(key);
        return false;
    }

    // While a settings value is being edited, keystrokes (including q/e/a/r)
    // feed the edit buffer rather than triggering global shortcuts.
    let editing_setting = app.tab == app::Tab::Settings && app.settings_edit.is_some();
    if editing_setting && !matches!(key.code, KeyCode::Tab | KeyCode::BackTab) {
        app.handle_tab_key(key);
        return false;
    }

    // The chat input only captures global keys (q/r/Esc) while the Chat tab is
    // focused AND its input is focused. On every other tab the input is not on
    // screen, so q/r/Esc act as global shortcuts immediately from startup
    // (chat input defaults focused, but that must not gate other tabs).
    let chat_capturing = app.tab == app::Tab::Chat && app.chat.is_input_focused();

    match key.code {
        KeyCode::Char('q') if !chat_capturing => return true,
        KeyCode::Esc => {
            // While editing a settings value OR previewing an enum (5b), Esc
            // cancels that instead of quitting / blurring chat.
            if app.tab == app::Tab::Settings
                && (app.settings_edit.is_some() || app.settings_pending.is_some())
            {
                app.handle_tab_key(key);
            } else if app.chat.sending {
                app.chat.request_abort();
            } else if chat_capturing {
                // Only blur the chat input when the Chat tab is the one focused;
                // on other tabs Esc quits as before.
                app.chat.blur_input();
            } else {
                return true;
            }
        }
        KeyCode::Tab => app.next_tab(),
        KeyCode::BackTab => app.prev_tab(),
        KeyCode::Char('r') if !chat_capturing => app.reload(),
        KeyCode::Char('e') if app.tab == app::Tab::Settings => app.set_settings_easy(true),
        KeyCode::Char('a') if app.tab == app::Tab::Settings => app.set_settings_easy(false),
        // `?` opens the keybinding help overlay (unless the chat input is
        // capturing text, where `?` is a literal character).
        KeyCode::Char('?') if !chat_capturing => app.show_help = true,
        _ => app.handle_tab_key(key),
    }

    false
}

/// Mouse handling: click a tab in the header to switch to it; scroll the wheel
/// to move within the active tab's list. All other mouse events are ignored.
/// A click or scroll also dismisses an open help overlay.
fn handle_mouse(app: &mut App, mouse: MouseEvent) {
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            if app.show_help {
                app.show_help = false;
                return;
            }
            // Tab-bar click: map the clicked column to a tab via the hit regions
            // the renderer recorded this frame. Leaving the Settings tab discards
            // an armed reset-all confirm so it can't linger (and re-arm itself
            // when the user returns) after the modal key-guard no longer applies.
            if let Some(tab) = app.tab_at(mouse.column, mouse.row) {
                if tab != app::Tab::Settings {
                    // Leaving Settings discards an armed reset-all confirm and an
                    // uncommitted enum preview (neither should linger off-tab).
                    app.confirm_reset_all = false;
                    app.settings_pending = None;
                }
                app.tab = tab;
            }
        }
        MouseEventKind::ScrollDown => {
            if app.show_help {
                app.show_help = false;
                return;
            }
            app.handle_tab_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        }
        MouseEventKind::ScrollUp => {
            if app.show_help {
                app.show_help = false;
                return;
            }
            app.handle_tab_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app::Tab;

    fn test_app() -> App {
        App::load().expect("App::load")
    }
    fn k(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }
    fn mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        // High bytes (0x80-0xFF) — the `& 63` masking keeps table indexing valid.
        assert_eq!(base64_encode(&[0xFF, 0x00, 0x80]), "/wCA");
        // Multibyte UTF-8 round-trips through the byte encoder.
        assert_eq!(base64_encode("é".as_bytes()), "w6k=");
    }

    #[test]
    fn osc52_frame_shape() {
        assert_eq!(osc52_sequence("foo"), "\x1b]52;c;Zm9v\x07");
    }

    #[test]
    fn question_mark_opens_help_then_any_key_closes_it() {
        let mut app = test_app();
        app.tab = Tab::Home;
        assert!(!app.show_help);
        let quit = handle_key(&mut app, k(KeyCode::Char('?')));
        assert!(app.show_help, "? opens the help overlay");
        assert!(!quit);
        // Any subsequent key closes it, and must not quit or do anything else.
        let quit = handle_key(&mut app, k(KeyCode::Char('j')));
        assert!(!app.show_help, "any key closes the help overlay");
        assert!(!quit, "dismissing help must not quit");
    }

    #[test]
    fn left_click_on_tab_region_switches_tab() {
        let mut app = test_app();
        app.tab = Tab::Home;
        app.tab_row_y = 3;
        app.tab_hitboxes = vec![(0, 6, Tab::Home), (9, 20, Tab::Dashboard)];
        handle_mouse(
            &mut app,
            mouse(MouseEventKind::Down(MouseButton::Left), 12, 3),
        );
        assert_eq!(app.tab, Tab::Dashboard);
    }

    #[test]
    fn click_off_the_tab_bar_does_not_switch() {
        let mut app = test_app();
        app.tab = Tab::Home;
        app.tab_row_y = 3;
        app.tab_hitboxes = vec![(0, 6, Tab::Home), (9, 20, Tab::Dashboard)];
        handle_mouse(
            &mut app,
            mouse(MouseEventKind::Down(MouseButton::Left), 12, 10),
        );
        assert_eq!(
            app.tab,
            Tab::Home,
            "a click off the tab row changes nothing"
        );
    }

    #[test]
    fn scroll_dismisses_help_overlay() {
        let mut app = test_app();
        app.show_help = true;
        handle_mouse(&mut app, mouse(MouseEventKind::ScrollDown, 0, 0));
        assert!(!app.show_help);
    }

    #[test]
    fn scroll_moves_list_selection() {
        // The wheel must actually move the active list (its real purpose), not
        // just dismiss the overlay. Settings easy-mode has a fixed row set, so
        // the selection movement is deterministic.
        let mut app = test_app();
        app.tab = Tab::Settings;
        app.settings_easy = true;
        app.settings_selected = 0;
        handle_mouse(&mut app, mouse(MouseEventKind::ScrollDown, 0, 0));
        assert_eq!(
            app.settings_selected, 1,
            "scroll down advances the selection"
        );
        handle_mouse(&mut app, mouse(MouseEventKind::ScrollUp, 0, 0));
        assert_eq!(app.settings_selected, 0, "scroll up retreats it");
    }
}
