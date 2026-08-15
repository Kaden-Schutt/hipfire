// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Named conversation sessions persisted to `~/.hipfire/chat_history.json`, so a
//! chat can be saved, reloaded, listed, and deleted across runs.
//!
//! A missing or corrupt file degrades to an empty history, never an error.
//! Writes are atomic (temp file + rename), mirroring the config / ui_state
//! writers. The file holds a small list of `{name, messages}` records.

use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

use super::chat::ChatMessage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub name: String,
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatHistory {
    pub sessions: Vec<ChatSession>,
}

impl ChatHistory {
    pub fn load(path: &Path) -> Self {
        fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let body = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("chat_history.json");
        // pid-tag the temp so two instances sharing ~/.hipfire never collide on
        // the temp write before the (atomic) rename.
        let tmp = path.with_file_name(format!("{name}.{}.tmp", std::process::id()));
        fs::write(&tmp, body.as_bytes())?;
        fs::rename(&tmp, path)
    }

    /// Insert or replace the session named `name`.
    pub fn upsert(&mut self, name: &str, messages: Vec<ChatMessage>) {
        if let Some(s) = self.sessions.iter_mut().find(|s| s.name == name) {
            s.messages = messages;
        } else {
            self.sessions.push(ChatSession {
                name: name.to_string(),
                messages,
            });
        }
    }

    pub fn get(&self, name: &str) -> Option<&[ChatMessage]> {
        self.sessions
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.messages.as_slice())
    }

    /// Remove the named session; returns whether it existed.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.sessions.len();
        self.sessions.retain(|s| s.name != name);
        self.sessions.len() != before
    }

    pub fn names(&self) -> Vec<&str> {
        self.sessions.iter().map(|s| s.name.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
        }
    }

    #[test]
    fn upsert_get_remove_round_trip() {
        let dir = std::env::temp_dir().join(format!("hipfire-chathist-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let p = dir.join("chat_history.json");

        let mut h = ChatHistory::default();
        h.upsert("debug", vec![msg("user", "hi"), msg("assistant", "hello")]);
        h.upsert("debug", vec![msg("user", "again")]); // replace, not duplicate
        assert_eq!(h.sessions.len(), 1);
        assert_eq!(h.get("debug").unwrap().len(), 1);

        h.upsert("other", vec![msg("user", "x")]);
        h.save(&p).unwrap();
        let back = ChatHistory::load(&p);
        assert_eq!(back.names(), vec!["debug", "other"]);
        assert_eq!(back.get("debug").unwrap()[0].content, "again");

        let mut back = back;
        assert!(back.remove("debug"));
        assert!(!back.remove("debug"));
        assert_eq!(back.names(), vec!["other"]);

        let tmp = p.with_file_name(format!("chat_history.json.{}.tmp", std::process::id()));
        assert!(!tmp.exists(), "temp file is renamed away, not left behind");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_file_is_empty_not_error() {
        let dir = std::env::temp_dir().join(format!("hipfire-chathist-bad-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let p = dir.join("chat_history.json");
        fs::write(&p, b"{not json").unwrap();
        assert!(ChatHistory::load(&p).sessions.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_and_empty_object_degrade_to_empty() {
        // The documented degrade-to-empty contract on both failure modes:
        let dir = std::env::temp_dir().join(format!("hipfire-chathist-m-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        // (a) nonexistent path -> read error -> empty
        let missing = dir.join("nope.json");
        let _ = fs::remove_file(&missing);
        assert!(ChatHistory::load(&missing).sessions.is_empty());
        // (b) valid JSON missing the sessions key -> serde(default) -> empty
        let empty = dir.join("empty.json");
        fs::write(&empty, b"{}").unwrap();
        assert!(ChatHistory::load(&empty).sessions.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
