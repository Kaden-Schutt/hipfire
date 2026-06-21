// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::time::Duration;

use hipfire_operator::training::{
    list_training_runs, load_training_run_detail, TrainingRunDetail, TrainingRunList,
};

use super::{config::ConfigState, HipfirePaths};

#[derive(Clone, Debug)]
pub struct TrainingState {
    pub list: TrainingRunList,
    pub detail: Option<TrainingRunDetail>,
    pub selected: usize,
    pub source: String,
    pub warning: Option<String>,
}

impl TrainingState {
    pub fn load(paths: &HipfirePaths, config: &ConfigState) -> Self {
        match load_remote(config) {
            Ok(state) => state,
            Err(err) => {
                let list = list_training_runs(&paths.training_runs);
                let detail = list
                    .runs
                    .first()
                    .and_then(|run| load_training_run_detail(&paths.training_runs, &run.id, 80));
                Self {
                    list,
                    detail,
                    selected: 0,
                    source: "local files".into(),
                    warning: Some(format!("operator API unavailable: {err}")),
                }
            }
        }
    }

    pub fn selected_run_id(&self) -> Option<&str> {
        self.list.runs.get(self.selected).map(|run| run.id.as_str())
    }

    pub fn active_count(&self) -> usize {
        self.list.runs.iter().filter(|run| run.is_active()).count()
    }

    pub fn stale_count(&self) -> usize {
        self.list.runs.iter().filter(|run| run.stale).count()
    }

    pub fn select_delta(&mut self, delta: isize, paths: &HipfirePaths, config: &ConfigState) {
        if self.list.runs.is_empty() {
            self.selected = 0;
            self.detail = None;
            return;
        }
        let max = self.list.runs.len() as isize - 1;
        self.selected = (self.selected as isize + delta).clamp(0, max) as usize;
        self.reload_detail(paths, config);
    }

    fn reload_detail(&mut self, paths: &HipfirePaths, config: &ConfigState) {
        let Some(id) = self.selected_run_id().map(str::to_string) else {
            self.detail = None;
            return;
        };
        self.detail = if self.source == "operator API" {
            load_remote_detail(config, &id).ok()
        } else {
            load_training_run_detail(&paths.training_runs, &id, 80)
        };
    }
}

fn load_remote(config: &ConfigState) -> Result<TrainingState, String> {
    let url = format!(
        "http://{}:{}/admin/training/runs",
        config.probe_host(),
        config.port
    );
    let body = agent()
        .get(&url)
        .call()
        .map_err(|err| err.to_string())?
        .into_string()
        .map_err(|err| err.to_string())?;
    let list: TrainingRunList = serde_json::from_str(&body).map_err(|err| err.to_string())?;
    let detail = list
        .runs
        .first()
        .and_then(|run| load_remote_detail(config, &run.id).ok());
    Ok(TrainingState {
        list,
        detail,
        selected: 0,
        source: "operator API".into(),
        warning: None,
    })
}

fn load_remote_detail(config: &ConfigState, run_id: &str) -> Result<TrainingRunDetail, String> {
    let url = format!(
        "http://{}:{}/admin/training/runs/{}",
        config.probe_host(),
        config.port,
        run_id
    );
    let body = agent()
        .get(&url)
        .call()
        .map_err(|err| err.to_string())?
        .into_string()
        .map_err(|err| err.to_string())?;
    serde_json::from_str(&body).map_err(|err| err.to_string())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(650))
        .build()
}
