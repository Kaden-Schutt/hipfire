use crate::plan::ReapPlan;

/// Arch-specific REAP extras. Default impls are no-ops so arches that need
/// nothing (qwen35, lfm2moe, minimax) don't implement anything.
pub trait ReapArchHook {
    /// Path to an arch sidecar file inside the plan dir, if the arch uses one.
    fn sidecar_path(&self, plan: &ReapPlan, name: &str) -> std::path::PathBuf {
        plan.dir.join(name)
    }
    /// Whether this layer's auxiliary head (e.g. ds4 MTP) is skipped under reap.
    fn skip_aux_head(&self, _plan: &ReapPlan, _layer: usize) -> bool {
        false
    }
}
