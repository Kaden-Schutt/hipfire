# DS4 MI300X agentmaxx OMP profile

This project-local profile defines a 34-agent topology: one Opus main conductor from
`SYSTEM.md` plus 33 explicitly named task agents. A separate `task.md` guard shadows
generic unnamed delegation. It does not modify `~/.omp/agent`.

Start OMP from this worktree with the task overlay:

```bash
cd /home/kaden/ClaudeCode/autorocm/hipfire/.claude/worktrees/ds4-mi300x-agentmaxx
omp --config .omp/agentmaxx.yml --no-prewalk @.omp/GOAL.md @.omp/KICKOFF.md
```

The main model resolves from `.omp/config.yml` to Claude Opus 5 at `high`. Agent files
use direct provider/model selectors rather than shared aliases. The source worktree's
dirty gfx942 prototypes are intentionally not copied here; see `DS4-INVARIANTS.md`.

The initial read-only waves may run concurrently. Composer writers are serialized by
default, and `ds4-mi300x-operator` exclusively owns remote GPU command execution.
The task overlay disables the generic user-level agents, including the Kimi-max
backend profile, so this campaign can dispatch only the project `ds4-*` roster or the
unnamed-task guard.

The overlay enables OMP goal continuation, async jobs, compaction continuation, and
unattended approvals for this isolated campaign. `GOAL.md` is the authoritative
acceptance contract; `KICKOFF.md` is the one-time operating instruction. The safety
boundary remains `.omp/DS4-INVARIANTS.md`: no destructive actions, no push/upload,
and no mutation of the protected source worktree.
