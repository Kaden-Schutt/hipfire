# AGENTS.md - agent skills

This subtree stores reusable workflows for agents. Keep repo-wide rules in the
root `AGENTS.md`; use this directory for task-specific procedures, references,
and scripts that should be loaded only when relevant.

## Skill Authoring

- Each skill lives in `.agents/skills/<name>/` with a `SKILL.md` file. Optional
  `skill.json`, `scripts/`, `references/`, and `assets/` files should stay in
  that skill directory.
- Before using a skill, read its `SKILL.md` completely. If it points to a
  directly relevant reference file, read that file before acting.
- Prefer updating an existing skill when a workflow is recurring and too long
  for root guidance.
- Do not duplicate the full skill index in root `AGENTS.md`; skill metadata is
  discoverable from the skill directories.
