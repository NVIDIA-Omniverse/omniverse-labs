---
name: legacy-retirement-clean-history
description: Identify and remove obsolete legacy, fallback, staging, compatibility, duplicate, generated, and reference code before creating clean initial commits or new project history. Use when a repo is being reset, consolidated, or prepared for a cleaner public history.
---

# Legacy Retirement Clean History

## Purpose

Create a clean codebase that starts from the current intended architecture, not from the migration path that produced it.

## Retirement Workflow

1. Define the supported present.
   - Current default binaries.
   - Current package entrypoints.
   - Current public contracts.
   - Current supported platforms/backends.
   - Current test and smoke commands.

2. Search for transition language.
   - `legacy`, `fallback`, `old`, `thin`, `new`, `staging`, `phase`, `temporary`, `for reference`, `compat`, `deprecated`.
   - Treat matches as leads, not proof.

3. Classify each path.
   - **Keep:** still default, tested, documented as supported, or unique capability.
   - **Rename:** concept is current but name reveals migration history.
   - **Delete:** duplicate implementation, unused reference copy, old fallback, staging package, dead test harness, old generated/native extension.
   - **Defer:** unclear ownership or untested deletion risk.

4. Remove in dependency order.
   - Remove docs and scripts that advertise dead paths.
   - Remove build targets and fallback dispatch.
   - Remove source files and assets only used by removed targets.
   - Remove tests/tools that validate old architecture.
   - Clean names so current paths do not carry transition labels.

5. Prove absence.
   - Clean build at least once.
   - Search for deleted target names and transition terms.
   - Run current smoke paths.
   - Inspect produced binaries/packages to ensure old outputs are gone.

## Deletion Evidence

A deletion is usually safe when all are true:

- No current build target references it.
- No current runtime path imports or executes it.
- Its behavior is owned elsewhere.
- The replacement path has passed smoke verification.
- Docs can describe the system without mentioning it.

Be cautious when:

- It is a public API surface.
- It supports a deployed version range.
- It contains test fixtures still useful for the new path.
- It encodes platform-specific knowledge not yet moved elsewhere.

## Clean-History Rules

- Prefer deleting migration scaffolding before creating the first clean commit.
- Prefer current names over transitional names: `viewer`, not `viewer_thin`; `renderer`, not `renderer_new`.
- Keep compatibility aliases only when external users need them.
- Move useful fixtures into the owning repo before deleting old harnesses.
- Update docs to steady-state architecture.

## Git Discipline

- Inspect dirty state before deletion.
- Do not revert unrelated user changes.
- Stage and commit per repo/submodule when possible.
- For submodules, prepare each clean initial commit independently.
- If history will be rewritten remotely, coordinate branch replacement deliberately; local cleanup is separate from remote policy.

## Post-Cleanup Report

Include:

- Deleted groups by purpose.
- Renamed current paths.
- Current build/run entrypoints.
- Verification commands.
- Remaining intentional compatibility or future-work placeholders.
