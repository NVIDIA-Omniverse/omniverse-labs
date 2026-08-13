# Dev Variant Presenter — project instructions

**Read [AGENTS.md](AGENTS.md).** It is the single source of truth for how to work in this repo:
architecture in one paragraph, how to start and stop the server, fresh-clone setup, and tests.

This file exists only so Claude Code auto-loads something at session start. Keeping the
instructions in one file means there is one copy to keep correct instead of three.

Two things are worth having in front of you before you touch anything:

- **Start the server ONLY via the watchdog** (`run_server.ps1` on Windows, `run_server.sh` on
  Linux/macOS). Never bare `python -m dev_variant_presenter`, never a generic preview/dev-server
  launcher. Those skip the watchdog, and a routine ovrtx/ovstream native crash then becomes a dead
  app with no relaunch and no logs.
- **The stream is single-client.** Opening a second browser or client steals the active session.

The `.claude/skills/` skills (`start-server`, `setup-environment`) carry the full procedures and
load automatically.
