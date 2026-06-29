# MCP setup — register the placement server with Claude Code

The MCP server is `engine/mcp_server.py` — a `FastMCP` stdio server. The
`command` field in `mcpServers` config must be a single executable path
(Claude Code spawns the process directly; shell metacharacters in
`command` are not evaluated). Use the venv's python directly.

Run `scripts/setup_venv.sh` once if the venv at
`~/.cache/sketch-stage/venv` doesn't exist yet — it's idempotent.

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "sketch-placement": {
      "command": "$HOME/.cache/sketch-stage/venv/bin/python",
      "args": [
        "<sketch-stage-skill-dir>/engine/mcp_server.py"
      ]
    }
  }
}
```

**No `env` block needed** — paths are passed *per task* via the
`open_session` tool, so one MCP registration serves every project /
scenario you ever work on. The previous design baked the anchor + pack +
out into env vars, which meant one MCP registration ↔ one fixed
workspace; that limitation is gone.

The LLM opens a session at the start of each task:

```
mcp__sketch-placement__open_session(
    anchor = "/path/to/anchor.sketch.json",   # or null for empty canvas
    pack   = "/path/to/asset_pack",
    out    = "/path/to/output_dir"
)
```

Returns `{ok, anchor, pack, out, placementCount, zoneCount,
archetypeCount, surfaceCount}`. After that, every stateful tool
(`query_stage_graph`, `query_surfaces`, `place`, `place_on`, …) operates
on the open session until the LLM either calls `open_session(...)` again
(replacing it) or `close_session()` (shutting it down).

The pack root may be either a flat pack or a manifest (multi-theme) pack;
the session loads either transparently (see `scripts/pack_loader.py`).

### Optional: legacy env-var defaults

If you genuinely want a "default workspace" so the LLM doesn't have to
call `open_session` first, you can still set env vars in the config and
the first stateful tool call will auto-open from those:

```json
{
  "mcpServers": {
    "sketch-placement": {
      "command": "$HOME/.cache/sketch-stage/venv/bin/python",
      "args": ["<sketch-stage-skill-dir>/engine/mcp_server.py"],
      "env": {
        "SKETCH_ANCHOR":  "<default-anchor>",
        "SKETCH_PACK":    "<default-pack>",
        "SKETCH_OUT_DIR": "<default-out>"
      }
    }
  }
}
```

Explicit `open_session(...)` always wins over env-var defaults.

Or with the CLI:

```bash
claude mcp add sketch-placement \
    --command $HOME/.cache/sketch-stage/venv/bin/python \
    --args <sketch-stage-skill-dir>/engine/mcp_server.py
```

Restart Claude Code. Tools then appear in the LLM's tool list as
`mcp__sketch-placement__open_session`,
`mcp__sketch-placement__query_surfaces`,
`mcp__sketch-placement__place_on`,
`mcp__sketch-placement__query_zones`,
`mcp__sketch-placement__place`, etc.

## Without MCP (this session, or any environment where you can't restart)

The same tools are exposed as HTTP endpoints (`POST
http://127.0.0.1:8766/tool/<name>`). Run `scripts/run.sh` to boot the HTTP
shim + the live viz, then drive it via `Bash` → `curl`. Same JSON shapes
as the MCP tools. This is what the `/sketch-stage` skill prompt actually
does internally — the LLM in the skill talks HTTP, not MCP.

## State persistence

The session is **in-memory**. State persists for as long as the HTTP / MCP server process is alive. The on-disk `events.jsonl` is the canonical history — replayable to reconstruct the in-memory state if the server restarts.
