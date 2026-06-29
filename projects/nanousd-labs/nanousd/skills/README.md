# Skills

> **This is *not* the skill graph.** The skill graph — the method that *generates*
> USD implementations from the standard — is a separate repo,
> [`usd-developer-skillgraph`](../../usd-developer-skillgraph)
> (the [fleet README](../..) explains it). This
> folder holds one **repo-local agent skill** that lives alongside `nanousd`,
> separate from that method.

A skill is a self-contained, model-readable procedure — a `SKILL.md` (YAML
front-matter naming it and describing when to use it) plus any supporting
references, scripts, and tests. An agent loads it on demand when its description
matches the task at hand.

> **For newcomers:** you do **not** need it to build, test, or use `nanousd` —
> see the top-level [`README.md`](../README.md). It isn't a runtime dependency of
> the library, and it isn't how the library is generated (`nanousd` was built directly).

## What's here

| Skill | Kind | What it is |
|---|---|---|
| [`sketch-stage`](sketch-stage/SKILL.md) | **LLM-driven scene-authoring tool** | An LLM-driven USD scene-authoring app — generate or densify a stage from sketch inputs (bounds / scale / asset pack / intent), with collision-gated incremental placement and a live 2D+3D browser view. Its USD I/O currently goes through **OpenUSD (pxr / `usd-core`)**, not `nanousd`; its **output** (large prim-count stages) is used to **benchmark `nanousd`**. |

## Status

`sketch-stage` is a tool we found useful while building the fleet and are making
available now; it will likely move to its own home as things settle. Treat the
copy here as the working version for now.
