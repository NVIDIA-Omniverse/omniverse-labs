# Agentic Skills Catalog

Repo-local Codex-style skills distilled from the nanousd module refactor.

Operational mindset for this renderer goal: no subsystem, script, metric, or
existing pass layout is sacred. Prefer small verification artifacts, but do not
confuse that with small implementation scope. When repeated bounded slices
preserve the same wrong image, frame time, or residual structure, the catalog
expects an evidence-backed architecture pivot or subsystem replacement.

## Start here: building a renderer from skills alone

These skills now carry the *connective tissue*, not only judgment, so a renderer can
be built without a reference codebase to copy from. Read `renderer-implementation-roadmap`
first — it sequences every other skill into a gated build order (contracts before
backends, one vertical slice before breadth) and ships the reduced-fixture corpus.

Two rules thread through all of them:

- **Contracts before backends.** Freeze the public capability ABI, the RHI seam, and
  the scene-extraction structs once, as a single canonical copy each, before standing
  up any GPU backend (`renderer-capability-abi`, `contract-first-ffi`,
  `nanousd-renderer-scene-extraction`). Skip it and the second backend forces a core
  rewrite.
- **One source of truth per contract.** A "shared" file hand-copied across repos
  drifts — point at the canonical artifact, never transcribe its values, and enforce
  it with a machine check (e.g. `tools/check_renderer_abi.py`).

| Skill | Use When |
|---|---|
| `agentic-codebase-cartography` | Map unfamiliar multi-repo systems, APIs, ownership, duplication, and refactor risk before editing. |
| `layered-boundary-refactor` | Move duplicated behavior behind canonical contracts and thin adapters. |
| `contract-first-ffi` | Keep native ABIs and Python/native bindings in exact sync. |
| `portable-build-runtime-paths` | Make native builds, CLIs, viewers, plugins, shaders, and packaged artifacts run after relocation without `cwd`, username, or absolute build-path assumptions. |
| `nanousd-renderer-scene-extraction` | Load USD files through NanoUSD and turn composed prims into renderer geometry, material, light, camera, transform, texture, and instancing representations. |
| `graphics-debugging-lab` | Debug renderer, shader, material, camera, lighting, and viewer visual defects with captures. |
| `physically-based-renderer-validation` | Validate PBR/path-traced/ray-traced behavior against reference renderer output, AOVs, USD semantics, transport invariants, and reduced physical fixtures. |
| `renderer-image-comparison-testing` | Build and run golden-image harnesses for renderer/backend correctness and parity. |
| `renderer-stalled-parity-triage` | Reset renderer/reference parity work when beauty images, scalar metrics, repeated local tweaks, or narrow slices stop improving; escalate to no-sacred-cows pivots when evidence warrants. |
| `renderer-target-architecture-design` | Design renderer features from the target-quality architecture first, including radical subsystem replacement when incremental slices are exhausted. |
| `renderer-feature-validation-matrix` | Track renderer feature status, fixtures, references, pass criteria, backend support, owners, and residual risk before delegating or closing milestones. |
| `renderer-performance-architecture` | Design and review scalable renderer architecture for large USD scenes, including instancing, batching, streaming, AS, textures, pass gating, telemetry, and replacement of dead-end paths. |
| `verification-led-development` | Choose and run the smallest meaningful builds, tests, smoke runs, logs, and screenshots. |
| `legacy-retirement-clean-history` | Delete old fallback, staging, compatibility, and duplicate paths before clean initial commits. |
| `ai-agent-orchestration` | Coordinate explorer/worker/reviewer agents without overlap or integration drift. |
| `renderer-implementation-roadmap` | Sequence a renderer build from nothing using only these skills — the contracts-first phase order, gates, and reduced-fixture corpus. **Read first.** |
| `renderer-backend-port` | Port a renderer to a new GPU API (Vulkan→Metal/DX12/WebGPU) behind the shared seam; decide share vs reimplement vs stub; cross-API pitfalls. |
| `renderer-capability-abi` | Govern the shared renderer C ABI + backend capability contract so backends stay interchangeable and parity is machine-checked. |
| `renderer-realtime-sensor` | Design real-time, headless, multi-camera renderers that produce reproducible, AOV-labeled sensor data for simulation/RL — naming the fidelity/throughput/determinism target rather than assuming throughput wins. |
| `renderer-gpu-interop` | Share renderer output zero-copy across GPU APIs/runtimes (Vulkan↔CUDA, Metal/MPS, DLPack) and own fd ownership, cross-API sync, double-buffering, and teardown order across that boundary. |
| `renderer-simulation-loop` | Drive a renderer from a physics/RL loop — per-step updates through narrow dirty paths, time handling, frame pacing — avoiding the transpose-zeroes-translation origin collapse. |
| `renderer-shader-host-layout-sync` | Keep host structs and the shaders that read them byte-for-byte in sync (sizeof/offsetof asserts, one-file bindings, shared push-constant blocks) so layout drift fails the build, not the image. |
| `renderer-constrained-backend` | Build a backend to a hard memory/capability budget (low-VRAM, raster-only, portable) — texture-budget clamp as invariant, MaterialX baking vs codegen, and degradation that matches the reference. |

Each skill lives in `agentic-skills/<name>/SKILL.md` with standard frontmatter. Codex consumes it via the per-skill `agents/openai.yaml`; Claude Code discovers it through the `.claude/skills/` symlinks at the repo root — one source of truth, no duplicated prose.
