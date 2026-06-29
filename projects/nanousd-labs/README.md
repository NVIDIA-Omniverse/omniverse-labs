# nanousd-labs

*Specs are durable. Agents are elastic.*

**nanousd-labs** is an experimental fleet — a mid-flight exploration of how far agents can take generating spec-faithful USD implementations and stacks from the [USD Core Specification](https://github.com/aousd/specifications-public/tree/main/core). The way of working is the point — the method is the deliverable; the implementations are where it's exercised and refined. This directory is the fleet's umbrella and front door: it ties the fleet together and tells you where to start.

The spec is the durable, shared reference; implementations and stacks are elastic outputs you can regenerate against your own constraints.

Your existing OpenUSD stack keeps working untouched — use what fits, file what doesn't.

> **Experimental.** Pre-1.0, no stability or support guarantees. Each component's own README carries its current status and gaps.

## Start here: the skill graph

**[`usd-developer-skillgraph`](usd-developer-skillgraph/) is the heart of the fleet** — the library of skills, prompts, contracts, and goldens, pinned to the USD Core Spec, that an agent walks to generate USD library code in a target language. Everything else in the fleet is where that method gets exercised; the skill graph is the durable deliverable.

For what the graph generates today and how the method actually works, see its **[README](usd-developer-skillgraph/)** and **[tutorial](usd-developer-skillgraph/docs/tutorial/)** — this umbrella points there rather than restating it, so the detail stays in one place and doesn't drift.

Agents take the mechanical spec-to-implementation work; humans own performance, tradeoffs, and design decisions.

**Today vs. where this is going.** The flagship [`nanousd`](nanousd/) and the renderers were built directly with agents (not regenerated from the graph) and are the most complete part of the fleet. Driving those implementations *from* the skill graph — regenerating them from the spec — is the experiment, and it's earlier along. What the graph generates today versus what's still defined-but-not-generated is laid out honestly in **[Where we are](STATE-OF-THE-FLEET.md)**, refreshed roughly monthly.

## The fleet

Each is its own component. The fleet is a **stack** — take as little or as much of its height as you need; each layer builds on the ones beneath it:

```
          nanousdview          ·  interactive viewer
    vulkan / opengl / metal    ·  renderer backends (headless)
        nanousd-python         ·  Python bindings
           nanousd             ·  core C ABI — the foundation
  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄   not wired yet
  usd-developer-skillgraph     ·  the generator
```

Take just the core, climb up through a renderer, or run the full viewer — each level includes what's beneath it. Beneath the runnable stack sits the skill graph: the **generator** we're working to make the driver for the whole stack above — see its [README](usd-developer-skillgraph/) for what it generates today and [Where we are](STATE-OF-THE-FLEET.md) for the frontier. Each component's own README carries its current status, architecture, and gaps.

**Heart — the method**

| Component | Role |
|---|---|
| [`usd-developer-skillgraph`](usd-developer-skillgraph/) | The skill graph that generates USD code from the USD Core Spec. **Start here.** |

**Flagship & bindings**

| Component | Role |
|---|---|
| [`nanousd`](nanousd/) | The flagship — a portable C-ABI implementation of the Core Spec. |
| [`nanousd-python`](nanousd-python/) | Python bindings to the nanousd C ABI. |

**Stacks & integrations**

Tools generated around the implementation, plus external projects integrated with it. The renderer backends load USD through `nanousd` and run headless — `nanousdview` is the interactive shell that drives them (on their own, the renderers produce image captures).

| Project | Role |
|---|---|
| [`nanousdview`](nanousdview/) | Backend-agnostic viewer / stage inspection over the C ABI. |
| [`nanousd-vulkan-renderer`](nanousd-vulkan-renderer/) | Vulkan renderer backend. |
| [`nanousd-metal-renderer`](nanousd-metal-renderer/) | Metal renderer backend (Apple platforms). |
| [`nanousd-opengl-renderer`](nanousd-opengl-renderer/) | OpenGL renderer backend. |
| **Newton** | Physics engine; runs on nanousd via the `pxr_compat` shim — public packaging to come. |
| **MuJoCo** | Physics engine; runs on nanousd via the `pxr_compat` shim — public packaging to come. |

## Renderer comparison

See [`comparisons/README.md`](comparisons/README.md) for the current OVRTX,
Vulkan RT, Vulkan Raster, and OpenGL GLES visual comparison gallery. It includes
highlight, scene-by-scene, and full overview images; subtract/diff panels are
not used.

See [`comparisons/performance/README.md`](comparisons/performance/README.md)
for the separate nanousdview-based renderer performance comparison across the
full Isaac Sim warehouse and NVIDIA DSX datacenter scenes.

## Contributing

These components are published together as one experiment under Omniverse Labs, so the fleet isn't set up to take inbound pull requests against these directories.

The invitation is to the **way of working**. Highest-leverage ways to engage, and where each lands:

- **Skills, contracts, and goldens** → into [`usd-developer-skillgraph`](usd-developer-skillgraph/); they improve how code is generated and define what "correct" means.
- **Spec-ambiguity reports** → filed as issues on the [AOUSD specifications-public repo](https://github.com/aousd/specifications-public); a fix to the spec benefits every implementation.
- **Your own implementations and stacks** → generated from the skill graph in your repos; the skills you write to get there are what come back here.

Adopting the generated code is welcome, but the activation is for people who want to write skills, generate their own implementations, and build new stacks.

## License

Dual license: [Apache 2.0](../../LICENSE) for source code and scripts; [CC-BY-4.0](../../LICENSE) for skill instructions, documentation, and other non-software content. Both license texts are in [LICENSE](../../LICENSE).

---

*Experimental work under [`omniverse-labs`](../..) — samples and ideas, not a product.*
