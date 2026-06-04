# Omniverse Projects

> **Experimental work in progress.** These projects are early-stage explorations
> that may evolve into standalone Omniverse libraries, developer tools, or
> collaborative workstreams with NVIDIA partners. They are not polished samples
> or demos — expect rough edges, breaking changes, and active iteration.

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![OpenUSD](https://img.shields.io/badge/OpenUSD-25.x-76B900?logo=nvidia)](https://openusd.org)
[![Status](https://img.shields.io/badge/Status-Experimental-orange)]()

---

## What's Here

The `projects/` folder contains experimental Omniverse work that doesn't fit
neatly into the [`samples/`](../samples/README.md) folder. Unlike samples —
which demonstrate finished capabilities — projects are living codebases at
various stages of maturity:

| Stage | What it means |
|---|---|
| **Exploration** | Proof-of-concept; APIs and structure may change dramatically |
| **Incubation** | Core functionality stabilizing; ready for early partner feedback |
| **Pre-release** | Targeting a library or SDK release; API is hardening |

If a project graduates to a standalone library or pip package, it will be
linked here and archived.

---

## Projects

| Project | Description | Stage | Docs |
| :--- | :--- | :---: | :---: |
| [omniverse-usd-performance-tuning](./omniverse-usd-performance-tuning/) | Skills for optimizing large or complex USD scenes — LOD, culling, instancing, and scene graph pruning | Incubation | [↗](./omniverse-usd-performance-tuning/README.md) |
| [omniverse-cad-to-simready](./omniverse-cad-to-simready/) | End-to-end pipeline skills for converting CAD geometry to SimReady USD assets | Incubation | [↗](./omniverse-cad-to-simready/README.md) |
| [omniverse-realtime-viewer](./omniverse-realtime-viewer/) | Reference viewer built on `ovrtx`, `ovui`, and `ovstream` — intended as a starting point for partner-built viewers | Incubation | [↗](./omniverse-realtime-viewer/README.md) |
| [physical-ai-skill-hub](./physical-ai-skill-hub/) | Repository of Physical AI recipes and skills for sim-ready robotics and industrial workflows | Exploration | — | [↗](./physical-ai-skill-hub/README.md) |

---

## How This Differs from Samples

| | `samples/` | `projects/` |
|---|---|---|
| **Purpose** | Demonstrate finished Omniverse features | Develop future libraries and partner capabilities |
| **Stability** | Stable — tied to a release | Experimental — may break between commits |
| **Audience** | Developers learning Omniverse | Partners co-developing with NVIDIA |
| **Lifecycle** | Published alongside a library release | May graduate to a standalone library or be deprecated |
| **Support** | Supported | Best-effort |

---

## Prerequisites

Requirements vary per project — check each project's own `README.md` for
specifics. Common dependencies:

- **Python 3.10+**
- **OpenUSD 25.x** — install via `pip install usd-core` or build from
  [source](https://github.com/PixarAnimationStudios/OpenUSD)
- **Omniverse Kit SDK** *(select projects only)* —
  [developer.nvidia.com/omniverse](https://developer.nvidia.com/omniverse)
- Relevant Omniverse libraries (`ovrtx`, `ovui`, `ovstream`, etc.) where noted
  in the project README

---

## Getting Started

Clone the repo and navigate to the project you want to explore:

```bash
git clone https://github.com/NVIDIA-dev/omniverse-labs.git
cd omniverse-labs/projects/<project-name>
pip install -r requirements.txt   # if present
