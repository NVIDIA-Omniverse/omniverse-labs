# nanousdview

> **Part of [`nanousd-labs`](..)** — an experimental fleet that generates USD implementations and stacks from the [USD Core Specification](https://github.com/aousd/specifications-public/tree/main/core). This component is the **backend-agnostic viewer / stage-inspection shell** over the `nanousd` C ABI. The [fleet README](..) has the full picture — the stack, how the repos fit together, and the skillgraph that drives it.

Backend-agnostic USD viewer over the OVRTX Python API. The viewer always talks
to `ovrtx.Renderer`; `--backend` only selects which implementation provides the
`ovrtx` module.

**Backend matrix.** What's available depends on platform:

| Platform | Available backends | Default |
|---|---|---|
| Linux   | `vulkan`, `opengl`, `ovrtx`  | `vulkan` |
| macOS   | `metal`,  `opengl`  | `metal`  |
| Windows | `vulkan`, `opengl` — under active development | — |

`vulkan`, `opengl`, and `metal` use the nanousd-backed `ovrtx` facade from
`nanousd-vulkan-renderer/python` with `NANOUSD_OVRTX_BACKEND` set to the
selected implementation. `ovrtx` routes through NVIDIA's OVRTX wheel directly.

**Windows:** Vulkan and OpenGL support is under active development; Metal is not available on Windows (Apple platforms only).

**Status:** active viewer shell. `run.sh` opens a default scene through the selected backend.

## Quick start

**Runtime prerequisites:** the viewer is a Python/Qt shell, so `run.sh` needs a Python venv with **`numpy`** + **`PySide6`** and the **[`nanousd-python`](../nanousd-python)** package installed (the viewer imports it; a missing-package error names it). For **headless/CI**, set **`QT_QPA_PLATFORM=offscreen`** or Qt aborts on the missing `xcb` platform plugin.

```bash
./build.sh     # clean release build
./run.sh       # opens the viewer on a default test scene (platform default backend)
./run.sh --backend opengl                     # OpenGL (Linux + macOS)
./run.sh --backend vulkan                     # Vulkan (Linux only)
./run.sh --backend metal                      # Metal  (macOS only)
./run.sh --backend ovrtx                      # OVRTX wheel (Linux only)

# headless screenshot (no display):
QT_QPA_PLATFORM=offscreen ./run.sh --backend opengl --screenshot out.ppm
```

Both scripts accept `--help`. The same `build.sh`/`run.sh` shape is used across the fleet repos — `cd` into any repo, run those two commands.

## Architecture

```
┌────────────────────────────────────┐
│  nanousdview                       │
│   --backend=vulkan|opengl|metal|ovrtx│
└─────────────┬──────────────────────┘
              │  ovrtx.Renderer / RenderProduct / RenderVarOutput
   ┌──────────┴──────────────┐
   ▼                         ▼
nanousd ovrtx facade      NVIDIA ovrtx wheel
   │
   ▼
vulkan / opengl / metal renderer implementation
```

Window, input, CLI, and UI overlays live here. Rendering is expressed as an
OVRTX render product and read back by mapping `LdrColor`; the viewer no longer
imports `nusd_renderer*` Python modules or calls `nu_*` renderer methods.

## Sibling repos

| Repo | Role |
|---|---|
| [`nanousd-vulkan-renderer`](../nanousd-vulkan-renderer) | Owns the nanousd OVRTX facade and Vulkan implementation |
| [`nanousd-opengl-renderer`](../nanousd-opengl-renderer) | OpenGL implementation selected through the OVRTX facade |
| [`nanousd-metal-renderer`](../nanousd-metal-renderer) | Metal implementation selected through the OVRTX facade |

## License

Apache-2.0.

## Contributing

This project is currently not accepting contributions.
