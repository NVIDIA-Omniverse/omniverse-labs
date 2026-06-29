# ovfmi — sample showcasing declarative FMI/SSP embedded in OpenUSD for physics simulation in digital twins

> **`bash apps/ov-fmi/setup.sh` → `python apps/ov-fmi/main.py <stage>.usda`** · Load FMI behavior models straight from a USD stage, step them with FMPy, render with RTX, and close the loop with rigid-body physics — all from a standalone Python app. Built on the disaggregated ov libraries (**ovrtx** rendering + **ovphysx** physics) plus the open **FMI**, **SSP**, and **OpenUSD** standards.
>
> *Pre-release / internal Omniverse Labs project (ov-fmi v0.1.0; ovrtx 0.3, ovphysx 0.4.9, fmpy 0.3.25). APIs and the USD-FMI schema may change; open-sourcing is on the roadmap.*

```bash
# Conveyor-belt digital twin: ovphysx rollers + package sensor + SSP controller
python apps/ov-fmi/main.py usd/conveyor/ConveyorFMI.usda --up-axis Z
# Minimal example: one FMU drives a sphere's height
python apps/ov-fmi/main.py usd/ov-fmi/fmi_parser_test.usda
```

---

## 1. What is ovfmi?

**ovfmi** is a USD-native integration that adds *behavior* to industrial digital twins by binding **FMI** co-simulation models to OpenUSD scenes through a custom schema. USD is declarative: it describes world state (*the what*), not behavior. FMI + SSP supply the imperative simulation behavior (*the how*). ovfmi joins the two so a single USD layer captures both the 3D scene and its simulation model, readable and composable by any USD tool.

ovfmi exposes a Python runtime (the `ov-fmi` app) plus a USD-FMI schema: you author `FmuInstance` / `SspInstance` prims that reference `.fmu` or `.ssp` archives, declaratively map model variables to USD attributes, and the runtime traverses the stage, instantiates the models via FMPy, and synchronizes data every simulation step. It runs standalone — no full Omniverse Kit install — on the disaggregated ov libraries. Behind the scenes **ovrtx** (NVIDIA RTX renderer and USD stage host) is paired with **ovphysx** (PhysX-based rigid-body simulation), exchanging state through ovphysx's in-memory tensor API rather than slow per-attribute USD I/O — fast enough for real-time.

---

## 2. What functionalities are available, and who are the target users?

**What you can do with it:**

- **Embed FMUs and SSPs in USD declaratively** — `FmuInstance` (single `.fmu`) and `SspInstance` (multi-FMU `.ssp` black box), each with `FmuConnection` → `FmuMapping` children that bind FMI variables to prim attributes.
- **Run FMI 2.0 and FMI 3.0 co-simulation** via the FMPy master, plus **SSP 1.0** archives that package networks of wired FMUs into one file (internal FMUs must be FMI 1.0/2.0 — an FMPy constraint).
- **Close the physics loop** — `physx:position` / `physx:velocity` inputs and `physx:force` (and articulation drive-velocity) outputs route directly to ovphysx tensors; physics auto-enables for prims carrying `PhysicsRigidBodyAPI`.
- **Map to any USD attribute with component selection** — `fmi:usdMapping = (offset, count)` picks scalar, single-component, or ranges of `xformOp:translate`, `omni:xform`, light intensity, or custom attributes.
- **Render and inspect live** — real-time RTX viewer (WASD/mouse navigation), plus `--headless` and `--png` modes for smoke tests and batch capture.
- **Deterministic stepping** — FmuInstances step in USD depth-first traversal order (authoring order sets causality between coupled FMUs); each SSP steps atomically.
- **Ready-made demo scenes** — bouncing ball, PD controller, two-FMU orbit, SSP orbit, and a full conveyor-belt digital twin.

**Who benefits:**

- **Industrial digital-twin engineers** — add controllers, sensors, and actuator behavior to factory scenes built on open standards.
- **Controls & simulation engineers from the FMI/Modelica ecosystem** — drop vendor-neutral FMUs/SSPs exported by 250+ compliant tools (Siemens, Dassault, dSPACE, Bosch, …) into a USD twin for simulations without rewriting them.
- **Robotics & physics developers (ovphysx / Isaac users)** — drive PhysX rigid bodies from FMU control logic through the tensor API for closed-loop simulation.
- **ISVs & integrators** — protect IP by shipping pre-packaged SSP black boxes whose internal wiring stays hidden from USD.

---

## 3. Documentation and reference links

- **Project README & quick start:** <https://github.com/NVIDIA-Omniverse/omniverse-labs-internal/tree/main/projects/ovfmi>
- **USD-FMI Schema Reference:** <https://github.com/NVIDIA-Omniverse/omniverse-labs-internal/blob/main/projects/ovfmi/docs/USD-FMI-SCHEMA.md>
- **Source (GitHub):** <https://github.com/NVIDIA-Omniverse/omniverse-labs-internal/tree/main/projects/ovfmi>
- **FMI standard:** <https://fmi-standard.org/> · **SSP standard:** <https://ssp-standard.org/> · **OpenUSD:** <https://openusd.org/>
- **FMPy (FMI runtime):** <https://github.com/CATIA-Systems/FMPy>
- **Background talk:** GTC 2025 — "Build Physics-Based Digital Twins for Co-Simulation" (S71963, with SoftServe), since extended with SSP support.

---

## 4. System requirements

- Python 3.10 through 3.13 (`requires-python >=3.10,<3.14`).
- Linux (Ubuntu/Debian and equivalents) and Windows (x86_64).
- NVIDIA RTX GPU + recent driver; display/OpenGL for the live viewer. CUDA Toolkit optional (only for CUDA/OpenGL zero-copy display).
- Git with Git LFS, and a C++17 compiler to build the demo FMUs (Windows: Visual Studio 2022 Build Tools, MSVC v143 x64; Linux: build-essential).
- Key dependencies installed by the setup scripts: `ovrtx==0.3.0.312915`, `ovphysx==0.4.9`, `fmpy==0.3.25`, NumPy ≥2.2, GLFW, PyOpenGL. `usd-core` lives in an isolated venv (ovrtx will not load alongside it); ovphysx must be installed **after** ovrtx to avoid Carbonite plugin conflicts.

---

## 5. Licensing

- **Source code:** Apache License 2.0 — permissive, free for commercial and non-commercial use, with an express patent grant (per project materials).
- **Pre-built binaries** of the underlying ov libraries (ovrtx, ovphysx wheels): distributed under the **NVIDIA Omniverse License**; installs additional third-party open-source components (FMPy, OpenUSD, etc.) — review their terms.
- **Third-party notices:** See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) for the project-specific notices and license references.

> **Note:** Pre-release Omniverse Labs project — currently hosted in an internal repository with open-sourcing planned ("get in touch for early access"). Known constraints: co-simulation only (no Model Exchange); SSP internal FMUs limited to FMI 1.0/2.0; single shared time step (no multi-rate); no algebraic-loop detection; declarative scene-query sensors (overlap/raycast) still need an app-level workaround.

---

## 6. Detailed setup and usage

The sections below expand the overview into a complete setup, operation, and
authoring guide. Run commands from the repository root unless a step says
otherwise.

### 6.1 Install platform prerequisites

Install the platform-specific packages that satisfy the system requirements
above.

#### Windows

Install:

- NVIDIA driver: https://www.nvidia.com/Download/index.aspx
- Python: https://www.python.org/downloads/
- Git for Windows: https://git-scm.com/download/win
- Git LFS: https://git-lfs.com/
- Visual Studio 2022 Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

In the Visual Studio Build Tools installer, select **Desktop development with
C++**. Make sure the selected components include **MSVC v143 x64/x86 build
tools**.

You do not need to open the Visual Studio command prompt for normal setup.
Run `setup.ps1` from a normal PowerShell; it invokes `vcvars64.bat` itself for
the FMU build. The generic Visual Studio "Developer Command Prompt" button can
open an environment that is not clearly x64, so avoid using that as the setup
instruction. If you manually rebuild FMUs, use **x64 Native Tools Command
Prompt for VS 2022**.

Optional CUDA Toolkit:

- https://developer.nvidia.com/cuda-downloads

After installing Git LFS, run once:

```powershell
git lfs install
```

The setup script will locate the Visual Studio compiler environment for the FMU
build.

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y \
  git git-lfs \
  python3 python3-venv python3-pip \
  build-essential \
  libgl1 libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6

git lfs install
```

Optional CUDA Toolkit:

- https://developer.nvidia.com/cuda-downloads

For other Linux distributions, install the equivalent packages: Git, Git LFS,
Python 3 with `venv` and `pip`, a C++17 compiler, and OpenGL/X11 runtime
libraries.

### 6.2 Clone the repository

Clone down the repository, and then in the repository directory, run: 

```bash
git submodule update --init --depth=1
```

Why the submodule exists: `third-party/ovrtx` is pinned to the ovrtx 0.3 source
snapshot. Normal Python setup installs the ovrtx wheel from NVIDIA's Python
package index. The submodule is used for matching source/reference files and
for explicit offline/native package override workflows.

### 6.3 Set up Python dependencies

The setup scripts install the normal Python runtime for the app. They install:

- the pinned `ovrtx` wheel from NVIDIA's Python package index
- this app as an editable Python package
- app dependencies such as FMPy, NumPy, GLFW, and PyOpenGL
- `ovphysx` for the physics demos
- an isolated `usd-core` environment used for USD parsing
- `apps/ov-fmi/.env`, which records paths the app needs at runtime
- generated demo `.fmu`, `.fmu3`, and `.ssp` archives in `usd/ov-fmi/`

They do not install optional `cuda-python` or test-only tools such as `pytest`.

If the setup output shows large `ovrtx` or `ovphysx` wheel downloads, that is
expected. Do not install them again manually.

#### Windows

From a normal PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File apps\ov-fmi\setup.ps1
```

This creates and populates:

- `apps/ov-fmi/.venv` for the app
- `apps/ov-fmi/.usd_venv` for USD parsing
- `apps/ov-fmi/.env` with paths used by the app

The script also builds the demo FMUs and SSPs. It can run from a normal
PowerShell as long as Visual Studio Build Tools are installed; it will locate
the `vcvars64.bat` compiler environment automatically.

Optional CUDA/OpenGL zero-copy display support:

```powershell
powershell -ExecutionPolicy Bypass -File apps\ov-fmi\setup.ps1 -InstallCudaPython
```

If you only want to install Python packages and skip FMU compilation:

```powershell
powershell -ExecutionPolicy Bypass -File apps\ov-fmi\setup.ps1 -SkipFmuBuild
```

#### Linux

Run:

```bash
bash apps/ov-fmi/setup.sh
```

This creates and populates `apps/ov-fmi/.venv`, creates
`apps/ov-fmi/.usd_venv` for USD parsing, and builds the demo FMUs and SSPs.

Optional CUDA/OpenGL zero-copy display support:

```bash
INSTALL_CUDA_PYTHON=1 bash apps/ov-fmi/setup.sh
```

If you only want to install Python packages and skip FMU compilation:

```bash
SKIP_FMU_BUILD=1 bash apps/ov-fmi/setup.sh
```

### 6.4 Run the demos

The app opens a live window by default. Close the window or press `Ctrl+C` to
stop. In the live viewer:

- `W/A/S/D`: move
- `Q/E`: down/up
- right mouse drag: mouse look
- `Shift`: faster
- `Ctrl`: slower

#### Windows

```powershell
# Basic FMI rendering demo
.\apps\ov-fmi\.venv\Scripts\python.exe apps\ov-fmi\main.py usd\ov-fmi\fmi_parser_test.usda

# Physics PD controller demo
.\apps\ov-fmi\.venv\Scripts\python.exe apps\ov-fmi\main.py usd\ov-fmi\pd_controller_test.usda

# SSP orbit demo
.\apps\ov-fmi\.venv\Scripts\python.exe apps\ov-fmi\main.py usd\ov-fmi\ssp_orbit.usda

# Conveyor FMI/SSP + ovphysx demo
.\apps\ov-fmi\.venv\Scripts\python.exe apps\ov-fmi\main.py usd\conveyor\ConveyorFMI.usda --up-axis Z
```

#### Linux

```bash
# Basic FMI rendering demo
apps/ov-fmi/.venv/bin/python apps/ov-fmi/main.py usd/ov-fmi/fmi_parser_test.usda

# Physics PD controller demo
apps/ov-fmi/.venv/bin/python apps/ov-fmi/main.py usd/ov-fmi/pd_controller_test.usda

# SSP orbit demo
apps/ov-fmi/.venv/bin/python apps/ov-fmi/main.py usd/ov-fmi/ssp_orbit.usda

# Conveyor FMI/SSP + ovphysx demo
apps/ov-fmi/.venv/bin/python apps/ov-fmi/main.py usd/conveyor/ConveyorFMI.usda --up-axis Z
```

#### Headless smoke test

Use this to verify a setup without opening a live display window.

Windows:

```powershell
.\apps\ov-fmi\.venv\Scripts\python.exe apps\ov-fmi\main.py usd\conveyor\ConveyorFMI.usda --up-axis Z --duration 0.05 --headless
```

Linux:

```bash
apps/ov-fmi/.venv/bin/python apps/ov-fmi/main.py usd/conveyor/ConveyorFMI.usda --up-axis Z --duration 0.05 --headless
```

Expected output includes:

```text
SSP loaded: 7 components, 5 inputs, 12 outputs
Articulation drive target routing enabled
Done: ...
```

## 7. Demo guide

| Stage | Purpose |
|---|---|
| `usd/ov-fmi/fmi_parser_test.usda` | Minimal FMI demo; one FMU drives a sphere height attribute. |
| `usd/ov-fmi/pd_controller_test.usda` | FMU reads rigid-body pose/velocity and writes a force through ovphysx tensors. |
| `usd/ov-fmi/two_fmu_orbit.usda` | Two FMU instances communicate through authored USD attributes. |
| `usd/ov-fmi/ssp_orbit.usda` | One SSP instance hides internal FMU wiring behind system-level connectors. |
| `usd/conveyor/ConveyorFMI.usda` | Conveyor demo with ovphysx rollers, package sensor, SSP controller, and five driven roller zones. |
| `usd/conveyor/Conveyor.usda` | Base conveyor USD asset without FMI overlay; useful for preview/debug. |

## 8. Troubleshooting

### `cl.exe` is not found on Windows

Install Visual Studio Build Tools with **Desktop development with C++**. The
setup script can usually find the Visual Studio compiler environment from a
normal PowerShell. If you need to skip compilation temporarily, rerun setup
with `-SkipFmuBuild`.

### The conveyor stage loads but physics is disabled

Rerun setup. It installs ovphysx after ovrtx, which is the supported order for
these packages.

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File apps\ov-fmi\setup.ps1
```

Linux:

```bash
bash apps/ov-fmi/setup.sh
```

### The app says generated FMU archives are missing

Rerun setup, or rebuild the generated archives directly.

Windows:

```powershell
.\apps\ov-fmi\.venv\Scripts\python.exe apps\ov-fmi\build_fmu.py
```

Linux:

```bash
apps/ov-fmi/.venv/bin/python apps/ov-fmi/build_fmu.py
```

### FMPy reports `WinError 193` while loading an FMU DLL

This usually means the FMU DLL was built for the wrong Windows architecture,
for example by an x86 Visual Studio compiler environment while running 64-bit
Python. Rerun setup from a normal PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File apps\ov-fmi\setup.ps1
```

Rerunning setup should print a line like:

```text
Using Visual Studio environment: ...\vcvars64.bat
```

If that line does not appear, setup did not find the x64 compiler environment.
Reopen the Visual Studio Build Tools installer and confirm **Desktop
development with C++** and **MSVC v143 x64/x86 build tools** are installed.
`build_fmu.py` checks generated DLL architecture before packaging FMUs so this
failure is caught during setup rather than later during simulation.

### Live display does not open on Linux

Check that the machine has a display server and OpenGL runtime libraries. On
headless systems, use `--headless` or `--png`.

### Duplicate Carbonite plugin warnings appear

Warnings can appear when ovrtx and ovphysx are loaded in the same Python
process. They are expected with the currently tested package combination as
long as physics initializes and the simulation runs.

## 9. Command reference

Useful app flags:

| Flag | Description |
|---|---|
| `--headless` | Run without a display window. |
| `--png` | Save rendered frames as PNG images to `_output/`; implies `--headless`. |
| `--duration SECONDS` | Stop after a fixed simulation time. Default is infinity. |
| `--no-physics` | Render authored transforms without starting ovphysx. |
| `--up-axis Y|Z` | Override stage up-axis inference. |
| `--nav-speed VALUE` | Live viewer movement speed. |
| `--mouse-sensitivity VALUE` | Mouse-look sensitivity in degrees per pixel. |
| `--render-product PATH` | Render product prim path. Default: `/Render/Camera`. |
| `--camera-prim PATH` | Camera prim controlled by live navigation. |

## 10. Testing

After setup, install `pytest` into the app venv if you want to run the test
suite.

Windows:

```powershell
.\apps\ov-fmi\.venv\Scripts\python.exe -m pip install pytest
.\apps\ov-fmi\.venv\Scripts\python.exe -m pytest apps\ov-fmi\tests
```

Linux:

```bash
apps/ov-fmi/.venv/bin/python -m pip install pytest
apps/ov-fmi/.venv/bin/python -m pytest apps/ov-fmi/tests
```

## 11. Notes on ovrtx and ovphysx

- `ovrtx==0.3.0.312915` is installed by the setup scripts from NVIDIA's Python
  package index.
- `ovphysx==0.4.9` is installed by the setup scripts after ovrtx because it is
  needed for the physics demos.
- Import order matters: the app imports and initializes ovrtx before ovphysx.
  Loading ovphysx first can fail because both packages use Carbonite plugins.
- On Windows, the app exits directly after a physics run to avoid native DLL
  unload crashes during process shutdown. The simulation has already completed
  at that point.
- `third-party/ovrtx` is a pinned submodule for matching source/reference code
  and offline override workflows. Normal users do not build ovrtx from source.

## 12. USD-FMI schema used by the demos

The app looks for `FmuInstance` and `SspInstance` prims in a USD stage.

Minimal `FmuInstance` shape:

```usda
def FmuInstance "Controller"
{
    bool fmi:enabled = 1
    asset fmi:fmu = @./Controller.fmu@

    def FmuConnection "Output"
    {
        rel fmi:targets = </World/Cube>

        def FmuMapping "WriteX"
        {
            token fmi:direction = "output"
            token fmi:fmuAttribute = "x"
            token fmi:usdAttribute = "xformOp:translate"
            int2 fmi:usdMapping = (0, 1)
        }
    }
}
```

Minimal `SspInstance` shape:

```usda
def SspInstance "System"
{
    bool fmi:enabled = 1
    asset fmi:ssp = @./system.ssp@

    def FmuConnection "PhysicsOutput"
    {
        rel fmi:targets = </World/Body>

        def FmuMapping "ForceY"
        {
            token fmi:direction = "output"
            token fmi:fmuAttribute = "force_y"
            token fmi:usdAttribute = "physx:force"
            int2 fmi:usdMapping = (1, 1)
        }
    }
}
```

Mapping rules:

- `direction = "input"` reads a USD/physics value into an FMU or SSP connector.
- `direction = "output"` writes an FMU or SSP connector value back to USD,
  ovrtx, or ovphysx.
- `fmi:usdMapping = (offset, count)` selects a component. `(0, 0)` means a
  scalar value.
- Transform writes should use `omni:xform` or `xformOp:translate` through the
  app's transform binding path.

Physics routing names:

| Attribute | Direction | Meaning |
|---|---|---|
| `physx:position` | input | Rigid-body position from ovphysx pose tensor. |
| `physx:velocity` | input | Rigid-body linear velocity from ovphysx velocity tensor. |
| `physx:force` | output | Force written to the ovphysx force tensor. |
| `physx:overlap` | input | Sphere overlap presence signal for sensor prims. |
| `drive:angular:physics:targetVelocity` | output | Articulation drive velocity target written through ovphysx tensors. |

These are routing directives, not real authored USD attributes.

For overlap sensors, a prim whose name starts with `Sensor` supplies the query
center from its world transform. A child `Sphere` can supply the query radius.

## 13. Optional developer workflows

### Offline ovrtx package override

Normal setup installs the ovrtx wheel. To test an extracted native ovrtx package
instead, pass its root directory. The directory must contain `bin/`.

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File apps\ov-fmi\setup.ps1 `
    -OVRTX_DIR C:\path\to\ovrtx-0.3.0.windows-x86_64
```

Linux:

```bash
OVRTX_DIR=/path/to/ovrtx-0.3.0.linux-x86_64 bash apps/ov-fmi/setup.sh
```

### Optional `fmi_usd_helper`

Do not build this for normal demo use. The default USD parser path is enough.

`fmi_usd_helper` is a C++ USD parser intended for experiments where the
`usd-core` Python fallback is not desired. It links against USD libraries from
an ovphysx SDK and OpenUSD headers, so it is version-sensitive.

## 14. Project layout

```text
ovfmi/
  apps/ov-fmi/                  Python app, setup scripts, tests
  fmu/                          C++ FMU source folders
  ssp/                          SSP source folders
  usd/ov-fmi/                   Small demo stages and generated FMU/SSP outputs
  usd/conveyor/                 Conveyor USD asset and FMI overlay stage
  third-party/ovrtx/            Pinned ovrtx source submodule
```

---

*ovfmi · OpenUSD + FMI/SSP · ovrtx · ovphysx · FMPy · Copyright (c) 2026 NVIDIA Corporation.*
