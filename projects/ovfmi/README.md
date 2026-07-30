<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ovfmi — FMI/SSP co-simulation for OpenUSD digital twins

> Load FMI behavior models from an OpenUSD stage, step them with FMPy, and
> synchronize simulation data through a reusable Python API. The included
> standalone demo application renders with ovrtx and can close the loop with
> ovphysx rigid-body simulation. The project builds on the disaggregated
> Omniverse libraries and the open FMI, SSP, and OpenUSD standards.
>
> *Pre-release ovfmi 0.2.0 project. APIs and the USD-FMI schema may change.*

## What is ovfmi?

`ovfmi` is a reusable, USD-native library that adds behavior to industrial
digital twins by binding FMI co-simulation models to OpenUSD scenes through a
custom schema. USD describes world state; FMI and SSP supply simulation
behavior. ovfmi joins them so a USD layer can describe both the scene and its
simulation model in a form that remains readable and composable by USD tools.

Applications use the ovfmi Python API to discover `FmuInstance` and
`SspInstance` prims, instantiate their referenced models through FMPy, route
values between model variables and USD attributes, and advance the models.
The `demoapp/` directory is separate sample code that shows how to combine the
library with ovstage, ovrtx rendering, and optional ovphysx simulation without
requiring a full Omniverse Kit installation.

## What functionalities are available, and who are the target users?

Available functionality includes:

- Declaratively embedding individual FMUs and packaged SSP systems in USD.
- Running FMI 2.0 and FMI 3.0 co-simulation models through the FMPy backend,
  plus SSP 1.0 archives whose internal FMUs meet FMPy's version constraints.
- Mapping FMI variables to USD attributes, including selected vector
  components and ranges.
- Deterministic, synchronous simulation stepping and explicit data routing
  through a documented Python API.
- Integrating FMU control logic with ovphysx rigid bodies in the sample
  application, using shared ovstage data rather than per-attribute USD I/O.
- Running the supplied examples with live ovrtx rendering or in headless mode.

ovfmi is intended for:

- Industrial digital-twin engineers adding controllers, sensors, and actuator
  behavior to OpenUSD scenes.
- Controls and simulation engineers bringing vendor-neutral FMUs and SSPs into
  USD-based workflows.
- Robotics and physics developers coupling model-based control logic to
  rigid-body simulations.
- ISVs and integrators assembling standards-based simulation applications.

## Documentation and reference links

- [USD-FMI Schema Reference](docs/USD-FMI-SCHEMA.md)
- [Architecture and API design](docs/architecture.md)
- [FMI standard](https://fmi-standard.org/)
- [SSP standard](https://ssp-standard.org/)
- [OpenUSD](https://openusd.org/)
- [FMPy FMI runtime](https://github.com/CATIA-Systems/FMPy)
- GTC 2025 session S71963, "Build Physics-Based Digital Twins for
  Co-Simulation" (with SoftServe), which presents the original project.

Install the library from the repository root with:

```bash
python -m pip install .
```

## Fast Start

The commands below are the normal path from a fresh machine to running the
demos. Use the Windows block on Windows and the Linux block on Linux.
Run commands from the repository root unless a step says otherwise.

### 1. Install Prerequisites

#### Library requirements

The ovfmi library requires:

- Python 3.10 through 3.13.
- Windows or Linux on x86_64.
- A compatible ovstage package when attaching ovfmi to a USD stage.

An NVIDIA GPU, graphics driver, display server, OpenGL, Git LFS, and a C++
compiler are not requirements of the ovfmi library itself.

#### Demo application requirements

Building and running the included demo application additionally requires:

- Git with Git LFS.
- A C++17 compiler for building the supplied demo FMUs.
- Compatible ovrtx and ovstage packages.
- ovphysx for the physics-enabled examples.
- An NVIDIA RTX GPU and recent NVIDIA driver for ovrtx rendering.
- Display/OpenGL support only for the live viewer.

The `--headless` option removes the display-window and OpenGL requirement, but
the current demo still initializes ovrtx and therefore still requires an RTX
GPU. This distinction does not apply to applications that use ovfmi without
the ovrtx renderer.

The CUDA Toolkit is optional. The app can display frames through a CPU upload
path without CUDA. Install CUDA only if you want CUDA/OpenGL zero-copy display
or other CUDA developer tools.

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

#### Linux Ubuntu/Debian

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

### 2. Clone the Repository

If you already cloned the repo, skip to step 3.

```bash
git clone https://github.com/NVIDIA-dev/ovfmi.git
cd ovfmi
```

### 3. Set Up Python Dependencies

First install the three NVIDIA runtime packages into the Python environment
that will launch setup. The packages may come from a local wheel, an internal
index, or a public package index once published:

Windows:

```powershell
py -m pip install ovrtx ovstage ovphysx
```

Linux:

```bash
python3 -m pip install ovrtx ovstage ovphysx
```

The setup scripts do not install or pin these packages. They create the demo
virtual environment with access to the base environment, validate that all
three packages are visible, and then install:

- the root `ovfmi` product and `demoapp` as editable Python packages
- ovfmi's FMPy backend dependency and demo dependencies such as GLFW, PyOpenGL,
  and pytest
- an isolated `usd-core` environment used for USD parsing
- `demoapp/.env`, which records paths the app needs at runtime
- generated demo `.fmu`, `.fmu3`, and `.ssp` archives in `demoapp/usd/`

They do not install optional `cuda-python` unless requested.

If `demoapp/.venv` was created by an older checkout, delete it before running
the new setup so it can inherit the packages installed in the base environment.

#### Windows

From a normal PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File demoapp\setup.ps1
```

This creates and populates:

- `demoapp/.venv` for the app
- `demoapp/.usd_venv` for USD parsing
- `demoapp/.env` with paths used by the app

The script also builds the demo FMUs and SSPs. It can run from a normal
PowerShell as long as Visual Studio Build Tools are installed; it will locate
the `vcvars64.bat` compiler environment automatically.

Optional CUDA/OpenGL zero-copy display support:

```powershell
powershell -ExecutionPolicy Bypass -File demoapp\setup.ps1 -InstallCudaPython
```

If you only want to install Python packages and skip FMU compilation:

```powershell
powershell -ExecutionPolicy Bypass -File demoapp\setup.ps1 -SkipFmuBuild
```

#### Linux

Run:

```bash
bash demoapp/setup.sh
```

This creates and populates `demoapp/.venv`, creates
`demoapp/.usd_venv` for USD parsing, and builds the demo FMUs and SSPs.

Optional CUDA/OpenGL zero-copy display support:

```bash
INSTALL_CUDA_PYTHON=1 bash demoapp/setup.sh
```

If you only want to install Python packages and skip FMU compilation:

```bash
SKIP_FMU_BUILD=1 bash demoapp/setup.sh
```

### 4. Run the Demos

The app opens a live window by default. Close the window or press `Ctrl+C` to
stop. The first ovrtx run can take several minutes while shaders compile;
subsequent runs reuse the shader cache. In the live viewer:

- `W/A/S/D`: move
- `Q/E`: down/up
- right mouse drag: mouse look
- `Shift`: faster
- `Ctrl`: slower

#### Windows

```powershell
# Basic FMI rendering demo
.\demoapp\.venv\Scripts\python.exe demoapp\main.py demoapp\usd\fmi_parser_test.usda

# Physics PD controller demo
.\demoapp\.venv\Scripts\python.exe demoapp\main.py demoapp\usd\pd_controller_test.usda

# SSP orbit demo
.\demoapp\.venv\Scripts\python.exe demoapp\main.py demoapp\usd\ssp_orbit.usda

# Conveyor FMI/SSP + ovphysx demo
.\demoapp\.venv\Scripts\python.exe demoapp\main.py demoapp\usd\conveyor\ConveyorFMI.usda --up-axis Z
```

#### Linux

```bash
# Basic FMI rendering demo
demoapp/.venv/bin/python demoapp/main.py demoapp/usd/fmi_parser_test.usda

# Physics PD controller demo
demoapp/.venv/bin/python demoapp/main.py demoapp/usd/pd_controller_test.usda

# SSP orbit demo
demoapp/.venv/bin/python demoapp/main.py demoapp/usd/ssp_orbit.usda

# Conveyor FMI/SSP + ovphysx demo
demoapp/.venv/bin/python demoapp/main.py demoapp/usd/conveyor/ConveyorFMI.usda --up-axis Z
```

#### Headless Smoke Test

Use this to verify a setup without opening a live display window.

Windows:

```powershell
.\demoapp\.venv\Scripts\python.exe demoapp\main.py demoapp\usd\conveyor\ConveyorFMI.usda --up-axis Z --duration 0.05 --headless
```

Linux:

```bash
demoapp/.venv/bin/python demoapp/main.py demoapp/usd/conveyor/ConveyorFMI.usda --up-axis Z --duration 0.05 --headless
```

Expected output includes:

```text
SSP loaded: 7 components, 5 inputs, 12 outputs
FMI physics controls: shared ovstage control ordinals
Done: ...
```

## What Each Demo Shows

| Stage | Purpose |
|---|---|
| `demoapp/usd/fmi_parser_test.usda` | Minimal FMI demo; one FMU drives a sphere height attribute. |
| `demoapp/usd/pd_controller_test.usda` | FMU reads rigid-body pose/velocity and writes a force through ovphysx tensors. |
| `demoapp/usd/two_fmu_orbit.usda` | Two FMU instances communicate through authored USD attributes. |
| `demoapp/usd/ssp_orbit.usda` | One SSP instance hides internal FMU wiring behind system-level connectors. |
| `demoapp/usd/conveyor/ConveyorFMI.usda` | Conveyor demo with ovphysx rollers, package sensor, SSP controller, and five driven roller zones. |
| `demoapp/usd/conveyor/Conveyor.usda` | Base conveyor USD asset without FMI overlay; useful for preview/debug. |

## Known Issues

### Custom USD-FMI schema discovery requires source parsing

The currently supported ovstage release does not expose data authored with
unregistered custom schemas, including the USD-FMI `FmuInstance`,
`SspInstance`, `FmuConnection`, and `FmuMapping` prims used by ovfmi.
Consequently, ovfmi cannot yet discover FMI/SSP instances and mappings from
the populated ovstage alone.

As a compatibility measure, `FmiHost.attach_ovstage()` accepts the original
USD file through its `source_asset` argument. ovfmi pre-parses that source with
OpenUSD in an isolated helper process and uses the resulting schema description
to populate its FMI/SSP runtime directly. Simulation values continue to flow
through the caller-owned ovstage; the auxiliary parse is used only for schema
discovery and initial authored values. The demo setup configures the isolated
OpenUSD environment automatically.

## Common Problems

### `cl.exe` is not found on Windows

Install Visual Studio Build Tools with **Desktop development with C++**. The
setup script can usually find the Visual Studio compiler environment from a
normal PowerShell. If you need to skip compilation temporarily, rerun setup
with `-SkipFmuBuild`.

### The conveyor stage loads but physics is disabled

Confirm that all three NVIDIA runtime packages are installed in the base Python
environment:

Windows:

```powershell
py -m pip show ovrtx ovstage ovphysx
```

Linux:

```bash
python3 -m pip show ovrtx ovstage ovphysx
```

If a package is missing or outdated, install the required wheels in that base
environment, delete `demoapp/.venv`, and rerun setup. The application imports
ovrtx before ovphysx at runtime, as required for shared plugin initialization.

### The app says generated FMU archives are missing

Rerun setup, or rebuild the generated archives directly.

Windows:

```powershell
.\demoapp\.venv\Scripts\python.exe demoapp\build_fmu.py
```

Linux:

```bash
demoapp/.venv/bin/python demoapp/build_fmu.py
```

### FMPy reports `WinError 193` while loading an FMU DLL

This usually means the FMU DLL was built for the wrong Windows architecture,
for example by an x86 Visual Studio compiler environment while running 64-bit
Python. Rerun setup from a normal PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File demoapp\setup.ps1
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

## Command Reference

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

## Testing

The regular setup installs pytest into the demo environment.

Windows:

```powershell
.\demoapp\.venv\Scripts\python.exe -m pytest tests
.\demoapp\.venv\Scripts\python.exe -m pytest demoapp\tests
```

Linux:

```bash
demoapp/.venv/bin/python -m pytest tests
demoapp/.venv/bin/python -m pytest demoapp/tests
```

## Public Python API

Applications should import supported symbols directly from `ovfmi`. The
package's `__all__` is the compatibility boundary; submodules and names that
begin with an underscore are implementation details.

```python
from ovfmi import AttributeWrite, FmiHost

with FmiHost() as fmi:
    report = fmi.attach_ovstage(stage, source_asset=usd_path)
    fmi.update_from_ovstage(input_ordinal, input_ordinal)
    fmi.write(
        [
            AttributeWrite(
                prim_paths=("/World/Body",),
                attribute_name="sim:input",
                values=[[1.0]],
            )
        ]
    )
    fmi.step_sync(1.0 / 60.0)

    with fmi.read() as result:
        consume_outputs(result.groups)
```

The public symbols are:

| Symbol | Purpose |
|---|---|
| `FmiHost` | Owns FMI/SSP instances, routing state, and simulation lifecycle for one caller-owned ovstage. |
| `FmiHostConfig` | Configures discovery, schema validation, SSP support, and missing-input behavior. |
| `MissingInputPolicy` | Selects how unavailable mapped inputs are initialized. |
| `PopulationReport`, `InstanceInfo` | Describe instances created during attachment. |
| `AttributeWrite` | Supplies input values identified by USD prim path and attribute name. |
| `ReadResult`, `ReadGroup` | Provide an owned snapshot of output values in USD space. |

All public classes and methods carry detailed Python docstrings, including
ownership, filtering, operation-token, ordinal, and current backend semantics.
The present FMPy backend is synchronous; `step()` and `write()` nevertheless
use the backend-neutral API's completion-token lifecycle.

## Versioning and Releases

The library distribution version is recorded in `VERSION.md` and in
`[project].version` in the root `pyproject.toml`; a unit test requires them to
match. Versions follow Python's PEP 440 format, and a public release is tagged
`v<version>` from the commit used to build its wheel and source distribution.
For example, version `0.2.0` uses tag `v0.2.0`.

The demo application has separate package metadata but tracks the library's
minor release. Its ovfmi dependency accepts compatible patch releases within
that minor series.

## Notes on ovstage, ovrtx, and ovphysx

- The project deliberately does not pin or install ovrtx, ovstage, or ovphysx.
  Install compatible wheels in the base Python environment before running
  setup. Pip can resolve them from a configured package index or local wheel
  paths.
- Setup creates `demoapp/.venv` with access to base-environment packages and
  validates that all three distributions are visible. It does not copy,
  upgrade, downgrade, or otherwise manage them.
- Stage population now goes through ovstage, which ovrtx renders directly in
  zero-copy BORROW mode. There is no private replicated rendering stage or
  legacy compatibility population pass.
- ovphysx attaches to the same caller-owned ovstage as ovrtx. Physics
  outputs are read with `PhysX.read()` and published to output-only ovstage
  ordinals that are never drained back into physics. Renderer-consumed local
  transforms are authored to the same ordinal before it is sealed and rendered
  directly from the shared Fabric.
- Initialization order matters: the app initializes ovrtx, then populates
  ovstage, and only then imports ovphysx. Loading ovphysx first can fail because
  the packages use shared Carbonite plugins.
- On Windows, the app exits directly after a physics run to avoid native DLL
  unload crashes during process shutdown. The simulation has already completed
  at that point.
## USD FMI Schema Used by the Demos

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

Legacy tensor-routing names:

| Attribute | Direction | Meaning |
|---|---|---|
| `physx:position` | input | Rigid-body position from ovphysx pose tensor. |
| `physx:velocity` | input | Rigid-body linear velocity from ovphysx velocity tensor. |
| `physx:force` | output | Force written to the ovphysx force tensor. |

These are routing directives, not real authored USD attributes.

Stage-routed physics values:

| Attribute | Direction | Meaning |
|---|---|---|
| `sensor:presence` | input | Real runtime USD attribute populated from an ovphysx overlap query and read by FMI on the next frame. |
| `drive:angular:physics:targetVelocity` | output | Real USD drive attribute written to an ovstage control ordinal and consumed through `PhysX.update_from_ovstage()`. |

The conveyor FMUs retain the deprecated articulation tensor convention of
radians per second. At the ovstage boundary, the app converts those drive
outputs to the USD angular-drive unit of degrees per second.

For overlap sensors, a prim whose name starts with `Sensor` identifies the
sensor and a child `UsdGeomSphere` supplies the preferred shape query. The app
runs an ovphysx `SHAPE`/`ANY` overlap after each physics step, publishes the
boolean result as `sensor:presence` on the sensor prim, and consumes that stage
attribute during the next FMI step. If no child shape is available, its world
position and radius are used for a sphere query.

## Optional Developer Workflows

### Optional `fmi_usd_helper`

Do not build this for normal demo use. The default USD parser path is enough.

`fmi_usd_helper` is a C++ USD parser intended for experiments where the
`usd-core` Python fallback is not desired. It links against USD libraries from
an ovphysx SDK and OpenUSD headers, so it is version-sensitive.

## Project Layout

```text
ovfmi/
  python/                       Shipping ovfmi Python package
  tests/                        Library unit and integration tests
  demoapp/                      Rendering/physics sample application
    fmu/                        Demo FMU source folders
    ssp/                        Demo SSP source folders
    usd/                        Demo stages and generated model archives
  docs/                         Architecture, schema, and API design documents
```

## License

The ovfmi source code is licensed under the
[Apache License 2.0](LICENSE.md). Bundled demo assets and third-party
dependencies may have separate terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
