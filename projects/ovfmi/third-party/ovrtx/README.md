

# NVIDIA ovrtx

ovrtx is a lightweight C and Python SDK for Omniverse RTX, allowing developers to integrate RTX sensor simulation and visualization easily into their applications.

Omniverse RTX provides real-time, physically accurate sensor simulation and rendering for [Physical AI](https://www.nvidia.com/en-us/glossary/generative-physical-ai/), targeting robotics learning, synthetic data generation, and industrial and design workflows.

* [Get started in Python](#getting-started-in-python)
* [Get started in C](#getting-started-in-c)

> [!NOTE]
> ovrtx is currently **pre-release** software.

![warehouse](img/warehouse.jpg)

## Features
* Physically accurate simulation of cameras, lidar, radar, and other sensors.
* Scalable simulation performance from reinforcement learning in-the-loop with tens of thousands of frames per second, through real-time, photorealistic, interactive viewport and navigation, to offline predictive rendering.
* [OpenUSD](https://aousd.org/) scene description allowing interchange with a vast ecosystem of content creation, CAD and simulation tools.
* Easy integration with Python simulation and learning ecosystem.

## Getting Started in Python

During Early Access, we recommend using the [uv](https://docs.astral.sh/uv/getting-started/installation/) Python package and project manager. All the examples in this repository contain pyproject.toml files that are tested with uv.

Python 3.10–3.13 is required.

To get started, first clone this repository and run the first example with uv:

```bash
git clone https://github.com/NVIDIA-Omniverse/ovrtx.git
cd ovrtx/examples/python/minimal
uv run main.py
```

The minimal example shows how to create the renderer, load an OpenUSD scene and render a single image, copying the results back to the CPU for display.

![minimal example](img/example-minimal.jpg)


Note that the first time a program built against ovrtx is run, it will compile and cache necessary shaders, which may take some time depending on your system. Subsequent runs will use the cached shaders and will be fast.

## Getting Started in C

The C/C++ examples require CMake and a development environment. On Windows this is provided by [Visual Studio 2017 or newer](https://visualstudio.microsoft.com/). On Linux (Ubuntu):

```bash
sudo apt-get build-essential cmake
```

To get started, first clone this repository and build and run the first example using CMake:

```bash
git clone https://github.com/NVIDIA-Omniverse/ovrtx.git
cd ovrtx/examples/c/minimal
cmake -B build 
```

Then, on Windows:
```
cmake --build build --config Release
.\build\Release\minimal.exe
```

On Linux:
```
cmake --build build --config Release
./build/minimal
```

The minimal example shows how to create the renderer, load an OpenUSD scene and render a single image, copying the results back to the CPU for writing out as a PNG.

![minimal example](img/example-minimal.jpg)

The resulting image will be written to `./out.png` and can be inspected with any image viewer.

Note that the first time a program built against ovrtx is run, it will compile and cache necessary shaders, which may take some time depending on your system. Subsequent runs will use the cached shaders and will be fast.

## Examples

Further examples using both the C and Python APIs are available in the [examples](examples/README.md) directory. See the individual examples for building and usage instructions.

## Releases

The Releases page of this repository contains binary builds for the official releases of the ovrtx C library and the corresponding Python wheels.
These binaries are provided for the supported platforms:
 * Windows x86_64
 * Linux x86_64
 * Linux aarch64

The libraries require a compatible NVIDIA RTX-capable GPU with a compatible NVIDIA driver on the system to be able to initialize correctly.
More detailed system requirements can be found at https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html 

<!--
## Badges
TODO - show badges for
* PyPI package
* License
* GitHub Commit activity
* downloads/month
* CI pipeline results
-->

## AI Coding Agents

The [skills](skills) directory contains a series of Skills to help AI coding agents to understand how to use the API (and they're useful for humans too). Copy this directory to your project and point your agent at it.


## Testing

Tests live under `tests/docs/` in three independent suites: Python, C, and USD validation.

### Python tests (GPU required)

```bash
cd tests/docs/python

# Run the full suite
uv run pytest -v

# Run a single test
uv run pytest test_base.py::test_base -v
uv run pytest test_base.py::test_bind_material -v
```

### C tests (GPU required)

```bash
cd tests/docs/c
cmake -B build
cmake --build build --config Release

# Run the full suite
cd build && ctest --output-on-failure

# Run a single test
ctest -R BaseTest.RenderLdrColor --output-on-failure

# Or use the gtest binary directly
./ovrtx_docs_tests --gtest_filter=BaseTest.BindMaterial
```

### USD validation tests (no GPU required)

```bash
cd tests/docs/usd
uv run pytest -v
```

Rendered images are written to `_output/` under the respective suite directory.

## Documentation
<!-- TODO - Pointer to Tutorials and API Documentation -->
Documentation is published at https://nvidia-omniverse.github.io/ovrtx

### Prerequisites for Documentation Build

* [uv](https://docs.astral.sh/uv/)
* [Doxygen](https://www.doxygen.nl/)

```bash
cd docs
make html
```

Then, to view the built docs:
```bash
uv run python -m http.server 8000 -d _build/html
```

Then open http://localhost:8000/ in a browser.


## Support
https://forums.developer.nvidia.com/c/omniverse/300

## Roadmap
To be announced

## Contributing
At this time this project is not open to external contributions.

## Authors and acknowledgment
NVIDIA Corporation

## License

The software and materials are governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and the [Product Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
