# Minimal Example

This example shows basic initialization of the renderer, rendering a single frame from an RGB camera, mapping the output and writing the result to disk as a PNG.

The example will load an example scene from S3 and output the resulting image to `out.png`.

The first time the example is run, driver shader compilation will be performed and cached. Subsequent runs will be much faster. 

> _“Create the smallest useful C/C++ example that initializes ovrtx, loads a USD scene asynchronously, waits for it, renders one camera frame, fetches and CPU-maps the color output, writes it to an image file, and releases all ovrtx resources explicitly.”_

![output](../../../img/out.jpg)


## Linux

### Prerequisites

- `sudo apt install build-essential cmake`

The ovrtx library is downloaded automatically at configure time. If ovrtx is already installed and available via `CMAKE_PREFIX_PATH`, the local installation is used instead.


### Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Running

```bash
./build/minimal
```

## Windows

### Prerequisites

- [Visual Studio 2017+](https://visualstudio.microsoft.com/downloads/)

The ovrtx library is downloaded automatically at configure time. If ovrtx is already installed and available via `CMAKE_PREFIX_PATH`, the local installation is used instead.

### Building

```pwsh
cmake -B build
cmake --build --config Release
```

### Running

```pwsh
.\build\Release\minimal
```

# Licensing

This example contains stb_image_write.h, © Sean Barrett, released under Public Domain. 
