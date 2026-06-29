# Lidar Sensor Composite Tensor Example

This example loads `lidar_example.usda`, renders a lidar `PointCloud` output,
maps the composite render variable to CPU memory, and reads the named tensor
channels.

> _“Create a C/C++ lidar sensor example that applies required sensor runtime settings before renderer creation, loads a lidar scene, warms up the sensor pipeline, renders one point-cloud output, reads valid point data safely through the count channel, prints summary statistics, and cleans up all results and mappings.”_

## Scene

The scene is Z-up and contains:

- a lidar at `(0, 0, 1)` rotated to look along world +X
- an asphalt ground plane spanning `X=-200..200` and `Y=-200..200`
- a concrete cube at `(10, 0, 0.75)`

## Render Output

The USD requests the lidar `PointCloud` render variable with these channels:

- `Coordinates`
- `Intensity`
- `Counts`
- `TimeOffsetNs`

The executable maps the output to CPU, uses `Counts` as the number of valid
point entries, and prints the mean `Intensity` and maximum `TimeOffsetNs` over
the valid point range.

## API Flow

The example demonstrates this ovrtx flow:

1. Create a renderer with `OVRTX_CONFIG_ENABLE_MOTION_BVH` enabled.
2. Open `lidar_example.usda`.
3. Step the render product after warmup.
4. Fetch step results and locate the `PointCloud` render variable output.
5. Map the composite render variable to CPU memory.
6. Read the named DLTensor channels and unmap the output.

## Renderer Config

Motion BVH is required by the lidar sensor pipeline. The example enables it via
the renderer config at creation time:

```c
ovrtx_config_entry_t config_entries[] = {
    ovrtx_config_entry_enable_motion_bvh(true),
};
ovrtx_config_t config = { config_entries, 1 };
ovrtx_create_renderer(&config, &renderer);
```

## Building

From this directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Running

```bash
./build/lidar-composite-tensor
```

You can also pass an explicit scene path:

```bash
./build/lidar-composite-tensor path/to/lidar_example.usda
```

Expected output values vary, but a successful run prints:

```text
valid points=..., mean intensity=..., max time offset=... ns
```
