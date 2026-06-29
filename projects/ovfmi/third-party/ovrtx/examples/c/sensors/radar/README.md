# Radar Sensor Composite Tensor Example

This example loads `radar_example.usda`, renders a radar `PointCloud` output,
maps the composite render variable to CPU memory, and reads the named tensor
channels. The moving target approaches the radar, so its `RadialVelocityMs`
values are expected to be negative.

> _“Create a C/C++ radar sensor example that applies required runtime settings before renderer creation, loads an animated radar scene, advances scene time across several simulation steps, reads valid detection data including signal strength and signed radial velocity, prints per-step summaries, reports moving-target observations, and cleans up all resources.”_

## Scene

The scene is Z-up and contains:

- a radar at `(0, 0, 1)` rotated to look along world +X
- an asphalt ground plane spanning `X=-200..200` and `Y=-200..200`
- a steel cube moving from `(30, 4, 0.75)` to `(20, 4, 0.75)`
- a concrete cube fixed at `(30, -4, 0.75)`

The moving cube advances 1 m toward the radar on each of 10 simulation steps.
Approaching radar detections use the radar sign convention and report negative
`RadialVelocityMs`.

## Render Output

The USD requests the radar `PointCloud` render variable with these channels:

- `Coordinates`
- `Counts`
- `RCS`
- `RadialVelocityMs`

The executable maps the output to CPU, uses `Counts` as the number of valid
point entries, and prints min/max `RCS` and `RadialVelocityMs` values for each
step.

## API Flow

The example demonstrates this ovrtx flow:

1. Create a renderer with `OVRTX_CONFIG_ENABLE_MOTION_BVH` enabled.
2. Open `radar_example.usda`.
3. Step the render product after warmup.
4. Fetch step results and locate the `PointCloud` render variable output.
5. Map the composite render variable to CPU memory.
6. Read the named DLTensor channels and unmap the output.

## Renderer Config

Motion BVH is required for moving-object radial velocity. The example enables
it via the renderer config at creation time:

```c
ovrtx_config_entry_t config_entries[] = {
    ovrtx_config_entry_enable_motion_bvh(true),
};
ovrtx_config_t config = { config_entries, 1 };
ovrtx_create_renderer(&config, &renderer);
```

The radar orientation is authored in `radar_example.usda`.

## Building

From this directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Running

```bash
./build/radar-composite-tensor
```

You can also pass an explicit scene path:

```bash
./build/radar-composite-tensor path/to/radar_example.usda
```

Expected output values vary, but a successful run prints 10 steps and a final
observation line:

```text
Stepping moving cube toward radar...
  step 1: valid points=..., RCS min/max=[..., ...], radial velocity min/max=[-..., ...] m/s
  ...
Observed ... detections with |radial velocity| > 0.1 m/s; max |radial velocity| = ... m/s
```
