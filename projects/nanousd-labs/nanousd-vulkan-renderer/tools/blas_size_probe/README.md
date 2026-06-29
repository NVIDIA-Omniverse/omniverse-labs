# blas_size_probe

Standalone Vulkan utility that calls `vkGetAccelerationStructureBuildSizesKHR`
twice on the same logical BLAS geometry — once with
`VK_FORMAT_R32G32B32_SFLOAT` vertex input, once with
`VK_FORMAT_R16G16B16A16_SFLOAT` — and prints the resulting BLAS / scratch
sizes side by side.

Purpose: settle whether NVIDIA preserves input vertex precision in BVH
leaves on the target GPU. The Vulkan spec is silent on this; the answer must
come from the driver itself.

## What it measures

For a fixed mesh of 1000 indexed triangles (3000 vertices, `uint32` indices,
flags = `PREFER_FAST_TRACE | ALLOW_COMPACTION` — matching the production
renderer in `src/gpu_vulkan.c`):

- `accelerationStructureSize` for fp32 positions
- `accelerationStructureSize` for fp16 positions (4-component, since
  3-component fp16 is not universally supported as an AS vertex format)
- `buildScratchSize` and `updateScratchSize` for both

The size-query API does **not** dereference vertex/index device addresses —
no buffers are allocated, no commands are submitted, no actual BLAS is
built. The driver computes sizes purely from format, stride, max vertex,
and primitive count.

## How to build

```bash
cd nanousd-vulkan-renderer/tools/blas_size_probe
mkdir build && cd build
cmake ..
make
```

Requires the Vulkan SDK / loader to be installed and discoverable by
CMake's `find_package(Vulkan)`.

## How to run

```bash
./blas_size_probe
```

No arguments. No display required (headless — instance is created without a
surface / swapchain).

## Output format

```
device       = NVIDIA GeForce RTX 5090         <- VkPhysicalDeviceProperties.deviceName
api_version  = 1.3.0                            <- driver-reported Vulkan API version
driver_ver   = 0x83400000                       <- VkPhysicalDeviceProperties.driverVersion
vendor_id    = 0x10de                           <- PCI vendor ID (0x10de = NVIDIA)
tri_count    = 1000                             <- primitives passed to the size query
vertex_count = 3000 (indexed, uint32)
flags        = PREFER_FAST_TRACE | ALLOW_COMPACTION
fp32 fmt     = R32G32B32_SFLOAT  stride=12  as_supported=1
fp16 fmt     = R16G16B16A16_SFLOAT  stride=8  as_supported=1

fp32_size=NNNNN  fp16_size=MMMMM  delta=DDDD  delta_pct=X.XX%
fp32  buildScratchSize=...  updateScratchSize=...
fp16  buildScratchSize=...  updateScratchSize=...
```

### How to read it

- **`as_supported=1`** for both formats means the driver advertises
  `VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR` on that
  format. If fp16 reports `as_supported=0`, the tool exits with status 3
  and prints a clear error — the fp16 BLAS path is not usable on this GPU.
- **`fp32_size`** and **`fp16_size`** are the driver's reported
  `accelerationStructureSize` in bytes for an identically-shaped mesh that
  differs only in the input vertex format.
- **`delta = fp16_size - fp32_size`**. Negative means fp16 produces a
  smaller BLAS (driver preserves the lower precision in leaves). Zero or
  near-zero means the driver internally requantizes — most likely to
  fp32 — and fp16 input buys nothing in BLAS size.
- **`buildScratchSize` / `updateScratchSize`** are the transient working
  memory the driver needs during build / update. A delta here is
  uncommon but worth recording.

### Decision rules

| Result | Action |
|--------|--------|
| `delta_pct` clearly negative (e.g. < -10%) on NVIDIA | fp16 BLAS positions preserve precision; quantization PR is worth designing. |
| `delta_pct` near zero on NVIDIA | Driver requantizes; drop fp16 BLAS positions and pursue compaction / OMM / SER / `PREFER_FAST_TRACE` instead. |
| fp16 not supported (`as_supported=0`) | Path is not portable; rely on alternatives. |

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success — both formats queried, results printed. |
| 1 | Vulkan init / extension / device-create failure. |
| 2 | R32G32B32_SFLOAT not advertised for AS (unexpected). |
| 3 | R16G16B16A16_SFLOAT not advertised for AS — fp16 path unsupported. |
