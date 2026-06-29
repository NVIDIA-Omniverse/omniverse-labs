# mtlx2msl

Tiny standalone driver that loads a `.mtlx` document and runs
`MaterialXGenMsl` against every renderable element, dumping the emitted
Metal Shading Language source to stdout. Phase 1 of the closure-based
MaterialX integration plan — get the codegen path wired end-to-end
before plumbing closures into the path tracer.

Mirror of the Vulkan-side
[`tools/mtlx2glsl`](../../../nanousd-vulkan-renderer/tools/mtlx2glsl).
The only structural differences are:

* `MslShaderGenerator` instead of `GlslShaderGenerator`.
* Default libraries path defaults to
  `$HOME/OpenUSD-install/src/MaterialX-1.39.4` (the chess comparison
  harness already depends on this layout).
* `--rt` mode dropped — the closure-based `RtSurfaceNodeMsl` is
  Phase 2 work and lives in a separate commit.

## Build

```sh
cd tools/mtlx2msl
cmake -B build -S .
cmake --build build
```

Override the MaterialX install prefix with `MATERIALX_INSTALL`:

```sh
cmake -B build -S . -DMATERIALX_INSTALL=/opt/materialx
```

## Run

```sh
./build/mtlx2msl path/to/material.mtlx
```

`stderr` shows a per-element summary (vertex/pixel byte counts),
`stdout` carries the actual MSL source.

## Verified

Smoke-tested against three Standard Surface assets shipped with
MaterialX 1.39.4:

| Asset | Pixel-stage size |
| --- | --- |
| `standard_surface_default.mtlx` | 100 KB |
| `standard_surface_brick_procedural.mtlx` | 134 KB |
| `standard_surface_chess_set.mtlx` (15 materials) | 113 KB / material |

…plus the chess-set per-piece `.mtlx` files at
`/tmp/usd-chess/full_assets/OpenChessSet/assets/<Piece>/<Piece>_mat.mtlx`.

The brick test exercises a multi-node nodegraph
(`noise2d` / `mix` / `multiply`); the chess set exercises the full
Standard Surface BSDF tree (oren_nayar_diffuse / dielectric / conductor /
sheen / subsurface / layer-add-mix combinators) plus textures.

## Phase 2 (next)

* Custom `RtSurfaceNodeMsl` that emits a closure-returning function
  (`SurfaceClosure evalSurface_<name>(...)`) instead of the stock
  forward-shaded fragment shader, so the path tracer can call into
  the generated MSL at hit time.
* Plumb the closure result through `shade_hit` so authored
  Standard-Surface materials drive the shader rather than the
  hand-written approximation in `raytrace.metal`.
