# Vendored robot test assets

These USD robot assets are vendored verbatim from the **Newton** physics engine
(<https://github.com/newton-physics/newton>), from `newton/tests/assets/`.

They are used here only as fixtures for the OpenUSD ⇄ nanousd differential
conformance test (`tests/test_usdphysics_conformance.py`), which parses each
robot through both real OpenUSD (`usd-core`) and nanousd's `pxr_compat` and
flags any divergence in `UsdPhysics` parsing.

## License

Newton is licensed under the **Apache License, Version 2.0**.

    SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
    SPDX-License-Identifier: Apache-2.0

The full license text is in Newton's `LICENSE.md`
(<https://github.com/newton-physics/newton/blob/main/LICENSE.md>). A copy is
included alongside this file as `LICENSE-Apache-2.0.txt`.

## Provenance / attribution

| File | Notes |
|------|-------|
| `humanoid.usda` | `defaultPrim = "nv_humanoid"` — an NVIDIA-derived humanoid (Isaac Gym lineage), redistributed by Newton under Apache-2.0. |
| `ant.usda`, `ant_mixed.usda` | Newton "ant" articulation fixtures. |
| `cartpole_mjc.usda` | Cartpole articulation (MuJoCo-converted) fixture. |
| `four_link_chain_articulation.usda`, `revolute_articulation.usda` | Articulation/joint fixtures. |
| `simple_articulation_with_mesh.usda` | Articulation with an inline mesh collider. |
| `cube_cylinder.usda` | Primitive-collider fixture. |

All files are self-contained `.usda` (no external references/payloads), so both
USD backends parse them without additional dependencies.

Only self-contained, in-repo Newton assets are vendored. Newton's downloadable
robots (Unitree G1/H1, ANYbotics ANYmal) live in the separate `newton-assets`
and `mujoco_menagerie` repositories under their own upstream licenses and are
intentionally **not** vendored here.
