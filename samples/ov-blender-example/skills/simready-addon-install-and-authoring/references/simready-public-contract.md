# SimReady public contract

## Installed authoring surface

Probe the running add-on. Known distributions expose collection creation and
physics assignment through operators such as
`bpy.ops.sr_core.create_simready_collections()` and
`bpy.ops.simready.assign_physx_properties()`. Export integration varies by
release: inspect registered operator RNA and the distribution documentation
instead of assuming `export_scene.simready_usd` exists.

Do not create `pxr:*` custom properties as a fallback. Their presence alone
does not establish that the official add-on authored or validated the asset.

## Pinned OVRTX conversion boundary

At the revision recorded by the repository, public conversion is implemented
in `addon/ovrtx_blender_example/simready_physics_conversion.py` and tested by
`tests/test_simready_physics_conversion.py`. It accepts one uni-body:

- `Export` contains `Geometry`, `ReferencePrims`, and `Colliders` collections;
- Geometry contains exactly one mesh body;
- ReferencePrims contains exactly one reference object;
- Colliders contains no separate collider mesh;
- the body has exactly one `CHILD_OF` constraint to that reference and no
  unsupported joint constraint;
- exactly one physics material is present;
- mass is finite and positive; center of mass, diagonal inertia, principal
  axes, density, friction, and restitution values are present and finite.

Structured errors such as `separate_simready_colliders_unsupported`,
`simready_unibody_requires_one_geometry_mesh`,
`simready_joint_constraints_unsupported`, or
`simready_unibody_requires_one_physics_material` define honest unsupported
boundaries. Check the pinned source before relying on this list after updates.
