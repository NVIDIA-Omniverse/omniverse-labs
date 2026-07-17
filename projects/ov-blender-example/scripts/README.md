# Scripts

- For a source-only add-on change, run
  `python -m pytest <test-path>` or
  `python scripts/validate_suite.py <suite> --runtime-root <prepared-runtime>`
  directly from this checkout. Developers choose the relevant tests; packaging
  is not required.
- `validate.py`: runs Task Validation's five semantic suites once each against
  one supplied add-on and materialized runtime. CI prepares a disposable test
  tree with the exact built add-on at `addon/`, then runs these same scripts
  from that tree with `--addon-root` and `--runtime-root`.
- `validate_suite.py`: directly runs one of the seven semantic suites. Use
  `validate_suite.py unit --list` to print the authoritative assignment of
  every pytest test, golden workload, performance workload, and validation
  probe (including documented diagnostic exclusions). Repeat `--measurement`
  to select named `performance-large` members from that inventory; omitting it
  runs the complete suite.
- `validate_extended.py`: runs Extended Validation by invoking `golden-large`
  and `performance-large` once each through `validate_suite.py`, retaining
  their separate output directories and result semantics.
- `run_blender_navigation.py`: measures the Junk Shop moving-view workload through the
  production Blender viewport using explicitly supplied deployed packages and
  records software frame service latency from each render invocation to the
  matching newly drawn publication's `POST_PIXEL` callback. It advances a
  deterministic turntable view after each newly presented frame, measuring
  unthrottled render throughput without synthetic input overhead.
- `run_blender_light_edit_responsiveness.py`: targets 120 Hz for 240 Junk Shop
  `Area.019` color edits through Blender's depsgraph path, correlates exact
  scheduler revisions with matching `POST_PIXEL` publications, records the
  achieved offer rate, and verifies that the settled terminal image visibly
  differs from the settled same-run baseline.
- `report_navigation.py`: prints presented FPS from one schema-12, single-run
  navigation throughput record. `performance-large` runs and reports distinct
  LDR and HDR navigation measurements.
- `run_ovrtx_live_transform_probe.py`: validates that a stock Blender transform edit
  reaches `update_transforms` in the same OVRTX render session. It is the
  representative live-transform integration adapter.
- `run_ovrtx_light_value_probe.py`, `run_ovrtx_material_value_probe.py`,
  `run_ovrtx_world_dome_probe.py`, and `run_ovrtx_primvars_st_probe.py`: run
  focused real-runtime diagnostics for same-session value updates that are not
  covered by the fast Python suite.
- `run_ovrtx_orthographic_camera_probe.py`: renders perspective and
  orthographic USD camera controls through the OVRTX bridge/client render-flow
  path and records projection-consistency diagnostics.
- `run_blender_orthographic_view_parity_probe.py`: opens real Blender View3D
  windows and verifies add-on request shaping for orthographic scene cameras,
  active camera view, and orthographic user-view zoom and pan.
- `run_ovrtx_operator_seam_probe.py`: launches bounded Blender GUI, moves a
  tagged imported body interaction object, and records whether the production depsgraph
  bridge reaches the live OVRTX render session.
- `run_ovphysx_drop_probe.py`: launches the configured OVPhysX bridge as a
  managed local subprocess and records drop/step/read diagnostics, or a
  blocked preflight artifact when the subprocess is not available.
- `run_shared_stage_composition_probe.py`: runs the deterministic OVRTX +
  OVPhysX shared-stage composition validation inside background Blender through
  the OVRTX native client and the public OVPhysX gRPC service surface.

Fixture specs and the explicit preparation command live under `tests/fixtures/`.
Native runtime components resolve through the active installed or explicitly
configured runtime.

## License

Project-authored scripts are licensed under Apache-2.0. See the repository
[LICENSE](../LICENSE).
