# From a demonstration to its implementation

Omniverse Libraries Playground is a working application built with the Omniverse
libraries. Start with the small runnable example for a capability, then follow
its room implementation to see how it connects to controls and the shared scene.

| Library | Runnable example | Application implementation |
| --- | --- | --- |
| OVRTX | [Materials](../scripts/examples/render_materials.py), [LiDAR and cameras](../scripts/examples/lidar_points.py) | [Native renderer](../source/nvidia/ov_library_lab/native_runtime.py), [sensor worker](../source/nvidia/ov_library_lab/lidar_worker.py), [camera definitions](../source/nvidia/ov_library_lab/robot_cameras.py) |
| OVPhysX | [Gravity and contacts](../scripts/examples/physics_drop.py) | [Native simulation](../source/nvidia/ov_library_lab/native_runtime.py), [physics controls](../source/nvidia/ov_library_lab/panels.py) |
| OVStage | [Shared scene state](../scripts/examples/shared_stage.py) | [Scene bridge](../source/nvidia/ov_library_lab/native_runtime.py), [fleet transforms](../source/nvidia/ov_library_lab/workshops.py) |
| OVUI | [Shared widget model](../scripts/examples/operator_ui.py) | [Application shell](../source/nvidia/ov_library_lab/presentation.py), [scene-anchored controls](../source/nvidia/ov_library_lab/equipment_panel.py) |
| OVStorage | [Write and read revisions](../scripts/examples/storage_revisions.py) | [Storage operations](../source/nvidia/ov_library_lab/native_storage.py), [guided controls](../source/nvidia/ov_library_lab/storage_panel.py) |
| OVStream | [Frame and command round trip](../scripts/examples/stream_roundtrip.py) | [Native transport](../source/nvidia/ov_library_lab/stream_transport.py), [received image display](../source/nvidia/ov_library_lab/native_view.py) |
| OVNewton | [Articulated gantry](../scripts/examples/gantry_newton.py) | [Solver worker](../source/nvidia/ov_library_lab/newton_worker.py), [room state](../source/nvidia/ov_library_lab/newton_room.py) |
| OVPackage | [Package and verify](../scripts/examples/asset_package.py) | [Package operations](../source/nvidia/ov_library_lab/package_worker.py), [publish/install workflow](../source/nvidia/ov_library_lab/package_flow.py) |

The [example guide](../scripts/examples/README.md) identifies the Python environment
for each script. The in-app information button opens explanations and these same
examples. Package and Newton use separate environments to preserve their tested
dependency versions.

## How the application fits together

1. `app.py` owns the window, update loop and worker lifetimes.
2. `scene.py` assembles USD assets and exhibit geometry. `layout.py` supplies
   room placement and navigation; artwork is distinct from live library state.
3. `native_runtime.py` publishes scene edits through OVStage and requests OVRTX
   images. OVPhysX results supply the physics objects' displayed poses.
4. Room modules own their experiment state; UI modules turn visitor actions into
   library requests and show the resulting state.
5. `library_worker.py` keeps expensive or incompatible SDK work in owned
   processes. Sensors prepare during loading before live frame submission;
   interactive updates poll for results without waiting for the worker.

## What each example does and does not establish

LiDAR coordinates and camera images come from OVRTX. The robot follows a scripted
route, and its camera/sensor calibration is illustrative. Fleet motion and the
OVUI equipment are kinematic; they are not presented as physics simulations.
OVPhysX and OVNewton compute the motion in their own examples. Storage uses local
files, and streaming uses a local shared-memory connection. These demonstrate
library operations without claiming a production deployment architecture.

Repository paths, the `ov_library_lab` Python package and existing runtime folders
retain their stable identifiers after the product rename. Existing data and
launch scripts therefore keep working.
