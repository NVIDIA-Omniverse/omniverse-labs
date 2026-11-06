# ABB IRB 120 visual reference

Source: [ROS-Industrial abb_irb120_support, noetic branch](https://github.com/ros-industrial/abb/tree/noetic/abb_irb120_support).

Authors listed upstream: Mark Culleton and Kevin Kelly (Trinity College Dublin), and Levi Armstrong (SwRI). Maintained by the ROS-Industrial community. Upstream package license: Apache-2.0; the complete license is included in `LICENSE.txt`. Model source URLs and SHA-256 hashes are recorded in `manifest.json`.

Changes for this exhibit: seven binary visual STL meshes were converted to USD, smooth corner normals were generated while preserving sharp edges, the original URDF joint offsets were retained, a static display pose was authored, and the coordinate system was changed from Z-up to Y-up. The gallery displays the model at 2.1 times its source scale and applies illustrative surface finishes. It is a visual material/lighting study, not a commissioned robot configuration or motion simulation. Source metre units remain intact in the USD file.

Rebuild with `tools/convert_robot.py` using the original files listed in the manifest. The original kinematics and package metadata are retained here for provenance. ABB is the referenced equipment manufacturer; no endorsement is implied.
