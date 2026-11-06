# ABB IRB 4600-60/2.05 visual geometry

Source: [ROS-Industrial abb_irb4600_support](https://github.com/ros-industrial/abb/tree/eefdf01c956a6f2129aacb88d598201feff3cf19/abb_irb4600_support), Apache-2.0. Upstream authors: Jonathan Meyer (SwRI), Rik Tonnaer (SAM|XL), Levi Armstrong (SwRI). The original license and package metadata accompany the model.

Adaptation: seven COLLADA meshes (60,195 triangles) were converted to USD with source normals and URDF joint offsets retained. A static pose within the authored joint limits was selected, and ROS Z-up coordinates were converted to the gallery's Y-up frame. Source metre dimensions are preserved with a display scale of 1:1. The gallery applies illustrative blue, orange and bare-metal finishes; these are material studies, not manufacturer finish specifications. The robot is not actuated or simulated in the RTX exhibit.

Rebuild with `tools/prepare_industrial_robot.py` from the repository root using Python with usd-core and numpy. The source revision, file hashes, selected joint angles and bounds are in `manifest.json`. ABB is the referenced equipment manufacturer; no endorsement is implied.
