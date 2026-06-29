# UsdPhysics parsing conformance: nanousd vs OpenUSD

- nanousd shim: `$HOME/.venv/bin/python`
- OpenUSD (usd-core): `/tmp/usdcore-venv/bin/python`

Each robot is parsed via `UsdPhysics.LoadUsdPhysicsFromRange` through both
backends; divergences are grouped into structural, shim-only, OpenUSD-only,
and **value-mismatch** (the parser-bug list).

| robot | obj-count | path-set | shim-only | openusd-only | **value-mismatch** |
|---|--:|--:|--:|--:|--:|
| `ant` | 0 | 0 | 60 | 114 | **9** |
| `ant_mixed` | 0 | 0 | 60 | 114 | **9** |
| `cartpole_mjc` | 0 | 0 | 22 | 34 | **3** |
| `cube_cylinder` | 0 | 0 | 8 | 15 | **9** |
| `four_link_chain_articulation` | 0 | 0 | 31 | 49 | **22** |
| `humanoid` | 0 | 0 | 120 | 296 | **91** |
| `revolute_articulation` | 0 | 0 | 15 | 25 | **8** |
| `simple_articulation_with_mesh` | 0 | 0 | 12 | 26 | **1** |

## Value mismatches by field (parser bugs to investigate)

- `jointLimits` — 42
- `collisionGroups` — 19
- `position` — 18
- `jointDrives` — 16
- `rotation` — 12
- `localScale` — 12
- `collisions` — 7
- `limit` — 6
- `localPos` — 5
- `localRot` — 4
- `articulatedBodies` — 3
- `drive` — 3
- `localPose1Orientation` — 3
- `localPose0Orientation` — 2

## OpenUSD-only fields by name (shim coverage gaps)

- `primPath` — 133
- `simulationOwners` — 103
- `filteredCollisions` — 52
- `scale` — 45
- `startsAsleep` — 45
- `breakForce` — 40
- `breakTorque` — 40
- `collisionEnabled` — 40
- `rel0` — 40
- `rel1` — 40
- `filteredGroups` — 20
- `invertFilteredGroups` — 20
- `mergeGroupName` — 20
- `mergedGroups` — 20
- `rootPrims` — 7
- `approximation` — 4
- `doubleSided` — 4

## `ant` — details


<details><summary>value mismatches</summary>

- `RigidBody@/ant/front_left_foot.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/front_left_leg.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/front_right_foot.position[2]`: shim=`-0.0` openusd=`1.0`
- `RigidBody@/ant/front_right_leg.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/left_back_foot.position[2]`: shim=`-0.0` openusd=`1.0`
- `RigidBody@/ant/left_back_leg.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/right_back_foot.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/right_back_leg.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/torso.position[2]`: shim=`0.0` openusd=`1.0`

</details>

## `ant_mixed` — details


<details><summary>value mismatches</summary>

- `RigidBody@/ant/front_left_foot.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/front_left_leg.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/front_right_foot.position[2]`: shim=`-0.0` openusd=`1.0`
- `RigidBody@/ant/front_right_leg.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/left_back_foot.position[2]`: shim=`-0.0` openusd=`1.0`
- `RigidBody@/ant/left_back_leg.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/right_back_foot.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/right_back_leg.position[2]`: shim=`0.0` openusd=`1.0`
- `RigidBody@/ant/torso.position[2]`: shim=`0.0` openusd=`1.0`

</details>

## `cartpole_mjc` — details


<details><summary>value mismatches</summary>

- `Articulation@/test_cartpole/Geometry/cart.articulatedBodies.len`: shim=`2` openusd=`3`
- `RevoluteJoint@/test_cartpole/Geometry/cart/pole/hinge.limit.enabled`: shim=`True` openusd=`False`
- `RigidBody@/test_cartpole/Geometry/cart.collisions.len`: shim=`2` openusd=`1`

</details>

## `cube_cylinder` — details


<details><summary>value mismatches</summary>

- `MeshShape@/World/Cube_static/cube2/mesh_0.localPos[2]`: shim=`0.0` openusd=`-1.5`
- `MeshShape@/World/Cube_static/cube2/mesh_0.localRot[1]`: shim=`1.0` openusd=`0.70711`
- `MeshShape@/World/Cube_static/cube2/mesh_0.localRot[2]`: shim=`0.0` openusd=`0.70711`
- `MeshShape@/World/Cylinder_dynamic/cylinder_reverse/mesh_0.localRot[1]`: shim=`1.0` openusd=`0.70711`
- `MeshShape@/World/Cylinder_dynamic/cylinder_reverse/mesh_0.localRot[2]`: shim=`0.0` openusd=`0.70711`
- `RigidBody@/World/Cylinder_dynamic.rotation[1]`: shim=`1.0` openusd=`0.91`
- `RigidBody@/World/Cylinder_dynamic.rotation[2]`: shim=`0.0` openusd=`-0.28367`
- `RigidBody@/World/Cylinder_dynamic.rotation[3]`: shim=`0.0` openusd=`0.29795`
- `RigidBody@/World/Cylinder_dynamic.rotation[4]`: shim=`0.0` openusd=`-0.05157`

</details>

## `four_link_chain_articulation` — details


<details><summary>value mismatches</summary>

- `Articulation@/Articulation.articulatedBodies.len`: shim=`4` openusd=`5`
- `CubeShape@/Articulation/Body1.localPos[0]`: shim=`1.0` openusd=`0.0`
- `CubeShape@/Articulation/Body1.localScale[0]`: shim=`2.0` openusd=`1.0`
- `CubeShape@/Articulation/Body1.localScale[1]`: shim=`0.1` openusd=`1.0`
- `CubeShape@/Articulation/Body1.localScale[2]`: shim=`0.1` openusd=`1.0`
- `CubeShape@/Articulation/Body2.localPos[0]`: shim=`2.0` openusd=`0.0`
- `CubeShape@/Articulation/Body2.localScale[0]`: shim=`2.0` openusd=`1.0`
- `CubeShape@/Articulation/Body2.localScale[1]`: shim=`0.1` openusd=`1.0`
- `CubeShape@/Articulation/Body2.localScale[2]`: shim=`0.1` openusd=`1.0`
- `CubeShape@/Articulation/Body3.localPos[0]`: shim=`3.0` openusd=`0.0`
- `CubeShape@/Articulation/Body3.localScale[0]`: shim=`2.0` openusd=`1.0`
- `CubeShape@/Articulation/Body3.localScale[1]`: shim=`0.1` openusd=`1.0`
- `CubeShape@/Articulation/Body3.localScale[2]`: shim=`0.1` openusd=`1.0`
- `RevoluteJoint@/Articulation/Joint1.drive.damping`: shim=`100000.0` openusd=`0.0`
- `RevoluteJoint@/Articulation/Joint1.drive.enabled`: shim=`True` openusd=`False`
- `RevoluteJoint@/Articulation/Joint1.drive.stiffness`: shim=`100000.0` openusd=`0.0`
- `RevoluteJoint@/Articulation/Joint1.limit.enabled`: shim=`True` openusd=`False`
- `RevoluteJoint@/Articulation/Joint2.limit.enabled`: shim=`True` openusd=`False`
- `RevoluteJoint@/Articulation/Joint3.limit.enabled`: shim=`True` openusd=`False`
- `RigidBody@/Articulation/Body1.collisions.len`: shim=`0` openusd=`1`
- `RigidBody@/Articulation/Body2.collisions.len`: shim=`0` openusd=`1`
- `RigidBody@/Articulation/Body3.collisions.len`: shim=`0` openusd=`1`

</details>

## `humanoid` — details


<details><summary>value mismatches</summary>

- `CapsuleShape@/nv_humanoid/left_foot/collisions/left_left_foot.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/left_foot/collisions/right_left_foot.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/left_lower_arm/collisions/left_lower_arm.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/left_shin/collisions/left_shin.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/left_thigh/collisions/left_thigh.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/left_upper_arm/collisions/left_upper_arm.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/lower_waist/collisions/lower_waist.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/pelvis/collisions/butt.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/right_foot/collisions/left_right_foot.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/right_foot/collisions/right_right_foot.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/right_lower_arm/collisions/right_lower_arm.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/right_shin/collisions/right_shin.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/right_thigh/collisions/right_thigh.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/right_upper_arm/collisions/right_upper_arm.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/torso/collisions/torso.collisionGroups.len`: shim=`2` openusd=`1`
- `CapsuleShape@/nv_humanoid/torso/collisions/upper_waist.collisionGroups.len`: shim=`2` openusd=`1`
- `D6Joint@/nv_humanoid/joints/left_foot.jointDrives[0].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/left_foot.jointDrives[1].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/left_foot.jointLimits[0].first`: shim=`<str>transX` openusd=`<int>1`
- `D6Joint@/nv_humanoid/joints/left_foot.jointLimits[1].first`: shim=`<str>transY` openusd=`<int>2`
- `D6Joint@/nv_humanoid/joints/left_foot.jointLimits[2].first`: shim=`<str>transZ` openusd=`<int>3`
- `D6Joint@/nv_humanoid/joints/left_foot.jointLimits[3].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/left_foot.jointLimits[4].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/left_foot.jointLimits[5].first`: shim=`<str>rotZ` openusd=`<int>6`
- `D6Joint@/nv_humanoid/joints/left_thigh.jointDrives[0].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/left_thigh.jointDrives[1].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/left_thigh.jointDrives[2].first`: shim=`<str>rotZ` openusd=`<int>6`
- `D6Joint@/nv_humanoid/joints/left_thigh.jointLimits[0].first`: shim=`<str>transX` openusd=`<int>1`
- `D6Joint@/nv_humanoid/joints/left_thigh.jointLimits[1].first`: shim=`<str>transY` openusd=`<int>2`
- `D6Joint@/nv_humanoid/joints/left_thigh.jointLimits[2].first`: shim=`<str>transZ` openusd=`<int>3`
- `D6Joint@/nv_humanoid/joints/left_thigh.jointLimits[3].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/left_thigh.jointLimits[4].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/left_thigh.jointLimits[5].first`: shim=`<str>rotZ` openusd=`<int>6`
- `D6Joint@/nv_humanoid/joints/left_upper_arm.jointDrives[0].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/left_upper_arm.jointDrives[1].first`: shim=`<str>rotZ` openusd=`<int>6`
- `D6Joint@/nv_humanoid/joints/left_upper_arm.jointLimits[0].first`: shim=`<str>transX` openusd=`<int>1`
- `D6Joint@/nv_humanoid/joints/left_upper_arm.jointLimits[1].first`: shim=`<str>transY` openusd=`<int>2`
- `D6Joint@/nv_humanoid/joints/left_upper_arm.jointLimits[2].first`: shim=`<str>transZ` openusd=`<int>3`
- `D6Joint@/nv_humanoid/joints/left_upper_arm.jointLimits[3].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/left_upper_arm.jointLimits[4].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/left_upper_arm.jointLimits[5].first`: shim=`<str>rotZ` openusd=`<int>6`
- `D6Joint@/nv_humanoid/joints/lower_waist.jointDrives[0].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/lower_waist.jointDrives[1].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/lower_waist.jointLimits[0].first`: shim=`<str>transX` openusd=`<int>1`
- `D6Joint@/nv_humanoid/joints/lower_waist.jointLimits[1].first`: shim=`<str>transY` openusd=`<int>2`
- `D6Joint@/nv_humanoid/joints/lower_waist.jointLimits[2].first`: shim=`<str>transZ` openusd=`<int>3`
- `D6Joint@/nv_humanoid/joints/lower_waist.jointLimits[3].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/lower_waist.jointLimits[4].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/lower_waist.jointLimits[5].first`: shim=`<str>rotZ` openusd=`<int>6`
- `D6Joint@/nv_humanoid/joints/lower_waist.localPose0Orientation[1]`: shim=`0.70569` openusd=`-0.70569`
- `D6Joint@/nv_humanoid/joints/lower_waist.localPose0Orientation[3]`: shim=`-0.70852` openusd=`0.70852`
- `D6Joint@/nv_humanoid/joints/lower_waist.localPose1Orientation[1]`: shim=`0.70711` openusd=`-0.70711`
- `D6Joint@/nv_humanoid/joints/lower_waist.localPose1Orientation[3]`: shim=`-0.70711` openusd=`0.70711`
- `D6Joint@/nv_humanoid/joints/right_foot.jointDrives[0].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/right_foot.jointDrives[1].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/right_foot.jointLimits[0].first`: shim=`<str>transX` openusd=`<int>1`
- `D6Joint@/nv_humanoid/joints/right_foot.jointLimits[1].first`: shim=`<str>transY` openusd=`<int>2`
- `D6Joint@/nv_humanoid/joints/right_foot.jointLimits[2].first`: shim=`<str>transZ` openusd=`<int>3`
- `D6Joint@/nv_humanoid/joints/right_foot.jointLimits[3].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/right_foot.jointLimits[4].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/right_foot.jointLimits[5].first`: shim=`<str>rotZ` openusd=`<int>6`
- `D6Joint@/nv_humanoid/joints/right_thigh.jointDrives[0].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/right_thigh.jointDrives[1].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/right_thigh.jointDrives[2].first`: shim=`<str>rotZ` openusd=`<int>6`
- `D6Joint@/nv_humanoid/joints/right_thigh.jointLimits[0].first`: shim=`<str>transX` openusd=`<int>1`
- `D6Joint@/nv_humanoid/joints/right_thigh.jointLimits[1].first`: shim=`<str>transY` openusd=`<int>2`
- `D6Joint@/nv_humanoid/joints/right_thigh.jointLimits[2].first`: shim=`<str>transZ` openusd=`<int>3`
- `D6Joint@/nv_humanoid/joints/right_thigh.jointLimits[3].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/right_thigh.jointLimits[4].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/right_thigh.jointLimits[5].first`: shim=`<str>rotZ` openusd=`<int>6`
- `D6Joint@/nv_humanoid/joints/right_upper_arm.jointDrives[0].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/right_upper_arm.jointDrives[1].first`: shim=`<str>rotZ` openusd=`<int>6`
- `D6Joint@/nv_humanoid/joints/right_upper_arm.jointLimits[0].first`: shim=`<str>transX` openusd=`<int>1`
- `D6Joint@/nv_humanoid/joints/right_upper_arm.jointLimits[1].first`: shim=`<str>transY` openusd=`<int>2`
- `D6Joint@/nv_humanoid/joints/right_upper_arm.jointLimits[2].first`: shim=`<str>transZ` openusd=`<int>3`
- `D6Joint@/nv_humanoid/joints/right_upper_arm.jointLimits[3].first`: shim=`<str>rotX` openusd=`<int>4`
- `D6Joint@/nv_humanoid/joints/right_upper_arm.jointLimits[4].first`: shim=`<str>rotY` openusd=`<int>5`
- `D6Joint@/nv_humanoid/joints/right_upper_arm.jointLimits[5].first`: shim=`<str>rotZ` openusd=`<int>6`
- `RigidBody@/nv_humanoid/left_foot.rotation[3]`: shim=`0.004` openusd=`-0.004`
- `RigidBody@/nv_humanoid/left_shin.rotation[3]`: shim=`0.004` openusd=`-0.004`
- `RigidBody@/nv_humanoid/left_thigh.rotation[3]`: shim=`0.004` openusd=`-0.004`
- `RigidBody@/nv_humanoid/lower_waist.rotation[3]`: shim=`0.002` openusd=`-0.002`
- `RigidBody@/nv_humanoid/pelvis.rotation[3]`: shim=`0.004` openusd=`-0.004`
- `RigidBody@/nv_humanoid/right_foot.collisions[0]`: shim=`/nv_humanoid/right_foot/collisions/right_right_foot` openusd=`/nv_humanoid/right_foot/collisions/left_right_foot`
- `RigidBody@/nv_humanoid/right_foot.collisions[1]`: shim=`/nv_humanoid/right_foot/collisions/left_right_foot` openusd=`/nv_humanoid/right_foot/collisions/right_right_foot`
- `RigidBody@/nv_humanoid/right_foot.rotation[3]`: shim=`0.004` openusd=`-0.004`
- `RigidBody@/nv_humanoid/right_shin.rotation[3]`: shim=`0.004` openusd=`-0.004`
- `RigidBody@/nv_humanoid/right_thigh.rotation[3]`: shim=`0.004` openusd=`-0.004`
- `SphereShape@/nv_humanoid/head/collisions/head.collisionGroups.len`: shim=`2` openusd=`1`
- `SphereShape@/nv_humanoid/left_hand/collisions/left_hand.collisionGroups.len`: shim=`2` openusd=`1`
- `SphereShape@/nv_humanoid/right_hand/collisions/right_hand.collisionGroups.len`: shim=`2` openusd=`1`

</details>

## `revolute_articulation` — details


<details><summary>value mismatches</summary>

- `Articulation@/Articulation.articulatedBodies.len`: shim=`2` openusd=`3`
- `CubeShape@/Articulation/Arm.localPos[0]`: shim=`1.0` openusd=`0.0`
- `CubeShape@/Articulation/Arm.localScale[0]`: shim=`2.0` openusd=`1.0`
- `CubeShape@/Articulation/Arm.localScale[1]`: shim=`0.1` openusd=`1.0`
- `CubeShape@/Articulation/Arm.localScale[2]`: shim=`0.1` openusd=`1.0`
- `FixedJoint@/Articulation/CenterPivot/FixedJoint.localPose1Orientation[1]`: shim=`0.63246` openusd=`1.0`
- `RevoluteJoint@/Articulation/Arm/RevoluteJoint.limit.enabled`: shim=`True` openusd=`False`
- `RigidBody@/Articulation/Arm.collisions.len`: shim=`0` openusd=`1`

</details>

## `simple_articulation_with_mesh` — details


<details><summary>value mismatches</summary>

- `RevoluteJoint@/Robot/joints/RevoluteJoint.limit.enabled`: shim=`True` openusd=`False`

</details>
