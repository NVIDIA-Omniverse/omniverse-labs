# Full-stage read-API conformance: nanousd vs OpenUSD

Every prim of every robot walked through both backends; the broad read
surface (type, applied schemas, every authored `attr.Get()`, relationships,
metadata) is diffed. Coverage shows how much API surface each robot exercises.

| robot | prims | authored attrs | value types | rels | **value-mismatch** | shim-only | openusd-only |
|---|--:|--:|--:|--:|--:|--:|--:|
| `ant` | 64 | 437 | 15 | 85 | **129** | 20 | 17 |
| `ant_mixed` | 64 | 506 | 16 | 85 | **129** | 20 | 17 |
| `cartpole_mjc` | 17 | 100 | 14 | 48 | **40** | 27 | 4 |
| `cube_cylinder` | 15 | 99 | 17 | 26 | **19** | 9 | 7 |
| `four_link_chain_articulation` | 10 | 85 | 12 | 18 | **6** | 2 | 6 |
| `humanoid` | 124 | 778 | 10 | 256 | **196** | 113 | 52 |
| `revolute_articulation` | 7 | 73 | 12 | 13 | **10** | 5 | 4 |
| `simple_articulation_with_mesh` | 7 | 49 | 13 | 14 | **15** | 4 | 3 |

**Totals across 8 robots:** 308 prims, 2127 authored attributes, 545 relationships compared; 544 value-mismatches.

## `ant` — details

<details><summary>value mismatches</summary>

- `@/ant.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant.schemas.len`: shim=`2` openusd=`1`
- `@/ant/front_left_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/collisions/left_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/collisions/left_ankle_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/collisions/left_ankle_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/collisions/left_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/visuals/left_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/visuals/left_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/collisions/left_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/collisions/left_leg_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/collisions/left_leg_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/collisions/left_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/visuals/left_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/visuals/left_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/collisions/right_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/collisions/right_ankle_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/collisions/right_ankle_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/collisions/right_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/visuals/right_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/visuals/right_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/collisions/right_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/collisions/right_leg_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/collisions/right_leg_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/collisions/right_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/visuals/right_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/visuals/right_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/front_left_foot.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/front_left_foot.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_left_foot.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>1`
- `@/ant/joints/front_left_foot.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>1`
- `@/ant/joints/front_left_foot.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/front_left_leg.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/front_left_leg.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_left_leg.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_left_leg.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_left_leg.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/front_right_foot.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/front_right_foot.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_right_foot.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>3`
- `@/ant/joints/front_right_foot.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>3`
- `@/ant/joints/front_right_foot.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/front_right_leg.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/front_right_leg.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_right_leg.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>2`
- `@/ant/joints/front_right_leg.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>2`
- `@/ant/joints/front_right_leg.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/left_back_foot.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/left_back_foot.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/left_back_foot.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>5`
- `@/ant/joints/left_back_foot.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>5`
- `@/ant/joints/left_back_foot.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/left_back_leg.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/left_back_leg.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/left_back_leg.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>4`
- `@/ant/joints/left_back_leg.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>4`
- `@/ant/joints/left_back_leg.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/right_back_foot.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/right_back_foot.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/right_back_foot.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>7`
- `@/ant/joints/right_back_foot.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>7`
- `@/ant/joints/right_back_foot.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/right_back_leg.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/right_back_leg.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/right_back_leg.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>6`
- `@/ant/joints/right_back_leg.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>6`
- `@/ant/joints/right_back_leg.schemas.len`: shim=`3` openusd=`1`
- `@/ant/left_back_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/collisions/third_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/collisions/third_ankle_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/collisions/third_ankle_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/collisions/third_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/visuals/third_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/visuals/third_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/collisions/back_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/collisions/back_leg_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/collisions/back_leg_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/collisions/back_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/visuals/back_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/visuals/back_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/collisions/fourth_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/collisions/fourth_ankle_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/collisions/fourth_ankle_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/collisions/fourth_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/visuals/fourth_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/visuals/fourth_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/collisions/rightback_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/collisions/rightback_leg_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/collisions/rightback_leg_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/collisions/rightback_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/visuals/rightback_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/visuals/rightback_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso.attrs.physics:newton:articulation_index.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/torso.schemas.len`: shim=`3` openusd=`2`
- `@/ant/torso/collisions/aux_1_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_1_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_1_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_1_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_2_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_2_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_2_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_2_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_3_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_3_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_3_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_3_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_4_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_4_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_4_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_4_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/torso_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/torso_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/torso_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_1_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_1_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_2_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_2_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_3_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_3_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_4_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_4_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/torso_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/physicsScene.schemas.len`: shim=`1` openusd=`0`

</details>

## `ant_mixed` — details

<details><summary>value mismatches</summary>

- `@/ant.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant.schemas.len`: shim=`2` openusd=`1`
- `@/ant/front_left_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/collisions/left_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/collisions/left_ankle_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/collisions/left_ankle_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/collisions/left_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/visuals/left_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_foot/visuals/left_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/collisions/left_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/collisions/left_leg_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/collisions/left_leg_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/collisions/left_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/visuals/left_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_left_leg/visuals/left_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/collisions/right_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/collisions/right_ankle_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/collisions/right_ankle_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/collisions/right_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/visuals/right_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_foot/visuals/right_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/collisions/right_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/collisions/right_leg_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/collisions/right_leg_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/collisions/right_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/visuals/right_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/front_right_leg/visuals/right_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/front_left_foot.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/front_left_foot.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_left_foot.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>1`
- `@/ant/joints/front_left_foot.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>1`
- `@/ant/joints/front_left_foot.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/front_left_leg.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/front_left_leg.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_left_leg.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_left_leg.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_left_leg.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/front_right_foot.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/front_right_foot.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_right_foot.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>3`
- `@/ant/joints/front_right_foot.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>3`
- `@/ant/joints/front_right_foot.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/front_right_leg.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/front_right_leg.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/front_right_leg.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>2`
- `@/ant/joints/front_right_leg.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>2`
- `@/ant/joints/front_right_leg.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/left_back_foot.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/left_back_foot.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/left_back_foot.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>5`
- `@/ant/joints/left_back_foot.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>5`
- `@/ant/joints/left_back_foot.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/left_back_leg.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/left_back_leg.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/left_back_leg.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>4`
- `@/ant/joints/left_back_leg.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>4`
- `@/ant/joints/left_back_leg.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/right_back_foot.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/right_back_foot.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/right_back_foot.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>7`
- `@/ant/joints/right_back_foot.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>7`
- `@/ant/joints/right_back_foot.schemas.len`: shim=`3` openusd=`1`
- `@/ant/joints/right_back_leg.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/joints/right_back_leg.attrs.physics:tensor:angular:dofOffset.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/joints/right_back_leg.attrs.physics:tensor:jointDofsStartIndex.value`: shim=`<NoneType>None` openusd=`<int>6`
- `@/ant/joints/right_back_leg.attrs.physics:tensor:jointIndex.value`: shim=`<NoneType>None` openusd=`<int>6`
- `@/ant/joints/right_back_leg.schemas.len`: shim=`3` openusd=`1`
- `@/ant/left_back_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/collisions/third_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/collisions/third_ankle_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/collisions/third_ankle_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/collisions/third_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/visuals/third_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_foot/visuals/third_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/collisions/back_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/collisions/back_leg_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/collisions/back_leg_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/collisions/back_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/visuals/back_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/left_back_leg/visuals/back_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/collisions/fourth_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/collisions/fourth_ankle_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/collisions/fourth_ankle_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/collisions/fourth_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/visuals/fourth_ankle_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_foot/visuals/fourth_ankle_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/collisions/rightback_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/collisions/rightback_leg_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/collisions/rightback_leg_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/collisions/rightback_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/visuals/rightback_leg_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/right_back_leg/visuals/rightback_leg_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso.attrs.physics:newton:articulation_index.value`: shim=`<NoneType>None` openusd=`<int>0`
- `@/ant/torso.schemas.len`: shim=`3` openusd=`2`
- `@/ant/torso/collisions/aux_1_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_1_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_1_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_1_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_2_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_2_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_2_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_2_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_3_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_3_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_3_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_3_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_4_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_4_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_4_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/aux_4_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/torso_geom.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/torso_geom.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/collisions/torso_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_1_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_1_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_2_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_2_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_3_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_3_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_4_geom.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/aux_4_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/ant/torso/visuals/torso_geom.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/physicsScene.schemas.len`: shim=`1` openusd=`0`

</details>

## `cartpole_mjc` — details

<details><summary>value mismatches</summary>

- `@/PhysicsScene.attrs.mjc:compiler:inertiaFromGeom.variability`: shim=`varying` openusd=`uniform`
- `@/PhysicsScene.attrs.mjc:option:timestep.variability`: shim=`varying` openusd=`uniform`
- `@/PhysicsScene.schemas.len`: shim=`1` openusd=`0`
- `@/test_cartpole/Geometry/cart.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/cart.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/cart.schemas.len`: shim=`3` openusd=`2`
- `@/test_cartpole/Geometry/cart/pole.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/pole/cpole.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/pole/cpole.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/pole/cpole.schemas.len`: shim=`3` openusd=`2`
- `@/test_cartpole/Geometry/cart/pole/hinge.attrs.mjc:damping.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/pole/hinge.attrs.mjc:solreflimit.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/pole/hinge.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/pole/hinge.schemas.len`: shim=`1` openusd=`0`
- `@/test_cartpole/Geometry/cart/pole/tip.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/pole/tip.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/pole/tip.schemas.len`: shim=`1` openusd=`0`
- `@/test_cartpole/Geometry/cart/slider.attrs.mjc:damping.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/slider.attrs.mjc:solreflimit.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/slider.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/slider.schemas.len`: shim=`1` openusd=`0`
- `@/test_cartpole/Geometry/cart/tn__cartsensor_kA.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/tn__cartsensor_kA.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/cart/tn__cartsensor_kA.schemas.len`: shim=`1` openusd=`0`
- `@/test_cartpole/Geometry/floor.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/floor.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/floor.schemas.len`: shim=`3` openusd=`2`
- `@/test_cartpole/Geometry/rail1.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/rail1.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/rail1.schemas.len`: shim=`3` openusd=`2`
- `@/test_cartpole/Geometry/rail2.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/rail2.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Geometry/rail2.schemas.len`: shim=`3` openusd=`2`
- `@/test_cartpole/Physics/PhysicsMaterial.attrs.mjc:rollingfriction.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Physics/PhysicsMaterial.attrs.mjc:torsionalfriction.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Physics/PhysicsMaterial.schemas.len`: shim=`2` openusd=`1`
- `@/test_cartpole/Physics/slide.attrs.mjc:ctrlLimited.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Physics/slide.attrs.mjc:ctrlRange:max.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Physics/slide.attrs.mjc:ctrlRange:min.variability`: shim=`varying` openusd=`uniform`
- `@/test_cartpole/Physics/slide.attrs.mjc:gear.variability`: shim=`varying` openusd=`uniform`

</details>

## `cube_cylinder` — details

<details><summary>value mismatches</summary>

- `@/Environment.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Environment/defaultLight.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Environment/defaultLight.schemas.len`: shim=`1` openusd=`4`
- `@/World/Camera.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/World/Camera.attrs.stereoRole.variability`: shim=`varying` openusd=`uniform`
- `@/World/Camera.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/World/Cube_static.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/World/Cube_static.schemas.len`: shim=`6` openusd=`0`
- `@/World/Cube_static/cube2.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/World/Cube_static/cube2/mesh_0.attrs.orientation.variability`: shim=`varying` openusd=`uniform`
- `@/World/Cube_static/cube2/mesh_0.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/World/Cube_static/cube2/mesh_0.attrs.subdivisionScheme.variability`: shim=`varying` openusd=`uniform`
- `@/World/Cylinder_dynamic.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/World/Cylinder_dynamic.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/World/Cylinder_dynamic.schemas.len`: shim=`11` openusd=`5`
- `@/World/Cylinder_dynamic/cylinder_reverse.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/World/Cylinder_dynamic/cylinder_reverse/mesh_0.attrs.orientation.variability`: shim=`varying` openusd=`uniform`
- `@/World/Cylinder_dynamic/cylinder_reverse/mesh_0.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/World/Cylinder_dynamic/cylinder_reverse/mesh_0.attrs.subdivisionScheme.variability`: shim=`varying` openusd=`uniform`

</details>

## `four_link_chain_articulation` — details

<details><summary>value mismatches</summary>

- `@/Articulation.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Articulation/Body0.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Articulation/Body1.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Articulation/Body2.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Articulation/Body3.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Articulation/Joint1.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`

</details>

## `humanoid` — details

<details><summary>value mismatches</summary>

- `@/nv_humanoid/CollisionGroup_NonParticipants.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_butt.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_head.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_left_hand.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_left_left_foot.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_left_lower_arm.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_left_right_foot.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_left_shin.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_left_thigh.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_left_upper_arm.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_lower_waist.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_right_hand.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_right_left_foot.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_right_lower_arm.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_right_right_foot.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_right_shin.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_right_thigh.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_right_upper_arm.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_torso.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/CollisionGroup_upper_waist.attrs.collection:colliders:expansionRule.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/head.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/head.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/head/collisions/head.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/head/collisions/head.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/head/collisions/head.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/head/visuals/head.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/abdomen_x.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/abdomen_x.schemas.len`: shim=`1` openusd=`0`
- `@/nv_humanoid/joints/left_elbow.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/left_elbow.schemas.len`: shim=`1` openusd=`0`
- `@/nv_humanoid/joints/left_foot.attrs.drive:rotX:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/left_foot.attrs.drive:rotY:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/left_foot.schemas.len`: shim=`10` openusd=`8`
- `@/nv_humanoid/joints/left_knee.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/left_knee.schemas.len`: shim=`1` openusd=`0`
- `@/nv_humanoid/joints/left_thigh.attrs.drive:rotX:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/left_thigh.attrs.drive:rotY:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/left_thigh.attrs.drive:rotZ:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/left_thigh.schemas.len`: shim=`12` openusd=`9`
- `@/nv_humanoid/joints/left_upper_arm.attrs.drive:rotX:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/left_upper_arm.attrs.drive:rotZ:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/left_upper_arm.schemas.len`: shim=`10` openusd=`8`
- `@/nv_humanoid/joints/lower_waist.attrs.drive:rotX:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/lower_waist.attrs.drive:rotY:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/lower_waist.schemas.len`: shim=`10` openusd=`8`
- `@/nv_humanoid/joints/right_elbow.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/right_elbow.schemas.len`: shim=`1` openusd=`0`
- `@/nv_humanoid/joints/right_foot.attrs.drive:rotX:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/right_foot.attrs.drive:rotY:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/right_foot.schemas.len`: shim=`10` openusd=`8`
- `@/nv_humanoid/joints/right_knee.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/right_knee.schemas.len`: shim=`1` openusd=`0`
- `@/nv_humanoid/joints/right_thigh.attrs.drive:rotX:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/right_thigh.attrs.drive:rotY:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/right_thigh.attrs.drive:rotZ:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/right_thigh.schemas.len`: shim=`12` openusd=`9`
- `@/nv_humanoid/joints/right_upper_arm.attrs.drive:rotX:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/right_upper_arm.attrs.drive:rotZ:physics:type.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/joints/right_upper_arm.schemas.len`: shim=`10` openusd=`8`
- `@/nv_humanoid/left_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/left_foot/collisions/left_left_foot.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot/collisions/left_left_foot.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot/collisions/left_left_foot.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot/collisions/left_left_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot/collisions/right_left_foot.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot/collisions/right_left_foot.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot/collisions/right_left_foot.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot/collisions/right_left_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot/visuals/left_left_foot.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot/visuals/left_left_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot/visuals/right_left_foot.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_foot/visuals/right_left_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_hand.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_hand.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/left_hand/collisions/left_hand.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_hand/collisions/left_hand.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_hand/collisions/left_hand.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_hand/visuals/left_hand.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_lower_arm.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_lower_arm.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/left_lower_arm/collisions/left_lower_arm.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_lower_arm/collisions/left_lower_arm.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_lower_arm/collisions/left_lower_arm.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_lower_arm/collisions/left_lower_arm.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_lower_arm/visuals/left_lower_arm.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_lower_arm/visuals/left_lower_arm.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_shin.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_shin.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/left_shin/collisions/left_shin.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_shin/collisions/left_shin.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_shin/collisions/left_shin.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_shin/collisions/left_shin.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_shin/visuals/left_shin.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_shin/visuals/left_shin.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_thigh.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_thigh.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/left_thigh/collisions/left_thigh.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_thigh/collisions/left_thigh.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_thigh/collisions/left_thigh.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_thigh/collisions/left_thigh.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_thigh/visuals/left_thigh.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_thigh/visuals/left_thigh.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_upper_arm.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_upper_arm.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/left_upper_arm/collisions/left_upper_arm.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_upper_arm/collisions/left_upper_arm.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_upper_arm/collisions/left_upper_arm.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_upper_arm/collisions/left_upper_arm.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_upper_arm/visuals/left_upper_arm.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/left_upper_arm/visuals/left_upper_arm.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/lower_waist.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/lower_waist.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/lower_waist/collisions/lower_waist.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/lower_waist/collisions/lower_waist.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/lower_waist/collisions/lower_waist.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/lower_waist/collisions/lower_waist.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/lower_waist/visuals/lower_waist.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/lower_waist/visuals/lower_waist.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/pelvis.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/pelvis.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/pelvis/collisions/butt.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/pelvis/collisions/butt.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/pelvis/collisions/butt.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/pelvis/collisions/butt.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/pelvis/visuals/butt.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/pelvis/visuals/butt.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/right_foot/collisions/left_right_foot.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot/collisions/left_right_foot.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot/collisions/left_right_foot.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot/collisions/left_right_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot/collisions/right_right_foot.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot/collisions/right_right_foot.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot/collisions/right_right_foot.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot/collisions/right_right_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot/visuals/left_right_foot.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot/visuals/left_right_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot/visuals/right_right_foot.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_foot/visuals/right_right_foot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_hand.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_hand.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/right_hand/collisions/right_hand.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_hand/collisions/right_hand.attrs.purpose.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_hand/collisions/right_hand.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_hand/visuals/right_hand.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_lower_arm.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/nv_humanoid/right_lower_arm.schemas.len`: shim=`3` openusd=`2`
- `@/nv_humanoid/right_lower_arm/collisions/right_lower_arm.attrs.axis.variability`: shim=`varying` openusd=`uniform`
- … 46 more

</details>

## `revolute_articulation` — details

<details><summary>value mismatches</summary>

- `@/Articulation.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Articulation.schemas.len`: shim=`2` openusd=`1`
- `@/Articulation/Arm.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Articulation/Arm.schemas.len`: shim=`5` openusd=`3`
- `@/Articulation/Arm/RevoluteJoint.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/Articulation/CenterPivot.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Articulation/CenterPivot.schemas.len`: shim=`3` openusd=`2`
- `@/physicsScene.attrs.physxScene:broadphaseType.variability`: shim=`varying` openusd=`uniform`
- `@/physicsScene.attrs.physxScene:solverType.variability`: shim=`varying` openusd=`uniform`
- `@/physicsScene.schemas.len`: shim=`1` openusd=`0`

</details>

## `simple_articulation_with_mesh` — details

<details><summary>value mismatches</summary>

- `@/Robot.schemas.len`: shim=`2` openusd=`1`
- `@/Robot/joints.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Robot/joints/RevoluteJoint.attrs.physics:axis.variability`: shim=`varying` openusd=`uniform`
- `@/Robot/left_rigid.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Robot/left_rigid.schemas.len`: shim=`2` openusd=`1`
- `@/Robot/left_rigid/left_collider.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/Robot/left_rigid/left_collider.attrs.subdivisionScheme.variability`: shim=`varying` openusd=`uniform`
- `@/Robot/left_rigid/left_collider.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Robot/left_rigid/left_collider.schemas.len`: shim=`4` openusd=`2`
- `@/Robot/right_rigid.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Robot/right_rigid.schemas.len`: shim=`3` openusd=`2`
- `@/Robot/right_rigid/right_collider.attrs.physics:approximation.variability`: shim=`varying` openusd=`uniform`
- `@/Robot/right_rigid/right_collider.attrs.subdivisionScheme.variability`: shim=`varying` openusd=`uniform`
- `@/Robot/right_rigid/right_collider.attrs.xformOpOrder.variability`: shim=`varying` openusd=`uniform`
- `@/Robot/right_rigid/right_collider.schemas.len`: shim=`4` openusd=`2`

</details>
