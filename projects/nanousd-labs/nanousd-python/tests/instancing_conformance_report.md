# Instancing conformance: nanousd vs OpenUSD

Each fixture is a tiny instancing repro walked through both backends (`dump_instancing.py`): native/scene-graph instancing, point instancers, nested instancing, instance proxies, per-prototype mesh enumeration, and composed world transforms. A divergence here is a subtle instancing bug.

| fixture | value-mismatch | shim-missing (vs OpenUSD) | shim-extra |
|---|--:|--:|--:|
| `bare_pointinstancer` | **2** | 8 | 0 |
| `mixed_protoindices` | **2** | 13 | 0 |
| `native_class` | **5** | 4 | 2 |
| `native_deactivated` | **4** | 5 | 0 |
| `native_external` | **5** | 4 | 0 |
| `native_multi` | **5** | 7 | 0 |
| `native_over` | **7** | 6 | 0 |
| `nested_instance_in_instance` | **6** | 14 | 0 |
| `nested_pi_in_proto` | **6** | 20 | 0 |
| `nested_pi_proto_instanceable` | **6** | 11 | 0 |
| `pawn_pointinstancer` | **2** | 12 | 0 |
| `pi_invisible_ids` | **2** | 8 | 0 |
| `pi_pawn_multimesh` | **2** | 10 | 0 |
| `pi_payload` | **2** | 10 | 0 |
| `pi_single_proto` | **2** | 8 | 0 |
| `pi_two_protos` | **6** | 9 | 0 |

**Totals across 16 fixtures:** 64 value-mismatches, 149 surfaces present in OpenUSD but missing in the shim, 2 shim-only.

## `bare_pointinstancer`
- **mismatch** `@/World/Empty.pointinstancer.is_a_pointinstancer`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`4` openusd=`5`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Empty.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Empty.pointinstancer.computed_transforms`: openusd=`{'n': 2, 'values': [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]]]}`
- **shim-missing** `@/World/Empty.pointinstancer.instance_count`: openusd=`2`
- **shim-missing** `@/World/Empty.pointinstancer.mask`: openusd=`[]`
- **shim-missing** `@/World/Empty/Protos.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Empty/Protos/P.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `mixed_protoindices`
- **mismatch** `@/World/Mixed.pointinstancer.is_a_pointinstancer`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`9` openusd=`10`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Mixed.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Mixed.pointinstancer.computed_transforms`: openusd=`{'n': 4, 'values': [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [4.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [6.0, 0.0, 0.0, 1.0]]]}`
- **shim-missing** `@/World/Mixed.pointinstancer.instance_count`: openusd=`4`
- **shim-missing** `@/World/Mixed.pointinstancer.mask`: openusd=`[]`
- **shim-missing** `@/World/Mixed/Protos.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Mixed/Protos/EmptyProto.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Mixed/Protos/PawnProto.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Mixed/Protos/PawnProto/Geom_Body.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Mixed/Protos/PawnProto/Geom_Body/Render.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Mixed/Protos/PawnProto/Geom_Top.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Mixed/Protos/PawnProto/Geom_Top/Render.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `native_class`
- **mismatch** `@/World/A.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@/World/B.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`7` openusd=`6`
- **mismatch** `@__prototypes__.count`: shim=`0` openusd=`1`
- **mismatch** `@__prototypes__.shapes.len`: shim=`0` openusd=`1`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/A.is_prototype`: openusd=`False`
- **shim-missing** `@/World/B.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`
- **shim-extra** `@__instance_proxies__.detail./World/_ProtoClass`: shim=`{'displayColor': None, 'purpose': 'default', 'type': 'Xform', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`
- **shim-extra** `@__instance_proxies__.detail./World/_ProtoClass/Geom`: shim=`{'displayColor': None, 'purpose': 'default', 'type': 'Mesh', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `native_deactivated`
- **mismatch** `@/World/A.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`5` openusd=`6`
- **mismatch** `@__prototypes__.count`: shim=`0` openusd=`1`
- **mismatch** `@__prototypes__.shapes.len`: shim=`0` openusd=`1`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/A.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto/Geom.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `native_external`
- **mismatch** `@/World/A.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@/World/B.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`7` openusd=`8`
- **mismatch** `@__prototypes__.count`: shim=`0` openusd=`1`
- **mismatch** `@__prototypes__.shapes.len`: shim=`0` openusd=`1`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/A.is_prototype`: openusd=`False`
- **shim-missing** `@/World/B.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `native_multi`
- **mismatch** `@/World/A.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@/World/B.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`10` openusd=`11`
- **mismatch** `@__prototypes__.count`: shim=`0` openusd=`1`
- **mismatch** `@__prototypes__.shapes.len`: shim=`0` openusd=`1`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/A.is_prototype`: openusd=`False`
- **shim-missing** `@/World/B.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto/Body.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto/Top.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `native_over`
- **mismatch** `@/World/A.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@/World/B.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`7` openusd=`8`
- **mismatch** `@__instance_proxies__.detail./World/A/Geom.displayColor[0][0]`: shim=`0.0` openusd=`1.0`
- **mismatch** `@__instance_proxies__.detail./World/A/Geom.displayColor[0][1]`: shim=`1.0` openusd=`0.0`
- **mismatch** `@__prototypes__.count`: shim=`0` openusd=`1`
- **mismatch** `@__prototypes__.shapes.len`: shim=`0` openusd=`1`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/A.is_prototype`: openusd=`False`
- **shim-missing** `@/World/B.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto/Geom.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `nested_instance_in_instance`
- **mismatch** `@/World/A.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@/World/B.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@/World/Outer/InnerInstance.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`15` openusd=`20`
- **mismatch** `@__prototypes__.count`: shim=`0` openusd=`2`
- **mismatch** `@__prototypes__.shapes.len`: shim=`0` openusd=`2`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/A.is_prototype`: openusd=`False`
- **shim-missing** `@/World/B.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Leaf.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Leaf/Geom_Body.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Leaf/Geom_Top.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Outer.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Outer/InnerInstance.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Outer/OuterShell.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`
- **shim-missing** `@__instance_proxies__.detail./World/A/InnerInstance/Geom_Body`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': 'Mesh', 'visibility': 'inherited', 'world_translate': [5.0, 0.0, 1.0]}`
- **shim-missing** `@__instance_proxies__.detail./World/A/InnerInstance/Geom_Top`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': 'Mesh', 'visibility': 'inherited', 'world_translate': [5.0, 0.0, 1.0]}`
- **shim-missing** `@__instance_proxies__.detail./World/B/InnerInstance/Geom_Body`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': 'Mesh', 'visibility': 'inherited', 'world_translate': [10.0, 0.0, 1.0]}`
- **shim-missing** `@__instance_proxies__.detail./World/B/InnerInstance/Geom_Top`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': 'Mesh', 'visibility': 'inherited', 'world_translate': [10.0, 0.0, 1.0]}`

## `nested_pi_in_proto`
- **mismatch** `@/World/A.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@/World/B.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@/World/Proto/PI.pointinstancer.is_a_pointinstancer`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`18` openusd=`23`
- **mismatch** `@__prototypes__.count`: shim=`0` openusd=`1`
- **mismatch** `@__prototypes__.shapes.len`: shim=`0` openusd=`1`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/A.is_prototype`: openusd=`False`
- **shim-missing** `@/World/B.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PawnProto.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PawnProto/Geom_Body.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PawnProto/Geom_Top.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto/PI.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto/PI.pointinstancer.computed_transforms`: openusd=`{'n': 2, 'values': [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [3.0, 0.0, 0.0, 1.0]]]}`
- **shim-missing** `@/World/Proto/PI.pointinstancer.instance_count`: openusd=`2`
- **shim-missing** `@/World/Proto/PI.pointinstancer.mask`: openusd=`[]`
- **shim-missing** `@/World/Proto/PI/Prototypes.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto/PI/Prototypes/Pawn.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto/PI/Prototypes/Pawn/Geom_Body.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Proto/PI/Prototypes/Pawn/Geom_Top.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`
- **shim-missing** `@__instance_proxies__.detail./World/A/PI/Prototypes/Pawn/Geom_Body`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': 'Mesh', 'visibility': 'inherited', 'world_translate': [5.0, 0.0, 0.0]}`
- **shim-missing** `@__instance_proxies__.detail./World/A/PI/Prototypes/Pawn/Geom_Top`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': 'Mesh', 'visibility': 'inherited', 'world_translate': [5.0, 0.0, 0.0]}`
- **shim-missing** `@__instance_proxies__.detail./World/B/PI/Prototypes/Pawn/Geom_Body`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': 'Mesh', 'visibility': 'inherited', 'world_translate': [10.0, 0.0, 0.0]}`
- **shim-missing** `@__instance_proxies__.detail./World/B/PI/Prototypes/Pawn/Geom_Top`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': 'Mesh', 'visibility': 'inherited', 'world_translate': [10.0, 0.0, 0.0]}`

## `nested_pi_proto_instanceable`
- **mismatch** `@/World/PI.pointinstancer.is_a_pointinstancer`: shim=`False` openusd=`True`
- **mismatch** `@/World/PI.pointinstancer.proto_mesh_counts./World/PI/Prototypes/Pawn`: shim=`2` openusd=`0`
- **mismatch** `@/World/PI/Prototypes/Pawn.is_instanceable`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`9` openusd=`10`
- **mismatch** `@__prototypes__.count`: shim=`0` openusd=`1`
- **mismatch** `@__prototypes__.shapes.len`: shim=`0` openusd=`1`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI.pointinstancer.computed_transforms`: openusd=`{'n': 3, 'values': [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [3.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [6.0, 0.0, 0.0, 1.0]]]}`
- **shim-missing** `@/World/PI.pointinstancer.instance_count`: openusd=`3`
- **shim-missing** `@/World/PI.pointinstancer.mask`: openusd=`[]`
- **shim-missing** `@/World/PI/Prototypes.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI/Prototypes/Pawn.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PawnSource.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PawnSource/Geom_Body.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PawnSource/Geom_Top.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `pawn_pointinstancer`
- **mismatch** `@/World/Pawns.pointinstancer.is_a_pointinstancer`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`8` openusd=`9`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns.pointinstancer.computed_transforms`: openusd=`{'n': 8, 'values': [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [4.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [6.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 2.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [2.0, 0.0, 2.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [4.0, 0.0, 2.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [6.0, 0.0, 2.0, 1.0]]]}`
- **shim-missing** `@/World/Pawns.pointinstancer.instance_count`: openusd=`8`
- **shim-missing** `@/World/Pawns.pointinstancer.mask`: openusd=`[]`
- **shim-missing** `@/World/Pawns/Prototypes.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns/Prototypes/Pawn.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns/Prototypes/Pawn/Geom_Body.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns/Prototypes/Pawn/Geom_Body/Render.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns/Prototypes/Pawn/Geom_Top.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns/Prototypes/Pawn/Geom_Top/Render.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `pi_invisible_ids`
- **mismatch** `@/World/PI.pointinstancer.is_a_pointinstancer`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`4` openusd=`5`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI.pointinstancer.computed_transforms`: openusd=`{'n': 3, 'values': [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [4.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [8.0, 0.0, 0.0, 1.0]]]}`
- **shim-missing** `@/World/PI.pointinstancer.instance_count`: openusd=`5`
- **shim-missing** `@/World/PI.pointinstancer.mask`: openusd=`[True, False, True, False, True]`
- **shim-missing** `@/World/PI/Prototypes.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI/Prototypes/Proto.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `pi_pawn_multimesh`
- **mismatch** `@/World/PI.pointinstancer.is_a_pointinstancer`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`6` openusd=`7`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI.pointinstancer.computed_transforms`: openusd=`{'n': 2, 'values': [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [4.0, 0.0, 0.0, 1.0]]]}`
- **shim-missing** `@/World/PI.pointinstancer.instance_count`: openusd=`2`
- **shim-missing** `@/World/PI.pointinstancer.mask`: openusd=`[]`
- **shim-missing** `@/World/PI/Prototypes.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI/Prototypes/Pawn.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI/Prototypes/Pawn/Geom_Body.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI/Prototypes/Pawn/Geom_Top.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `pi_payload`
- **mismatch** `@/World/Pawns.pointinstancer.is_a_pointinstancer`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`6` openusd=`7`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns.pointinstancer.computed_transforms`: openusd=`{'n': 3, 'values': [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [3.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [6.0, 0.0, 0.0, 1.0]]]}`
- **shim-missing** `@/World/Pawns.pointinstancer.instance_count`: openusd=`3`
- **shim-missing** `@/World/Pawns.pointinstancer.mask`: openusd=`[]`
- **shim-missing** `@/World/Pawns/Prototypes.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns/Prototypes/Pawn.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns/Prototypes/Pawn/Body.is_prototype`: openusd=`False`
- **shim-missing** `@/World/Pawns/Prototypes/Pawn/Head.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `pi_single_proto`
- **mismatch** `@/World/PI.pointinstancer.is_a_pointinstancer`: shim=`False` openusd=`True`
- **mismatch** `@__instance_proxies__.count`: shim=`4` openusd=`5`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI.pointinstancer.computed_transforms`: openusd=`{'n': 3, 'values': [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [3.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [6.0, 0.0, 0.0, 1.0]]]}`
- **shim-missing** `@/World/PI.pointinstancer.instance_count`: openusd=`3`
- **shim-missing** `@/World/PI.pointinstancer.mask`: openusd=`[]`
- **shim-missing** `@/World/PI/Prototypes.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI/Prototypes/Proto.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`

## `pi_two_protos`
- **mismatch** `@/World/PI.pointinstancer.is_a_pointinstancer`: shim=`False` openusd=`True`
- **mismatch** `@/World/PI.pointinstancer.orientations[0].len`: shim=`4` openusd=`5`
- **mismatch** `@/World/PI.pointinstancer.orientations[1].len`: shim=`4` openusd=`5`
- **mismatch** `@/World/PI.pointinstancer.orientations[2].len`: shim=`4` openusd=`5`
- **mismatch** `@/World/PI.pointinstancer.orientations[3].len`: shim=`4` openusd=`5`
- **mismatch** `@__instance_proxies__.count`: shim=`5` openusd=`6`
- **shim-missing** `@/World.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI.pointinstancer.computed_transforms`: openusd=`{'n': 4, 'values': [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[1.41457, 0.0, -1.41371, 0.0], [0.0, 1.0, 0.0, 0.0], [0.35343, 0.0, 0.35364, 0.0], [2.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [4.0, 0.0, 0.0, 1.0]], [[0.00011, 0.0, -0.49989, 0.0], [0.0, 0.5, 0.0, 0.0], [0.49989, 0.0, 0.00011, 0.0], [6.0, 0.0, 0.0, 1.0]]]}`
- **shim-missing** `@/World/PI.pointinstancer.instance_count`: openusd=`4`
- **shim-missing** `@/World/PI.pointinstancer.mask`: openusd=`[]`
- **shim-missing** `@/World/PI/Prototypes.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI/Prototypes/Cube.is_prototype`: openusd=`False`
- **shim-missing** `@/World/PI/Prototypes/Tri.is_prototype`: openusd=`False`
- **shim-missing** `@__instance_proxies__.detail./`: openusd=`{'displayColor': None, 'purpose': 'default', 'type': '', 'visibility': 'inherited', 'world_translate': [0.0, 0.0, 0.0]}`
