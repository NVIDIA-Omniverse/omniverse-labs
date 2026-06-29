# OpenUSD schema tests vs nanousd `pxr_compat`

Upstream OpenUSD unit tests run against the shim. Report-only — the
tally is the conformance matrix, not a pass/fail gate.

| schema | test module | ran | pass | fail | error | status |
|---|---|--:|--:|--:|--:|---|
| usdPhysics | `testUsdPhysicsCollisionGroupAPI` | 4 | 0 | 0 | 4 | ❌ none pass |
| usdPhysics | `testUsdPhysicsMetrics` | 1 | 0 | 0 | 1 | ❌ none pass |
| usdPhysics | `testUsdPhysicsParsing` | 19 | 0 | 1 | 19 | ❌ none pass |
| usdPhysics | `testUsdPhysicsRigidBodyAPI` | 20 | 0 | 3 | 17 | ❌ none pass |
| usdGeom | `testUsdGeomBBoxCache` | 0 | 0 | 0 | 0 | ❌ none pass |
| usdGeom | `testUsdGeomBasisCurves` | 3 | 0 | 0 | 3 | ❌ none pass |
| usdGeom | `testUsdGeomCamera` | 4 | 0 | 0 | 4 | ❌ none pass |
| usdGeom | `testUsdGeomComputeAtTime` | – | – | – | – | ❌ load: AttributeError: type object '_PointInstancer' has no attribute 'IncludeProtoXfor |
| usdGeom | `testUsdGeomConstraintTarget` | 1 | 0 | 0 | 1 | ❌ none pass |
| usdGeom | `testUsdGeomConsts` | 1 | 0 | 1 | 0 | ❌ none pass |
| usdGeom | `testUsdGeomCurves` | 4 | 0 | 0 | 4 | ❌ none pass |
| usdGeom | `testUsdGeomExtentFromPlugins` | 3 | 0 | 0 | 3 | ❌ none pass |
| usdGeom | `testUsdGeomExtentTransform` | 9 | 0 | 0 | 9 | ❌ none pass |
| usdGeom | `testUsdGeomHermiteCurves` | 3 | 0 | 0 | 3 | ❌ none pass |
| usdGeom | `testUsdGeomImageable` | 1 | 0 | 1 | 0 | ❌ none pass |
| usdGeom | `testUsdGeomMesh` | 2 | 0 | 0 | 2 | ❌ none pass |
| usdGeom | `testUsdGeomMetrics` | 2 | 0 | 0 | 2 | ❌ none pass |
| usdGeom | `testUsdGeomMotionAPI` | 1 | 0 | 0 | 1 | ❌ none pass |
| usdGeom | `testUsdGeomNoPlugLoad` | 1 | 0 | 0 | 1 | ❌ none pass |
| usdGeom | `testUsdGeomPointInstancer` | 9 | 0 | 0 | 9 | ❌ none pass |
| usdGeom | `testUsdGeomPoints` | 1 | 0 | 0 | 1 | ❌ none pass |
| usdGeom | `testUsdGeomPrimvar` | 6 | 0 | 1 | 5 | ❌ none pass |
| usdGeom | `testUsdGeomPurposeVisibility` | 6 | 0 | 1 | 5 | ❌ none pass |
| usdGeom | `testUsdGeomSchemata` | 16 | 0 | 2 | 14 | ❌ none pass |
| usdGeom | `testUsdGeomSubset` | 5 | 0 | 0 | 5 | ❌ none pass |
| usdGeom | `testUsdGeomTetMesh` | 3 | 0 | 0 | 3 | ❌ none pass |
| usdGeom | `testUsdGeomTypeRegistry` | 3 | 0 | 3 | 0 | ❌ none pass |
| usdGeom | `testUsdGeomXformCommonAPI` | 11 | 0 | 0 | 11 | ❌ none pass |
| usdGeom | `testUsdGeomXformable` | 28 | 2 | 2 | 24 | ⚠️ partial |
| usdSkel | `testUsdSkelAnimMapper` | 4 | 0 | 0 | 4 | ❌ none pass |
| usdSkel | `testUsdSkelAnimQuery` | 1 | 0 | 0 | 1 | ❌ none pass |
| usdSkel | `testUsdSkelBakeSkinning` | 6 | 0 | 5 | 1 | ❌ none pass |
| usdSkel | `testUsdSkelBindingAPI` | 1 | 0 | 0 | 1 | ❌ none pass |
| usdSkel | `testUsdSkelBlendShape` | 2 | 0 | 0 | 2 | ❌ none pass |
| usdSkel | `testUsdSkelCache` | 5 | 0 | 0 | 5 | ❌ none pass |
| usdSkel | `testUsdSkelRoot` | 1 | 0 | 0 | 1 | ❌ none pass |
| usdSkel | `testUsdSkelSkeletonQuery` | 1 | 0 | 0 | 1 | ❌ none pass |
| usdSkel | `testUsdSkelSkinningQuery` | 1 | 0 | 0 | 1 | ❌ none pass |
| usdSkel | `testUsdSkelTopology` | 2 | 0 | 1 | 1 | ❌ none pass |
| usdSkel | `testUsdSkelUtils` | 9 | 4 | 4 | 1 | ⚠️ partial |

## Per-schema totals

| schema | ran | pass | fail | error |
|---|--:|--:|--:|--:|
| usdPhysics | 44 | 0 | 4 | 41 |
| usdGeom | 123 | 2 | 11 | 110 |
| usdSkel | 33 | 4 | 10 | 19 |

## Representative failures (first few per module)

### `testUsdPhysicsCollisionGroupAPI`
- test_collision_group_complex_merging :: AttributeError: '_UsdPhysicsModule' object has no attribute 'CollisionGroup'
- test_collision_group_inversion :: AttributeError: '_UsdPhysicsModule' object has no attribute 'CollisionGroup'
- test_collision_group_simple_merging :: AttributeError: '_UsdPhysicsModule' object has no attribute 'CollisionGroup'
- test_collision_group_table :: AttributeError: '_UsdPhysicsModule' object has no attribute 'CollisionGroup'

### `testUsdPhysicsMetrics`
- test_kilogramsPerUnit :: AttributeError: '_UsdPhysicsModule' object has no attribute 'MassUnits'

### `testUsdPhysicsParsing`
- setUpClass :: TypeError: object of type 'NoneType' has no len()
- test_articulation_kinematic_parse :: AttributeError: '_RigidBodyAPI' object has no attribute 'GetKinematicEnabledAttr'. Did you mean: 'CreateKinematicEnabledAttr'?
- test_articulation_parse :: AttributeError: '_ArticulationDesc' object has no attribute 'rootPrims'
- test_articulation_simulation_owner_parse :: AttributeError: '_UsdPhysicsModule' object has no attribute 'CustomUsdPhysicsTokens'
- test_collision_groups_collider_parse :: AttributeError: '_UsdPhysicsModule' object has no attribute 'CollisionGroup'
- test_collision_groups_parse :: AttributeError: '_UsdPhysicsModule' object has no attribute 'CollisionGroup'

### `testUsdPhysicsRigidBodyAPI`
- test_mass_rigid_body_cube_collider_com :: AttributeError: 'numpy.ndarray' object has no attribute 'SetDiagonal'. Did you mean: 'diagonal'?
- test_mass_rigid_body_cube_collider_density :: AttributeError: 'numpy.ndarray' object has no attribute 'SetDiagonal'. Did you mean: 'diagonal'?
- test_mass_rigid_body_cube_collider_inertia :: AttributeError: 'numpy.ndarray' object has no attribute 'SetDiagonal'. Did you mean: 'diagonal'?
- test_mass_rigid_body_cube_collider_mass :: AttributeError: 'numpy.ndarray' object has no attribute 'SetDiagonal'. Did you mean: 'diagonal'?
- test_mass_rigid_body_cube_com_precedence :: AttributeError: 'numpy.ndarray' object has no attribute 'SetDiagonal'. Did you mean: 'diagonal'?
- test_mass_rigid_body_cube_compound :: AttributeError: 'numpy.ndarray' object has no attribute 'SetDiagonal'. Did you mean: 'diagonal'?

### `testUsdGeomBasisCurves`
- test_ComputeCurveCount :: AttributeError: '_BasisCurves' object has no attribute 'GetCurveCount'
- test_ComputeSegmentCount :: AttributeError: '_BasisCurves' object has no attribute 'ComputeSegmentCounts'
- test_InterpolationTypes :: AttributeError: '_BasisCurves' object has no attribute 'ComputeUniformDataSize'

### `testUsdGeomCamera`
- test_ComputeLinearExposureScale :: AttributeError: 'Camera' object has no attribute 'GetExposureAttr'
- test_GetCamera :: AttributeError: 'Camera' object has no attribute 'GetLocalTransformation'
- test_SetFromCamera :: AttributeError: 'Camera' object has no attribute 'GetProjectionAttr'
- test_SetFromCameraWithComposition :: AttributeError: 'Camera' object has no attribute 'SetFromCamera'

### `testUsdGeomComputeAtTime`
- AttributeError: type object '_PointInstancer' has no attribute 'IncludeProtoXform'

### `testUsdGeomConstraintTarget`
- test_Basic :: TypeError: only 0-dimensional arrays can be converted to Python scalars

### `testUsdGeomConsts`
- test_Basic :: AssertionError: 10.0 != 9.999999680285692e+37 : Round-tripped constant 'SHARPNESS_INFINITE' did not compare equal to schema constant. (schema: 10.0, roundtrippe

### `testUsdGeomCurves`
- test_create :: RuntimeError: Failed to open USD file: anon:0xf06ab40-1: Failed to parse anon:0xf06ab40-1: Unsupported resource scheme: anon
- test_schema :: RuntimeError: Failed to open USD file: anon:0xf06ab40-2: Failed to parse anon:0xf06ab40-2: Unsupported resource scheme: anon
- test_create :: RuntimeError: Failed to open USD file: anon:0xf06ab40-3: Failed to parse anon:0xf06ab40-3: Unsupported resource scheme: anon
- test_schema :: RuntimeError: Failed to open USD file: anon:0xf06ab40-4: Failed to parse anon:0xf06ab40-4: Unsupported resource scheme: anon

### `testUsdGeomExtentFromPlugins`
- test_ComputeExtentAndExtentFromPlugin :: AttributeError: 'Boundable' object has no attribute 'ComputeExtent'
- test_Default :: AttributeError: type object 'Boundable' has no attribute 'ComputeExtentFromPlugins'
- test_TimeSampled :: AttributeError: type object 'Boundable' has no attribute 'ComputeExtentFromPlugins'

### `testUsdGeomExtentTransform`
- test_Capsule :: AttributeError: type object 'Boundable' has no attribute 'ComputeExtentFromPlugins'
- test_Cone :: AttributeError: type object 'Boundable' has no attribute 'ComputeExtentFromPlugins'
- test_Cube :: AttributeError: type object 'Boundable' has no attribute 'ComputeExtentFromPlugins'
- test_Curves :: AttributeError: type object 'Boundable' has no attribute 'ComputeExtentFromPlugins'
- test_Cylinder :: AttributeError: type object 'Boundable' has no attribute 'ComputeExtentFromPlugins'
- test_PointBased :: AttributeError: type object 'Boundable' has no attribute 'ComputeExtentFromPlugins'

### `testUsdGeomHermiteCurves`
- testInterleave :: AttributeError: '_UsdGeomModule' object has no attribute 'HermiteCurves'
- testPointAndTangents :: AttributeError: '_UsdGeomModule' object has no attribute 'HermiteCurves'
- testSeparate :: AttributeError: '_UsdGeomModule' object has no attribute 'HermiteCurves'

### `testUsdGeomImageable`
- test_MakeVisible :: + inherited

### `testUsdGeomMesh`
- test_ComputeFaceCount :: AttributeError: '_Mesh' object has no attribute 'GetFaceCount'
- test_ValidateTopology :: AttributeError: type object '_Mesh' has no attribute 'ValidateTopology'

### `testUsdGeomMetrics`
- test_metersPerUnit :: AttributeError: '_UsdGeomModule' object has no attribute 'LinearUnits'
- test_upAxis :: AttributeError: '_UsdGeomModule' object has no attribute 'GetFallbackUpAxis'

### `testUsdGeomMotionAPI`
- test_Basic :: AttributeError: '_UsdGeomModule' object has no attribute 'MotionAPI'

### `testUsdGeomNoPlugLoad`
- test_Basic :: AttributeError: '_UsdModule' object has no attribute 'Typed'

### `testUsdGeomPointInstancer`
- test_BBoxCache :: AttributeError: '_Xformable' object has no attribute 'AddRotateZOp'. Did you mean: 'AddRotateXYZOp'?
- test_ComputeInstancerCount :: AttributeError: '_Xformable' object has no attribute 'AddRotateZOp'. Did you mean: 'AddRotateXYZOp'?
- test_ExtentMultipleCubeInstances :: AttributeError: '_Xformable' object has no attribute 'AddRotateZOp'. Did you mean: 'AddRotateXYZOp'?
- test_ExtentOneOriginCubeInstance :: AttributeError: '_Xformable' object has no attribute 'AddRotateZOp'. Did you mean: 'AddRotateXYZOp'?
- test_ExtentOneOriginScaledCubeInstance :: AttributeError: '_Xformable' object has no attribute 'AddRotateZOp'. Did you mean: 'AddRotateXYZOp'?
- test_ExtentOneRotatedCubeInstance :: AttributeError: '_Xformable' object has no attribute 'AddRotateZOp'. Did you mean: 'AddRotateXYZOp'?

### `testUsdGeomPoints`
- test_ComputePointCount :: AttributeError: '_Points' object has no attribute 'GetPointCount'

### `testUsdGeomPrimvar`
- test_Bug124579 :: AttributeError: '_UsdPrim' object has no attribute 'has_attribute'. Did you mean: 'HasAttribute'?
- test_InvalidPrimvar :: TypeError: _PrimvarProxy.__init__() missing 1 required positional argument: 'prim_or_attr'
- test_PrimvarIndicesBlock :: AttributeError: '_UsdPrim' object has no attribute 'has_attribute'. Did you mean: 'HasAttribute'?
- test_PrimvarInheritance :: AttributeError: '_UsdPrim' object has no attribute 'has_attribute'. Did you mean: 'HasAttribute'?
- test_PrimvarsAPI :: AttributeError: type object '_PrimvarProxy' has no attribute 'IsPrimvar'
- test_Hash :: AssertionError: <nanousd.pxr_compat._native_compat._PrimvarProxy object at 0x78ae00d6f530> is not true

### `testUsdGeomPurposeVisibility`
- test_ComputePurpose :: AttributeError: '_Imageable' object has no attribute 'GetPurposeAttr'
- test_ComputePurposeVisibility :: AttributeError: '_UsdGeomModule' object has no attribute 'VisibilityAPI'
- test_ComputeVisibility :: AttributeError: '_Imageable' object has no attribute 'GetPrim'
- test_MakeVisInvis :: AttributeError: '_Scope' object has no attribute 'ComputeVisibility'
- test_ProxyPrim :: AttributeError: '_Scope' object has no attribute 'CreatePurposeAttr'
- test_ComputePurposeVisibilityWithInstancing :: + visible

### `testUsdGeomSchemata`
- test_Apply :: AttributeError: '_UsdGeomModule' object has no attribute 'MotionAPI'
- test_Basic :: RuntimeError: Failed to open USD file: anon:0x346f0b30-1: Failed to parse anon:0x346f0b30-1: Unsupported resource scheme: anon
- test_BasicMetadataCases :: AttributeError: '_Attribute' object has no attribute 'HasMetadata'. Did you mean: 'GetMetadata'?
- test_Bug116593 :: AttributeError: '_UsdGeomModule' object has no attribute 'ModelAPI'
- test_Camera :: AttributeError: 'NoneType' object has no attribute 'IsA'
- test_ComputeExtent :: AttributeError: type object 'PointBased' has no attribute 'ComputeExtent'

### `testUsdGeomSubset`
- test_CreateGeomSubset :: AttributeError: type object '_Subset' has no attribute 'CreateGeomSubset'
- test_GetUnassignedIndicesForEdges :: AttributeError: type object '_Subset' has no attribute 'CreateGeomSubset'
- test_GetUnassignedIndicesForSegments :: AttributeError: type object '_Subset' has no attribute 'CreateGeomSubset'
- test_PointInstancer :: AttributeError: type object '_Subset' has no attribute 'GetGeomSubsets'
- test_SubsetRetrievalAndValidity :: AttributeError: type object '_Subset' has no attribute 'GetGeomSubsets'

### `testUsdGeomTetMesh`
- test_ComputeSurfaceExtractionFromUsdGeomTetMeshLeftHanded :: AttributeError: '_GfModule' object has no attribute 'Vec4i'. Did you mean: 'Vec4d'?
- test_ComputeSurfaceExtractionFromUsdGeomTetMeshRightHanded :: AttributeError: '_GfModule' object has no attribute 'Vec4i'. Did you mean: 'Vec4d'?
- test_UsdGeomTetMeshFindInvertedElements :: AttributeError: '_GfModule' object has no attribute 'Vec4i'. Did you mean: 'Vec4d'?

### `testUsdGeomTypeRegistry`
- test_AbstractTyped :: AssertionError: 'pxr.UsdGeom' unexpectedly found in {'sys': <module 'sys' (built-in)>, 'builtins': <module 'builtins' (built-in)>, '_frozen_importlib': <module 
- test_Applied :: AssertionError: 'pxr.UsdGeom' unexpectedly found in {'sys': <module 'sys' (built-in)>, 'builtins': <module 'builtins' (built-in)>, '_frozen_importlib': <module 
- test_ConcreteTyped :: AssertionError: 'pxr.UsdGeom' unexpectedly found in {'sys': <module 'sys' (built-in)>, 'builtins': <module 'builtins' (built-in)>, '_frozen_importlib': <module 

### `testUsdGeomXformCommonAPI`
- test_Bug116955 :: AttributeError: '_UsdGeomModule' object has no attribute 'XformCommonAPI'
- test_CreateXformOps :: AttributeError: '_UsdGeomModule' object has no attribute 'XformCommonAPI'
- test_Doulbe3AndFloat3PivotPosition :: AttributeError: '_UsdGeomModule' object has no attribute 'XformCommonAPI'
- test_EmptyXformable :: AttributeError: '_UsdGeomModule' object has no attribute 'XformCommonAPI'
- test_GetRotationTransform :: AttributeError: '_Xformable' object has no attribute 'AddXformOp'
- test_GetXformVectorsByAccumulation :: AttributeError: '_Xformable' object has no attribute 'AddRotateXOp'. Did you mean: 'AddRotateXYZOp'?

### `testUsdGeomXformable`
- test_GetTimeSamples :: AttributeError: '_Xformable' object has no attribute 'GetTimeSamples'
- test_GetXformOp :: AttributeError: '_Xformable' object has no attribute 'GetXformOp'
- test_ImplicitConversions :: AttributeError: '_Attribute' object has no attribute 'FlattenTo'
- test_InvalidXformOps :: TypeError: _XformOp.__init__() missing 2 required positional arguments: 'attr_name' and 'op_type'
- test_InvalidXformable :: AttributeError: 'NoneType' object has no attribute '_prim'
- test_InverseOps :: AttributeError: '_Xformable' object has no attribute 'AddRotateXOp'. Did you mean: 'AddRotateXYZOp'?

### `testUsdSkelAnimMapper`
- test_EmptyArrays :: AttributeError: 'NoneType' object has no attribute 'IsIdentity'
- test_Errors :: AttributeError: 'NoneType' object has no attribute 'Remap'
- test_IdentityRemapping :: AttributeError: 'NoneType' object has no attribute 'IsIdentity'
- test_Remapping :: AttributeError: 'NoneType' object has no attribute 'Remap'

### `testUsdSkelAnimQuery`
- test_SkelAnimation :: AttributeError: 'function' object has no attribute 'Define'

### `testUsdSkelBakeSkinning`
- test_BlendShapes :: RuntimeError: Failed to open USD file: blendshapes.usda: Failed to parse blendshapes.usda: Failed to open resource: blendshapes.usda
- test_BlendShapesWithNormals :: AssertionError: None is not true
- test_DQSkinning :: AssertionError: None is not true
- test_DQSkinningWithInterval :: AssertionError: None is not true
- test_LinearBlendSkinning :: AssertionError: None is not true
- test_LinearBlendSkinningWithInterval :: AssertionError: None is not true

### `testUsdSkelBindingAPI`
- test_JointInfluences :: AttributeError: 'NoneType' object has no attribute 'CreateJointIndicesPrimvar'

### `testUsdSkelBlendShape`
- test_BlendShape :: AttributeError: 'function' object has no attribute 'Define'
- test_BlendShapeWithInbetweens :: AttributeError: 'function' object has no attribute 'Define'

### `testUsdSkelCache`
- test_AnimQuery :: AttributeError: 'function' object has no attribute 'Define'
- test_InheritedAnimBinding :: AttributeError: 'NoneType' object has no attribute 'Populate'
- test_InheritedSkeletonBinding :: AttributeError: 'NoneType' object has no attribute 'Populate'
- test_InstancedBlendShape :: AttributeError: 'NoneType' object has no attribute 'Populate'
- test_InstancedSkeletonBinding :: AttributeError: 'NoneType' object has no attribute 'Populate'

### `testUsdSkelRoot`
- test_ComputeExtentPlugin :: AttributeError: type object 'Boundable' has no attribute 'ComputeExtentFromPlugins'

### `testUsdSkelSkeletonQuery`
- test_SkeletonQuery :: AttributeError: 'function' object has no attribute 'Define'

### `testUsdSkelSkinningQuery`
- test_JointInfluences :: AttributeError: 'NoneType' object has no attribute 'Populate'

### `testUsdSkelTopology`
- test_Topology :: AttributeError: 'NoneType' object has no attribute 'GetNumJoints'
- test_ValidateTopology :: AssertionError: None is not true

### `testUsdSkelUtils`
- test_TransformCompositionAndDecomposition :: TypeError: unsupported operand type(s) for *: 'Rotation' and 'Rotation'
- test_ExpandConstantInfluencesToVarying :: AssertionError
- test_NormalizeWeights :: AssertionError
- test_ResizeInfluences :: AssertionError
- test_SortInfluences :: AssertionError
