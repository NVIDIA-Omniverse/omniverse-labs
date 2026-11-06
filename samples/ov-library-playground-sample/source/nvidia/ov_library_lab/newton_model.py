# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original, metre-scale commissioning rig consumed by ovnewton (no host physics)."""

GANTRY_USDA = '''#usda 1.0
(
    defaultPrim = "Gantry"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Y"
)
def PhysicsScene "Physics" {
    vector3f physics:gravityDirection = (0, -1, 0)
    float physics:gravityMagnitude = 9.81
}
def Xform "Gantry" (prepend apiSchemas = ["NewtonArticulationRootAPI"]) {
    bool newton:selfCollisionEnabled = false
    def Xform "Rail" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        double3 xformOp:translate = (0, 3.6, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 500
        float3 physics:diagonalInertia = (10, 500, 500)
    }
    def PhysicsFixedJoint "Anchor" {
        rel physics:body1 = </Gantry/Rail>
    }
    def Xform "Trolley" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        double3 xformOp:translate = (-1.6, 3.6, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 120
        float3 physics:diagonalInertia = (8, 8, 8)
        def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
            double size = 1
            float3 xformOp:scale = (0.8, 0.3, 0.6)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }
    }
    def Xform "Load" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        double3 xformOp:translate = (-1.6, 1.9, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 35
        float3 physics:diagonalInertia = (7, 5, 7)
        def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
            double size = 1
            float3 xformOp:scale = (1.05, 0.7, 0.8)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }
    }
    def PhysicsPrismaticJoint "Travel" (prepend apiSchemas = ["PhysicsDriveAPI:linear", "PhysicsJointStateAPI:linear"]) {
        rel physics:body0 = </Gantry/Rail>
        rel physics:body1 = </Gantry/Trolley>
        uniform token physics:axis = "X"
        float physics:lowerLimit = -2.7
        float physics:upperLimit = 2.7
        float state:linear:physics:position = -1.6
        float state:linear:physics:velocity = 0
        float drive:linear:physics:targetPosition = -1.6
        float drive:linear:physics:targetVelocity = 0
        float drive:linear:physics:stiffness = 45000
        float drive:linear:physics:damping = 1800
        float drive:linear:physics:maxForce = 10000
    }
    def PhysicsRevoluteJoint "Suspension" (prepend apiSchemas = ["PhysicsJointStateAPI:angular"]) {
        rel physics:body0 = </Gantry/Trolley>
        rel physics:body1 = </Gantry/Load>
        uniform token physics:axis = "Z"
        point3f physics:localPos1 = (0, 1.7, 0)
        float physics:lowerLimit = -75
        float physics:upperLimit = 75
        float state:angular:physics:position = 0
        float state:angular:physics:velocity = 0
    }
}
'''
