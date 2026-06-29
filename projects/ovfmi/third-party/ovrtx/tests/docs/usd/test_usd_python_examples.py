# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for USD Python API examples shown in doc USDA tabs."""

from pxr import Gf, Sdf, Usd, UsdRender


def test_camera_sensor_render_product():
    """Build the camera_sensors.rst RenderProduct using the USD Python API."""
    stage = Usd.Stage.CreateInMemory()

    # [snippet:doc-camera-sensor-render-product-python]
    render_scope = stage.DefinePrim("/Render")
    camera_product = UsdRender.Product.Define(stage, "/Render/Camera")
    camera_product.GetResolutionAttr().Set(Gf.Vec2i(1920, 1080))
    camera_product.GetCameraRel().SetTargets(["/World/Camera"])

    ldr_var = UsdRender.Var.Define(stage, "/Render/Camera/LdrColor")
    ldr_var.GetSourceNameAttr().Set("LdrColor")

    hdr_var = UsdRender.Var.Define(stage, "/Render/Camera/HdrColor")
    hdr_var.GetSourceNameAttr().Set("HdrColor")

    camera_product.GetOrderedVarsRel().SetTargets([
        "/Render/Camera/LdrColor",
        "/Render/Camera/HdrColor",
    ])
    # [/snippet:doc-camera-sensor-render-product-python]

    # Verify
    assert stage.GetPrimAtPath("/Render/Camera").IsValid()
    assert stage.GetPrimAtPath("/Render/Camera/LdrColor").IsValid()
    assert stage.GetPrimAtPath("/Render/Camera/HdrColor").IsValid()
    assert camera_product.GetResolutionAttr().Get() == Gf.Vec2i(1920, 1080)
    assert camera_product.GetCameraRel().GetTargets() == [Sdf.Path("/World/Camera")]


def test_minimal_render_product():
    """Build the sensor_configuration.rst minimal RenderProduct using the USD Python API."""
    stage = Usd.Stage.CreateInMemory()

    # [snippet:doc-minimal-render-product-python]
    render_scope = stage.DefinePrim("/Render")
    camera_product = UsdRender.Product.Define(stage, "/Render/Camera")
    camera_product.GetResolutionAttr().Set(Gf.Vec2i(1920, 1080))
    camera_product.GetCameraRel().SetTargets(["/World/Camera"])

    ldr_var = UsdRender.Var.Define(stage, "/Render/Camera/LdrColor")
    ldr_var.GetSourceNameAttr().Set("LdrColor")

    camera_product.GetOrderedVarsRel().SetTargets(["/Render/Camera/LdrColor"])
    # [/snippet:doc-minimal-render-product-python]

    # Verify
    assert stage.GetPrimAtPath("/Render/Camera").IsValid()
    assert stage.GetPrimAtPath("/Render/Camera/LdrColor").IsValid()
    assert camera_product.GetResolutionAttr().Get() == Gf.Vec2i(1920, 1080)


def test_multi_render_product_shared_vars():
    """Build the sensor_configuration.rst multi-sensor RenderProduct using the USD Python API."""
    stage = Usd.Stage.CreateInMemory()

    # [snippet:doc-multi-render-product-shared-vars-python]
    render_scope = stage.DefinePrim("/Render")

    front_product = UsdRender.Product.Define(stage, "/Render/FrontCamera")
    front_product.GetCameraRel().SetTargets(["/World/FrontCamera"])
    front_product.GetOrderedVarsRel().SetTargets(["/Render/Vars/LdrColor"])

    rear_product = UsdRender.Product.Define(stage, "/Render/RearCamera")
    rear_product.GetCameraRel().SetTargets(["/World/RearCamera"])
    rear_product.GetOrderedVarsRel().SetTargets(["/Render/Vars/LdrColor"])

    vars_scope = stage.DefinePrim("/Render/Vars")
    ldr_var = UsdRender.Var.Define(stage, "/Render/Vars/LdrColor")
    ldr_var.GetSourceNameAttr().Set("LdrColor")
    # [/snippet:doc-multi-render-product-shared-vars-python]

    # Verify
    assert stage.GetPrimAtPath("/Render/FrontCamera").IsValid()
    assert stage.GetPrimAtPath("/Render/RearCamera").IsValid()
    assert stage.GetPrimAtPath("/Render/Vars/LdrColor").IsValid()
    assert front_product.GetCameraRel().GetTargets() == [Sdf.Path("/World/FrontCamera")]
    assert rear_product.GetCameraRel().GetTargets() == [Sdf.Path("/World/RearCamera")]
