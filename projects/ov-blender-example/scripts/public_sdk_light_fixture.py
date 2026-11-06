"""Reusable USD fixture for direct public SDK light and DLSS probes."""
from pathlib import Path
import textwrap

def _generated_light_probe_fixture(path: Path, width: int, height: int) -> dict[str, str]:
    render_product_path = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"
    path.write_text(
        textwrap.dedent(
            f"""
            #usda 1.0
            (
                defaultPrim = "World"
                metersPerUnit = 1
                upAxis = "Z"
            )

            def Xform "World"
            {{
                def Scope "Looks"
                {{
                    def Material "NeutralGrey"
                    {{
                        token outputs:surface.connect = </World/Looks/NeutralGrey/PreviewSurface.outputs:surface>
                        def Shader "PreviewSurface"
                        {{
                            uniform token info:id = "UsdPreviewSurface"
                            color3f inputs:diffuseColor = (0.6, 0.6, 0.6)
                            float inputs:roughness = 0.65
                            float inputs:metallic = 0
                            token outputs:surface
                        }}
                    }}
                }}

                def Cube "LitCube" (
                    prepend apiSchemas = ["MaterialBindingAPI"]
                )
                {{
                    rel material:binding = </World/Looks/NeutralGrey>
                    double size = 1.6
                    double3 xformOp:translate = (0, 0, 0.8)
                    uniform token[] xformOpOrder = ["xformOp:translate"]
                }}

                def Camera "Camera"
                {{
                    double3 xformOp:translate = (0, -5, 1.5)
                    quatd xformOp:orient = (0.7071068, 0.7071068, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
                    float focalLength = 42
                    float horizontalAperture = 36
                    float verticalAperture = 20.25
                    float2 clippingRange = (0.05, 100)
                }}

                def DistantLight "KeyLight"
                {{
                    float inputs:intensity = 50000
                    color3f inputs:color = (1, 0.95, 0.9)
                    float inputs:angle = 0.53
                    bool inputs:normalize = false
                }}
            }}

            def "Render"
            {{
                def "OmniverseKit"
                {{
                    def "HydraTextures"
                    {{
                        def RenderProduct "ViewportTexture0"
                        {{
                            rel camera = </World/Camera>
                            token omni:rtx:rendermode = "RealTimePathTracing"
                            token omni:rtx:background:source:type = "color"
                            color3f omni:rtx:background:color = (0, 0, 0)
                            rel orderedVars = </Render/OmniverseKit/HydraTextures/ViewportTexture0/LdrColor>
                            uniform int2 resolution = ({int(width)}, {int(height)})
                            bool omni:rtx:autoExposure:enabled = false
                            bool omni:rtx:rt:ecoMode:enabled = false
                            def RenderVar "LdrColor"
                            {{
                                uniform string sourceName = "LdrColor"
                            }}
                        }}
                    }}
                }}
            }}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return {"usd_path": str(path), "render_product_path": render_product_path}

