#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Test-only Pillow: run in a developer SDK test environment with Pillow added; the shipped SDK runtime lock has no Pillow.
"""Measured add/delete/topology/material updates in one live renderer and stage."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def scene(generation, material_model="preview"):
    def mesh(name, x, material, triangle=False):
        points = [(x-.7,-.7,0),(x+.7,-.7,0),(x+.7,.7,0),(x-.7,.7,0)]
        if triangle:
            points = points[:3]
        color = "" if triangle else "color3f[] primvars:displayColor = [(0.2,0.3,0.4)]"
        return f'''def Mesh "{name}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
            {color}
            point3f[] points = {points}
            int[] faceVertexCounts = [{len(points)}]
            int[] faceVertexIndices = {list(range(len(points)))}
            uniform token subdivisionScheme = "none"
            rel material:binding = </World/Looks/{material}>
        }}\n'''
    def material(name, color, shader):
        mtlx = material_model == "openpbr"
        terminal, output = ("mtlx:surface", "out") if mtlx else ("surface", "surface")
        node = "ND_open_pbr_surface_surfaceshader" if mtlx else "UsdPreviewSurface"
        color_input = "base_color" if mtlx else "diffuseColor"
        roughness_input = "specular_roughness" if mtlx else "roughness"
        schema = '(prepend apiSchemas = ["NodeGraphNodeAPI"])' if mtlx else ""
        return f'''def Material "{name}" {{
            token outputs:{terminal}.connect = </World/Looks/{name}/{shader}.outputs:{output}>
            def Shader "{shader}" {schema} {{
                uniform token info:id = "{node}"
                color3f inputs:{color_input} = {color}
                float inputs:{roughness_input} = 0.5
                token outputs:{output}
            }}
        }}\n'''
    return '''#usda 1.0
(
    upAxis = "Y"
    metersPerUnit = 1
)
def Xform "World" {
''' + mesh("Left", -1, "LeftMaterial", generation == 3) + (
        mesh("Right", 1, "RightMaterial") if generation in (1,2) else "") + '''
    def Scope "Looks" {
''' + material("LeftMaterial", "(0.02,0.8,0.02)" if generation >= 2 else "(0.8,0.02,0.02)",
               "NewGreenShader" if generation >= 2 else "OldRedShader") + (
        material("RightMaterial", "(0.02,0.02,0.8)", "BlueShader") if generation in (1,2) else "") + '''
    }
    def Camera "Camera" {
        float focalLength = 35
        float horizontalAperture = 36
        float verticalAperture = 24
        double3 xformOp:translate = (0,0,7)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def DomeLight "Light" {
        float inputs:intensity = 1500
    }
}
def "Render" {
    def RenderProduct "Camera" {
        int2 resolution = (320,240)
        rel camera = </World/Camera>
        rel orderedVars = [<LdrColor>,<HdrColor>]
        def RenderVar "LdrColor" {
            string sourceName = "LdrColor"
        }
        def RenderVar "HdrColor" {
            string sourceName = "HdrColor"
        }
    }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--material-model", choices=("preview","openpbr"), default="preview")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
    from ovstage_scene_updates import SceneGenerationPopulation, SceneGenerationUpdateError
    import numpy as np
    import ovstage
    import ovrtx
    from PIL import Image
    stage = ovstage.Stage("live-scene-update-probe")
    renderer = ovrtx.Renderer(ovrtx.RendererConfig(active_cuda_gpus=str(args.gpu),
        log_level="warn", log_file_path=str(args.output/"renderer.log"), keep_system_alive=True))
    renderer.attach_ovstage(stage)
    owner = SceneGenerationPopulation(stage)
    identity = (os.getpid(), id(stage), id(renderer))
    records, images = [], []
    direct_material = None
    def render_pixels():
        for _ in range(16):
            products = renderer.step({"/Render/Camera"},delta_time=1/60,ordinal=owner.ordinal)
            if _ != 15:
                del products
        frame = products["/Render/Camera"].frames[-1]
        var = next(v for k,v in frame.render_vars.items() if k.rsplit("/",1)[-1] == "LdrColor")
        mapped = var.map(device=ovrtx.Device.CPU)
        try:
            view = np.from_dlpack(mapped)
            pixels = view.copy()
            del view
        finally:
            mapped.unmap()
        del mapped, var, frame, products
        return pixels
    def values(path, attr, ordinal):
        found = []
        with ovstage.PathDictionary(stage) as paths:
            with paths.create_path_list_from_strings([path]) as plist:
                with stage.query_from_path_list(plist) as query:
                    with stage.read_attributes(query,[paths.intern_token(attr)],ovstage.OrdinalRange.latest(ordinal)) as read:
                        read.wait()
                        for group in read.groups():
                            try:
                                if not group.is_delete:
                                    found.append(np.asarray(group.array(0)).copy().tolist())
                            finally:
                                stage.release_group(group)
        return found
    try:
        for generation in range(4):
            source = args.output / f"generation-{generation}.usda"
            source.write_text(scene(generation,args.material_model))
            start = time.monotonic()
            result = (owner.initialize(source) if generation == 0 else
                      owner.replace(source, ordinal=owner.ordinal+1))
            stage.advance_write_floor(owner.ordinal,ovstage.Scope.ALL).wait()
            update_seconds = time.monotonic()-start
            renderer.update_from_stage(owner.ordinal)
            # This public call clears image accumulation, not the live stage or
            # renderer. Material-only changes otherwise retain previous pixels.
            renderer.reset(time=generation*16/60)
            pixels = render_pixels()
            Image.fromarray(pixels).save(args.output/f"generation-{generation}.png")
            assert identity == (os.getpid(),id(stage),id(renderer))
            records.append(dict(generation=generation,update_seconds=update_seconds,**result,
                left_points=values("/World/Left","points",owner.ordinal),
                left_display_color=values("/World/Left","primvars:displayColor",owner.ordinal),
                right_points=values("/World/Right","points",owner.ordinal)))
            images.append(pixels)
            if generation == 1:
                settled = render_pixels()
                no_edit_mae = float(np.abs(settled.astype(float)-pixels.astype(float)).mean())
            if generation == 2:
                shader = "/World/Looks/LeftMaterial/NewGreenShader"
                color_attribute = "inputs:base_color" if args.material_model == "openpbr" else "inputs:diffuseColor"
                def write_color(color):
                    owner.ordinal += 1
                    with ovstage.PathDictionary(stage) as paths:
                        with paths.create_path_list_from_strings([shader]) as plist:
                            with stage.query_from_path_list(plist) as query:
                                stage.write_attribute(query,color_attribute,owner.ordinal,
                                    np.asarray([color],dtype=np.float32),is_array=False).wait()
                    stage.advance_write_floor(owner.ordinal).wait()
                    renderer.update_from_stage(owner.ordinal)
                    renderer.reset(time=2*16/60)
                write_color([0.02,0.02,0.8])
                blue = render_pixels()
                Image.fromarray(blue).save(args.output/"direct-material-blue.png")
                patch = blue[100:140,100:135,:3].astype(float).mean(axis=(0,1))
                direct_material = dict(shader=shader,attribute=color_attribute,
                    ovstage_readback=values(shader,color_attribute,owner.ordinal),
                    patch_rgb=patch.tolist(),source_generation_unchanged=owner.path==source,
                    pixel_blue=bool(patch[2] > patch[1]+20 and patch[2] > patch[0]+20))
                write_color([0.02,0.8,0.02])
                render_pixels()
        # Force a failure *after* native candidate projection changed the stage,
        # then verify the helper restores the accepted predecessor itself.
        class FailAfterNativeApply:
            def __init__(self):
                self.armed = True

            def __getattr__(self, name):
                return getattr(ovstage.population, name)

            def apply_usd_changes(self, target, **kwargs):
                ovstage.population.apply_usd_changes(target, **kwargs)
                if self.armed:
                    self.armed = False
                    raise RuntimeError("injected failure after native apply")
        owner.population = FailAfterNativeApply()
        try:
            owner.replace(args.output/"generation-1.usda", ordinal=owner.ordinal+1)
        except SceneGenerationUpdateError as failure:
            assert failure.rollback_succeeded and not owner.failed
        else:
            raise AssertionError("Injected native failure was not raised")
        finally:
            owner.population = ovstage.population
        renderer.update_from_stage(owner.ordinal)
        renderer.reset(time=4*16/60)
        restored = render_pixels()
        Image.fromarray(restored).save(args.output/"restored-predecessor.png")
        rollback_mae = float(np.abs(restored.astype(float)-images[3].astype(float)).mean())
        assert not values("/World/Right","points",owner.ordinal)
        assert np.asarray(values("/World/Left","points",owner.ordinal)[0]).size == 9
        assert rollback_mae < 2, rollback_mae
        bad = args.output/"invalid.usda"
        bad.write_text("not a USD layer\n")
        previous = owner.path, owner.ordinal, owner.handle
        try:
            owner.replace(bad, ordinal=owner.ordinal+1)
        except Exception:
            assert (owner.path, owner.ordinal, owner.handle) == previous
        else:
            raise AssertionError("Invalid candidate source was accepted")
        # Local changed regions separately prove addition, material change, and
        # deletion. Comparing whole-image means can hide a retained old mesh.
        differences = [float(np.abs(images[i].astype(float)-images[i-1].astype(float)).mean()) for i in range(1,4)]
        assert differences[0] > 1 and differences[2] > 1, differences
        assert records[1]["right_points"] and records[2]["right_points"]
        assert not records[3]["right_points"], "Deleted mesh points survived population"
        assert np.asarray(records[0]["left_points"][0]).size == 12
        assert np.asarray(records[3]["left_points"][0]).size == 9
        assert records[0]["left_display_color"] and not records[3]["left_display_color"]
        left = images[2][100:140,100:135,:3].astype(float).mean(axis=(0,1))
        assert left[1] > left[0]+20, left.tolist()
        before = images[1][100:140,100:135,:3].astype(float).mean(axis=(0,1))
        assert before[0] > before[1]+20, before.tolist()
        assert differences[1] > max(0.2, 3*no_edit_mae), (differences,no_edit_mae)
        assert direct_material["pixel_blue"], direct_material
        report = dict(status="passed",material_model=args.material_model,pid=identity[0],stage_identity=identity[1],renderer_identity=identity[2],
            same_process_stage_renderer=True,generations=records,pixel_mae=differences,
            material_green_patch_rgb=left.tolist(),material_red_patch_rgb=before.tolist(),
            no_edit_mae=no_edit_mae,history_cleared_after_scene_edit=True,
            native_apply_failure_restored=True,rollback_pixel_mae=rollback_mae,
            invalid_candidate_preserved_predecessor=True,
            removed_prim_attribute_absent=True,
            direct_shader_input_write=direct_material,
            deleted_mesh_absent=True,topology_updated=True)
        (args.output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))
    finally:
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()


if __name__ == "__main__":
    main()
