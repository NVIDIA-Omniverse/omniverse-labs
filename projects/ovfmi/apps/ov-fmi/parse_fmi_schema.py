#!/usr/bin/env python3
"""
parse_fmi_schema.py — FMI schema + initial USD attribute extractor.

Run as a subprocess: the main process cannot import pxr (usd-core) together
with ovrtx in the same process (they link different USD versions). This script
runs in isolation, prints JSON to stdout, and exits before ovrtx is loaded.

Usage:
    python3 parse_fmi_schema.py <usd_file>

Output JSON schema:
{
  "instances": {
    "/World/FmuInstance1": {
      "enabled": true,
      "fmu": "/resolved/path/to/model.fmu",
      "path": "/World/FmuInstance1",
      "connections": [
        {
          "enabled": true,
          "targets": ["/World/Cube1"],
          "mappings": [
            {
              "fmiAttributeName": "posX",
              "usdAttributeName": "xformOp:translate",
              "direction": "output",
              "usdMapping": [0, 1]
            }
          ]
        }
      ]
    }
  },
  "initial_values": {
    "/World/Cube1": {
      "xformOp:translate": [0.0, 5.0, 0.0]
    }
  },
  "body_prims": [],
  "render_products": ["/Render/Camera"],
  "cameras": ["/World/Camera"],
  "stage_bounds": {
    "valid": true,
    "min": [-1.0, 0.0, -1.0],
    "max": [1.0, 2.0, 1.0],
    "center": [0.0, 1.0, 0.0],
    "extent": [2.0, 2.0, 2.0],
    "radius": 1.7320508075688772
  },
  "world_up": {
    "vector": [0.0, 1.0, 0.0],
    "source": "stage upAxis=Y"
  }
}
"""

import json
import math
import sys


def _read_attr(prim, attr_name):
    """Read a USD attribute and convert it to a JSON-serialisable Python value."""
    attr = prim.GetAttribute(attr_name)
    if not attr.IsValid():
        return None
    val = attr.Get()
    if val is None:
        return None
    # GfVec* and GfMatrix* have iteration support; plain scalars do not.
    try:
        return list(val)
    except TypeError:
        return val


def _stage_bounds(stage) -> dict:
    """Compute a conservative world-space bound for renderable scene content."""
    try:
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        bounds = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
        if bounds.IsEmpty():
            return {"valid": False}

        mn = bounds.GetMin()
        mx = bounds.GetMax()
        center = [(float(mn[i]) + float(mx[i])) * 0.5 for i in range(3)]
        extent = [max(0.0, float(mx[i]) - float(mn[i])) for i in range(3)]
        radius = math.sqrt(sum((e * 0.5) ** 2 for e in extent))
        return {
            "valid": True,
            "min": [float(mn[i]) for i in range(3)],
            "max": [float(mx[i]) for i in range(3)],
            "center": center,
            "extent": extent,
            "radius": float(radius),
        }
    except Exception as exc:
        print(f"[parse_fmi_schema] WARNING: could not compute stage bounds: {exc}", file=sys.stderr)
        return {"valid": False}


def _normalised_vec3(vec, fallback):
    length = math.sqrt(sum(float(v) * float(v) for v in vec))
    if length < 1e-8:
        return list(fallback)
    return [float(v) / length for v in vec]


def _world_up(stage) -> dict:
    """Infer the stage up vector, preferring authored physics gravity."""
    for prim in stage.Traverse():
        attr = prim.GetAttribute("physics:gravityDirection")
        if not attr.IsValid():
            continue
        gravity = attr.Get()
        if gravity is None:
            continue
        try:
            direction = [float(gravity[i]) for i in range(3)]
        except Exception:
            continue
        up = _normalised_vec3([-direction[0], -direction[1], -direction[2]], [0.0, 1.0, 0.0])
        return {
            "vector": up,
            "source": f"physics:gravityDirection at {prim.GetPath()}",
        }

    try:
        from pxr import UsdGeom  # noqa: PLC0415
        up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
    except Exception:
        up_axis = "Y"

    if up_axis == "Z":
        return {"vector": [0.0, 0.0, 1.0], "source": "stage upAxis=Z"}
    return {"vector": [0.0, 1.0, 0.0], "source": "stage upAxis=Y"}


def _world_position(prim) -> list[float] | None:
    """Return the prim's world-space translation, if it can be computed."""
    try:
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        matrix = cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        return [float(translation[i]) for i in range(3)]
    except Exception as exc:
        print(
            f"[parse_fmi_schema] WARNING: could not compute world position "
            f"for {prim.GetPath()}: {exc}",
            file=sys.stderr,
        )
        return None


def _sensor_sphere_proxies(sensor_prim) -> tuple[float, list[str]]:
    """Find standard UsdGeomSphere children that can configure a sensor."""
    proxies = []
    radius = 0.1

    def visit(prim):
        nonlocal radius
        if prim.GetTypeName() == "Sphere":
            proxies.append(str(prim.GetPath()))
            attr = prim.GetAttribute("radius")
            if attr.IsValid():
                authored_radius = attr.Get()
                if authored_radius is not None:
                    try:
                        radius = float(authored_radius)
                    except Exception:
                        pass
        for child in prim.GetChildren():
            visit(child)

    visit(sensor_prim)
    return radius, proxies


def parse(usd_path: str) -> dict:
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    instances = {}
    initial_values = {}

    for prim in stage.Traverse():
        prim_type = prim.GetTypeName()
        if prim_type not in ("FmuInstance", "SspInstance"):
            continue

        is_ssp = (prim_type == "SspInstance")

        if is_ssp:
            ssp_asset = prim.GetAttribute("fmi:ssp").Get()
            if ssp_asset is None:
                print(f"[parse_fmi_schema] WARNING: {prim.GetPath()} has no fmi:ssp attribute", file=sys.stderr)
                continue
            resolved_path = ssp_asset.resolvedPath
        else:
            fmu_asset = prim.GetAttribute("fmi:fmu").Get()
            if fmu_asset is None:
                print(f"[parse_fmi_schema] WARNING: {prim.GetPath()} has no fmi:fmu attribute", file=sys.stderr)
                continue
            resolved_path = fmu_asset.resolvedPath

        enabled = prim.GetAttribute("fmi:enabled").Get() if prim.HasAttribute("fmi:enabled") else True
        connections = []

        for child in prim.GetChildren():
            if child.GetTypeName() != "FmuConnection":
                continue

            conn_enabled = child.GetAttribute("fmi:enabled").Get() if child.HasAttribute("fmi:enabled") else True
            targets = [str(t) for t in child.GetRelationship("fmi:targets").GetTargets()]
            mappings = []

            for mapping_child in child.GetChildren():
                if mapping_child.GetTypeName() != "FmuMapping":
                    continue
                fmi_attr = mapping_child.GetAttribute("fmi:fmuAttribute").Get()
                usd_attr = mapping_child.GetAttribute("fmi:usdAttribute").Get()
                direction = mapping_child.GetAttribute("fmi:direction").Get()
                usd_mapping = [0, 0]
                if mapping_child.HasAttribute("fmi:usdMapping"):
                    usd_mapping = list(mapping_child.GetAttribute("fmi:usdMapping").Get())
                if None in (fmi_attr, usd_attr, direction):
                    print(f"[parse_fmi_schema] WARNING: incomplete mapping at {mapping_child.GetPath()}", file=sys.stderr)
                    continue
                mappings.append({
                    "fmiAttributeName": fmi_attr,
                    "usdAttributeName": usd_attr,
                    "direction": direction,
                    "usdMapping": usd_mapping,
                })

                # Capture current USD attribute value for each target prim
                if direction == "input":
                    for target_path in targets:
                        target = stage.GetPrimAtPath(target_path)
                        if not target.IsValid():
                            continue
                        val = _read_attr(target, usd_attr)
                        if val is not None:
                            initial_values.setdefault(target_path, {})[usd_attr] = val if isinstance(val, list) else [val]

            connections.append({
                "enabled": conn_enabled,
                "targets": targets,
                "mappings": mappings,
            })

        instances[str(prim.GetPath())] = {
            "enabled": enabled,
            "fmu": resolved_path if not is_ssp else None,
            "ssp": resolved_path if is_ssp else None,
            "path": str(prim.GetPath()),
            "connections": connections,
        }

    # --- Detect rigid body prims (PhysicsRigidBodyAPI) and articulation roots ---
    body_prims = []
    articulation_roots = []
    try:
        from pxr import UsdPhysics  # noqa: PLC0415
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
                if enabled is not False:
                    body_prims.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(str(prim.GetPath()))
    except ImportError:
        # UsdPhysics may not be available; fall back to attribute check
        for prim in stage.Traverse():
            if prim.GetAttribute("physics:rigidBodyEnabled").Get():
                body_prims.append(str(prim.GetPath()))
            if "PhysicsArticulationRootAPI" in [str(api) for api in prim.GetAppliedSchemas()]:
                articulation_roots.append(str(prim.GetPath()))

    render_products = []
    cameras = []
    sensor_positions = {}
    overlap_sensors = {}
    for prim in stage.Traverse():
        prim_type = prim.GetTypeName()
        if prim_type == "RenderProduct":
            render_products.append(str(prim.GetPath()))
        elif prim_type == "Camera":
            cameras.append(str(prim.GetPath()))
        if prim.GetName().lower().startswith("sensor"):
            position = _world_position(prim)
            if position is not None:
                sensor_path = str(prim.GetPath())
                sensor_positions[sensor_path] = position
                radius, proxies = _sensor_sphere_proxies(prim)
                overlap_sensors[sensor_path] = {
                    "position": position,
                    "radius": radius,
                    "proxies": proxies,
                }

    return {
        "instances": instances,
        "initial_values": initial_values,
        "body_prims": body_prims,
        "articulation_roots": articulation_roots,
        "render_products": render_products,
        "cameras": cameras,
        "sensor_positions": sensor_positions,
        "overlap_sensors": overlap_sensors,
        "stage_bounds": _stage_bounds(stage),
        "world_up": _world_up(stage),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: parse_fmi_schema.py <usd_file>", file=sys.stderr)
        sys.exit(1)

    try:
        result = parse(sys.argv[1])
        print(json.dumps(result))
    except Exception as e:
        print(f"[parse_fmi_schema] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
