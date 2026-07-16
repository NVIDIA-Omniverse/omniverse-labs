"""Deterministic PBR inspection renders and side-by-side material comparison."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import RealToSimError
from .learned_materials import MATFUSE_BACKEND, STABLEMATERIALS_BACKEND


def _image(path: Path, mode: str, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        return (
            np.asarray(
                source.convert(mode).resize(size, Image.Resampling.BILINEAR),
                dtype=np.float64,
            )
            / 255.0
        )


def _sample(texture: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = texture.shape[:2]
    x = (np.mod(u, 1.0) * (width - 1)).astype(np.int64)
    y = (np.clip(v, 0.0, 1.0) * (height - 1)).astype(np.int64)
    return texture[y, x]


def render_material_ball(bundle_role: Path, *, size: int = 640) -> Path:
    """Render a compact deterministic PBR inspector from the five-map contract."""

    root = Path(bundle_role).resolve()
    required = (
        "baseColor.png",
        "normal.png",
        "roughness.png",
        "metallic.png",
        "ao.png",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RealToSimError(f"material inspector is missing maps: {missing}")
    yy, xx = np.mgrid[0:size, 0:size]
    px = (xx + 0.5) / size * 2.0 - 1.0
    py = 1.0 - (yy + 0.5) / size * 2.0
    radius = 0.78
    sx = px / radius
    sy = py / radius
    radial = sx * sx + sy * sy
    sz = np.sqrt(np.clip(1.0 - radial, 0.0, 1.0))
    geometric = np.stack((sx, sy, sz), axis=2)
    geometric /= np.maximum(np.linalg.norm(geometric, axis=2, keepdims=True), 1e-8)
    u = np.arctan2(geometric[..., 0], geometric[..., 2]) / (2.0 * math.pi) + 0.5
    v = np.arccos(np.clip(geometric[..., 1], -1.0, 1.0)) / math.pi

    texture_size = (512, 512)
    base = _sample(_image(root / "baseColor.png", "RGB", texture_size), u, v)
    tangent_normal = (
        _sample(_image(root / "normal.png", "RGB", texture_size), u, v) * 2.0 - 1.0
    )
    tangent_normal /= np.maximum(
        np.linalg.norm(tangent_normal, axis=2, keepdims=True), 1e-8
    )
    roughness = _sample(_image(root / "roughness.png", "L", texture_size), u, v)
    metallic = _sample(_image(root / "metallic.png", "L", texture_size), u, v)
    ao = _sample(_image(root / "ao.png", "L", texture_size), u, v)

    tangent = np.stack(
        (geometric[..., 2], np.zeros_like(sz), -geometric[..., 0]), axis=2
    )
    tangent /= np.maximum(np.linalg.norm(tangent, axis=2, keepdims=True), 1e-8)
    bitangent = np.cross(geometric, tangent)
    normal = (
        tangent * tangent_normal[..., 0:1]
        + bitangent * tangent_normal[..., 1:2]
        + geometric * tangent_normal[..., 2:3]
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-8)

    view = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    light = np.asarray((-0.38, 0.62, 0.69), dtype=np.float64)
    light /= np.linalg.norm(light)
    fill = np.asarray((0.72, 0.18, 0.67), dtype=np.float64)
    fill /= np.linalg.norm(fill)
    half_vector = (view + light) / np.linalg.norm(view + light)
    ndotl = np.clip(np.sum(normal * light, axis=2), 0.0, 1.0)
    ndotfill = np.clip(np.sum(normal * fill, axis=2), 0.0, 1.0)
    ndoth = np.clip(np.sum(normal * half_vector, axis=2), 0.0, 1.0)
    exponent = 2.0 + (1.0 - roughness) ** 2 * 180.0
    fresnel = 0.04 * (1.0 - metallic[..., None]) + base * metallic[..., None]
    specular = (
        fresnel
        * np.power(ndoth[..., None], exponent[..., None])
        * (0.25 + 1.8 * (1.0 - roughness[..., None]))
    )
    diffuse = (
        base
        * (1.0 - metallic[..., None])
        * (0.16 + 0.78 * ndotl[..., None] + 0.24 * ndotfill[..., None])
    )
    color = (diffuse + specular) * (0.55 + 0.45 * ao[..., None])
    rim = np.power(1.0 - np.clip(geometric[..., 2], 0.0, 1.0), 3.0)[..., None]
    color += rim * np.asarray((0.10, 0.16, 0.24))

    background = np.zeros((size, size, 3), dtype=np.float64)
    background[..., 0] = 0.018 + 0.018 * (1.0 - (yy / size))
    background[..., 1] = 0.026 + 0.025 * (1.0 - (yy / size))
    background[..., 2] = 0.042 + 0.038 * (1.0 - (yy / size))
    edge = np.clip((1.0 - radial) * size * 0.025, 0.0, 1.0)[..., None]
    image = background * (1.0 - edge) + np.clip(color, 0.0, 1.0) * edge
    output = root / "material-ball.png"
    Image.fromarray(
        np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
    ).save(output)
    return output


def _manifest(root: Path) -> dict[str, Any]:
    path = Path(root).resolve() / "manifest.json"
    if not path.is_file():
        raise RealToSimError(f"learned material bundle has no manifest: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RealToSimError(f"invalid learned material manifest: {path}") from exc
    if value.get("contract") != "nanousd-rts-learned-pbr-bundle-v1":
        raise RealToSimError(f"unsupported learned material bundle: {path}")
    return value


def write_material_comparison(
    matfuse_bundle: Path,
    stablematerials_bundle: Path,
    *,
    output: Path,
    matfuse_scene: str | None = None,
    stablematerials_scene: str | None = None,
) -> Path:
    """Write one side-by-side report with role and map switching."""

    roots = {
        "matfuse": Path(matfuse_bundle).resolve(),
        "stablematerials": Path(stablematerials_bundle).resolve(),
    }
    manifests = {key: _manifest(value) for key, value in roots.items()}
    expected_backends = {
        "matfuse": MATFUSE_BACKEND,
        "stablematerials": STABLEMATERIALS_BACKEND,
    }
    for key, expected in expected_backends.items():
        actual = manifests[key].get("backend")
        if actual != expected:
            raise RealToSimError(
                f"{key} comparison slot requires {expected}, received {actual!r}"
            )
    role_sets = [
        set(item["role"] for item in manifest["roles"])
        for manifest in manifests.values()
    ]
    roles = sorted(set.intersection(*role_sets))
    if not roles:
        raise RealToSimError(
            "MatFuse and StableMaterials bundles have no matching roles"
        )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"roles": roles, "models": {}}
    for key, root in roots.items():
        manifest = manifests[key]
        role_by_name = {item["role"]: item for item in manifest["roles"]}
        role_payload: dict[str, Any] = {}
        for role in roles:
            directory = root / role
            ball = render_material_ball(directory)
            names = [
                name
                for name in (
                    "baseColor.png",
                    "normal.png",
                    "roughness.png",
                    "metallic.png",
                    "ao.png",
                    "specular.png",
                    "height.png",
                )
                if (directory / name).is_file()
            ]
            role_payload[role] = {
                "prompt": role_by_name[role]["prompt"],
                "seconds": role_by_name[role]["inference_seconds"],
                "ball": os.path.relpath(ball, output.parent),
                "maps": {
                    name.removesuffix(".png"): os.path.relpath(
                        directory / name, output.parent
                    )
                    for name in names
                },
            }
        payload["models"][key] = {
            "backend": manifest["backend"],
            "model": manifest["model"],
            "runtime": manifest["runtime"],
            "generation": manifest["generation"],
            "roles": role_payload,
            "scene": matfuse_scene if key == "matfuse" else stablematerials_scene,
        }
    data = json.dumps(payload, sort_keys=True).replace("</", "<\\/")
    output.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Home Scan · learned material comparison</title>
<style>
:root{{color-scheme:dark;--bg:#06080d;--panel:#0e141f;--line:#263247;--text:#ecf2fb;--muted:#91a0b5;--cyan:#73d6ff;--orange:#ff9b57}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 50% -20%,#16243a,var(--bg) 48%);color:var(--text);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
header{{padding:26px clamp(18px,4vw,54px) 18px;border-bottom:1px solid var(--line)}} h1{{margin:0;font:650 clamp(22px,3vw,36px)/1.1 system-ui,sans-serif}} .lede{{max-width:900px;color:var(--muted);margin-top:9px}}
.toolbar,#sceneLinks{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}} .toolbar{{margin-top:18px}} select,button,a.scene{{border:1px solid var(--line);background:#111a28;color:var(--text);border-radius:7px;padding:9px 12px;font:inherit;text-decoration:none}} button.active{{border-color:var(--cyan);color:var(--cyan)}}
main{{padding:24px clamp(18px,4vw,54px) 60px}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}} .card{{min-width:0;background:linear-gradient(150deg,#111a28,#0b1019);border:1px solid var(--line);border-radius:14px;overflow:hidden}}
.card-head{{padding:18px 20px;border-bottom:1px solid var(--line)}} h2{{font:650 20px/1.2 system-ui,sans-serif;margin:0}} .meta,.prompt{{color:var(--muted);font-size:12px;margin-top:7px}} .hero{{display:block;width:100%;aspect-ratio:1.35;object-fit:contain;background:#05070b}}
.maps{{display:flex;gap:8px;flex-wrap:wrap;padding:13px 16px;border-top:1px solid var(--line)}} .maps button{{padding:6px 9px;font-size:12px}} .facts{{display:grid;grid-template-columns:repeat(3,1fr);border-top:1px solid var(--line)}} .fact{{padding:12px 14px;border-right:1px solid var(--line)}} .fact:last-child{{border-right:0}} .fact b{{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px}}
.honesty{{margin-top:20px;padding:15px 18px;border:1px solid #3b3327;background:#18130d;border-radius:10px;color:#d8c4ad}} code{{color:#b6ddff}} @media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Official MatFuse vs StableMaterials</h1><div class="lede">The same Home Scan oven material request, generated locally on Apple Metal. Model maps are shown unchanged; the sphere is a deterministic inspector render using base color, tangent normal, roughness, metallic, and AO.</div>
<div class="toolbar"><label>Interior role <select id="role"></select></label><span id="sceneLinks"></span></div></header>
<main><div class="grid" id="grid"></div><div class="honesty"><strong>Map contract:</strong> MatFuse natively predicts diffuse, normal, roughness, and specular; its specular map is preserved and metallic/AO receive explicit neutral values. StableMaterials natively predicts base color, normal, height, roughness, and metallic; only AO is neutral. Generated hidden surfaces remain <code>measured=false</code>.</div></main>
<script>
const data={data};
const esc=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;');
const role=document.querySelector('#role'); role.innerHTML=data.roles.map(x=>`<option value="${{esc(x)}}">${{esc(x)}}</option>`).join('');
const labels={{matfuse:'MatFuse · CVPR 2024',stablematerials:'StableMaterials'}};
function render(){{const r=role.value;document.querySelector('#grid').innerHTML=Object.entries(data.models).map(([key,m])=>{{const x=m.roles[r];const buttons=[['material ball',x.ball],...Object.entries(x.maps)];return `<article class="card"><div class="card-head"><h2>${{esc(labels[key])}}</h2><div class="meta">${{esc(m.model.repo_id)}} @ ${{esc(m.model.revision.slice(0,9))}} · ${{esc(m.model.variant)}}</div><div class="prompt">${{esc(x.prompt)}}</div></div><img class="hero" id="hero-${{esc(key)}}" src="${{esc(x.ball)}}"><div class="maps">${{buttons.map(([name,url],i)=>`<button data-model="${{esc(key)}}" data-url="${{esc(url)}}" class="${{i===0?'active':''}}">${{esc(name)}}</button>`).join('')}}</div><div class="facts"><div class="fact"><b>device</b>${{esc(m.runtime.device)}} · ${{esc(m.runtime.dtype)}}</div><div class="fact"><b>sampling</b>${{esc(m.generation.steps)}} steps · CFG ${{esc(m.generation.guidance_scale)}}</div><div class="fact"><b>role time</b>${{esc(x.seconds.toFixed(1))}} s</div></div></article>`}}).join('');
document.querySelectorAll('.maps button').forEach(b=>b.onclick=()=>{{document.querySelector(`#hero-${{b.dataset.model}}`).src=b.dataset.url;b.parentElement.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active')}});
document.querySelector('#sceneLinks').innerHTML=Object.entries(data.models).filter(([,m])=>m.scene).map(([key,m])=>`<a class="scene" href="${{esc(m.scene)}}">Open ${{esc(labels[key])}} scene</a>`).join(' ');
}} role.onchange=render;render();
</script></body></html>""",
        encoding="utf-8",
    )
    return output
