"""Pinned MatFuse and StableMaterials workers for Apple Metal inference.

The heavyweight diffusion runtime deliberately lives outside the scene-authoring
path.  It consumes the deterministic ``nanousd-rts-pbr-atlas-v1`` request bundle
and emits the same five-map contract accepted by ``external-pbr-atlas-v1``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from .core import RealToSimError, content_digest, sha256_file


MATERIAL_BUNDLE_SCHEMA = 1
MATFUSE_BACKEND = "matfuse-paper-hf-mps-v1"
STABLEMATERIALS_BACKEND = "stablematerials-hf-mps-v1"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    backend: str
    repo_id: str
    revision: str
    license: str
    paper: str
    native_resolution: int
    default_steps: int
    default_guidance: float


MODEL_SPECS = {
    "matfuse": ModelSpec(
        backend=MATFUSE_BACKEND,
        repo_id="gvecchio/MatFuse",
        revision="5b8131f9559c5f35b52a5d4e49df380fa7cf3fd5",
        license="MIT",
        paper="https://arxiv.org/abs/2308.11408",
        native_resolution=256,
        default_steps=50,
        default_guidance=4.0,
    ),
    "stablematerials": ModelSpec(
        backend=STABLEMATERIALS_BACKEND,
        repo_id="gvecchio/StableMaterials",
        revision="4d452731535bd5c74261d6645623a573326f6f36",
        license="OpenRAIL",
        paper="https://arxiv.org/abs/2406.09293",
        native_resolution=512,
        default_steps=4,
        default_guidance=10.0,
    ),
}


MATFUSE_PATTERNS = (
    "model_index.json",
    "pipeline_matfuse.py",
    "condition_encoders.py",
    "vae_matfuse.py",
    "unet/*",
    "vae/*",
    "scheduler/*",
    "condition_encoder/*",
)
STABLEMATERIALS_COMMON_PATTERNS = (
    "model_index.json",
    "pipeline.py",
    "processor/*",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "vae/*",
    "vision_encoder/*",
)


MODEL_PROMPTS = {
    "oven-door": (
        "charcoal gray heat-resistant porcelain enamel for an oven interior, "
        "fine orange-peel texture, subtle baked-on wear, seamless photorealistic PBR material"
    ),
    "refrigerator-door": (
        "clean warm-white molded refrigerator interior plastic, very fine satin texture, "
        "subtle manufacturing variation, seamless photorealistic PBR material"
    ),
    "cabinet-door": (
        "warm off-white painted maple cabinet interior, fine wood grain beneath satin paint, "
        "subtle realistic wear, seamless photorealistic PBR material"
    ),
    "drawer": (
        "light maple wood drawer interior, fine straight grain, matte clear finish, "
        "subtle realistic variation, seamless photorealistic PBR material"
    ),
}


def _require_runtime() -> tuple[Any, Any]:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
        import diffusers
    except ImportError as exc:
        raise RealToSimError(
            "learned material inference dependencies are missing; create the isolated "
            "Python 3.12 worker with `UV_PROJECT_ENVIRONMENT=.venv-materials uv sync "
            "--python 3.12 --extra learned-materials`"
        ) from exc
    return torch, diffusers


def learned_material_status() -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "recommended_python": "3.12",
        "dependencies": False,
        "mps_built": False,
        "mps_available": False,
    }
    try:
        torch, diffusers = _require_runtime()
    except RealToSimError as exc:
        runtime["error"] = str(exc)
    else:
        runtime.update(
            {
                "dependencies": True,
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "mps_built": bool(torch.backends.mps.is_built()),
                "mps_available": bool(torch.backends.mps.is_available()),
            }
        )
    return {
        "schema_version": MATERIAL_BUNDLE_SCHEMA,
        "runtime": runtime,
        "models": {
            key: {
                "backend": value.backend,
                "repo_id": value.repo_id,
                "revision": value.revision,
                "license": value.license,
                "paper": value.paper,
                "native_resolution": value.native_resolution,
                "default_steps": value.default_steps,
                "default_guidance": value.default_guidance,
            }
            for key, value in MODEL_SPECS.items()
        },
    }


def _snapshot(
    backend: str,
    *,
    stable_variant: str,
    local_files_only: bool,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RealToSimError(
            "huggingface-hub is required for learned material inference"
        ) from exc
    spec = MODEL_SPECS[backend]
    if backend == "matfuse":
        patterns = MATFUSE_PATTERNS
    else:
        unet_pattern = "unet_lcm/*" if stable_variant == "lcm" else "unet/*"
        patterns = (*STABLEMATERIALS_COMMON_PATTERNS, unet_pattern)
    try:
        return Path(
            snapshot_download(
                spec.repo_id,
                revision=spec.revision,
                allow_patterns=list(patterns),
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        mode = "cached" if local_files_only else "pinned Hugging Face"
        raise RealToSimError(
            f"failed to resolve {mode} weights for {spec.repo_id}@{spec.revision}: {exc}"
        ) from exc


@contextmanager
def _import_root(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass


def _module_from_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RealToSimError(f"cannot import pinned model pipeline: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RealToSimError(
            "MPS was requested but torch.backends.mps.is_available() is false"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RealToSimError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )
    return torch.device(requested)


def _dtype(torch: Any, backend: str, device: Any, requested: str) -> Any:
    if requested == "auto":
        # MatFuse lazy-loads the paper's OpenAI CLIP encoder after pipeline.to().
        # float32 keeps that model and its placeholder image in one dtype on MPS.
        if backend == "matfuse" or device.type == "cpu":
            return torch.float32
        return torch.float16
    return {"float32": torch.float32, "float16": torch.float16}[requested]


def _load_matfuse(snapshot: Path, torch: Any, device: Any, dtype: Any) -> Any:
    with _import_root(snapshot):
        module = _module_from_file(
            "_nanousd_matfuse_pipeline", snapshot / "pipeline_matfuse.py"
        )
        pipeline = module.MatFusePipeline.from_pretrained(
            str(snapshot), torch_dtype=dtype
        )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(desc="MatFuse")
    return pipeline


def _load_stablematerials(
    snapshot: Path,
    torch: Any,
    device: Any,
    dtype: Any,
    *,
    variant: str,
) -> Any:
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        LCMScheduler,
        UNet2DConditionModel,
    )
    from transformers import (
        CLIPProcessor,
        CLIPTextModelWithProjection,
        CLIPTokenizerFast,
        CLIPVisionModelWithProjection,
    )

    module = _module_from_file(
        "_nanousd_stablematerials_pipeline",
        snapshot / "pipeline.py",
    )
    unet_folder = "unet_lcm" if variant == "lcm" else "unet"
    unet = UNet2DConditionModel.from_pretrained(
        snapshot / unet_folder,
        torch_dtype=dtype,
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(
        snapshot / "vae",
        torch_dtype=dtype,
        local_files_only=True,
    )
    scheduler = DDIMScheduler.from_pretrained(
        snapshot / "scheduler", local_files_only=True
    )
    if variant == "lcm":
        scheduler = LCMScheduler.from_config(scheduler.config)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        snapshot / "text_encoder",
        torch_dtype=dtype,
        local_files_only=True,
    )
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
        snapshot / "vision_encoder",
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipeline = module.StableMaterialsPipeline(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=CLIPTokenizerFast.from_pretrained(
            snapshot / "tokenizer", local_files_only=True
        ),
        vision_encoder=vision_encoder,
        processor=CLIPProcessor.from_pretrained(
            snapshot / "processor", local_files_only=True
        ),
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(desc=f"StableMaterials {variant.upper()}")
    return pipeline


def _requests(root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    root = Path(root).resolve()
    if root.is_file():
        paths = [root]
    elif (root / "material-request.json").is_file():
        paths = [root / "material-request.json"]
    else:
        paths = sorted(root.glob("*/material-request.json"))
    if not paths:
        raise RealToSimError(f"no material-request.json files found under {root}")
    records: list[tuple[str, Path, dict[str, Any]]] = []
    roles: set[str] = set()
    for path in paths:
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RealToSimError(f"invalid material request: {path}") from exc
        if request.get("contract") != "nanousd-rts-pbr-atlas-v1":
            raise RealToSimError(f"unsupported material request contract in {path}")
        role = str(request.get("role", "")).replace("_", "-")
        if not role or role in roles:
            raise RealToSimError("material request roles must be non-empty and unique")
        roles.add(role)
        records.append((role, path, request))
    return records


def _prompt(request: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    if request.get("model_prompt"):
        return str(request["model_prompt"])
    kind = str(request.get("template_kind", ""))
    return MODEL_PROMPTS.get(
        kind, str(request.get("prompt", "high-quality PBR material"))
    )


def _palette(request: dict[str, Any]) -> np.ndarray:
    source = request.get("measured_palette", {})
    try:
        dark = np.asarray(source["dark"], dtype=np.float32)
        median = np.asarray(source["median"], dtype=np.float32)
        light = np.asarray(source["light"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise RealToSimError(
            "MatFuse palette conditioning requires dark/median/light RGB"
        ) from exc
    if any(value.shape != (3,) for value in (dark, median, light)):
        raise RealToSimError("material request palette colors must be RGB triples")
    return np.clip(
        np.stack((dark, (dark + median) * 0.5, median, (median + light) * 0.5, light)),
        0.0,
        1.0,
    )


def _save_rgb(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path)


def _save_scalar(image: Image.Image, path: Path) -> None:
    image.convert("L").save(path)


def _save_matfuse(result: dict[str, Any], destination: Path) -> dict[str, Any]:
    diffuse = result["diffuse"][0]
    normal = result["normal"][0]
    roughness = result["roughness"][0]
    specular = result["specular"][0]
    _save_rgb(diffuse, destination / "baseColor.png")
    _save_rgb(normal, destination / "normal.png")
    _save_scalar(roughness, destination / "roughness.png")
    _save_rgb(specular, destination / "specular.png")
    Image.new("L", diffuse.size, 0).save(destination / "metallic.png")
    Image.new("L", diffuse.size, 255).save(destination / "ao.png")
    return {
        "native_outputs": ["baseColor", "normal", "roughness", "specular"],
        "adapter": {
            "diffuse_to_baseColor": True,
            "roughness_rgb_to_scalar": True,
            "specular_preserved_as": "specular.png",
            "metallic": "neutral dielectric value because MatFuse predicts specular, not metalness",
            "ao": "neutral white because MatFuse does not predict ambient occlusion",
        },
    }


def _save_stablematerials(material: Any, destination: Path) -> dict[str, Any]:
    _save_rgb(material.basecolor, destination / "baseColor.png")
    _save_rgb(material.normal, destination / "normal.png")
    _save_scalar(material.height, destination / "height.png")
    _save_scalar(material.roughness, destination / "roughness.png")
    _save_scalar(material.metallic, destination / "metallic.png")
    Image.new("L", material.basecolor.size, 255).save(destination / "ao.png")
    return {
        "native_outputs": ["baseColor", "normal", "height", "roughness", "metallic"],
        "adapter": {
            "height_preserved_as": "height.png",
            "ao": "neutral white because StableMaterials does not predict ambient occlusion",
        },
    }


def _map_manifest(directory: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "baseColor.png",
        "normal.png",
        "roughness.png",
        "metallic.png",
        "ao.png",
        "specular.png",
        "height.png",
    ):
        path = directory / name
        if path.is_file():
            with Image.open(path) as image:
                size = list(image.size)
                mode = image.mode
            result[name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "size": size,
                "mode": mode,
            }
    return result


def _sync(torch: Any, device: Any) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _replace_directory(destination: Path, staged: Path) -> None:
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    staged.rename(destination)
    if backup.exists():
        shutil.rmtree(backup)


def generate_material_bundle(
    request_bundle: Path,
    output_bundle: Path,
    *,
    backend: str,
    device: str = "auto",
    dtype: str = "auto",
    seed: int = 42,
    steps: int | None = None,
    guidance_scale: float | None = None,
    stable_variant: str = "lcm",
    prompt_override: str | None = None,
    use_matfuse_palette: bool = True,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Run one pinned official model and emit an external PBR atlas bundle."""

    if backend not in MODEL_SPECS:
        raise RealToSimError(f"unsupported learned material backend: {backend}")
    if stable_variant not in {"base", "lcm"}:
        raise RealToSimError("StableMaterials variant must be base or lcm")
    if not isinstance(seed, int) or seed < 0:
        raise RealToSimError("material seed must be a non-negative integer")
    spec = MODEL_SPECS[backend]
    steps = spec.default_steps if steps is None else steps
    guidance_scale = spec.default_guidance if guidance_scale is None else guidance_scale
    if not 1 <= steps <= 200:
        raise RealToSimError("material inference steps must be within [1, 200]")
    if not math.isfinite(guidance_scale) or not 0.0 <= guidance_scale <= 30.0:
        raise RealToSimError("material guidance scale must be within [0, 30]")
    records = _requests(request_bundle)
    torch, diffusers = _require_runtime()
    execution_device = _device(torch, device)
    execution_dtype = _dtype(torch, backend, execution_device, dtype)
    snapshot = _snapshot(
        backend,
        stable_variant=stable_variant,
        local_files_only=local_files_only,
    )
    load_started = time.perf_counter()
    if backend == "matfuse":
        pipeline = _load_matfuse(snapshot, torch, execution_device, execution_dtype)
    else:
        pipeline = _load_stablematerials(
            snapshot,
            torch,
            execution_device,
            execution_dtype,
            variant=stable_variant,
        )
    _sync(torch, execution_device)
    load_seconds = time.perf_counter() - load_started

    destination = Path(output_bundle).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    role_manifests: list[dict[str, Any]] = []
    inference_total = 0.0
    try:
        for role, request_path, request in records:
            role_directory = staged / role
            role_directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(request_path, role_directory / "material-request.json")
            # One articulated object should retain one material identity across
            # its world-attached cavity and joint-attached moving interior.
            role_seed = seed
            prompt = _prompt(request, prompt_override)
            generator = torch.Generator(device=execution_device).manual_seed(role_seed)
            _sync(torch, execution_device)
            started = time.perf_counter()
            if backend == "matfuse":
                result = pipeline(
                    text=prompt,
                    palette=_palette(request) if use_matfuse_palette else None,
                    height=spec.native_resolution,
                    width=spec.native_resolution,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="pil",
                )
                adapter = _save_matfuse(result, role_directory)
                conditioning = (
                    ["text", "measured-front-palette"]
                    if use_matfuse_palette
                    else ["text"]
                )
            else:
                result = pipeline(
                    prompt=prompt,
                    height=spec.native_resolution,
                    width=spec.native_resolution,
                    tileable=True,
                    num_images_per_prompt=1,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="pil",
                )
                adapter = _save_stablematerials(result.images[0], role_directory)
                conditioning = ["text"]
            _sync(torch, execution_device)
            elapsed = time.perf_counter() - started
            inference_total += elapsed
            role_manifest = {
                "role": role,
                "request_sha256": sha256_file(role_directory / "material-request.json"),
                "prompt": prompt,
                "conditioning": conditioning,
                "seed": role_seed,
                "inference_seconds": elapsed,
                "maps": _map_manifest(role_directory),
                **adapter,
            }
            (role_directory / "manifest.json").write_text(
                json.dumps(role_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            role_manifests.append(role_manifest)

        manifest = {
            "schema_version": MATERIAL_BUNDLE_SCHEMA,
            "contract": "nanousd-rts-learned-pbr-bundle-v1",
            "backend": spec.backend,
            "model": {
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "license": spec.license,
                "paper": spec.paper,
                "variant": stable_variant
                if backend == "stablematerials"
                else "paper-diffusers",
                "official_weights": True,
            },
            "runtime": {
                "device": execution_device.type,
                "dtype": str(execution_dtype).replace("torch.", ""),
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "platform": platform.platform(),
                "model_load_seconds": load_seconds,
                "inference_seconds": inference_total,
            },
            "generation": {
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "native_resolution": spec.native_resolution,
                "tileable": backend == "stablematerials",
            },
            "request_digest": content_digest([item[2] for item in records]),
            "roles": role_manifests,
            "provenance": {
                "measured": False,
                "learned": True,
                "source_requests_preserved": True,
                "checkpoint_cache_excluded_from_bundle": True,
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _replace_directory(destination, staged)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    finally:
        del pipeline
        if execution_device.type == "mps":
            torch.mps.empty_cache()
        elif execution_device.type == "cuda":
            torch.cuda.empty_cache()
    return manifest
