from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from nanousd_rts.core import RealToSimError, sha256_file
from nanousd_rts.learned_materials import (
    MATFUSE_BACKEND,
    STABLEMATERIALS_BACKEND,
    _palette,
    _requests,
    _save_matfuse,
    _save_stablematerials,
)
from nanousd_rts.material_preview import write_material_comparison
from nanousd_rts.mesh_completion import PBR_MAPS, _external_material_maps


def _rgb(color: tuple[int, int, int], size: int = 128) -> Image.Image:
    return Image.new("RGB", (size, size), color)


def _scalar(value: int, size: int = 128) -> Image.Image:
    return Image.new("L", (size, size), value)


class LearnedMaterialTests(unittest.TestCase):
    def test_request_discovery_and_measured_palette_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {
                "contract": "nanousd-rts-pbr-atlas-v1",
                "role": "static-cavity",
                "measured_palette": {
                    "dark": [0.0, 0.1, 0.2],
                    "median": [0.4, 0.5, 0.6],
                    "light": [0.8, 0.9, 1.0],
                },
            }
            role = root / "static-cavity"
            role.mkdir()
            path = role / "material-request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            records = _requests(root)
            self.assertEqual(records[0][0], "static-cavity")
            palette = _palette(records[0][2])
            self.assertEqual(palette.shape, (5, 3))
            np.testing.assert_allclose(
                palette[2], request["measured_palette"]["median"]
            )

    def test_matfuse_adapter_preserves_specular_without_calling_it_metallic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = {
                "diffuse": [_rgb((80, 70, 60))],
                "normal": [_rgb((128, 128, 255))],
                "roughness": [_rgb((100, 110, 120))],
                "specular": [_rgb((30, 40, 50))],
            }
            metadata = _save_matfuse(result, root)
            self.assertTrue((root / "specular.png").is_file())
            self.assertEqual(Image.open(root / "metallic.png").getextrema(), (0, 0))
            self.assertIn("specular, not metalness", metadata["adapter"]["metallic"])

    def test_stablematerials_adapter_preserves_height(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            material = SimpleNamespace(
                basecolor=_rgb((120, 110, 100)),
                normal=_rgb((128, 128, 255)),
                height=_scalar(90),
                roughness=_scalar(140),
                metallic=_scalar(10),
            )
            metadata = _save_stablematerials(material, root)
            self.assertTrue((root / "height.png").is_file())
            self.assertEqual(Image.open(root / "ao.png").getextrema(), (255, 255))
            self.assertIn("height", metadata["native_outputs"])

    def test_comparison_writes_both_material_ball_inspectors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundles = {}
            for key, backend in (
                ("matfuse", "matfuse-paper-hf-mps-v1"),
                ("stablematerials", "stablematerials-hf-mps-v1"),
            ):
                bundle = root / key
                role = bundle / "static-cavity"
                role.mkdir(parents=True)
                _rgb((90, 80, 70)).save(role / "baseColor.png")
                _rgb((128, 128, 255)).save(role / "normal.png")
                _scalar(130).save(role / "roughness.png")
                _scalar(0).save(role / "metallic.png")
                _scalar(255).save(role / "ao.png")
                manifest = {
                    "contract": "nanousd-rts-learned-pbr-bundle-v1",
                    "backend": backend,
                    "model": {
                        "repo_id": f"gvecchio/{key}",
                        "revision": "1" * 40,
                        "variant": "test",
                    },
                    "runtime": {"device": "mps", "dtype": "float32"},
                    "generation": {"steps": 1, "guidance_scale": 1.0},
                    "roles": [
                        {
                            "role": "static-cavity",
                            "prompt": "test material",
                            "inference_seconds": 0.1,
                        }
                    ],
                }
                (bundle / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                bundles[key] = bundle
            output = write_material_comparison(
                bundles["matfuse"],
                bundles["stablematerials"],
                output=root / "index.html",
            )
            self.assertIn(
                "Official MatFuse vs StableMaterials",
                output.read_text(encoding="utf-8"),
            )
            self.assertTrue(
                (bundles["matfuse"] / "static-cavity" / "material-ball.png").is_file()
            )
            self.assertTrue(
                (
                    bundles["stablematerials"] / "static-cavity" / "material-ball.png"
                ).is_file()
            )

    def test_learned_bundle_import_verifies_every_map_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "matfuse"
            role = bundle / "static-cavity"
            role.mkdir(parents=True)
            for name in PBR_MAPS:
                if name in {"baseColor.png", "normal.png"}:
                    _rgb((80, 90, 100)).save(role / name)
                else:
                    _scalar(120).save(role / name)
            manifest = {
                "contract": "nanousd-rts-learned-pbr-bundle-v1",
                "backend": MATFUSE_BACKEND,
                "model": {"repo_id": "gvecchio/MatFuse", "revision": "1" * 40},
                "generation": {"steps": 50},
                "runtime": {"device": "mps"},
                "roles": [
                    {
                        "role": "static-cavity",
                        "maps": {
                            name: {"sha256": sha256_file(role / name)}
                            for name in PBR_MAPS
                        },
                    }
                ],
            }
            (bundle / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            imported = _external_material_maps(
                bundle,
                root / "accepted",
                role_slug="static-cavity",
            )
            self.assertTrue(imported["provenance"]["learned"])
            self.assertEqual(
                imported["provenance"]["learned_bundle"]["backend"],
                MATFUSE_BACKEND,
            )
            _scalar(121).save(role / "roughness.png")
            with self.assertRaises(RealToSimError):
                _external_material_maps(
                    bundle,
                    root / "rejected",
                    role_slug="static-cavity",
                )

    def test_comparison_rejects_swapped_model_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundles = {}
            for key, backend in (
                ("matfuse", STABLEMATERIALS_BACKEND),
                ("stablematerials", MATFUSE_BACKEND),
            ):
                bundle = root / key
                role = bundle / "static-cavity"
                role.mkdir(parents=True)
                for name in PBR_MAPS:
                    (
                        _rgb((90, 80, 70))
                        if name in {"baseColor.png", "normal.png"}
                        else _scalar(100)
                    ).save(role / name)
                (bundle / "manifest.json").write_text(
                    json.dumps(
                        {
                            "contract": "nanousd-rts-learned-pbr-bundle-v1",
                            "backend": backend,
                            "model": {},
                            "runtime": {},
                            "generation": {},
                            "roles": [
                                {
                                    "role": "static-cavity",
                                    "prompt": "test",
                                    "inference_seconds": 0.1,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                bundles[key] = bundle
            with self.assertRaises(RealToSimError):
                write_material_comparison(
                    bundles["matfuse"],
                    bundles["stablematerials"],
                    output=root / "index.html",
                )


if __name__ == "__main__":
    unittest.main()
