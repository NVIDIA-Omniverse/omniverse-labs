# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal ovstage-backed adapter for the ovrtx 0.4 Python renderer.

The prerelease ovrtx C API supports attaching an externally-owned ovstage, but
its Python ``Renderer`` does not expose those calls yet. This module adds only
that narrow bridge. The renderer always uses BORROW mode and renders directly
from the caller-owned ovstage without per-frame replication.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

from ovrtx import Renderer, RendererConfig
from ovrtx._src import bindings
from ovrtx._src.renderer import Operation, RenderProductSetOutputs


class OvstageRenderer(Renderer):
    """An ovrtx ``Renderer`` populated from an attached ovstage."""

    _ORDINAL = 1

    @property
    def ovstage(self):
        """Caller-owned stage shared with ovrtx and ovphysx."""
        return self._ovstage

    @property
    def ordinal(self) -> int:
        """Latest sealed ovstage ordinal visible to rendering."""
        return self._render_ordinal

    def __init__(
        self,
        *,
        usd_file: str | os.PathLike[str] | None = None,
        usda: str | None = None,
    ):
        if (usd_file is None) == (usda is None):
            raise ValueError("Pass exactly one of usd_file or usda")

        # ``_attach_mode`` is intentionally private in the ovrtx 0.4 Python
        # bindings.  It is nevertheless the bridge's required native config
        # entry: BORROW lets the renderer consume the caller-owned ovstage
        # directly rather than making a replicated renderer stage.
        self._ovstage = None
        self._ovstage_attached = False
        self._render_ordinal = self._ORDINAL
        self._dll_dir_cookies = []
        config = RendererConfig()
        config._attach_mode = bindings._AttachMode.BORROW
        self._prepare_windows_dll_search()

        # ovrtx must own the shared Carbonite/usdrt framework before ovstage
        # population starts. The opposite order can produce an all-black frame.
        super().__init__(config=config)
        try:
            self._configure_attach_api()
            self._create_and_populate_stage(usd_file=usd_file, usda=usda)
            self._attach_and_initialize_stage()
        except Exception:
            self.__del__()
            raise

    def _prepare_windows_dll_search(self) -> None:
        """Expose ovrtx's nested flat-runtime DLL closure on Windows."""
        if os.name != "nt":
            return
        hint = bindings.OVRTX_LIBRARY_PATH_HINT or os.environ.get(
            "OVRTX_LIBRARY_PATH_HINT"
        )
        if hint:
            ovrtx_bin = Path(hint)
        else:
            import ovrtx  # noqa: PLC0415

            ovrtx_bin = Path(ovrtx.__file__).resolve().parent / "bin"
        dll_dirs = sorted({path.parent for path in ovrtx_bin.rglob("*.dll")})
        for directory in dll_dirs:
            try:
                self._dll_dir_cookies.append(os.add_dll_directory(str(directory)))
            except OSError:
                pass
        os.environ["PATH"] = os.pathsep.join(
            [*(str(path) for path in dll_dirs), os.environ.get("PATH", "")]
        )

    def _configure_attach_api(self) -> None:
        lib = self._bindings._lib
        renderer_p = ctypes.POINTER(bindings.ovrtx_renderer_t)
        lib.ovrtx_attach_ovstage.argtypes = [renderer_p, ctypes.c_void_p]
        lib.ovrtx_attach_ovstage.restype = bindings.ovrtx_result_t
        lib.ovrtx_detach_ovstage.argtypes = [renderer_p]
        lib.ovrtx_detach_ovstage.restype = bindings.ovrtx_result_t
        lib.ovrtx_update_from_stage.argtypes = [renderer_p, ctypes.c_uint64]
        lib.ovrtx_update_from_stage.restype = bindings.ovrtx_enqueue_result_t
        lib.ovrtx_step_with_stage.argtypes = [
            renderer_p,
            bindings.ovrtx_render_product_set_t,
            ctypes.c_double,
            ctypes.c_uint64,
            ctypes.POINTER(bindings.ovrtx_step_result_handle_t),
        ]
        lib.ovrtx_step_with_stage.restype = bindings.ovrtx_enqueue_result_t

    def _create_and_populate_stage(self, *, usd_file, usda) -> None:
        # Load the native ovstage bundled with this exact ovrtx build. The
        # standalone prerelease artifacts currently have binary build skew.
        ovrtx_bin = Path(self._bindings.library_path).resolve().parent
        os.environ["OVSTAGE_LIBRARY_PATH_HINT"] = str(ovrtx_bin)

        import ovstage  # noqa: PLC0415 - must follow ovrtx initialization

        ovstage._src.bindings.OVSTAGE_LIBRARY_PATH_HINT = str(ovrtx_bin)
        stage = ovstage.Stage("ov-fmi")
        try:
            if usd_file is not None:
                ovstage.population.open_usd(
                    stage,
                    str(Path(usd_file).resolve()),
                    ordinal=self._ORDINAL,
                    domains=(
                        ovstage.PopulationDomain.RENDERING
                        | ovstage.PopulationDomain.PHYSICS
                    ),
                )
            else:
                ovstage.population.open_usd_from_string(
                    stage,
                    usda,
                    ordinal=self._ORDINAL,
                    domains=(
                        ovstage.PopulationDomain.RENDERING
                        | ovstage.PopulationDomain.PHYSICS
                    ),
                )
            stage.advance_write_floor(self._ORDINAL).wait()
        except Exception:
            stage.destroy()
            raise
        self._ovstage = stage

    def _attach_and_initialize_stage(self) -> None:
        stage_ptr = ctypes.cast(self._ovstage._inst, ctypes.c_void_p)
        result = self._bindings._lib.ovrtx_attach_ovstage(self._handle, stage_ptr)
        if result.status != bindings.OVRTX_API_SUCCESS:
            error = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to attach ovstage: {error}")
        self._ovstage_attached = True
        result = self._bindings._lib.ovrtx_update_from_stage(
            self._handle, self._ORDINAL
        )
        self._operation_from_result(result, "ovrtx_update_from_stage").wait()

    def update_from_ovstage(self, ordinal: int) -> None:
        """Select a sealed ovstage ordinal for the next frame.

        The native update is a no-op in BORROW mode because ovrtx already shares
        ovstage's Fabric; the call keeps ordinal selection explicit and uniform.
        """
        if ordinal < self._render_ordinal:
            raise ValueError(
                f"ovstage ordinal cannot move backwards: {ordinal} < {self._render_ordinal}"
            )
        result = self._bindings._lib.ovrtx_update_from_stage(self._handle, ordinal)
        self._operation_from_result(result, "ovrtx_update_from_stage").wait()
        self._render_ordinal = ordinal

    def _operation_from_result(self, result, name: str, **kwargs) -> Operation:
        if result.status != bindings.OVRTX_API_SUCCESS:
            error = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue {name}: {error}")
        return Operation(
            renderer=self,
            op_id=result.op_index.value,
            operation_name=name,
            **kwargs,
        )

    def step_async(
        self,
        render_products: set[str],
        delta_time: float,
        *,
        ordinal: int | None = None,
    ) -> "Operation":
        """Use the attached-stage counterpart of ``Renderer.step_async``."""
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")
        if delta_time < 0:
            raise ValueError(f"delta_time must be non-negative, got {delta_time}")

        if ordinal is None:
            ordinal = self._render_ordinal
        else:
            ordinal = self._normalize_ordinal(ordinal)
            if ordinal < self._render_ordinal:
                raise ValueError(
                    "ovstage ordinal cannot move backwards: "
                    f"{ordinal} < {self._render_ordinal}"
                )
            self._render_ordinal = ordinal

        product_strings = [
            bindings.ovx_string_t(path)
            for path in render_products
            if path and str(path).strip()
        ]
        if not product_strings:
            raise ValueError("At least one valid render product is required")
        product_array = (bindings.ovx_string_t * len(product_strings))(*product_strings)
        product_set = bindings.ovrtx_render_product_set_t(
            render_products=product_array,
            num_render_products=len(product_strings),
        )
        step_handle = bindings.ovrtx_step_result_handle_t()
        result = self._bindings._lib.ovrtx_step_with_stage(
            self._handle,
            product_set,
            delta_time,
            ordinal,
            ctypes.byref(step_handle),
        )

        def fetch_step(timeout_ns: Optional[int] = None) -> RenderProductSetOutputs:
            try:
                return self._fetch_step_results(step_handle, timeout_ns)
            except Exception as exc:
                try:
                    self._bindings.destroy_results(self._handle, step_handle)
                except Exception:
                    pass
                raise RuntimeError(f"Failed to fetch step results: {exc}") from exc

        return self._operation_from_result(
            result,
            f"step_with_stage(dt={delta_time}, ordinal={ordinal})",
            handle=step_handle,
            fetch_fn=fetch_step,
            cleanup_fn=lambda: self._bindings.destroy_results(
                self._handle, step_handle
            ),
        )

    def _detach_ovstage(self) -> None:
        if getattr(self, "_ovstage_attached", False) and getattr(
            self, "_handle", None
        ) is not None:
            try:
                self._bindings._lib.ovrtx_detach_ovstage(self._handle)
            except Exception:
                pass
            self._ovstage_attached = False

    def __del__(self):
        stage = getattr(self, "_ovstage", None)
        try:
            self._detach_ovstage()
            self._ovstage = None
            if getattr(self, "_handle", None) is not None:
                try:
                    self._force_unbind_all()
                    self._force_unmap_all()
                    self._bindings.destroy_renderer(self._handle)
                except Exception:
                    pass
                finally:
                    self._handle = None
                    self._bindings = None
                    self._config = None
                    self._path_dict = None
        finally:
            if stage is not None:
                try:
                    stage.destroy()
                except Exception:
                    pass
