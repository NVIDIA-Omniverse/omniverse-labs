# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nanousd-backed OVRTX compatibility layer.

This package intentionally uses the public ``ovrtx`` module name. Put this
directory ahead of the real OVRTX wheel on ``PYTHONPATH`` when you want a
nanousd renderer to be a drop-in replacement for OVRTX-shaped callers.

The implementation keeps the OVRTX Python object model at the boundary and
uses private nanousd backend bindings internally. It is not a fork of the
OVRTX runtime; unsupported Fabric/runtime-stage semantics fail explicitly
instead of silently pretending to work.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import math as _math
import os
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

__version__ = "0.3.0+nanousd"
OVRTX_LIBRARY_PATH_HINT = None
OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP = "omni:selectionOutlineGroup"
OVRTX_ATTR_NAME_PICKABLE = "omni:pickable"
OVRTX_RENDER_VAR_PICK_HIT = "ovrtx_pick_hit"
OVRTX_PICK_FLAG_GIZMO = 1
OVRTX_PICK_FLAG_INCLUDE_TRACKED_INFO = 2
OVRTX_PICK_HIT_MAGIC = 1448105032
OVRTX_PICK_HIT_VERSION = 1
_USD_DEFAULT_HORIZONTAL_APERTURE = 20.955
_USD_DEFAULT_VERTICAL_APERTURE = 15.2908
_USD_DEFAULT_CAMERA_APERTURE_ASPECT = _USD_DEFAULT_HORIZONTAL_APERTURE / _USD_DEFAULT_VERTICAL_APERTURE
_USD_DEFAULT_INCLUDED_PURPOSES = ("default", "render")
_USD_DEFAULT_MATERIAL_BINDING_PURPOSES = ("full", "")
_USD_DISABLE_DEPTH_OF_FIELD_FSTOP = 1.0e30
_NATIVE_OVRTX_CAMERA_EXPOSURE_ATTRIBUTES = frozenset(
    {
        "exposure:fStop",
        "exposure:responsivity",
        "exposure:time",
    }
)
_NATIVE_OVRTX_RENDER_EXPOSURE_FLOAT_ATTRIBUTES = frozenset(
    {
        *_NATIVE_OVRTX_CAMERA_EXPOSURE_ATTRIBUTES,
        "omni:rtx:autoExposure:whitePointScale",
        "rtx:post:tonemap:fNumber",
        "omni:rtx:post:tonemap:fNumber",
        "rtx:post:tonemap:cm2Factor",
        "omni:rtx:post:tonemap:cm2Factor",
    }
)
_NATIVE_OVRTX_RENDER_EXPOSURE_BOOL_ATTRIBUTES = frozenset(
    {
        "omni:rtx:autoExposure:enabled",
    }
)
_NATIVE_OVRTX_RENDER_EXPOSURE_ATTRIBUTES = (
    _NATIVE_OVRTX_RENDER_EXPOSURE_FLOAT_ATTRIBUTES | _NATIVE_OVRTX_RENDER_EXPOSURE_BOOL_ATTRIBUTES
)

_NATIVE_MUTABLE_COLOR_ATTRIBUTES = frozenset(
    {
        "primvars:displayColor",
        "displayColor",
        "inputs:diffuseColor",
        "inputs:diffuse_color_constant",
        "inputs:base_color",
        "inputs:baseColor",
        "inputs:color",
        "color",
    }
)


def _is_native_mutable_xform_attribute(attribute_name: str, semantic: Semantic) -> bool:
    return semantic == Semantic.XFORM_MAT4x4 or attribute_name in ("omni:xform", "xformOp:transform")


def _is_native_mutable_color_attribute(attribute_name: str) -> bool:
    return attribute_name in _NATIVE_MUTABLE_COLOR_ATTRIBUTES


def _is_native_mutable_visibility_attribute(attribute_name: str) -> bool:
    return attribute_name == "visibility"


_NATIVE_RUNTIME_CAMERA_ATTRIBUTES = frozenset(
    {
        "omni:xform",
        "xformOp:transform",
        "focalLength",
        "horizontalAperture",
        "verticalAperture",
        "horizontalApertureOffset",
        "verticalApertureOffset",
        "clippingRange",
    }
)

_NATIVE_RUNTIME_RENDER_PRODUCT_ATTRIBUTES = frozenset(
    {
        "camera",
        "products",
        "orderedVars",
        "resolution",
        "renderMode",
        "nanousd:renderMode",
        "omni:rtx:rendermode",
        "omni:rtx:minimal:mode",
        "pixelAspectRatio",
        "aspectRatioConformPolicy",
        "dataWindowNDC",
        "renderingColorSpace",
    }
)


def _reject_dlpack_copy(kwargs: dict[str, Any]) -> None:
    if kwargs.get("copy") is True:
        raise BufferError("copy=True not supported")


def _normalize_cuda_sync_stream(sync_stream: Optional[int]) -> Optional[int]:
    if sync_stream is None:
        return None
    try:
        value = int(sync_stream)
    except Exception as exc:
        raise TypeError("sync_stream must be an integer CUDA stream handle") from exc
    if value < 0:
        raise ValueError("sync_stream must be non-negative")
    return value


def register_schema_paths(binary_package_root: Optional[str] = None) -> None:
    del binary_package_root


def usd_pluginpath_env_keys() -> tuple[str, ...]:
    return (
        "OV_PXR_PLUGINPATH_2511",
        "PXR_PLUGINPATH_NAME",
        "OMNI_USD_PLUGINS_BASE_PATH",
        "OMNI_USD_RTX_SETTINGS_PATH",
    )


class EventStatus(IntEnum):
    PENDING = 0
    COMPLETED = 1
    FAILURE = 2


class Device(IntEnum):
    DEFAULT = 0
    CPU = 1
    CUDA = 2
    CUDA_ARRAY = 3


class Semantic(IntEnum):
    NONE = 0
    XFORM_MAT4x4 = 1
    PATH_STRING = 4
    TOKEN_STRING = 5
    TOKEN_ID = 6
    PATH_ID = 7
    TAG = 8


class PrimMode(IntEnum):
    EXISTING_ONLY = 0
    MUST_EXIST = 1
    CREATE_NEW = 2


class BindingFlag(IntFlag):
    NONE = 0
    OPTIMIZE = 1 << 0


class DataAccess(IntEnum):
    ASYNC = 0
    SYNC = 1


class FilterKind(IntEnum):
    PRIM_TYPE = 0
    HAS_ATTRIBUTE = 1


class AttributeFilterMode(IntEnum):
    NONE = 0
    ALL = 1
    SPECIFIC = 2


class SelectionFillMode(IntEnum):
    EDGE_ONLY = 0
    GLOBAL = 1
    GROUP_OUTLINE_COLOR = 2
    GROUP_FILL_COLOR = 3


@dataclass
class RendererConfig:
    sync_mode: Optional[bool] = None
    log_file_path: Optional[str] = None
    log_level: Optional[str] = None
    enable_profiling: Optional[bool] = None
    read_gpu_transforms: Optional[bool] = None
    keep_system_alive: Optional[bool] = None
    active_cuda_gpus: Optional[str] = None
    use_vulkan: Optional[bool] = None
    selection_outline_enabled: Optional[bool] = None
    selection_outline_width: Optional[int] = None
    selection_fill_mode: Optional[SelectionFillMode] = None
    enable_geometry_streaming: Optional[bool] = None
    enable_geometry_streaming_lod: Optional[bool] = None
    enable_spg: Optional[bool] = None
    enable_motion_bvh: Optional[bool] = None
    enable_rt: Optional[bool] = None
    enable_materials: Optional[bool] = None


@dataclass
class OperationCounter:
    name: str
    current: int
    total: int


@dataclass
class OperationStatus:
    state: EventStatus
    progress: float
    counters: list[OperationCounter]


@dataclass
class AttributeInfo:
    name: str
    dtype: Any
    is_array: bool
    semantic: Semantic
    value: Any = None
    time_samples: Optional[dict[float, Any]] = None


@dataclass
class SelectionGroupStyle:
    outline_color: tuple[float, float, float, float]
    fill_color: tuple[float, float, float, float]


class RenderVarParam:
    def __init__(self, parent: Any = None, record: Any = None):
        self.parent = parent
        self.record = record

    @property
    def name(self) -> str:
        return str(self.record[0]) if self.record is not None else ""

    @property
    def doc(self) -> str:
        return ""

    @property
    def tensor(self) -> Any:
        return self.record[1] if self.record is not None else None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(getattr(self.tensor, "shape", ()))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self):
        return getattr(self.tensor, "dtype", None)

    @property
    def device(self):
        return getattr(self.tensor, "device", "cpu")

    def __dlpack__(self, *args, **kwargs):
        _reject_dlpack_copy(kwargs)
        return self.tensor.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


class RenderVarTensor:
    def __init__(self, parent: Any = None, record: Any = None):
        self.parent = parent
        self.record = record

    @property
    def name(self) -> str:
        return str(self.record[0]) if self.record is not None else ""

    @property
    def doc(self) -> str:
        return ""

    @property
    def tensor(self) -> Any:
        return self.record[1] if self.record is not None else None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(getattr(self.tensor, "shape", ()))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self):
        return getattr(self.tensor, "dtype", None)

    @property
    def device(self):
        return getattr(self.tensor, "device", "cpu")

    def __dlpack__(self, *args, **kwargs):
        _reject_dlpack_copy(kwargs)
        return self.tensor.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


@dataclass
class _RenderProductSpec:
    path: str
    width: Optional[int] = None
    height: Optional[int] = None
    camera_path: Optional[str] = None
    camera_paths: Optional[list[str]] = None
    render_vars: list[str] = None
    render_var_paths: list[str] = None
    render_mode: Optional[str] = None
    minimal_mode: Optional[int] = None
    pixel_aspect_ratio: Optional[float] = None
    aspect_ratio_conform_policy: Optional[str] = None
    data_window_ndc: Optional[tuple[float, float, float, float]] = None
    disable_motion_blur: Optional[bool] = None
    disable_depth_of_field: Optional[bool] = None
    instantaneous_shutter: Optional[bool] = None
    product_name: Optional[str] = None
    product_type: Optional[str] = None
    included_purposes: Optional[list[str]] = None
    material_binding_purposes: Optional[list[str]] = None
    rendering_color_space: Optional[str] = None

    def vars_or_default(self) -> list[str]:
        return list(self.render_vars or ["LdrColor"])

    def cameras_or_default(self) -> list[Optional[str]]:
        if self.camera_paths:
            return list(self.camera_paths)
        return [self.camera_path]


@dataclass
class _RenderSettingsSpec:
    path: str
    product_paths: list[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    camera_path: Optional[str] = None
    camera_paths: Optional[list[str]] = None
    render_mode: Optional[str] = None
    minimal_mode: Optional[int] = None
    pixel_aspect_ratio: Optional[float] = None
    aspect_ratio_conform_policy: Optional[str] = None
    data_window_ndc: Optional[tuple[float, float, float, float]] = None
    disable_motion_blur: Optional[bool] = None
    disable_depth_of_field: Optional[bool] = None
    instantaneous_shutter: Optional[bool] = None
    included_purposes: Optional[list[str]] = field(default_factory=lambda: list(_USD_DEFAULT_INCLUDED_PURPOSES))
    material_binding_purposes: Optional[list[str]] = field(
        default_factory=lambda: list(_USD_DEFAULT_MATERIAL_BINDING_PURPOSES)
    )
    rendering_color_space: Optional[str] = None


class DLDataType:
    """Small Python stand-in for OVRTX's DLPack dtype descriptor."""

    TYPE_MAP = {
        "int8": ("int", 8, 1),
        "int16": ("int", 16, 1),
        "int32": ("int", 32, 1),
        "int64": ("int", 64, 1),
        "uint8": ("uint", 8, 1),
        "uint16": ("uint", 16, 1),
        "uint32": ("uint", 32, 1),
        "uint64": ("uint", 64, 1),
        "float16": ("float", 16, 1),
        "float32": ("float", 32, 1),
        "float64": ("float", 64, 1),
        "bool": ("bool", 8, 1),
    }

    def __init__(self, code: str = "float", bits: int = 32, lanes: int = 1):
        self.code = code
        self.bits = int(bits)
        self.lanes = int(lanes)

    @classmethod
    def from_str(cls, type_name: str, lanes: Optional[int] = None) -> "DLDataType":
        if type_name not in cls.TYPE_MAP:
            raise ValueError(f"Unknown type: {type_name}")
        code, bits, default_lanes = cls.TYPE_MAP[type_name]
        return cls(code, bits, lanes if lanes is not None else default_lanes)

    def __repr__(self) -> str:
        lane = "" if self.lanes == 1 else f"x{self.lanes}"
        return f"DLDataType({self.code}{self.bits}{lane})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, DLDataType)
            and self.code == other.code
            and self.bits == other.bits
            and self.lanes == other.lanes
        )


def _numpy_dtype(dtype: Any, semantic: Semantic = Semantic.NONE):
    if semantic == Semantic.XFORM_MAT4x4:
        return np.float64
    if dtype is None:
        return np.float32
    if isinstance(dtype, DLDataType):
        if dtype.code == "float":
            return np.dtype(f"float{dtype.bits}")
        if dtype.code == "int":
            return np.dtype(f"int{dtype.bits}")
        if dtype.code == "uint":
            return np.dtype(f"uint{dtype.bits}")
        if dtype.code == "bool":
            return np.bool_
    if isinstance(dtype, str):
        if dtype in ("double", "float64"):
            return np.float64
        if dtype in ("float", "float32"):
            return np.float32
        if dtype in ("int", "int32"):
            return np.int32
        if dtype in ("uint", "uint32"):
            return np.uint32
        return np.dtype(dtype)
    try:
        return np.dtype(dtype)
    except Exception:
        return np.float32


def _copy_to_cuda_array(array: Any, context: str, sync_stream: Optional[int] = None):
    try:
        import cupy as cp
    except Exception as exc:
        raise NotImplementedError(
            f"{context} requested CUDA mapping, but CuPy is not importable. "
            "Install CuPy or map on Device.CPU."
        ) from exc
    stream = _normalize_cuda_sync_stream(sync_stream)
    try:
        if stream == 1 and hasattr(cp.cuda.Stream, "null"):
            with cp.cuda.Stream.null:
                return cp.asarray(array)
        if stream and stream > 1 and hasattr(cp.cuda, "ExternalStream"):
            with cp.cuda.ExternalStream(stream):
                return cp.asarray(array)
        return cp.asarray(array)
    except Exception as exc:
        raise RuntimeError(f"{context} failed to copy data to CUDA: {exc}") from exc


def _copy_from_cuda_array(array: Any) -> np.ndarray:
    get = getattr(array, "get", None)
    if callable(get):
        return np.asarray(get())
    try:
        import cupy as cp
        return cp.asnumpy(array)
    except Exception:
        return np.asarray(array)


class _CUExternalMemoryWin32(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _CUExternalMemoryHandle(ctypes.Union):
    _fields_ = [
        ("fd", ctypes.c_int),
        ("win32", _CUExternalMemoryWin32),
        ("nvSciBufObject", ctypes.c_void_p),
    ]


class _CUExternalMemoryHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _CUExternalMemoryHandle),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _CUExternalMemoryBufferDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _CudaDriver:
    _lib = None

    @classmethod
    def lib(cls):
        if cls._lib is None:
            path = ctypes.util.find_library("cuda") or "libcuda.so.1"
            lib = ctypes.CDLL(path)
            lib.cuInit.argtypes = [ctypes.c_uint]
            lib.cuInit.restype = ctypes.c_int
            lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            lib.cuDeviceGet.restype = ctypes.c_int
            lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            lib.cuCtxGetCurrent.restype = ctypes.c_int
            lib.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
            lib.cuCtxGetDevice.restype = ctypes.c_int
            lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
            lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
            lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
            lib.cuCtxSetCurrent.restype = ctypes.c_int
            lib.cuImportExternalMemory.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(_CUExternalMemoryHandleDesc),
            ]
            lib.cuImportExternalMemory.restype = ctypes.c_int
            lib.cuExternalMemoryGetMappedBuffer.argtypes = [
                ctypes.POINTER(ctypes.c_ulonglong),
                ctypes.c_void_p,
                ctypes.POINTER(_CUExternalMemoryBufferDesc),
            ]
            lib.cuExternalMemoryGetMappedBuffer.restype = ctypes.c_int
            lib.cuDestroyExternalMemory.argtypes = [ctypes.c_void_p]
            lib.cuDestroyExternalMemory.restype = ctypes.c_int
            cls._lib = lib
        return cls._lib

    @classmethod
    def _check(cls, code: int, operation: str) -> None:
        if int(code) != 0:
            raise RuntimeError(f"{operation} failed with CUDA driver error {int(code)}")

    @classmethod
    def ensure_context(cls, device_id: int) -> None:
        lib = cls.lib()
        cls._check(lib.cuInit(0), "cuInit")
        current = ctypes.c_void_p()
        cls._check(lib.cuCtxGetCurrent(ctypes.byref(current)), "cuCtxGetCurrent")
        if current.value:
            current_dev = ctypes.c_int()
            if int(lib.cuCtxGetDevice(ctypes.byref(current_dev))) == 0 and int(current_dev.value) == int(device_id):
                return
        dev = ctypes.c_int()
        cls._check(lib.cuDeviceGet(ctypes.byref(dev), int(device_id)), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        cls._check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value), "cuDevicePrimaryCtxRetain")
        cls._check(lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")


class _CudaExternalFloatBuffer:
    def __init__(self, fd: int, mem_size: int, count: int, device_id: int):
        self.fd = int(fd)
        self.mem_size = int(mem_size)
        self.count = int(count)
        self.device_id = int(device_id)
        self.ext_mem = ctypes.c_void_p()
        self.ptr = 0
        self._closed = False
        imported = False
        try:
            _CudaDriver.ensure_context(self.device_id)
            lib = _CudaDriver.lib()
            handle_desc = _CUExternalMemoryHandleDesc()
            handle_desc.type = 1  # CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
            handle_desc.handle.fd = self.fd
            handle_desc.size = self.mem_size
            handle_desc.flags = 0
            _CudaDriver._check(
                lib.cuImportExternalMemory(ctypes.byref(self.ext_mem), ctypes.byref(handle_desc)),
                "cuImportExternalMemory",
            )
            imported = True
            self.fd = -1
            buf_desc = _CUExternalMemoryBufferDesc()
            buf_desc.offset = 0
            buf_desc.size = self.mem_size
            buf_desc.flags = 0
            ptr = ctypes.c_ulonglong()
            _CudaDriver._check(
                lib.cuExternalMemoryGetMappedBuffer(ctypes.byref(ptr), self.ext_mem, ctypes.byref(buf_desc)),
                "cuExternalMemoryGetMappedBuffer",
            )
            self.ptr = int(ptr.value)
        finally:
            if not imported and self.fd >= 0:
                try:
                    os.close(self.fd)
                except OSError:
                    pass
                self.fd = -1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.ext_mem.value:
            try:
                _CudaDriver.lib().cuDestroyExternalMemory(self.ext_mem)
            except Exception:
                pass
            self.ext_mem = ctypes.c_void_p()
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = -1

    def __del__(self):
        self.close()


_GPU_XFORM_EXPAND_KERNEL = None


def _get_gpu_xform_expand_kernel(wp: Any):
    global _GPU_XFORM_EXPAND_KERNEL
    if _GPU_XFORM_EXPAND_KERNEL is not None:
        return _GPU_XFORM_EXPAND_KERNEL

    @wp.kernel
    def _expand_xforms(
        source: wp.array(dtype=wp.mat44d),  # type: ignore
        parent_indices: wp.array(dtype=wp.int32),  # type: ignore
        relative_xforms: wp.array(dtype=wp.mat44f),  # type: ignore
        out_xforms: wp.array(dtype=wp.float32, ndim=2),  # type: ignore
    ):
        i = wp.tid()
        p = parent_indices[i]
        s = source[p]
        r = relative_xforms[i]
        for row in range(4):
            for col in range(4):
                out_xforms[i, row * 4 + col] = wp.float32(
                    s[row, 0] * wp.float64(r[0, col])
                    + s[row, 1] * wp.float64(r[1, col])
                    + s[row, 2] * wp.float64(r[2, col])
                    + s[row, 3] * wp.float64(r[3, col])
                )

    _GPU_XFORM_EXPAND_KERNEL = _expand_xforms
    return _GPU_XFORM_EXPAND_KERNEL


def _is_string_semantic(semantic: Any) -> bool:
    try:
        return Semantic(semantic) in (Semantic.TOKEN_STRING, Semantic.PATH_STRING)
    except Exception:
        return False


def _is_string_write_payload(data: Any, *, is_array: bool = False) -> bool:
    if is_array:
        return (
            isinstance(data, list)
            and bool(data)
            and all(isinstance(item, list) and all(isinstance(s, str) for s in item) for item in data)
        )
    return isinstance(data, list) and all(isinstance(item, str) for item in data)


def _validate_prim_paths(prim_paths: list[str]) -> None:
    if not prim_paths or not all(isinstance(p, str) and p.strip() for p in prim_paths):
        raise ValueError(f"all prim_paths must not be empty, got: {prim_paths}")


_ATTRIBUTE_DTYPE_MAP = {
    "float16": ("float", 16),
    "float32": ("float", 32),
    "float64": ("float", 64),
    "int8": ("int", 8),
    "int16": ("int", 16),
    "int32": ("int", 32),
    "int64": ("int", 64),
    "uint8": ("uint", 8),
    "uint16": ("uint", 16),
    "uint32": ("uint", 32),
    "uint64": ("uint", 64),
    "bool": ("bool", 8),
}
_PYTHON_BUILTIN_DTYPE_MAP = {
    float: "float64",
    int: "int64",
    bool: "bool",
}
_NUMPY_KIND_MAP = {
    "f": "float",
    "i": "int",
    "u": "uint",
    "b": "bool",
}


def _resolve_semantic_and_dtype(context: str, semantic: Any, dtype: Any) -> tuple[Semantic, DLDataType]:
    semantic = Semantic(semantic)
    resolved_dtype = dtype if isinstance(dtype, DLDataType) else None
    if semantic == Semantic.XFORM_MAT4x4:
        inferred = DLDataType("float", 64, 16)
        if resolved_dtype is not None and (resolved_dtype.code, resolved_dtype.bits) != (inferred.code, inferred.bits):
            raise ValueError(
                f"{context}: provided dtype (code={resolved_dtype.code}, bits={resolved_dtype.bits}, "
                f"components={resolved_dtype.lanes}) must match scalar type for {semantic!r} "
                f"(code={inferred.code}, bits={inferred.bits})"
            )
        return semantic, inferred
    if semantic in (Semantic.TOKEN_STRING, Semantic.PATH_STRING):
        inferred = DLDataType("uint", 128, 1)
        if resolved_dtype is not None and resolved_dtype != inferred:
            raise ValueError(
                f"{context}: provided dtype (code={resolved_dtype.code}, bits={resolved_dtype.bits}, "
                f"components={resolved_dtype.lanes}) doesn't match required dtype for {semantic!r} "
                f"(code={inferred.code}, bits={inferred.bits}, components={inferred.lanes})"
            )
        return semantic, inferred
    if resolved_dtype is None:
        raise ValueError(f"{context}: dtype is required when semantic is {semantic!r}")
    return semantic, resolved_dtype


def _resolve_new_dtype_and_shape(
    context: str,
    dtype: Any,
    shape: Optional[tuple],
) -> tuple[Semantic, DLDataType, Optional[tuple]]:
    if isinstance(dtype, DLDataType):
        return Semantic.NONE, dtype, None
    if isinstance(dtype, str):
        if dtype == "token":
            return Semantic.TOKEN_STRING, DLDataType("uint", 128, 1), None
        if dtype == "path":
            return Semantic.PATH_STRING, DLDataType("uint", 128, 1), None
        dtype_str = dtype
    elif dtype in _PYTHON_BUILTIN_DTYPE_MAP:
        dtype_str = _PYTHON_BUILTIN_DTYPE_MAP[dtype]
    elif isinstance(dtype, type) and getattr(dtype, "__name__", "") in _ATTRIBUTE_DTYPE_MAP:
        dtype_str = dtype.__name__
    else:
        dtype_obj = dtype
        if hasattr(dtype_obj, "dtype") and hasattr(dtype_obj.dtype, "kind") and hasattr(dtype_obj.dtype, "itemsize"):
            dtype_obj = dtype_obj.dtype
        if hasattr(dtype_obj, "kind") and hasattr(dtype_obj, "itemsize"):
            code = _NUMPY_KIND_MAP.get(dtype_obj.kind)
            if code is None:
                raise ValueError(
                    f"{context}: dtype={dtype!r} is not a supported numeric type. "
                    "Use a float, int, uint, or bool dtype (e.g. np.float32, np.int32, np.uint8, np.bool_)."
                )
            lanes = int(_math.prod(shape)) if shape else 1
            return Semantic.NONE, DLDataType(code, int(dtype_obj.itemsize) * 8, lanes), shape if shape else None
        dtype_str = str(dtype)

    entry = _ATTRIBUTE_DTYPE_MAP.get(dtype_str)
    if entry is None:
        raise ValueError(
            f"{context}: unrecognized dtype={dtype!r}. Use a numpy dtype, Python built-in (float, int), "
            "or a string name ('float32', 'int32', 'token', 'path', etc.)."
        )
    code, bits = entry
    lanes = int(_math.prod(shape)) if shape else 1
    return Semantic.NONE, DLDataType(code, bits, lanes), shape if shape else None


def _resolve_attribute_type_args(
    context: str,
    dtype: Any,
    shape: Optional[tuple],
    semantic: Any,
    flags: Any = None,
) -> tuple[Semantic, DLDataType, Optional[tuple]]:
    semantic = Semantic(semantic)
    if flags is not None:
        valid_flags = int(BindingFlag.OPTIMIZE)
        try:
            flag_value = int(flags)
        except Exception as exc:
            raise ValueError(
                f"Invalid binding flags: {flags!r}. "
                f"Valid flags are: {BindingFlag.NONE!r}, {BindingFlag.OPTIMIZE!r} (combinable with |)."
            ) from exc
        if flag_value & ~valid_flags:
            raise ValueError(
                f"Invalid binding flags: {flags!r}. "
                f"Valid flags are: {BindingFlag.NONE!r}, {BindingFlag.OPTIMIZE!r} (combinable with |)."
            )
    if semantic != Semantic.NONE:
        if shape is not None:
            raise ValueError("shape= must not be provided when semantic= is specified")
        resolved_semantic, resolved_dtype = _resolve_semantic_and_dtype(context, semantic, dtype)
        return resolved_semantic, resolved_dtype, None
    if dtype is None:
        raise ValueError(f"{context}: dtype is required. Use e.g. dtype='float32', shape=(3,)")
    return _resolve_new_dtype_and_shape(context, dtype, shape)


def _numpy_code_bits(dtype: Any, context: str) -> tuple[str, int]:
    np_dtype = np.dtype(dtype)
    code = _NUMPY_KIND_MAP.get(np_dtype.kind)
    if code is None:
        raise TypeError(
            f"{context}: dtype={np_dtype!r} is not a supported numeric type. "
            "Use a float, int, uint, or bool dtype."
        )
    bits = 8 if np_dtype.kind == "b" else np_dtype.itemsize * 8
    return code, int(bits)


def _binding_element_shape(binding: "AttributeBinding") -> tuple[int, ...]:
    if binding.shape is not None:
        return tuple(binding.shape)
    if binding.semantic == Semantic.XFORM_MAT4x4:
        return (4, 4)
    dtype = binding.dtype
    if isinstance(dtype, DLDataType) and dtype.lanes > 1:
        return (int(dtype.lanes),)
    return ()


def _validate_binding_numeric_dtype(binding: "AttributeBinding", arr: np.ndarray, context: str) -> None:
    code, bits = _numpy_code_bits(arr.dtype, context)
    expected = binding.dtype
    if isinstance(expected, DLDataType) and (code, bits) != (expected.code, expected.bits):
        raise ValueError(
            f"{context}: tensor element type (code={code}, bits={bits}) must match binding's scalar type "
            f"(code={expected.code}, bits={expected.bits})"
        )


def _validate_scalar_binding_array(binding: "AttributeBinding", arr: np.ndarray, context: str) -> np.ndarray:
    _validate_binding_numeric_dtype(binding, arr, context)
    prim_count = len(binding._prim_paths)
    element_shape = _binding_element_shape(binding)
    expected_shape = (prim_count,) + element_shape
    if arr.shape == () and expected_shape == (1,):
        return np.ascontiguousarray(arr.reshape(expected_shape))
    if tuple(arr.shape) != expected_shape:
        raise ValueError(f"{context}: tensor shape {tuple(arr.shape)} must match binding shape {expected_shape}")
    return np.ascontiguousarray(arr)


def _validate_array_binding_tensor(binding: "AttributeBinding", arr: np.ndarray, context: str) -> np.ndarray:
    _validate_binding_numeric_dtype(binding, arr, context)
    element_shape = _binding_element_shape(binding)
    if element_shape:
        if arr.ndim < len(element_shape) or tuple(arr.shape[-len(element_shape):]) != element_shape:
            raise ValueError(
                f"{context}: array tensor shape {tuple(arr.shape)} must end with element shape {element_shape}"
            )
    return np.ascontiguousarray(arr)


class ManagedDLTensor:
    """DLPack-compatible wrapper over a NumPy/Torch/Warp-like tensor."""

    def __init__(
        self,
        array: Any,
        manager_ctx: Any = None,
        deleter_callback: Optional[Callable[[Any], None]] = None,
        readonly: bool = False,
    ):
        self._array = array
        self._manager_ctx = manager_ctx
        self._deleter_callback = deleter_callback
        self._readonly = readonly

    def __dlpack__(self, *args, **kwargs):
        return self._array.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        if hasattr(self._array, "__dlpack_device__"):
            return self._array.__dlpack_device__()
        return (1, 0)

    def __array__(self, dtype=None):
        arr = np.asarray(self._array)
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

    @property
    def raw_dltensor(self):
        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(getattr(self._array, "shape", ()))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self):
        return getattr(self._array, "dtype", None)

    @property
    def device(self):
        if hasattr(self._array, "device"):
            return self._array.device
        return "cpu"

    @property
    def data(self) -> int:
        if isinstance(self._array, np.ndarray):
            return int(self._array.__array_interface__["data"][0])
        if hasattr(self._array, "data_ptr"):
            return int(self._array.data_ptr())
        data = getattr(self._array, "data", None)
        ptr = getattr(data, "ptr", None)
        if ptr is not None:
            return int(ptr)
        if hasattr(self._array, "ptr"):
            return int(self._array.ptr)
        return 0

    def numpy(self) -> np.ndarray:
        if isinstance(self._array, np.ndarray):
            return self._array
        get = getattr(self._array, "get", None)
        if callable(get):
            return np.asarray(get())
        if hasattr(self._array, "cpu") and hasattr(self._array, "numpy"):
            return self._array.cpu().numpy()
        return np.from_dlpack(self)

    def to_bytes(self) -> bytes:
        return self.numpy().tobytes()

    def __del__(self):
        if self._deleter_callback is not None:
            try:
                self._deleter_callback(self._manager_ctx)
            except Exception:
                pass
            self._deleter_callback = None


class ManagedStringDLTensor(ManagedDLTensor):
    """DLPack-compatible ovx_string_t-style tensor with Python string access."""

    def __init__(
        self,
        strings: Iterable[str],
        semantic: Semantic = Semantic.TOKEN_STRING,
        manager_ctx: Any = None,
        readonly: bool = True,
    ):
        self._strings = tuple(str(s) for s in strings)
        self._semantic = Semantic(semantic)
        self._byte_arrays = [
            np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
            for text in self._strings
        ]
        records = np.zeros((len(self._strings), 2), dtype=np.uint64)
        for index, encoded in enumerate(self._byte_arrays):
            records[index, 0] = int(encoded.ctypes.data) if encoded.size else 0
            records[index, 1] = int(encoded.size)
        super().__init__(records, manager_ctx=manager_ctx, readonly=readonly)

    @property
    def dtype(self):
        return DLDataType("uint", 128, 1)

    @property
    def semantic(self) -> Semantic:
        return self._semantic

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._strings),)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def strings(self) -> tuple[str, ...]:
        return self._strings

    def to_strings(self) -> list[str]:
        return list(self._strings)


class PendingFetch:
    def __init__(self, fetch_fn: Callable[[Optional[int]], Any]):
        self._fetch_fn = fetch_fn
        self._fetched = False
        self._result = None

    def fetch(self, timeout_ns: Optional[int] = None):
        if self._fetched:
            return self._result
        result = self._fetch_fn(timeout_ns)
        if result is not None:
            self._result = result
            self._fetched = True
            self._fetch_fn = None
        return self._result


class Operation:
    _next_id = 1

    def __init__(
        self,
        renderer: "Renderer",
        op_id: Optional[int] = None,
        handle: Any = None,
        operation_name: str = "",
        fetch_fn: Optional[Callable[[Optional[int]], Any]] = None,
        cleanup_fn: Optional[Callable[[], None]] = None,
        error: Optional[BaseException] = None,
    ):
        self._renderer = renderer
        self._op_id = op_id if op_id is not None else Operation._next_id
        Operation._next_id = max(Operation._next_id + 1, self._op_id + 1)
        self._handle = handle
        self._operation_name = operation_name
        self._fetch_fn = fetch_fn
        self._cleanup_fn = cleanup_fn
        self._error = error
        self._pending_fetch: Optional[PendingFetch] = None
        self._completed = False
        self._completed_result = None

    @property
    def op_id(self) -> int:
        return self._op_id

    def wait(self, timeout_ns: Optional[int] = None):
        del timeout_ns
        if self._error is not None:
            raise RuntimeError(f"Operation '{self._operation_name}' failed: {self._error}") from self._error
        if self._fetch_fn is not None:
            if self._pending_fetch is None:
                self._pending_fetch = PendingFetch(self._fetch_fn)
                self._completed = True
            return self._pending_fetch
        if not self._completed:
            self._completed_result = self._handle if self._handle is not None else True
            self._completed = True
        return self._completed_result

    def query_status(self) -> OperationStatus:
        if self._completed:
            raise RuntimeError("Operation already completed — query_status is no longer available")
        state = EventStatus.FAILURE if self._error is not None else EventStatus.COMPLETED
        return OperationStatus(state=state, progress=1.0, counters=[])


@dataclass
class FrameOutput:
    start_time: float
    end_time: float
    render_vars: dict[str, "RenderVarOutput"]


@dataclass
class ProductOutput:
    name: str
    frames: list[FrameOutput]


class RenderProductSetOutputs:
    def __init__(self, destroy_fn: Callable[[], None], products: dict[str, ProductOutput]):
        self._destroy_fn = destroy_fn
        self._outputs = products
        self._destroyed = False

    def __getitem__(self, key: str) -> ProductOutput:
        return self._outputs[key]

    def __iter__(self):
        return iter(self._outputs)

    def __len__(self) -> int:
        return len(self._outputs)

    def __contains__(self, key: str) -> bool:
        return key in self._outputs

    def keys(self):
        return self._outputs.keys()

    def values(self):
        return self._outputs.values()

    def items(self):
        return self._outputs.items()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False

    def destroy(self):
        if not self._destroyed:
            for product in self._outputs.values():
                for frame in product.frames:
                    for render_var in frame.render_vars.values():
                        render_var._mark_result_destroyed()
            self._destroy_fn()
            self._destroyed = True

    def __del__(self):
        if not self._destroyed:
            try:
                self.destroy()
            except Exception:
                pass


class _MappedRenderVar:
    def __init__(
        self,
        render_var: "RenderVarOutput",
        device: Device = Device.CPU,
        sync_stream: Optional[int] = None,
    ):
        self._render_var = render_var
        self._device = device
        self._sync_stream = _normalize_cuda_sync_stream(sync_stream)
        self.name = render_var.name
        self.type = getattr(render_var, "type", "tensor")
        self.doc = getattr(render_var, "doc", "")
        self.version = getattr(render_var, "version", 1)
        self._tensors: dict[str, ManagedDLTensor] = {}
        self._params: dict[str, ManagedDLTensor] = {}
        self._entered = False
        self._unmapped = False

    @property
    def device(self) -> Device:
        return self._device

    @property
    def tensor(self) -> ManagedDLTensor:
        tensor = self._require_single_tensor(".tensor")
        warnings.warn(
            ".tensor is deprecated for single-tensor render variables; use the mapping directly: "
            "np.from_dlpack(mapped).",
            DeprecationWarning,
            stacklevel=2,
        )
        return tensor

    @property
    def params(self) -> dict[str, RenderVarParam]:
        self._require_mapped()
        return {name: RenderVarParam(parent=self, record=(name, tensor)) for name, tensor in self._params.items()}

    @property
    def wait_event(self) -> Optional[int]:
        return None

    def wait(self) -> None:
        return None

    def wait_on(self, stream: int) -> None:
        del stream
        return None

    def __dlpack__(self, *args, **kwargs):
        _reject_dlpack_copy(kwargs)
        return self._require_single_tensor("__dlpack__").__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self._require_single_tensor("__dlpack_device__").__dlpack_device__()

    def __getitem__(self, key: str) -> RenderVarTensor:
        self._require_mapped()
        try:
            tensor = self._tensors[key]
        except KeyError as exc:
            raise KeyError(
                f"Render variable '{self.name}' has no tensor named {key!r}; available: {list(self._tensors)}"
            ) from exc
        return RenderVarTensor(parent=self, record=(key, tensor))

    def __contains__(self, key: object) -> bool:
        return key in self._tensors

    def __iter__(self):
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def keys(self):
        return self._tensors.keys()

    def values(self):
        return (RenderVarTensor(parent=self, record=(name, tensor)) for name, tensor in self._tensors.items())

    def items(self):
        return (
            (name, RenderVarTensor(parent=self, record=(name, tensor)))
            for name, tensor in self._tensors.items()
        )

    def unmap(self, event: Optional[int] = None, stream: Optional[int] = None) -> None:
        if self._unmapped:
            return
        if event is not None and stream is not None:
            raise ValueError("Cannot specify both event and stream")
        if self._device in (Device.DEFAULT, Device.CPU) and (event is not None or stream is not None):
            raise ValueError("CUDA sync parameters are not valid for CPU render-var mappings")
        self._unmapped = True
        if getattr(self._render_var, "_map_handle", None) is self:
            self._render_var._map_handle = None

    def __enter__(self):
        if self._entered:
            return self
        self._tensors = {
            name: ManagedDLTensor(self._map_array(array), manager_ctx=self, readonly=True)
            for name, array in self._render_var._tensors.items()
        }
        self._params = {
            name: ManagedDLTensor(np.ascontiguousarray(array), manager_ctx=self, readonly=True)
            for name, array in self._render_var._params.items()
        }
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unmap()
        return False

    def __del__(self):
        if not self._unmapped:
            try:
                self.unmap()
            except Exception:
                pass

    def _map_array(self, array: Any) -> Any:
        if self._device in (Device.DEFAULT, Device.CPU):
            return array
        if self._device in (Device.CUDA, Device.CUDA_ARRAY):
            return _copy_to_cuda_array(array, "RenderVarOutput.map", sync_stream=self._sync_stream)
        raise NotImplementedError(f"Unsupported render output mapping device: {self._device!r}")

    def _require_mapped(self) -> None:
        if self._unmapped or not self._entered:
            raise RuntimeError("Mapping not entered or already unmapped")

    def _require_single_tensor(self, operation: str) -> ManagedDLTensor:
        self._require_mapped()
        if len(self._tensors) != 1:
            raise RuntimeError(
                f"Render variable '{self.name}' has {len(self._tensors)} tensors; {operation} requires "
                "a single-tensor render variable; use mapping['<tensor_name>']"
            )
        return next(iter(self._tensors.values()))


MappedRenderVar = _MappedRenderVar


class RenderVarOutput:
    def __init__(
        self,
        name: str,
        handle: Any,
        renderer: "Renderer",
        tensors: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.handle = handle
        self._renderer = renderer
        self._array = handle
        self.type = "tensor"
        self.doc = ""
        self.version = 1
        self._tensors = {str(k): v for k, v in (tensors or {name: handle}).items()}
        self._params = {str(k): v for k, v in (params or {}).items()}
        self._map_handle: Optional[_MappedRenderVar] = None
        self._result_destroyed = False

    def map(self, device: Device = Device.CPU, sync_stream: Optional[int] = None) -> _MappedRenderVar:
        mapped_device = Device(device)
        normalized_stream = _normalize_cuda_sync_stream(sync_stream)
        if mapped_device in (Device.DEFAULT, Device.CPU) and normalized_stream not in (None, 0):
            raise ValueError("sync_stream is only valid for CUDA render-var mappings")
        if mapped_device in (Device.CUDA, Device.CUDA_ARRAY) and self._renderer._backend == "metal":
            raise NotImplementedError("CUDA render-var mapping is not supported by the Metal backend")
        if self._result_destroyed:
            raise RuntimeError(f"Render var '{self.name}' belongs to a destroyed RenderProductSetOutputs")
        if self._map_handle is not None:
            raise RuntimeError(f"Render var '{self.name}' already mapped")
        mapped = _MappedRenderVar(self, device=mapped_device, sync_stream=normalized_stream)
        self._map_handle = mapped
        try:
            return mapped.__enter__()
        except Exception:
            if self._map_handle is mapped:
                self._map_handle = None
            raise

    def release(self) -> None:
        mapped = self._map_handle
        if mapped is not None:
            mapped.unmap()

    def _mark_result_destroyed(self) -> None:
        self._result_destroyed = True


class AttributeMapping:
    def __init__(
        self,
        binding: "AttributeBinding",
        array: Any,
        device: Device = Device.CPU,
        *,
        mapped_array: Any = None,
        commit_callback: Optional[Callable[["AttributeMapping"], None]] = None,
    ):
        self._binding = binding
        self._array = array
        self._device = device
        self._commit_callback = commit_callback
        self._unmapped = False
        self._device_array = None
        if mapped_array is not None:
            tensor_array = mapped_array
            if device in (Device.CUDA, Device.CUDA_ARRAY):
                self._device_array = mapped_array
        elif device in (Device.CUDA, Device.CUDA_ARRAY):
            self._device_array = _copy_to_cuda_array(self._array, "AttributeBinding.map")
            tensor_array = self._device_array
        else:
            tensor_array = self._array
        self._tensor = ManagedDLTensor(tensor_array, manager_ctx=self, readonly=False)

    @property
    def tensor(self) -> ManagedDLTensor:
        if self._unmapped:
            raise RuntimeError("Attribute mapping already unmapped")
        return self._tensor

    @property
    def map_handle(self) -> int:
        return id(self._array)

    @property
    def binding_desc(self):
        return None

    @property
    def device(self) -> Device:
        return self._device

    def _validate_unmap_sync(self, event: Optional[int], stream: Optional[int]) -> None:
        if self._device == Device.CPU and (event is not None or stream is not None):
            raise ValueError("CUDA sync parameters (event/stream) not applicable for CPU-mapped attributes")
        if event is not None and stream is not None:
            raise ValueError("Cannot specify both event and stream; use one or the other")

    def _commit_unmap(self) -> None:
        if self._commit_callback is not None:
            self._commit_callback(self)
        elif self._device_array is not None:
            self._array[...] = _copy_from_cuda_array(self._device_array)
            self._binding.write(self._array)
        else:
            self._binding.write(self._array)
        self._unmapped = True

    def unmap(self, event: Optional[int] = None, stream: Optional[int] = None) -> None:
        if self._unmapped:
            return
        self._validate_unmap_sync(event, stream)
        self._commit_unmap()

    def unmap_async(self, event: Optional[int] = None, stream: Optional[int] = None) -> Operation:
        if self._unmapped:
            raise RuntimeError("Mapping already unmapped")
        self._validate_unmap_sync(event, stream)
        self._commit_unmap()
        return Operation(self._binding._renderer, operation_name="unmap_attribute")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unmap()
        return False


class _NativeGpuTransformBindingState:
    def __init__(
        self,
        renderer: "Renderer",
        binding_handle: int,
        device_id: int,
        source_initial: np.ndarray,
        mesh_ids: np.ndarray,
        parent_indices: np.ndarray,
        relative_xforms: np.ndarray,
    ):
        self.renderer = renderer
        self.binding_handle = int(binding_handle)
        self.device_id = int(device_id)
        self.device = f"cuda:{self.device_id}"
        self.count = int(source_initial.shape[0])
        self.mesh_count = int(mesh_ids.shape[0])
        self.mesh_ids = np.ascontiguousarray(mesh_ids, dtype=np.int32)
        self.parent_indices_host = np.ascontiguousarray(parent_indices, dtype=np.int32)
        self.relative_xforms_host = np.ascontiguousarray(relative_xforms, dtype=np.float32).reshape(self.mesh_count, 4, 4)
        self.external: Optional[_CudaExternalFloatBuffer] = None
        self.source = None
        self.parent_indices = None
        self.relative_xforms = None
        self.out_xforms = None
        self._wp = None
        self._closed = False
        self._initialize(source_initial)

    def _initialize(self, source_initial: np.ndarray) -> None:
        try:
            import warp as wp
        except Exception as exc:
            raise NotImplementedError(
                "RendererConfig(read_gpu_transforms=True) with Device.CUDA attribute mapping requires NVIDIA Warp."
            ) from exc
        self._wp = wp
        nu = self.renderer._nu
        if nu is None:
            raise RuntimeError("Native renderer is not initialized")
        if not all(
            hasattr(nu, name)
            for name in ("get_transforms_interop_info", "set_transform_layout", "translate_instances_gpu")
        ):
            raise NotImplementedError("Native renderer does not expose CUDA/Vulkan transform interop.")
        info = nu.get_transforms_interop_info(self.mesh_count)
        self.external = _CudaExternalFloatBuffer(
            int(info["mem_fd"]),
            int(info["mem_size"]),
            int(info["count"]),
            self.device_id,
        )
        initial = np.ascontiguousarray(source_initial, dtype=np.float64).reshape(self.count, 4, 4)
        self.source = wp.array(initial, dtype=wp.mat44d, device=self.device)
        self.parent_indices = wp.array(self.parent_indices_host, dtype=wp.int32, device=self.device)
        self.relative_xforms = wp.array(self.relative_xforms_host, dtype=wp.mat44f, device=self.device)
        try:
            self.out_xforms = wp.array(
                ptr=int(self.external.ptr),
                shape=(self.mesh_count, 16),
                dtype=wp.float32,
                device=self.device,
                copy=False,
                owner=False,
                capacity=int(info["mem_size"]),
            )
        except TypeError:
            self.out_xforms = wp.array(
                ptr=int(self.external.ptr),
                shape=(self.mesh_count, 16),
                dtype=wp.float32,
                device=self.device,
                copy=False,
                owner=False,
            )
        nu.set_transform_layout(self.mesh_ids)

    def compatible(
        self,
        device_id: int,
        source_count: int,
        mesh_ids: np.ndarray,
        parent_indices: np.ndarray,
        relative_xforms: np.ndarray,
    ) -> bool:
        return (
            not self._closed
            and int(device_id) == self.device_id
            and int(source_count) == self.count
            and np.array_equal(self.mesh_ids, np.asarray(mesh_ids, dtype=np.int32))
            and np.array_equal(self.parent_indices_host, np.asarray(parent_indices, dtype=np.int32))
            and np.array_equal(
                self.relative_xforms_host.reshape(-1, 16),
                np.asarray(relative_xforms, dtype=np.float32).reshape(-1, 16),
            )
        )

    def commit(self) -> None:
        if self._closed or self.mesh_count <= 0:
            return
        wp = self._wp
        if wp is None:
            raise RuntimeError("Warp state is not initialized")
        kernel = _get_gpu_xform_expand_kernel(wp)
        wp.launch(
            kernel=kernel,
            dim=self.mesh_count,
            inputs=[self.source, self.parent_indices, self.relative_xforms, self.out_xforms],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        self.renderer._nu.translate_instances_gpu()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.external is not None:
            self.external.close()
            self.external = None
        self.source = None
        self.parent_indices = None
        self.relative_xforms = None
        self.out_xforms = None


class AttributeBinding:
    def __init__(
        self,
        handle: int,
        semantic: int,
        dtype: Any,
        renderer: "Renderer",
        is_array: bool = False,
        shape: Optional[tuple] = None,
        prim_paths: Optional[list[str]] = None,
        attribute_name: str = "",
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
    ):
        self._handle = handle
        self._semantic = Semantic(semantic)
        self._dtype = dtype
        self._renderer = renderer
        self._is_array = bool(is_array)
        self._shape = shape
        self._prim_paths = list(prim_paths or [])
        self._attribute_name = attribute_name
        self._prim_mode = PrimMode(prim_mode)

    @property
    def handle(self) -> int:
        return self._handle

    @property
    def semantic(self) -> int:
        return self._semantic

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_array(self) -> bool:
        return self._is_array

    @property
    def shape(self) -> Optional[tuple]:
        return self._shape

    @property
    def prim_mode(self) -> PrimMode:
        return self._prim_mode

    def write(
        self,
        data: Any,
        dirty_bits: Optional[bytes] = None,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> None:
        del dirty_bits, cuda_stream, cuda_event
        if self._handle is None:
            raise RuntimeError("AttributeBinding.write() called after unbind(). Create a new binding with bind_attribute().")
        if _is_string_semantic(self._semantic) and DataAccess(data_access) == DataAccess.ASYNC:
            raise ValueError("String attributes (token, path) require DataAccess.SYNC")
        self._renderer._write_bound_attribute(self, data)

    def write_async(
        self,
        data: Any,
        dirty_bits: Optional[bytes] = None,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> Operation:
        if self._handle is None:
            raise RuntimeError(
                "AttributeBinding.write_async() called after unbind(). Create a new binding with bind_attribute()."
            )
        self.write(data, dirty_bits, data_access, cuda_stream, cuda_event)
        return Operation(self._renderer, operation_name="write_attribute")

    def map(self, device: Device = Device.CPU, device_id: int = 0) -> AttributeMapping:
        if self._handle is None:
            raise RuntimeError("AttributeBinding.map() called after unbind(). Create a new binding with bind_attribute().")
        return self._renderer._map_bound_attribute(self, device, device_id)

    def unbind(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            self._renderer._release_native_gpu_transform_binding(int(handle))
        self._renderer._bindings.pop(id(self), None)


_RENDER_PRODUCT_DEFAULTS = (
    "/NUSDRender/Render/OmniverseKit/HydraTextures/ViewportTexture0",
    "/Render/OmniverseKit/HydraTextures/ViewportTexture0",
    "/Render/Product",
)
_RE_RESOLUTION = re.compile(r"\b(?:uniform\s+)?int2\s+resolution\s*=\s*\((\d+)\s*,\s*(\d+)\)")
_RE_CAMERA_REL = re.compile(r"\brel\s+camera\s*=\s*(?:\[(.*?)\]|<([^>]+)>)")
_RE_ORDERED_VARS = re.compile(r"\brel\s+orderedVars\s*=\s*(?:\[(.*?)\]|<([^>]+)>)")
_RE_MATERIAL_BINDING = re.compile(r"\brel\s+material:binding\s*=\s*(?:\[(.*?)\]|<([^>]+)>)")
_RE_SOURCE_NAME = re.compile(r'\b(?:uniform\s+)?string\s+sourceName\s*=\s*"([^"]+)"')
_RE_RENDER_MODE = re.compile(
    r'\b(?:custom\s+)?(?:uniform\s+)?token\s+((?:nanousd:)?renderMode|omni:rtx:rendermode)\s*=\s*"([^"]+)"'
)
_RE_MINIMAL_MODE = re.compile(r"\b(?:custom\s+)?(?:uniform\s+)?int\s+omni:rtx:minimal:mode\s*=\s*(-?\d+)")
_RE_RENDER_PRODUCT = re.compile(r'\b(?:def|over)\s+RenderProduct\s+"([^"]+)"')
_RE_PRIM = re.compile(r'^\s*(?:def|over)\s+(?:(\w+)\s+)?"([^"]+)"')
_RE_REL_ATTR = re.compile(r'^\s*(?:custom\s+)?(?:(prepend|append|add|delete|reorder)\s+)?rel\s+([\w:]+)\s*=\s*(.*?)\s*$')
_RE_ATTR = re.compile(r'^\s*(?:custom\s+)?(?:uniform\s+)?([\w:<>]+(?:\[\])?)\s+([\w:]+)\s*=\s*(.*?)\s*$')
_RE_ATTR_TIME_SAMPLES = re.compile(
    r'^\s*(?:custom\s+)?(?:uniform\s+)?([\w:<>]+(?:\[\])?)\s+([\w:]+)\.timeSamples\s*=\s*(.*?)\s*$'
)
_RE_SUBLAYERS = re.compile(r"\bsubLayers\s*=\s*\[(.*?)\]", re.DOTALL)
_RE_ASSET_REF = re.compile(r"@([^@]+)@")
_RE_AUTHORING_LAYER = re.compile(r'\bstring\s+authoring_layer\s*=\s*"([^"]+)"')
_RE_DEFAULT_PRIM = re.compile(r'\bdefaultPrim\s*=\s*"([^"]+)"')
_RE_TIME_SAMPLE_KEY = re.compile(r'(?<![\w.])([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*:')


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _is_text_usd_layer(path: Path) -> bool:
    """Return true for layers that are safe to scan with USDA regex helpers."""
    try:
        with path.open("rb") as f:
            head = f.read(16)
    except Exception:
        return False
    return head.lstrip().startswith(b"#usda")


def _safe_sidecar_asset_ref(ref: str) -> bool:
    if not ref or "://" in ref or len(ref) > 1024:
        return False
    return not any(ord(ch) < 32 for ch in ref)


def _asset_ref(path: str) -> str:
    return str(Path(path).resolve()).replace("\\", "/")


def _scan_render_metadata(text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for m in _RE_RESOLUTION.finditer(text):
        meta["width"] = int(m.group(1))
        meta["height"] = int(m.group(2))
        break
    for m in _RE_CAMERA_REL.finditer(text):
        paths = _camera_rel_paths(m)
        if paths:
            meta["camera_path"] = paths[0]
            meta["camera_paths"] = paths
        break
    for m in _RE_RENDER_PRODUCT.finditer(text):
        name = m.group(1)
        if name.startswith("/"):
            meta["render_product"] = name
        else:
            # Good enough for the standard OVRTX/Kit viewport product.
            if "ViewportTexture" in name:
                meta["render_product"] = f"/Render/OmniverseKit/HydraTextures/{name}"
            else:
                meta["render_product"] = f"/Render/{name}"
        break
    return meta


def _default_prim_name(text: str) -> Optional[str]:
    match = _RE_DEFAULT_PRIM.search(text)
    return match.group(1) if match else None


def _root_prim_types(text: str) -> dict[str, str]:
    roots: dict[str, str] = {}
    for path, attrs in _scan_prim_index(text).items():
        if path.count("/") != 1:
            continue
        prim_type = getattr(attrs.get("__type__"), "value", None)
        roots[path[1:]] = str(prim_type or "Xform")
    return roots


def _normalize_prefix(path_prefix: Optional[str]) -> Optional[str]:
    if not path_prefix:
        return None
    prefix = str(path_prefix).strip()
    if not prefix or prefix == "/":
        return None
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return prefix.rstrip("/")


def _remap_reference_path(path: Optional[str], path_prefix: Optional[str], default_prim: Optional[str]) -> Optional[str]:
    if not path:
        return path
    prefix = _normalize_prefix(path_prefix)
    if prefix is None:
        return path
    text = str(path)
    default_root = f"/{default_prim}" if default_prim else None
    if default_root and text == default_root:
        return prefix
    if default_root and text.startswith(default_root + "/"):
        return prefix + text[len(default_root):]
    if default_prim:
        return text
    return prefix + text if text.startswith("/") else f"{prefix}/{text}"


def _remap_path_string_value(value: Any, path_prefix: Optional[str], default_prim: Optional[str]) -> Any:
    prefix = _normalize_prefix(path_prefix)
    if prefix is None:
        return value
    if isinstance(value, str):
        return _remap_reference_path(value, prefix, default_prim)
    if isinstance(value, (list, tuple)):
        return [
            str(mapped)
            for mapped in (_remap_reference_path(str(item), prefix, default_prim) for item in value)
            if mapped
        ]
    arr = np.asarray(value)
    if arr.dtype.kind in "OUS":
        return [
            str(mapped)
            for mapped in (
                _remap_reference_path(str(item), prefix, default_prim)
                for item in arr.reshape(-1).tolist()
            )
            if mapped
        ]
    return value


def _remap_path_string_attribute_values(
    attrs: dict[str, AttributeInfo],
    path_prefix: Optional[str],
    default_prim: Optional[str],
) -> dict[str, AttributeInfo]:
    prefix = _normalize_prefix(path_prefix)
    if prefix is None:
        return attrs
    remapped: dict[str, AttributeInfo] = {}
    for name, info in attrs.items():
        value = info.value
        time_samples = info.time_samples
        if info.semantic == Semantic.PATH_STRING:
            value = _remap_path_string_value(value, prefix, default_prim)
            if isinstance(time_samples, dict):
                time_samples = {
                    float(time): _remap_path_string_value(sample, prefix, default_prim)
                    for time, sample in time_samples.items()
                }
        remapped[name] = AttributeInfo(
            info.name,
            info.dtype,
            info.is_array,
            info.semantic,
            value,
            time_samples,
        )
    return remapped


def _remap_cloned_path_value(value: Any, source_root: str, target_root: str) -> Any:
    def remap_one(path: str) -> str:
        path = str(path)
        if path == source_root:
            return target_root
        if path.startswith(source_root + "/"):
            return target_root + path[len(source_root):]
        return path

    if isinstance(value, str):
        return remap_one(value)
    if isinstance(value, (list, tuple)):
        return [remap_one(str(item)) for item in value]
    arr = np.asarray(value)
    if arr.dtype.kind in "OUS":
        return [remap_one(str(item)) for item in arr.reshape(-1).tolist()]
    return value


def _clone_attribute_info(
    info: AttributeInfo,
    source_root: str,
    target_root: str,
) -> AttributeInfo:
    value = getattr(info, "value", None)
    if info.semantic == Semantic.PATH_STRING:
        value = _remap_cloned_path_value(value, source_root, target_root)
    elif isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value.copy())
    time_samples = getattr(info, "time_samples", None)
    if isinstance(time_samples, dict):
        time_samples = {
            float(time): (
                _remap_cloned_path_value(sample, source_root, target_root)
                if info.semantic == Semantic.PATH_STRING
                else np.ascontiguousarray(sample.copy()) if isinstance(sample, np.ndarray) else sample
            )
            for time, sample in time_samples.items()
        }
    return AttributeInfo(
        info.name,
        info.dtype,
        info.is_array,
        info.semantic,
        value,
        time_samples,
    )


def _reference_token(asset_path: str, prim_path: Optional[str] = None) -> str:
    target = f"<{prim_path}>" if prim_path else ""
    return f"@{_asset_ref(asset_path)}@{target}"


def _rel_paths_from_match(match: re.Match) -> list[str]:
    body = match.group(1) if match.group(1) is not None else match.group(2)
    if body is None:
        return []
    paths = re.findall(r"<([^>]+)>", body)
    if not paths and match.group(2):
        paths = [match.group(2)]
    return [p for p in paths if p]


def _scan_material_bindings(text: str) -> dict[str, list[str]]:
    stack: list[tuple[str, str]] = []
    bindings: dict[str, list[str]] = {}
    pending_push: Optional[tuple[str, str]] = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if stack:
            bm = _RE_MATERIAL_BINDING.search(raw)
            if bm:
                paths = _rel_paths_from_match(bm)
                if paths:
                    bindings[stack[-1][0]] = paths

        pm = _RE_PRIM.match(raw)
        if pm:
            typ = pm.group(1) or ""
            name = pm.group(2)
            parent = stack[-1][0] if stack else ""
            path = name if name.startswith("/") else f"{parent.rstrip('/')}/{name}" if parent else f"/{name}"
            open_count = raw.count("{")
            close_count = raw.count("}")
            if open_count > close_count:
                stack.append((path, typ))
                pending_push = None
            elif open_count == 0:
                pending_push = (path, typ)
            else:
                pending_push = None
        elif pending_push and "{" in raw:
            stack.append(pending_push)
            pending_push = None

        close_count = max(0, stripped.count("}") - stripped.count("{"))
        if close_count:
            for _ in range(min(close_count, len(stack))):
                stack.pop()
    return bindings


def _make_material_binding_overrides(
    source_text: str,
    source_root: str,
    target_root: str,
    indent: int,
) -> list[str]:
    source_root = (_normalize_prefix(source_root) or "/").rstrip("/") or "/"
    target_root = (_normalize_prefix(target_root) or "/").rstrip("/") or "/"

    def map_binding_target(target: str) -> str:
        if target == source_root:
            return target_root
        if source_root != "/" and target.startswith(source_root + "/"):
            return target_root + target[len(source_root):]
        if source_root == "/" and target.startswith("/"):
            return target_root + target
        return target

    tree: dict[str, Any] = {"targets": [], "children": {}}
    for source_path, targets in _scan_material_bindings(source_text).items():
        if source_path == source_root:
            parts: list[str] = []
        elif source_path.startswith(source_root + "/"):
            parts = [p for p in source_path[len(source_root) + 1 :].split("/") if p]
        else:
            continue
        node = tree
        for part in parts:
            node = node["children"].setdefault(part, {"targets": [], "children": {}})
        node["targets"] = [map_binding_target(target) for target in targets]

    def emit_node(node: dict[str, Any], depth: int) -> list[str]:
        out: list[str] = []
        pad = " " * depth
        targets = [target for target in node.get("targets", []) if target]
        if targets:
            target_text = f"<{targets[0]}>" if len(targets) == 1 else "[" + ", ".join(f"<{t}>" for t in targets) + "]"
            out.append(f"{pad}rel material:binding = {target_text}")
        for name, child in sorted(node.get("children", {}).items()):
            out.append(f'{pad}over "{name}"')
            out.append(f"{pad}{{")
            out.extend(emit_node(child, depth + 4))
            out.append(f"{pad}}}")
        return out

    return emit_node(tree, indent)


def _def_with_reference(
    indent: int,
    prim_type: str,
    name: str,
    ref: str,
    body_lines: Optional[list[str]] = None,
) -> list[str]:
    pad = " " * indent
    return [
        f'{pad}def {prim_type or "Xform"} "{name}" (',
        f"{pad}    prepend references = {ref}",
        f"{pad})",
        f"{pad}{{",
        *(body_lines or []),
        f"{pad}}}",
    ]


def _make_reference_wrapper(asset_path: str, path_prefix: str, source_text: str) -> str:
    prefix = _normalize_prefix(path_prefix)
    if prefix is None:
        raise ValueError("path_prefix must not be root for a reference wrapper")
    segments = [seg for seg in prefix.strip("/").split("/") if seg]
    if not segments:
        raise ValueError("path_prefix must contain at least one prim")

    default_prim = _default_prim_name(source_text)
    root_types = _root_prim_types(source_text)
    default_type = root_types.get(default_prim or "", "Xform")
    roots = root_types or ({default_prim: default_type} if default_prim else {})

    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{segments[0]}"',
        ")",
        "",
    ]
    for depth, segment in enumerate(segments):
        indent = depth * 4
        if depth == len(segments) - 1:
            if default_prim:
                overrides = _make_material_binding_overrides(
                    source_text,
                    f"/{default_prim}",
                    prefix,
                    indent + 4,
                )
                lines.extend(
                    _def_with_reference(
                        indent,
                        default_type,
                        segment,
                        _reference_token(asset_path),
                        body_lines=overrides,
                    )
                )
            else:
                lines.append(f'{" " * indent}def Xform "{segment}"')
                lines.append(f'{" " * indent}{{')
                for root_name, root_type in sorted(roots.items()):
                    lines.extend(
                        _def_with_reference(
                            indent + 4,
                            root_type,
                            root_name,
                            _reference_token(asset_path, f"/{root_name}"),
                            body_lines=_make_material_binding_overrides(
                                source_text,
                                f"/{root_name}",
                                f"{prefix}/{root_name}",
                                indent + 8,
                            ),
                        )
                    )
                lines.append(f'{" " * indent}}}')
        else:
            lines.append(f'{" " * indent}def Xform "{segment}"')
            lines.append(f'{" " * indent}{{')
    for depth in range(len(segments) - 2, -1, -1):
        lines.append(f'{" " * (depth * 4)}}}')
    return "\n".join(lines) + "\n"


def _make_prim_reference_wrapper(
    asset_path: str,
    source_prim_path: str,
    target_path: str,
    prim_type: str,
    source_text: str = "",
) -> str:
    target = _normalize_prefix(target_path)
    if target is None:
        raise ValueError("target_path must not be root for clone_usd")
    segments = [seg for seg in target.strip("/").split("/") if seg]
    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{segments[0]}"',
        ")",
        "",
    ]
    for depth, segment in enumerate(segments):
        indent = depth * 4
        if depth == len(segments) - 1:
            lines.extend(
                _def_with_reference(
                    indent,
                    prim_type or "Xform",
                    segment,
                    _reference_token(asset_path, source_prim_path),
                    body_lines=_make_material_binding_overrides(
                        source_text,
                        source_prim_path,
                        target,
                        indent + 4,
                    ),
                )
            )
        else:
            lines.append(f'{" " * indent}def Xform "{segment}"')
            lines.append(f'{" " * indent}{{')
    for depth in range(len(segments) - 2, -1, -1):
        lines.append(f'{" " * (depth * 4)}}}')
    return "\n".join(lines) + "\n"


def _render_product_path(parent: str, name: str) -> str:
    if name.startswith("/"):
        return name
    if parent:
        return f"{parent.rstrip('/')}/{name}"
    if "ViewportTexture" in name:
        return f"/Render/OmniverseKit/HydraTextures/{name}"
    return f"/Render/{name}"


def _render_var_name(path_or_name: str) -> str:
    text = path_or_name.strip().strip("<>").strip()
    if not text:
        return ""
    return text.rsplit("/", 1)[-1].split(".")[-1]


def _camera_rel_paths(match: re.Match) -> list[str]:
    body = match.group(1) if match.group(1) is not None else match.group(2)
    if body is None:
        return []
    paths = re.findall(r"<([^>]+)>", body)
    if not paths and match.group(2):
        paths = [match.group(2)]
    return [p for p in paths if p]


def _resolve_relationship_target(base_prim_path: str, target: str) -> str:
    target = target.strip().strip("<>").strip()
    if not target:
        return ""
    if target.startswith("/"):
        return target
    parts = [part for part in base_prim_path.split("/") if part]
    for part in target.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return "/" + "/".join(parts)


def _relationship_targets_from_text(base_prim_path: str, value_text: str) -> list[str]:
    body = _strip_usda_comment(value_text).strip().rstrip(",").strip()
    if not body:
        return []
    targets = re.findall(r"<([^>]+)>", body)
    if not targets:
        stripped = body.strip("[]").strip().strip("<>").strip()
        if stripped:
            targets = [part.strip().strip("<>").strip() for part in stripped.split(",")]
    return [
        resolved
        for resolved in (_resolve_relationship_target(base_prim_path, target) for target in targets)
        if resolved
    ]


def _merge_relationship_targets(existing: Iterable[str], targets: Iterable[str], list_op: Optional[str]) -> list[str]:
    current = [str(path) for path in existing if str(path)]
    incoming = [str(path) for path in targets if str(path)]
    op = (list_op or "explicit").lower()
    if op == "delete":
        removed = set(incoming)
        return [path for path in current if path not in removed]
    if op == "reorder":
        requested = [path for path in incoming if path in current]
        seen = set(requested)
        return requested + [path for path in current if path not in seen]
    if op == "prepend":
        seen = set(incoming)
        return incoming + [path for path in current if path not in seen]
    if op in ("append", "add"):
        out = list(current)
        seen = set(out)
        for path in incoming:
            if path not in seen:
                out.append(path)
                seen.add(path)
        return out
    return incoming


def _string_values_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        return [value.decode("utf-8", errors="replace")]
    if isinstance(value, str):
        return [value]
    arr = np.asarray(value)
    if arr.dtype.kind not in "OUS":
        return []
    return [str(item) for item in arr.reshape(-1).tolist()]


def _first_float_value(value: Any) -> Optional[float]:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size:
            return float(arr[0])
    except Exception:
        return None
    return None


def _first_bool_value(value: Any) -> Optional[bool]:
    try:
        arr = np.asarray(value, dtype=np.bool_).reshape(-1)
        if arr.size:
            return bool(arr[0])
    except Exception:
        return None
    return None


def _float4_tuple_value(value: Any) -> Optional[tuple[float, float, float, float]]:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= 4:
            return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    except Exception:
        return None
    return None


def _apply_render_settings_base_attribute(spec: Any, attribute_name: str, value: Any) -> bool:
    if attribute_name == "pixelAspectRatio":
        parsed = _first_float_value(value)
        if parsed is not None:
            spec.pixel_aspect_ratio = parsed
        return True
    if attribute_name == "aspectRatioConformPolicy":
        values = _string_values_from_value(value)
        if values:
            spec.aspect_ratio_conform_policy = values[0]
        return True
    if attribute_name == "dataWindowNDC":
        parsed = _float4_tuple_value(value)
        if parsed is not None:
            spec.data_window_ndc = parsed
        return True
    if attribute_name == "disableMotionBlur":
        parsed = _first_bool_value(value)
        if parsed is not None:
            spec.disable_motion_blur = parsed
        return True
    if attribute_name == "disableDepthOfField":
        parsed = _first_bool_value(value)
        if parsed is not None:
            spec.disable_depth_of_field = parsed
        return True
    if attribute_name == "instantaneousShutter":
        parsed = _first_bool_value(value)
        if parsed is not None:
            spec.instantaneous_shutter = parsed
        return True
    return False


def _scan_render_var_sources(text: str) -> dict[str, str]:
    sources: dict[str, str] = {}
    stack: list[tuple[str, str]] = []
    pending_push: Optional[tuple[str, str]] = None

    for raw in text.splitlines():
        stripped = raw.strip()
        current_var = next((path for path, typ in reversed(stack) if typ == "RenderVar"), None)
        if current_var:
            sm = _RE_SOURCE_NAME.search(raw)
            if sm:
                sources[current_var] = sm.group(1).strip()

        pm = _RE_PRIM.match(raw)
        if pm:
            typ = pm.group(1) or ""
            name = pm.group(2)
            parent = stack[-1][0] if stack else ""
            path = name if name.startswith("/") else f"{parent.rstrip('/')}/{name}" if parent else f"/{name}"
            if typ == "RenderVar":
                sources.setdefault(path, name)
                sm = _RE_SOURCE_NAME.search(raw)
                if sm:
                    sources[path] = sm.group(1).strip()
            open_count = raw.count("{")
            close_count = raw.count("}")
            if open_count > close_count:
                stack.append((path, typ))
                pending_push = None
            elif open_count == 0:
                pending_push = (path, typ)
            else:
                pending_push = None
        elif pending_push and "{" in raw:
            stack.append(pending_push)
            pending_push = None

        close_count = max(0, stripped.count("}") - stripped.count("{"))
        if close_count:
            for _ in range(min(close_count, len(stack))):
                stack.pop()

    return sources


def _scan_render_products(text: str) -> tuple[dict[str, _RenderProductSpec], dict[str, _RenderSettingsSpec]]:
    products: dict[str, _RenderProductSpec] = {}
    settings: dict[str, _RenderSettingsSpec] = {}
    render_var_sources = _scan_render_var_sources(text)
    stack: list[tuple[str, str]] = []
    pending_push: Optional[tuple[str, str]] = None
    pending_product_relationship: Optional[tuple[str, str, Optional[str], list[str], int]] = None
    pending_settings_relationship: Optional[tuple[str, str, Optional[str], list[str], int]] = None

    def _apply_product_relationship(product_path: str, attr_name: str, list_op: Optional[str], value_text: str) -> None:
        spec = products.setdefault(product_path, _RenderProductSpec(path=product_path, render_vars=[]))
        targets = _relationship_targets_from_text(product_path, value_text)
        if attr_name == "camera":
            existing = list(spec.camera_paths or ([spec.camera_path] if spec.camera_path else []))
            merged = _merge_relationship_targets(existing, targets, list_op)
            if merged:
                spec.camera_path = merged[0]
                spec.camera_paths = merged
            else:
                spec.camera_path = None
                spec.camera_paths = None
        elif attr_name == "orderedVars":
            merged = _merge_relationship_targets(spec.render_var_paths or [], targets, list_op)
            spec.render_var_paths = merged
            names = [render_var_sources.get(target, _render_var_name(target)) for target in merged]
            spec.render_vars = [name for name in names if name]

    def _apply_settings_relationship(settings_path: str, attr_name: str, list_op: Optional[str], value_text: str) -> None:
        spec = settings.setdefault(settings_path, _RenderSettingsSpec(path=settings_path, product_paths=[]))
        targets = _relationship_targets_from_text(settings_path, value_text)
        if attr_name == "camera":
            existing = list(spec.camera_paths or ([spec.camera_path] if spec.camera_path else []))
            merged = _merge_relationship_targets(existing, targets, list_op)
            if merged:
                spec.camera_path = merged[0]
                spec.camera_paths = merged
            else:
                spec.camera_path = None
                spec.camera_paths = None
        elif attr_name == "products":
            spec.product_paths = _merge_relationship_targets(spec.product_paths or [], targets, list_op)

    for raw in text.splitlines():
        stripped = raw.strip()
        if pending_product_relationship is not None:
            product_path, attr_name, list_op, value_lines, bracket_depth = pending_product_relationship
            value_lines.append(raw)
            bracket_depth += raw.count("[") - raw.count("]")
            if bracket_depth <= 0:
                _apply_product_relationship(product_path, attr_name, list_op, "\n".join(value_lines))
                pending_product_relationship = None
            else:
                pending_product_relationship = (product_path, attr_name, list_op, value_lines, bracket_depth)
            continue
        if pending_settings_relationship is not None:
            settings_path, attr_name, list_op, value_lines, bracket_depth = pending_settings_relationship
            value_lines.append(raw)
            bracket_depth += raw.count("[") - raw.count("]")
            if bracket_depth <= 0:
                _apply_settings_relationship(settings_path, attr_name, list_op, "\n".join(value_lines))
                pending_settings_relationship = None
            else:
                pending_settings_relationship = (settings_path, attr_name, list_op, value_lines, bracket_depth)
            continue
        current_product = next((path for path, typ in reversed(stack) if typ == "RenderProduct"), None)
        if current_product:
            spec = products[current_product]
            rm = _RE_RESOLUTION.search(raw)
            if rm:
                spec.width = int(rm.group(1))
                spec.height = int(rm.group(2))
            rel = _RE_REL_ATTR.match(raw)
            if rel and rel.group(2) in {"camera", "orderedVars"}:
                list_op = rel.group(1)
                attr_name = rel.group(2)
                value_text = rel.group(3)
                bracket_depth = value_text.count("[") - value_text.count("]")
                if bracket_depth > 0:
                    pending_product_relationship = (current_product, attr_name, list_op, [value_text], bracket_depth)
                else:
                    _apply_product_relationship(current_product, attr_name, list_op, value_text)
            mm = _RE_RENDER_MODE.search(raw)
            if mm:
                spec.render_mode = mm.group(2).strip().lower()
            minimal = _RE_MINIMAL_MODE.search(raw)
            if minimal:
                spec.minimal_mode = int(minimal.group(1))
            am = _RE_ATTR.match(raw)
            if am:
                attr_name = am.group(2)
                value = _parse_usda_value(am.group(1), am.group(3))
                if _apply_render_settings_base_attribute(spec, attr_name, value):
                    continue
                if attr_name in {"productName", "productType"}:
                    values = _string_values_from_value(value)
                    if values:
                        if attr_name == "productName":
                            spec.product_name = values[0]
                        else:
                            spec.product_type = values[0]

        current_settings = next((path for path, typ in reversed(stack) if typ == "RenderSettings"), None)
        if current_settings and not current_product:
            spec = settings[current_settings]
            rm = _RE_RESOLUTION.search(raw)
            if rm:
                spec.width = int(rm.group(1))
                spec.height = int(rm.group(2))
            rel = _RE_REL_ATTR.match(raw)
            if rel and rel.group(2) in {"camera", "products"}:
                list_op = rel.group(1)
                attr_name = rel.group(2)
                value_text = rel.group(3)
                bracket_depth = value_text.count("[") - value_text.count("]")
                if bracket_depth > 0:
                    pending_settings_relationship = (current_settings, attr_name, list_op, [value_text], bracket_depth)
                else:
                    _apply_settings_relationship(current_settings, attr_name, list_op, value_text)
            mm = _RE_RENDER_MODE.search(raw)
            if mm:
                spec.render_mode = mm.group(2).strip().lower()
            minimal = _RE_MINIMAL_MODE.search(raw)
            if minimal:
                spec.minimal_mode = int(minimal.group(1))
            am = _RE_ATTR.match(raw)
            if am:
                attr_name = am.group(2)
                value = _parse_usda_value(am.group(1), am.group(3))
                if _apply_render_settings_base_attribute(spec, attr_name, value):
                    continue
                values = _string_values_from_value(value)
                if attr_name == "includedPurposes":
                    spec.included_purposes = values
                elif attr_name == "materialBindingPurposes":
                    spec.material_binding_purposes = values
                elif attr_name == "renderingColorSpace" and values:
                    spec.rendering_color_space = values[0]

        pm = _RE_PRIM.match(raw)
        if pm:
            typ = pm.group(1) or ""
            name = pm.group(2)
            parent = stack[-1][0] if stack else ""
            path = _render_product_path(parent, name) if typ == "RenderProduct" else (
                name if name.startswith("/") else f"{parent.rstrip('/')}/{name}" if parent else f"/{name}"
            )
            if typ == "RenderProduct":
                products.setdefault(path, _RenderProductSpec(path=path, render_vars=[]))
            elif typ == "RenderSettings":
                settings.setdefault(path, _RenderSettingsSpec(path=path, product_paths=[]))
            if raw.count("{") > raw.count("}"):
                stack.append((path, typ))
                pending_push = None
            else:
                pending_push = (path, typ)
        elif pending_push and "{" in raw:
            stack.append(pending_push)
            pending_push = None

        close_count = stripped.count("}")
        if close_count:
            for _ in range(min(close_count, len(stack))):
                stack.pop()

    for settings_spec in settings.values():
        for product_path in settings_spec.product_paths or []:
            product = products.setdefault(product_path, _RenderProductSpec(path=product_path, render_vars=[]))
            if product.camera_path is None and settings_spec.camera_path is not None:
                product.camera_path = settings_spec.camera_path
                product.camera_paths = list(settings_spec.camera_paths or [settings_spec.camera_path])
            if product.width is None and settings_spec.width is not None:
                product.width = settings_spec.width
            if product.height is None and settings_spec.height is not None:
                product.height = settings_spec.height
            if product.render_mode is None and settings_spec.render_mode is not None:
                product.render_mode = settings_spec.render_mode
            if product.minimal_mode is None and settings_spec.minimal_mode is not None:
                product.minimal_mode = settings_spec.minimal_mode
            if product.pixel_aspect_ratio is None and settings_spec.pixel_aspect_ratio is not None:
                product.pixel_aspect_ratio = settings_spec.pixel_aspect_ratio
            if (
                product.aspect_ratio_conform_policy is None
                and settings_spec.aspect_ratio_conform_policy is not None
            ):
                product.aspect_ratio_conform_policy = settings_spec.aspect_ratio_conform_policy
            if product.data_window_ndc is None and settings_spec.data_window_ndc is not None:
                product.data_window_ndc = settings_spec.data_window_ndc
            if product.disable_motion_blur is None and settings_spec.disable_motion_blur is not None:
                product.disable_motion_blur = settings_spec.disable_motion_blur
            if product.disable_depth_of_field is None and settings_spec.disable_depth_of_field is not None:
                product.disable_depth_of_field = settings_spec.disable_depth_of_field
            if product.instantaneous_shutter is None and settings_spec.instantaneous_shutter is not None:
                product.instantaneous_shutter = settings_spec.instantaneous_shutter
            if settings_spec.included_purposes is not None:
                product.included_purposes = list(settings_spec.included_purposes)
            if settings_spec.material_binding_purposes is not None:
                product.material_binding_purposes = list(settings_spec.material_binding_purposes)
            if settings_spec.rendering_color_space is not None:
                product.rendering_color_space = settings_spec.rendering_color_space

    return products, settings


def _scan_sublayers(text: str, base_dir: Optional[Path] = None) -> list[str]:
    out: list[str] = []
    for block in _RE_SUBLAYERS.findall(text):
        for ref in _RE_ASSET_REF.findall(block):
            if "://" in ref:
                continue
            path = Path(ref)
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            out.append(str(path))
    for ref in _RE_AUTHORING_LAYER.findall(text):
        if "://" in ref:
            continue
        path = Path(ref)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        out.append(str(path))
    return out


def _strip_usda_comment(text: str) -> str:
    in_quote = False
    escaped = False
    for i, ch in enumerate(text):
        if ch == "\\" and in_quote:
            escaped = not escaped
            continue
        if ch == '"' and not escaped:
            in_quote = not in_quote
        elif ch == "#" and not in_quote:
            return text[:i].strip()
        escaped = False
    return text.strip()


def _usda_vector_lanes(type_name: str) -> Optional[int]:
    base = type_name.replace("[]", "").lower()
    match = re.search(r"([234])(?:[fdh]?)$", base)
    if match:
        return int(match.group(1))
    if base.startswith(("color3", "point3", "normal3", "vector3")):
        return 3
    if base.startswith("texcoord2"):
        return 2
    return None


def _usda_numpy_dtype(type_name: str):
    base = type_name.replace("[]", "").lower()
    if base in ("double", "float64", "matrix4d") or base.endswith("d"):
        return np.float64
    if base.startswith("half") or base.endswith("h"):
        return np.float16
    if base.startswith("uint"):
        return np.uint32
    if base.startswith("int"):
        return np.int32
    if base.startswith("bool"):
        return np.bool_
    if base.startswith(("float", "color", "point", "normal", "vector", "texcoord")):
        return np.float32
    return None


def _usda_dl_dtype(type_name: str) -> tuple[Any, Semantic]:
    base = type_name.replace("[]", "").lower()
    lanes = 1
    semantic = Semantic.NONE
    if base == "matrix4d":
        return DLDataType("float", 64, 16), Semantic.XFORM_MAT4x4
    if base in ("string", "token"):
        return DLDataType("uint", 128, 1), Semantic.TOKEN_STRING
    if base in ("asset", "path"):
        return DLDataType("uint", 128, 1), Semantic.PATH_STRING
    vector_lanes = _usda_vector_lanes(type_name)
    if vector_lanes:
        lanes = vector_lanes
    if base.startswith("uint"):
        return DLDataType("uint", 32, lanes), semantic
    if base.startswith("int"):
        return DLDataType("int", 32, lanes), semantic
    if base.startswith("bool"):
        return DLDataType("bool", 8, lanes), semantic
    if base in ("double", "float64") or base.endswith("d"):
        return DLDataType("float", 64, lanes), semantic
    if base.startswith("half") or base.endswith("h"):
        return DLDataType("float", 16, lanes), semantic
    if base.startswith(("float", "color", "point", "normal", "vector", "texcoord")):
        return DLDataType("float", 32, lanes), semantic
    return type_name, semantic


def _array_dl_dtype(value: Any, semantic: Semantic = Semantic.NONE, is_array: bool = False) -> tuple[Any, Semantic]:
    if semantic == Semantic.XFORM_MAT4x4:
        return DLDataType("float", 64, 16), semantic
    if semantic in (Semantic.TOKEN_STRING, Semantic.PATH_STRING):
        return DLDataType("uint", 128, 1), semantic
    arr = np.asarray(value)
    if arr.dtype.kind in "OUS":
        return DLDataType("uint", 128, 1), Semantic.TOKEN_STRING
    dtype = np.dtype(arr.dtype)
    if dtype.kind == "f":
        code = "float"
    elif dtype.kind == "i":
        code = "int"
    elif dtype.kind == "u":
        code = "uint"
    elif dtype.kind == "b":
        code = "bool"
    else:
        return dtype, semantic
    bits = 8 if dtype.kind == "b" else dtype.itemsize * 8
    lanes = 1
    if arr.ndim:
        if is_array:
            if arr.ndim > 1 and arr.shape[-1] in (2, 3, 4):
                lanes = int(arr.shape[-1])
        else:
            lanes = int(_math.prod(arr.shape))
    return DLDataType(code, bits, max(lanes, 1)), semantic


def _parse_usda_value(type_name: str, value_text: str) -> Any:
    value_text = _strip_usda_comment(value_text).rstrip(",")
    base = type_name.replace("[]", "").lower()
    if not value_text:
        return None
    if base in ("string", "token", "asset", "path"):
        strings = re.findall(r'"([^"]*)"', value_text)
        if base == "path":
            path_targets = re.findall(r"<([^>]+)>", value_text)
            if path_targets:
                strings = path_targets
        if not strings:
            body = value_text.strip()
            if type_name.endswith("[]"):
                body = body.strip("[]").strip()
                if body:
                    strings = [
                        item.strip().strip("<>").strip('"')
                        for item in body.split(",")
                        if item.strip()
                    ]
            else:
                body = body.strip("<>").strip('"')
                if body:
                    strings = [body]
        if type_name.endswith("[]"):
            return strings
        return strings[0] if strings else value_text.strip()
    if base == "bool":
        tokens = re.findall(r"\b(?:true|false|0|1)\b", value_text, flags=re.IGNORECASE)
        values = [token.lower() in ("true", "1") for token in tokens]
        if type_name.endswith("[]"):
            return np.asarray(values, dtype=np.bool_)
        return np.asarray(values[0] if values else False, dtype=np.bool_)

    dtype = _usda_numpy_dtype(type_name)
    if dtype is None:
        return None
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value_text)
    if not numbers:
        return None
    arr = np.asarray(numbers, dtype=dtype)
    if base == "matrix4d" and arr.size == 16:
        arr = arr.reshape(4, 4)
    else:
        lanes = _usda_vector_lanes(type_name)
        if lanes and arr.size % lanes == 0:
            arr = arr.reshape((-1, lanes)) if type_name.endswith("[]") or arr.size > lanes else arr.reshape(lanes)
        elif arr.size == 1 and not type_name.endswith("[]"):
            arr = arr.reshape(())
    return np.ascontiguousarray(arr)


def _parse_usda_time_samples(type_name: str, sample_text: str) -> dict[float, Any]:
    start = sample_text.find("{")
    end = sample_text.rfind("}")
    if start < 0 or end <= start:
        return {}
    content = sample_text[start + 1:end]
    matches = list(_RE_TIME_SAMPLE_KEY.finditer(content))
    out: dict[float, Any] = {}
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        value_text = content[match.end():next_start].strip().rstrip(",")
        value = _parse_usda_value(type_name, value_text)
        if value is not None:
            out[float(match.group(1))] = value
    return out


def _sample_time_value(samples: dict[float, Any], usd_time: float) -> Any:
    if not samples:
        return None
    keys = sorted(samples)
    if usd_time <= keys[0]:
        return samples[keys[0]]
    if usd_time >= keys[-1]:
        return samples[keys[-1]]
    if usd_time in samples:
        return samples[usd_time]
    lower = keys[0]
    upper = keys[-1]
    for i in range(1, len(keys)):
        if keys[i] >= usd_time:
            lower = keys[i - 1]
            upper = keys[i]
            break
    low_value = samples[lower]
    high_value = samples[upper]
    low_arr = np.asarray(low_value)
    high_arr = np.asarray(high_value)
    if low_arr.shape == high_arr.shape and low_arr.dtype.kind == "f" and high_arr.dtype.kind == "f":
        alpha = (float(usd_time) - lower) / max(upper - lower, 1e-12)
        return np.ascontiguousarray(low_arr + (high_arr - low_arr) * alpha)
    return low_value


def _dlpackable_attr_value(value: Any, fallback: Any) -> np.ndarray:
    if value is None:
        return np.asarray(fallback)
    arr = np.asarray(value)
    if arr.dtype.kind in "OUS":
        encoded = str(arr.reshape(-1)[0] if arr.size else "").encode("utf-8")
        return np.frombuffer(encoded, dtype=np.uint8).copy()
    return np.ascontiguousarray(arr)


def _stable_prim_path_id(path: str) -> int:
    digest = hashlib.blake2b(str(path).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return value if value != 0 else 1


def _scan_prim_index(text: str) -> dict[str, dict[str, AttributeInfo]]:
    stack: list[tuple[str, str]] = []
    out: dict[str, dict[str, AttributeInfo]] = {}
    pending_path: Optional[str] = None
    pending_push: Optional[tuple[str, str]] = None
    pending_time_samples: Optional[tuple[str, str, str, list[str], int]] = None
    pending_relationship: Optional[tuple[str, str, Optional[str], list[str], int]] = None

    def _apply_relationship(prim_path: str, attr_name: str, list_op: Optional[str], value_text: str) -> None:
        attrs = out.setdefault(prim_path, {})
        existing = _string_values_from_value(getattr(attrs.get(attr_name), "value", None))
        targets = _relationship_targets_from_text(prim_path, value_text)
        attrs[attr_name] = AttributeInfo(
            attr_name,
            DLDataType("uint", 128, 1),
            True,
            Semantic.PATH_STRING,
            _merge_relationship_targets(existing, targets, list_op),
        )

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if pending_relationship is not None:
            prim_path, attr_name, list_op, value_lines, bracket_depth = pending_relationship
            value_lines.append(raw)
            bracket_depth += raw.count("[") - raw.count("]")
            if bracket_depth <= 0:
                _apply_relationship(prim_path, attr_name, list_op, "\n".join(value_lines))
                pending_relationship = None
            else:
                pending_relationship = (prim_path, attr_name, list_op, value_lines, bracket_depth)
            continue
        if pending_time_samples is not None:
            prim_path, type_name, attr_name, sample_lines, brace_depth = pending_time_samples
            sample_lines.append(raw)
            brace_depth += raw.count("{") - raw.count("}")
            if brace_depth <= 0:
                samples = _parse_usda_time_samples(type_name, "\n".join(sample_lines))
                dtype, semantic = _usda_dl_dtype(type_name)
                attrs = out.setdefault(prim_path, {})
                existing = attrs.get(attr_name)
                default_value = getattr(existing, "value", None) if existing is not None else None
                if default_value is None and samples:
                    default_value = _sample_time_value(samples, 0.0)
                attrs[attr_name] = AttributeInfo(
                    attr_name,
                    dtype,
                    type_name.endswith("[]"),
                    semantic,
                    default_value,
                    samples,
                )
                pending_time_samples = None
            else:
                pending_time_samples = (prim_path, type_name, attr_name, sample_lines, brace_depth)
            continue
        if line.startswith("{") and pending_push is not None:
            stack.append(pending_push)
            pending_path = pending_push[0]
            pending_push = None
            continue
        close_count = max(0, line.count("}") - line.count("{"))
        if close_count and not line.startswith(("def ", "over ")):
            for _ in range(min(close_count, len(stack))):
                stack.pop()
            pending_path = stack[-1][0] if stack else None
            pending_push = None
            continue
        m = _RE_PRIM.match(raw)
        if m:
            typ = m.group(1) or ""
            name = m.group(2)
            parent = stack[-1][0] if stack else ""
            path = f"{parent}/{name}" if parent else f"/{name}"
            out.setdefault(path, {})
            if typ:
                out[path]["__type__"] = AttributeInfo("__type__", str, False, Semantic.TOKEN_STRING)
                out[path]["__type__"].value = typ  # type: ignore[attr-defined]
            open_count = line.count("{")
            close_count = line.count("}")
            if open_count > close_count:
                stack.append((path, typ))
                pending_path = path
                pending_push = None
            elif open_count == 0:
                pending_push = (path, typ)
                pending_path = path
            else:
                pending_path = path
                pending_push = None
            continue
        tm = _RE_ATTR_TIME_SAMPLES.match(raw)
        if tm and pending_path:
            type_name = tm.group(1)
            attr_name = tm.group(2)
            sample_text = tm.group(3)
            brace_depth = sample_text.count("{") - sample_text.count("}")
            if brace_depth <= 0 and "}" in sample_text:
                samples = _parse_usda_time_samples(type_name, sample_text)
                dtype, semantic = _usda_dl_dtype(type_name)
                attrs = out.setdefault(pending_path, {})
                existing = attrs.get(attr_name)
                default_value = getattr(existing, "value", None) if existing is not None else None
                if default_value is None and samples:
                    default_value = _sample_time_value(samples, 0.0)
                attrs[attr_name] = AttributeInfo(
                    attr_name,
                    dtype,
                    type_name.endswith("[]"),
                    semantic,
                    default_value,
                    samples,
                )
            else:
                pending_time_samples = (pending_path, type_name, attr_name, [sample_text], brace_depth)
            continue
        rm = _RE_REL_ATTR.match(raw)
        if rm and pending_path:
            list_op = rm.group(1)
            attr_name = rm.group(2)
            value_text = rm.group(3)
            bracket_depth = value_text.count("[") - value_text.count("]")
            if bracket_depth > 0:
                pending_relationship = (pending_path, attr_name, list_op, [value_text], bracket_depth)
            else:
                _apply_relationship(pending_path, attr_name, list_op, value_text)
            continue
        am = _RE_ATTR.match(raw)
        if am and pending_path:
            type_name = am.group(1)
            attr_name = am.group(2)
            dtype, semantic = _usda_dl_dtype(type_name)
            existing = out.setdefault(pending_path, {}).get(attr_name)
            out.setdefault(pending_path, {})[attr_name] = AttributeInfo(
                attr_name,
                dtype,
                type_name.endswith("[]"),
                semantic,
                _parse_usda_value(type_name, am.group(3)),
                getattr(existing, "time_samples", None) if existing is not None else None,
            )
        if close_count:
            for _ in range(min(close_count, len(stack))):
                stack.pop()
            pending_path = stack[-1][0] if stack else None
    return out


_NATIVE_INLINE_SCENE_PRIM_TYPES = frozenset(
    {
        "DomeLight",
        "DistantLight",
        "RectLight",
        "DiskLight",
        "SphereLight",
        "CylinderLight",
        "GeometryLight",
        "PortalLight",
    }
)


def _inline_layer_has_native_scene_prims(text: str) -> bool:
    try:
        for attrs in _scan_prim_index(text).values():
            prim_type = str(getattr(attrs.get("__type__"), "value", "") or "")
            if prim_type in _NATIVE_INLINE_SCENE_PRIM_TYPES:
                return True
    except Exception:
        pass
    return False


def _runtime_inline_layer_text(text: str, base_dir: Path) -> str:
    def replace_asset_ref(match: re.Match[str]) -> str:
        ref = match.group(1)
        if "://" in ref:
            return match.group(0)
        path = Path(ref)
        if not path.is_absolute():
            path = base_dir / path
        return f"@{str(path)}@"

    def replace_sublayers(match: re.Match[str]) -> str:
        block = match.group(0)
        return _RE_ASSET_REF.sub(replace_asset_ref, block)

    return _RE_SUBLAYERS.sub(replace_sublayers, text)


def _colocated_inline_layer_text(text: str, base_dir: Path) -> str:
    base_dir = Path(base_dir)

    def replace_asset_ref(match: re.Match[str]) -> str:
        ref = match.group(1)
        if "://" in ref:
            return match.group(0)
        path = Path(ref)
        if not path.is_absolute():
            path = base_dir / path
        try:
            rel = os.path.relpath(path, base_dir)
        except Exception:
            rel = str(path)
        rel = rel.replace("\\", "/")
        if not rel.startswith("."):
            rel = f"./{rel}"
        return f"@{rel}@"

    def replace_sublayers(match: re.Match[str]) -> str:
        block = match.group(0)
        return _RE_ASSET_REF.sub(replace_asset_ref, block)

    return _RE_SUBLAYERS.sub(replace_sublayers, text)


def _matrix_to_camera(matrix: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    eye = m[3, :3]
    forward = -m[2, :3]
    n = np.linalg.norm(forward)
    if n <= 1e-12:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        forward = forward / n
    up = m[1, :3]
    un = np.linalg.norm(up)
    if un <= 1e-12:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        up = up / un
    target = eye + forward
    return tuple(eye.tolist()), tuple(target.tolist()), tuple(up.tolist())


def _normalize3(value: Any, fallback: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n <= 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return arr / n


def _default_camera_from_bounds(
    bounds: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    if bounds is None:
        return (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    try:
        bmin = np.asarray(bounds[0], dtype=np.float32)
        bmax = np.asarray(bounds[1], dtype=np.float32)
        if bmin.shape != (3,) or bmax.shape != (3,):
            raise ValueError
        if not np.all(np.isfinite(bmin)) or not np.all(np.isfinite(bmax)):
            raise ValueError
        extent = np.maximum(bmax - bmin, 0.0)
        center = (bmin + bmax) * 0.5
        radius = max(float(np.linalg.norm(extent) * 0.5), 1.0)
        eye = center + np.array([0.0, 0.0, radius * 3.0], dtype=np.float32)
        return tuple(eye.tolist()), tuple(center.tolist()), (0.0, 1.0, 0.0)
    except Exception:
        return (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)


def _make_view_matrix(
    eye: Any,
    target: Any,
    up: Any,
) -> np.ndarray:
    eye_v = np.asarray(eye, dtype=np.float32).reshape(3)
    target_v = np.asarray(target, dtype=np.float32).reshape(3)
    forward = _normalize3(target_v - eye_v, (0.0, 0.0, -1.0))
    up_v = _normalize3(up, (0.0, 1.0, 0.0))
    right = np.cross(forward, up_v)
    if float(np.linalg.norm(right)) <= 1e-6:
        up_v = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        right = np.cross(forward, up_v)
    right = _normalize3(right, (1.0, 0.0, 0.0))
    camera_up = _normalize3(np.cross(right, forward), (0.0, 1.0, 0.0))

    view = np.zeros(16, dtype=np.float32)
    view[0] = right[0]
    view[1] = right[1]
    view[2] = right[2]
    view[3] = -float(np.dot(right, eye_v))
    view[4] = camera_up[0]
    view[5] = camera_up[1]
    view[6] = camera_up[2]
    view[7] = -float(np.dot(camera_up, eye_v))
    view[8] = -forward[0]
    view[9] = -forward[1]
    view[10] = -forward[2]
    view[11] = float(np.dot(forward, eye_v))
    view[15] = 1.0
    return view


def _invert_view(view: np.ndarray) -> np.ndarray:
    vi = np.zeros(16, dtype=np.float32)
    vi[0] = view[0]
    vi[1] = view[4]
    vi[2] = view[8]
    vi[4] = view[1]
    vi[5] = view[5]
    vi[6] = view[9]
    vi[8] = view[2]
    vi[9] = view[6]
    vi[10] = view[10]
    vi[3] = -(vi[0] * view[3] + vi[1] * view[7] + vi[2] * view[11])
    vi[7] = -(vi[4] * view[3] + vi[5] * view[7] + vi[6] * view[11])
    vi[11] = -(vi[8] * view[3] + vi[9] * view[7] + vi[10] * view[11])
    vi[15] = 1.0
    return vi


def _make_proj_matrix(
    fov_degrees: float,
    aspect: float,
    near: float,
    far: float,
    projection_shift: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    fov = _math.radians(float(fov_degrees))
    t = _math.tan(fov * 0.5)
    if not np.isfinite(t) or t <= 1e-8:
        t = _math.tan(_math.radians(60.0) * 0.5)
    aspect = max(float(aspect), 1e-6)
    near = max(float(near), 1e-6)
    far = max(float(far), near + 1.0)
    proj = np.zeros(16, dtype=np.float32)
    proj[0] = 1.0 / (aspect * t)
    proj[5] = -1.0 / t
    proj[2] = float(projection_shift[0])
    proj[6] = float(projection_shift[1])
    proj[10] = far / (near - far)
    proj[11] = -(far * near) / (far - near)
    proj[14] = -1.0
    return proj


def _invert_proj(proj: np.ndarray) -> np.ndarray:
    pi = np.zeros(16, dtype=np.float32)
    pi[0] = 1.0 / proj[0]
    pi[5] = 1.0 / proj[5]
    pi[3] = proj[2] / proj[0]
    pi[7] = proj[6] / proj[5]
    pi[11] = 1.0 / proj[14]
    pi[14] = -1.0
    pi[15] = proj[10] / proj[14]
    return pi


def _make_vp_inv(
    eye: Any,
    target: Any,
    up: Any,
    fov_degrees: float,
    width: int,
    height: int,
    near: float,
    far: float,
    projection_shift: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    view = _make_view_matrix(eye, target, up)
    proj = _make_proj_matrix(
        fov_degrees,
        max(float(width), 1.0) / max(float(height), 1.0),
        near,
        far,
        projection_shift,
    )
    return np.concatenate([_invert_view(view), _invert_proj(proj)]).astype(np.float32, copy=False)


def _positions_from_depth(
    depth: np.ndarray,
    camera: Optional[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    fov_degrees: float,
    projection_shift: tuple[float, float] = (0.0, 0.0),
) -> Optional[np.ndarray]:
    if camera is None:
        return None
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        return None
    h, w = depth.shape
    eye, target, up = camera
    eye_v = np.asarray(eye, dtype=np.float32).reshape(3)
    target_v = np.asarray(target, dtype=np.float32).reshape(3)
    forward = _normalize3(target_v - eye_v, (0.0, 0.0, -1.0))
    up_v = _normalize3(up, (0.0, 1.0, 0.0))
    right = np.cross(forward, up_v)
    if float(np.linalg.norm(right)) <= 1e-6:
        up_v = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        right = np.cross(forward, up_v)
    right = _normalize3(right, (1.0, 0.0, 0.0))
    camera_up = _normalize3(np.cross(right, forward), (0.0, 1.0, 0.0))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    aspect = max(float(w), 1.0) / max(float(h), 1.0)
    tan_half_fov = _math.tan(_math.radians(float(fov_degrees)) * 0.5)
    x_ndc = ((xx + 0.5) / max(float(w), 1.0)) * 2.0 - 1.0
    y_ndc = 1.0 - ((yy + 0.5) / max(float(h), 1.0)) * 2.0
    x = (x_ndc + float(projection_shift[0])) * aspect * tan_half_fov
    y = (y_ndc + float(projection_shift[1])) * tan_half_fov
    rays = forward.reshape(1, 1, 3) + right.reshape(1, 1, 3) * x[..., None] + camera_up.reshape(1, 1, 3) * y[..., None]
    rays /= np.maximum(np.linalg.norm(rays, axis=2, keepdims=True), 1e-8)

    mask = np.isfinite(depth) & (depth > 0.0)
    pos = np.zeros((h, w, 4), dtype=np.float32)
    pos[..., :3] = eye_v.reshape(1, 1, 3) + rays * np.where(mask, depth, 0.0)[..., None]
    pos[~mask, :3] = 0.0
    pos[..., 3] = mask.astype(np.float32)
    return pos


def _fallback_camera_positions(x_norm: np.ndarray, y_norm: np.ndarray, distance: np.ndarray) -> np.ndarray:
    h, w = distance.shape
    pos = np.zeros((h, w, 4), dtype=np.float32)
    pos[..., 0] = x_norm * distance
    pos[..., 1] = y_norm * distance
    pos[..., 2] = -distance
    pos[..., 3] = np.isfinite(distance).astype(np.float32)
    return np.ascontiguousarray(pos)


def _image_plane_distance_from_positions(
    positions: np.ndarray,
    camera: Optional[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    fallback: np.ndarray,
) -> np.ndarray:
    fallback = np.asarray(fallback, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    if camera is None or positions.ndim != 3 or positions.shape[:2] != fallback.shape or positions.shape[2] < 3:
        return np.ascontiguousarray(fallback, dtype=np.float32)
    eye, target, _up = camera
    eye_v = np.asarray(eye, dtype=np.float32).reshape(3)
    target_v = np.asarray(target, dtype=np.float32).reshape(3)
    forward = _normalize3(target_v - eye_v, (0.0, 0.0, -1.0))
    plane = np.tensordot(positions[..., :3] - eye_v.reshape(1, 1, 3), forward, axes=([-1], [0]))
    valid = np.isfinite(plane)
    if positions.shape[2] >= 4:
        valid &= positions[..., 3] > 0.0
    valid &= fallback > 0.0
    return np.ascontiguousarray(np.where(valid, np.maximum(plane, 0.0), 0.0).astype(np.float32, copy=False))


def _normals_from_positions(pos: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 3 or pos.shape[2] < 3:
        return None
    xyz = pos[..., :3]
    dx = np.empty_like(xyz)
    dy = np.empty_like(xyz)
    dx[:, 1:-1] = xyz[:, 2:] - xyz[:, :-2]
    dx[:, 0] = xyz[:, 1] - xyz[:, 0] if xyz.shape[1] > 1 else 0.0
    dx[:, -1] = xyz[:, -1] - xyz[:, -2] if xyz.shape[1] > 1 else 0.0
    dy[1:-1, :] = xyz[2:, :] - xyz[:-2, :]
    dy[0, :] = xyz[1, :] - xyz[0, :] if xyz.shape[0] > 1 else 0.0
    dy[-1, :] = xyz[-1, :] - xyz[-2, :] if xyz.shape[0] > 1 else 0.0
    normal3 = np.cross(dx, dy).astype(np.float32, copy=False)
    length = np.linalg.norm(normal3, axis=2)
    finite = np.isfinite(length) & (length > 1e-8)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    if not np.any(finite):
        return None
    normal3 = np.divide(normal3, np.where(length > 1e-8, length, 1.0)[..., None])
    normal3 = np.nan_to_num(normal3, copy=False)
    normal3[~finite] = 0.0
    normals = np.zeros(pos.shape[:2] + (4,), dtype=np.float32)
    normals[..., :3] = normal3
    normals[..., 3] = finite.astype(np.float32)
    return np.ascontiguousarray(normals)


def _assemble_camera_tiles(tiles: np.ndarray, width: int, height: int) -> np.ndarray:
    tiles = np.asarray(tiles)
    if tiles.ndim not in (3, 4) or tiles.shape[0] <= 0:
        return np.asarray(tiles)
    num_cameras = int(tiles.shape[0])
    tile_h = int(tiles.shape[1])
    tile_w = int(tiles.shape[2])
    num_cols = int(_math.ceil(_math.sqrt(num_cameras)))
    num_rows = int(_math.ceil(num_cameras / max(num_cols, 1)))
    grid_shape = (num_rows * tile_h, num_cols * tile_w) + tuple(tiles.shape[3:])
    grid = np.zeros(grid_shape, dtype=tiles.dtype)
    for cam in range(num_cameras):
        row = cam // num_cols
        col = cam % num_cols
        grid[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w, ...] = tiles[cam]
    return np.ascontiguousarray(grid[:height, :width, ...])


def _camera_tile_layout(num_cameras: int, width: int, height: int) -> tuple[int, int, int, int]:
    num_cols = int(_math.ceil(_math.sqrt(max(num_cameras, 1))))
    num_rows = int(_math.ceil(num_cameras / max(num_cols, 1)))
    tile_w = max(1, int(_math.ceil(width / max(num_cols, 1))))
    tile_h = max(1, int(_math.ceil(height / max(num_rows, 1))))
    return num_cols, num_rows, tile_w, tile_h


def _fit_camera_tile(tile: np.ndarray, width: int, height: int) -> np.ndarray:
    tile = np.asarray(tile)
    if tile.ndim < 2:
        return np.asarray(tile)
    if tile.shape[0] == height and tile.shape[1] == width:
        return np.ascontiguousarray(tile)
    fitted = np.zeros((height, width) + tuple(tile.shape[2:]), dtype=tile.dtype)
    copy_h = min(height, int(tile.shape[0]))
    copy_w = min(width, int(tile.shape[1]))
    if copy_h > 0 and copy_w > 0:
        fitted[:copy_h, :copy_w, ...] = tile[:copy_h, :copy_w, ...]
    return np.ascontiguousarray(fitted)


def _data_window_mask(
    data_window_ndc: Optional[tuple[float, float, float, float]],
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    if data_window_ndc is None or width <= 0 or height <= 0:
        return None
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in data_window_ndc)
    except Exception:
        return None
    if not all(np.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
        return None
    if xmin <= 0.0 and ymin <= 0.0 and xmax >= 1.0 and ymax >= 1.0:
        return None
    x_centers = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    # USD dataWindowNDC uses bottom-left origin; image arrays are top-left first.
    y_centers = 1.0 - ((np.arange(height, dtype=np.float32) + 0.5) / float(height))
    x_mask = (x_centers >= xmin) & (x_centers <= xmax)
    y_mask = (y_centers >= ymin) & (y_centers <= ymax)
    return np.ascontiguousarray(y_mask[:, None] & x_mask[None, :])


def _apply_data_window_ndc(
    arrays: dict[str, np.ndarray],
    data_window_ndc: Optional[tuple[float, float, float, float]],
) -> dict[str, np.ndarray]:
    if data_window_ndc is None or not arrays:
        return arrays
    first = next(iter(arrays.values()))
    if getattr(first, "ndim", 0) < 2:
        return arrays
    height, width = int(first.shape[0]), int(first.shape[1])
    mask = _data_window_mask(data_window_ndc, width, height)
    if mask is None:
        return arrays
    out: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.shape[:2] == (height, width):
            shaped_mask = mask if arr.ndim == 2 else mask.reshape((height, width) + (1,) * (arr.ndim - 2))
            out[name] = np.ascontiguousarray(np.where(shaped_mask, arr, np.zeros((), dtype=arr.dtype)))
        else:
            out[name] = value
    return out


def _usda_string_literal(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _usda_float_literal(value: float) -> str:
    number = float(value)
    if not np.isfinite(number):
        number = 0.0
    return f"{number:.9g}"


def _native_runtime_attribute_line(attribute_name: str, value: Any) -> Optional[str]:
    if attribute_name in _NATIVE_OVRTX_RENDER_EXPOSURE_FLOAT_ATTRIBUTES:
        parsed = _first_float_value(value)
        if parsed is not None:
            return f"custom float {attribute_name} = {_usda_float_literal(parsed)}"
        return None
    if attribute_name in _NATIVE_OVRTX_RENDER_EXPOSURE_BOOL_ATTRIBUTES:
        string_values = [item.lower() for item in _string_values_from_value(value)]
        if string_values:
            if string_values[0] in ("true", "1", "yes", "on"):
                parsed = True
            elif string_values[0] in ("false", "0", "no", "off"):
                parsed = False
            else:
                parsed = None
        else:
            parsed = _first_bool_value(value)
        if parsed is not None:
            return f"custom bool {attribute_name} = {'true' if parsed else 'false'}"
    return None


def _nested_usda_overrides(overrides: dict[str, list[str]]) -> str:
    tree: dict[str, Any] = {}
    attrs_key = "__attrs__"
    for path, attr_lines in sorted(overrides.items()):
        parts = [part for part in str(path).split("/") if part]
        if not parts:
            continue
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
        node.setdefault(attrs_key, []).extend(attr_lines)

    lines: list[str] = []

    def emit(node: dict[str, Any], depth: int) -> None:
        indent = "    " * depth
        for name in sorted(key for key in node if key != attrs_key):
            child = node[name]
            lines.append(f'{indent}over "{_usda_string_literal(name)}"')
            lines.append(f"{indent}{{")
            for attr_line in child.get(attrs_key, []):
                lines.append(f"{indent}    {attr_line}")
            emit(child, depth + 1)
            lines.append(f"{indent}}}")

    emit(tree, 0)
    return "\n".join(lines)


def _relationship_target_text(targets: Iterable[str]) -> Optional[str]:
    clean = [str(target) for target in targets if str(target)]
    if not clean:
        return None
    if len(clean) == 1:
        return f"<{clean[0]}>"
    return "[" + ", ".join(f"<{target}>" for target in clean) + "]"


def _backend_name_from_config(config: RendererConfig) -> str:
    env = os.environ.get("NANOUSD_OVRTX_BACKEND")
    if env:
        return env.strip().lower()
    if config.use_vulkan is not None:
        return "vulkan" if config.use_vulkan else "opengl"
    return "metal" if sys.platform == "darwin" else "vulkan"


def _single_channel_image(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[..., None]
    return np.ascontiguousarray(array)


def _load_backend(backend: str):
    if backend == "vulkan":
        from nusd_renderer._bindings import NuRenderer, NU_RENDER_RASTER, NU_RENDER_SHADOW, NU_RENDER_RT
    elif backend == "opengl":
        from nusd_renderer_opengl._bindings import NuRenderer, NU_RENDER_RASTER, NU_RENDER_SHADOW, NU_RENDER_RT
    elif backend == "metal":
        from nusd_renderer_metal._bindings import NuRenderer, NU_RENDER_RASTER, NU_RENDER_SHADOW, NU_RENDER_RT
    else:
        raise ValueError(f"Unsupported NANOUSD_OVRTX_BACKEND={backend!r}")
    return NuRenderer, NU_RENDER_RASTER, NU_RENDER_SHADOW, NU_RENDER_RT


def _render_var_arrays(
    ldr: np.ndarray,
    names: Iterable[str],
    native_depth: Optional[np.ndarray] = None,
    native_normals: Optional[np.ndarray] = None,
    native_segmentation: Optional[np.ndarray] = None,
    camera: Optional[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = None,
    fov_degrees: float = 60.0,
    projection_shift: tuple[float, float] = (0.0, 0.0),
) -> dict[str, np.ndarray]:
    ldr = np.ascontiguousarray(ldr, dtype=np.uint8)
    h, w = ldr.shape[:2]
    requested = list(dict.fromkeys(names))
    out: dict[str, np.ndarray] = {}
    # Color-only fast path: an interactive viewport requesting only LdrColor (no
    # depth/normals/segmentation) does not need the per-pixel mask, np.mgrid
    # coordinate grids, or fallback-distance arrays built below — that is
    # ~52ms/frame @1080p of numpy work whose result the LdrColor branch discards
    # (it simply returns `ldr`). Return it directly. Bit-identical to the full
    # path for names==["LdrColor"]; everything else falls through unchanged.
    if (requested == ["LdrColor"]
            and native_depth is None
            and native_normals is None
            and native_segmentation is None):
        return {"LdrColor": ldr}
    mask = (ldr[..., :3].sum(axis=2) > 0).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x_norm = (xx / max(w - 1, 1)) * 2.0 - 1.0
    y_norm = (yy / max(h - 1, 1)) * 2.0 - 1.0
    fallback_distance = (1.0 + 9.0 * (0.25 + 0.75 * mask)).astype(np.float32)
    depth = None
    if native_depth is not None:
        native_depth = np.asarray(native_depth, dtype=np.float32)
        if native_depth.shape == (h, w):
            cleaned = np.where(np.isfinite(native_depth) & (native_depth > 0.0), native_depth, 0.0).astype(np.float32)
            if np.any(cleaned > 0.0):
                depth = np.ascontiguousarray(cleaned)
    distance = depth if depth is not None else fallback_distance

    normals_native = None
    if native_normals is not None:
        native_normals = np.asarray(native_normals, dtype=np.float32)
        if native_normals.shape == (h, w, 3) and np.any(np.isfinite(native_normals)):
            normals_native = np.nan_to_num(native_normals, copy=False)
    segmentation = None
    if native_segmentation is not None:
        native_segmentation = np.asarray(native_segmentation, dtype=np.uint32)
        if native_segmentation.shape == (h, w):
            segmentation = np.ascontiguousarray(native_segmentation)

    positions_from_distance: Optional[np.ndarray] = None
    image_plane_distance: Optional[np.ndarray] = None

    def camera_positions() -> Optional[np.ndarray]:
        nonlocal positions_from_distance
        if positions_from_distance is None and camera is not None:
            positions_from_distance = _positions_from_depth(distance, camera, fov_degrees, projection_shift)
        return positions_from_distance

    def distance_to_image_plane() -> np.ndarray:
        nonlocal image_plane_distance
        if image_plane_distance is not None:
            return image_plane_distance
        pos = camera_positions()
        if pos is None:
            image_plane_distance = np.ascontiguousarray(distance, dtype=np.float32)
        else:
            image_plane_distance = _image_plane_distance_from_positions(pos, camera, distance)
        return image_plane_distance

    for name in requested:
        if name == "LdrColor":
            out[name] = ldr
        elif name == "HdrColor":
            hdr = (ldr.astype(np.float32) / 255.0).astype(np.float16)
            out[name] = np.ascontiguousarray(hdr)
        elif name == "DiffuseAlbedoSD":
            out[name] = ldr.copy()
        elif name == "NormalSD":
            normals = np.zeros((h, w, 4), dtype=np.float32)
            if normals_native is not None and np.any(np.abs(normals_native) > 1e-6):
                normals[..., :3] = normals_native
                normals[..., 3] = (np.linalg.norm(normals_native, axis=2) > 1e-6).astype(np.float32)
            else:
                pos = camera_positions()
                if pos is None:
                    pos = _fallback_camera_positions(x_norm, y_norm, distance)
                valid = np.isfinite(distance) & (distance > 0.0)
                if depth is None:
                    valid &= mask > 0.0
                derived = _normals_from_positions(pos, valid)
                if derived is not None:
                    normals = derived
                else:
                    normals[..., 2] = 1.0
                    normals[..., 3] = valid.astype(np.float32)
            out[name] = np.ascontiguousarray(normals)
        elif name in ("DepthSD", "DistanceToImagePlaneSD"):
            out[name] = _single_channel_image(distance_to_image_plane().astype(np.float32, copy=False))
        elif name == "DistanceToCameraSD":
            out[name] = _single_channel_image(distance.astype(np.float32, copy=False))
        elif name == "Camera3dPositionSD":
            pos = camera_positions()
            if pos is None:
                pos = _fallback_camera_positions(x_norm, y_norm, distance)
            out[name] = np.ascontiguousarray(pos, dtype=np.float32)
        elif name in ("SemanticSegmentation", "SemanticSegmentationSD", "InstanceSegmentationSD"):
            if segmentation is not None:
                out[name] = _single_channel_image(segmentation.astype(np.uint32, copy=False))
            else:
                out[name] = _single_channel_image((mask > 0.0).astype(np.uint32))
        else:
            out[name] = ldr.copy()
    if "LdrColor" not in out:
        out["LdrColor"] = ldr
    return out


class Renderer:
    Semantic = Semantic
    PrimMode = PrimMode
    DataAccess = DataAccess
    Device = Device

    def __init__(self, config: Optional[RendererConfig] = None):
        self._config = config if config is not None else RendererConfig()
        self._backend = _backend_name_from_config(self._config)
        self._NuRenderer, self._NU_RENDER_RASTER, self._NU_RENDER_SHADOW, self._NU_RENDER_RT = _load_backend(self._backend)
        self._nu = None
        self._default_width = int(os.environ.get("NANOUSD_OVRTX_WIDTH", "1280"))
        self._default_height = int(os.environ.get("NANOUSD_OVRTX_HEIGHT", "720"))
        self._width = self._default_width
        self._height = self._default_height
        self._usd_handles: dict[int, str] = {}
        self._usd_prefixes: dict[int, Optional[str]] = {}
        self._usd_layers: dict[int, tuple[str, Optional[str]]] = {}
        self._removable_usd_handles: set[int] = set()
        self._clone_requests: list[tuple[str, tuple[str, ...]]] = []
        self._next_usd_handle = 1
        self._runtime_dir = tempfile.TemporaryDirectory(prefix="nanousd_ovrtx_")
        self._external_runtime_layers: set[str] = set()
        self._runtime_layer_index = 0
        self._native_stage_dirty = True
        self._native_load_paths: list[str] = []
        self._native_mesh_paths: list[tuple[int, str]] = []
        self._native_mesh_ids_by_prim_path: dict[str, np.ndarray] = {}
        self._native_mesh_initial_transforms_by_id: dict[int, np.ndarray] = {}
        self._native_mesh_relative_xforms_by_prim_path: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._native_gpu_transform_bindings: dict[int, _NativeGpuTransformBindingState] = {}
        self._native_mesh_index_valid = False
        self._active_included_purposes: Optional[tuple[str, ...]] = None
        self._active_material_binding_purposes: Optional[tuple[str, ...]] = None
        self._active_camera_render_overrides: Optional[tuple[Any, ...]] = None
        self._active_native_runtime_attribute_overrides: Optional[tuple[tuple[str, str, str], ...]] = None
        self._render_products: set[str] = set()
        self._render_product_specs: dict[str, _RenderProductSpec] = {}
        self._render_settings_specs: dict[str, _RenderSettingsSpec] = {}
        self._attrs: dict[tuple[str, str], np.ndarray] = {}
        self._attr_metadata: dict[tuple[str, str], tuple[Semantic, bool]] = {}
        self._bindings: dict[int, AttributeBinding] = {}
        self._next_binding_handle = 1
        self._prim_index: dict[str, dict[str, AttributeInfo]] = {}
        self._prim_path_ids: dict[str, int] = {}
        self._paths_by_prim_id: dict[int, str] = {}
        self._selection_group_styles: dict[int, SelectionGroupStyle] = {}
        self._camera_matrix: Optional[np.ndarray] = None
        self._camera_matrices: dict[str, np.ndarray] = {}
        self._camera_fovs: dict[str, float] = {}
        self._camera_clip_ranges: dict[str, tuple[float, float]] = {}
        self._camera_apertures: dict[str, tuple[float, float]] = {}
        self._camera_aperture_aspects: dict[str, float] = {}
        self._camera_aperture_offsets: dict[str, tuple[float, float]] = {}
        self._default_camera_path: Optional[str] = None
        self._last_camera: Optional[
            tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        ] = None
        self._camera_fov = 60.0
        self._near_clip = 0.1
        self._far_clip = 10000.0
        self._time = 0.0
        self._usd_time = 0.0

    @property
    def version(self) -> tuple:
        return (0, 3, 0)

    @property
    def config(self) -> RendererConfig:
        return self._config

    def _ensure_renderer(self):
        if self._nu is None:
            enable_rt = self._backend != "opengl"
            if self._config.enable_rt is not None:
                enable_rt = bool(self._config.enable_rt)
            enable_materials = True
            if self._config.enable_materials is not None:
                enable_materials = bool(self._config.enable_materials)
            self._nu = self._NuRenderer(
                width=self._width,
                height=self._height,
                enable_rt=enable_rt,
                enable_materials=enable_materials,
            )
        return self._nu

    def _reset_runtime_defaults(self) -> None:
        self._width = self._default_width
        self._height = self._default_height
        self._camera_fov = 60.0
        self._near_clip = 0.1
        self._far_clip = 10000.0

    def _clear_composed_stage(
        self,
        clear_handles: bool = False,
        clear_attrs: bool = False,
        clear_native: bool = True,
    ) -> None:
        if clear_native and self._nu is not None and hasattr(self._nu, "clear_scene"):
            self._nu.clear_scene()
        if clear_native:
            self._cleanup_external_runtime_layers()
        if clear_handles:
            self._usd_handles.clear()
            self._usd_prefixes.clear()
            self._usd_layers.clear()
            self._removable_usd_handles.clear()
            self._clone_requests.clear()
            self._active_included_purposes = None
            self._active_material_binding_purposes = None
            self._active_camera_render_overrides = None
            self._active_native_runtime_attribute_overrides = None
        if clear_attrs:
            self._attrs.clear()
            self._attr_metadata.clear()
        self._reset_runtime_defaults()
        self._native_load_paths.clear()
        self._invalidate_native_mesh_index()
        self._prim_index.clear()
        self._prim_path_ids.clear()
        self._paths_by_prim_id.clear()
        self._render_products.clear()
        self._render_product_specs.clear()
        self._render_settings_specs.clear()
        self._camera_matrix = None
        self._camera_matrices.clear()
        self._camera_fovs.clear()
        self._camera_clip_ranges.clear()
        self._camera_apertures.clear()
        self._camera_aperture_aspects.clear()
        self._camera_aperture_offsets.clear()
        self._default_camera_path = None
        self._last_camera = None

    def _write_runtime_layer(self, label: str, text: str) -> str:
        self._runtime_layer_index += 1
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "layer"
        path = Path(self._runtime_dir.name) / f"{self._runtime_layer_index:06d}_{safe_label}.usda"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def _cleanup_external_runtime_layers(self) -> None:
        for path in list(self._external_runtime_layers):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
        self._external_runtime_layers.clear()

    def _write_external_runtime_layer(self, directory: Path, label: str, text: str) -> str:
        self._runtime_layer_index += 1
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "layer"
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(directory),
            prefix=f".nanousd_ovrtx_{self._runtime_layer_index:06d}_{safe_label}_",
            suffix=".usda",
            delete=False,
        ) as f:
            f.write(text)
            path = f.name
        self._external_runtime_layers.add(path)
        return path

    def _queue_native_load(self, usd_path: str) -> None:
        self._native_load_paths.append(str(usd_path))

    def _release_native_gpu_transform_binding(self, handle: int) -> None:
        state = self._native_gpu_transform_bindings.pop(int(handle), None)
        if state is not None:
            state.close()

    def _release_native_gpu_transform_bindings(self) -> None:
        for state in list(self._native_gpu_transform_bindings.values()):
            state.close()
        self._native_gpu_transform_bindings.clear()

    def _invalidate_native_mesh_index(self) -> None:
        self._release_native_gpu_transform_bindings()
        self._native_mesh_paths.clear()
        self._native_mesh_ids_by_prim_path.clear()
        self._native_mesh_initial_transforms_by_id.clear()
        self._native_mesh_relative_xforms_by_prim_path.clear()
        self._native_mesh_index_valid = False

    def _refresh_native_mesh_index(self) -> None:
        self._native_mesh_paths.clear()
        self._native_mesh_ids_by_prim_path.clear()
        self._native_mesh_initial_transforms_by_id.clear()
        self._native_mesh_relative_xforms_by_prim_path.clear()
        self._native_mesh_index_valid = True
        nu = self._nu
        if nu is None:
            return
        try:
            count = int(getattr(nu, "mesh_count", 0))
        except Exception:
            return
        get_name = getattr(nu, "get_mesh_name", None)
        if not callable(get_name):
            return
        get_transform = getattr(nu, "get_mesh_transform", None)
        for mesh_id in range(max(count, 0)):
            try:
                name = str(get_name(mesh_id) or "")
            except Exception:
                continue
            if name:
                self._native_mesh_paths.append((mesh_id, name))
            if callable(get_transform):
                try:
                    self._native_mesh_initial_transforms_by_id[mesh_id] = np.ascontiguousarray(
                        np.asarray(get_transform(mesh_id), dtype=np.float32).reshape(4, 4)
                    )
                except Exception:
                    pass

    def _native_mesh_ids_for_prim_path(self, prim_path: str) -> np.ndarray:
        path = str(prim_path)
        cached = self._native_mesh_ids_by_prim_path.get(path)
        if cached is not None:
            return cached
        if not self._native_mesh_index_valid:
            self._refresh_native_mesh_index()
        prefix = path.rstrip("/") + "/"
        ids = [
            mesh_id
            for mesh_id, mesh_path in self._native_mesh_paths
            if mesh_path == path or mesh_path.startswith(prefix)
        ]
        arr = np.ascontiguousarray(np.asarray(ids, dtype=np.int32))
        self._native_mesh_ids_by_prim_path[path] = arr
        return arr

    @staticmethod
    def _native_transform_matrix_from_value(value: Any) -> np.ndarray:
        mat = np.asarray(value, dtype=np.float32).reshape(4, 4)
        return np.ascontiguousarray(mat)

    def _native_initial_transform_for_prim_path(self, prim_path: str, attribute_name: str = "omni:xform") -> np.ndarray:
        path = str(prim_path)
        exact_ids = [
            mesh_id
            for mesh_id, mesh_path in self._native_mesh_paths
            if mesh_path == path and mesh_id in self._native_mesh_initial_transforms_by_id
        ]
        if exact_ids:
            return self._native_mesh_initial_transforms_by_id[exact_ids[0]]
        for name in (attribute_name, "omni:xform", "xformOp:transform"):
            read_value = self._attribute_value_for_read(path, name)
            if read_value is None:
                continue
            value, semantic = read_value
            if semantic != Semantic.XFORM_MAT4x4 and name not in ("omni:xform", "xformOp:transform"):
                continue
            try:
                return self._native_transform_matrix_from_value(value)
            except Exception:
                continue
        return np.eye(4, dtype=np.float32)

    def _native_mesh_relative_xforms_for_prim_path(
        self,
        prim_path: str,
        attribute_name: str = "omni:xform",
    ) -> tuple[np.ndarray, np.ndarray]:
        path = str(prim_path)
        cache_key = (path, str(attribute_name))
        cached = self._native_mesh_relative_xforms_by_prim_path.get(cache_key)
        if cached is not None:
            return cached
        ids = self._native_mesh_ids_for_prim_path(path)
        if ids.size == 0:
            empty_ids = np.ascontiguousarray(np.zeros((0,), dtype=np.int32))
            empty_rel = np.ascontiguousarray(np.zeros((0, 4, 4), dtype=np.float32))
            self._native_mesh_relative_xforms_by_prim_path[cache_key] = (empty_ids, empty_rel)
            return empty_ids, empty_rel
        parent_initial = self._native_initial_transform_for_prim_path(path, attribute_name)
        try:
            parent_inv = np.linalg.inv(parent_initial.astype(np.float64)).astype(np.float32)
        except Exception:
            parent_inv = np.eye(4, dtype=np.float32)
        rels: list[np.ndarray] = []
        for mesh_id in ids:
            mesh_initial = self._native_mesh_initial_transforms_by_id.get(int(mesh_id))
            if mesh_initial is None:
                mesh_initial = np.eye(4, dtype=np.float32)
            rels.append(np.ascontiguousarray(parent_inv @ mesh_initial, dtype=np.float32))
        rel_arr = np.ascontiguousarray(np.stack(rels, axis=0), dtype=np.float32)
        result = (ids, rel_arr)
        self._native_mesh_relative_xforms_by_prim_path[cache_key] = result
        return result

    def _native_gpu_transforms_enabled(self) -> bool:
        return (
            self._backend == "vulkan"
            and bool(getattr(self._config, "read_gpu_transforms", False))
            and self._nu is not None
        )

    def _ensure_native_stage_for_attribute_mapping(self) -> None:
        if self._native_stage_dirty or self._nu is None:
            self._active_included_purposes = None
            self._active_material_binding_purposes = None
            self._active_camera_render_overrides = None
            self._active_native_runtime_attribute_overrides = None
            self._rebuild_composed_stage(load_native=True)

    def _try_map_native_gpu_transforms(
        self,
        binding: AttributeBinding,
        map_paths: list[str],
        initial_array: np.ndarray,
        device: Device,
        device_id: int,
    ) -> Optional[AttributeMapping]:
        if device not in (Device.CUDA, Device.CUDA_ARRAY):
            return None
        if binding.is_array:
            return None
        if not _is_native_mutable_xform_attribute(binding._attribute_name, binding.semantic):
            return None
        if self._backend != "vulkan" or not bool(getattr(self._config, "read_gpu_transforms", False)):
            return None

        self._ensure_native_stage_for_attribute_mapping()
        if not self._native_gpu_transforms_enabled():
            return None
        nu = self._nu
        if nu is None or not all(
            hasattr(nu, name)
            for name in ("get_transforms_interop_info", "set_transform_layout", "translate_instances_gpu")
        ):
            return None

        mesh_id_chunks: list[np.ndarray] = []
        parent_index_chunks: list[np.ndarray] = []
        relative_chunks: list[np.ndarray] = []
        for parent_index, prim_path in enumerate(map_paths):
            if self._is_camera_path(prim_path):
                continue
            ids, rels = self._native_mesh_relative_xforms_for_prim_path(prim_path, binding._attribute_name)
            if ids.size == 0:
                continue
            mesh_id_chunks.append(ids)
            parent_index_chunks.append(np.full((int(ids.size),), parent_index, dtype=np.int32))
            relative_chunks.append(rels)
        if not mesh_id_chunks:
            return None

        mesh_ids = np.ascontiguousarray(np.concatenate(mesh_id_chunks), dtype=np.int32)
        parent_indices = np.ascontiguousarray(np.concatenate(parent_index_chunks), dtype=np.int32)
        relative_xforms = np.ascontiguousarray(np.concatenate(relative_chunks, axis=0), dtype=np.float32)
        state = self._native_gpu_transform_bindings.get(int(binding.handle))
        if state is None or not state.compatible(device_id, len(map_paths), mesh_ids, parent_indices, relative_xforms):
            self._release_native_gpu_transform_binding(int(binding.handle))
            state = _NativeGpuTransformBindingState(
                self,
                int(binding.handle),
                int(device_id),
                np.ascontiguousarray(initial_array, dtype=np.float64).reshape(len(map_paths), 4, 4),
                mesh_ids,
                parent_indices,
                relative_xforms,
            )
            self._native_gpu_transform_bindings[int(binding.handle)] = state
        return AttributeMapping(
            binding,
            state.source,
            device=Device.CUDA,
            mapped_array=state.source,
            commit_callback=lambda _mapping, s=state: s.commit(),
        )

    def _native_attribute_can_mutate_without_reload(
        self,
        prim_path: str,
        attribute_name: str,
        semantic: Semantic,
    ) -> bool:
        if self._is_camera_path(prim_path):
            return attribute_name in _NATIVE_RUNTIME_CAMERA_ATTRIBUTES
        return (
            _is_native_mutable_xform_attribute(attribute_name, semantic)
            or _is_native_mutable_color_attribute(attribute_name)
            or _is_native_mutable_visibility_attribute(attribute_name)
        )

    def _apply_native_runtime_attribute_overrides(self) -> None:
        if self._nu is None:
            return
        xform_records: list[tuple[str, str, np.ndarray, Semantic]] = []
        color_records: list[tuple[str, np.ndarray, Semantic]] = []
        visibility_records: list[tuple[str, np.ndarray, Semantic]] = []
        for (path, attribute_name), value in list(self._attrs.items()):
            semantic, _is_array = self._attr_metadata.get((path, attribute_name), (Semantic.NONE, False))
            if self._is_camera_path(path):
                continue
            if _is_native_mutable_xform_attribute(attribute_name, semantic):
                xform_records.append((path, attribute_name, np.asarray(value), semantic))
            elif _is_native_mutable_color_attribute(attribute_name):
                color_records.append((path, np.asarray(value), semantic))
            elif _is_native_mutable_visibility_attribute(attribute_name):
                visibility_records.append((path, np.asarray(value), semantic))
        if xform_records:
            by_attribute: dict[str, tuple[list[str], list[np.ndarray]]] = {}
            for path, attribute_name, value, _semantic in xform_records:
                paths, values = by_attribute.setdefault(attribute_name, ([], []))
                paths.append(path)
                values.append(value)
            for attribute_name, (paths, values) in by_attribute.items():
                self._apply_native_transform_batch(paths, values, attribute_name)
        if color_records:
            self._apply_native_color_batch(
                [path for path, _value, _semantic in color_records],
                [value for _path, value, _semantic in color_records],
            )
        if visibility_records:
            self._apply_native_visibility_batch(
                [path for path, _value, _semantic in visibility_records],
                [value for _path, value, _semantic in visibility_records],
            )

    def _apply_native_attribute_batch(
        self,
        prim_paths: list[str],
        attribute_name: str,
        values: list[Any],
        semantic: Semantic,
    ) -> None:
        if self._nu is None or not prim_paths:
            return
        if _is_native_mutable_xform_attribute(attribute_name, semantic):
            self._apply_native_transform_batch(prim_paths, values, attribute_name)
        elif _is_native_mutable_color_attribute(attribute_name):
            self._apply_native_color_batch(prim_paths, values)
        elif _is_native_mutable_visibility_attribute(attribute_name):
            self._apply_native_visibility_batch(prim_paths, values)

    def _apply_native_transform_batch(
        self,
        prim_paths: list[str],
        values: list[Any],
        attribute_name: str = "omni:xform",
    ) -> None:
        nu = self._nu
        if nu is None or not hasattr(nu, "set_transforms"):
            return
        mesh_ids: list[int] = []
        transforms: list[np.ndarray] = []
        for prim_path, value in zip(prim_paths, values):
            if self._is_camera_path(prim_path):
                continue
            ids, relative_xforms = self._native_mesh_relative_xforms_for_prim_path(prim_path, attribute_name)
            if ids.size == 0:
                continue
            try:
                mat = np.asarray(value, dtype=np.float32).reshape(4, 4)
            except Exception:
                continue
            expanded = np.matmul(mat.astype(np.float32), relative_xforms)
            mesh_ids.extend(int(i) for i in ids)
            transforms.extend(np.ascontiguousarray(x.reshape(16), dtype=np.float32) for x in expanded)
        if not mesh_ids:
            return
        ids_arr = np.ascontiguousarray(np.asarray(mesh_ids, dtype=np.int32))
        xforms_arr = np.ascontiguousarray(np.stack(transforms, axis=0), dtype=np.float32)
        nu.set_transforms(ids_arr, xforms_arr)

    @staticmethod
    def _rgb_from_native_color_value(value: Any) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
        if arr.size < 3:
            return None
        if arr.ndim == 0:
            return None
        if arr.shape[-1] >= 3:
            rgb = arr.reshape((-1, arr.shape[-1]))[0, :3]
        else:
            flat = arr.ravel()
            if flat.size < 3:
                return None
            rgb = flat[:3]
        if not np.all(np.isfinite(rgb)):
            return None
        return np.ascontiguousarray(rgb.astype(np.float32, copy=False))

    def _apply_native_color_batch(self, prim_paths: list[str], values: list[Any]) -> None:
        nu = self._nu
        if nu is None or not hasattr(nu, "set_colors"):
            return
        mesh_ids: list[int] = []
        colors: list[np.ndarray] = []
        for prim_path, value in zip(prim_paths, values):
            ids = self._native_mesh_ids_for_prim_path(prim_path)
            if ids.size == 0:
                continue
            rgb = self._rgb_from_native_color_value(value)
            if rgb is None:
                continue
            mesh_ids.extend(int(i) for i in ids)
            colors.extend(rgb for _ in range(int(ids.size)))
        if not mesh_ids:
            return
        ids_arr = np.ascontiguousarray(np.asarray(mesh_ids, dtype=np.int32))
        colors_arr = np.ascontiguousarray(np.stack(colors, axis=0), dtype=np.float32)
        nu.set_colors(ids_arr, colors_arr)

    @staticmethod
    def _visible_from_native_visibility_value(value: Any) -> Optional[bool]:
        values = _string_values_from_value(value)
        if values:
            return values[0] != "invisible"
        try:
            arr = np.asarray(value)
        except Exception:
            return None
        if arr.size == 0:
            return None
        first = arr.ravel()[0]
        if isinstance(first, (bytes, str)):
            return str(first.decode("utf-8") if isinstance(first, bytes) else first) != "invisible"
        try:
            return bool(first)
        except Exception:
            return None

    def _apply_native_visibility_batch(self, prim_paths: list[str], values: list[Any]) -> None:
        nu = self._nu
        if nu is None or not hasattr(nu, "set_visibility"):
            return
        mesh_ids: list[int] = []
        visible_values: list[bool] = []
        for prim_path, value in zip(prim_paths, values):
            ids = self._native_mesh_ids_for_prim_path(prim_path)
            if ids.size == 0:
                continue
            visible = self._visible_from_native_visibility_value(value)
            if visible is None:
                continue
            mesh_ids.extend(int(i) for i in ids)
            visible_values.extend(bool(visible) for _ in range(int(ids.size)))
        if not mesh_ids:
            return
        nu.set_visibility(np.ascontiguousarray(np.asarray(mesh_ids, dtype=np.int32)), visible_values)

    def _render_settings_override_layer_text(self) -> str:
        overrides: dict[str, list[str]] = {}
        if self._active_included_purposes is not None:
            included = {str(purpose) for purpose in self._active_included_purposes}
            for prim_path, attrs in self._prim_index.items():
                purpose_info = attrs.get("purpose")
                purpose_values = _string_values_from_value(getattr(purpose_info, "value", None))
                if not purpose_values:
                    continue
                purpose = purpose_values[0]
                if purpose in included:
                    if purpose in {"guide", "proxy"}:
                        overrides.setdefault(prim_path, []).append('uniform token purpose = "default"')
                elif purpose in {"default", "render", "guide", "proxy"}:
                    overrides.setdefault(prim_path, []).append('uniform token visibility = "invisible"')

        if self._active_material_binding_purposes is not None:
            ordered = [str(purpose) for purpose in self._active_material_binding_purposes]
            rel_names = [f"material:binding:{purpose}" if purpose else "material:binding" for purpose in ordered]
            for prim_path, attrs in self._prim_index.items():
                selected_targets: Optional[list[str]] = None
                for rel_name in rel_names:
                    info = attrs.get(rel_name)
                    targets = _string_values_from_value(getattr(info, "value", None))
                    if targets:
                        selected_targets = targets
                        break
                if selected_targets is None:
                    continue
                target_text = _relationship_target_text(selected_targets)
                if target_text is not None:
                    overrides.setdefault(prim_path, []).append(f"rel material:binding = {target_text}")

        if self._active_camera_render_overrides is not None:
            camera_paths, motion_mode, disable_depth_of_field, shutter_opens = self._active_camera_render_overrides
            for index, camera_path in enumerate(camera_paths):
                attr_lines: list[str] = []
                if motion_mode == "disable":
                    attr_lines.extend(["double shutter:open = 0", "double shutter:close = 0"])
                elif motion_mode == "instantaneous":
                    shutter_open = shutter_opens[index] if index < len(shutter_opens) else 0.0
                    attr_lines.append(f"double shutter:close = {_usda_float_literal(shutter_open)}")
                if disable_depth_of_field:
                    attr_lines.append(f"float fStop = {_usda_float_literal(_USD_DISABLE_DEPTH_OF_FIELD_FSTOP)}")
                if attr_lines:
                    overrides.setdefault(str(camera_path), []).extend(attr_lines)

        if self._active_native_runtime_attribute_overrides is not None:
            for prim_path, _attribute_name, attr_line in self._active_native_runtime_attribute_overrides:
                overrides.setdefault(prim_path, []).append(attr_line)

        return _nested_usda_overrides(overrides)

    def _mirror_material_sidecars(self, usd_paths: Iterable[str]) -> None:
        runtime_dir = Path(self._runtime_dir.name)
        search_dirs: set[Path] = set()
        material_suffixes = {".mtlx", ".mdl", ".mdle"}
        for usd_path in usd_paths:
            path = Path(usd_path)
            if path.exists():
                search_dirs.add(path.parent)
                if not _is_text_usd_layer(path):
                    continue
                text = _read_text(str(path))
                for ref in _RE_ASSET_REF.findall(text):
                    if not _safe_sidecar_asset_ref(ref):
                        continue
                    try:
                        ref_path = Path(ref)
                    except Exception:
                        continue
                    if not ref_path.is_absolute():
                        ref_path = path.parent / ref_path
                    try:
                        exists = ref_path.exists()
                    except OSError:
                        exists = False
                    if exists:
                        search_dirs.add(ref_path.parent)
        for directory in search_dirs:
            try:
                for sidecar in directory.iterdir():
                    if not sidecar.is_file() or sidecar.suffix.lower() not in material_suffixes:
                        continue
                    dst = runtime_dir / sidecar.name
                    if not dst.exists():
                        dst.symlink_to(sidecar)
            except Exception:
                pass

    def _flush_native_loads(self) -> None:
        if not self._native_load_paths:
            return
        loaded_native_stage = False
        try:
            nu = self._ensure_renderer()
            if hasattr(nu, "set_current_time"):
                nu.set_current_time(float(self._usd_time))
            if hasattr(nu, "set_render_size"):
                nu.set_render_size(self._width, self._height)
            self._mirror_material_sidecars(self._native_load_paths)
            render_settings_overrides = self._render_settings_override_layer_text()
            if render_settings_overrides:
                sublayers = "\n".join(f"        @{_asset_ref(path)}@" for path in self._native_load_paths)
                layer = f"""#usda 1.0
(
    subLayers = [
{sublayers}
    ]
)

{render_settings_overrides}
"""
                nu.load_usd(self._write_runtime_layer("purpose_filtered_stage", layer))
                loaded_native_stage = True
            elif len(self._native_load_paths) == 1:
                nu.load_usd(self._native_load_paths[0])
                loaded_native_stage = True
            else:
                sublayers = "\n".join(f"        @{_asset_ref(path)}@" for path in self._native_load_paths)
                layer = f"""#usda 1.0
(
    subLayers = [
{sublayers}
    ]
)
"""
                nu.load_usd(self._write_runtime_layer("composed_stage", layer))
                loaded_native_stage = True
            if loaded_native_stage:
                self._refresh_native_mesh_index()
                self._apply_native_runtime_attribute_overrides()
        finally:
            self._native_load_paths.clear()

    def _layer_path_for_composed_path(
        self,
        composed_path: str,
        source_text: str,
        path_prefix: Optional[str],
    ) -> Optional[tuple[str, str]]:
        prefix = _normalize_prefix(path_prefix)
        default_prim = _default_prim_name(source_text)
        if prefix is None:
            layer_path = composed_path
        elif default_prim:
            if composed_path == prefix:
                layer_path = f"/{default_prim}"
            elif composed_path.startswith(prefix + "/"):
                layer_path = f"/{default_prim}{composed_path[len(prefix):]}"
            else:
                return None
        else:
            if composed_path == prefix or not composed_path.startswith(prefix + "/"):
                return None
            layer_path = composed_path[len(prefix):] or "/"
        if layer_path == "/":
            return None
        indexed = _scan_prim_index(source_text)
        attrs = indexed.get(layer_path)
        if attrs is None:
            return None
        prim_type = getattr(attrs.get("__type__"), "value", None) or "Xform"
        return layer_path, str(prim_type)

    def _resolve_clone_source(self, source_path: str) -> Optional[tuple[str, str, str]]:
        source = _normalize_prefix(source_path)
        if source is None:
            return None
        for handle in sorted(self._usd_handles):
            path = self._usd_handles[handle]
            if path != "<inline>":
                text = _read_text(path)
                mapped = self._layer_path_for_composed_path(source, text, self._usd_prefixes.get(handle))
                if mapped:
                    layer_path, prim_type = mapped
                    return os.path.abspath(path), layer_path, prim_type
            elif handle in self._usd_layers:
                text, layer_prefix = self._usd_layers[handle]
                mapped = self._layer_path_for_composed_path(source, text, layer_prefix)
                if mapped:
                    layer_path, prim_type = mapped
                    return self._write_runtime_layer("clone_inline_source", text), layer_path, prim_type
        return None

    def _index_layer_text(self, text: str, path_prefix: Optional[str] = None) -> None:
        default_prim = _default_prim_name(text)
        meta = _scan_render_metadata(text)
        specs, settings_specs = _scan_render_products(text)
        self._width = int(meta.get("width", self._width))
        self._height = int(meta.get("height", self._height))
        if "camera_path" in meta:
            self._default_camera_path = str(_remap_reference_path(meta["camera_path"], path_prefix, default_prim))
        if "render_product" in meta:
            render_product = _remap_reference_path(str(meta["render_product"]), path_prefix, default_prim)
            if render_product:
                self._render_products.add(str(render_product))
        for path, spec in specs.items():
            mapped_path = _remap_reference_path(path, path_prefix, default_prim) or path
            mapped_spec = _RenderProductSpec(
                path=mapped_path,
                width=spec.width,
                height=spec.height,
                camera_path=_remap_reference_path(spec.camera_path, path_prefix, default_prim),
                camera_paths=[
                    str(_remap_reference_path(camera_path, path_prefix, default_prim))
                    for camera_path in (spec.camera_paths or [])
                    if _remap_reference_path(camera_path, path_prefix, default_prim)
                ] or None,
                render_vars=list(spec.render_vars or []),
                render_var_paths=[
                    str(_remap_reference_path(render_var_path, path_prefix, default_prim))
                    for render_var_path in (spec.render_var_paths or [])
                    if _remap_reference_path(render_var_path, path_prefix, default_prim)
                ],
                render_mode=spec.render_mode,
                minimal_mode=spec.minimal_mode,
                pixel_aspect_ratio=spec.pixel_aspect_ratio,
                aspect_ratio_conform_policy=spec.aspect_ratio_conform_policy,
                data_window_ndc=spec.data_window_ndc,
                disable_motion_blur=spec.disable_motion_blur,
                disable_depth_of_field=spec.disable_depth_of_field,
                instantaneous_shutter=spec.instantaneous_shutter,
                product_name=spec.product_name,
                product_type=spec.product_type,
                included_purposes=list(spec.included_purposes or []) if spec.included_purposes is not None else None,
                material_binding_purposes=(
                    list(spec.material_binding_purposes or []) if spec.material_binding_purposes is not None else None
                ),
                rendering_color_space=spec.rendering_color_space,
            )
            self._render_product_specs[mapped_path] = mapped_spec
            self._render_products.add(mapped_path)
            if mapped_spec.camera_path and self._default_camera_path is None:
                self._default_camera_path = mapped_spec.camera_path
            if mapped_spec.width and mapped_spec.height:
                self._width, self._height = int(mapped_spec.width), int(mapped_spec.height)
        for path, spec in settings_specs.items():
            mapped_path = _remap_reference_path(path, path_prefix, default_prim) or path
            self._render_settings_specs[mapped_path] = _RenderSettingsSpec(
                path=mapped_path,
                product_paths=[
                    str(_remap_reference_path(product_path, path_prefix, default_prim))
                    for product_path in (spec.product_paths or [])
                    if _remap_reference_path(product_path, path_prefix, default_prim)
                ],
                width=spec.width,
                height=spec.height,
                camera_path=_remap_reference_path(spec.camera_path, path_prefix, default_prim),
                camera_paths=[
                    str(_remap_reference_path(camera_path, path_prefix, default_prim))
                    for camera_path in (spec.camera_paths or [])
                    if _remap_reference_path(camera_path, path_prefix, default_prim)
                ] or None,
                render_mode=spec.render_mode,
                minimal_mode=spec.minimal_mode,
                pixel_aspect_ratio=spec.pixel_aspect_ratio,
                aspect_ratio_conform_policy=spec.aspect_ratio_conform_policy,
                data_window_ndc=spec.data_window_ndc,
                disable_motion_blur=spec.disable_motion_blur,
                disable_depth_of_field=spec.disable_depth_of_field,
                instantaneous_shutter=spec.instantaneous_shutter,
                included_purposes=list(spec.included_purposes or []),
                material_binding_purposes=list(spec.material_binding_purposes or []),
                rendering_color_space=spec.rendering_color_space,
            )
        indexed_attrs: dict[str, dict[str, AttributeInfo]] = {}
        for path, attrs in _scan_prim_index(text).items():
            mapped_path = _remap_reference_path(path, path_prefix, default_prim) or path
            mapped_attrs = _remap_path_string_attribute_values(attrs, path_prefix, default_prim)
            self._prim_index[mapped_path] = mapped_attrs
            indexed_attrs[mapped_path] = mapped_attrs
        self._apply_indexed_attribute_side_effects(indexed_attrs)

    def _prim_type(self, path: str) -> Optional[str]:
        return getattr(self._prim_index.get(path, {}).get("__type__"), "value", None)

    def _is_render_settings_path(self, path: str) -> bool:
        return self._prim_type(path) == "RenderSettings"

    def _is_render_product_path(self, path: str) -> bool:
        return self._prim_type(path) == "RenderProduct"

    def _render_settings_spec_for(self, settings_path: str) -> _RenderSettingsSpec:
        return self._render_settings_specs.setdefault(
            settings_path,
            _RenderSettingsSpec(path=settings_path, product_paths=[]),
        )

    def _product_has_own_attribute(self, product_path: str, attribute_names: Iterable[str]) -> bool:
        attrs = self._prim_index.get(product_path, {})
        return any(name in attrs or (product_path, name) in self._attrs for name in attribute_names)

    def _apply_render_settings_defaults(self, settings_path: str) -> None:
        settings_spec = self._render_settings_specs.get(settings_path)
        if settings_spec is None:
            return
        for product_path in settings_spec.product_paths or []:
            self._render_products.add(product_path)
            product = self._render_product_specs.setdefault(
                product_path,
                _RenderProductSpec(path=product_path, render_vars=["LdrColor"]),
            )
            if settings_spec.camera_path and not self._product_has_own_attribute(product_path, ("camera",)):
                product.camera_path = settings_spec.camera_path
                product.camera_paths = list(settings_spec.camera_paths or [settings_spec.camera_path])
                if self._default_camera_path is None:
                    self._default_camera_path = settings_spec.camera_path
            if (
                settings_spec.width is not None
                and settings_spec.height is not None
                and not self._product_has_own_attribute(product_path, ("resolution",))
            ):
                product.width = settings_spec.width
                product.height = settings_spec.height
            if (
                settings_spec.render_mode is not None
                and not self._product_has_own_attribute(
                    product_path,
                    ("renderMode", "nanousd:renderMode", "omni:rtx:rendermode"),
                )
            ):
                product.render_mode = settings_spec.render_mode
            if (
                settings_spec.minimal_mode is not None
                and not self._product_has_own_attribute(product_path, ("omni:rtx:minimal:mode",))
            ):
                product.minimal_mode = settings_spec.minimal_mode
            if (
                settings_spec.pixel_aspect_ratio is not None
                and not self._product_has_own_attribute(product_path, ("pixelAspectRatio",))
            ):
                product.pixel_aspect_ratio = settings_spec.pixel_aspect_ratio
            if (
                settings_spec.aspect_ratio_conform_policy is not None
                and not self._product_has_own_attribute(product_path, ("aspectRatioConformPolicy",))
            ):
                product.aspect_ratio_conform_policy = settings_spec.aspect_ratio_conform_policy
            if (
                settings_spec.data_window_ndc is not None
                and not self._product_has_own_attribute(product_path, ("dataWindowNDC",))
            ):
                product.data_window_ndc = settings_spec.data_window_ndc
            if (
                settings_spec.disable_motion_blur is not None
                and not self._product_has_own_attribute(product_path, ("disableMotionBlur",))
            ):
                product.disable_motion_blur = settings_spec.disable_motion_blur
            if (
                settings_spec.disable_depth_of_field is not None
                and not self._product_has_own_attribute(product_path, ("disableDepthOfField",))
            ):
                product.disable_depth_of_field = settings_spec.disable_depth_of_field
            if (
                settings_spec.instantaneous_shutter is not None
                and not self._product_has_own_attribute(product_path, ("instantaneousShutter",))
            ):
                product.instantaneous_shutter = settings_spec.instantaneous_shutter
            if settings_spec.included_purposes is not None:
                product.included_purposes = list(settings_spec.included_purposes)
            if settings_spec.material_binding_purposes is not None:
                product.material_binding_purposes = list(settings_spec.material_binding_purposes)
            if settings_spec.rendering_color_space is not None:
                product.rendering_color_space = settings_spec.rendering_color_space

    def _apply_indexed_attribute_side_effects(self, indexed_attrs: dict[str, dict[str, AttributeInfo]]) -> None:
        for prim_path, attrs in indexed_attrs.items():
            for attribute_name, info in attrs.items():
                if attribute_name == "__type__" or (prim_path, attribute_name) in self._attrs:
                    continue
                value = getattr(info, "value", None)
                if value is None:
                    continue
                self._apply_attribute_side_effect(
                    prim_path,
                    attribute_name,
                    value,
                    info.semantic,
                    use_overrides=False,
                )

    def _load_usd_path_recursive(
        self,
        usd_file_path: str,
        seen: Optional[set[tuple[str, Optional[str], bool]]] = None,
        path_prefix: Optional[str] = None,
        load_native: bool = True,
    ) -> None:
        seen = seen if seen is not None else set()
        abs_path = os.path.abspath(usd_file_path)
        prefix = _normalize_prefix(path_prefix)
        seen_key = (abs_path, prefix, bool(load_native))
        if seen_key in seen:
            return
        seen.add(seen_key)

        text = _read_text(abs_path)
        self._index_layer_text(text, path_prefix)
        if load_native:
            load_path = abs_path
            if prefix is not None:
                wrapper = _make_reference_wrapper(abs_path, prefix, text)
                load_path = self._write_runtime_layer(f"ref_{Path(abs_path).stem}", wrapper)
            self._queue_native_load(load_path)

        for layer_path in _scan_sublayers(text, Path(abs_path).parent):
            if Path(layer_path).exists():
                self._load_usd_path_recursive(
                    str(layer_path),
                    seen,
                    path_prefix,
                    load_native=False,
                )

    def _load_inline_layer_text(
        self,
        usd_layer_content: str,
        path_prefix: Optional[str],
        seen: Optional[set[tuple[str, Optional[str], bool]]] = None,
        load_native: bool = True,
    ) -> None:
        prefix = _normalize_prefix(path_prefix)
        self._index_layer_text(usd_layer_content, path_prefix)
        base_dir = Path.cwd()
        sublayers = [layer_path for layer_path in _scan_sublayers(usd_layer_content, base_dir) if Path(layer_path).exists()]
        inline_has_native_scene_prims = _inline_layer_has_native_scene_prims(usd_layer_content)
        if load_native:
            if prefix is None and sublayers and not inline_has_native_scene_prims:
                for layer_path in sublayers:
                    self._load_usd_path_recursive(layer_path, seen, path_prefix, load_native=True)
            else:
                if prefix is None and inline_has_native_scene_prims and len(sublayers) == 1:
                    colocated_dir = Path(sublayers[0]).parent
                    source_text = _colocated_inline_layer_text(usd_layer_content, colocated_dir)
                    try:
                        source_path = self._write_external_runtime_layer(colocated_dir, "inline_source", source_text)
                    except Exception:
                        source_text = _runtime_inline_layer_text(usd_layer_content, base_dir)
                        source_path = self._write_runtime_layer("inline_source", source_text)
                else:
                    source_text = _runtime_inline_layer_text(usd_layer_content, base_dir)
                    source_path = self._write_runtime_layer("inline_source", source_text)
                load_path = source_path
                if prefix is not None:
                    wrapper = _make_reference_wrapper(source_path, prefix, source_text)
                    load_path = self._write_runtime_layer("inline_ref", wrapper)
                self._queue_native_load(load_path)

        if not (load_native and prefix is None and sublayers and not inline_has_native_scene_prims):
            for layer_path in sublayers:
                self._load_usd_path_recursive(layer_path, seen, path_prefix, load_native=False)

    def _rebuild_composed_stage(self, load_native: bool = True) -> None:
        self._clear_composed_stage(clear_handles=False, clear_attrs=False, clear_native=load_native)
        if load_native:
            self._native_load_paths = []
        seen: set[tuple[str, Optional[str], bool]] = set()
        for handle in sorted(self._usd_handles):
            path = self._usd_handles[handle]
            if path != "<inline>":
                self._load_usd_path_recursive(path, seen, self._usd_prefixes.get(handle), load_native=load_native)
        for handle in sorted(self._usd_layers):
            text, path_prefix = self._usd_layers[handle]
            self._load_inline_layer_text(text, path_prefix, seen, load_native=load_native)
        self._index_clone_requests()
        self._apply_usd_time_samples()
        self._reapply_attribute_overrides()
        if load_native:
            self._load_native_clone_requests()
            self._flush_native_loads()
        if self._usd_time and self._nu is not None and hasattr(self._nu, "set_current_time"):
            self._nu.set_current_time(float(self._usd_time))
        self._refresh_path_dictionary()
        self._native_stage_dirty = not load_native

    def _refresh_path_dictionary(self) -> None:
        self._prim_path_ids.clear()
        self._paths_by_prim_id.clear()
        for path in sorted(self._prim_index):
            path_id = _stable_prim_path_id(path)
            while path_id in self._paths_by_prim_id and self._paths_by_prim_id[path_id] != path:
                path_id = (path_id + 1) & 0xFFFFFFFFFFFFFFFF
                if path_id == 0:
                    path_id = 1
            self._prim_path_ids[path] = path_id
            self._paths_by_prim_id[path_id] = path

    def _prim_path_id(self, prim_path: str) -> int:
        path = str(prim_path)
        if path in self._prim_index and path not in self._prim_path_ids:
            self._refresh_path_dictionary()
        return int(self._prim_path_ids.get(path, 0))

    def _reapply_attribute_overrides(self) -> None:
        for (path, name), arr in list(self._attrs.items()):
            semantic, is_array = self._attr_metadata.get((path, name), (Semantic.NONE, False))
            self._write_attr_array([path], name, arr, semantic, is_array)

    def _apply_usd_time_samples(self) -> None:
        for path, attrs in list(self._prim_index.items()):
            for name, info in list(attrs.items()):
                samples = getattr(info, "time_samples", None)
                if not samples:
                    continue
                value = _sample_time_value(samples, float(self._usd_time))
                if value is None:
                    continue
                value_arr = np.ascontiguousarray(value)
                info.value = value_arr
                self._apply_attribute_side_effect(path, name, value_arr, info.semantic, use_overrides=False)

    def _index_clone_requests(self) -> None:
        for source_path, target_paths in self._clone_requests:
            source = _normalize_prefix(source_path)
            if source is None:
                continue
            matches = [
                (path, attrs)
                for path, attrs in list(self._prim_index.items())
                if path == source or path.startswith(source + "/")
            ]
            override_matches = [
                (path, attr_name, arr, self._attr_metadata.get((path, attr_name), (Semantic.NONE, False)))
                for (path, attr_name), arr in list(self._attrs.items())
                if path == source or path.startswith(source + "/")
            ]
            for target_path in target_paths:
                target = _normalize_prefix(target_path)
                if target is None:
                    continue
                cloned_attrs: dict[str, dict[str, AttributeInfo]] = {}
                for path, attrs in matches:
                    suffix = path[len(source):]
                    cloned_path = target + suffix
                    cloned = {
                        name: _clone_attribute_info(info, source, target)
                        for name, info in attrs.items()
                    }
                    self._prim_index[cloned_path] = cloned
                    cloned_attrs[cloned_path] = cloned
                self._apply_indexed_attribute_side_effects(cloned_attrs)
                for path, attr_name, arr, (semantic, is_array) in override_matches:
                    suffix = path[len(source):]
                    cloned_path = target + suffix
                    if (cloned_path, attr_name) in self._attrs:
                        continue
                    self._write_attr_array(
                        [cloned_path],
                        attr_name,
                        np.ascontiguousarray(np.asarray(arr).copy()),
                        semantic,
                        is_array,
                    )

    def _load_native_clone_requests(self) -> None:
        for source_path, target_paths in self._clone_requests:
            resolved = self._resolve_clone_source(source_path)
            if resolved is None:
                continue
            asset_path, layer_path, prim_type = resolved
            source_text = _read_text(asset_path)
            for target_path in target_paths:
                wrapper = _make_prim_reference_wrapper(asset_path, layer_path, target_path, prim_type, source_text)
                self._queue_native_load(self._write_runtime_layer("clone_ref", wrapper))

    def open_usd(self, usd_file_path: str) -> None:
        self.open_usd_async(usd_file_path).wait()

    def open_usd_async(self, usd_file_path: str) -> Operation:
        handle = self._next_usd_handle
        abs_path = os.path.abspath(usd_file_path)
        try:
            self._clear_composed_stage(clear_handles=True, clear_attrs=True, clear_native=True)
            self._next_usd_handle += 1
            self._usd_handles[handle] = abs_path
            self._usd_prefixes[handle] = None
            self._rebuild_composed_stage(load_native=False)
            return Operation(self, handle=True, operation_name=f"open_usd({usd_file_path})")
        except Exception as exc:
            self._clear_composed_stage(clear_handles=True, clear_attrs=True, clear_native=True)
            return Operation(self, operation_name=f"open_usd({usd_file_path})", error=exc)

    def open_usd_from_string(self, root_layer_content: str) -> None:
        self.open_usd_from_string_async(root_layer_content).wait()

    def open_usd_from_string_async(self, root_layer_content: str) -> Operation:
        if not root_layer_content or not root_layer_content.strip():
            raise ValueError("root_layer_content cannot be empty")
        handle = self._next_usd_handle
        try:
            self._clear_composed_stage(clear_handles=True, clear_attrs=True, clear_native=True)
            self._next_usd_handle += 1
            self._usd_handles[handle] = "<inline>"
            self._usd_prefixes[handle] = None
            self._usd_layers[handle] = (root_layer_content, None)
            self._rebuild_composed_stage(load_native=False)
            return Operation(self, handle=True, operation_name="open_usd_from_string")
        except Exception as exc:
            self._clear_composed_stage(clear_handles=True, clear_attrs=True, clear_native=True)
            return Operation(self, operation_name="open_usd_from_string", error=exc)

    def add_usd_reference(self, layer_file: str, prefix_path: str) -> Any:
        return self.add_usd_reference_async(layer_file, prefix_path).wait()

    def add_usd_reference_async(self, layer_file: str, prefix_path: str) -> Operation:
        handle = self._next_usd_handle
        abs_path = os.path.abspath(layer_file)
        try:
            self._next_usd_handle += 1
            self._usd_handles[handle] = abs_path
            self._usd_prefixes[handle] = prefix_path
            self._removable_usd_handles.add(handle)
            self._rebuild_composed_stage(load_native=False)
            return Operation(self, handle=handle, operation_name=f"add_usd_reference({layer_file})")
        except Exception as exc:
            self._usd_handles.pop(handle, None)
            self._usd_prefixes.pop(handle, None)
            self._removable_usd_handles.discard(handle)
            return Operation(self, operation_name=f"add_usd_reference({layer_file})", error=exc)

    def add_usd_reference_from_string(self, layer_content: str, prefix_path: str) -> Any:
        return self.add_usd_reference_from_string_async(layer_content, prefix_path).wait()

    def add_usd_reference_from_string_async(self, layer_content: str, prefix_path: str) -> Operation:
        if not layer_content or not layer_content.strip():
            raise ValueError("layer_content cannot be empty")
        handle = self._next_usd_handle
        try:
            self._next_usd_handle += 1
            self._usd_handles[handle] = "<inline>"
            self._usd_prefixes[handle] = prefix_path
            self._usd_layers[handle] = (layer_content, prefix_path)
            self._removable_usd_handles.add(handle)
            self._rebuild_composed_stage(load_native=False)
            return Operation(self, handle=handle, operation_name="add_usd_reference_from_string")
        except Exception as exc:
            self._usd_handles.pop(handle, None)
            self._usd_prefixes.pop(handle, None)
            self._usd_layers.pop(handle, None)
            self._removable_usd_handles.discard(handle)
            return Operation(self, operation_name="add_usd_reference_from_string", error=exc)

    def remove_usd(self, usd_handle: Any) -> None:
        self.remove_usd_async(usd_handle).wait()

    def remove_usd_async(self, usd_handle: Any) -> Operation:
        try:
            handle = int(usd_handle)
        except Exception as exc:
            raise TypeError("usd_handle must be a handle returned from add_usd_reference()") from exc
        if handle not in self._removable_usd_handles:
            raise RuntimeError(f"Failed to enqueue remove_usd: invalid or already removed USD handle: {usd_handle!r}")
        try:
            self._removable_usd_handles.discard(handle)
            self._usd_handles.pop(handle, None)
            self._usd_prefixes.pop(handle, None)
            self._usd_layers.pop(handle, None)
            self._rebuild_composed_stage(load_native=False)
            return Operation(self, operation_name="remove_usd")
        except Exception as exc:
            return Operation(self, operation_name="remove_usd", error=exc)

    def update_from_usd_time(self, usd_time: float) -> None:
        self.update_from_usd_time_async(usd_time).wait()

    def update_from_usd_time_async(self, usd_time: float) -> Operation:
        previous_usd_time = float(self._usd_time)
        self._usd_time = float(usd_time)
        self._apply_usd_time_samples()
        self._reapply_attribute_overrides()
        if self._nu is not None and hasattr(self._nu, "set_current_time"):
            self._nu.set_current_time(float(usd_time))
        if (
            self._nu is not None
            and float(self._usd_time) != previous_usd_time
            and (self._usd_handles or self._usd_layers or self._clone_requests)
        ):
            self._native_stage_dirty = True
        return Operation(self, operation_name="update_from_usd_time")

    def clone_usd(self, source_path: str, target_paths: list[str]) -> None:
        self.clone_usd_async(source_path, target_paths).wait()

    def clone_usd_async(self, source_path: str, target_paths: list[str]) -> Operation:
        if not target_paths:
            raise ValueError("At least one target path is required")
        try:
            source = _normalize_prefix(source_path)
            targets = tuple(_normalize_prefix(path) for path in target_paths)
            if source is None:
                raise ValueError("source_path must be an absolute prim path")
            if any(path is None for path in targets):
                raise ValueError("At least one absolute target path is required")
            if not any(path == source or path.startswith(source + "/") for path in self._prim_index):
                raise ValueError(f"clone_usd source does not exist in the indexed stage: {source}")
            if self._resolve_clone_source(source) is None:
                raise NotImplementedError(
                    "clone_usd currently requires a source prim backed by a loaded file or inline layer"
                )
            self._clone_requests.append((source, tuple(str(path) for path in targets if path is not None)))
            self._rebuild_composed_stage(load_native=False)
            return Operation(self, operation_name="clone_usd")
        except Exception as exc:
            return Operation(self, operation_name="clone_usd", error=exc)

    def enqueue_pick_query(
        self,
        render_product_path: str,
        left: int,
        top: int,
        right: int,
        bottom: int,
        *,
        flags: int = 0,
    ) -> None:
        self.enqueue_pick_query_async(render_product_path, left, top, right, bottom, flags=flags).wait()

    def enqueue_pick_query_async(
        self,
        render_product_path: str,
        left: int,
        top: int,
        right: int,
        bottom: int,
        *,
        flags: int = 0,
    ) -> Operation:
        del render_product_path, left, top, right, bottom, flags
        return Operation(
            self,
            operation_name="enqueue_pick_query",
            error=NotImplementedError("nanousd ovrtx compatibility does not yet implement OVRTX picking"),
        )

    def set_selection_group_styles(self, styles: dict[int, SelectionGroupStyle]) -> None:
        self.set_selection_group_styles_async(styles).wait()

    def set_selection_group_styles_async(self, styles: dict[int, SelectionGroupStyle]) -> Operation:
        normalized: dict[int, SelectionGroupStyle] = {}
        for group_id, style in styles.items():
            gid = int(group_id)
            if not 0 <= gid <= 255:
                raise ValueError(f"selection group id {gid} out of range; expected 0..255")
            if len(style.outline_color) != 4 or len(style.fill_color) != 4:
                raise ValueError(
                    f"selection group {gid}: outline_color and fill_color must each have 4 RGBA components"
                )
            normalized[gid] = SelectionGroupStyle(
                outline_color=tuple(float(c) for c in style.outline_color),
                fill_color=tuple(float(c) for c in style.fill_color),
            )
        self._selection_group_styles.update(normalized)
        return Operation(self, handle=True, operation_name=f"set_selection_group_styles({len(normalized)} group(s))")

    def resolve_prim_path_id(self, prim_path_id: int) -> str:
        return self._paths_by_prim_id.get(int(prim_path_id), "")

    def reset(self, time: float = 0.0) -> None:
        self.reset_async(time).wait()

    def reset_async(self, time: float = 0.0) -> Operation:
        self._time = float(time)
        return Operation(self, operation_name="reset")

    def reset_stage(self) -> None:
        self.reset_stage_async().wait()

    def reset_stage_async(self) -> Operation:
        self._clear_composed_stage(clear_handles=True, clear_attrs=True, clear_native=True)
        self._time = 0.0
        self._usd_time = 0.0
        self._native_stage_dirty = False
        return Operation(self, operation_name="reset_stage")

    def _is_camera_path(self, path: str) -> bool:
        prim_type = getattr(self._prim_index.get(path, {}).get("__type__"), "value", None)
        if prim_type == "Camera":
            return True
        if path == self._default_camera_path:
            return True
        if any(spec.camera_path == path for spec in self._render_product_specs.values()):
            return True
        return "camera" in path.rsplit("/", 1)[-1].lower()

    def _resolve_camera(
        self,
        camera_path: Optional[str] = None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        candidates = [camera_path, self._default_camera_path]
        candidates.extend(self._camera_matrices.keys())
        for path in candidates:
            if path and path in self._camera_matrices:
                return _matrix_to_camera(self._camera_matrices[path])
        if self._camera_matrix is not None:
            return _matrix_to_camera(self._camera_matrix)
        bounds = None
        if self._nu is not None and hasattr(self._nu, "get_scene_bounds"):
            try:
                bounds = self._nu.get_scene_bounds()
            except Exception:
                bounds = None
        return _default_camera_from_bounds(bounds)

    def _attribute_scalar_for_projection(
        self,
        prim_path: str,
        attribute_name: str,
        fallback: float,
        *,
        use_overrides: bool,
    ) -> float:
        value = self._attrs.get((prim_path, attribute_name)) if use_overrides else None
        if value is None:
            info = self._prim_index.get(prim_path, {}).get(attribute_name)
            value = getattr(info, "value", info)
        parsed = _first_float_value(value)
        return float(parsed) if parsed is not None else float(fallback)

    def _refresh_camera_projection(self, prim_path: str, *, use_overrides: bool = True) -> None:
        focal = self._attribute_scalar_for_projection(prim_path, "focalLength", 50.0, use_overrides=use_overrides)
        vertical = self._attribute_scalar_for_projection(
            prim_path,
            "verticalAperture",
            _USD_DEFAULT_VERTICAL_APERTURE,
            use_overrides=use_overrides,
        )
        horizontal = self._attribute_scalar_for_projection(
            prim_path,
            "horizontalAperture",
            _USD_DEFAULT_HORIZONTAL_APERTURE,
            use_overrides=use_overrides,
        )
        horizontal_offset = self._attribute_scalar_for_projection(
            prim_path,
            "horizontalApertureOffset",
            0.0,
            use_overrides=use_overrides,
        )
        vertical_offset = self._attribute_scalar_for_projection(
            prim_path,
            "verticalApertureOffset",
            0.0,
            use_overrides=use_overrides,
        )
        if focal <= 0.0 or vertical <= 0.0:
            return
        fov = _math.degrees(2.0 * _math.atan(vertical / (2.0 * focal)))
        self._camera_fovs[prim_path] = fov
        self._camera_fov = fov
        self._camera_apertures[prim_path] = (horizontal, vertical)
        self._camera_aperture_offsets[prim_path] = (horizontal_offset, vertical_offset)
        if horizontal > 0.0:
            self._camera_aperture_aspects[prim_path] = horizontal / vertical

    def _camera_fov_for_path(self, camera_path: Optional[str]) -> float:
        if camera_path and camera_path in self._camera_fovs:
            return self._camera_fovs[camera_path]
        if self._default_camera_path and self._default_camera_path in self._camera_fovs:
            return self._camera_fovs[self._default_camera_path]
        return float(self._camera_fov)

    def _camera_aperture_aspect_for_path(self, camera_path: Optional[str]) -> float:
        if camera_path and camera_path in self._camera_aperture_aspects:
            return self._camera_aperture_aspects[camera_path]
        if self._default_camera_path and self._default_camera_path in self._camera_aperture_aspects:
            return self._camera_aperture_aspects[self._default_camera_path]
        return float(_USD_DEFAULT_CAMERA_APERTURE_ASPECT)

    def _camera_aperture_for_path(self, camera_path: Optional[str]) -> tuple[float, float]:
        if camera_path and camera_path in self._camera_apertures:
            return self._camera_apertures[camera_path]
        if self._default_camera_path and self._default_camera_path in self._camera_apertures:
            return self._camera_apertures[self._default_camera_path]
        return (_USD_DEFAULT_HORIZONTAL_APERTURE, _USD_DEFAULT_VERTICAL_APERTURE)

    def _camera_aperture_offset_for_path(self, camera_path: Optional[str]) -> tuple[float, float]:
        if camera_path and camera_path in self._camera_aperture_offsets:
            return self._camera_aperture_offsets[camera_path]
        if self._default_camera_path and self._default_camera_path in self._camera_aperture_offsets:
            return self._camera_aperture_offsets[self._default_camera_path]
        return (0.0, 0.0)

    def _set_camera_clip_range(self, prim_path: str, value: Any) -> None:
        cr = np.asarray(value, dtype=np.float32).ravel()
        if cr.size < 2:
            return
        near_clip = float(cr[0])
        far_clip = float(cr[1])
        if not (np.isfinite(near_clip) and np.isfinite(far_clip)):
            return
        self._camera_clip_ranges[prim_path] = (near_clip, far_clip)
        self._near_clip = near_clip
        self._far_clip = far_clip

    def _camera_clip_range_for_path(self, camera_path: Optional[str]) -> tuple[float, float]:
        if camera_path and camera_path in self._camera_clip_ranges:
            return self._camera_clip_ranges[camera_path]
        if camera_path:
            return (0.1, 10000.0)
        if self._default_camera_path and self._default_camera_path in self._camera_clip_ranges:
            return self._camera_clip_ranges[self._default_camera_path]
        return (float(self._near_clip), float(self._far_clip))

    def _effective_fov_for_render_product(
        self,
        spec: _RenderProductSpec,
        width: int,
        height: int,
        camera_path: Optional[str] = None,
    ) -> float:
        return self._effective_projection_for_render_product(spec, width, height, camera_path)[0]

    def _effective_projection_for_render_product(
        self,
        spec: _RenderProductSpec,
        width: int,
        height: int,
        camera_path: Optional[str] = None,
    ) -> tuple[float, tuple[float, float]]:
        resolved_camera_path = camera_path or spec.camera_path
        base_fov = self._camera_fov_for_path(camera_path or spec.camera_path)
        try:
            base_tan = _math.tan(_math.radians(float(base_fov)) * 0.5)
        except Exception:
            return float(base_fov), (0.0, 0.0)
        if not np.isfinite(base_tan) or base_tan <= 1e-8:
            return float(base_fov), (0.0, 0.0)
        raster_aspect = max(float(width), 1.0) / max(float(height), 1.0)
        pixel_aspect = float(spec.pixel_aspect_ratio) if spec.pixel_aspect_ratio is not None else 1.0
        if not np.isfinite(pixel_aspect) or pixel_aspect <= 1e-8:
            pixel_aspect = 1.0
        image_aspect = max(raster_aspect * pixel_aspect, 1e-8)
        camera_aspect = max(self._camera_aperture_aspect_for_path(resolved_camera_path), 1e-8)
        policy = re.sub(r"[^a-z0-9]+", "", (spec.aspect_ratio_conform_policy or "expandAperture").lower())
        adjusted_tan = base_tan
        if policy == "adjustapertureheight":
            adjusted_tan = base_tan * camera_aspect / image_aspect
        elif policy == "adjustpixelaspectratio":
            adjusted_tan = base_tan * camera_aspect / max(raster_aspect, 1e-8)
        elif policy == "expandaperture":
            if image_aspect < camera_aspect:
                adjusted_tan = base_tan * camera_aspect / image_aspect
        elif policy == "cropaperture":
            if image_aspect > camera_aspect:
                adjusted_tan = base_tan * camera_aspect / image_aspect
        elif policy == "adjustaperturewidth":
            adjusted_tan = base_tan
        if not np.isfinite(adjusted_tan) or adjusted_tan <= 1e-8:
            return float(base_fov), (0.0, 0.0)

        horizontal, vertical = self._camera_aperture_for_path(resolved_camera_path)
        horizontal_offset, vertical_offset = self._camera_aperture_offset_for_path(resolved_camera_path)
        shift_x = 0.0
        if horizontal > 0.0:
            horizontal_tan = base_tan * camera_aspect
            effective_horizontal_tan = adjusted_tan * raster_aspect
            if np.isfinite(horizontal_tan) and np.isfinite(effective_horizontal_tan) and effective_horizontal_tan > 1e-8:
                shift_x = (2.0 * horizontal_offset / horizontal) * (horizontal_tan / effective_horizontal_tan)
        shift_y = 0.0
        if vertical > 0.0:
            shift_y = (2.0 * vertical_offset / vertical) * (base_tan / adjusted_tan)
        shift = (
            float(shift_x) if np.isfinite(shift_x) else 0.0,
            float(shift_y) if np.isfinite(shift_y) else 0.0,
        )
        return float(min(_math.degrees(2.0 * _math.atan(adjusted_tan)), 179.0)), shift

    def _sync_camera(
        self,
        camera_path: Optional[str] = None,
        fov_degrees: Optional[float] = None,
        projection_shift: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        if self._nu is None:
            return _default_camera_from_bounds(None)
        eye, target, up = self._resolve_camera(camera_path)
        fov = float(fov_degrees if fov_degrees is not None else self._camera_fov_for_path(camera_path))
        near_clip, far_clip = self._camera_clip_range_for_path(camera_path)
        if hasattr(self._nu, "set_camera_explicit_window"):
            self._nu.set_camera_explicit_window(eye, target, up, fov, near_clip, far_clip, projection_shift)
        elif hasattr(self._nu, "set_camera_explicit"):
            self._nu.set_camera_explicit(eye, target, up, fov, near_clip, far_clip)
        else:
            self._nu.set_camera(eye, target, fov, near_clip, far_clip)
        self._last_camera = (eye, target, up)
        return self._last_camera

    def _native_render_mode(self, render_mode: Optional[str], nu: Any) -> int:
        token = re.sub(r"[^a-z0-9]+", "", (render_mode or "rt").strip().lower())
        if token in ("raster", "rasteronly", "hydraraster"):
            mode = self._NU_RENDER_RASTER
        elif token in ("shadow", "rastershadow", "hydrashadow"):
            mode = self._NU_RENDER_SHADOW
        else:
            mode = self._NU_RENDER_RT
        if self._backend == "opengl":
            mode = self._NU_RENDER_RASTER
        elif not getattr(nu, "rt_available", True) and mode == self._NU_RENDER_RT:
            mode = self._NU_RENDER_RASTER
        return mode

    def _native_fast_mode(self, spec: _RenderProductSpec) -> bool:
        token = re.sub(r"[^a-z0-9]+", "", (spec.render_mode or "").strip().lower())
        return token == "minimal"

    def _included_purposes_key_for_product(self, spec: _RenderProductSpec) -> Optional[tuple[str, ...]]:
        if spec.included_purposes is None:
            return None
        return tuple(str(purpose) for purpose in spec.included_purposes)

    def _material_binding_purposes_key_for_product(self, spec: _RenderProductSpec) -> Optional[tuple[str, ...]]:
        if spec.material_binding_purposes is None:
            return None
        return tuple(str(purpose) for purpose in spec.material_binding_purposes)

    def _camera_paths_for_render_product(self, spec: _RenderProductSpec) -> tuple[str, ...]:
        out: list[str] = []
        for camera_path in spec.cameras_or_default():
            path = str(camera_path or self._default_camera_path or "")
            if path and path not in out:
                out.append(path)
        return tuple(out)

    def _camera_shutter_open_for_path(self, camera_path: str) -> float:
        info = self._prim_index.get(camera_path, {}).get("shutter:open")
        parsed = _first_float_value(getattr(info, "value", None))
        return float(parsed) if parsed is not None else 0.0

    def _camera_render_overrides_key_for_product(self, spec: _RenderProductSpec) -> Optional[tuple[Any, ...]]:
        motion_mode = ""
        if spec.disable_motion_blur:
            motion_mode = "disable"
        elif spec.instantaneous_shutter:
            motion_mode = "instantaneous"
        disable_depth_of_field = bool(spec.disable_depth_of_field)
        if not motion_mode and not disable_depth_of_field:
            return None
        camera_paths = self._camera_paths_for_render_product(spec)
        if not camera_paths:
            return None
        shutter_opens: tuple[float, ...] = ()
        if motion_mode == "instantaneous":
            shutter_opens = tuple(self._camera_shutter_open_for_path(path) for path in camera_paths)
        return (camera_paths, motion_mode, disable_depth_of_field, shutter_opens)

    def _native_runtime_attribute_overrides_key_for_product(
        self,
        spec: _RenderProductSpec,
    ) -> Optional[tuple[tuple[str, str, str], ...]]:
        records: list[tuple[str, str, str]] = []

        def add_attr(prim_path: str, attribute_name: str) -> bool:
            key = (prim_path, attribute_name)
            if key not in self._attrs:
                return False
            attr_line = _native_runtime_attribute_line(attribute_name, self._attrs[key])
            if attr_line is None:
                return False
            records.append((prim_path, attribute_name, attr_line))
            return True

        for camera_path in self._camera_paths_for_render_product(spec):
            for attribute_name in sorted(_NATIVE_OVRTX_CAMERA_EXPOSURE_ATTRIBUTES):
                add_attr(camera_path, attribute_name)

        product_attributes = sorted(_NATIVE_OVRTX_RENDER_EXPOSURE_ATTRIBUTES)
        selected_product_attributes: set[str] = set()
        for attribute_name in product_attributes:
            if add_attr(spec.path, attribute_name):
                selected_product_attributes.add(attribute_name)

        for settings_path, settings_spec in sorted(self._render_settings_specs.items()):
            if spec.path not in (settings_spec.product_paths or []):
                continue
            for attribute_name in product_attributes:
                if attribute_name in selected_product_attributes:
                    continue
                if self._product_has_own_attribute(spec.path, (attribute_name,)):
                    continue
                if add_attr(settings_path, attribute_name):
                    selected_product_attributes.add(attribute_name)

        return tuple(records) or None

    def _ensure_native_stage_for_product(self, spec: _RenderProductSpec) -> None:
        included_key = self._included_purposes_key_for_product(spec)
        material_key = self._material_binding_purposes_key_for_product(spec)
        camera_override_key = self._camera_render_overrides_key_for_product(spec)
        runtime_override_key = self._native_runtime_attribute_overrides_key_for_product(spec)
        if (
            self._native_stage_dirty
            or self._nu is None
            or self._active_included_purposes != included_key
            or self._active_material_binding_purposes != material_key
            or self._active_camera_render_overrides != camera_override_key
            or self._active_native_runtime_attribute_overrides != runtime_override_key
        ):
            self._active_included_purposes = included_key
            self._active_material_binding_purposes = material_key
            self._active_camera_render_overrides = camera_override_key
            self._active_native_runtime_attribute_overrides = runtime_override_key
            self._rebuild_composed_stage(load_native=True)

    def _fetch_native_aovs(
        self,
        nu: Any,
        names: Iterable[str],
        width: int,
        height: int,
        mode: int,
        camera: Optional[
            tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        ],
        fov_degrees: float,
        clip_range: Optional[tuple[float, float]] = None,
        projection_shift: tuple[float, float] = (0.0, 0.0),
    ) -> dict[str, np.ndarray]:
        requested = set(names)
        need_depth = bool(requested & {"DepthSD", "DistanceToCameraSD", "DistanceToImagePlaneSD", "Camera3dPositionSD"})
        need_normals = "NormalSD" in requested
        need_segmentation = bool(requested & {"SemanticSegmentation", "SemanticSegmentationSD", "InstanceSegmentationSD"})
        if not (need_depth or need_normals or need_segmentation):
            return {}
        if self._backend == "opengl" or mode != self._NU_RENDER_RT or not getattr(nu, "rt_available", True):
            return {}
        if not hasattr(nu, "render_tiled"):
            return {}

        can_depth = need_depth and hasattr(nu, "set_depth_enabled") and hasattr(nu, "fetch_depth_tiled")
        can_normals = need_normals and hasattr(nu, "set_normals_enabled") and hasattr(nu, "fetch_normals_tiled")
        can_segmentation = (
            need_segmentation
            and hasattr(nu, "set_segmentation_enabled")
            and hasattr(nu, "fetch_segmentation_tiled")
        )
        if not (can_depth or can_normals or can_segmentation):
            return {}

        toggled_fast_mode = False
        try:
            if hasattr(nu, "set_fast_mode"):
                nu.set_fast_mode(True)
                toggled_fast_mode = True
            if can_depth:
                nu.set_depth_enabled(True)
            if can_normals:
                nu.set_normals_enabled(True)
            if can_segmentation:
                nu.set_segmentation_enabled(True)
            eye, target, up = camera if camera is not None else self._resolve_camera(None)
            near_clip, far_clip = clip_range or self._camera_clip_range_for_path(None)
            vp = _make_vp_inv(
                eye,
                target,
                up,
                fov_degrees,
                width,
                height,
                near_clip,
                far_clip,
                projection_shift,
            )
            nu.render_tiled(vp.reshape(1, 32), num_cameras=1, tile_w=width, tile_h=height, mode=mode)
            out: dict[str, np.ndarray] = {}
            if can_depth:
                depth = np.asarray(nu.fetch_depth_tiled(num_cameras=1, tile_w=width, tile_h=height)[0], dtype=np.float32)
                if depth.shape == (height, width) and np.any(np.isfinite(depth) & (depth > 0.0)):
                    out["depth"] = np.ascontiguousarray(depth)
            if can_normals:
                normals = np.asarray(nu.fetch_normals_tiled(num_cameras=1, tile_w=width, tile_h=height)[0], dtype=np.float32)
                if normals.shape == (height, width, 3) and np.any(np.isfinite(normals) & (np.abs(normals) > 1e-6)):
                    out["normals"] = np.ascontiguousarray(normals)
            if can_segmentation:
                segmentation = np.asarray(
                    nu.fetch_segmentation_tiled(num_cameras=1, tile_w=width, tile_h=height)[0],
                    dtype=np.uint32,
                )
                if segmentation.shape == (height, width):
                    out["segmentation"] = np.ascontiguousarray(segmentation)
            return out
        except Exception:
            return {}
        finally:
            if toggled_fast_mode:
                try:
                    nu.set_fast_mode(False)
                except Exception:
                    pass

    def _render_tiled_product_arrays(
        self,
        nu: Any,
        spec: _RenderProductSpec,
        names: list[str],
        width: int,
        height: int,
        mode: int,
        fast_mode: bool = False,
    ) -> Optional[
        tuple[
            np.ndarray,
            dict[str, np.ndarray],
            Optional[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
        ]
    ]:
        camera_paths = spec.cameras_or_default()
        if len(camera_paths) <= 1:
            return None
        if self._backend == "opengl" or mode != self._NU_RENDER_RT or not getattr(nu, "rt_available", True):
            return None
        if not hasattr(nu, "render_tiled") or not hasattr(nu, "fetch_pixels_tiled"):
            return None

        num_cameras = len(camera_paths)
        _, _, tile_w, tile_h = _camera_tile_layout(num_cameras, width, height)
        cameras = [self._resolve_camera(path) for path in camera_paths]
        vps = np.stack(
            [
                _make_vp_inv(eye, target, up, fov, tile_w, tile_h, near_clip, far_clip, shift)
                for camera_path, (eye, target, up) in zip(camera_paths, cameras)
                for fov, shift in [self._effective_projection_for_render_product(spec, tile_w, tile_h, camera_path)]
                for near_clip, far_clip in [self._camera_clip_range_for_path(camera_path)]
            ]
        )
        requested = set(names)
        need_depth = bool(requested & {"DepthSD", "DistanceToCameraSD", "DistanceToImagePlaneSD", "Camera3dPositionSD"})
        need_normals = "NormalSD" in requested
        need_segmentation = bool(requested & {"SemanticSegmentation", "SemanticSegmentationSD", "InstanceSegmentationSD"})
        native: dict[str, np.ndarray] = {}
        toggled_fast_mode = False
        try:
            if fast_mode and hasattr(nu, "set_fast_mode"):
                nu.set_fast_mode(True)
                toggled_fast_mode = True
            if hasattr(nu, "set_tiled_srgb"):
                nu.set_tiled_srgb(True)
            if need_depth and hasattr(nu, "set_depth_enabled"):
                nu.set_depth_enabled(True)
            if need_normals and hasattr(nu, "set_normals_enabled"):
                nu.set_normals_enabled(True)
            if need_segmentation and hasattr(nu, "set_segmentation_enabled"):
                nu.set_segmentation_enabled(True)

            nu.render_tiled(vps, num_cameras=num_cameras, tile_w=tile_w, tile_h=tile_h, mode=mode)
            ldr = _assemble_camera_tiles(
                nu.fetch_pixels_tiled(num_cameras=num_cameras, tile_w=tile_w, tile_h=tile_h),
                width,
                height,
            )
            if need_depth and hasattr(nu, "fetch_depth_tiled"):
                depth_tiles = nu.fetch_depth_tiled(num_cameras=num_cameras, tile_w=tile_w, tile_h=tile_h)
                native["depth"] = _assemble_camera_tiles(depth_tiles, width, height)
            if need_normals and hasattr(nu, "fetch_normals_tiled"):
                normal_tiles = nu.fetch_normals_tiled(num_cameras=num_cameras, tile_w=tile_w, tile_h=tile_h)
                native["normals"] = _assemble_camera_tiles(normal_tiles, width, height)
            if need_segmentation and hasattr(nu, "fetch_segmentation_tiled"):
                segmentation_tiles = nu.fetch_segmentation_tiled(num_cameras=num_cameras, tile_w=tile_w, tile_h=tile_h)
                native["segmentation"] = _assemble_camera_tiles(segmentation_tiles, width, height)
            return np.ascontiguousarray(ldr), native, cameras[0] if cameras else None
        except Exception:
            return None
        finally:
            if toggled_fast_mode:
                try:
                    nu.set_fast_mode(False)
                except Exception:
                    pass

    def _render_serial_tiled_product_arrays(
        self,
        nu: Any,
        spec: _RenderProductSpec,
        names: list[str],
        width: int,
        height: int,
        mode: int,
        fast_mode: bool = False,
    ) -> Optional[
        tuple[
            np.ndarray,
            dict[str, np.ndarray],
            Optional[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
        ]
    ]:
        camera_paths = spec.cameras_or_default()
        if len(camera_paths) <= 1:
            return None
        if not hasattr(nu, "render") or not hasattr(nu, "fetch_pixels"):
            return None

        num_cameras = len(camera_paths)
        _, _, tile_w, tile_h = _camera_tile_layout(num_cameras, width, height)
        cameras: list[
            tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        ] = []
        ldr_tiles: list[np.ndarray] = []
        native_tiles_by_camera: list[dict[str, np.ndarray]] = []
        try:
            for camera_path in camera_paths:
                if hasattr(nu, "set_render_size"):
                    nu.set_render_size(tile_w, tile_h)
                tile_fov, tile_shift = self._effective_projection_for_render_product(spec, tile_w, tile_h, camera_path)
                camera = self._sync_camera(camera_path, tile_fov, tile_shift)
                cameras.append(camera)
                toggled_fast_mode = False
                try:
                    if fast_mode and hasattr(nu, "set_fast_mode"):
                        nu.set_fast_mode(True)
                        toggled_fast_mode = True
                    nu.render(mode)
                    ldr_tiles.append(_fit_camera_tile(nu.fetch_pixels(), tile_w, tile_h))
                finally:
                    if toggled_fast_mode:
                        try:
                            nu.set_fast_mode(False)
                        except Exception:
                            pass
                tile_clip_range = self._camera_clip_range_for_path(camera_path)
                tile_native = self._fetch_native_aovs(
                    nu,
                    names,
                    tile_w,
                    tile_h,
                    mode,
                    camera,
                    tile_fov,
                    tile_clip_range,
                    tile_shift,
                )
                native_tiles_by_camera.append(
                    {key: _fit_camera_tile(arr, tile_w, tile_h) for key, arr in tile_native.items()}
                )
            if len(ldr_tiles) != num_cameras:
                return None
            common_native_keys: set[str] = set()
            if native_tiles_by_camera:
                common_native_keys = set(native_tiles_by_camera[0])
                for tile_native in native_tiles_by_camera[1:]:
                    common_native_keys &= set(tile_native)
            native = {
                key: _assemble_camera_tiles(
                    np.stack([tile_native[key] for tile_native in native_tiles_by_camera]),
                    width,
                    height,
                )
                for key in common_native_keys
            }
            ldr = _assemble_camera_tiles(np.stack(ldr_tiles), width, height)
            return np.ascontiguousarray(ldr), native, cameras[0] if cameras else None
        except Exception:
            return None
        finally:
            if hasattr(nu, "set_render_size"):
                try:
                    nu.set_render_size(width, height)
                except Exception:
                    pass
            if camera_paths:
                try:
                    fov, shift = self._effective_projection_for_render_product(spec, width, height, camera_paths[0])
                    self._sync_camera(
                        camera_paths[0],
                        fov,
                        shift,
                    )
                except Exception:
                    pass

    def step(self, render_products: set[str], delta_time: float) -> RenderProductSetOutputs:
        return self.step_async(render_products, delta_time).wait().fetch()

    def _validate_step_render_products(self, products: set[str]) -> None:
        invalid: list[str] = []
        settings: list[str] = []
        for path in sorted(products):
            prim_type = getattr(self._prim_index.get(path, {}).get("__type__"), "value", None)
            if prim_type == "RenderProduct":
                continue
            if prim_type == "RenderSettings":
                settings.append(path)
            else:
                invalid.append(path)
        if settings:
            raise ValueError(
                "step() expects paths to RenderProduct prims, not RenderSettings prims: "
                f"{settings}. Read the RenderSettings 'products' relationship and pass those product paths."
            )
        if invalid:
            raise ValueError(f"step() requested unknown or non-RenderProduct paths: {invalid}")

    def step_async(self, render_products: set[str], delta_time: float) -> Operation:
        if delta_time < 0:
            raise ValueError("delta_time must be non-negative")

        products = {str(p) for p in render_products if p and str(p).strip()}
        if not products:
            raise ValueError("At least one valid render product is required to step the renderer")
        self._validate_step_render_products(products)

        def _fetch(_: Optional[int] = None) -> RenderProductSetOutputs:
            frames_by_product: dict[str, ProductOutput] = {}
            for product_path in products:
                spec = self._render_product_specs.get(
                    product_path,
                    _RenderProductSpec(path=product_path, render_vars=["LdrColor"]),
                )
                self._ensure_native_stage_for_product(spec)
                nu = self._ensure_renderer()
                spec = self._render_product_specs.get(product_path, spec)
                mode = self._native_render_mode(spec.render_mode, nu)
                if mode != self._NU_RENDER_RASTER and hasattr(nu, "build_accel"):
                    try:
                        nu.build_accel()
                    except Exception:
                        pass
                fast_mode = self._native_fast_mode(spec)
                width = int(spec.width or self._width)
                height = int(spec.height or self._height)
                if width != self._width or height != self._height:
                    self._width, self._height = width, height
                    if hasattr(nu, "set_render_size"):
                        nu.set_render_size(width, height)
                names = spec.vars_or_default()
                tiled = self._render_tiled_product_arrays(nu, spec, names, width, height, mode, fast_mode)
                if tiled is None:
                    tiled = self._render_serial_tiled_product_arrays(nu, spec, names, width, height, mode, fast_mode)
                if tiled is not None:
                    pixels, native, camera = tiled
                else:
                    fov_degrees, projection_shift = self._effective_projection_for_render_product(
                        spec,
                        width,
                        height,
                        spec.camera_path,
                    )
                    camera = self._sync_camera(spec.camera_path, fov_degrees, projection_shift)
                    toggled_fast_mode = False
                    try:
                        if fast_mode and hasattr(nu, "set_fast_mode"):
                            nu.set_fast_mode(True)
                            toggled_fast_mode = True
                        nu.render(mode)
                        pixels = np.ascontiguousarray(nu.fetch_pixels())
                    finally:
                        if toggled_fast_mode:
                            try:
                                nu.set_fast_mode(False)
                            except Exception:
                                pass
                    native = self._fetch_native_aovs(
                        nu,
                        names,
                        width,
                        height,
                        mode,
                        camera,
                        fov_degrees,
                        self._camera_clip_range_for_path(spec.camera_path),
                        projection_shift,
                    )
                output_fov_degrees, output_projection_shift = self._effective_projection_for_render_product(
                    spec,
                    width,
                    height,
                    spec.camera_path,
                )
                var_arrays = _render_var_arrays(
                    pixels,
                    names,
                    native_depth=native.get("depth"),
                    native_normals=native.get("normals"),
                    native_segmentation=native.get("segmentation"),
                    camera=camera,
                    fov_degrees=output_fov_degrees,
                    projection_shift=output_projection_shift,
                )
                var_arrays = _apply_data_window_ndc(var_arrays, spec.data_window_ndc)
                frame = FrameOutput(
                    start_time=self._time,
                    end_time=self._time + float(delta_time),
                    render_vars={name: RenderVarOutput(name, arr, self) for name, arr in var_arrays.items()},
                )
                frames_by_product[product_path] = ProductOutput(name=product_path, frames=[frame])
            self._time += float(delta_time)
            return RenderProductSetOutputs(
                destroy_fn=lambda: None,
                products=frames_by_product,
            )

        return Operation(self, operation_name=f"step(dt={delta_time})", handle=object(), fetch_fn=_fetch)

    def query_prims(
        self,
        require_all: Optional[list[tuple[int, str]]] = None,
        require_any: Optional[list[tuple[int, str]]] = None,
        exclude: Optional[list[tuple[int, str]]] = None,
        attribute_filter_mode: int = AttributeFilterMode.NONE,
        attribute_names: Optional[list[str]] = None,
    ) -> dict[str, dict[str, AttributeInfo]]:
        return self.query_prims_async(
            require_all, require_any, exclude, attribute_filter_mode, attribute_names
        ).wait().fetch()

    def query_prims_async(
        self,
        require_all: Optional[list[tuple[int, str]]] = None,
        require_any: Optional[list[tuple[int, str]]] = None,
        exclude: Optional[list[tuple[int, str]]] = None,
        attribute_filter_mode: int = AttributeFilterMode.NONE,
        attribute_names: Optional[list[str]] = None,
    ) -> Operation:
        def _matches(path: str, attrs: dict[str, AttributeInfo], flt: tuple[int, str]) -> bool:
            kind, name = flt
            if int(kind) == int(FilterKind.PRIM_TYPE):
                return getattr(attrs.get("__type__"), "value", None) == name
            if int(kind) == int(FilterKind.HAS_ATTRIBUTE):
                return name in attrs
            return False

        def _fetch(_: Optional[int] = None):
            result = {}
            for path, attrs in self._prim_index.items():
                if require_all and not all(_matches(path, attrs, f) for f in require_all):
                    continue
                if require_any and not any(_matches(path, attrs, f) for f in require_any):
                    continue
                if exclude and any(_matches(path, attrs, f) for f in exclude):
                    continue
                visible = {k: v for k, v in attrs.items() if k != "__type__"}
                if int(attribute_filter_mode) == int(AttributeFilterMode.NONE):
                    visible = {}
                elif int(attribute_filter_mode) == int(AttributeFilterMode.SPECIFIC) and attribute_names:
                    visible = {k: v for k, v in visible.items() if k in set(attribute_names)}
                result[path] = visible
            return result

        return Operation(self, operation_name="query_prims", handle=object(), fetch_fn=_fetch)

    def read_attribute(
        self,
        attribute_name: str,
        prim_paths: list[str],
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        dest: Optional[Any] = None,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> ManagedDLTensor:
        return self.read_attribute_async(
            attribute_name, prim_paths, prim_mode, dest, cuda_stream, cuda_event
        ).wait().fetch()

    def _attribute_value_for_read(self, prim_path: str, attribute_name: str) -> Optional[tuple[Any, Semantic]]:
        override = self._attrs.get((prim_path, attribute_name))
        if override is not None:
            semantic, _ = self._attr_metadata.get((prim_path, attribute_name), (Semantic.NONE, False))
            return override, Semantic(semantic)
        info = self._prim_index.get(prim_path, {}).get(attribute_name)
        if info is not None:
            return getattr(info, "value", None), Semantic(getattr(info, "semantic", Semantic.NONE))
        return None

    def _prim_mode_indices(
        self,
        prim_paths: list[str],
        attribute_name: str,
        prim_mode: PrimMode,
        context: str,
        *,
        for_read: bool = False,
    ) -> list[int]:
        mode = PrimMode(prim_mode)
        if for_read and mode == PrimMode.CREATE_NEW:
            raise RuntimeError(f"{context}: PrimMode.CREATE_NEW is not supported for reads")
        if mode == PrimMode.CREATE_NEW:
            return list(range(len(prim_paths)))

        existing: list[int] = []
        missing: list[str] = []
        for index, path in enumerate(prim_paths):
            attrs = self._prim_index.get(path)
            if attrs is not None and attribute_name in attrs:
                existing.append(index)
            else:
                missing.append(path)

        if mode == PrimMode.MUST_EXIST and missing:
            raise RuntimeError(
                f"{context}: PrimMode.MUST_EXIST missing prim or attribute '{attribute_name}' "
                f"for {missing}"
            )
        return existing

    @staticmethod
    def _select_tensor_rows_for_indices(arr: np.ndarray, original_count: int, indices: list[int]) -> np.ndarray:
        if len(indices) == original_count or not indices:
            return arr
        if arr.shape and arr.shape[0] == original_count:
            return np.ascontiguousarray(arr[indices])
        return arr

    def read_attribute_async(
        self,
        attribute_name: str,
        prim_paths: list[str],
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        dest: Optional[Any] = None,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> Operation:
        del cuda_stream, cuda_event
        if dest is not None and not hasattr(dest, "__dlpack__"):
            raise TypeError("read_attribute dest must support the DLPack protocol")
        read_indices = self._prim_mode_indices(
            prim_paths,
            attribute_name,
            PrimMode(prim_mode),
            "read_attribute",
            for_read=True,
        )
        read_paths = [prim_paths[i] for i in read_indices]

        def _fetch(_: Optional[int] = None):
            vals = []
            string_vals: list[str] = []
            string_semantic = Semantic.TOKEN_STRING
            for p in read_paths:
                read_value = self._attribute_value_for_read(p, attribute_name)
                if read_value is None:
                    continue
                value, semantic = read_value
                is_string_value = semantic in (Semantic.TOKEN_STRING, Semantic.PATH_STRING) or (
                    value is not None and np.asarray(value).dtype.kind in "OUS"
                )
                if is_string_value:
                    values = _string_values_from_value(value)
                    string_vals.append(values[0] if values else "")
                    string_semantic = semantic if semantic in (Semantic.TOKEN_STRING, Semantic.PATH_STRING) else Semantic.TOKEN_STRING
                else:
                    vals.append(_dlpackable_attr_value(value, np.array(0.0, dtype=np.float32)))
            if string_vals:
                if vals:
                    raise RuntimeError(f"Read returned mixed string and numeric data for '{attribute_name}'")
                if dest is not None:
                    raise RuntimeError("read_attribute dest is not supported for token/path string attributes")
                return ManagedStringDLTensor(string_vals, string_semantic)
            if not vals:
                raise RuntimeError(
                    f"Read returned no data for '{attribute_name}' (attribute may not exist on the given prims)"
                )
            arr = np.ascontiguousarray(np.stack(vals))
            if dest is not None:
                try:
                    dest_arr = np.asarray(dest)
                except Exception as exc:
                    raise TypeError(f"read_attribute dest could not be viewed as an array: {exc}") from exc
                if tuple(dest_arr.shape) != tuple(arr.shape):
                    raise RuntimeError(
                        f"read_attribute dest shape {tuple(dest_arr.shape)} does not match result shape {tuple(arr.shape)}"
                    )
                try:
                    dest_arr[...] = arr
                except Exception as exc:
                    raise RuntimeError(f"read_attribute failed to write into dest: {exc}") from exc
                return ManagedDLTensor(dest)
            return ManagedDLTensor(arr)

        return Operation(self, operation_name="read_attribute", handle=object(), fetch_fn=_fetch)

    def read_array_attribute(
        self,
        attribute_name: str,
        prim_paths: list[str],
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
    ) -> dict[str, ManagedDLTensor]:
        return self.read_array_attribute_async(attribute_name, prim_paths, prim_mode).wait().fetch()

    def read_array_attribute_async(
        self,
        attribute_name: str,
        prim_paths: list[str],
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
    ) -> Operation:
        read_indices = self._prim_mode_indices(
            prim_paths,
            attribute_name,
            PrimMode(prim_mode),
            "read_array_attribute",
            for_read=True,
        )
        read_paths = [prim_paths[i] for i in read_indices]

        def _fetch(_: Optional[int] = None):
            result = {}
            for p in read_paths:
                read_value = self._attribute_value_for_read(p, attribute_name)
                if read_value is None:
                    continue
                value, semantic = read_value
                is_string_value = semantic in (Semantic.TOKEN_STRING, Semantic.PATH_STRING) or (
                    value is not None and np.asarray(value).dtype.kind in "OUS"
                )
                if is_string_value:
                    string_semantic = (
                        semantic if semantic in (Semantic.TOKEN_STRING, Semantic.PATH_STRING) else Semantic.TOKEN_STRING
                    )
                    result[p] = ManagedStringDLTensor(_string_values_from_value(value), string_semantic)
                else:
                    result[p] = ManagedDLTensor(
                        np.ascontiguousarray(_dlpackable_attr_value(value, np.array([], dtype=np.float32)))
                    )
            if not result:
                raise RuntimeError(
                    f"Read returned no data for '{attribute_name}' (attribute may not exist on the given prims)"
                )
            return result

        return Operation(self, operation_name="read_array_attribute", handle=object(), fetch_fn=_fetch)

    def write_attribute(
        self,
        prim_paths: list[str],
        attribute_name: str,
        tensor: Any,
        semantic: Semantic = Semantic.NONE,
        dirty_bits: Optional[bytes] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> None:
        self.write_attribute_async(
            prim_paths, attribute_name, tensor, semantic, dirty_bits, prim_mode, data_access, cuda_stream, cuda_event
        ).wait()

    def write_attribute_async(
        self,
        prim_paths: list[str],
        attribute_name: str,
        tensor: Any,
        semantic: Semantic = Semantic.NONE,
        dirty_bits: Optional[bytes] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> Operation:
        del dirty_bits, cuda_stream, cuda_event
        _validate_prim_paths(prim_paths)
        data_access = DataAccess(data_access)
        semantic = Semantic(semantic)
        if semantic == Semantic.PATH_STRING:
            raise ValueError(
                "Semantic.PATH_STRING requires write_array_attribute (is_array=True). "
                "Use write_array_attribute(semantic=Semantic.PATH_STRING, ...) instead of write_attribute."
            )
        if (_is_string_semantic(semantic) or (semantic == Semantic.NONE and _is_string_write_payload(tensor))) and (
            data_access == DataAccess.ASYNC
        ):
            raise ValueError("String attributes (token, path) require DataAccess.SYNC")
        write_indices = self._prim_mode_indices(
            prim_paths,
            attribute_name,
            PrimMode(prim_mode),
            "write_attribute",
        )
        write_paths = [prim_paths[i] for i in write_indices]
        if not write_paths:
            return Operation(self, operation_name="write_attribute")
        try:
            array_method = getattr(tensor, "__array__", None)
            if callable(array_method) and not isinstance(tensor, np.ndarray):
                arr = np.asarray(array_method())
            else:
                arr = np.asarray(tensor)
            arr = self._select_tensor_rows_for_indices(arr, len(prim_paths), write_indices)
            self._write_attr_array(write_paths, attribute_name, arr, semantic, is_array=False)
            return Operation(self, operation_name="write_attribute")
        except Exception as exc:
            return Operation(self, operation_name="write_attribute", error=exc)

    def write_array_attribute(
        self,
        prim_paths: list[str],
        attribute_name: str,
        tensors: list[Any],
        semantic: Semantic = Semantic.NONE,
        is_token: bool = False,
        dirty_bits: Optional[bytes] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> None:
        self.write_array_attribute_async(
            prim_paths, attribute_name, tensors, semantic, is_token, dirty_bits, prim_mode, data_access,
            cuda_stream, cuda_event
        ).wait()

    def write_array_attribute_async(
        self,
        prim_paths: list[str],
        attribute_name: str,
        tensors: list[Any],
        semantic: Semantic = Semantic.NONE,
        is_token: bool = False,
        dirty_bits: Optional[bytes] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> Operation:
        del dirty_bits, cuda_stream, cuda_event
        _validate_prim_paths(prim_paths)
        if not tensors:
            raise ValueError("tensors list cannot be empty")
        if len(tensors) != len(prim_paths):
            raise ValueError(
                f"tensors length must match prim_paths length, got {len(tensors)} tensors "
                f"for {len(prim_paths)} prim paths"
            )
        data_access = DataAccess(data_access)
        semantic = Semantic(semantic)
        if (_is_string_semantic(semantic) or (semantic == Semantic.NONE and _is_string_write_payload(tensors, is_array=True))) and (
            data_access == DataAccess.ASYNC
        ):
            raise ValueError("String attributes (token, path) require DataAccess.SYNC")
        write_indices = self._prim_mode_indices(
            prim_paths,
            attribute_name,
            PrimMode(prim_mode),
            "write_array_attribute",
        )
        if not write_indices:
            return Operation(self, operation_name="write_array_attribute")
        try:
            if semantic == Semantic.NONE and _is_string_write_payload(tensors, is_array=True):
                array_semantic = Semantic.TOKEN_STRING if is_token else Semantic.PATH_STRING
            else:
                array_semantic = semantic
            for i in write_indices:
                self._write_attr_array([prim_paths[i]], attribute_name, np.asarray(tensors[i]), array_semantic, is_array=True)
            return Operation(self, operation_name="write_array_attribute")
        except Exception as exc:
            return Operation(self, operation_name="write_array_attribute", error=exc)

    def bind_attribute(
        self,
        prim_paths: list[str],
        attribute_name: str,
        dtype: Any = None,
        shape: Optional[tuple] = None,
        semantic: Semantic = Semantic.NONE,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        flags: BindingFlag = BindingFlag.NONE,
    ) -> AttributeBinding:
        return self.bind_attribute_async(prim_paths, attribute_name, dtype, shape, semantic, prim_mode, flags).wait()

    def bind_attribute_async(
        self,
        prim_paths: list[str],
        attribute_name: str,
        dtype: Any = None,
        shape: Optional[tuple] = None,
        semantic: Semantic = Semantic.NONE,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        flags: BindingFlag = BindingFlag.NONE,
    ) -> Operation:
        resolved_semantic, resolved_dtype, element_shape = _resolve_attribute_type_args(
            "bind_attribute", dtype, shape, semantic, flags
        )
        handle = self._next_binding_handle
        self._next_binding_handle += 1
        binding = AttributeBinding(
            handle=handle,
            semantic=int(resolved_semantic),
            dtype=resolved_dtype,
            renderer=self,
            is_array=False,
            shape=element_shape,
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            prim_mode=PrimMode(prim_mode),
        )
        self._bindings[id(binding)] = binding
        return Operation(self, handle=binding, operation_name="bind_attribute")

    def bind_array_attribute(
        self,
        prim_paths: list[str],
        attribute_name: str,
        dtype: Any = None,
        shape: Optional[tuple] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        flags: BindingFlag = BindingFlag.NONE,
    ) -> AttributeBinding:
        return self.bind_array_attribute_async(prim_paths, attribute_name, dtype, shape, prim_mode, flags).wait()

    def bind_array_attribute_async(
        self,
        prim_paths: list[str],
        attribute_name: str,
        dtype: Any = None,
        shape: Optional[tuple] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        flags: BindingFlag = BindingFlag.NONE,
    ) -> Operation:
        op = self.bind_attribute_async(prim_paths, attribute_name, dtype, shape, Semantic.NONE, prim_mode, flags)
        binding = op.wait()
        binding._is_array = True
        return Operation(self, handle=binding, operation_name="bind_array_attribute")

    def map_attribute(
        self,
        prim_paths: list[str],
        attribute_name: str,
        dtype: Any = None,
        shape: Optional[tuple] = None,
        semantic: Semantic = Semantic.NONE,
        device: Device = Device.CPU,
        device_id: int = 0,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
    ) -> AttributeMapping:
        resolved_semantic, resolved_dtype, element_shape = _resolve_attribute_type_args(
            "map_attribute", dtype, shape, semantic
        )
        handle = self._next_binding_handle
        self._next_binding_handle += 1
        binding = AttributeBinding(
            handle=handle,
            semantic=int(resolved_semantic),
            dtype=resolved_dtype,
            renderer=self,
            is_array=False,
            shape=element_shape,
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            prim_mode=PrimMode(prim_mode),
        )
        self._bindings[id(binding)] = binding
        return self._map_bound_attribute(binding, device, device_id)

    def _map_bound_attribute(
        self,
        binding: AttributeBinding,
        device: Device = Device.CPU,
        device_id: int = 0,
    ) -> AttributeMapping:
        if device not in (Device.DEFAULT, Device.CPU, Device.CUDA, Device.CUDA_ARRAY):
            raise NotImplementedError(f"Unsupported attribute mapping device: {device!r}")
        if Device(device) in (Device.CUDA, Device.CUDA_ARRAY) and self._backend == "metal":
            raise NotImplementedError("CUDA attribute mapping is not supported by the Metal backend")
        map_indices = self._prim_mode_indices(
            binding._prim_paths,
            binding._attribute_name,
            binding.prim_mode,
            "map_attribute",
        )
        map_paths = [binding._prim_paths[i] for i in map_indices]
        if len(map_paths) != len(binding._prim_paths):
            binding = AttributeBinding(
                handle=binding.handle,
                semantic=int(binding.semantic),
                dtype=binding.dtype,
                renderer=self,
                is_array=binding.is_array,
                shape=binding.shape,
                prim_paths=map_paths,
                attribute_name=binding._attribute_name,
                prim_mode=binding.prim_mode,
            )
        dtype = _numpy_dtype(binding.dtype, binding.semantic)
        elem_shape = _binding_element_shape(binding)
        full_shape = (len(binding._prim_paths),) + tuple(elem_shape)
        arr = np.zeros(full_shape, dtype=dtype)
        if binding.semantic == Semantic.XFORM_MAT4x4 or binding._attribute_name in ("omni:xform", "xformOp:transform"):
            arr[:] = np.eye(4, dtype=dtype)
        for i, path in enumerate(binding._prim_paths):
            read_value = self._attribute_value_for_read(path, binding._attribute_name)
            if read_value is None:
                continue
            value, semantic = read_value
            if semantic in (Semantic.TOKEN_STRING, Semantic.PATH_STRING):
                continue
            value_arr = np.asarray(value)
            if value_arr.dtype.kind in "OUS":
                continue
            try:
                value_arr = np.asarray(value, dtype=dtype)
                if value_arr.shape == arr[i].shape:
                    arr[i] = value_arr
                elif value_arr.size == arr[i].size:
                    arr[i] = value_arr.reshape(arr[i].shape)
                elif arr[i].shape == () and value_arr.size == 1:
                    arr[i] = value_arr.reshape(())
            except Exception:
                continue
        mapped_device = Device.CPU if device == Device.DEFAULT else Device(device)
        native_gpu_mapping = self._try_map_native_gpu_transforms(
            binding,
            binding._prim_paths,
            np.ascontiguousarray(arr),
            mapped_device,
            device_id,
        )
        if native_gpu_mapping is not None:
            return native_gpu_mapping
        return AttributeMapping(binding, np.ascontiguousarray(arr), device=mapped_device)

    def unmap_attribute(
        self,
        mapping: AttributeMapping,
        event: Optional[int] = None,
        stream: Optional[int] = None,
    ) -> None:
        mapping.unmap(event=event, stream=stream)

    def unmap_attribute_async(
        self,
        mapping: AttributeMapping,
        event: Optional[int] = None,
        stream: Optional[int] = None,
    ) -> Operation:
        return mapping.unmap_async(event=event, stream=stream)

    def _write_bound_attribute(self, binding: AttributeBinding, data: Any) -> None:
        prim_count = len(binding._prim_paths)
        if not binding.is_array and _is_native_mutable_xform_attribute(binding._attribute_name, binding.semantic):
            self._release_native_gpu_transform_binding(int(binding.handle))
        if binding.is_array:
            if not isinstance(data, list):
                raise TypeError("AttributeBinding.write(): array bindings require a list of tensors, one per prim")
            if not data:
                raise ValueError("tensors list cannot be empty")
            if len(data) != prim_count:
                raise ValueError(
                    f"tensors length must match bound prim count, got {len(data)} tensors for {prim_count} prim paths"
                )
            write_indices = self._prim_mode_indices(
                binding._prim_paths,
                binding._attribute_name,
                binding.prim_mode,
                "AttributeBinding.write()",
            )
            if not write_indices:
                return
            if _is_string_semantic(binding.semantic):
                for i in write_indices:
                    value = data[i]
                    if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
                        raise TypeError(
                            f"tensors[{i}]: expected a list of strings for this string attribute binding, "
                            f"got {type(value).__name__}"
                        )
                    self._write_attr_array(
                        [binding._prim_paths[i]],
                        binding._attribute_name,
                        np.asarray(value),
                        binding.semantic,
                        is_array=True,
                    )
                return
            for i in write_indices:
                value = data[i]
                arr = _validate_array_binding_tensor(
                    binding,
                    np.asarray(value),
                    f"AttributeBinding.write(tensors[{i}])",
                )
                self._write_attr_array([binding._prim_paths[i]], binding._attribute_name, arr, binding.semantic, True)
            return

        if _is_string_semantic(binding.semantic):
            if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
                raise TypeError(
                    "AttributeBinding.write(): expected a list of strings for this string attribute binding, "
                    f"got {type(data).__name__}"
                )
            if len(data) != prim_count:
                raise ValueError(
                    f"string value count must match bound prim count, got {len(data)} strings for {prim_count} prim paths"
                )
            write_indices = self._prim_mode_indices(
                binding._prim_paths,
                binding._attribute_name,
                binding.prim_mode,
                "AttributeBinding.write()",
            )
            if not write_indices:
                return
            arr = self._select_tensor_rows_for_indices(np.asarray(data), prim_count, write_indices)
            paths = [binding._prim_paths[i] for i in write_indices]
            self._write_attr_array(paths, binding._attribute_name, arr, binding.semantic, False)
            return

        arr = _validate_scalar_binding_array(binding, np.asarray(data), "AttributeBinding.write()")
        write_indices = self._prim_mode_indices(
            binding._prim_paths,
            binding._attribute_name,
            binding.prim_mode,
            "AttributeBinding.write()",
        )
        if not write_indices:
            return
        arr = self._select_tensor_rows_for_indices(arr, prim_count, write_indices)
        paths = [binding._prim_paths[i] for i in write_indices]
        self._write_attr_array(paths, binding._attribute_name, arr, binding.semantic, binding.is_array)

    def _apply_attribute_side_effect(
        self,
        prim_path: str,
        attribute_name: str,
        value_arr: Any,
        semantic: Semantic,
        *,
        use_overrides: bool = True,
    ) -> None:
        prim_type = self._prim_type(prim_path)
        if prim_type == "RenderSettings":
            settings_spec = self._render_settings_spec_for(prim_path)
            if attribute_name == "resolution":
                try:
                    wh = np.asarray(value_arr, dtype=np.int32).ravel()
                    if wh.size >= 2:
                        settings_spec.width = max(int(wh[0]), 1)
                        settings_spec.height = max(int(wh[1]), 1)
                        self._apply_render_settings_defaults(prim_path)
                except Exception:
                    pass
                return
            if attribute_name in ("renderMode", "nanousd:renderMode", "omni:rtx:rendermode"):
                try:
                    values = _string_values_from_value(value_arr)
                    token = (values[0] if values else str(np.asarray(value_arr).ravel()[0])).strip().lower()
                    if token:
                        settings_spec.render_mode = token
                        self._apply_render_settings_defaults(prim_path)
                except Exception:
                    pass
                return
            if attribute_name == "omni:rtx:minimal:mode":
                try:
                    settings_spec.minimal_mode = int(np.asarray(value_arr).ravel()[0])
                    self._apply_render_settings_defaults(prim_path)
                except Exception:
                    pass
                return
            if _apply_render_settings_base_attribute(settings_spec, attribute_name, value_arr):
                self._apply_render_settings_defaults(prim_path)
                return
            if attribute_name in ("includedPurposes", "materialBindingPurposes"):
                values = _string_values_from_value(value_arr)
                if attribute_name == "includedPurposes":
                    settings_spec.included_purposes = values
                else:
                    settings_spec.material_binding_purposes = values
                self._apply_render_settings_defaults(prim_path)
                return
            if attribute_name == "renderingColorSpace":
                values = _string_values_from_value(value_arr)
                settings_spec.rendering_color_space = values[0] if values else None
                self._apply_render_settings_defaults(prim_path)
                return
            if semantic == Semantic.PATH_STRING and attribute_name in ("camera", "products"):
                targets = [
                    target
                    for target in (
                        _resolve_relationship_target(prim_path, value)
                        for value in _string_values_from_value(value_arr)
                    )
                    if target
                ]
                if attribute_name == "camera":
                    settings_spec.camera_path = targets[0] if targets else None
                    settings_spec.camera_paths = list(targets) or None
                else:
                    settings_spec.product_paths = list(targets)
                self._apply_render_settings_defaults(prim_path)
                return

        if prim_type == "RenderProduct" and attribute_name in ("productName", "productType"):
            values = _string_values_from_value(value_arr)
            if values:
                spec = self._render_product_specs.setdefault(
                    prim_path, _RenderProductSpec(path=prim_path, render_vars=["LdrColor"])
                )
                if attribute_name == "productName":
                    spec.product_name = values[0]
                else:
                    spec.product_type = values[0]
            return

        if prim_type == "RenderProduct":
            spec = self._render_product_specs.setdefault(
                prim_path, _RenderProductSpec(path=prim_path, render_vars=["LdrColor"])
            )
            if _apply_render_settings_base_attribute(spec, attribute_name, value_arr):
                return

        if semantic == Semantic.XFORM_MAT4x4 or attribute_name in ("omni:xform", "xformOp:transform"):
            matrix = np.asarray(value_arr, dtype=np.float64).reshape(4, 4)
            if self._is_camera_path(prim_path):
                self._camera_matrices[prim_path] = matrix
                self._camera_matrix = matrix
        elif attribute_name in (
            "focalLength",
            "horizontalAperture",
            "verticalAperture",
            "horizontalApertureOffset",
            "verticalApertureOffset",
        ):
            try:
                self._refresh_camera_projection(prim_path, use_overrides=use_overrides)
            except Exception:
                pass
        elif attribute_name == "clippingRange":
            try:
                if self._is_camera_path(prim_path):
                    self._set_camera_clip_range(prim_path, value_arr)
            except Exception:
                pass
        elif attribute_name == "resolution":
            try:
                wh = np.asarray(value_arr, dtype=np.int32).ravel()
                if wh.size >= 2:
                    spec = self._render_product_specs.setdefault(
                        prim_path, _RenderProductSpec(path=prim_path, render_vars=["LdrColor"])
                    )
                    spec.width = max(int(wh[0]), 1)
                    spec.height = max(int(wh[1]), 1)
                    self._width = spec.width
                    self._height = spec.height
                    if self._nu is not None and hasattr(self._nu, "set_render_size"):
                        self._nu.set_render_size(self._width, self._height)
            except Exception:
                pass
        elif attribute_name in ("renderMode", "nanousd:renderMode", "omni:rtx:rendermode"):
            try:
                token = str(np.asarray(value_arr).ravel()[0]).strip().lower()
                if token:
                    spec = self._render_product_specs.setdefault(
                        prim_path, _RenderProductSpec(path=prim_path, render_vars=["LdrColor"])
                    )
                    spec.render_mode = token
            except Exception:
                pass
        elif attribute_name == "omni:rtx:minimal:mode":
            try:
                mode = int(np.asarray(value_arr).ravel()[0])
                spec = self._render_product_specs.setdefault(
                    prim_path, _RenderProductSpec(path=prim_path, render_vars=["LdrColor"])
                )
                spec.minimal_mode = mode
            except Exception:
                pass
        elif semantic == Semantic.PATH_STRING and attribute_name in ("camera", "orderedVars", "products"):
            targets = [
                target
                for target in (_resolve_relationship_target(prim_path, value) for value in _string_values_from_value(value_arr))
                if target
            ]
            if attribute_name == "camera" and targets:
                spec = self._render_product_specs.setdefault(
                    prim_path, _RenderProductSpec(path=prim_path, render_vars=["LdrColor"])
                )
                spec.camera_path = targets[0]
                spec.camera_paths = list(targets)
                if self._default_camera_path is None:
                    self._default_camera_path = targets[0]
            elif attribute_name == "orderedVars":
                spec = self._render_product_specs.setdefault(
                    prim_path, _RenderProductSpec(path=prim_path, render_vars=[])
                )
                names: list[str] = []
                for target in targets:
                    source_value = self._attrs.get((target, "sourceName"))
                    if source_value is None:
                        source_info = self._prim_index.get(target, {}).get("sourceName")
                        source_value = getattr(source_info, "value", None)
                    values = _string_values_from_value(source_value)
                    names.append(values[0] if values else _render_var_name(target))
                spec.render_vars = [name for name in names if name]
            elif attribute_name == "products":
                for target in targets:
                    self._render_products.add(target)
                    self._render_product_specs.setdefault(
                        target, _RenderProductSpec(path=target, render_vars=["LdrColor"])
                    )

    def _attribute_affects_native_stage(self, prim_path: str, attribute_name: str, semantic: Semantic) -> bool:
        if attribute_name in _NATIVE_OVRTX_RENDER_EXPOSURE_ATTRIBUTES:
            return True
        if attribute_name in {"purpose", "visibility", "apiSchemas"}:
            return True
        if attribute_name.startswith("material:binding"):
            return True
        prim_type = self._prim_type(prim_path)
        if prim_type in {"RenderSettings", "RenderProduct"}:
            if attribute_name in _NATIVE_RUNTIME_RENDER_PRODUCT_ATTRIBUTES:
                return False
            return attribute_name in {
                "disableMotionBlur",
                "disableDepthOfField",
                "instantaneousShutter",
                "includedPurposes",
                "materialBindingPurposes",
            }
        if self._is_camera_path(prim_path):
            if attribute_name in _NATIVE_RUNTIME_CAMERA_ATTRIBUTES:
                return False
            return attribute_name in {
                "shutter:open",
                "shutter:close",
                "fStop",
                "focusDistance",
            }
        return semantic == Semantic.PATH_STRING and attribute_name in {"references", "payload"}

    def _write_attr_array(
        self,
        prim_paths: list[str],
        attribute_name: str,
        arr: np.ndarray,
        semantic: Semantic,
        is_array: bool,
    ) -> None:
        effective_semantic = (
            Semantic.XFORM_MAT4x4
            if semantic == Semantic.XFORM_MAT4x4 or attribute_name in ("omni:xform", "xformOp:transform")
            else semantic
        )
        if is_array:
            values = [arr] if len(prim_paths) == 1 else (
                list(arr) if arr.shape and arr.shape[0] == len(prim_paths) else [arr] * len(prim_paths)
            )
            for p, value in zip(prim_paths, values):
                value_arr = np.ascontiguousarray(value)
                self._attrs[(p, attribute_name)] = value_arr
                dtype, info_semantic = _array_dl_dtype(value_arr, effective_semantic, is_array=True)
                self._attr_metadata[(p, attribute_name)] = (info_semantic, True)
                self._prim_index.setdefault(p, {})[attribute_name] = AttributeInfo(
                    attribute_name,
                    dtype,
                    True,
                    info_semantic,
                    value_arr,
                )
                self._apply_attribute_side_effect(p, attribute_name, value_arr, info_semantic)
                if (
                    self._attribute_affects_native_stage(p, attribute_name, info_semantic)
                    and not self._native_attribute_can_mutate_without_reload(p, attribute_name, info_semantic)
                ):
                    self._native_stage_dirty = True
            self._apply_native_attribute_batch(prim_paths, attribute_name, values, effective_semantic)
            return
        if len(prim_paths) == 1 and arr.shape and arr.shape[0] != 1:
            values = [arr]
        else:
            values = list(arr) if arr.shape and arr.shape[0] == len(prim_paths) else [arr] * len(prim_paths)
        for p, value in zip(prim_paths, values):
            value_arr = np.ascontiguousarray(value)
            self._attrs[(p, attribute_name)] = value_arr
            dtype, info_semantic = _array_dl_dtype(value_arr, effective_semantic, is_array=False)
            self._attr_metadata[(p, attribute_name)] = (info_semantic, False)
            self._prim_index.setdefault(p, {})[attribute_name] = AttributeInfo(
                attribute_name,
                dtype,
                False,
                info_semantic,
                value_arr,
            )
            self._apply_attribute_side_effect(p, attribute_name, value_arr, info_semantic)
            if (
                self._attribute_affects_native_stage(p, attribute_name, info_semantic)
                and not self._native_attribute_can_mutate_without_reload(p, attribute_name, info_semantic)
            ):
                self._native_stage_dirty = True
        self._apply_native_attribute_batch(prim_paths, attribute_name, values, effective_semantic)

    def _unbind_attribute(self, binding: AttributeBinding) -> None:
        binding.unbind()

    def _unbind_attribute_async(self, binding: AttributeBinding) -> Operation:
        binding.unbind()
        return Operation(self, operation_name="unbind_attribute")

    def __bool__(self) -> bool:
        return True


__all__ = [
    "__version__",
    "OVRTX_LIBRARY_PATH_HINT",
    "OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP",
    "OVRTX_ATTR_NAME_PICKABLE",
    "OVRTX_RENDER_VAR_PICK_HIT",
    "OVRTX_PICK_FLAG_GIZMO",
    "OVRTX_PICK_FLAG_INCLUDE_TRACKED_INFO",
    "OVRTX_PICK_HIT_MAGIC",
    "OVRTX_PICK_HIT_VERSION",
    "register_schema_paths",
    "usd_pluginpath_env_keys",
    "BindingFlag",
    "DataAccess",
    "Device",
    "DLDataType",
    "EventStatus",
    "PrimMode",
    "SelectionFillMode",
    "Semantic",
    "AttributeFilterMode",
    "FilterKind",
    "AttributeInfo",
    "OperationCounter",
    "OperationStatus",
    "Renderer",
    "RendererConfig",
    "PendingFetch",
    "SelectionGroupStyle",
    "AttributeBinding",
    "AttributeMapping",
    "FrameOutput",
    "ManagedDLTensor",
    "MappedRenderVar",
    "Operation",
    "ProductOutput",
    "RenderProductSetOutputs",
    "RenderVarParam",
    "RenderVarTensor",
    "RenderVarOutput",
]
