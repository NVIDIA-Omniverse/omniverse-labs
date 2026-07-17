# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import statistics
import time

import bpy
import gpu


WIDTH = 1280
HEIGHT = 720
WARMUP = 2
ITERATIONS = 12


def rgba8_payload(width, height):
    return bytes((i * 17 + 31) & 0xFF for i in range(width * height * 4))


def rgba8_to_float_array(payload):
    try:
        import numpy as np

        return np.frombuffer(payload, dtype=np.uint8).astype(np.float32) * np.float32(1.0 / 255.0)
    except Exception:
        from array import array

        return array("f", (value / 255.0 for value in payload))


def stats_ms(values):
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "p50": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def measure_current_gpu_texture(payload, width, height):
    timings = []
    convert_timings = []
    buffer_timings = []
    texture_timings = []
    texture = None
    for index in range(WARMUP + ITERATIONS):
        started = time.perf_counter_ns()
        convert_started = started
        rgba = rgba8_to_float_array(payload)
        buffer_started = time.perf_counter_ns()
        buffer = gpu.types.Buffer("FLOAT", len(rgba), rgba)
        texture_started = time.perf_counter_ns()
        texture = gpu.types.GPUTexture((width, height), format="RGBA8", data=buffer)
        texture.filter_mode(True)
        completed = time.perf_counter_ns()
        if index >= WARMUP:
            convert_timings.append((buffer_started - convert_started) / 1_000_000.0)
            buffer_timings.append((texture_started - buffer_started) / 1_000_000.0)
            texture_timings.append((completed - texture_started) / 1_000_000.0)
            timings.append((completed - started) / 1_000_000.0)
    return {
        "total_ms": stats_ms(timings),
        "convert_ms": stats_ms(convert_timings),
        "buffer_ms": stats_ms(buffer_timings),
        "texture_ctor_ms": stats_ms(texture_timings),
        "last_texture": repr(texture),
    }


def measure_gpu_texture_update(payload, width, height):
    if not hasattr(gpu.types.GPUTexture, "update"):
        return {
            "available": False,
            "reason": "gpu.types.GPUTexture.update is not available",
        }

    initial_rgba = rgba8_to_float_array(payload)
    initial_buffer = gpu.types.Buffer("FLOAT", len(initial_rgba), initial_rgba)
    texture = gpu.types.GPUTexture((width, height), format="RGBA8", data=initial_buffer)
    texture.filter_mode(True)

    timings = []
    buffer_timings = []
    update_timings = []
    for index in range(WARMUP + ITERATIONS):
        started = time.perf_counter_ns()
        buffer_started = started
        buffer = gpu.types.Buffer("UBYTE", len(payload), payload)
        update_started = time.perf_counter_ns()
        texture.update(buffer, format="UBYTE")
        completed = time.perf_counter_ns()
        if index >= WARMUP:
            buffer_timings.append((update_started - buffer_started) / 1_000_000.0)
            update_timings.append((completed - update_started) / 1_000_000.0)
            timings.append((completed - started) / 1_000_000.0)

    return {
        "available": True,
        "total_ms": stats_ms(timings),
        "buffer_ms": stats_ms(buffer_timings),
        "texture_update_ms": stats_ms(update_timings),
        "last_texture": repr(texture),
    }


def measure_image_from_image(payload, width, height, *, float_buffer):
    timings = []
    convert_timings = []
    pixels_timings = []
    from_image_timings = []
    image = bpy.data.images.new(
        "ovrtx_image_float_probe" if float_buffer else "ovrtx_image_byte_probe",
        width,
        height,
        alpha=True,
        float_buffer=float_buffer,
    )
    image.colorspace_settings.name = "Non-Color"
    texture = None
    for index in range(WARMUP + ITERATIONS):
        started = time.perf_counter_ns()
        convert_started = started
        rgba = rgba8_to_float_array(payload)
        pixels_started = time.perf_counter_ns()
        image.pixels.foreach_set(rgba)
        from_image_started = time.perf_counter_ns()
        texture = gpu.texture.from_image(image)
        texture.filter_mode(True)
        completed = time.perf_counter_ns()
        if index >= WARMUP:
            convert_timings.append((pixels_started - convert_started) / 1_000_000.0)
            pixels_timings.append((from_image_started - pixels_started) / 1_000_000.0)
            from_image_timings.append((completed - from_image_started) / 1_000_000.0)
            timings.append((completed - started) / 1_000_000.0)
    return {
        "total_ms": stats_ms(timings),
        "convert_ms": stats_ms(convert_timings),
        "pixels_foreach_set_ms": stats_ms(pixels_timings),
        "from_image_ms": stats_ms(from_image_timings),
        "image_is_float": bool(image.is_float),
        "float_buffer_requested": bool(float_buffer),
        "last_texture": repr(texture),
    }


def measure_image_cached_from_image(payload, width, height):
    image = bpy.data.images.new("ovrtx_image_cache_probe", width, height, alpha=True, float_buffer=False)
    image.pixels.foreach_set(rgba8_to_float_array(payload))
    texture = gpu.texture.from_image(image)
    timings = []
    for index in range(WARMUP + ITERATIONS):
        started = time.perf_counter_ns()
        texture = gpu.texture.from_image(image)
        completed = time.perf_counter_ns()
        if index >= WARMUP:
            timings.append((completed - started) / 1_000_000.0)
    return {
        "from_image_cached_ms": stats_ms(timings),
        "last_texture": repr(texture),
    }


def main():
    payload = rgba8_payload(WIDTH, HEIGHT)
    ubyte_constructor_error = None
    try:
        buffer = gpu.types.Buffer("UBYTE", len(payload), payload)
        gpu.types.GPUTexture((WIDTH, HEIGHT), format="RGBA8", data=buffer)
    except Exception as exc:
        ubyte_constructor_error = str(exc)

    result = {
        "blender_version": bpy.app.version_string,
        "size": [WIDTH, HEIGHT],
        "payload_bytes": len(payload),
        "iterations": ITERATIONS,
        "warmup": WARMUP,
        "ubyte_constructor_error": ubyte_constructor_error,
        "current_gpu_texture": measure_current_gpu_texture(payload, WIDTH, HEIGHT),
        "gpu_texture_update": measure_gpu_texture_update(payload, WIDTH, HEIGHT),
        "image_from_image_byte_buffer": measure_image_from_image(
            payload, WIDTH, HEIGHT, float_buffer=False
        ),
        "image_from_image_float_buffer": measure_image_from_image(
            payload, WIDTH, HEIGHT, float_buffer=True
        ),
        "image_cached_from_image": measure_image_cached_from_image(payload, WIDTH, HEIGHT),
    }
    print("OVRTX_IMAGE_TEXTURE_PROBE_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("OVRTX_IMAGE_TEXTURE_PROBE_JSON_END")
    bpy.ops.wm.quit_blender()


bpy.app.timers.register(main, first_interval=0.1)
