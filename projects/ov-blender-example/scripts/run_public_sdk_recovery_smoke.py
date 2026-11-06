#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Test-only Pillow: taken from the consumer/test environment (scripts/requirements-public-sdk-consumer.txt), never the shipped SDK runtime.
"""Real render/edit/resize/SIGKILL/recovery through the existing typed client.

The supplied scene must contain the selected render product and camera. Only
the worker owned by this invocation is killed. Outputs and exact owned-resource
names are recorded so crash-cleanup failures cannot be hidden by global sweeps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time
from types import SimpleNamespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sensor", default="/Render/Camera")
    parser.add_argument("--camera", default="/World/Camera")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--render-var", choices=("LdrColor", "HdrColor"), default="LdrColor")
    parser.add_argument("--stage-metadata", type=Path,
                        help="JSON of authored stage metadata when the client Python lacks pxr")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Public source directory supplying the client and worker")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["OV_BLENDER_PUBLIC_SDK_PYTHON"] = args.python
    os.environ["OV_BLENDER_PUBLIC_SDK_OUTPUT"] = str(output / "workers")
    os.environ["OV_BLENDER_PUBLIC_SDK_GPU"] = str(args.gpu)
    sys.path.insert(0, str(args.source_root.resolve() / "addon"))
    sys.path.insert(0, str(args.source_root.resolve() / "scripts"))
    from run_public_sdk_first_frame_smoke import local_caches
    local_caches(output.parent / 'native-cache')
    from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
    from ovrtx_blender_example.ovrtx_scene_composition import OvrtxSceneComposition
    from ovrtx_blender_example.ovrtx_session import OvrtxSessionSpec
    from ovrtx_blender_example.ovrtx_value_updates import OvrtxTransformValue
    from PIL import Image
    scene = args.scene.resolve()
    composition = OvrtxSceneComposition(str(scene), str(scene), (),
        hashlib.sha256(scene.read_bytes()).hexdigest(), True)
    spec = OvrtxSessionSpec(composition, (args.sensor,), 320, 240, args.camera,
                           "composed_scene", render_var=args.render_var, output_size_value_route=True)
    if args.stage_metadata:
        spec = SimpleNamespace(**vars(spec), stage_metadata=json.loads(args.stage_metadata.read_text()))
    client = PublicSdkRenderClient()
    report = {"status": "running", "gpu_requested": args.gpu, "render_var": args.render_var,
              "frames": [], "owned_streams": [], "command": sys.argv,
              "fixture_sha256": hashlib.sha256(scene.read_bytes()).hexdigest()}
    owned_names = set()
    frames = {}
    previews = {}
    def payload(frame):
        return frame.linear_rgba16f if args.render_var == 'HdrColor' else frame.rgba8
    def preview(frame):
        if args.render_var != 'HdrColor':
            return bytes(frame.rgba8)
        # Diagnostic PNG only. Numeric HDR checks below always use the exact
        # owned binary16 plane, never this explicitly tone-mapped preview.
        pixels = bytearray(frame.width*frame.height*4)
        for index, components in enumerate(struct.iter_unpack('<4e', frame.linear_rgba16f)):
            for channel, value in enumerate(components[:3]):
                assert math.isfinite(value)
                value = max(0., value)
                pixels[index*4+channel] = round(255*value/(1+value))
            pixels[index*4+3] = 255
        return bytes(pixels)
    def render(label):
        started = time.monotonic()
        result = client.render_result(identity, selected_sensor_paths=(args.sensor,),
                                      render_var=args.render_var, additional_samples=args.samples)
        assert result.native_timings["transport"] == 'ovstream_shm_raw'
        assert len(payload(result)) == result.width * result.height * (8 if args.render_var == 'HdrColor' else 4)
        name, stream_epoch, producer_pid = client._reader_key
        owned_names.add(name)
        frames[label] = result
        pixels = previews[label] = preview(result)
        Image.frombytes("RGBA", (result.width, result.height), pixels).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM).save(output / (label + ".png"))
        center = (result.height // 2 * result.width + result.width // 2) * 4
        corner_rgb = pixels[:3]
        center_rgb = pixels[center:center+3]
        report["frames"].append(dict(label=label, width=result.width, height=result.height,
            sha256=hashlib.sha256(payload(result)).hexdigest(), samples=result.completed_samples,
            epoch=client.runtime.epoch, pid=client.runtime.pid, stream_name=name,
            stream_epoch=stream_epoch, center_rgb=list(center_rgb), corner_rgb=list(corner_rgb),
            seconds=time.monotonic()-started, preview_encoding='reinhard_rgb_alpha1' if args.render_var == 'HdrColor' else 'exact_rgba8'))
        if args.render_var == "HdrColor":
            assert result.frame_format == "rgba16f" and len(result.linear_rgba16f) == result.width * result.height * 8
            report["frames"][-1].update(hdr_sha256=hashlib.sha256(result.linear_rgba16f).hexdigest(),
                hdr_max=max(v[0] for v in struct.iter_unpack("<e", result.linear_rgba16f)))
            rgb = [channel for pixel in struct.iter_unpack('<4e', result.linear_rgba16f) for channel in pixel[:3]]
            assert all(math.isfinite(value) for value in rgb) and max(rgb) > min(rgb)
        assert max(pixels[0::4]) != min(pixels[0::4]) or max(pixels[1::4]) != min(pixels[1::4]), "Rendered image is uniform"
        return result
    def difference(first, second):
        assert (first.width, first.height) == (second.width, second.height)
        first_pixels, second_pixels = preview(first), preview(second)
        return sum(abs(a-b) for a,b in zip(first_pixels, second_pixels)) / len(first_pixels)
    try:
        identity = client.start_session(spec)
        report['runtime_identity'] = client.runtime.request('health')['runtime_identity']
        render("initial")
        original_pid = client.runtime.pid
        original_epoch = client.runtime.epoch
        matrix = ((1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,1.,0.),(2.,0.,8.,1.))
        client.update_transforms(identity, (OvrtxTransformValue(args.camera, matrix),))
        render("camera-edited")
        report["camera_edit_mean_absolute_byte_difference"] = difference(frames["initial"], frames["camera-edited"])
        assert report["camera_edit_mean_absolute_byte_difference"] > 1, "Camera edit had no measured image effect"
        client.resize_output(identity, 400, 240)
        before = render("before-kill")
        assert (before.width, before.height) == (400, 240)
        assert client.runtime.pid == original_pid and client.runtime.epoch == original_epoch
        gpu_query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        gpu_rows = gpu_query.stdout
        report["gpu_process_rows"] = [row for row in gpu_rows.splitlines() if row.split(",")[0].strip() == str(original_pid)]
        # WDDM may omit graphics-process rows. Actual rendered pixels and
        # renderer ownership remain mandatory; do not fabricate a GPU row.
        report["gpu_process_query_returncode"] = gpu_query.returncode
        if sys.platform != 'win32':
            assert report["gpu_process_rows"], "Owned renderer not visible as a GPU process"
        report['initial_worker_identity'] = vars(client.runtime.identity)
        # The authenticated producer can be a venv launcher child. The retained
        # owner terminates and confirms the whole owned tree, not an assumed PID.
        client.runtime._owner.stop()
        assert client.runtime.returncode is not None
        after = render("recovered")
        assert client.runtime.pid != original_pid and client.runtime.epoch == original_epoch + 1
        assert (after.width, after.height) == (400, 240), "Recovered resize was not replayed"
        assert client.runtime.diagnostics()["retained_values"] >= 2, "Camera update ledger was lost"
        report["recovery_mean_absolute_byte_difference"] = difference(before, after)
        assert report["recovery_mean_absolute_byte_difference"] < 5, "Recovery failed to reproduce edited scene"
        if args.render_var == "HdrColor":
            report["recovery_hdr_mean_absolute_difference"] = sum(
                abs(a[0]-b[0]) for a,b in zip(struct.iter_unpack("<e",before.linear_rgba16f),
                                            struct.iter_unpack("<e",after.linear_rgba16f))) / (len(before.linear_rgba16f)//2)
            assert report["recovery_hdr_mean_absolute_difference"] < .1
        assert report["frames"][-1]["stream_epoch"] != report["frames"][-2]["stream_epoch"]
        recovered_pid = client.runtime.pid
        client.resize_output(identity, 256, 192)
        resized = render("resized-after-recovery")
        assert (resized.width, resized.height) == (256,192)
        assert client.runtime.pid == recovered_pid
        client.resize_output(identity, 320, 240)
        for _ in range(2):
            client.render_result(identity, selected_sensor_paths=(args.sensor,),
                                 render_var=args.render_var, additional_samples=1)
        durations = []
        benchmark_started = time.perf_counter()
        for _ in range(30):
            frame_started = time.perf_counter()
            timed = client.render_result(identity, selected_sensor_paths=(args.sensor,),
                                         render_var=args.render_var, additional_samples=1)
            durations.append((time.perf_counter() - frame_started) * 1000)
            assert (timed.width, timed.height) == (320, 240)
            assert len(payload(timed)) == 320 * 240 * (8 if args.render_var == 'HdrColor' else 4)
        benchmark_seconds = time.perf_counter() - benchmark_started
        owned_names.add(client._reader_key[0])
        report["warm_native_render_copy_ack"] = dict(
            width=320, height=240, frames=30, samples_per_request=1,
            p50_ms=statistics.median(durations), p95_ms=sorted(durations)[28],
            total_seconds=benchmark_seconds, frames_per_second=30 / benchmark_seconds,
            includes="native render, SHM owned copy, frame acknowledgment; excludes Blender viewport upload",
        )
        report["runtime"] = client.runtime.diagnostics()
        report["vendor_modules_in_client"] = sorted({"ovrtx","ovphysx","ovstage","ovstream","warp"} & sys.modules.keys())
        assert not report["vendor_modules_in_client"]
        if args.render_var == "HdrColor":
            report["hdr_disk_payloads"] = [str(p) for p in (output / "workers").rglob("*.bin")]
            assert not report["hdr_disk_payloads"], "HDR smoke must not use disk payload fallback"
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        client.shutdown()
        report['runtime_after_shutdown'] = client.runtime.diagnostics()
        report["owned_streams"] = sorted(owned_names)
        if sys.platform == 'win32':
            from ovrtx_blender_example.win32_shm import existing_resources
            report["leftover_owned_resources"] = [resource for name in owned_names
                for resource, exists in existing_resources(name).items() if exists]
        else:
            paths = [Path("/dev/shm") / ("ovstream-" + name) for name in owned_names]
            paths += [Path("/dev/shm") / ("sem.ovstream-" + name + "-wake") for name in owned_names]
            paths += [Path(os.environ.get("TMPDIR") or "/tmp") / ("ovstream-" + name + ".sock") for name in owned_names]
            report["leftover_owned_resources"] = [str(path) for path in paths if path.exists()]
        report["cleanup_passed"] = (not report["leftover_owned_resources"]
            and report['runtime_after_shutdown']['closed']
            and report['runtime_after_shutdown']['returncode'] == 0
            and not report['runtime_after_shutdown']['cleanup_errors'])
        if not report["cleanup_passed"]:
            report["status"] = "failed"
        (output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
