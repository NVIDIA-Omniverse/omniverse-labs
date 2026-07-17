#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared execution helpers for repository validation commands."""

from __future__ import annotations

import hashlib
import ctypes
import json
import os
import platform
import re
import signal
import shlex
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Protocol, Sequence

try:
    import visual_comparison
except ModuleNotFoundError:  # Imported as scripts.validation in tests and tooling.
    from scripts import visual_comparison


REPOSITORY = "ov-blender-example"
VISUAL_CASES = (
    "demo_stair_drop_1280x720",
    "perf_junk_shop_1280x720",
    "perf_blender_classroom_1280x720",
    "hero_cube",
)
VISUAL_DIMENSIONS = {
    "demo_stair_drop_1280x720": (1280, 720),
    "perf_junk_shop_1280x720": (1280, 720),
    "perf_blender_classroom_1280x720": (1280, 720),
    "hero_cube": (640, 480),
}
EXIT_CODES = {"pass": 0, "regression": 1, "unavailable": 2}
COLOR_PRESENTATION = "scene_linear_hdr"
LDR_COLOR_PRESENTATION = "ldr_rgba8_display_passthrough"
SUPPORTED_COLOR_PRESENTATIONS = {COLOR_PRESENTATION, LDR_COLOR_PRESENTATION}
VISUAL_SAMPLES = {
    case: 16 if case == "hero_cube" else 64 for case in VISUAL_CASES
}
GOLDEN_KIND = "ov-blender-example-golden-image"
VALIDATION_SURFACE = (
    "scripts/validation.py",
    "scripts/navigation.py",
    "scripts/visual_comparison.py",
)
VALIDATION_PYTHON = sys.executable
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
COMMAND_KEYS = {
    "argv",
    "cwd",
    "exit_status",
    "started_at_ns",
    "ended_at_ns",
    "stdout_path",
    "stderr_path",
}
DEPLOYED_ROOT_NAMES = (
    "ovrtx-bridge-server",
    "ovphysx-bridge-server",
    "ovrtx-bridge-client",
    "ovphysx-bridge-client",
)
DIRECT_DEPLOY_ROOT = "direct_deploy"
TERMINATION_SIGNALS = tuple(
    handled
    for name in ("SIGHUP", "SIGINT", "SIGTERM")
    if (handled := getattr(signal, name, None)) is not None
)
BLENDER_PROCESS_NAMES = ("blender", "Blender", "blender.exe")


def _portable_path(value: str | Path) -> Path | PurePosixPath | PureWindowsPath:
    if isinstance(value, Path):
        return value
    text = str(value)
    if PureWindowsPath(text).is_absolute():
        return PureWindowsPath(text)
    if PurePosixPath(text).is_absolute():
        return PurePosixPath(text)
    return Path(text)


def _is_absolute_path(value: Any) -> bool:
    return isinstance(value, str) and _portable_path(value).is_absolute()


def _test_environment_python(
    source: str | Path,
) -> Path | PurePosixPath | PureWindowsPath:
    environment = _portable_path(source) / "out" / "validation-test-env"
    for relative in (Path("bin/python"), Path("Scripts/python.exe")):
        candidate = environment / relative
        if candidate.is_file():
            return candidate
    fallback = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return environment / fallback


class ValidationError(RuntimeError):
    """The supplied roots or output directory violate the fixed interface."""


def kill_blender_processes() -> None:
    """Terminate Blender before and after validation on dedicated workers."""

    if os.name == "nt":
        _run_process_killer(
            ["taskkill", "/IM", "blender.exe", "/T", "/F"],
            {0, 128},
            allowed_failure_output=(
                'the process "blender.exe" not found',
                "no tasks are running",
            ),
        )
        return
    for process_name in BLENDER_PROCESS_NAMES:
        _run_process_killer(["pkill", "-TERM", "-x", process_name], {0, 1})
    time.sleep(0.5)
    for process_name in BLENDER_PROCESS_NAMES:
        _run_process_killer(["pkill", "-KILL", "-x", process_name], {0, 1})


def _run_process_killer(
    argv: Sequence[str],
    allowed_returncodes: set[int],
    *,
    allowed_failure_output: Sequence[str] = (),
) -> None:
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as error:
        raise ValidationError(f"missing process killer: {argv[0]}") from error
    if completed.returncode not in allowed_returncodes:
        detail = (completed.stderr or completed.stdout).strip()
        if detail and any(
            allowed in detail.casefold() for allowed in allowed_failure_output
        ):
            return
        raise ValidationError(
            "failed to terminate Blender processes with "
            + shlex.join(argv)
            + (f": {detail}" if detail else "")
        )


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    cwd: str
    exit_status: int
    started_at_ns: int
    ended_at_ns: int
    stdout_path: str
    stderr_path: str

    def evidence(self) -> dict[str, Any]:
        return {
            "argv": self.argv,
            "cwd": self.cwd,
            "exit_status": self.exit_status,
            "started_at_ns": self.started_at_ns,
            "ended_at_ns": self.ended_at_ns,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
        }


class Executor(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        output_dir: Path,
        label: str,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult: ...


def _tee_output(
    source: Any,
    log: Any,
    verbose: bool,
    errors: list[BaseException],
) -> None:
    try:
        while chunk := os.read(source.fileno(), 64 * 1024):
            log.write(chunk)
            log.flush()
            if verbose:
                stream = getattr(sys.stdout, "buffer", None)
                if stream is None:
                    sys.stdout.write(chunk.decode(errors="replace"))
                    sys.stdout.flush()
                else:
                    stream.write(chunk)
                    stream.flush()
    except BaseException as error:
        errors.append(error)
    finally:
        source.close()


def _command_timeout(argv: Sequence[str]) -> int:
    if any(
        str(argument).endswith("run_blender_light_edit_responsiveness.py")
        for argument in argv
    ):
        return 960 if "--inside-blender" in argv else 975
    if any(str(argument).endswith("run_blender_navigation.py") for argument in argv):
        return 600
    return 180 if argv and (argv[0] == "blender" or "--window-geometry" in argv) else 7200


class SubprocessExecutor:
    def __init__(self, *, verbose: bool = False) -> None:
        self.verbose = verbose

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        output_dir: Path,
        label: str,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        log_path = output_dir / f"{label}.log"
        if self.verbose:
            command = (
                subprocess.list2cmdline(argv)
                if os.name == "nt"
                else shlex.join(argv)
            )
            print(f"[validation] {label}: {command}", flush=True)
        started = time.time_ns()
        baseline_children = _subreaper_children()
        process: subprocess.Popen[Any] | None = None
        windows_job: _WindowsJob | None = None
        output_thread: threading.Thread | None = None
        output_errors: list[BaseException] = []
        failure_message = ""
        command_env = dict(os.environ)
        if env is not None:
            command_env.update(env)
        command_env["OVRTX_ACTIVE_CUDA_GPUS"] = "0"
        with log_path.open("wb") as log:
            try:
                if os.name == "nt":
                    process = subprocess.Popen(
                        list(argv),
                        cwd=cwd,
                        env=command_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                    try:
                        windows_job = _WindowsJob(process)
                    except OSError:
                        process.kill()
                        process.wait()
                        raise
                else:
                    previous_mask = signal.pthread_sigmask(
                        signal.SIG_BLOCK, TERMINATION_SIGNALS
                    )
                    try:
                        process = subprocess.Popen(
                            list(argv),
                            cwd=cwd,
                            env=command_env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            preexec_fn=lambda: signal.pthread_sigmask(
                                signal.SIG_SETMASK, previous_mask
                            ),
                        )
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                output_thread = threading.Thread(
                    target=_tee_output,
                    args=(process.stdout, log, self.verbose, output_errors),
                )
                output_thread.start()
                try:
                    exit_status = process.wait(
                        timeout=_command_timeout(argv)
                    )
                except subprocess.TimeoutExpired as error:
                    _terminate_process_group(
                        process,
                        baseline_children=baseline_children,
                        windows_job=windows_job,
                    )
                    failure_message = f"TimeoutExpired: {error}\n"
                    exit_status = 124
                else:
                    _terminate_process_group(
                        process,
                        grace_seconds=0,
                        baseline_children=baseline_children,
                        windows_job=windows_job,
                    )
            except OSError as error:
                failure_message = f"{type(error).__name__}: {error}\n"
                exit_status = 127
            except BaseException:
                if process is not None:
                    _terminate_process_group(
                        process,
                        baseline_children=baseline_children,
                        windows_job=windows_job,
                    )
                raise
            finally:
                if output_thread is not None:
                    output_thread.join()
            if output_errors:
                raise output_errors[0]
            if failure_message:
                log.write(failure_message.encode())
                if self.verbose:
                    print(failure_message, end="", flush=True)
        ended = time.time_ns()
        if self.verbose:
            print(f"[validation] {label}: exit {exit_status}", flush=True)
        return CommandResult(
            argv=list(argv),
            cwd=str(cwd),
            exit_status=exit_status,
            started_at_ns=started,
            ended_at_ns=ended,
            stdout_path=str(log_path),
            stderr_path=str(log_path),
        )


def _terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = 5,
    baseline_children: set[int] | None = None,
    windows_job: _WindowsJob | None = None,
) -> None:
    if windows_job is not None:
        windows_job.close()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return
    adopted = (
        _child_pids(os.getpid()) - baseline_children
        if baseline_children is not None
        else set()
    )
    groups = _process_groups(process.pid, adopted)
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    for _ in range(100):
        adopted = (
            _child_pids(os.getpid()) - baseline_children
            if baseline_children is not None
            else adopted
        )
        groups = _process_groups(process.pid, adopted)
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        for pid in adopted:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        if baseline_children is None or not (
            _child_pids(os.getpid()) - baseline_children
        ):
            return
    raise RuntimeError("could not terminate all command descendants")


def _process_groups(root_pid: int, adopted: set[int] | None = None) -> list[int]:
    groups = [root_pid]
    seen: set[int] = set()
    pending = [root_pid, *(adopted or set())]
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if group not in groups:
            groups.append(group)
        pending.extend(_child_pids(pid))
    return groups


def _child_pids(pid: int) -> set[int]:
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return set()
    return {int(child) for child in children.split()}


def _subreaper_children() -> set[int]:
    if os.name == "nt":
        return set()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return _child_pids(os.getpid())


class _WindowsJob:
    """Kill a Windows command and every descendant when its boundary closes."""

    def __init__(self, process: subprocess.Popen[Any]) -> None:
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            kernel32.CloseHandle(job)
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process._handle)):
            kernel32.CloseHandle(job)
            raise ctypes.WinError(ctypes.get_last_error())
        self._kernel32 = kernel32
        self._job = job

    def close(self) -> None:
        if self._job:
            self._kernel32.CloseHandle(self._job)
            self._job = None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_file_result(path: Path, exit_status: int) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {"status": "invalid", "error": str(error)}
    if not isinstance(value, Mapping):
        return {"status": "invalid", "error": "result record must be an object"}
    if (
        exit_status != 0
        and isinstance(value, Mapping)
        and (value.get("status") == "pass" or value.get("outcome") == "pass")
    ):
        return {"status": "invalid", "error": f"command exited {exit_status}"}
    return value


def _result_file_result(path: Path, exit_status: int) -> Any:
    if not path.is_file():
        return {
            "status": "missing",
            "error": f"command exited {exit_status} without result evidence",
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {"status": "invalid", "error": str(error)}
    if (
        exit_status != 0
        and isinstance(value, Mapping)
        and value.get("status")
        in {
            "pass",
            "pass-real",
        }
    ):
        return {
            "status": "invalid",
            "error": "command reported pass but exited nonzero",
        }
    return value


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    excluded = {"result.json"}
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.relative_to(root).as_posix() not in excluded
    ]


def _public_dependency_evidence(command: CommandResult, label: str) -> dict[str, Any]:
    message = f"{label}: exit_status={command.exit_status}\n"
    Path(command.stdout_path).write_text(message, encoding="utf-8")
    Path(command.stderr_path).write_text(message, encoding="utf-8")
    evidence = command.evidence()
    evidence["argv"] = ["dependencies", label]
    evidence["cwd"] = "."
    return evidence


def _dependency_project_record(project: Path) -> None:
    required = ("conanfile.py", "conan.lock", "recipes")
    missing = [relative for relative in required if not (project / relative).exists()]
    if missing:
        raise RuntimeError(
            "dependency project is missing required paths: " + ", ".join(missing)
        )
    _selected_dependency_profiles(project)
    _dependency_recipe_dirs(project)
    if _git(project, "status", "--porcelain"):
        raise RuntimeError("dependency project must be clean")


def _dependency_install_commands(project: Path, destination: Path) -> list[list[str]]:
    build_profile, host_profile = _selected_dependency_profiles(project)
    recipes = _dependency_recipe_dirs(project)
    commands = [["conan", "export", f"recipes/{recipe.name}"] for recipe in recipes]
    commands.append(
        [
            "conan",
            "install",
            ".",
            "--profile:build",
            str(Path("profiles") / build_profile.name),
            "--profile:host",
            str(Path("profiles") / host_profile.name),
            "--lockfile",
            "conan.lock",
            "--lockfile-partial",
            "--build=missing",
            "--deployer=direct_deploy",
            "--deployer-folder",
            str(destination),
        ]
    )
    return commands


def _selected_dependency_profiles(project: Path) -> tuple[Path, Path]:
    expected_os, expected_arch = _current_conan_platform()
    profiles = project / "profiles"
    if not profiles.is_dir():
        raise RuntimeError("dependency project is missing required paths: profiles")
    host = profiles / f"{expected_os.lower()}-{expected_arch}"
    build = host.with_name(f"{host.name}-build")
    if not host.is_file() or not build.is_file():
        raise RuntimeError(
            "dependency project must provide host and build profiles for "
            f"Conan os={expected_os}, arch={expected_arch}"
        )
    return build, host


def _current_conan_platform() -> tuple[str, str]:
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return "Windows", "x64"
    if sys.platform.startswith("linux") and machine in {"amd64", "x86_64"}:
        return "Linux", "x64"
    if sys.platform.startswith("linux") and machine in {"aarch64", "arm64"}:
        return "Linux", "aarch64"
    raise RuntimeError(f"unsupported validation platform: {sys.platform}-{machine}")


def _dependency_recipe_dirs(project: Path) -> list[Path]:
    recipes = sorted(
        path for path in (project / "recipes").iterdir()
        if (path / "conanfile.py").is_file()
    )
    if not recipes:
        raise RuntimeError("dependency project is missing project recipes")
    return recipes


def _dependency_runtime_paths(destination: Path) -> dict[str, str]:
    deployment = destination.expanduser().resolve() / DIRECT_DEPLOY_ROOT
    missing = [name for name in DEPLOYED_ROOT_NAMES if not (deployment / name).is_dir()]
    if missing:
        raise RuntimeError("dependency deployment is incomplete: " + ", ".join(missing))
    roots = {name: str(deployment / name) for name in DEPLOYED_ROOT_NAMES}
    return _dependency_runtime_paths_from_roots(roots)


def _dependency_runtime_paths_from_roots(roots: dict[str, str]) -> dict[str, str]:
    ovrtx_bridge = Path(roots["ovrtx-bridge-server"])
    ovphysx_bridge = Path(roots["ovphysx-bridge-server"])
    return {
        **roots,
        "ovrtx_worker": str(
            _dependency_runtime_executable(
                ovrtx_bridge / "bin", "ovrtx-bridge-server"
            )
        ),
        "ovrtx_package_root": str(ovrtx_bridge),
        "ovrtx_blender_client": roots["ovrtx-bridge-client"],
        "ovphysx_server": str(
            _dependency_runtime_executable(
                ovphysx_bridge / "bin", "ovphysx-bridge-server"
            )
        ),
        "ovphysx_blender_client": roots["ovphysx-bridge-client"],
        "ovphysx_bridge_root": roots["ovphysx-bridge-server"],
    }


def _dependency_runtime_executable(directory: Path, stem: str) -> Path:
    for candidate in (directory / stem, directory / f"{stem}.exe"):
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"dependency deployment is missing executable: {stem}")


def _prepare_dependency(
    executor: Executor,
    repo: Path,
    project: Path,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    commands: list[dict[str, Any]] = []
    try:
        _dependency_project_record(project)
    except (OSError, RuntimeError) as error:
        return {"status": "failed", "error": str(error)}, {}, commands
    failed_command: CommandResult | None = None
    failure_error: str | None = None
    deployment = repo / "out" / "validation-dependencies"
    runtime: dict[str, str] = {}
    for index, argv in enumerate(_dependency_install_commands(project, deployment)):
        command = executor.run(
            argv, cwd=project, output_dir=output, label=f"conan-{index}"
        )
        commands.append(_public_dependency_evidence(command, f"install-{index}"))
        if command.exit_status:
            failed_command = command
            break
    if failed_command is None:
        try:
            runtime = _dependency_runtime_paths(deployment)
        except (OSError, RuntimeError) as error:
            failure_error = str(error)
    status = "pass" if failed_command is None and runtime else "failed"
    if failed_command is not None:
        failure_error = "dependency installation failed"
    result = {"status": status, "deployment_verified": status == "pass"}
    if status == "failed":
        result["error"] = failure_error or "dependency installation failed"
    return result, runtime, commands

def _golden_evidence(
    case: str,
    root: Path,
    *,
    expected_presentation: str | None = COLOR_PRESENTATION,
) -> dict[str, Any]:
    metadata_path = root / "metadata.json"
    frame_path = root / "frame.png"
    try:
        metadata = _read_json(metadata_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {
            "status": "unavailable",
            "reason": f"golden metadata is invalid: {error}",
        }
    width, height = VISUAL_DIMENSIONS[case]
    frame = metadata.get("frame") if isinstance(metadata, Mapping) else None
    size = metadata.get("size") if isinstance(metadata, Mapping) else None
    valid = (
        isinstance(metadata, Mapping)
        and metadata.get("schema_version") == 1
        and type(metadata.get("schema_version")) is int
        and metadata.get("kind") == GOLDEN_KIND
        and metadata.get("case") == case
        and size == {"width": width, "height": height}
        and (
            metadata.get("presentation") in SUPPORTED_COLOR_PRESENTATIONS
            if expected_presentation is None
            else metadata.get("presentation") == expected_presentation
        )
        and isinstance(frame, Mapping)
        and set(frame) == {"path", "sha256"}
        and frame.get("path") == "frame.png"
        and DIGEST_RE.fullmatch(str(frame.get("sha256"))) is not None
    )
    if not valid:
        return {
            "status": "unavailable",
            "reason": "golden metadata does not match the fixed case",
        }
    approval = metadata.get("approval")
    if not _golden_approval_is_valid(approval):
        return {
            "status": "unavailable",
            "reason": (
                "golden is not explicitly approved: approval requires a non-empty "
                "approved_by and a valid UTC approved_at_utc"
            ),
        }
    try:
        frame_sha256 = _file_sha256(frame_path)
        metadata_sha256 = _file_sha256(metadata_path)
    except OSError as error:
        return {
            "status": "unavailable",
            "reason": f"golden artifact is missing: {error}",
        }
    if frame_sha256 != frame["sha256"] or metadata_sha256 is None:
        return {
            "status": "unavailable",
            "reason": "golden artifact digest does not match metadata",
        }
    return {
        "status": "pass",
        "metadata": dict(metadata),
        "metadata_sha256": metadata_sha256,
        "frame_sha256": frame_sha256,
    }


def _golden_approval_is_valid(approval: Any) -> bool:
    if not isinstance(approval, Mapping):
        return False
    approved_by = approval.get("approved_by")
    approved_at_utc = approval.get("approved_at_utc")
    if not isinstance(approved_by, str) or not approved_by.strip():
        return False
    if not isinstance(approved_at_utc, str):
        return False
    try:
        timestamp = datetime.fromisoformat(approved_at_utc.replace("Z", "+00:00"))
    except ValueError:
        return False
    return timestamp.tzinfo == timezone.utc


def _worker_command(runtime: Any) -> str:
    if not isinstance(runtime, Mapping):
        return ""
    command = runtime.get("ovrtx_worker_command")
    if isinstance(command, str) and command.strip():
        return command
    executable = runtime.get("ovrtx_worker")
    package_root = runtime.get("ovrtx_package_root")
    if not isinstance(executable, str) or not isinstance(package_root, str):
        return ""
    arguments = [
        executable,
        "--address",
        "127.0.0.1",
        "--port",
        "50051",
        "--package-root",
        package_root,
    ]
    return (
        subprocess.list2cmdline(arguments)
        if os.name == "nt"
        else shlex.join(arguments)
    )


def _blender_python(
    executor: Executor, repo: Path, output_dir: Path
) -> tuple[str | None, CommandResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = executor.run(
        [
            "blender",
            "--background",
            "--factory-startup",
            "--python-expr",
            "import sys; print('OVRTX_BLENDER_PYTHON=' + sys.executable)",
        ],
        cwd=repo,
        output_dir=output_dir,
        label="blender-python",
    )
    return (
        _blender_python_result(Path(command.stdout_path), command.exit_status),
        command,
    )


def _blender_python_result(path: Path, exit_status: int) -> str | None:
    marker = "OVRTX_BLENDER_PYTHON="
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    value = next((line[len(marker) :] for line in lines if line.startswith(marker)), "")
    if exit_status != 0 or not _is_absolute_path(value):
        return None
    return value


def _probe_failure(
    check_id: str,
    result: Any,
    output_dir: Path | None = None,
    *,
    allow_legacy_live_transform: bool = False,
) -> str:
    if (
        not isinstance(result, Mapping)
        or type(result.get("schema_version")) is not int
        or result.get("schema_version") != 1
        or result.get("runtime_artifacts_available") is not True
    ):
        return f"{check_id} evidence is incomplete"
    if check_id == "live-transform":
        if result.get("artifact_id") != "ovrtx-live-transform-probe" or result.get(
            "status"
        ) not in {"pass", "pass-real"}:
            return "live-transform probe did not pass"
        fixture = result.get("fixture")
        probe = result.get("probe")
        correctness = probe.get("correctness") if isinstance(probe, Mapping) else None
        required_correctness = {
            "depsgraph_edit_found",
            "workflow_accepted",
            "workflow_unsupported_recorded",
            "tick_values_written",
            "update_values_written",
            "edit_records_written",
            "per_edit_values_written_recorded",
            "per_edit_unsupported_recorded",
            "same_ovrtx_session",
            "whole_scene_export_avoided",
            "render_results_differ",
        }
        if (
            not isinstance(fixture, Mapping)
            or fixture.get("id") != "demo_stair_drop_1280x720"
            or DIGEST_RE.fullmatch(str(fixture.get("fixture_usd_sha256"))) is None
            or DIGEST_RE.fullmatch(str(fixture.get("fixture_content_sha256"))) is None
            or not isinstance(correctness, Mapping)
            or set(correctness) != required_correctness
            or any(value is not True for value in correctness.values())
        ):
            return "live-transform correctness evidence is invalid"
        arguments = result.get("arguments")
        if not isinstance(arguments, Mapping) or output_dir is None:
            return "live-transform render evidence is invalid"
        presentation = _live_transform_color_presentation(
            result, allow_legacy_ldr=allow_legacy_live_transform
        )
        if presentation is None or (
            "validation_color_presentation" in result
            and result.get("validation_color_presentation") != presentation
        ):
            return "live-transform render evidence is invalid"
        width, height = arguments.get("width"), arguments.get("height")
        if presentation == LDR_COLOR_PRESENTATION:
            resolution = fixture.get("resolution")
            if not isinstance(resolution, Mapping):
                return "live-transform render evidence is invalid"
            width, height = resolution.get("width"), resolution.get("height")
        if type(width) is not int or type(height) is not int:
            return "live-transform render evidence is invalid"
        initial_pixels = _validated_probe_image(
            probe.get("initial_render_result"),
            output_dir / "initial.png",
            width,
            height,
            require_rgba_digest=True,
        )
        post_pixels = _validated_probe_image(
            probe.get("post_edit_render_result"),
            output_dir / "post-edit.png",
            width,
            height,
            require_rgba_digest=True,
        )
        if (
            initial_pixels is None
            or post_pixels is None
            or _visible_pixels(initial_pixels) == _visible_pixels(post_pixels)
        ):
            return "live-transform render evidence is invalid"
        return ""
    return f"{check_id} is not repository validation"


def _live_transform_color_presentation(
    result: Mapping[str, Any], *, allow_legacy_ldr: bool
) -> str | None:
    probe = result.get("probe")
    if not isinstance(probe, Mapping):
        return None
    reads = [
        probe.get("initial_render_result"),
        probe.get("post_edit_render_result"),
    ]
    if all(
        isinstance(read, Mapping)
        and _hdr_probe_presentation_valid(_probe_color_presentation(read))
        for read in reads
    ):
        return COLOR_PRESENTATION
    legacy_fields = {
        "path",
        "width",
        "height",
        "size_bytes",
        "sha256",
        "rgba8_sha256",
        "completed_samples",
        "session_completed_samples",
        "simulation_time_ns",
    }
    if allow_legacy_ldr and all(
        isinstance(read, Mapping)
        and set(read) == legacy_fields
        and type(read["width"]) is int
        and type(read["height"]) is int
        and read["width"] > 0
        and read["height"] > 0
        for read in reads
    ):
        return LDR_COLOR_PRESENTATION
    return None


def _hdr_probe_presentation_valid(value: Any) -> bool:
    expected = {
        "requested_mode": COLOR_PRESENTATION,
        "active_mode": COLOR_PRESENTATION,
        "status": "current_behavior",
        "frame_format": "rgba16f",
        "frame_color_mode": "scene_linear",
        "render_var": "HdrColor",
        "result_frame_format": "rgba16f",
        "result_frame_color_mode": "scene_linear",
        "result_render_var": "HdrColor",
    }
    return isinstance(value, Mapping) and all(
        value.get(key) == expected_value for key, expected_value in expected.items()
    )


def _probe_color_presentation(value: Any) -> Any:
    return value.get("color_presentation") if isinstance(value, Mapping) else None


def _validated_probe_image(
    image: Any,
    path: Path,
    width: int,
    height: int,
    *,
    require_rgba_digest: bool,
) -> bytes | None:
    if (
        not isinstance(image, Mapping)
        or not _absolute_path_matches_tail(image.get("path"), path, 3)
        or image.get("width") != width
        or image.get("height") != height
        or type(image.get("size_bytes")) is not int
        or image["size_bytes"] <= 0
        or DIGEST_RE.fullmatch(str(image.get("sha256"))) is None
    ):
        return None
    try:
        data = path.read_bytes()
        decoded_width, decoded_height, pixels = visual_comparison._read_png_rgba8(path)
    except (OSError, ValueError, zlib.error):
        return None
    if not visual_comparison._nonblank(pixels):
        return None
    if (
        (decoded_width, decoded_height) != (width, height)
        or image["size_bytes"] != len(data)
        or image["sha256"] != hashlib.sha256(data).hexdigest()
    ):
        return None
    if (
        require_rgba_digest
        and image.get("rgba8_sha256") != hashlib.sha256(pixels).hexdigest()
    ):
        return None
    return pixels


def _absolute_path_matches_tail(value: Any, path: Path, parts: int) -> bool:
    if not isinstance(value, str):
        return False
    recorded = _portable_path(value)
    return (
        recorded.is_absolute()
        and len(recorded.parts) >= parts
        and recorded.parts[-parts:] == path.parts[-parts:]
    )


def _visible_pixels(pixels: bytes) -> bytes:
    visible = bytearray()
    for index in range(0, len(pixels), 4):
        alpha = pixels[index + 3]
        visible.extend(
            round(pixels[index + channel] * alpha / 255) for channel in range(3)
        )
    return bytes(visible)


def _probe_fixture_identity(result: Any) -> Any:
    if not isinstance(result, Mapping):
        return None
    fixture = result.get("fixture")
    if not isinstance(fixture, Mapping):
        return None
    return {
        "id": fixture.get("id"),
        "fixture_usd_sha256": fixture.get("fixture_usd_sha256"),
        "fixture_content_sha256": fixture.get("fixture_content_sha256"),
        "camera_prim_path": fixture.get("camera_prim_path"),
        "render_product_prim_path": fixture.get("render_product_path"),
    }


def _fixture_manifest_record(
    repo: Path, fixture_id: str
) -> tuple[dict[str, Any], Path] | None:
    try:
        fixture = json.loads(
            (repo / "tests" / "fixtures" / fixture_id / "spec.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(fixture, dict) or fixture.get("id") != fixture_id:
            return None
        identity = {
            "id": fixture_id,
            "fixture_usd_sha256": fixture["fixture_usd_sha256"],
            "fixture_content_sha256": fixture["fixture_content_sha256"],
            "camera_prim_path": fixture["camera_prim_path"],
            "render_product_prim_path": fixture["render_product_prim_path"],
        }
        fixture_path_value = fixture.get("fixture_usd_path")
        runtime_files = fixture.get("runtime_files")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ):
        return None
    if any(
        DIGEST_RE.fullmatch(str(identity[key])) is None
        for key in ("fixture_usd_sha256", "fixture_content_sha256")
    ):
        return None
    if not isinstance(fixture_path_value, str) or not fixture_path_value:
        return None
    if not isinstance(runtime_files, list) or not runtime_files:
        return None
    fixture_path = (repo / "tests" / fixture_path_value).resolve()
    if (
        not fixture_path.is_file()
        or not fixture_path.is_relative_to((repo / "tests").resolve())
        or _file_sha256(fixture_path) != identity["fixture_usd_sha256"]
    ):
        return None
    runtime_identity: list[tuple[str, str]] = []
    for item in runtime_files:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or DIGEST_RE.fullmatch(str(item.get("sha256"))) is None
        ):
            return None
        path = (repo / "tests" / item["path"]).resolve()
        if (
            not path.is_file()
            or not path.is_relative_to((repo / "tests").resolve())
            or _file_sha256(path) != item["sha256"]
        ):
            return None
        runtime_identity.append((item["path"], item["sha256"]))
    if (fixture_path_value, identity["fixture_usd_sha256"]) not in runtime_identity:
        return None
    digest = hashlib.sha256(b"ovrtx-fixture-content-v2\0")
    for path, sha256 in sorted(runtime_identity):
        digest.update(path.encode("utf-8") + b"\0")
        digest.update(sha256.encode("ascii") + b"\0")
    if digest.hexdigest() != identity["fixture_content_sha256"]:
        return None
    return identity, fixture_path


def _load_result(path: Path, command: CommandResult) -> Any:
    return _result_file_result(path, command.exit_status)


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _json_from_stdout(command: CommandResult) -> Any:
    return _json_file_result(Path(command.stdout_path), command.exit_status)


def _command_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != COMMAND_KEYS:
        return False
    argv = value["argv"]
    start = value["started_at_ns"]
    end = value["ended_at_ns"]
    return (
        isinstance(argv, list)
        and bool(argv)
        and all(isinstance(item, str) and item for item in argv)
        and isinstance(value["cwd"], str)
        and bool(value["cwd"])
        and type(value["exit_status"]) is int
        and type(start) is int
        and type(end) is int
        and 0 <= start <= end
        and isinstance(value["stdout_path"], str)
        and bool(value["stdout_path"])
        and isinstance(value["stderr_path"], str)
        and bool(value["stderr_path"])
    )


def _repository(root: Path) -> Path:
    repository = root.expanduser().resolve()
    if not repository.is_dir():
        raise ValidationError(f"source root is not a Git repository: {repository}")
    try:
        is_work_tree = _git(repository, "rev-parse", "--is-inside-work-tree")
    except ValidationError:
        is_work_tree = "false"
    if is_work_tree != "true":
        raise ValidationError(f"source root is not a Git repository: {repository}")
    return repository


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise ValidationError(completed.stderr.strip() or "git identity check failed")
    return completed.stdout.strip()


def _empty_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValidationError(f"output must be an existing empty directory: {resolved}")
    if any(resolved.iterdir()):
        raise ValidationError(f"output must be empty: {resolved}")
    return resolved


def _validator_digest(entrypoint: Path) -> str:
    scripts = entrypoint.resolve().parent
    digest = hashlib.sha256(b"ov-blender-example-validator-v2\0")
    names = {
        entrypoint.name,
        "navigation.py",
        "validation.py",
        "visual_comparison.py",
    }
    for name in sorted(names):
        path = scripts / name
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
