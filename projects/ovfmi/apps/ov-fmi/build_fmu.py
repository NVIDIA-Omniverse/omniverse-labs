#!/usr/bin/env python3
"""
build_fmu.py - Build FMU and SSP archives from source directories.

This script:
  1. Compiles FMU shared libraries from C++ source files
  2. Packages each FMU source dir + compiled binary into .fmu/.fmu3 archives
  3. Packages the SSP source dir + built FMU archives into .ssp archive

All output goes to usd/ov-fmi/ next to the sample USD stages.

Usage:
    python3 apps/ov-fmi/build_fmu.py

Requirements:
    - C++ compiler (g++ or clang++)
    - FMU source dirs in fmu/ (each contains .cpp + modelDescription.xml)
    - SSP source dirs in ssp/
"""

import os
import sys
import subprocess
import struct
import zipfile
import platform
import shutil
import shlex
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FMU_SRC_DIR = REPO_ROOT / "fmu"
SSP_SRC_DIR = REPO_ROOT / "ssp"
DATA_DIR = REPO_ROOT / "usd" / "ov-fmi"
BUILD_DIR = REPO_ROOT / "apps" / "ov-fmi" / "_build"

# Platform detection for FMI directory structure
SYSTEM = platform.system()
MACHINE = platform.machine()


def _normalise_machine(machine: str) -> str:
    machine = machine.lower()
    if machine in ("amd64", "x64"):
        return "x86_64"
    if machine == "arm64":
        return "aarch64"
    return machine


ARCH = _normalise_machine(MACHINE)

if SYSTEM == "Linux":
    FMI3_PLATFORM = f"{ARCH}-linux"  # e.g. x86_64-linux
    FMI2_PLATFORM = "linux64"
    LIB_EXT = ".so"
elif SYSTEM == "Darwin":
    FMI3_PLATFORM = f"{ARCH}-darwin"
    FMI2_PLATFORM = "darwin64"
    LIB_EXT = ".dylib"
elif SYSTEM == "Windows":
    FMI3_PLATFORM = f"{ARCH}-windows"
    FMI2_PLATFORM = "win64"
    LIB_EXT = ".dll"
else:
    print(f"Unsupported platform: {SYSTEM}", file=sys.stderr)
    sys.exit(1)


_PE_MACHINE_NAMES = {
    0x014C: "x86",
    0x8664: "x86_64",
    0xAA64: "arm64",
}


def _read_pe_machine(path: Path) -> int | None:
    """Return the PE machine type for a Windows DLL, or None if not PE."""
    with path.open("rb") as f:
        dos_header = f.read(64)
        if len(dos_header) < 64 or dos_header[:2] != b"MZ":
            return None
        pe_offset = struct.unpack_from("<I", dos_header, 0x3C)[0]
        f.seek(pe_offset)
        pe_header = f.read(6)
        if len(pe_header) < 6 or pe_header[:4] != b"PE\0\0":
            return None
        return struct.unpack_from("<H", pe_header, 4)[0]


def validate_library_arch(lib_path: Path) -> bool:
    """Catch wrong-architecture Windows DLLs before packaging FMUs."""
    if SYSTEM != "Windows":
        return True

    expected = {
        "x86_64": 0x8664,
        "aarch64": 0xAA64,
    }.get(ARCH)
    if expected is None:
        return True

    try:
        actual = _read_pe_machine(lib_path)
    except OSError as exc:
        print(f"  ERROR: Could not inspect {lib_path.name}: {exc}", file=sys.stderr)
        return False

    if actual is None:
        print(f"  ERROR: {lib_path.name} is not a valid Windows PE DLL.", file=sys.stderr)
        return False

    if actual != expected:
        actual_name = _PE_MACHINE_NAMES.get(actual, f"0x{actual:04x}")
        expected_name = _PE_MACHINE_NAMES.get(expected, f"0x{expected:04x}")
        print(
            f"  ERROR: {lib_path.name} was built for {actual_name}, "
            f"but this Python/runtime expects {expected_name}.",
            file=sys.stderr,
        )
        print(
            "  Rebuild with apps\\ov-fmi\\setup.ps1 after installing "
            "Visual Studio Build Tools with the MSVC x64/x86 tools.",
            file=sys.stderr,
        )
        print(
            "  Setup output should include: Using Visual Studio environment: "
            "...\\vcvars64.bat",
            file=sys.stderr,
        )
        return False

    return True

# FMU definitions: (source_dir_path, source_cpp, output_lib_name, fmi_version)
FMUS = [
    # FMI 3.0 FMUs
    ("fmi3/bouncing_ball", "bouncing_ball.cpp", "BouncingBall", 3),
    ("fmi3/pd_controller", "pd_controller.cpp", "PDController", 3),
    ("fmi3/trajectory_generator", "trajectory_generator.cpp", "TrajectoryGenerator", 3),
    # FMI 2.0 FMUs (for SSP)
    ("fmi2/pd_controller", "pd_controller.cpp", "PDController", 2),
    ("fmi2/trajectory_generator", "trajectory_generator.cpp", "TrajectoryGenerator", 2),
    ("fmi2/presence_sensor", "presence_sensor.cpp", "PresenceSensor", 2),
    ("fmi2/conveyor_controller", "conveyor_controller.cpp", "ConveyorController", 2),
    ("fmi2/motor_drive", "motor_drive.cpp", "MotorDrive", 2),
]

# SSP definitions: (source_dir_name, output_filename, list of FMU2 lib_names to include)
SSPS = [
    ("orbit_controller", "orbit_controller.ssp", ["PDController", "TrajectoryGenerator"]),
    (
        "conveyor_demo",
        "conveyor_demo.ssp",
        ["PresenceSensor", "ConveyorController", "MotorDrive"],
    ),
]


# ============================================================
# Build Functions
# ============================================================

def compile_fmu(src_file: Path, output_lib: Path, fmi_version: int) -> bool:
    """Compile a single FMU shared library."""
    output_lib.parent.mkdir(parents=True, exist_ok=True)

    # Compiler selection
    default_cc = "cl" if SYSTEM == "Windows" else "g++"
    cc = shlex.split(os.environ.get("CXX", default_cc), posix=False)
    compiler_name = Path(cc[0]).name.lower()

    # FMI header defines
    if SYSTEM == "Windows" and compiler_name in {"cl", "cl.exe", "clang-cl", "clang-cl.exe"}:
        obj_file = output_lib.with_suffix(".obj")
        import_lib = output_lib.with_suffix(".lib")
        cmd = [
            *cc,
            "/nologo",
            "/LD",
            "/O2",
            "/std:c++17",
            "/EHsc",
            "/utf-8",
            f"/DFMI_VERSION={fmi_version}",
            f"/Fo{obj_file}",
            f"/Fe{output_lib}",
            str(src_file),
            "/link",
            "/NOLOGO",
            f"/IMPLIB:{import_lib}",
        ]
    else:
        cmd = [
            *cc,
            "-shared",
            "-O2",
            "-std=c++17",
            f"-DFMI_VERSION={fmi_version}",
            "-o", str(output_lib),
            str(src_file),
        ]
        if SYSTEM != "Windows":
            cmd.insert(2, "-fPIC")

    print(f"  Compiling: {src_file.name} -> {output_lib.name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"  ERROR: Compiler not found: {cc[0]}", file=sys.stderr)
        if SYSTEM == "Windows":
            print(
                "  Run from an x64 Visual Studio Developer Command Prompt, "
                "or set CXX to a C++17 compiler.",
                file=sys.stderr,
            )
        return False
    if result.returncode != 0:
        print(f"  ERROR: Compilation failed for {src_file.name}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False
    if not validate_library_arch(output_lib):
        return False
    return True


def package_fmu(fmu_name: str, lib_path: Path, model_desc: Path,
                fmi_version: int, output_path: Path) -> bool:
    """Package an FMU source dir + compiled binary into an archive."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmi_version == 3:
        platform_dir = f"binaries/{FMI3_PLATFORM}"
    else:
        platform_dir = f"binaries/{FMI2_PLATFORM}"

    ext = ".fmu3" if fmi_version == 3 else ".fmu"

    print(f"  Packaging: {output_path.name}")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add modelDescription.xml at root
        zf.write(model_desc, "modelDescription.xml")
        # Add binary in platform directory
        zf.write(lib_path, f"{platform_dir}/{lib_path.name}")

    return True


def package_ssp(ssp_name: str, ssp_src: Path, fmu_archives: dict,
                output_path: Path) -> bool:
    """Package an SSP source dir + FMU archives into an .ssp archive."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Packaging SSP: {output_path.name}")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add SystemStructure.ssd at root
        ssd_file = ssp_src / "SystemStructure.ssd"
        if not ssd_file.exists():
            print(f"  ERROR: {ssd_file} not found", file=sys.stderr)
            return False
        zf.write(ssd_file, "SystemStructure.ssd")

        # Add any other non-FMU files from the SSP source dir
        for f in ssp_src.rglob("*"):
            if f.is_file() and f.name != "SystemStructure.ssd":
                arcname = str(f.relative_to(ssp_src))
                zf.write(f, arcname)

        # Add FMU archives into resources/
        for fmu_name, fmu_path in fmu_archives.items():
            zf.write(fmu_path, f"resources/{fmu_name}.fmu")

    return True


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("FMU/SSP Build System")
    print(f"Platform: {SYSTEM} {MACHINE}")
    print(f"FMI 3.0 platform dir: binaries/{FMI3_PLATFORM}")
    print(f"FMI 2.0 platform dir: binaries/{FMI2_PLATFORM}")
    print("=" * 60)

    # Clean build directory
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    success = True
    built_fmus = {}  # name -> archive path
    output_artifacts = []

    # Step 1: Compile and package FMUs
    print("\n--- Step 1: Building FMUs ---")
    for fmu_dir_name, src_name, lib_name, fmi_version in FMUS:
        print(f"\n[FMU] {fmu_dir_name} (FMI {fmi_version}.0)")

        fmu_src = FMU_SRC_DIR / fmu_dir_name
        src_file = fmu_src / src_name
        if not src_file.exists():
            print(f"  ERROR: Source file not found: {src_file}", file=sys.stderr)
            success = False
            continue

        model_desc = fmu_src / "modelDescription.xml"
        if not model_desc.exists():
            print(f"  ERROR: modelDescription.xml not found: {model_desc}", file=sys.stderr)
            success = False
            continue

        # Compile shared library
        lib_output = BUILD_DIR / f"{lib_name}{LIB_EXT}"
        if not compile_fmu(src_file, lib_output, fmi_version):
            success = False
            continue

        # Package into archive
        ext = ".fmu3" if fmi_version == 3 else ".fmu"
        archive_output = BUILD_DIR / f"{lib_name}{ext}"
        if not package_fmu(lib_name, lib_output, model_desc, fmi_version, archive_output):
            success = False
            continue

        built_fmus[(lib_name, fmi_version)] = archive_output

        # Copy FMU archives to data directory.  FMI 3 samples are exploded
        # because the current runtime expects them that way; FMI 2 samples stay
        # zipped so they can also be embedded in SSPs.
        if fmi_version == 3:
            data_output = DATA_DIR / f"{lib_name}.fmu3"
            # Remove old exploded directory if present
            if data_output.exists():
                shutil.rmtree(data_output)
            # Extract to data dir (runtime expects exploded dir for FMU3)
            with zipfile.ZipFile(archive_output) as zf:
                zf.extractall(data_output)
            print(f"  Extracted to: {data_output}")
            output_artifacts.append(data_output)
        else:
            data_output = DATA_DIR / f"{lib_name}.fmu"
            shutil.copy2(archive_output, data_output)
            print(f"  Copied to: {data_output}")
            output_artifacts.append(data_output)

    # Step 2: Package SSPs
    print("\n--- Step 2: Building SSPs ---")
    for ssp_dir_name, ssp_output_name, fmu2_names in SSPS:
        print(f"\n[SSP] {ssp_dir_name}")

        ssp_src = SSP_SRC_DIR / ssp_dir_name
        if not ssp_src.exists():
            print(f"  ERROR: SSP source dir not found: {ssp_src}", file=sys.stderr)
            success = False
            continue

        # Gather FMU2 archives for this SSP
        fmu_archives = {}
        for fmu_name in fmu2_names:
            key = (fmu_name, 2)
            if key not in built_fmus:
                print(f"  ERROR: Required FMU '{fmu_name}' was not built", file=sys.stderr)
                success = False
                continue
            fmu_archives[fmu_name] = built_fmus[key]

        if not success:
            continue

        # Package SSP
        ssp_output = DATA_DIR / ssp_output_name
        if not package_ssp(ssp_dir_name, ssp_src, fmu_archives, ssp_output):
            success = False
            continue
        output_artifacts.append(ssp_output)

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("BUILD SUCCEEDED")
        print(f"\nOutput artifacts in {DATA_DIR}:")
        for f in sorted(output_artifacts):
            print(f"  {f.name}")
    else:
        print("BUILD FAILED - see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
