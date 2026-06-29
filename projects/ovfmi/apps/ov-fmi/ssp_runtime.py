# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# SSP runtime: load and step an SSP (System Structure and Parameterization)
# archive as a single aggregate "super FMU" that exposes system-level connectors
# as its inputs/outputs.

import os
import shutil
import sys
import tempfile
import zipfile
from io import BytesIO
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET

from fmi_parser import FmuParserInstance
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave
from fmpy.ssp.ssd import read_ssd, find_components, find_connectors, get_connections


def _safe_temp_dir(prefix: str) -> str:
    tempdir = tempfile.mkdtemp(prefix=prefix)
    if sys.platform.startswith("win") and "~" in tempdir:
        import ntpath
        tempdir = ntpath.realpath(tempdir)
    return tempdir


def _extract_archive(filename: str, prefix: str) -> str:
    return extract(filename, unzipdir=_safe_temp_dir(prefix))


def _default_system_connector_value(name: str | None) -> float:
    """Return conservative defaults for unconnected SSP system inputs."""
    if name is not None and name.lower() in {"enable", "enabled"}:
        return 1.0
    return 0.0




def _find_staged_windows_fmu2_binary(model_identifier: str) -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = []
    for config in ("Release", "RelWithDebInfo", "Debug"):
        candidates.append(repo_root / "build" / config / f"{model_identifier}.dll")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _add_windows_binary_to_fmu(fmu_bytes: bytes) -> bytes | None:
    with zipfile.ZipFile(BytesIO(fmu_bytes), "r") as fmu:
        model_xml = fmu.read("modelDescription.xml")
        root = ET.fromstring(model_xml)
        co_sim = root.find("CoSimulation")
        if co_sim is None:
            return None

        model_identifier = co_sim.get("modelIdentifier")
        if not model_identifier:
            return None

        binary_name = f"binaries/win64/{model_identifier}.dll"
        if binary_name in fmu.namelist():
            return None

        staged = _find_staged_windows_fmu2_binary(model_identifier)
        if staged is None:
            return None

        out_bytes = BytesIO()
        with zipfile.ZipFile(out_bytes, "w", zipfile.ZIP_DEFLATED) as out_fmu:
            for info in fmu.infolist():
                out_fmu.writestr(info, fmu.read(info.filename))
            out_fmu.write(staged, binary_name)
        return out_bytes.getvalue()


def prepare_ssp_archive_for_current_platform(ssp_path: str | Path) -> str:
    """Return an SSP path usable on the current platform.

    The checked-in demo SSP currently carries Linux FMU binaries.  On Windows,
    use locally built FMI 2.0 DLLs from build/<config>/ and create a temporary
    SSP archive whose nested FMUs also contain binaries/win64/*.dll.  The
    original archive remains unchanged.
    """
    if not sys.platform.startswith("win"):
        return str(ssp_path)

    ssp_path = Path(ssp_path)
    replacements: dict[str, bytes] = {}

    with zipfile.ZipFile(ssp_path, "r") as ssp:
        for name in ssp.namelist():
            if not name.endswith(".fmu"):
                continue
            patched = _add_windows_binary_to_fmu(ssp.read(name))
            if patched is not None:
                replacements[name] = patched

        if not replacements:
            return str(ssp_path)

        out_dir = Path(_safe_temp_dir("ov_fmi_ssp_archive_"))
        out_path = out_dir / ssp_path.name
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as out_ssp:
            for info in ssp.infolist():
                data = replacements.get(info.filename)
                if data is None:
                    data = ssp.read(info.filename)
                out_ssp.writestr(info, data)

    return str(out_path)


class SspRuntimeInstance:
    """Wraps an SSP as a single FMU-like instance for the FMIRuntime framework.

    From the outside (OvrtxInputHead, OvrtxOutputTail, PhysxInputHead) this
    behaves identically to FmuRuntimeInstance:
      - get_parser_instance() returns the parser metadata
      - step() accepts inputs dict, returns a numpy structured array row
      - set_start_values() sets initial parameter values
    """

    def __init__(self, parser_instance: FmuParserInstance, ssp_path: str):
        self._parser_instance = parser_instance
        self._ssp_path = ssp_path
        self._unzip_dir = None
        self._components = []
        self._connectors = []
        self._connections = []
        self._system_inputs = []   # system-level input connectors
        self._system_outputs = []  # system-level output connectors
        self._initialized = False
        self._last_successful_time = 0.0
        self._start_values = {}

        self._load_ssp()

    def _load_ssp(self):
        """Extract the SSP and instantiate all internal FMUs."""
        # Extract SSP archive
        self._unzip_dir = _extract_archive(self._ssp_path, "ov_fmi_ssp_")

        # Parse the SSD
        ssd = read_ssd(self._ssp_path)
        self._ssd = ssd

        # Add path info to the system tree
        from fmpy.ssp.ssd import add_tree_info
        add_tree_info(ssd.system)

        self._components = find_components(ssd.system)
        self._connectors = find_connectors(ssd.system)
        self._connections = get_connections(ssd.system)

        # Resolve connections (trace back through subsystems)
        connections_reversed = {}
        for a, b in self._connections:
            connections_reversed[b] = a

        new_connections = []
        for a, b in self._connections:
            while hasattr(a, 'parent') and hasattr(a.parent, 'parent') and a.parent.parent is not None:
                if a in connections_reversed:
                    a = connections_reversed[a]
                else:
                    break
            new_connections.append((a, b))
        self._connections = new_connections

        # Classify system-level connectors
        for connector in ssd.system.connectors:
            connector.value = _default_system_connector_value(connector.name)
            if connector.kind == 'input':
                self._system_inputs.append(connector)
            elif connector.kind == 'output':
                self._system_outputs.append(connector)

        # Initialize all connector values to 0
        for connector in self._connectors:
            connector.value = 0.0

        # Instantiate FMUs
        for component in self._components:
            self._instantiate_component(component)

        print(f"SSP loaded: {len(self._components)} components, "
              f"{len(self._system_inputs)} inputs, {len(self._system_outputs)} outputs")

    def _instantiate_component(self, component):
        """Instantiate a single FMU component within the SSP."""
        fmu_filename = os.path.join(self._unzip_dir, component.source)
        component.unzipdir = _extract_archive(fmu_filename, "ov_fmi_ssp_fmu_")

        model_description = read_model_description(fmu_filename, validate=False)
        if model_description.coSimulation is None:
            raise RuntimeError(f"{component.source} does not support co-simulation.")

        self._add_bundled_windows_binary_if_needed(
            Path(component.unzipdir),
            model_description.coSimulation.modelIdentifier,
        )

        # Build variable lookup
        component.variables = {}
        for variable in model_description.modelVariables:
            component.variables[variable.name] = variable

        fmu_kwargs = {
            'guid': model_description.guid,
            'unzipDirectory': component.unzipdir,
            'modelIdentifier': model_description.coSimulation.modelIdentifier,
            'instanceName': component.name,
        }

        component.fmu = FMU2Slave(**fmu_kwargs)
        component.fmu.instantiate()
        component.fmu.setupExperiment(startTime=0.0)
        component.fmu.enterInitializationMode()
        component.fmu.exitInitializationMode()
        component.model_description = model_description

    def _add_bundled_windows_binary_if_needed(self, unzipdir: Path, model_identifier: str):
        if not sys.platform.startswith("win"):
            return

        expected = unzipdir / "binaries" / "win64" / f"{model_identifier}.dll"
        if expected.exists():
            return

        staged = self._find_staged_windows_fmu2_binary(model_identifier)
        if staged is None:
            return

        expected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged, expected)
        print(f"Using staged Windows SSP FMU binary: {staged}")

    def _find_staged_windows_fmu2_binary(self, model_identifier: str) -> Path | None:
        return _find_staged_windows_fmu2_binary(model_identifier)

    def get_parser_instance(self) -> FmuParserInstance:
        return self._parser_instance

    def set_start_values(self, start_values: dict):
        self._start_values = start_values

    def step(self, filename: str, inputs: dict, outputs: list, time: float):
        """Step the entire SSP by one communication interval.

        Args:
            filename: ignored (SSP manages its own FMUs)
            inputs: {connector_name: value} for system-level inputs
            outputs: list of system-level output connector names to read
            time: current simulation time (target end time for this step)

        Returns:
            numpy structured array row with 'time' + output fields, or None
        """
        if time <= self._last_successful_time:
            return None

        step_size = time - self._last_successful_time

        # Set system-level inputs from the provided dict
        for connector in self._system_inputs:
            if connector.name in inputs:
                connector.value = float(inputs[connector.name])

        # Propagate connections: system inputs → component inputs,
        # and previous component outputs → downstream component inputs
        for start_connector, end_connector in self._connections:
            end_connector.value = start_connector.value

        # Step all components in order
        for component in self._components:
            # Set inputs for this component
            self._set_component_inputs(component)
            # Step this component
            component.fmu.doStep(
                currentCommunicationPoint=self._last_successful_time,
                communicationStepSize=step_size,
            )
            # Get outputs from this component
            self._get_component_outputs(component)
            # Propagate connections again so downstream components see updated values
            for start_connector, end_connector in self._connections:
                end_connector.value = start_connector.value

        # Build result structured array
        self._last_successful_time = time

        # Create dtype: time + all requested outputs
        dtype_list = [('time', np.float64)]
        for name in outputs:
            dtype_list.append((name, np.float64))

        row_values = [time]
        for name in outputs:
            # Find the system-level output connector
            val = 0.0
            for connector in self._system_outputs:
                if connector.name == name:
                    val = connector.value
                    break
            row_values.append(val)

        result = np.array([tuple(row_values)], dtype=np.dtype(dtype_list))
        return result[0]

    def _set_component_inputs(self, component):
        """Set input values on an FMU component from its connectors."""
        for connector in component.connectors:
            if connector.kind == 'input':
                variable = component.variables.get(connector.name)
                if variable is not None:
                    component.fmu.setReal(
                        [variable.valueReference], [connector.value]
                    )

    def _get_component_outputs(self, component):
        """Read output values from an FMU component into its connectors."""
        for connector in component.connectors:
            if connector.kind == 'output':
                variable = component.variables.get(connector.name)
                if variable is not None:
                    values = component.fmu.getReal([variable.valueReference])
                    connector.value = values[0]

    def destroy(self):
        """Terminate and free all internal FMUs, clean up temp dirs."""
        for component in self._components:
            try:
                component.fmu.terminate()
                component.fmu.freeInstance()
            except Exception:
                pass
            if hasattr(component, 'unzipdir') and component.unzipdir:
                try:
                    shutil.rmtree(component.unzipdir)
                except Exception:
                    pass

        if self._unzip_dir:
            try:
                shutil.rmtree(self._unzip_dir)
            except Exception:
                pass
            self._unzip_dir = None

    def reset(self):
        """Reset all internal FMUs."""
        self._last_successful_time = 0.0
        for component in self._components:
            component.fmu.reset()
            component.fmu.setupExperiment(startTime=0.0)
            component.fmu.enterInitializationMode()
            component.fmu.exitInitializationMode()
        for connector in self._connectors:
            connector.value = 0.0

    def resume(self):
        pass


class SspRuntimeExtracted:
    """Analogous to FmuRuntimeExtractedFMU but for SSP archives.

    Manages a single SSP archive and its runtime instances.
    """

    def __init__(self, ssp_path: str):
        self._ssp_path = ssp_path
        self._runtime_instances: list[SspRuntimeInstance] = []

    def get_runtime_instances(self) -> list[SspRuntimeInstance]:
        return self._runtime_instances

    def get_unzip_dir(self) -> str:
        # SSP instances manage their own extraction; return empty string
        return ""

    def add_instance(self, parser_instance: FmuParserInstance):
        if not parser_instance.enabled:
            return
        instance = SspRuntimeInstance(parser_instance, self._ssp_path)
        self._runtime_instances.append(instance)

    def destroy(self):
        for instance in self._runtime_instances:
            instance.destroy()
        self._runtime_instances.clear()
