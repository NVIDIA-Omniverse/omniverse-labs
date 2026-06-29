# Adapted from usd_fmi (omni/source/extensions/fmi/source/fmi.usd.runtime)
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Change from upstream: import from local fmi_parser instead of fmi.usd.parser package.

from fmi_parser import FmuParserInstance
import fmpy
import os
import tempfile
import sys
from fmpy.fmi1 import FMU1Slave, FMU1Model
from fmpy.fmi2 import FMU2Slave, FMU2Model
from fmpy.fmi3 import FMU3Slave, FMU3Model, FMU3ScheduledExecution
from fmpy.model_description import ModelDescription, ScalarVariable
from fmpy.simulation import SimulationResult
import shutil
import copy
from pathlib import Path

FMPYType = FMU1Slave | FMU2Slave | FMU3Slave | FMU1Model | FMU2Model | FMU3Model | FMU3ScheduledExecution


def fmpy_resolve_model_description_shapes(model_description: ModelDescription, fmu_instance: FMPYType):
    resolved_variables: dict[int, tuple[ScalarVariable, int]] = {}
    for variable in model_description.modelVariables:
        if variable.shape is not None:
            for index, dimension in enumerate(variable.dimensions):
                if dimension.valueReference is not None:
                    if dimension.valueReference in resolved_variables:
                        _, resolved_value = resolved_variables[dimension.valueReference]
                    else:
                        refVariable = next(
                            (v for v in model_description.modelVariables
                             if v.valueReference == dimension.valueReference),
                            None,
                        )
                        if refVariable is None:
                            raise Exception(f"Variable {refVariable.name} not found in model description")
                        type = refVariable.type
                        getter = getattr(fmu_instance, "get" + type)
                        resolved_value = getter(vr=[refVariable.valueReference], nValues=1)[0]
                        resolved_variables[refVariable.valueReference] = (refVariable, resolved_value)
                    shape = list(variable.shape)
                    shape[index] = int(resolved_value)
                    variable.shape = tuple(shape)


class FmuRuntimeInstance:
    def __init__(self, parser_instance: FmuParserInstance, model_description: ModelDescription, fmu_instance: FMPYType):
        self._parser_instance = parser_instance
        self._model_description = copy.deepcopy(model_description)
        self._fmu_instance = fmu_instance
        self._must_terminate = False
        self._last_successful_time = 0
        self._prev_time = 0
        self._prev_results = []
        self._start_values = {}
        self._structural_parameters = {}

    def get_parser_instance(self) -> FmuParserInstance:
        return self._parser_instance

    def get_model_description(self) -> ModelDescription:
        return self._model_description

    def get_fmu_instance(self) -> FMPYType:
        return self._fmu_instance

    def set_start_values(self, start_values: dict):
        self._start_values = start_values

    def destroy(self):
        if self._must_terminate:
            try:
                self._fmu_instance.terminate()
            except Exception as e:
                print(e)
            finally:
                self._fmu_instance.freeInstance()
                self._fmu_instance = None
                self._must_terminate = False

    @staticmethod
    def find_result_earlier_than(time: float, results: SimulationResult) -> int:
        if len(results) == 0:
            return None
        idx = -1
        for result in results:
            idx = idx + 1
            if result[0] > time:
                idx = idx - 1
                break
        if idx == -1:
            return 0
        return idx

    def reset(self):
        self._prev_results = []
        self._structural_parameters = {}
        self._last_successful_time = 0
        self._fmu_instance.reset()
        self._must_terminate = False

    def resume(self):
        pass

    def _set_variable_values(self, variable: ScalarVariable, values):
        if not isinstance(values, list):
            values = [values]
        if variable.type == "Float32":
            self._fmu_instance.setFloat32([variable.valueReference], values)
        elif variable.type in ("Float64", "Real"):
            setter = getattr(self._fmu_instance, "setFloat64", None)
            if setter is None:
                setter = getattr(self._fmu_instance, "setReal")
            setter([variable.valueReference], values)
        elif variable.type in ("Int32", "Integer"):
            setter = getattr(self._fmu_instance, "setInt32", None)
            if setter is None:
                setter = getattr(self._fmu_instance, "setInteger")
            setter([variable.valueReference], values)
        elif variable.type == "UInt32":
            self._fmu_instance.setUInt32([variable.valueReference], values)
        elif variable.type == "Boolean":
            bool_values = [bool(v) for v in values]
            self._fmu_instance.setBoolean([variable.valueReference], bool_values)

    def step(self, filename: str, inputs, outputs: list, time: float):
        if self._prev_time > time:
            self._prev_results = []
            self._last_successful_time = 0
            self._fmu_instance.reset()
            self._must_terminate = False
        self._prev_time = time

        if time - self._last_successful_time == 0:
            return

        if len(self._prev_results):
            result_idx = self.find_result_earlier_than(time, self._prev_results)
            if result_idx is not None:
                result = self._prev_results[result_idx]
                self._prev_results = self._prev_results[result_idx + 1:]
                return result

        if self._last_successful_time == 0:
            for variable in self._model_description.modelVariables:
                if (
                    variable.causality == "parameter"
                    and variable.variability == "fixed"
                    and variable.type == "String"
                    and variable.name in self._start_values
                ):
                    self._fmu_instance.setString([variable.valueReference], [self._start_values[variable.name]])
                if variable.causality == "structuralParameter" and variable.name in self._start_values:
                    self._structural_parameters[variable.name] = self._start_values[variable.name]
                    self._fmu_instance.fmi3EnterConfigurationMode(self._fmu_instance.component)
                    self._fmu_instance.setUInt32([variable.valueReference], [self._start_values[variable.name]])
                    self._fmu_instance.fmi3ExitConfigurationMode(self._fmu_instance.component)

            fmpy_resolve_model_description_shapes(self._model_description, self._fmu_instance)

            results = fmpy.simulate_fmu(
                filename=filename,
                fmi_type="CoSimulation",
                model_description=self._model_description,
                fmu_instance=self._fmu_instance,
                start_values=self._start_values,
                output=outputs,
                stop_time=time,
                set_stop_time=False,
                terminate=False,
                fmi_call_logger=lambda s: print("[FMI] " + s),
            )
        else:
            needs_configuration_mode = False
            for variable in self._model_description.modelVariables:
                if variable.causality == "structuralParameter" and variable.name in inputs:
                    if (variable.name not in self._structural_parameters
                            or self._structural_parameters[variable.name] != inputs[variable.name]):
                        self._structural_parameters[variable.name] = inputs[variable.name]
                        needs_configuration_mode = True

            if needs_configuration_mode:
                self._fmu_instance.fmi3EnterConfigurationMode(self._fmu_instance.component)

            try:
                for variable in self._model_description.modelVariables:
                    if not variable.name in inputs:
                        continue
                    if variable.variability == "fixed":
                        continue
                    if variable.causality == "output":
                        continue
                    if not needs_configuration_mode and variable.causality == "structuralParameter":
                        continue
                    self._set_variable_values(variable, inputs[variable.name])
            finally:
                if needs_configuration_mode:
                    self._fmu_instance.fmi3ExitConfigurationMode(self._fmu_instance.component)

            results = fmpy.simulate_fmu(
                filename=filename,
                fmi_type="CoSimulation",
                model_description=self._model_description,
                fmu_instance=self._fmu_instance,
                start_time=self._last_successful_time,
                stop_time=time,
                set_stop_time=False,
                initialize=False,
                terminate=False,
                output=outputs,
                fmi_call_logger=lambda s: print("[FMI] " + s),
            )

        self._must_terminate = True
        if len(results) > 0:
            self._last_successful_time = results[-1][0]
            result_idx = self.find_result_earlier_than(time, results)
            if result_idx is not None:
                result = results[result_idx]
                self._prev_results = results[result_idx + 1:]
                return result
        else:
            self._last_successful_time = time
        return None


class FmuRuntimeExtractedFMU:
    def __init__(self, fmu_location: str):
        self._unzip_fmu(fmu_location)
        self._runtime_fmus: list[FmuRuntimeInstance] = []

    def get_runtime_instances(self) -> list[FmuRuntimeInstance]:
        return self._runtime_fmus

    def get_unzip_dir(self) -> str:
        return self._unzip_dir

    def destroy(self):
        for instance in self._runtime_fmus:
            instance.destroy()
        self._runtime_fmus.clear()
        try:
            if self._should_delete_unzip_directory:
                shutil.rmtree(self._unzip_dir)
        finally:
            self._unzip_dir = None
            self._model_description = None

    def _unzip_fmu(self, fmu_location: str):
        self._fmu_location = fmu_location
        self._should_delete_unzip_directory = False
        if os.path.isdir(self._fmu_location):
            self._unzip_dir = self._fmu_location
        else:
            unzipdir = tempfile.mkdtemp()
            if sys.platform.startswith("win") and "~" in unzipdir:
                import ntpath
                unzipdir = ntpath.realpath(unzipdir)
            try:
                self._unzip_dir = unzipdir
                fmpy.extract(filename=self._fmu_location, unzipdir=unzipdir)
            except Exception as e:
                print(e)
                return
            self._should_delete_unzip_directory = True
        self._model_description = fmpy.read_model_description(filename=self._unzip_dir)
        self._add_bundled_windows_binary_if_needed()
        print(f"fmu guid={self._model_description.guid}")
        print(f"fmu model={self._model_description.coSimulation.modelIdentifier}")

    def _add_bundled_windows_binary_if_needed(self):
        if not sys.platform.startswith("win"):
            return
        if self._model_description.coSimulation is None:
            return

        model_identifier = self._model_description.coSimulation.modelIdentifier
        platform_dir = (
            "win64"
            if str(self._model_description.fmiVersion).startswith("2")
            else "x86_64-windows"
        )
        expected = (
            Path(self._unzip_dir)
            / "binaries"
            / platform_dir
            / f"{model_identifier}.dll"
        )
        if expected.exists():
            return

        staged = self._find_staged_windows_fmu_binary(model_identifier)
        if staged is None:
            return

        augmented_dir = Path(tempfile.mkdtemp(prefix="ov_fmi_"))
        shutil.copytree(self._unzip_dir, augmented_dir, dirs_exist_ok=True)
        target = (
            augmented_dir
            / "binaries"
            / platform_dir
            / f"{model_identifier}.dll"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged, target)

        if self._should_delete_unzip_directory:
            shutil.rmtree(self._unzip_dir)
        self._unzip_dir = str(augmented_dir)
        self._should_delete_unzip_directory = True
        print(f"Using staged Windows FMU binary: {staged}")

    def _find_staged_windows_fmu_binary(self, model_identifier: str) -> Path | None:
        repo_root = Path(__file__).resolve().parents[2]
        fmu_name = Path(self._fmu_location).name
        candidates = []
        for config in ("Release", "RelWithDebInfo", "Debug"):
            candidates.append(
                repo_root
                / "build"
                / config
                / "fmus"
                / fmu_name
                / "binaries"
                / "x86_64-windows"
                / f"{model_identifier}.dll"
            )
            candidates.append(
                repo_root
                / "build"
                / config
                / "fmus"
                / fmu_name
                / "binaries"
                / "win64"
                / f"{model_identifier}.dll"
            )
            candidates.append(repo_root / "build" / config / f"{model_identifier}.dll")

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def add_instance(self, parser_instance: FmuParserInstance):
        if not parser_instance.enabled:
            return
        fmu_instance = fmpy.instantiate_fmu(
            unzipdir=self._unzip_dir, model_description=self._model_description
        )
        runtime_instance = FmuRuntimeInstance(
            parser_instance=parser_instance,
            fmu_instance=fmu_instance,
            model_description=self._model_description,
        )
        self._runtime_fmus.append(runtime_instance)


class FMIRuntimeInputHead:
    """Abstract base: implement to feed USD/external values into FMU inputs."""
    def cache_connections(self, instance: FmuRuntimeInstance): pass
    def empty_cache_for(self, instance: FmuRuntimeInstance): pass
    def write_start_values(self, instance: FmuRuntimeInstance): pass
    def get_inputs_for(self, instance: FmuRuntimeInstance): pass


class FMIRuntimeOutputTail:
    """Abstract base: implement to consume FMU outputs and write to USD/renderer."""
    def cache_connections(self, instance: FmuRuntimeInstance): pass
    def empty_cache_for(self, instance: FmuRuntimeInstance): pass
    def write_outputs(self, instance: FmuRuntimeInstance, outputs, result): pass
    def get_outputs_for(self, instance: FmuRuntimeInstance): pass


class FMIRuntime:
    def __init__(self, runtime_input_head: FMIRuntimeInputHead, runtime_output_tail: FMIRuntimeOutputTail):
        self._runtime_fmus: dict[str, FmuRuntimeExtractedFMU] = {}
        self._runtime_input_head = runtime_input_head
        self._runtime_output_tail = runtime_output_tail

    def init(self, parser_instances: dict):
        for parser_instance in parser_instances.values():
            # Check if this is an SSP instance (tagged by _deserialise_instances)
            is_ssp = getattr(parser_instance, '_is_ssp', False)
            if is_ssp:
                from ssp_runtime import SspRuntimeExtracted  # noqa: PLC0415
                if self._runtime_fmus.get(parser_instance.fmu) is None:
                    self._runtime_fmus[parser_instance.fmu] = SspRuntimeExtracted(parser_instance.fmu)
                self._runtime_fmus[parser_instance.fmu].add_instance(parser_instance)
            else:
                if self._runtime_fmus.get(parser_instance.fmu) is None:
                    self._runtime_fmus[parser_instance.fmu] = FmuRuntimeExtractedFMU(parser_instance.fmu)
                self._runtime_fmus[parser_instance.fmu].add_instance(parser_instance)

    def resume(self):
        for runtime_fmu in self._runtime_fmus.values():
            for instance in runtime_fmu.get_runtime_instances():
                self._runtime_input_head.empty_cache_for(instance)
                self._runtime_input_head.cache_connections(instance)
                self._runtime_input_head.write_start_values(instance)
                self._runtime_output_tail.empty_cache_for(instance)
                self._runtime_output_tail.cache_connections(instance)
                instance.resume()

    def step(self, time: float):
        for runtime_fmu in self._runtime_fmus.values():
            for instance in runtime_fmu.get_runtime_instances():
                inputs = self._runtime_input_head.get_inputs_for(instance)
                outputs = self._runtime_output_tail.get_outputs_for(instance)
                result = instance.step(
                    filename=runtime_fmu.get_unzip_dir(),
                    inputs=inputs,
                    outputs=outputs,
                    time=time,
                )
                if result is not None:
                    self._runtime_output_tail.write_outputs(instance, outputs, result)

    def reset(self):
        for runtime_fmu in self._runtime_fmus.values():
            for instance in runtime_fmu.get_runtime_instances():
                self._runtime_input_head.empty_cache_for(instance)
                self._runtime_output_tail.empty_cache_for(instance)
                instance.reset()

    def destroy(self):
        for runtime_fmu in self._runtime_fmus.values():
            for instance in runtime_fmu.get_runtime_instances():
                instance.destroy()
