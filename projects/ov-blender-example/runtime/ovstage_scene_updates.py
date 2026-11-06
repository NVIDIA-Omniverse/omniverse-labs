# SPDX-License-Identifier: Apache-2.0
"""Replace accepted composed scene generations inside one attached OVStage.

The controller validates/accepts immutable scene files. This worker-side owner
only manages public USD population references; it never owns a renderer and
never creates a replacement stage. Direct property writes remain the fast path.
"""
from pathlib import Path
import json
import math


class SceneGenerationUpdateError(RuntimeError):
    def __init__(self, message, *, rollback_succeeded, ordinal):
        super().__init__(message)
        self.rollback_succeeded = rollback_succeeded
        self.ordinal = ordinal


class SceneGenerationPopulation:
    def __init__(self, stage, *, population=None):
        if population is None:
            from ovstage import population
        self.stage = stage
        self.population = population
        self.handle = None
        self.path = None
        self.base_path = None
        self.ordinal = 0
        self.failed = False
        self.overlays = []
        self.overlay_paths = set()
        self.visibility = None
        self.structure = None
        self.instance_graphs = None
        self.materials = None
        self.root_header = "#usda 1.0\n"

    def initialize(self, path, *, ordinal=1, domains=None, stage_metadata=None):
        if self.handle is not None or self.ordinal:
            raise RuntimeError("Scene population is already initialized")
        source = self._source(path)
        # No authored root opinions may mask the managed source layer.
        kwargs = {"ordinal": ordinal}
        if domains is not None:
            kwargs["domains"] = domains
        metadata = stage_metadata or {}
        allowed = {"upAxis", "metersPerUnit", "timeCodesPerSecond", "framesPerSecond",
                   "startTimeCode", "endTimeCode", "defaultPrim", "renderSettingsPrimPath"}
        if set(metadata) - allowed:
            raise ValueError("Unsupported stage metadata")
        lines = []
        for key, value in metadata.items():
            if isinstance(value, str):
                literal = json.dumps(value)
            elif type(value) in (int, float) and math.isfinite(value):
                literal = repr(value)
            else:
                raise ValueError(f"Invalid stage metadata {key}")
            lines.append(f"    {key} = {literal}")
        header = "#usda 1.0\n" + ("(\n" + "\n".join(lines) + "\n)\n" if lines else "")
        self.root_header = header
        self.population.open_usd_from_string(self.stage, header, **kwargs)
        self.ordinal = ordinal
        return self.replace(source, ordinal=ordinal + 1)

    @staticmethod
    def _source(path):
        source = Path(path).resolve(strict=True)
        if not source.is_file():
            raise ValueError("Accepted scene generation must be a file")
        return source

    def replace(self, path, *, ordinal):
        if self.failed:
            raise RuntimeError("Failed scene population requires explicit recovery")
        if type(ordinal) is not int or ordinal <= self.ordinal:
            raise ValueError("Scene generation ordinal must increase")
        source = self._source(path)
        if source == self.path or source == self.base_path:
            # Native USD may intern equal file references: removing the old
            # reference can remove the just-added identical candidate as well.
            # Rebuild the owned composition to reset values and inline opinions
            # without relying on duplicate-reference identity semantics.
            old = self.handle
            self.restore(source, ordinal=ordinal)
            return {"ordinal": self.ordinal, "scene_path": str(source),
                    "source_handle": self.handle, "replaced_source_handle": old}
        # Parsing the candidate first leaves the active reference intact on
        # enqueue/parse failure. Both source edits are then projected together.
        candidate = self.population.add_usd_reference(self.stage, str(source), "/")
        old = self.handle
        try:
            if old is not None:
                self.population.remove_usd(self.stage, old)
            for overlay in self.overlays:
                self.population.remove_usd(self.stage, overlay)
            if self.visibility is not None:
                self.visibility.clear()
            if self.structure is not None:
                self.structure.clear()
            if self.instance_graphs is not None:
                self.instance_graphs.clear()
            if self.materials is not None:
                self.materials.clear()
            self.population.apply_usd_changes(self.stage, ordinal=ordinal)
        except Exception as error:
            # Source edits are not SDK transactions. Restore the previously
            # accepted immutable source before the controller sees a failure;
            # its ensure(old-spec) path may correctly choose to reuse this stage.
            rollback_ordinal = ordinal + 1
            try:
                self.population.reset_usd(self.stage)
                self._forget_inline_owners()
                self.population.open_usd_from_string(self.stage, self.root_header, ordinal=rollback_ordinal)
                rollback_ordinal += 1
                restored = (self.population.add_usd_reference(self.stage, str(self.path), "/")
                            if self.path is not None else None)
                self.population.apply_usd_changes(self.stage, ordinal=rollback_ordinal)
                self.stage.advance_write_floor(rollback_ordinal).wait()
                self.handle, self.ordinal = restored, rollback_ordinal
                self.overlays = []
                self.base_path = self.path
                self.overlay_paths.clear()
            except Exception as rollback_error:
                self.failed = True
                raise SceneGenerationUpdateError(
                    f"Scene update failed ({error}); predecessor restoration failed ({rollback_error})",
                    rollback_succeeded=False, ordinal=rollback_ordinal) from error
            raise SceneGenerationUpdateError(
                f"Scene update failed; predecessor restored: {error}",
                rollback_succeeded=True, ordinal=rollback_ordinal) from error
        self.handle, self.path, self.ordinal = candidate, source, ordinal
        self.base_path = source
        self.overlays = []
        self.overlay_paths.clear()
        return {"ordinal": ordinal, "scene_path": str(source), "source_handle": candidate,
                "replaced_source_handle": old}

    def apply_delta(self, path, *, predecessor_path, desired_path, ordinal, expected_predecessor_path=None):
        """Project a validated effective overlay while retaining accepted assets.

        The client generated this file from immutable accepted/candidate USD.
        A failed write restores the complete predecessor before returning.
        """
        if self.failed:
            raise RuntimeError("Failed scene population requires explicit recovery")
        if type(ordinal) is not int or ordinal <= self.ordinal:
            raise ValueError("Scene generation ordinal must increase")
        expected = self.path if expected_predecessor_path is None else self._source(expected_predecessor_path)
        if str(expected) != str(self._source(predecessor_path)):
            raise ValueError("Population delta predecessor is stale")
        patch, desired = self._source(path), self._source(desired_path)
        if patch in self.overlay_paths:
            # Public USD population interns an existing file reference. Adding
            # it again does not move it above later opinions (undo/redo A/B/A).
            # Rebuild the complete desired population instead of acknowledging
            # an overlay whose earlier, weaker reference cannot take effect.
            result = self.replace(desired, ordinal=ordinal)
            return {**result, 'population_update': 'replacement', 'overlay_count': 0,
                    'population_fallback_reason': 'reused_overlay_source'}
        old_path = self.path
        candidate = self.population.add_usd_reference(self.stage, str(patch), "/")
        try:
            if self.visibility is not None:
                self.visibility.clear()
            if self.structure is not None:
                self.structure.clear()
            if self.instance_graphs is not None:
                self.instance_graphs.clear()
            if self.materials is not None:
                self.materials.clear()
            self.population.apply_usd_changes(self.stage, ordinal=ordinal)
        except Exception as error:
            rollback_ordinal = ordinal+1
            try:
                self.population.reset_usd(self.stage)
                self._forget_inline_owners()
                self.population.open_usd_from_string(self.stage,self.root_header,ordinal=rollback_ordinal)
                rollback_ordinal += 1
                restored = self.population.add_usd_reference(self.stage,str(old_path),"/")
                self.population.apply_usd_changes(self.stage,ordinal=rollback_ordinal)
                self.stage.advance_write_floor(rollback_ordinal).wait()
                self.handle,self.ordinal,self.overlays = restored,rollback_ordinal,[]
                self.base_path = old_path
                self.overlay_paths.clear()
            except Exception as rollback_error:
                self.failed = True
                raise SceneGenerationUpdateError(
                    f"Delta failed ({error}); predecessor restoration failed ({rollback_error})",
                    rollback_succeeded=False,ordinal=rollback_ordinal) from error
            raise SceneGenerationUpdateError(f"Delta rejected; predecessor restored: {error}",
                rollback_succeeded=True,ordinal=rollback_ordinal) from error
        self.overlays.append(candidate)
        self.overlay_paths.add(patch)
        self.path,self.ordinal = desired,ordinal
        return {"ordinal":ordinal,"scene_path":str(desired),"source_handle":candidate,
                "population_update":"scoped_overlay","overlay_count":len(self.overlays)}

    def _forget_inline_owners(self):
        # reset_usd invalidates every source handle, including a helper whose
        # clear() raised before clearing its cache. Never remove/reuse those
        # handles during predecessor replay or the next generation change.
        self.visibility = self.structure = self.instance_graphs = self.materials = None

    def snapshot_inline_sources(self):
        """Copy only current acknowledged opinions before destructive source edits."""
        from copy import deepcopy
        return deepcopy({
            'structure': None if self.structure is None else (self.structure.text, self.structure.roots),
            'instances': [] if self.instance_graphs is None else [dict(owner_id=identity,
                projection_path=entry.projection_path, hidden_source_path=entry.hidden_source_path,
                layer_body=entry.text) for identity, entry in self.instance_graphs.entries.items()],
            'materials': {} if self.materials is None else self.materials.graphs,
            'visibility': None if self.visibility is None else (self.visibility.values, self.visibility.text),
        })

    def restore_inline_sources(self, snapshot):
        """Recreate native source handles; callers synchronize before replaying values."""
        try:
            from .inline_structure_population import InlineStructurePopulation
            from .instance_graph_updates import InstanceGraphPopulation
            from .inline_material_population import InlineMaterialPopulation
            from .inline_visibility_population import InlineVisibilityPopulation, VisibilityOpinion
        except ImportError:
            from inline_structure_population import InlineStructurePopulation
            from instance_graph_updates import InstanceGraphPopulation
            from inline_material_population import InlineMaterialPopulation
            from inline_visibility_population import InlineVisibilityPopulation, VisibilityOpinion
        structure = snapshot['structure']
        if structure and structure[1]:
            self.structure = InlineStructurePopulation(self)
            self.structure.replace(*structure, ordinal=self.ordinal + 1)
        if snapshot['instances']:
            self.instance_graphs = InstanceGraphPopulation(self)
            self.instance_graphs.apply(self.instance_graphs.prepare(snapshot['instances']), ordinal=self.ordinal + 1)
        if snapshot['materials']:
            self.materials = InlineMaterialPopulation(self)
            opinion = self.materials.prepare([dict(material_path=root, layer_body=text)
                for root, text in snapshot['materials'].items()])
            self.materials.apply(opinion, ordinal=self.ordinal + 1)
        visibility = snapshot['visibility']
        if visibility and visibility[1]:
            self.visibility = InlineVisibilityPopulation(self)
            self.visibility.apply(VisibilityOpinion(*visibility), ordinal=self.ordinal + 1)

    def restore(self, path, *, ordinal):
        """Rebuild predecessor ownership after an uncertain transaction boundary.

        Replacing known references alone cannot remove an unacknowledged source
        created by a failed native operation. Reset the entire owned source
        composition while retaining the same OVStage identity.
        """
        if self.failed:
            raise RuntimeError('Failed scene population requires explicit recovery')
        if type(ordinal) is not int or ordinal <= self.ordinal:
            raise ValueError('Scene generation ordinal must increase')
        source = self._source(path)
        try:
            self.population.reset_usd(self.stage)
            self._forget_inline_owners()
            self.population.open_usd_from_string(self.stage, self.root_header, ordinal=ordinal)
            restored = self.population.add_usd_reference(self.stage, str(source), '/')
            ordinal += 1
            self.population.apply_usd_changes(self.stage, ordinal=ordinal)
            self.stage.advance_write_floor(ordinal).wait()
        except Exception:
            self.failed = True
            raise
        self.handle, self.path, self.ordinal, self.overlays = restored, source, ordinal, []
        self.base_path = source
        self.overlay_paths.clear()
