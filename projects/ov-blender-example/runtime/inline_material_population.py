# SPDX-License-Identifier: Apache-2.0
"""Current material graphs compiled by native inline USD population.

No replacement stage, host scene export, temporary files, or edit history.
Caller drops this owner before replacing accepted base scene population.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class MaterialOpinion:
    graphs: dict
    changed_roots: tuple
    text: str


def _under(path, root):
    return path == root or path.startswith(root + '/')


class InlineMaterialPopulation:
    def __init__(self, owner):
        self.owner = owner
        self.handles = {}
        self.graphs = {}
        self.text = ''

    def prepare(self, updates):
        if not isinstance(updates, (list, tuple)) or not updates or len(updates) > 256:
            raise ValueError('Material update needs a bounded nonempty update list')
        graphs = dict(self.graphs); changed = []
        for update in updates:
            if set(update) != {'material_path', 'layer_body'}:
                raise ValueError('Material update requires material_path and layer_body')
            root, text = update['material_path'], update['layer_body']
            if (not isinstance(root, str) or not root.startswith('/') or root == '/'
                    or any(p in {'', '.', '..'} for p in root[1:].split('/'))):
                raise ValueError('Material update requires an absolute material prim path')
            if not isinstance(text, str) or not text.startswith('#usda 1.0\n') or len(text) > 8_000_000:
                raise ValueError('Material update requires bounded USDA text')
            if root in changed or any(_under(root, other) or _under(other, root) for other in graphs if other != root):
                raise ValueError('Overlapping material graph roots')
            graphs[root] = text
            changed.append(root)
        # Each graph is prevalidated by the authoring planner. Native parser
        # validates the cumulative candidate before removing its predecessor.
        combined = '#usda 1.0\n' + '\n'.join(graphs[root].split('\n', 1)[1] for root in sorted(graphs))
        return MaterialOpinion(graphs, tuple(changed), combined)

    def apply(self, opinion, *, ordinal):
        if self.owner.failed:
            raise RuntimeError('Failed graph population requires scene reload')
        if opinion.text == self.text:
            return False
        api, stage = self.owner.population, self.owner.stage
        candidates = {}
        try:
            # Separate per-material references avoid duplicate ancestor specs
            # and prevent unchanged material subtrees from being resampled.
            for root in opinion.changed_roots:
                candidates[root] = api.add_usd_reference_from_string(stage, opinion.graphs[root], '/')
        except Exception:
            for handle in candidates.values():
                api.remove_usd(stage, handle)
            raise
        try:
            for root in opinion.changed_roots:
                if root in self.handles:
                    api.remove_usd(stage, self.handles[root])
            api.apply_usd_changes(stage, ordinal=ordinal)
        except Exception:
            self.owner.failed = True
            raise
        self.handles.update(candidates)
        self.graphs, self.text = opinion.graphs, opinion.text
        self.owner.ordinal = max(self.owner.ordinal, ordinal)
        return True

    def clear(self):
        for handle in self.handles.values():
            self.owner.population.remove_usd(self.owner.stage, handle)
        self.handles, self.graphs, self.text = {}, {}, ''


def apply(backend, params):
    """Root-owned backend dispatch entrypoint; same native stage and renderer."""
    import time
    started = time.perf_counter()
    if set(params) != {'updates'}:
        raise ValueError('Material graph packet requires updates only')
    backend._require_idle_native_render()
    if backend.population_owner is None:
        raise RuntimeError('Material graphs require populated native scene')
    owner = getattr(backend.population_owner, 'materials', None)
    if owner is None:
        backend.population_owner.materials = owner = InlineMaterialPopulation(backend.population_owner)
    opinion = owner.prepare(params['updates'])
    # Confirm existing material roots before enqueueing any new source opinion.
    try:
        from .direct_native_structure import inspect
    except ImportError:
        from direct_native_structure import inspect
    columns = inspect(backend, {'prim_paths': list(opinion.changed_roots), 'attributes': ['usd-prim-type']})['columns']
    if any(columns.get(root, {}).get('usd-prim-type', {}).get('resolved') != ['Material'] for root in opinion.changed_roots):
        raise ValueError('Material graph target must already be a native Material')
    retained = {key: row for key, row in backend.desired_values.items()
                if not any(_under(key[0], root) for root in opinion.changed_roots)}
    backend.ordinal += 1
    try:
        owner.apply(opinion, ordinal=backend.ordinal)
        backend.stage.advance_write_floor(backend.ordinal, backend.ovstage.Scope.ALL).wait()
        backend.renderer.update_from_stage(backend.ordinal)
        backend.renderer.reset(time=backend.simulation_time_ns / 1e9)
        backend.sample_count = 0
        backend.desired_values = retained
        # Only changed material references are replaced. Unrelated material,
        # mesh/UV, camera and light opinions remain untouched; no blanket replay.
        replay = []

    except Exception:
        backend.experimental_structure_uncertain = True
        raise
    return dict(material_graph_updated=True, native_stage_id=backend.native_stage_id,
                ordinal=backend.ordinal, material_count=len(opinion.changed_roots),
                replayed_values=len(replay), total_ms=(time.perf_counter()-started)*1000,
                source_projection_stale=True)
