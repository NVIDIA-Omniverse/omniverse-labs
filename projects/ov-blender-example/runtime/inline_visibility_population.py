# SPDX-License-Identifier: Apache-2.0
"""One current inline visibility opinion; no files, edit log or renderer owner."""
from dataclasses import dataclass
import json

@dataclass(frozen=True)
class VisibilityOpinion:
    values: dict
    text: str

class InlineVisibilityPopulation:
    def __init__(self, owner):
        self.owner = owner
        self.handle = None
        self.values = {}
        self.text = ''

    def prepare(self, prepared):
        values = dict(self.values)
        for path, name, base, array, role, raw, width in prepared:
            if name != 'visibility':
                continue
            if base != 'token' or array or len(raw) != 1 or raw[0] not in {'inherited', 'invisible'}:
                raise ValueError('visibility requires scalar Token inherited or invisible')
            parts = path.split('/')[1:]
            if not parts or any(not part or part in {'.', '..'} or '.' in part for part in parts):
                raise ValueError('visibility requires an absolute prim path')
            values[path] = raw[0]
        tree = {}
        for path, token in sorted(values.items()):
            node = tree
            for part in path.split('/')[1:]:
                node = node.setdefault(part, {})
            node[None] = token
        lines = ['#usda 1.0']
        def emit(node, depth):
            for name, child in node.items():
                if name is None:
                    lines.append(' ' * depth + 'token visibility = ' + json.dumps(child))
                else:
                    lines.append(' ' * depth + 'over ' + json.dumps(name, ensure_ascii=False) + ' {')
                    emit(child, depth + 1)
                    lines.append(' ' * depth + '}')
        emit(tree, 0)
        return VisibilityOpinion(values, '\n'.join(lines) + '\n')

    def apply(self, opinion, *, ordinal):
        if self.owner.failed:
            raise RuntimeError('Failed visibility population requires scene reload')
        if opinion.text == self.text:
            return False
        api, stage = self.owner.population, self.owner.stage
        # Parse first. Native source changes are not transactional: a failure
        # after removing the old handle explicitly invalidates this projection.
        candidate = api.add_usd_reference_from_string(stage, opinion.text, '/')
        try:
            if self.handle is not None:
                api.remove_usd(stage, self.handle)
            api.apply_usd_changes(stage, ordinal=ordinal)
        except Exception:
            self.owner.failed = True
            raise
        self.handle, self.values, self.text = candidate, opinion.values, opinion.text
        self.owner.ordinal = max(self.owner.ordinal, ordinal)
        return True

    def clear(self):
        """Remove the source opinion before replacing/patching its generation."""
        if self.handle is not None:
            self.owner.population.remove_usd(self.owner.stage, self.handle)
        self.handle, self.values, self.text = None, {}, ''
