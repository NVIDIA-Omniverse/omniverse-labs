# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable, lazy OVUI item models for the native PVD ownership hierarchy."""
from dataclasses import replace
import re

import omni.ui as ui

from .array_settings import ArrayDisplaySettings, VISIBLE_ARRAY_WINDOW
from .hierarchy import HierarchyNode
from .row_actions import RowActionScope, RowIdentity


class ObjectItem(ui.AbstractItem):
    def __init__(self, key, name, type_name, instance_id=None):
        super().__init__()
        self.key = key
        self.instance_id = instance_id
        self.children = []
        self.parent = None
        self.class_name = type_name
        self.kind = "object" if instance_id is not None else "group"
        self.is_alias = False
        self.target_key = None
        self.renderable = False
        self.source_instance_id = None
        self.property_id = None
        self.element_index = None
        self.element_size = 0
        self.element_count = 0
        self.array_error = ''
        self.query_id = None
        self.values = [ui.SimpleStringModel(name), ui.SimpleStringModel(type_name),
                       ui.SimpleStringModel(""), ui.SimpleStringModel("")]


class _ItemCache(dict):
    """Preserve old cache lookups while materializing only requested UI rows."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get(self, key, default=None):
        if key not in self.model._specs:
            return default
        return self.model._item(key)

    def __missing__(self, key):
        if key in self.model._specs:
            return self.model._item(key)
        raise KeyError(key)


class ObjectTreeModel(ui.AbstractItemModel):
    def __init__(self, grouping="scene", page_size=10):
        super().__init__()
        self.grouping = grouping
        self.array_settings = ArrayDisplaySettings(group_size=page_size)
        self.page_size = self.array_settings.group_size
        self._page_generation = 0
        self._specs = {}
        self._action_specs = {}
        self.hidden_keys = set()
        self._root_keys = []
        self._ordered_keys = []
        self.cache = _ItemCache(self)
        self.roots = []
        self._filter = ""
        self._filter_options = (False, False, ("name", "type", "id"))
        self._visible_keys = set()
        self._revealed_keys = set()
        self.expanded_keys = set()
        self._adapter = None
        self._arrays = {}
        self._session_scope = None
        self._segment_epoch = None
        self._source_projection = None
        self.revision = 0
        self.object_count = 0

    @property
    def filter(self):
        return self._filter

    @property
    def capture_key(self):
        return self._session_scope

    @property
    def segment_epoch(self):
        return self._segment_epoch

    def clear(self):
        """Drop a capture's projection and every derived row/search identity."""
        self.cache.clear()
        self.roots=[]
        self._specs={}
        self._action_specs={}
        self.hidden_keys.clear()
        self._root_keys=[]
        self._ordered_keys=[]
        self._visible_keys.clear()
        self._revealed_keys.clear()
        self.expanded_keys.clear()
        self._adapter=None
        self._arrays={}
        self._session_scope=None
        self._segment_epoch=None
        self._source_projection=None
        self.object_count=0
        self.revision+=1
        self._item_changed(None)

    @filter.setter
    def filter(self, text):
        self.set_filter(text)

    def set_filter(self, text, *, match_case=False, whole_word=False,
                   fields=("name", "type", "id")):
        self._filter = str(text).strip()
        self._filter_options = (bool(match_case), bool(whole_word), tuple(fields))
        self._revealed_keys.clear()
        self._update_filter()
        self._item_changed(None)

    def set_grouping(self, grouping):
        if grouping not in ("scene", "class"):
            raise ValueError("Tree grouping must be scene or class")
        if grouping != self.grouping:
            self.grouping = grouping
            if self._adapter:
                self.populate(self._adapter)

    def ensure_native_visible(self, identities):
        if not any(f'object:{identity}' in self.hidden_keys for identity in identities):
            return False
        self.set_grouping('class')
        return True

    def set_array_settings(self, settings, *, selected_rows=()):
        """Re-page explicitly after a committed display-policy change.

        Native/group/alias/record identities survive; page identities belong to
        the old partition and are rejected, even after returning to its size.
        Only explicitly restored rows acquire UI items. Expansion limits never
        remove native references or query records from the hierarchy.
        """
        if not isinstance(settings, ArrayDisplaySettings):
            raise TypeError("Expected ArrayDisplaySettings")
        changed = settings.group_size != self.page_size
        self.array_settings = settings
        if changed:
            self.page_size = settings.group_size
            self._page_generation += 1
            obsolete = {key for key, node in self._specs.items()
                        if node.kind in ("page", "array_page")}
            for key in obsolete:
                self.cache.pop(key, None)
            self.expanded_keys.difference_update(obsolete)
            self._revealed_keys.difference_update(obsolete)
            self._source_projection = None
            if self._adapter is not None:
                self.populate(self._adapter)
        return self.restore_rows(selected_rows)

    def populate(self, adapter):
        self._adapter = adapter
        # Creation IDs survive a live segment transition. A fresh native
        # Session owns a new ID space; only that boundary discards UI identity.
        scope = getattr(adapter.session, "capture_key", id(adapter.session))
        self._segment_epoch = getattr(adapter.session, "segment_epoch", getattr(adapter, "_schema_epoch", 0))
        if scope != self._session_scope:
            self.cache.clear()
            self.expanded_keys.clear()
            self._revealed_keys.clear()
            self._session_scope = scope
            self._source_projection = None
        source, roots = adapter.hierarchy(self.grouping)
        patch = getattr(adapter, "_hierarchy_patch", None)
        if patch is not None and self.hidden_keys:
            # Native query patches may touch a compacted shape's geometry.
            self._source_projection = None
        if patch is not None and source is self._source_projection and source is patch[0]:
            _, deleted, updates, renamed = patch
            # A parent whose UI children were paged needs the general rebuild.
            if all(key not in self._specs or self._specs[key].kind != "group" or
                   not any(self._specs[child].kind == "page" for child in self._specs[key].children)
                   for key in updates):
                for key in deleted:
                    self._specs.pop(key, None)
                self._specs.update(updates)
                self._ordered_keys = [renamed.get(key, key) for key in self._ordered_keys]
                for key in updates:
                    if dict.get(self.cache, key) is not None:
                        self._item(key)
                if self._filter:
                    self._update_filter()
                else:
                    self._visible_keys.difference_update(deleted)
                    self._visible_keys.update(updates)
                self.revision += 1
                self._item_changed(None)
                return
            self._source_projection = None
        if source is self._source_projection:
            return
        self._source_projection = source
        previous = self._specs
        self._action_specs = source
        if self.grouping == 'scene':
            from .compact_hierarchy import compact_scene_hierarchy
            self._specs, self.hidden_keys = compact_scene_hierarchy(source)
        else:
            self._specs, self.hidden_keys = dict(source), set()
        self.object_count = sum(node.kind == "object" for node in source.values())
        self._arrays = {(node.source_instance_id, node.property_id): node.key
                        for node in source.values() if node.kind == "array"}
        self._root_keys = roots
        for node in tuple(self._specs.values()):
            if node.key in self.hidden_keys:
                continue
            if len(node.children) > min(self.page_size, VISIBLE_ARRAY_WINDOW):
                self._paginate_children(self._specs[node.key])
        self._ordered_keys = []
        stack = list(reversed(roots))
        while stack:
            key = stack.pop()
            self._ordered_keys.append(key)
            stack.extend(reversed(self._specs[key].children))
        self.roots = [self._item(key) for key in roots]
        changed = previous != self._specs
        # Synchronize only rows already materialized. Unopened reference lists
        # and typed arrays allocate no OVUI models or delegates at all.
        for key in tuple(dict.keys(self.cache)):
            if key in self._specs:
                self._item(key)
        self._update_filter()
        self.revision += 1
        if changed:
            self._item_changed(None)

    def _paginate_children(self, node):
        children = node.children
        fanout = min(self.page_size, VISIBLE_ARRAY_WINDOW)
        page_span = self.page_size if len(children) > self.page_size else fanout
        while (len(children) + page_span - 1) // page_span > fanout:
            page_span *= fanout
        pages = []
        for first in range(0, len(children), page_span):
            last = min(first + page_span, len(children))
            key = f"page:{node.key}:{first}:{page_span}:p{self._page_generation}"
            page = HierarchyNode(key, f"{node.name} [{first}–{last - 1}]",
                                 kind="page", parent_key=node.key, children=children[first:last],
                                 source_instance_id=node.source_instance_id,
                                 property_id=node.property_id, element_index=first,
                                 element_count=last-first, segment_scoped=True)
            self._specs[key] = page
            pages.append(key)
            for child in page.children:
                self._specs[child] = replace(self._specs[child], parent_key=key)
            if len(page.children) > min(self.page_size, VISIBLE_ARRAY_WINDOW):
                self._paginate_children(page)
        self._specs[node.key] = replace(node, children=pages)

    def _item(self, key):
        node = self._specs[key]
        item = dict.get(self.cache, key)
        if item is None:
            item = ObjectItem(key, node.name, node.class_name, node.instance_id)
            self.cache[key] = item
        item.instance_id, item.class_name = node.instance_id, node.class_name
        item.kind, item.is_alias = node.kind, node.kind == "reference"
        item.target_key, item.renderable = node.target_key, node.renderable
        item.source_instance_id, item.property_id = node.source_instance_id, node.property_id
        item.element_index, item.element_size = node.element_index, node.element_size
        item.element_count = node.element_count
        item.array_error = node.array_error
        item.query_id = node.query_id
        if item.values[0].as_string != node.name:
            item.values[0].set_value(node.name)
        if item.values[1].as_string != node.class_name:
            item.values[1].set_value(node.class_name)
        # Iteratively materialize ancestors, avoiding recursion limits in
        # large articulation chains. Children are resolved only on expansion.
        chain = []
        cursor = node.parent_key
        while cursor and dict.get(self.cache, cursor) is None:
            chain.append(cursor)
            cursor = self._specs[cursor].parent_key
        parent = dict.get(self.cache, cursor) if cursor else None
        for parent_key in reversed(chain):
            spec = self._specs[parent_key]
            ancestor = ObjectItem(parent_key, spec.name, spec.class_name, spec.instance_id)
            ancestor.kind, ancestor.renderable = spec.kind, spec.renderable
            ancestor.parent = parent
            self.cache[parent_key] = ancestor
            parent = ancestor
        item.parent = dict.get(self.cache, node.parent_key) if node.parent_key else None
        return item

    def _array_children(self, node):
        if node.children or not node.element_count:
            return
        first = node.element_index or 0
        count = node.element_count
        last = first + count
        child_keys = []
        fanout = min(self.page_size, VISIBLE_ARRAY_WINDOW)
        if count > fanout:
            span = self.page_size if count > self.page_size else fanout
            while (count + span - 1) // span > fanout:
                span *= fanout
            ranges = [(index, min(span, last - index)) for index in range(first, last, span)]
            for start, length in ranges:
                key = f"array-page:{node.source_instance_id}:{node.property_id}:{start}:{span}:p{self._page_generation}"
                self._specs[key] = HierarchyNode(key, f"[{start}–{start + length - 1}]",
                    node.class_name, kind="array_page", parent_key=node.key,
                    source_instance_id=node.source_instance_id, property_id=node.property_id,
                    element_index=start, element_size=node.element_size, element_count=length)
                child_keys.append(key)
        else:
            for index in range(first, last):
                key = f"array:{node.source_instance_id}:{node.property_id}:{index}"
                query_id = (self._adapter.query_identity(node.source_instance_id, node.property_id, index)
                            if hasattr(self._adapter, "query_identity") else None)
                label = f"{node.class_name}[{index}]" + (f" [id:{query_id}]" if query_id is not None else "")
                self._specs[key] = HierarchyNode(key, label,
                    node.class_name, kind="array_element", parent_key=node.key,
                    source_instance_id=node.source_instance_id, property_id=node.property_id,
                    element_index=index, element_size=node.element_size, query_id=query_id)
                child_keys.append(key)
        # Adapter specifications remain reusable by another tree projection.
        self._specs[node.key] = replace(node, children=child_keys)
        # Generated rows participate in later filtering and expansion replay;
        # they are absent from the initial, deliberately unexpanded projection.
        position = self._ordered_keys.index(node.key) + 1
        self._ordered_keys[position:position] = child_keys
        if node.key in self._visible_keys:
            self._visible_keys.update(child_keys)

    def row_identity(self, key):
        """Describe one existing row without constructing a UI item or reading native data."""
        if isinstance(key, (ObjectItem, RowIdentity)):
            key = key.key
        node = self._specs.get(key)
        if node is None:
            return None
        # UI wrappers can be removed, but row actions retain native ownership.
        semantic_node = self._action_specs.get(key, node)
        owner = None
        source, property_id = node.source_instance_id, node.property_id
        cursor = semantic_node.parent_key
        seen = set()
        while cursor in self._specs and cursor not in seen:
            seen.add(cursor)
            parent = self._action_specs.get(cursor, self._specs[cursor])
            if parent.kind == "object":
                owner = parent.instance_id
                break
            if node.kind != "object":
                if source is None:
                    source = parent.source_instance_id
                if property_id is None:
                    property_id = parent.property_id
            cursor = parent.parent_key
        transient = node.segment_scoped or node.kind in ("page", "array", "array_page", "array_element")
        return RowIdentity(self.capture_key, node.key, node.kind, node.class_name,
            node.instance_id, owner, source, property_id,
            node.query_id, node.element_index, node.element_count,
            self.segment_epoch if transient else None)

    def row_identities(self, keys):
        """Retain only the explicitly supplied rows; selection never enumerates a subtree."""
        rows = {}
        for key in keys:
            row = self.row_identity(key)
            if row is not None:
                if isinstance(key, RowIdentity) and row.token != key.token:
                    continue
                rows.setdefault(row.token, row)
        return tuple(rows.values())

    def restore_rows(self, identities):
        """Restore available explicit rows, retaining caller-owned intent for absent rows.

        Only the requested rows and their ancestors acquire UI items. Query
        records can regenerate their bounded path after a projection rebuild;
        stale capture/segment identities never address a newly reused row.
        """
        restored = []
        seen = set()
        for identity in identities:
            if (identity.capture_key != self.capture_key or
                identity.segment_epoch is not None and identity.segment_epoch != self.segment_epoch):
                continue
            if identity.kind in ("page", "array_page") and not identity.key.endswith(f":p{self._page_generation}"):
                continue
            if identity.key not in self._specs and identity.kind in ("array_page", "array_element"):
                key = self._arrays.get((identity.source_instance_id, identity.property_id))
                while key and identity.key not in self._specs:
                    node = self._specs[key]
                    self._array_children(node)
                    key = next((child for child in self._specs[key].children
                        if self._specs[child].element_index <= identity.element_index <
                        self._specs[child].element_index + max(1, self._specs[child].element_count)), None)
            row = self.row_identity(identity.key)
            if row is not None and row.token == identity.token and row.token not in seen:
                seen.add(row.token)
                restored.append(self._item(row.key))
        return tuple(restored)

    def action_scope(self, keys, *, action="visibility"):
        """Resolve explicit rows against owned specifications, without expanding UI.

        Canonical positive roots use InteractionState's existing native graph.
        Aliases and query records have local flags only. Visibility stops at a
        PxScene; transparency continues. This deliberately retains the tested
        ownership cascade rather than reproducing Qt's expansion-dependent
        shape/joint walk and lowercase-only scene.actors exception.
        """
        if action not in ("visibility", "transparency"):
            raise ValueError("Unknown row action")
        selected = self.row_identities(keys)
        native_roots, local_rows = {}, {}
        visited = set()
        pending = [(row.key, False) for row in reversed(selected)]
        while pending:
            key, has_native_root = pending.pop()
            if key in visited:
                continue
            visited.add(key)
            row = self.row_identity(key)
            if row is None:
                continue
            node = self._action_specs.get(key, self._specs[key])
            if row.native:
                if not has_native_root:
                    native_roots.setdefault(row.instance_id, None)
                has_native_root = True
                if action == "visibility" and row.class_name == "PxScene":
                    continue
            else:
                local_rows.setdefault(row.token, row)
            if row.kind == "reference":
                continue
            # Canonical edges preserve native ownership; existing presentation
            # edges also carry lazy page/record flags. Never expand new rows.
            children = dict.fromkeys([*node.children, *self._specs[key].children])
            pending.extend((child, has_native_root) for child in reversed(children))
        return RowActionScope(self.capture_key, self.segment_epoch, action,
                              selected, tuple(native_roots), tuple(local_rows.values()))

    def _matches(self, node, query):
        match_case, whole_word, fields = self._filter_options
        parts = []
        if "name" in fields:
            parts.append(node.name)
        if "type" in fields:
            parts.append(node.class_name)
        if "id" in fields and node.instance_id is not None:
            parts.append(f"{node.instance_id} id:{node.instance_id}")
        if "id" in fields and node.query_id is not None:
            parts.append(f"{node.query_id} id:{node.query_id}")
        haystack = " ".join(parts)
        if not match_case:
            haystack, query = haystack.casefold(), query.casefold()
        words = query.split()
        if whole_word:
            return all(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", haystack) for word in words)
        return all(word in haystack for word in words)

    def _update_filter(self):
        if not self._filter:
            self._visible_keys = set(self._specs)
            return
        visible = set(self._revealed_keys)
        inherited = set()
        for key in self._ordered_keys:
            node = self._specs[key]
            if node.parent_key in inherited or self._matches(node, self._filter):
                visible.add(key)
                inherited.add(key)
        for key in reversed(self._ordered_keys):
            if key in visible and self._specs[key].parent_key:
                visible.add(self._specs[key].parent_key)
        self._visible_keys = visible

    def canonical_item(self, instance_id):
        if instance_id < -1 and hasattr(self._adapter, "query_record"):
            record = self._adapter.query_record(instance_id)
            if record is None:
                return None
            key = self._arrays.get((record.scene_instance_id, record.property_id))
            while key:
                node = self._specs[key]
                self._array_children(node)
                children = self._specs[key].children
                key = next((child for child in children
                            if self._specs[child].element_index <= record.element_index <
                            self._specs[child].element_index + max(1, self._specs[child].element_count)), None)
                if key and self._specs[key].kind == "array_element":
                    return self._item(key)
            return None
        return self.cache.get(f"object:{instance_id}")

    def _key(self, item_or_id):
        if isinstance(item_or_id, ObjectItem):
            return item_or_id.key
        if isinstance(item_or_id, str) and item_or_id in self._specs:
            return item_or_id
        if isinstance(item_or_id, int) and item_or_id < -1:
            item = self.canonical_item(item_or_id)
            if item:
                return item.key
        return f"object:{item_or_id}"

    def ancestors(self, item_or_id):
        key = self._key(item_or_id)
        result = []
        while key in self._specs and self._specs[key].parent_key:
            key = self._specs[key].parent_key
            result.append(self._item(key))
        return list(reversed(result))

    def reveal_ancestors(self, item_or_id, expand_callback=None):
        key = self._key(item_or_id)
        if key not in self._specs:
            return []
        ancestors = self.ancestors(key)
        self._revealed_keys.update([key, *(item.key for item in ancestors)])
        self._update_filter()
        self.expanded_keys.update(item.key for item in ancestors)
        if expand_callback:
            for item in ancestors:
                expand_callback(item)
        if self._filter:
            self._item_changed(None)
        return ancestors

    def remember_expanded(self, item, expanded=True):
        if expanded:
            self.expanded_keys.add(item.key)
        else:
            self.expanded_keys.discard(item.key)

    def expanded_items(self):
        return [self._item(key) for key in self._ordered_keys if key in self.expanded_keys]

    def find_matches(self, query=None):
        query = self._filter if query is None else str(query).strip()
        if not query:
            return []
        # A picked or explicitly searched synthetic ID can reveal a record in
        # an unopened page without enumerating every record widget first.
        synthetic = re.fullmatch(r"(?:id:)?(-\d+)", query, re.IGNORECASE)
        if synthetic and "id" in self._filter_options[2]:
            self.canonical_item(int(synthetic.group(1)))
        # Compaction hides wrappers, not searchable native identities. The
        # browser reveals a selected hidden shape through its class view.
        keys = [*self._ordered_keys, *sorted(key for key in self.hidden_keys
                if self._specs[key].kind == 'object')]
        return [self._item(key) for key in keys
                if self._specs[key].kind in ("object", "array", "array_element") and self._matches(self._specs[key], query)]

    def next_match(self, current=None, *, backwards=False):
        matches = self.find_matches()
        if not matches:
            return None
        keys = [item.key for item in matches]
        key = self._key(current) if current is not None else None
        if key not in keys:
            return matches[-1 if backwards else 0]
        return matches[(keys.index(key) + (-1 if backwards else 1)) % len(matches)]

    def previous_match(self, current=None):
        return self.next_match(current, backwards=True)

    def get_item_children(self, item=None):
        if item is None:
            return [self._item(key) for key in self._root_keys if key in self._visible_keys]
        node = self._specs.get(item.key)
        if node is None:
            return []
        if node.kind in ("array", "array_page"):
            self._array_children(node)
            node = self._specs[item.key]
        item.children = [self._item(key) for key in node.children
                         if not self._filter or key in self._visible_keys]
        return item.children

    def can_item_have_children(self, item):
        if item is None:
            return bool(self._root_keys)
        node = self._specs.get(item.key)
        return bool(node and (node.children or node.element_count))

    def get_item_value_model_count(self, item=None):
        return 4

    def get_item_value_model(self, item, column_id=0):
        return item.values[column_id] if item else None
