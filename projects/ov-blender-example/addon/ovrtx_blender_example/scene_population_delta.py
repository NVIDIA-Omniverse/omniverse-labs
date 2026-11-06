# SPDX-License-Identifier: Apache-2.0
"""Conservative effective-USD deltas for the existing population owner.

Blender's stock exporter remains the geometry authority. This compiler compares
immutable accepted and candidate compositions; it never reads or mutates RNA.
Absent properties, schema changes, and metadata removals fall back to replacement.
"""
from dataclasses import dataclass, field
from pathlib import Path
import time
import hashlib
import tempfile
import shutil


@dataclass(frozen=True)
class SourcePopulationSnapshot:
    source_path: str
    fields: dict = field(repr=False)
    assets: dict = field(repr=False)
    sources: dict = field(default_factory=dict,repr=False)
    layer_signatures: dict = field(default_factory=dict,repr=False)
    changed_field_keys: frozenset | None = field(default=None,repr=False,compare=False)
    asset_signatures: dict = field(default_factory=dict,repr=False)
    source_order: tuple = field(default_factory=tuple,repr=False)


@dataclass(frozen=True)
class PopulationDelta:
    path: str | None
    predecessor_path: str
    changed_prims: int = 0
    changed_properties: int = 0
    created_prims: int = 0
    deleted_prims: int = 0
    preparation_ms: float = 0.0
    fallback_reason: str = ''
    candidate_snapshot: SourcePopulationSnapshot | None = field(default=None,repr=False,compare=False)
    phase_timings: dict = field(default_factory=dict,compare=False)
    # Effective patch roots, excluding implicit ancestor overs and layer-history
    # churn. None denotes an older/unavailable proof and keeps conservative
    # raw-source invalidation; an empty tuple certifies an effective no-op.
    changed_prim_paths: tuple[str, ...] | None = field(default=None,compare=False)


def _info(spec):
    return {key: spec.GetInfo(key) for key in spec.ListInfoKeys()}


def _file_signature(path):
    """Include bytes: same-size edits may share a filesystem timestamp tick."""
    path = Path(path)
    stat = path.stat()
    with path.open('rb') as source:
        content = hashlib.file_digest(source, 'sha256').hexdigest()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, content)


def _property_contributors(snapshot, property_path, sdf, source_specs):
    """Direct lookup in immutable source specs, with accepted-field fallback.

    Retired layers absent from the candidate are not covered by the compiler's
    shared-layer guard. Check each source once per request before reading it;
    if unavailable/changed, preserve ownership from its accepted field snapshot.
    """
    property_path_text = str(property_path)
    contributors = set()
    for identifier, (fields, _) in snapshot.sources.items():
        if identifier not in source_specs:
            layer = sdf.Layer.FindOrOpen(identifier)
            try:
                signature = _file_signature(layer.realPath or layer.identifier) if layer is not None else None
            except OSError:
                signature = None
            source_specs[identifier] = (layer if layer is not None and not layer.dirty
                and signature is not None and signature == snapshot.layer_signatures.get(identifier) else None)
        layer = source_specs[identifier]
        if layer is None:
            if any(key[1] == property_path_text for key in fields):
                contributors.add(identifier)
        else:
            spec = layer.GetObjectAtPath(property_path)
            if spec is not None and spec.ListInfoKeys():
                contributors.add(identifier)
    return contributors


def _source_layers(path, sdf):
    result=[]
    visiting=set()
    def visit(path):
        layer=sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise ValueError('source_layer_unavailable')
        if layer.identifier in visiting:
            raise ValueError('cyclic_source_layers')
        visiting.add(layer.identifier);result.append(layer)
        for child,offset in zip(layer.subLayerPaths,layer.subLayerOffsets):
            if offset.offset != 0 or offset.scale != 1:
                raise ValueError('source_layer_time_offset')
            visit(sdf.ComputeAssetPathRelativeToLayer(layer,child))
        visiting.remove(layer.identifier)
    visit(path)
    return result


def capture_population_snapshot(path, *, unchanged_source_snapshot=None):
    """Compare authored field contributors; let USD resolve only candidates.

    Contributor changes may over-admit a field masked by stronger opinions,
    but cannot suppress it. No custom USD composition implementation is used.
    """
    from pxr import Sdf as sdf, Usd
    assets={}
    active_dependencies=set()
    layer_signatures={}
    asset_signatures={}
    def asset_digest(path):
        if path not in assets:
            file=Path(path)
            try:
                signature=_file_signature(file)
            except FileNotFoundError:
                signature=None
            asset_signatures[path]=signature
            if (unchanged_source_snapshot is not None and path in unchanged_source_snapshot.asset_signatures
                and unchanged_source_snapshot.asset_signatures[path]==signature):
                assets[path]=unchanged_source_snapshot.assets[path]
            elif file.is_file():
                assets[path]=signature[-1]
            else:
                assets[path]=None
        return assets[path]
    images={'.png','.jpg','.jpeg','.tif','.tiff','.exr','.hdr','.bmp','.webp','.dds'}
    def normalized(value,layer):
        if isinstance(value,sdf.AssetPath):
            if not value.path:
                return ('asset','')
            resolved=sdf.ComputeAssetPathRelativeToLayer(layer,value.path)
            active_dependencies.add(resolved)
            digest=asset_digest(resolved)
            if digest is not None and Path(resolved).suffix.lower() in images:
                return ('standalone_image',Path(resolved).suffix.lower(),digest)
            return ('asset',resolved,digest)
        if isinstance(value,dict):
            return {k:normalized(v,layer) for k,v in value.items()}
        if isinstance(value,(sdf.Reference,sdf.Payload)):
            asset=sdf.ComputeAssetPathRelativeToLayer(layer,value.assetPath) if value.assetPath else ''
            return (type(value).__name__,asset,str(value.primPath),repr(value.layerOffset))
        if hasattr(value,'GetAppliedItems'):
            return (type(value).__name__,value.isExplicit,
                tuple((name,tuple(normalized(v,layer) for v in getattr(value,name))) for name in
                      ('explicitItems','addedItems','prependedItems','appendedItems','deletedItems','orderedItems')))
        if type(value).__name__=='AssetPathArray':
            return tuple(normalized(v,layer) for v in value)
        return value
    def fields(path):
        result={}
        sources={}
        layers=_source_layers(path,sdf)
        order=tuple(layer.identifier for layer in layers)
        for layer in layers:
            file=Path(layer.realPath or layer.identifier)
            signature=_file_signature(file)
            layer_signatures[layer.identifier]=signature
            cached = None if unchanged_source_snapshot is None else unchanged_source_snapshot.sources.get(layer.identifier)
            if cached is not None:
                previous_signature=unchanged_source_snapshot.layer_signatures.get(layer.identifier)
                if not layer.dirty and previous_signature != signature:
                    # USD may keep an open layer cached despite a subsecond
                    # on-disk edit (notably on Windows). Never fingerprint new
                    # file bytes while retaining the old parsed opinions.
                    if not layer.Reload(force=True):
                        raise ValueError('source_layer_reload_failed')
                if layer.dirty or previous_signature != signature or any(asset_digest(p) != digest for p,digest in cached[1].items()):
                    cached=None
            if cached is not None:
                local, dependencies = cached
                sources[layer.identifier]=cached
                continue
            local={}
            active_dependencies.clear()
            def visit(spec_path):
                if spec_path.ContainsPrimVariantSelection():
                    raise ValueError('source_usd_variants')
                if not (spec_path.IsPrimPath() or spec_path.IsPropertyPath()):
                    return
                spec=layer.GetObjectAtPath(spec_path)
                if spec is None:
                    return
                prim=str(spec_path.GetPrimPath())
                for key in spec.ListInfoKeys():
                    value=normalized(spec.GetInfo(key),layer)
                    identity=(prim,str(spec_path),key)
                    local.setdefault(identity,[]).append(value)
            layer.Traverse(sdf.Path.absoluteRootPath,visit)
            sources[layer.identifier]=(local,{p:assets[p] for p in active_dependencies})
        previous={} if unchanged_source_snapshot is None else unchanged_source_snapshot.sources
        # Only the order of shared contributors matters for reuse. New/removed
        # layers nominate their own keys; reordered shared layers nominate all
        # keys. Let USD continue to resolve effective fields downstream.
        common=set(previous)&set(sources)
        previous_order=() if unchanged_source_snapshot is None else unchanged_source_snapshot.source_order
        same_order=bool(previous_order) and ([p for p in previous_order if p in common]==[p for p in order if p in common])
        changed_keys=None
        if unchanged_source_snapshot is not None and same_order:
            changed_keys=set()
            for identifier in set(previous)|set(sources):
                before,after=previous.get(identifier),sources.get(identifier)
                if before is after:
                    continue
                if before is not None: changed_keys.update(before[0])
                if after is not None: changed_keys.update(after[0])
            result=unchanged_source_snapshot.fields.copy()
            for key in changed_keys:
                values=[]
                for identifier in order:
                    local,dependencies=sources[identifier]
                    for value in local.get(key,()):
                        if not values or values[-1] != value:
                            values.append(value)
                if values: result[key]=values
                else: result.pop(key,None)
        else:
            for identifier in order:
                local,dependencies=sources[identifier]
                for key,values in local.items():
                    bucket=result.setdefault(key,[])
                    for value in values:
                        if not bucket or bucket[-1] != value:
                            bucket.append(value)
        return result,sources,None if changed_keys is None else frozenset(changed_keys),order
    values,sources,changed_keys,order=fields(path)
    return SourcePopulationSnapshot(str(Path(path).resolve()),values,assets,sources,layer_signatures,changed_keys,asset_signatures,order)

def compile_population_delta(predecessor_path, candidate_path, *, accepted_snapshot=None,
                             retired_camera_projection=None):
    """Return a bounded sparse overlay, or an explicit conservative fallback."""
    started = time.perf_counter()
    old_path = str(Path(predecessor_path).resolve())
    candidate_snapshot = None
    phase_timings={}
    def fallback(reason):
        return PopulationDelta(None, old_path, preparation_ms=(time.perf_counter()-started)*1000,
                               fallback_reason=reason,candidate_snapshot=candidate_snapshot,phase_timings=phase_timings)
    try:
        from pxr import Sdf, Usd
    except ImportError:
        return fallback('usd_bindings_unavailable')
    try:
        if accepted_snapshot is not None and accepted_snapshot.source_path != old_path:
            return fallback('source_snapshot_predecessor_mismatch')
        phase=time.perf_counter()
        previous_snapshot = accepted_snapshot or capture_population_snapshot(old_path)
        candidate_snapshot = capture_population_snapshot(str(candidate_path),unchanged_source_snapshot=previous_snapshot)
        phase_timings['source_snapshot_ms']=(time.perf_counter()-phase)*1000
        # Effective predecessor composition is only recoverable while shared
        # source layers stay immutable. A mutated common layer requires native
        # replacement, just like a changed texture at the same filename.
        if any(previous_snapshot.layer_signatures[p] != candidate_snapshot.layer_signatures[p]
               or Sdf.Layer.FindOrOpen(p).dirty
               for p in set(previous_snapshot.layer_signatures)&set(candidate_snapshot.layer_signatures)):
            return fallback('mutable_source_layer_changed')
        phase=time.perf_counter()
        if accepted_snapshot is None and previous_snapshot.assets:
            return fallback('accepted_asset_snapshot_unavailable')
        if any(previous_snapshot.assets[path] != candidate_snapshot.assets[path]
               for path in set(previous_snapshot.assets)&set(candidate_snapshot.assets)):
            return fallback('mutable_asset_dependency_changed')
        before,after = previous_snapshot.fields,candidate_snapshot.fields
        keys=candidate_snapshot.changed_field_keys
        if keys is None: keys=set(before)|set(after)
        changed_keys = {key for key in keys if before.get(key) != after.get(key)}
        affected = {key[0] for key in changed_keys}
        # Layer-history churn can change an ancestor's authored over/def
        # contributor chain while leaving its effective USD specifier intact.
        # Admitting that ancestor to the flatten mask would include every mesh
        # below it. Ask USD about specifier-only nominations first; descendants
        # with actual field changes remain independently nominated.
        specifier_only = {path for path in affected
                          if {key[2] for key in changed_keys if key[0] == path} == {'specifier'}}
        if specifier_only:
            nomination_mask = Usd.StagePopulationMask()
            for path in sorted(specifier_only): nomination_mask.Add(path)
            old_nomination = Usd.Stage.OpenMasked(old_path, nomination_mask, load=Usd.Stage.LoadNone)
            new_nomination = Usd.Stage.OpenMasked(str(candidate_path), nomination_mask, load=Usd.Stage.LoadNone)
            if old_nomination is not None and new_nomination is not None:
                for path in specifier_only:
                    a, b = old_nomination.GetPrimAtPath(path), new_nomination.GetPrimAtPath(path)
                    if (a and b and a.GetSpecifier() == b.GetSpecifier()
                            and a.GetTypeName() == b.GetTypeName() and a.IsActive() == b.IsActive()):
                        affected.discard(path)
        phase_timings['source_difference_ms']=(time.perf_counter()-phase)*1000
        phase=time.perf_counter()
        mask = Usd.StagePopulationMask()
        for path in sorted(affected):
            mask.Add(path)
        old_stage = Usd.Stage.OpenMasked(old_path,mask)
        new_stage = Usd.Stage.OpenMasked(str(candidate_path),mask)
        if old_stage is None or new_stage is None:
            return fallback('composition_unavailable')
        old, new = old_stage.Flatten(False), new_stage.Flatten(False)
        if _info(old.pseudoRoot) != _info(new.pseudoRoot):
            return fallback('stage_metadata_changed')
        def prims(layer):
            result = {}
            def visit(path):
                if path.IsPrimPath():
                    result[str(path)] = layer.GetPrimAtPath(path)
            layer.Traverse(Sdf.Path.absoluteRootPath, visit)
            return result
        previous, desired = prims(old), prims(new)
        phase_timings['masked_composition_ms']=(time.perf_counter()-phase)*1000
        phase=time.perf_counter()
        patch = Sdf.Layer.CreateAnonymous('accepted-scene-delta.usda')
        changed = properties = created = deleted = 0
        added_roots, removed_roots = [], []
        changed_paths = set()
        retired_source_specs = {}  # Request-lifetime immutable spec lookup only.
        for path in sorted(set(previous) | set(desired), key=lambda p: (p.count('/'), p)):
            if any(path.startswith(root+'/') for root in added_roots+removed_roots):
                continue
            before, after = previous.get(path), desired.get(path)
            if before is None:
                Sdf.CreatePrimInLayer(patch, path)
                if not Sdf.CopySpec(new, path, patch, path):
                    return fallback('new_prim_copy_failed')
                patch.GetPrimAtPath(path).active = after.active
                added_roots.append(path); created += 1; changed += 1
                changed_paths.add(path)
                continue
            if after is None:
                Sdf.CreatePrimInLayer(patch, path).active = False
                removed_roots.append(path); deleted += 1; changed += 1
                changed_paths.add(path)
                continue
            if before.typeName != after.typeName or before.specifier != after.specifier:
                return fallback('existing_prim_schema_changed')
            old_info, new_info = _info(before), _info(after)
            for arc in ('references','payload','inheritPaths','specializes','variantSelection','variantSetNames'):
                if old_info.get(arc) != new_info.get(arc):
                    return fallback('existing_composition_arc_changed')
            removed_metadata = set(old_info)-set(new_info)
            if removed_metadata-{'active','instanceable'}:
                return fallback('prim_metadata_removed')
            metadata = {k:v for k,v in new_info.items() if k not in old_info or old_info[k] != v}
            metadata.update({k: (k == 'active') for k in removed_metadata})
            old_props = {p.name:p for p in before.properties}
            new_props = {p.name:p for p in after.properties}
            removed_props = set(old_props)-set(new_props)
            resets = {}
            if removed_props:
                # A selected-camera presentation layer is transient. Retiring
                # its settings is not a geometry deletion. Only reset the
                # explicitly owned old-camera fields, never arbitrary source
                # attributes, relationships, animation or shader connections.
                retirement = retired_camera_projection or {}
                allowed = {'fStop', 'horizontalApertureOffset', 'verticalApertureOffset',
                           'omni:rtx:autoExposure:enabled',
                           'omni:rtx:autoExposure:whitePointScale'}
                owned_layer = retirement.get('layer_path')
                # USD identifiers use canonical separators on Windows too.
                # Compare identities from the same resolver, not host strings.
                if owned_layer:
                    owned_source = Sdf.Layer.FindOrOpen(str(owned_layer))
                    owned_layer = owned_source.identifier if owned_source is not None else None
                if (path != retirement.get('camera_path') or after.typeName != 'Camera'
                        or not owned_layer or not removed_props <= allowed):
                    return fallback('existing_property_removed')
                for name in removed_props:
                    prop = old_props[name]
                    if (not isinstance(prop, Sdf.AttributeSpec)
                            or prop.HasInfo('timeSamples') or prop.HasInfo('connectionPaths')):
                        return fallback('existing_property_removed')
                    contributors = _property_contributors(previous_snapshot, prop.path, Sdf, retired_source_specs)
                    if contributors != {owned_layer}:
                        return fallback('existing_property_removed')
                    value = new_stage.GetPrimAtPath(path).GetAttribute(name).Get()
                    if value is None:
                        # Population does not reliably delete existing runtime
                        # columns for USD value blocks. The retired camera is
                        # inactive; use the adapter's explicit exposure defaults
                        # until a later camera retarget authors its saved values.
                        from .native_exposure import NativeExposure, ENABLED_ATTRIBUTE, TARGET_ATTRIBUTE
                        defaults = NativeExposure()
                        if name == ENABLED_ATTRIBUTE:
                            value = defaults.enabled
                        elif name == TARGET_ATTRIBUTE:
                            value = defaults.white_point_scale
                        else:
                            return fallback('existing_property_removed')
                    resets[name] = (prop, value)
            changed_props = [p for name,p in new_props.items()
                             if name not in old_props or _info(old_props[name]) != _info(p)]
            if any(p.name == 'xformOpOrder' or p.name.startswith('xformOp:')
                   for p in changed_props):
                # The locked native tuple acknowledges a scoped USD xform
                # overlay without reliably refreshing an existing rendered
                # hierarchy (real parented animation probe). Rebuild the
                # complete population until that projection is qualified.
                # Interactive world-pose writes retain their typed fast lane.
                return fallback('existing_transform_requires_replacement')
            if not metadata and not changed_props and not resets:
                continue
            target = Sdf.CreatePrimInLayer(patch, path)
            for key,value in metadata.items():
                target.SetInfo(key,value)
            for prop in changed_props:
                if not Sdf.CopySpec(new, prop.path, patch, prop.path):
                    return fallback('property_copy_failed')
            for name, (old_prop, value) in resets.items():
                restored = Sdf.AttributeSpec(target, name, old_prop.typeName,
                                            old_prop.variability, old_prop.custom)
                restored.default = value
            changed += 1; properties += len(changed_props) + len(resets)
            changed_paths.add(path)
        # Assets in a flattened composition are resolved by USD against their
        # source layer. Export beside the candidate, never edit accepted files.
        phase_timings['spec_comparison_ms']=(time.perf_counter()-phase)*1000
        phase=time.perf_counter()
        # Binary USD preserves exact typed values without formatting gigabytes
        # of numeric arrays as Python strings. Stream hashing/copying to keep
        # peak memory bounded even when a legitimately large subtree changes.
        with tempfile.TemporaryDirectory(prefix='population-delta-',dir=Path(candidate_path).parent) as temporary:
            staged=Path(temporary)/'delta.usdc'
            if not patch.Export(str(staged)):
                return fallback('delta_export_failed')
            digest=hashlib.sha256()
            with staged.open('rb') as source:
                for chunk in iter(lambda: source.read(1024*1024),b''):
                    digest.update(chunk)
            destination=Path(candidate_path).parent/('population-delta-'+digest.hexdigest()[:24]+'.usdc')
            try:
                with destination.open('xb') as output, staged.open('rb') as source:
                    shutil.copyfileobj(source,output)
            except FileExistsError:
                existing=hashlib.sha256()
                with destination.open('rb') as source:
                    for chunk in iter(lambda: source.read(1024*1024),b''):
                        existing.update(chunk)
                if existing.digest() != digest.digest():
                    return fallback('delta_immutable_file_conflict')
        phase_timings['delta_serialization_ms']=(time.perf_counter()-phase)*1000
        return PopulationDelta(str(destination),old_path,changed,properties,created,deleted,
                               (time.perf_counter()-started)*1000,candidate_snapshot=candidate_snapshot,
                               phase_timings=phase_timings,
                               changed_prim_paths=tuple(sorted(changed_paths)))
    except ValueError as error:
        return fallback(str(error))
    except Exception as error:
        return fallback('delta_compilation_failed:'+type(error).__name__)
