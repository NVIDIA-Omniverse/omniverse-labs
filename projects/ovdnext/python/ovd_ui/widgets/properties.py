# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable, bounded native property inspector and array windows."""
from collections import Counter
from copy import copy
from dataclasses import dataclass
import struct

import omni.ui as ui
from _pvd_native import SnapshotDataUnavailable

from ..analysis import can_plot_property
from ..array_settings import ArrayDisplaySettings, VISIBLE_ARRAY_WINDOW
from ..array_layout import array_layout
from ..compact_hierarchy import geometry_shape_ids
from ..formatting import ValueFormat,element_size,format_value
from ..frame_summary import native_object_counts,sorted_counts
from ..style import layout as fl

_COMPOUNDS={"PxTransform":(("Translation",16,28,"PxVec3"),("Rotation",0,16,"PxQuat")),
            "PxBounds3":(("min",0,12,"PxVec3"),("max",12,24,"PxVec3"))}
_SUMMARY_COUNT_WIDTH=96
_ARRAY_PLAN_BYTES=8*1024*1024
_NATIVE_COPY_BYTES=128*1024*1024
_COPY_RESERVE_BYTES=8*1024*1024
_ARRAY_UNAVAILABLE="Array window unavailable at this frame. It will retry when a complete frame restores it."


@dataclass(frozen=True)
class DisplayedValue:
    """Immutable text already displayed by a property row and its children."""
    text: str = ""
    children: tuple["DisplayedValue", ...] = ()

    @property
    def copy_text(self):
        # Qt copies immediate child Value cells, including the trailing separator.
        return "".join(child.text+"; " for child in self.children) if self.children else self.text


@dataclass
class _CopyEntry:
    value: DisplayedValue | None = None
    identity: int | None = None
    prop: object = None
    capture_key: object = None


def _display_value(type_name,raw,options,*,total_bytes=None):
    if type_name in _COMPOUNDS and len(raw)==element_size(type_name):
        return DisplayedValue(children=tuple(DisplayedValue(format_value(part_type,raw[start:stop],options))
            for _label,start,stop,part_type in _COMPOUNDS[type_name]))
    return DisplayedValue(format_value(type_name,raw,options,total_bytes=total_bytes))


@dataclass
class _ValueWidgets:
    """Retain labels and copy payloads, without retaining a native reader."""
    entry: _CopyEntry
    values: list

    def update(self,value):
        self.entry.value=value
        for (label,entry),part in zip(self.values,value.children or (value,)):
            label.text=part.text
            label.tooltip=part.text
            entry.value=part


def _copy_text(text):
    from ..clipboard import copy_text
    copy_text(text)


class PropertyInspector:
    def __init__(self,on_reference,on_plot,preferences,on_request=None,*,on_copy_text=None,on_error=None,on_summary=None):
        self.on_reference,self.on_plot,self.preferences=on_reference,on_plot,preferences
        self.on_request=on_request
        try:self.array_settings=ArrayDisplaySettings.from_preferences(preferences)
        except ValueError:self.array_settings=ArrayDisplaySettings()
        self._array_generation=0
        self._array_segment=None
        self._array_denied={}
        self._array_unavailable={}
        self._array_requested={}
        self._array_frame=None
        self.array_positions={}
        self.array_expanded=set()
        self.capture_key=None
        self.on_copy_text=on_copy_text or _copy_text
        self.on_error=on_error
        self.on_summary=on_summary
        self.summary_available=False
        self.summary_sort_column="type"
        self.summary_sort_descending=False
        self.copy_error=""
        self.menu=None
        self._destroyed=False
        self._menu_generation=0
        self._rows_key=None
        self._rows={}
        self._value_rows={}

    def destroy(self):
        self._destroyed=True
        self._rows_key=None
        self._rows.clear()
        self._value_rows.clear()
        self._array_generation+=1
        self._menu_generation+=1
        if self.menu is not None:
            self.menu.destroy()
            self.menu=None

    def _context(self,entry):
        return {"explicit_hover":True,"mouse_pressed_fn":lambda x,y,button,modifiers:
                self.show_context_menu(entry,x,y) if button==1 else None}

    def _report_copy_error(self,message):
        self.copy_error=message
        if self.on_error:self.on_error(message)

    def _copy_action(self,text,generation):
        if self._destroyed or generation!=self._menu_generation:return False
        try:self.on_copy_text(text)
        except (RuntimeError,ValueError,TypeError,OSError) as error:
            self._report_copy_error(f"Unable to copy value: {error}")
            return False
        self.copy_error=""
        return True

    def _plot_action(self,identity,prop,capture_key,generation):
        if self._destroyed or generation!=self._menu_generation:return False
        if capture_key!=self.capture_key:
            self._report_copy_error("The capture changed. Open the current property's menu to plot it.")
            return False
        self.on_plot(identity,prop)
        return True

    def show_context_menu(self,entry,x,y):
        if self._destroyed:return
        if entry is None and not self.summary_available:return
        # Capture strings/metadata now: subsequent frame/array rebuilds must not
        # change an open menu's payload or read from a newer native view.
        value,identity,prop,capture_key=(entry.value,entry.identity,entry.prop,entry.capture_key) if entry else (None,None,None,self.capture_key)
        summary_capture_key=self.capture_key
        if self.menu is not None:self.menu.destroy()
        self._menu_generation+=1
        generation=self._menu_generation
        self.menu=ui.Menu("Property value",tearable=False)
        with self.menu:
            text=value.copy_text if value is not None else ""
            if entry is not None:
                ui.MenuItem("Copy value",enabled=value is not None,
                    triggered_fn=lambda:self._copy_action(text,generation))
            if identity is not None and prop is not None and can_plot_property(prop):
                ui.MenuItem("Plot value",triggered_fn=lambda:
                    self._plot_action(identity,prop,capture_key,generation))
            if self.summary_available:
                if entry is not None:ui.Separator()
                ui.MenuItem("Frame summary",triggered_fn=lambda:
                    self._summary_action(summary_capture_key,generation))
        self.menu.show_at(x,y)

    def background_pressed(self,x,y,button,modifiers):
        if button==1:self.show_context_menu(None,x,y)

    def _summary_action(self,capture_key,generation):
        if self._destroyed or generation!=self._menu_generation:return False
        if capture_key!=self.capture_key or self.on_summary is None:return False
        return self.on_summary(capture_key)

    def _sort_summary(self,column):
        self.summary_sort_descending=(not self.summary_sort_descending
            if column==self.summary_sort_column else False)
        self.summary_sort_column=column
        self.summary_table.rebuild()

    def _build_summary(self,adapter):
        # Retain only immutable scalar rows for delayed table rebuilds.
        rows=native_object_counts(adapter)
        frame=adapter.session.frame
        ui.Label("Frame summary",name="title",height=26)
        ui.Label(f"Frame {frame} · {sum(row.count for row in rows):,} native objects",
                 name="muted",height=26,elided_text=True)
        ui.Label("Current-frame native objects only",name="muted",height=26,
                 explicit_hover=True, tooltip="Alive native objects counted once across this publication. Synthetic query/contact records, aliases and tree groups are excluded. Visibility and filtering do not change these counts.")
        def table():
            with ui.VStack(height=0,spacing=fl.pvd_spacing):
                with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                    for column,label,width in (("type","Type",ui.Fraction(1)),("count","Count",_SUMMARY_COUNT_WIDTH)):
                        suffix=(" (desc)" if self.summary_sort_descending else " (asc)") if column==self.summary_sort_column else ""
                        ui.Button(label+suffix,name="quiet",width=width,
                            clicked_fn=lambda key=column:self._sort_summary(key))
                for row in sorted_counts(rows,self.summary_sort_column,self.summary_sort_descending):
                    with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                        ui.Label(row.type_name,elided_text=True,explicit_hover=True, tooltip=row.type_name)
                        ui.Label(f"{row.count:,}",name="mono",width=_SUMMARY_COUNT_WIDTH,alignment=ui.Alignment.RIGHT_CENTER)
                if not rows:ui.Label("No alive native objects in this frame",name="muted",height=fl.pvd_control)
        self.summary_table=ui.Frame(build_fn=table)

    def set_array_settings(self,settings):
        """Adopt an already persisted policy; caller rebuilds/publishes as needed."""
        if not isinstance(settings,ArrayDisplaySettings):raise TypeError("Expected ArrayDisplaySettings")
        if settings==self.array_settings:return False
        self.array_settings=settings
        self._array_generation+=1
        self.array_positions.clear()
        self.array_expanded.clear()
        self._array_denied.clear()
        self._array_unavailable.clear()
        self._array_requested.clear()
        return True

    def reject_read_requests(self,requests,message):
        """Freeze optional-window rejection until an explicit user retry.

        The application retries a native copy-budget failure once without these
        optional reads. Capture/segment/policy changes clear this local state;
        navigating or reopening one array clears only that property's requests.
        """
        for request in requests:
            request=tuple(request)
            self._array_denied[request]=str(message)
            self._array_unavailable.pop(request,None)  # Budget rejection is explicit-retry only.

    def _retry_array(self,key):
        if len(key)!=2:return
        self._array_denied={request:reason for request,reason in self._array_denied.items() if request[:2]!=key}
        self._array_unavailable={request:stamp for request,stamp in self._array_unavailable.items() if request[:2]!=key}
        self._array_requested={request:stamp for request,stamp in self._array_requested.items() if request[:2]!=key}

    @staticmethod
    def _array_stamp(session,request,byte_count=None):
        # Generation changes on same-frame republishing; it is not evidence of
        # restored data. Only complete owned frame/size observations may retry.
        if not getattr(session,"ready",False) or not getattr(session,"request_applied",False):return None
        if byte_count is None:
            try:byte_count=session.property_size(*request[:2])
            except (IndexError,SnapshotDataUnavailable):pass
        return (session.scene_id,session.frame,byte_count)

    def _mark_array_unavailable(self,session,request):
        stamp=self._array_stamp(session,request)
        if stamp is None:return
        if request in self._array_denied:return
        self._array_unavailable[request]=stamp

    def _array_window_denial(self,session,request,byte_count=None):
        stamp=self._array_stamp(session,request,byte_count)
        if request in getattr(session,"unavailable_reads",()):
            self._mark_array_unavailable(session,request)
        previous=self._array_unavailable.get(request)
        if (stamp is not None and previous is not None and stamp!=previous and
                stamp[2] is not None and request[2]+request[3]<=stamp[2]):
            self._array_unavailable.pop(request)
            self._array_requested.pop(request,None)
        return self._array_denied.get(request) or (_ARRAY_UNAVAILABLE if request in self._array_unavailable else None)

    def _observe_array_frame(self,session):
        if not getattr(session,"ready",False) or not getattr(session,"request_applied",False):return
        frame=(session.scene_id,session.frame)
        if frame!=self._array_frame:
            self._array_frame=frame
            self._array_requested.clear()
        # Record death/shrink even when no array widget is drawn at this frame.
        for request in getattr(session,"unavailable_reads",()):
            self._mark_array_unavailable(session,tuple(request))

    def _request_array_window(self,request,stamp):
        if stamp is None or request in self._array_denied or request in self._array_unavailable or self._array_requested.get(request)==stamp:return
        if self.on_request:
            self._array_requested[request]=stamp
            self.on_request()

    @staticmethod
    def _array_stride(adapter,prop):
        stride=element_size(prop.type)
        if stride:return stride
        # Native class size is authoritative; never infer a record layout.
        sizes={getattr(cls,"byte_size",0) for cls in adapter.classes.values() if cls.name==prop.type}
        return sizes.pop() if len(sizes)==1 else 0

    def read_requests(self,adapter,identities):
        if adapter is None or len(identities)!=1:return []
        identity=identities[0]
        if identity not in adapter.instances or not adapter.instances[identity].alive:return []
        scope=(getattr(adapter.session,"capture_key",id(adapter.session)),
               getattr(adapter.session,"segment_epoch",None))
        if self.capture_key is not None and scope!=(self.capture_key,self._array_segment):return []
        requests=[]
        # Current copied_bytes includes previews, reference/name metadata and
        # prior explicit windows. Counting planned bytes again is conservative.
        remaining=min(_ARRAY_PLAN_BYTES,max(0,_NATIVE_COPY_BYTES-
            int(getattr(adapter.session,"copied_bytes",_NATIVE_COPY_BYTES))-_COPY_RESERVE_BYTES))
        for prop in adapter.properties(identity):
            key=(identity,prop.id)
            # Converted references already belong to the hierarchy metadata.
            if prop.type=="ObjectRef" or key not in self.array_expanded or key not in self.array_positions:continue
            value_type,stride=array_layout(adapter,prop)
            if not element_size(value_type):continue
            if not stride:continue
            byte_count=adapter.session.property_size(identity,prop.id)
            if byte_count%stride:continue
            count=byte_count//stride
            if not self.array_settings.allows_expansion(count):continue
            start,size=self.array_settings.window(count,self.array_positions[key],element_bytes=stride)
            if not size:continue
            request=(identity,prop.id,start*stride,size*stride)
            if self._array_window_denial(adapter.session,request,byte_count):continue
            if len(requests)>=1024 or request[3]>remaining:
                self._array_denied[request]="Array window paused at the owned-data budget. Close another array, then reopen this one to retry."
                continue
            requests.append(request)
            remaining-=request[3]
        return requests

    @property
    def options(self):
        return ValueFormat(self.preferences.get("float_precision",6),self.preferences.get("float_format","auto"),
                           self.preferences.get("show_hex",False))

    def _layout_signature(self,adapter,identities,record,summary):
        if adapter is None or record is not None or summary or len(identities)!=1:return None
        identity=identities[0]
        instance=adapter.instances.get(identity)
        if instance is None or not instance.alive:return None
        session=adapter.session
        schema=tuple((p.id,p.name,p.type,p.native_type,p.semantic,p.array,p.field_count)
                     for p in adapter.properties(identity))
        return (getattr(session,"capture_key",id(session)),getattr(session,"segment_epoch",None),
                getattr(session,"scene_id",None),identity,instance.class_id,
                adapter.name(identity),adapter.class_name(identity),schema,self.options,self.array_settings,
                geometry_shape_ids(adapter,identity))

    @staticmethod
    def _scalar_signature(adapter,identity,prop):
        # Compare this publication's availability and exact bytes, including
        # enum metadata. An array/reference row always rebuilds with its current
        # owned reader.
        if prop.array or prop.type=="ObjectRef":return None
        session=adapter.session
        try:
            property_availability=getattr(session,"property_availability",None)
            availability=property_availability(identity,prop.id) if property_availability else ""
            if prop.type=="StringHandle":return session.string_value(identity,prop.id),availability
            stride=element_size(prop.type)
            size=session.property_size(identity,prop.id)
            if not stride or size>stride:return None
            raw=session.read(identity,prop.id,limit=256)
            names=(tuple(session.named_values(prop.id))
                   if prop.semantic in ("Enumeration Value","Bitflag") and hasattr(session,"named_values") else ())
            return size,raw,names,availability
        except SnapshotDataUnavailable:
            return None

    def build(self,adapter,identities,record=None,*,row_titles=(),summary=False,retain=False):
        """Refresh values; retain rows only when the caller keeps the same host Frame."""
        key=self._layout_signature(adapter,identities,record,summary) if retain else None
        reuse=key is not None and key==self._rows_key
        if not reuse:
            self._rows_key=None
            self._rows.clear()
            self._value_rows.clear()
            with ui.VStack(spacing=2,height=0):
                self._build(adapter,identities,record,row_titles=row_titles,summary=summary,
                            retain=key is not None)
            self._rows_key=key
        else:
            # A Frame build callback may update its existing child without
            # replacing it. Changed rows replace only their own child stack.
            self._build(adapter,identities,record,row_titles=row_titles,summary=summary,retain=True,reuse=True)

    def _build(self,adapter,identities,record=None,*,row_titles=(),summary=False,retain=False,reuse=False):
        # Capture replacement must invalidate open Plot actions even when the
        # new inspector has no selection (or no capture at all).
        capture_key=getattr(adapter.session,"capture_key",id(adapter.session)) if adapter is not None else None
        segment=getattr(adapter.session,"segment_epoch",None) if adapter is not None else None
        self._array_generation+=1
        if self.capture_key!=capture_key or self._array_segment!=segment:
            self.capture_key=capture_key
            self._array_segment=segment
            self.array_positions.clear()
            self.array_expanded.clear()
            self._array_denied.clear()
            self._array_unavailable.clear()
            self._array_requested.clear()
            self._array_frame=None
        if adapter is not None:self._observe_array_frame(adapter.session)
        self.summary_available=bool(adapter is not None and self.on_summary is not None
                                    and (identities or record is not None or row_titles))
        if summary and adapter is not None:
            self._build_summary(adapter)
            return
        if adapter is None or (not identities and record is None):
            if adapter is not None and row_titles:
                title=row_titles[0] if len(row_titles)==1 else f"{len(row_titles):,} hierarchy rows selected"
                ui.Label(title,name="title",height=32,elided_text=True,explicit_hover=True, tooltip=title)
                ui.Label("Use the eye control or context menu\nto show or hide the selected contents.",
                    name="muted",height=44)
                return
            ui.Label("Inspect an object",name="title",height=32)
            ui.Label("Select an object to inspect its recorded\nproperties, arrays and references.",name="muted",height=44)
            return
        # Delayed array builds retain this frame's owned metadata and reader.
        adapter=copy(adapter)
        adapter.names=adapter.names.copy()
        adapter.schema=adapter.schema.copy()
        adapter._property_lookup=adapter._property_lookup.copy()
        if len(identities)>1:
            ui.Label(f"{len(identities):,} objects selected",name="title",height=36)
            counts=Counter(adapter.class_name(i) if i>0 else "Scene query" for i in identities)
            for type_name,count in sorted(counts.items()):
                with ui.HStack(height=28):
                    ui.Label(type_name,elided_text=True,explicit_hover=True, tooltip=type_name)
                    ui.Label(str(count),name="muted",width=60,alignment=ui.Alignment.RIGHT_CENTER)
            return
        if record is not None:
            self._record(adapter,record)
            return
        identity=identities[-1]
        if identity not in adapter.instances:
            ui.Label("Object unavailable at this frame",name="muted",height=40)
            return
        if not reuse:
            ui.Label(adapter.name(identity),name="title",height=26,elided_text=True)
            ui.Label(f"{adapter.class_name(identity)} · #{identity}",name="muted",height=16)
            ui.Spacer(height=10)
        if not adapter.instances[identity].alive:
            ui.Label("Not alive at this frame",name="muted",height=32)
            return
        if not reuse:
            for shape_id in geometry_shape_ids(adapter, identity):
                ui.Button(f'Inspect shape #{shape_id}', name='quiet', height=28,
                          tooltip='Open native shape properties and material references in class view',
                          clicked_fn=lambda target=shape_id:self.on_reference(target))
        if not reuse and adapter.class_name(identity)=="PxPlaneGeometry" and not adapter.properties(identity):
            ui.Label("Planes have no recorded geometry parameters. Their local normal is +X; "
                     "shape.localPose and actor.globalPose determine the world normal and position.",
                     name="muted",word_wrap=True,height=0)
        for prop in adapter.properties(identity):
            if retain:
                signature=self._scalar_signature(adapter,identity,prop)
                previous=self._rows.get(prop.id)
                if previous is not None and signature is not None and signature==previous[1]:continue
                values=self._value_rows.get(prop.id)
                if (previous is not None and values is not None and signature is not None
                        and signature[-1]==previous[1][-1]
                        and len(signature[1])==signature[0]==previous[1][0]):
                    values.update(_display_value(prop.type,signature[1],self.options,total_bytes=signature[0]))
                    self._rows[prop.id]=(previous[0],signature)
                    continue
                frame=previous[0] if previous is not None else ui.Frame(height=0)
                with frame:values=self._build_property_row(adapter,identity,prop)
                if values is None:self._value_rows.pop(prop.id,None)
                else:self._value_rows[prop.id]=values
                self._rows[prop.id]=(frame,signature)
            else:self._build_property_row(adapter,identity,prop)

    def _build_property_row(self,adapter,identity,prop):
        with ui.VStack(height=0,spacing=6):
            ui.Spacer(height=4)
            property_availability=getattr(adapter.session,"property_availability",None)
            availability=property_availability(identity,prop.id) if property_availability else ""
            if availability in ("absent_at_frame","not_projected"):
                # Check before partially constructing a scalar/compound row.
                # Every unavailable property keeps one identifiable heading.
                self._heading(prop, identity)
                text = {"absent_at_frame": "No recorded value at this frame",
                        "not_projected": "Value is not in this frame publication"}[availability]
                ui.Label(text,name="muted",height=26,elided_text=True)
                return
            try:return self._property(adapter,identity,prop)
            except SnapshotDataUnavailable:
                text = {"absent_at_frame": "No recorded value at this frame",
                        "not_projected": "Value is not in this frame publication"}.get(
                            availability, "Waiting for a published property value…")
                ui.Label(text,name="muted",height=26,elided_text=True)

    def _heading(self,prop,identity=None,entry=None):
        with ui.HStack(height=18):
            ui.Label(prop.name,name="muted",tooltip=prop.native_type or prop.type,elided_text=True,
                     **(self._context(entry) if entry is not None else {"explicit_hover":True}))
            if identity is not None and can_plot_property(prop):
                ui.Button("Plot",name="link",width=40,height=18,clicked_fn=lambda:self.on_plot(identity,prop))

    def _reference(self,adapter,identity,entry=None):
        available=identity in adapter.instances
        name=adapter.name(identity) if available else "Null" if identity==0 else "Unresolved"
        text=f"{name} [{identity}]"
        entry=entry if entry is not None else _CopyEntry()
        entry.value=DisplayedValue(text)
        ui.Button(text,name="quiet",height=28,enabled=available,
                  clicked_fn=lambda:self.on_reference(identity),tooltip=f"{name} · native ID {identity}",
                  **self._context(entry))
        return entry.value

    def _property(self,adapter,identity,prop):
        entry=_CopyEntry(identity=identity,prop=prop,capture_key=self.capture_key)
        if prop.type in _COMPOUNDS and not prop.array and adapter.session.property_size(identity,prop.id)==element_size(prop.type):
            raw=adapter.session.read(identity,prop.id,limit=256)
            widgets=[]
            with ui.ZStack(height=0):
                ui.Rectangle(name="subtle")
                with ui.HStack():
                    ui.Spacer(width=8)
                    with ui.VStack(height=0,spacing=8):
                        ui.Spacer(height=2)
                        self._heading(prop,identity,entry)
                        self._value(prop.type,raw,entry=entry,widgets=widgets)
                        ui.Spacer(height=2)
                    ui.Spacer(width=8)
            return _ValueWidgets(entry,widgets) if len(raw)==element_size(prop.type) else None
        self._heading(prop,identity,entry)
        session=adapter.session
        if prop.type=="StringHandle":
            value=session.string_value(identity,prop.id) or "—"
            entry.value=DisplayedValue("-" if value=="—" else value)
            ui.Label(entry.value.text,name="muted" if value=="—" else "",height=20,elided_text=True,tooltip=value,
                     **self._context(entry))
            return
        if prop.type=="ObjectRef":
            refs=session.references(identity,prop.id)
            if len(refs)>1 or prop.array:
                self._array(adapter,identity,prop,len(refs),lambda start,count:refs[start:start+count],references=True,entry=entry)
            elif refs:self._reference(adapter,refs[0],entry)
            else:
                entry.value=DisplayedValue("—")
                ui.Label("—",height=26,**self._context(entry))
            return
        size=session.property_size(identity,prop.id)
        value_type,_=array_layout(adapter,prop)
        stride=element_size(value_type)
        native_stride=stride or self._array_stride(adapter,prop)
        if native_stride and size%native_stride==0 and (prop.array or size>native_stride) and not self.array_settings.allows_expansion(size//native_stride):
            self._array(adapter,identity,prop,size//native_stride,None,entry=entry)
        elif stride and size%stride==0 and (prop.array or size>stride):
            self._array(adapter,identity,prop,size//stride,
                lambda start,count:session.read(identity,prop.id,offset=start*stride,limit=count*stride),
                stride=stride,entry=entry,value_type=value_type)
        else:
            raw=session.read(identity,prop.id,limit=256)
            if not self._named(adapter,prop,raw,entry):
                widgets=[]
                self._value(prop.type,raw,total_bytes=size,entry=entry,widgets=widgets)
                if stride and len(raw)==size==stride and prop.semantic not in ("Enumeration Value","Bitflag"):
                    return _ValueWidgets(entry,widgets)

    def _named(self,adapter,prop,raw,entry=None):
        if prop.semantic not in ("Enumeration Value","Bitflag") or not raw or len(raw)>8:
            return False
        if not hasattr(adapter.session,"named_values"):return False
        names=adapter.session.named_values(prop.id)
        value=int.from_bytes(raw,"little")
        entry=entry if entry is not None else _CopyEntry()
        if prop.semantic=="Enumeration Value":
            name=next((name for name,number in names if number==value),"Unknown")
            entry.value=DisplayedValue(f"{name} ({value})")
            ui.Label(entry.value.text,name="mono",height=20,elided_text=True,tooltip=entry.value.text,
                     **self._context(entry))
        elif value==0:
            entry.value=DisplayedValue("(none)")
            ui.Label("(none)",name="muted",height=20,**self._context(entry))
        else:
            known=0
            children=[]
            for name,bit in names:
                if bit and not bit & (bit-1) and value & bit:
                    child=DisplayedValue(name)
                    children.append(child)
                    ui.Label(name,name="mono",height=20,elided_text=True,tooltip=f"{name} · 0x{bit:x}",
                             **self._context(_CopyEntry(child)))
                    known|=bit
            if value & ~known:
                child=DisplayedValue(f"Unknown bits: 0x{value & ~known:x}")
                children.append(child)
                ui.Label(child.text,name="muted",height=26,**self._context(_CopyEntry(child)))
            entry.value=DisplayedValue(children=tuple(children))
        return True

    def _value(self,type_name,raw,*,total_bytes=None,entry=None,widgets=None):
        entry=entry if entry is not None else _CopyEntry()
        entry.value=_display_value(type_name,raw,self.options,total_bytes=total_bytes)
        if type_name in _COMPOUNDS and len(raw)==element_size(type_name):
            with ui.VStack(height=0,spacing=6):
                for (label,_start,_stop,_part_type),child in zip(_COMPOUNDS[type_name],entry.value.children):
                    value=child.text
                    child_entry=_CopyEntry(child)
                    context=self._context(child_entry)
                    with ui.HStack(height=0,spacing=10):
                        ui.Label(label,name="muted",width=52 if type_name=="PxBounds3" else 70,
                            height=20,alignment=ui.Alignment.LEFT_TOP,**context)
                        text_widget=ui.Label(value,name="mono_small",height=0,word_wrap=True,tooltip=value,**context)
                        if widgets is not None:widgets.append((text_widget,child_entry))
        else:
            text=entry.value.text
            text_widget=ui.Label(text,name="mono",height=0,word_wrap=True,tooltip=text,**self._context(entry))
            if widgets is not None:widgets.append((text_widget,entry))
        return entry.value

    def _array(self,adapter,identity,prop,count,read,*,stride=0,references=False,entry=None,key=None,value_type=None):
        key=(identity,prop.id) if key is None else key
        policy=self.array_settings
        generation=self._array_generation
        entry=entry if entry is not None else _CopyEntry()
        summary=f"{count:,} elements"
        entry.value=DisplayedValue(summary)
        if not policy.allows_expansion(count):
            self.array_positions.pop(key,None)
            self.array_expanded.discard(key)
            ui.Label(summary,name="muted",height=20,**self._context(entry))
            ui.Label(f"Expansion limit: {policy.max_array_to_load:,} elements. Change it in General Settings.",
                     name="muted",height=0,word_wrap=True)
            return
        if count==0:
            self.array_positions.pop(key,None)
            self.array_expanded.discard(key)
            ui.Label(summary,name="muted",height=20,**self._context(entry))
            return
        start,window=policy.window(count,self.array_positions.get(key,0),element_bytes=stride or 1)
        if start!=self.array_positions.get(key,0) and key in self.array_expanded and stride and len(key)==2:
            request=(identity,prop.id,start*stride,window*stride)
            self._array_window_denial(adapter.session,request)
            self._request_array_window(request,self._array_stamp(adapter.session,request))
        if window==0:
            ui.Label(summary,name="muted",height=20,**self._context(entry))
            ui.Label("One recorded element exceeds the owned window budget.",name="muted",height=0,word_wrap=True)
            return
        self.array_positions[key]=start
        with ui.CollapsableFrame(summary,height=0,collapsed=key not in self.array_expanded) as section:
            def expanded(collapsed):
                if self._destroyed or generation!=self._array_generation:return
                if collapsed:self.array_expanded.discard(key)
                else:
                    self.array_expanded.add(key)
                    self._retry_array(key)
                    if stride and len(key)==2:
                        first,size=policy.window(count,self.array_positions.get(key,start),element_bytes=stride)
                        request=(identity,prop.id,first*stride,size*stride)
                        self._request_array_window(request,self._array_stamp(adapter.session,request))
            section.set_collapsed_changed_fn(expanded)
            with ui.VStack(height=0,spacing=4):
                content=ui.Frame()
                def current():return not self._destroyed and generation==self._array_generation
                def navigate(position):
                    if not current():return
                    self.array_positions[key],size=policy.window(count,position,element_bytes=stride or 1)
                    self._retry_array(key)
                    if stride and len(key)==2:
                        request=(identity,prop.id,self.array_positions[key]*stride,size*stride)
                        self._request_array_window(request,self._array_stamp(adapter.session,request))
                    content.rebuild()
                def slider(value,maximum,changed):
                    control=ui.IntSlider(min=0,max=maximum,height=28)
                    control.model.set_value(value)
                    control.model.add_value_changed_fn(lambda model:changed(model.as_int))
                def build():
                    # Retained builders may display their old immutable view,
                    # but their callbacks never request/alter a newer one.
                    first,size=policy.window(count,self.array_positions.get(key,start) if current() else start,element_bytes=stride or 1)
                    entry.value=DisplayedValue(summary)
                    with ui.VStack(height=0,spacing=4):
                        range_first,range_count=0,count
                        while range_count>policy.group_size:
                            span=policy.range_span(range_count)
                            index=(first-range_first)//span
                            child_first=range_first+index*span
                            child_count=min(span,range_first+range_count-child_first)
                            ui.Label(f"Range [{child_first:,}–{child_first+child_count-1:,}]",name="muted",height=20)
                            slider(index,(range_count+span-1)//span-1,
                                lambda value,base=range_first,step=span:navigate(base+value*step))
                            range_first,range_count=child_first,child_count
                        if range_count>size:
                            ui.Label(f"Showing [{first:,}–{first+size-1:,}] of [{range_first:,}–{range_first+range_count-1:,}]",
                                     name="muted",height=20)
                            slider(first-range_first,range_count-size,
                                lambda value,base=range_first:navigate(base+value))
                        request=(identity,prop.id,first*stride,size*stride)
                        denied=(self._array_window_denial(adapter.session,request)
                                if current() and stride and len(key)==2 else None)
                        if not current() and stride and len(key)==2 and request in getattr(adapter.session,"unavailable_reads",()):
                            denied="Array window unavailable at this frame."
                        if denied:
                            ui.Label(denied,name="muted",height=0,word_wrap=True)
                            return
                        try:values=read(first,size)
                        except SnapshotDataUnavailable:
                            if current() and key in self.array_expanded and stride and len(key)==2:
                                self._request_array_window(request,self._array_stamp(adapter.session,request))
                            ui.Label("This array window is waiting for a published frame.",name="muted",height=32,word_wrap=True)
                            return
                        # No guessed rows when a raw/native window is incomplete.
                        available=min(size,len(values) if references else len(values)//stride)
                        children=[]
                        with ui.VStack(height=0,spacing=3):
                            for offset in range(available):
                                child_entry=_CopyEntry()
                                with ui.HStack(height=0,spacing=4):
                                    ui.Label(f"[{first+offset}]",name="muted",width=56,**self._context(child_entry))
                                    if references:child=self._reference(adapter,int(values[offset]),child_entry)
                                    else:child=self._value(value_type or prop.type,values[offset*stride:(offset+1)*stride],entry=child_entry)
                                    children.append(child)
                        if available<size:ui.Label("Incomplete recorded array window",name="muted",height=20)
                        entry.value=DisplayedValue(summary,tuple(children))
                content.set_build_fn(build)

    def _record(self,adapter,item):
        if getattr(item, 'kind', '') == 'unresolved':
            ui.Label('Unresolved material', name='title', height=26)
            ui.Label(f'Shape #{item.source_instance_id} · materials[{item.element_index}]',
                     name='muted', height=26)
            ui.Label('This shape records a material reference, but its target is unavailable in this recording/frame. '
                     'Material properties cannot be inspected without the recorded material object.',
                     name='muted', word_wrap=True, height=0)
            ui.Button('Inspect shape properties', name='quiet', height=28,
                      clicked_fn=lambda:self.on_reference(item.source_instance_id))
            return
        ui.Label(item.values[0].as_string,name="title",height=32,elided_text=True)
        ui.Label(item.class_name,name="muted",height=24)
        if element_size(item.class_name):
            try:
                raw=adapter.array_record(item.source_instance_id,item.property_id,item.element_index,item.element_size)
                self._value(item.class_name,raw)
            except SnapshotDataUnavailable:
                request=(item.source_instance_id,item.property_id,item.element_index*item.element_size,item.element_size)
                unavailable=request in getattr(adapter.session,"unavailable_reads",())
                ui.Label("Array element unavailable at this frame" if unavailable else "Waiting for a published array element…",
                         name="muted",height=32)
            except (ValueError,IndexError,RuntimeError) as error:
                ui.Label(str(error),name="muted",word_wrap=True,height=60)
            return
        try:record=adapter.session.array_record(item.source_instance_id,item.property_id,item.element_index)
        except SnapshotDataUnavailable:
            request=(item.source_instance_id,item.property_id,item.element_index)
            unavailable=request in getattr(adapter.session,"unavailable_records",[])
            ui.Label("Record unavailable at this frame" if unavailable else "Waiting for a published record…",
                     name="muted",word_wrap=True,height=60)
            return
        except (ValueError,IndexError,RuntimeError) as error:
            ui.Label(str(error),name="muted",word_wrap=True,height=60)
            return
        for field in record.fields:
            entry=_CopyEntry()
            self._heading(field,None,entry)
            key=("record",item.source_instance_id,item.property_id,item.element_index,field.id)
            stride=self._array_stride(adapter,field)
            if field.type=="ObjectRef" and len(field.data)%4==0:
                count=len(field.data)//4
                if field.array or count>1:
                    self._array(adapter,None,field,count,
                        lambda start,size,data=field.data:struct.unpack(f"<{size}i",data[start*4:(start+size)*4]),
                        references=True,entry=entry,key=key)
                elif count:self._reference(adapter,struct.unpack("<i",field.data)[0],entry)
                else:self._value(field.type,field.data,entry=entry)
            elif field.type=="StringHandle":
                entry.value=DisplayedValue(field.string_value or "—")
                ui.Label(entry.value.text,height=26,elided_text=True,tooltip=field.string_value,
                         **self._context(entry))
            elif stride and len(field.data)%stride==0 and (field.array or len(field.data)>stride) and not self.array_settings.allows_expansion(len(field.data)//stride):
                self._array(adapter,None,field,len(field.data)//stride,None,entry=entry,key=key)
            elif element_size(field.type) and len(field.data)%stride==0 and (field.array or len(field.data)>stride):
                self._array(adapter,None,field,len(field.data)//stride,
                    lambda start,count,data=field.data,width=stride:data[start*width:(start+count)*width],
                    stride=stride,entry=entry,key=key)
            elif not self._named(adapter,field,field.data,entry):self._value(field.type,field.data,entry=entry)
