/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Session.h"
#include "../PVDCore/LoadControl.h"
#include "Presentation.h"
#include <array>
#include <limits>
#include "Profiling.h"
#include "ErrorStream.h"
#include "AnalysisAvailability.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace ovdnext;
namespace ovdnext { void bindCaptureControl(py::class_<Session,std::shared_ptr<Session>>&); }

template<class Owner, class T>
py::array ownedArray(const std::shared_ptr<Owner>& owner, const std::vector<T>& values, std::vector<py::ssize_t> shape) {
    std::vector<py::ssize_t> strides(shape.size());
    py::ssize_t stride=sizeof(T);
    for(size_t j=shape.size(); j>0; --j) { strides[j-1]=stride; stride*=shape[j-1]; }
    // py::cast(owner) pins this owning immutable snapshot until the final NumPy
    // view dies, even after the Session has sought, closed or been destroyed.
    py::array result(py::dtype::of<T>(),shape,strides,values.data(),py::cast(owner));
    result.attr("setflags")(false);
    return result;
}
PYBIND11_MODULE(_pvd_native, m) {
    m.doc() = "Owned, bounded Python access to the original PVD/OVD engine";
    py::class_<SceneInfo>(m, "SceneInfo")
        .def_readonly("id", &SceneInfo::id).def_readonly("name", &SceneInfo::name)
        .def_readonly("frame_count", &SceneInfo::frameCount).def_readonly("current_frame", &SceneInfo::currentFrame)
        .def_readonly("instance_id", &SceneInfo::instanceId)
        .def_readonly("stream_id", &SceneInfo::streamId).def_readonly("alive", &SceneInfo::alive)
        .def_readonly("recording_segment", &SceneInfo::recordingSegment);
    py::class_<ClassInfo>(m, "ClassInfo")
        .def_readonly("id", &ClassInfo::id).def_readonly("base_id", &ClassInfo::baseId).def_readonly("name", &ClassInfo::name)
        .def_readonly("byte_size", &ClassInfo::byteSize);
    py::class_<InstanceInfo>(m, "InstanceInfo")
        .def_readonly("id", &InstanceInfo::id).def_readonly("class_id", &InstanceInfo::classId)
        .def_readonly("scene_id", &InstanceInfo::sceneId).def_readonly("physics_id", &InstanceInfo::physicsId)
        .def_readonly("alive", &InstanceInfo::alive).def_readonly("scene_joint", &InstanceInfo::sceneJoint);
    py::class_<PropertyInfo>(m, "PropertyInfo")
        .def_readonly("id", &PropertyInfo::id).def_readonly("name", &PropertyInfo::name)
        .def_readonly("type", &PropertyInfo::type).def_readonly("native_type", &PropertyInfo::nativeType)
        .def_readonly("semantic", &PropertyInfo::semantic)
        .def_readonly("array", &PropertyInfo::array).def_readonly("field_count", &PropertyInfo::fieldCount);
    py::class_<QueryRecord>(m,"QueryRecord")
        .def_readonly("query_id",&QueryRecord::queryId).def_readonly("scene_instance_id",&QueryRecord::sceneInstanceId)
        .def_readonly("property_id",&QueryRecord::propertyId).def_readonly("class_id",&QueryRecord::classId)
        .def_readonly("element_index",&QueryRecord::elementIndex).def_readonly("element_size",&QueryRecord::elementSize);
    py::class_<HierarchyChanges>(m,"HierarchyChanges")
        .def_readonly("revision",&HierarchyChanges::revision).def_readonly("schema_changed",&HierarchyChanges::schemaChanged)
        .def_readonly("query_records_changed",&HierarchyChanges::queryRecordsChanged)
        .def_readonly("changed_ids",&HierarchyChanges::changedIds).def_readonly("removed_ids",&HierarchyChanges::removedIds);
    py::class_<ExportCapabilities>(m,"ExportCapabilities")
        .def_readonly("format",&ExportCapabilities::format).def_readonly("save",&ExportCapabilities::save)
        .def_readonly("clip",&ExportCapabilities::clip).def_readonly("repx",&ExportCapabilities::repx)
        .def_readonly("save_reason",&ExportCapabilities::saveReason).def_readonly("clip_reason",&ExportCapabilities::clipReason).def_readonly("repx_reason",&ExportCapabilities::repxReason);
    py::class_<RecordField>(m,"RecordField")
        .def_readonly("id",&RecordField::id).def_readonly("name",&RecordField::name).def_readonly("type",&RecordField::type)
        .def_readonly("native_type",&RecordField::nativeType).def_readonly("semantic",&RecordField::semantic)
        .def_readonly("array",&RecordField::array).def_readonly("string_value",&RecordField::stringValue)
        .def_property_readonly("data",[](const RecordField& f){return py::bytes(reinterpret_cast<const char*>(f.data.data()),f.data.size());});
    py::class_<RecordInfo>(m,"RecordInfo").def_readonly("class_id",&RecordInfo::classId).def_readonly("fields",&RecordInfo::fields);
    py::class_<SessionInfo>(m, "SessionInfo")
        .def_readonly("ovd", &SessionInfo::ovd).def_readonly("instance_count", &SessionInfo::instanceCount)
        .def_readonly("max_frame", &SessionInfo::maxFrame).def_readonly("live", &SessionInfo::live)
        .def_readonly("connected", &SessionInfo::connected).def_readonly("ended", &SessionInfo::ended)
        .def_readonly("completed_serial", &SessionInfo::completedSerial)
        .def_readonly("segment_epoch", &SessionInfo::segmentEpoch).def_readonly("error", &SessionInfo::error)
        .def_readonly("unpublished_changes", &SessionInfo::unpublishedChanges)
        .def_readonly("live_max_frame", &SessionInfo::liveMaxFrame).def_readonly("live_current_frame", &SessionInfo::liveCurrentFrame)
        .def_readonly("recording_size", &SessionInfo::recordingSize);
    py::class_<LiveUpdate>(m, "LiveUpdate")
        .def_readonly("ready", &LiveUpdate::ready).def_readonly("changed", &LiveUpdate::changed)
        .def_readonly("serial", &LiveUpdate::serial).def_readonly("segment_epoch", &LiveUpdate::segmentEpoch)
        .def_readonly("segment_transition_serial", &LiveUpdate::segmentTransitionSerial)
        .def_readonly("current_frame", &LiveUpdate::currentFrame).def_readonly("max_frame", &LiveUpdate::maxFrame)
        .def_readonly("scenes", &LiveUpdate::scenes);
    py::enum_<GeometryKind>(m, "GeometryKind")
        .value("SPHERE",GeometryKind::Sphere).value("BOX",GeometryKind::Box)
        .value("CAPSULE",GeometryKind::Capsule).value("PLANE",GeometryKind::Plane)
        .value("MESH",GeometryKind::Mesh).value("LINES",GeometryKind::Lines).value("POINTS",GeometryKind::Points);
    py::class_<GeometryMesh,std::shared_ptr<GeometryMesh>>(m,"GeometryMesh")
        .def_readonly("id",&GeometryMesh::id).def_readonly("topology_id",&GeometryMesh::topologyId)
        .def_readonly("source_id",&GeometryMesh::sourceId)
        .def_property_readonly("vertices",[](const std::shared_ptr<GeometryMesh>& o){return ownedArray(o,o->vertices,{py::ssize_t(o->vertices.size()/3),3});})
        .def_property_readonly("indices",[](const std::shared_ptr<GeometryMesh>& o){return ownedArray(o,*o->indices,{py::ssize_t(o->indices->size())});});
    auto geometry=py::class_<GeometrySnapshot,std::shared_ptr<GeometrySnapshot>>(m,"GeometrySnapshot")
        .def_readonly("segment_epoch",&GeometrySnapshot::segmentEpoch)
        .def_readonly("meshes",&GeometrySnapshot::meshes).def_readonly("diagnostics",&GeometrySnapshot::diagnostics);
#define GEOMETRY_ARRAY(name,member) \
    geometry.def_property_readonly(name,[](const std::shared_ptr<GeometrySnapshot>& o){return ownedArray(o,o->member,{py::ssize_t(o->objectIds.size())});});
    GEOMETRY_ARRAY("object_ids",objectIds)
    GEOMETRY_ARRAY("shape_ids",shapeIds)
    GEOMETRY_ARRAY("geometry_ids",geometryIds)
    GEOMETRY_ARRAY("scene_ids",sceneIds)
    GEOMETRY_ARRAY("actor_classes",actorClasses)
    GEOMETRY_ARRAY("kinds",kinds)
    GEOMETRY_ARRAY("sleeping",sleeping)
    GEOMETRY_ARRAY("double_sided",doubleSided)
    GEOMETRY_ARRAY("particle_radius_modes",particleRadiusModes)
    GEOMETRY_ARRAY("shape_flags",shapeFlags)
    GEOMETRY_ARRAY("actor_flags",actorFlags)
    GEOMETRY_ARRAY("resource_ids",resourceIds)
#undef GEOMETRY_ARRAY
    geometry.def_property_readonly("override_colors",[](const std::shared_ptr<GeometrySnapshot>& o){return ownedArray(o,o->overrideColors,{py::ssize_t(o->overrideColors.size()/4),4});})
        .def_property_readonly("transforms",[](const std::shared_ptr<GeometrySnapshot>& o){return ownedArray(o,o->transforms,{py::ssize_t(o->objectIds.size()),4,4});})
        .def_property_readonly("dimensions",[](const std::shared_ptr<GeometrySnapshot>& o){return ownedArray(o,o->dimensions,{py::ssize_t(o->objectIds.size()),3});})
        .def_property_readonly("bounds",[](const std::shared_ptr<GeometrySnapshot>& o){return ownedArray(o,o->bounds,{py::ssize_t(o->objectIds.size()),2,3});});
    py::enum_<OverlayKind>(m,"OverlayKind")
        .value("POINT",OverlayKind::Point).value("LINE",OverlayKind::Line).value("TRIANGLE",OverlayKind::Triangle).value("ARROW",OverlayKind::Arrow);
    py::enum_<OverlayColorRole>(m,"OverlayColorRole")
        .value("CAPTURED",OverlayColorRole::Captured).value("BOUNDS",OverlayColorRole::Bounds)
        .value("CENTER_OF_MASS",OverlayColorRole::CenterOfMass).value("VELOCITY",OverlayColorRole::Velocity)
        .value("AXIS_X",OverlayColorRole::AxisX).value("AXIS_Y",OverlayColorRole::AxisY).value("AXIS_Z",OverlayColorRole::AxisZ)
        .value("JOINT_PARENT",OverlayColorRole::JointParent).value("JOINT_CHILD",OverlayColorRole::JointChild)
        .value("CONTACT_POINT",OverlayColorRole::ContactPoint).value("CONTACT_NORMAL",OverlayColorRole::ContactNormal);
    py::enum_<OverlayCategory>(m,"OverlayCategory")
        .value("BOUNDS",OverlayBounds).value("AXES",OverlayAxes).value("VELOCITY",OverlayVelocity).value("ANGULAR_VELOCITY",OverlayAngularVelocity)
        .value("JOINTS",OverlayJoints).value("CONTACTS",OverlayContacts).value("QUERIES",OverlayQueries).value("CUSTOM",OverlayCustom)
        .value("CENTER_OF_MASS",OverlayCenterOfMass).value("ALL",OverlayAll);
    auto overlays=py::class_<OverlaySnapshot,std::shared_ptr<OverlaySnapshot>>(m,"OverlaySnapshot")
        .def_readonly("segment_epoch",&OverlaySnapshot::segmentEpoch).def_readonly("current_frame",&OverlaySnapshot::currentFrame)
        .def_readonly("query_geometry",&OverlaySnapshot::queryGeometry).def_readonly("query_records",&OverlaySnapshot::queryRecords)
        .def_readonly("mask",&OverlaySnapshot::mask).def_readonly("ready",&OverlaySnapshot::ready)
        .def_readonly("family_scales",&OverlaySnapshot::familyScales)
        .def_readonly("master_scale",&OverlaySnapshot::masterScale)
        .def_property_readonly("joint_scale",[](const OverlaySnapshot& o)->py::object{return o.jointScaling.independent?py::cast(o.jointScaling.scale):py::none();})
        .def_property_readonly("joint_base_length",[](const OverlaySnapshot& o){return o.jointScaling.baseLength;}).def_readonly("diagnostics",&OverlaySnapshot::diagnostics);
#define OVERLAY_ARRAY(name,member) \
    overlays.def_property_readonly(name,[](const std::shared_ptr<OverlaySnapshot>& o){return ownedArray(o,o->member,{py::ssize_t(o->kinds.size())});});
    OVERLAY_ARRAY("kinds",kinds)
    OVERLAY_ARRAY("color_roles",colorRoles)
    OVERLAY_ARRAY("categories",categories)
    OVERLAY_ARRAY("object_ids",objectIds)
    OVERLAY_ARRAY("scene_ids",sceneIds)
#undef OVERLAY_ARRAY
    overlays.def_property_readonly("related_ids",[](const std::shared_ptr<OverlaySnapshot>& o){return ownedArray(o,o->relatedIds,{py::ssize_t(o->kinds.size()),2});})
        .def_property_readonly("positions",[](const std::shared_ptr<OverlaySnapshot>& o){return ownedArray(o,o->positions,{py::ssize_t(o->kinds.size()),3,3});})
        .def_property_readonly("colors",[](const std::shared_ptr<OverlaySnapshot>& o){return ownedArray(o,o->colors,{py::ssize_t(o->kinds.size()),3,4});});
    py::class_<MessageInfo>(m,"MessageInfo")
        .def_readonly("frame_id",&MessageInfo::frameId).def_readonly("type_name",&MessageInfo::typeName)
        .def_readonly("message",&MessageInfo::message).def_readonly("file",&MessageInfo::file)
        .def_readonly("line",&MessageInfo::line).def_readonly("type",&MessageInfo::type).def_readonly("handle",&MessageInfo::handle);
    py::class_<FrameTiming>(m,"FrameTiming")
        .def_readonly("supported",&FrameTiming::supported).def_readonly("inner_ms",&FrameTiming::innerMs).def_readonly("total_ms",&FrameTiming::totalMs);
    py::class_<ScalarHistory,std::shared_ptr<ScalarHistory>>(m,"ScalarHistory")
        .def_readonly("property_name",&ScalarHistory::propertyName).def_readonly("type",&ScalarHistory::type)
        .def_property_readonly("frames",[](const std::shared_ptr<ScalarHistory>& o){return ownedArray(o,o->frames,{py::ssize_t(o->frames.size())});})
        .def_property_readonly("values",[](const std::shared_ptr<ScalarHistory>& o){return ownedArray(o,o->values,{py::ssize_t(o->values.size())});})
        .def_property_readonly("valid",[](const std::shared_ptr<ScalarHistory>& o){return ownedArray(o,o->valid,{py::ssize_t(o->valid.size())});});
    bindEvidence(m);
    bindPresentation(m);
    py::class_<pvdapp::LoadProgress>(m, "LoadProgress")
        .def_readonly("stage", &pvdapp::LoadProgress::stage)
        .def_readonly("processed", &pvdapp::LoadProgress::processed)
        .def_readonly("total", &pvdapp::LoadProgress::total)
        .def_readonly("interruptible", &pvdapp::LoadProgress::interruptible);
    py::class_<pvdapp::LoadControl, std::shared_ptr<pvdapp::LoadControl>>(m, "LoadControl")
        .def(py::init<>())
        .def("cancel", &pvdapp::LoadControl::cancel)
        .def_property_readonly("cancelled", &pvdapp::LoadControl::cancelled)
        .def("progress", &pvdapp::LoadControl::progress);
    auto session=py::class_<Session, std::shared_ptr<Session>>(m, "Session");
    session.def(py::init<const std::string&, const std::string&, std::shared_ptr<pvdapp::LoadControl>>(),
            py::arg("path"), py::arg("temporary_directory"), py::arg("control")=nullptr,
             py::call_guard<py::gil_scoped_release>())
        .def_static("listen", &Session::listen, py::arg("port"), py::arg("recording_path") = "",
                    py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("capture_key",&Session::captureKey)
        .def("presentation",[](Session& s,uint64_t scene,int64_t frame,const std::vector<int32_t>& inspect,
            const std::vector<std::array<int64_t,4>>& rawReads,const std::vector<std::array<int64_t,3>>& rawRecords,uint32_t mask,float scale,py::object speedObject,bool allowUnavailable,py::object jointObject,float jointBaseLength,const std::map<uint32_t,float>& familyScales,float masterScale,bool includeEvidence){
            JointScaleRequest jointScaling; jointScaling.independent=!jointObject.is_none();
            jointScaling.scale=jointScaling.independent?jointObject.cast<float>():1.0f; jointScaling.baseLength=jointBaseLength;
            const bool capturedSpeed=speedObject.is_none();
            const float speed=capturedSpeed?1.0f:speedObject.cast<float>();
            auto id=[](int64_t value){if(value<std::numeric_limits<int32_t>::min() || value>std::numeric_limits<int32_t>::max())throw std::out_of_range("Presentation identity exceeds int32");return int32_t(value);};
            auto index=[](int64_t value){if(value<0 || uint64_t(value)>std::numeric_limits<uint32_t>::max())throw std::out_of_range("Presentation byte/record index exceeds uint32");return uint32_t(value);};
            std::vector<ReadRequest> reads;for(const auto& r:rawReads)reads.push_back({id(r[0]),id(r[1]),index(r[2]),index(r[3])});
            std::vector<RecordRequest> records;for(const auto& r:rawRecords)records.push_back({id(r[0]),id(r[1]),index(r[2])});
            py::gil_scoped_release release;return s.presentation(scene,frame,inspect,reads,records,mask,scale,speed,allowUnavailable,capturedSpeed,jointScaling,familyScales,masterScale,includeEvidence);
        },py::arg("scene_id")=0,py::arg("frame")=-1,py::arg("inspect_ids")=std::vector<int32_t>{},
          py::arg("read_requests")=std::vector<std::array<int64_t,4>>{},py::arg("record_requests")=std::vector<std::array<int64_t,3>>{},
          py::arg("overlay_mask")=0,py::arg("gizmo_scale")=1.0f,py::arg("tolerance_speed")=10.0f,
          py::arg("allow_unavailable_requests")=false,py::kw_only(),py::arg("joint_scale")=py::none(),py::arg("joint_base_length")=1.0f,py::arg("family_scales")=std::map<uint32_t,float>{},py::arg("master_scale")=1.0f,py::arg("include_evidence")=false)
        .def("evidence_metadata",&Session::evidenceMetadata,py::arg("offset")=0,py::arg("limit")=512,py::call_guard<py::gil_scoped_release>())
        .def("info", &Session::info, py::call_guard<py::gil_scoped_release>())
        .def("geometry_snapshot", &Session::geometrySnapshot, py::call_guard<py::gil_scoped_release>())
        .def("overlay_snapshot",[](Session& s,uint32_t mask,float scale,float speed,py::object jointObject,float jointBaseLength,const std::map<uint32_t,float>& familyScales,float masterScale){
            JointScaleRequest jointScaling; jointScaling.independent=!jointObject.is_none();
            jointScaling.scale=jointScaling.independent?jointObject.cast<float>():1.0f; jointScaling.baseLength=jointBaseLength;
            py::gil_scoped_release release;return s.overlaySnapshot(mask,scale,speed,jointScaling,familyScales,masterScale);
        },py::arg("mask")=uint32_t(OverlayAll),py::arg("gizmo_scale")=1.0f,py::arg("tolerance_speed")=10.0f,
          py::kw_only(),py::arg("joint_scale")=py::none(),py::arg("joint_base_length")=1.0f,py::arg("family_scales")=std::map<uint32_t,float>{},py::arg("master_scale")=1.0f)
        .def("hierarchy_changes",&Session::hierarchyChanges,py::arg("previous_revision"),py::call_guard<py::gil_scoped_release>())
        .def("hierarchy_revision",&Session::hierarchyRevision,py::call_guard<py::gil_scoped_release>())
        .def("query_records",&Session::queryRecords,py::call_guard<py::gil_scoped_release>())
        .def("array_record",&Session::arrayRecord,py::arg("instance_id"),py::arg("property_id"),py::arg("element_index"),py::call_guard<py::gil_scoped_release>())
        .def("named_values",&Session::namedValues,py::arg("property_id"),py::call_guard<py::gil_scoped_release>())
        .def("export_capabilities",&Session::exportCapabilities,py::call_guard<py::gil_scoped_release>())
        .def("save",&Session::save,py::arg("path"),py::arg("overwrite")=false,py::call_guard<py::gil_scoped_release>())
        .def("save_clip",&Session::saveClip,py::arg("path"),py::arg("scene_id"),py::arg("start"),py::arg("stop"),py::arg("overwrite")=false,py::call_guard<py::gil_scoped_release>())
        .def("export_repx",[](Session& s,const std::string& path,py::object ids,bool overwrite){
            const bool all=ids.is_none();const auto selected=all?std::vector<int32_t>{}:ids.cast<std::vector<int32_t>>();
            py::gil_scoped_release release;return s.exportRepX(path,selected,all,overwrite);
        },py::arg("path"),py::arg("instance_ids")=py::none(),py::arg("overwrite")=false)
        .def("message_count", &Session::messageCount, py::call_guard<py::gil_scoped_release>())
        .def("messages", &Session::messages, py::arg("offset")=0,py::arg("limit")=1000,py::call_guard<py::gil_scoped_release>())
        .def("frame_timing", &Session::frameTiming, py::call_guard<py::gil_scoped_release>())
        .def("scalar_history", &Session::scalarHistory,py::arg("scene_id"),py::arg("instance_id"),py::arg("property_id"),py::arg("start"),py::arg("stop"),py::call_guard<py::gil_scoped_release>())
        .def("scenes", &Session::scenes, py::call_guard<py::gil_scoped_release>())
        .def("classes", &Session::classes, py::call_guard<py::gil_scoped_release>())
        .def("instances", &Session::instances, py::arg("offset") = 0, py::arg("limit") = 10000,
             py::call_guard<py::gil_scoped_release>())
        .def("properties", &Session::properties, py::call_guard<py::gil_scoped_release>())
        .def("read", [](Session& s, int32_t id, int32_t prop, uint32_t offset, uint32_t limit) {
            std::vector<uint8_t> value;
            { py::gil_scoped_release release; value = s.read(id, prop, offset, limit); }
            return py::bytes(reinterpret_cast<const char*>(value.data()), value.size());
        }, py::arg("instance_id"), py::arg("property_id"), py::arg("offset") = 0, py::arg("limit") = 65536)
        .def("property_size", &Session::propertySize, py::call_guard<py::gil_scoped_release>())
        .def("references", &Session::references, py::call_guard<py::gil_scoped_release>())
        .def("string_value", &Session::stringValue, py::call_guard<py::gil_scoped_release>())
        .def("seek", &Session::seek, py::call_guard<py::gil_scoped_release>())
        .def("refresh", &Session::refresh, py::call_guard<py::gil_scoped_release>())
        .def("refresh_live", &Session::refreshLive, py::call_guard<py::gil_scoped_release>())
        .def("stop_live", &Session::stopLive, py::call_guard<py::gil_scoped_release>())
        .def("flush_recording", &Session::flushRecording, py::call_guard<py::gil_scoped_release>())
        .def("recording_path", &Session::recordingPath, py::call_guard<py::gil_scoped_release>());
    bindProfiling(m);
    bindCaptureControl(session);
    bindErrorStream(m);
    bindAnalysisAvailability(m);
}
