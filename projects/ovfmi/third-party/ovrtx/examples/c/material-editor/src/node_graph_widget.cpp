// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "node_graph_widget.hpp"

#include <QtNodes/ConnectionIdUtils>
#include <QtNodes/NodeData>
#include <QtNodes/NodeGroup>
#include <QtNodes/GroupGraphicsObject>
#include <QtNodes/StyleCollection>
#include <QAction>
#include <QColor>
#include <QFont>
#include <QGraphicsScene>
#include <QGraphicsSimpleTextItem>
#include <QMenu>

#include <algorithm>

using namespace QtNodes;

// ---------------------------------------------------------------------------
// ShaderGraphModel
// ---------------------------------------------------------------------------

ShaderGraphModel::ShaderGraphModel()
{
}

void ShaderGraphModel::setGraph(const std::vector<ShaderNode> &nodes,
                                const std::vector<GraphConnection> &connections,
                                const std::string &material_path)
{
    // Clear existing graph — connections first, then nodes
    for (const ConnectionId &cid : connections_) {
        emit connectionDeleted(cid);
    }
    connections_.clear();
    for (auto &[id, _] : nodes_) {
        Q_UNUSED(_);
        emit nodeDeleted(id);
    }
    nodes_.clear();
    next_id_ = 0;

    std::unordered_map<std::string, std::vector<std::string>> dest_port_names;
    auto addDestPort = [&dest_port_names](const std::string &path,
                                          const std::string &port) {
        std::vector<std::string> &ports = dest_port_names[path];
        if (std::find(ports.begin(), ports.end(), port) == ports.end()) {
            ports.push_back(port);
        }
    };
    for (const GraphConnection &conn : connections) {
        addDestPort(conn.dest_node_path, conn.dest_input);
    }

    // --- Create material node ---
    NodeInfo mat_node;
    mat_node.id = next_id_++;
    mat_node.prim_path = material_path;
    mat_node.caption = QString::fromStdString(
        material_path.substr(material_path.rfind('/') + 1));
    mat_node.subcaption = "Material";
    mat_node.position = QPointF(300.0, 0.0);

    // Material node gets input ports for connections targeting it
    auto mat_ports = dest_port_names.find(material_path);
    if (mat_ports != dest_port_names.end()) {
        for (const std::string &port : mat_ports->second) {
            mat_node.in_port_names.push_back(QString::fromStdString(port));
        }
    }
    nodes_[mat_node.id] = mat_node;

    // --- Create shader/nodegraph nodes ---
    std::unordered_map<std::string, NodeId> path_to_node;
    path_to_node[material_path] = mat_node.id;

    double y_offset = 0.0;
    for (const ShaderNode &shader : nodes) {
        NodeInfo info;
        info.id = next_id_++;
        info.prim_path = shader.prim_path;
        info.caption = QString::fromStdString(shader.display_name);
        info.subcaption = QString::fromStdString(
            shader.mdl_sub_id.empty() ? shader.mdl_asset : shader.mdl_sub_id);
        info.position = QPointF(0.0, y_offset);
        if (shader.has_node_graph_position) {
            info.position = QPointF(shader.node_graph_position[0],
                                    shader.node_graph_position[1]);
            info.has_authored_position = true;
        }
        info.is_node_graph = shader.is_node_graph;
        y_offset += 150.0;

        for (const ShaderInput &input : shader.inputs) {
            addDestPort(shader.prim_path, input.name);
        }
        for (const std::string &port : dest_port_names[shader.prim_path]) {
            info.in_port_names.push_back(QString::fromStdString(port));
        }
        for (const std::string &out_name : shader.output_names) {
            info.out_port_names.push_back(
                QString::fromStdString(out_name));
        }

        path_to_node[shader.prim_path] = info.id;
        nodes_[info.id] = info;
    }

    // Emit nodeCreated
    for (auto &[id, _] : nodes_) {
        Q_UNUSED(_);
        emit nodeCreated(id);
    }

    // --- Create connections ---
    for (const GraphConnection &conn : connections) {
        auto src_it = path_to_node.find(conn.source_node_path);
        auto dst_it = path_to_node.find(conn.dest_node_path);
        if (src_it == path_to_node.end() || dst_it == path_to_node.end()) {
            continue;
        }

        NodeId src_node = src_it->second;
        NodeId dst_node = dst_it->second;

        PortIndex src_port = InvalidPortIndex;
        const NodeInfo &src_info = nodes_[src_node];
        for (PortIndex i = 0; i < static_cast<PortIndex>(src_info.out_port_names.size()); ++i) {
            if (src_info.out_port_names[i] == QString::fromStdString(conn.source_output)) {
                src_port = i;
                break;
            }
        }

        PortIndex dst_port = InvalidPortIndex;
        const NodeInfo &dst_info = nodes_[dst_node];
        for (PortIndex i = 0; i < static_cast<PortIndex>(dst_info.in_port_names.size()); ++i) {
            if (dst_info.in_port_names[i] == QString::fromStdString(conn.dest_input)) {
                dst_port = i;
                break;
            }
        }

        if (src_port == InvalidPortIndex || dst_port == InvalidPortIndex) {
            continue;
        }

        ConnectionId cid{src_node, src_port, dst_node, dst_port};
        connections_.insert(cid);
        emit connectionCreated(cid);
    }
}

std::unordered_set<NodeId> ShaderGraphModel::allNodeIds() const
{
    std::unordered_set<NodeId> ids;
    for (auto &[id, _] : nodes_) {
        Q_UNUSED(_);
        ids.insert(id);
    }
    return ids;
}

std::unordered_set<ConnectionId> ShaderGraphModel::allConnectionIds(NodeId nodeId) const
{
    std::unordered_set<ConnectionId> result;
    for (const ConnectionId &cid : connections_) {
        if (cid.outNodeId == nodeId || cid.inNodeId == nodeId) {
            result.insert(cid);
        }
    }
    return result;
}

std::unordered_set<ConnectionId> ShaderGraphModel::connections(
    NodeId nodeId, PortType portType, PortIndex portIndex) const
{
    std::unordered_set<ConnectionId> result;
    for (const ConnectionId &cid : connections_) {
        if (portType == PortType::Out &&
            cid.outNodeId == nodeId && cid.outPortIndex == portIndex) {
            result.insert(cid);
        }
        if (portType == PortType::In &&
            cid.inNodeId == nodeId && cid.inPortIndex == portIndex) {
            result.insert(cid);
        }
    }
    return result;
}

bool ShaderGraphModel::connectionExists(ConnectionId connectionId) const
{
    return connections_.count(connectionId) > 0;
}

NodeId ShaderGraphModel::newNodeId() { return next_id_++; }

NodeId ShaderGraphModel::addNode(QString const) { return {}; }
bool ShaderGraphModel::connectionPossible(ConnectionId) const { return false; }
void ShaderGraphModel::addConnection(ConnectionId) {}
bool ShaderGraphModel::deleteConnection(ConnectionId) { return false; }
bool ShaderGraphModel::deleteNode(NodeId) { return false; }

bool ShaderGraphModel::nodeExists(NodeId nodeId) const
{
    return nodes_.count(nodeId) > 0;
}

QVariant ShaderGraphModel::nodeData(NodeId nodeId, NodeRole role) const
{
    auto it = nodes_.find(nodeId);
    if (it == nodes_.end()) {
        return {};
    }
    const NodeInfo &info = it->second;

    switch (role) {
    case NodeRole::Type:
        return QString("ShaderNode");
    case NodeRole::Position:
        return info.position;
    case NodeRole::Size:
        return info.size;
    case NodeRole::CaptionVisible:
        return true;
    case NodeRole::Caption:
        return info.caption;
    case NodeRole::Style: {
        QtNodes::NodeStyle style = QtNodes::StyleCollection::nodeStyle();
        if (info.is_node_graph) {
            style.NormalBoundaryColor = QColor(220, 160, 50);
            style.SelectedBoundaryColor = QColor(255, 200, 80);
            style.PenWidth = 2.5f;
        }
        return style.toJson().toVariantMap();
    }
    case NodeRole::InternalData:
        return {};
    case NodeRole::InPortCount:
        return static_cast<unsigned int>(info.in_port_names.size());
    case NodeRole::OutPortCount:
        return static_cast<unsigned int>(info.out_port_names.size());
    case NodeRole::Widget:
        return {};
    default:
        break;
    }
    return {};
}

bool ShaderGraphModel::setNodeData(NodeId nodeId, NodeRole role, QVariant value)
{
    auto it = nodes_.find(nodeId);
    if (it == nodes_.end()) return false;

    if (role == NodeRole::Position) {
        it->second.position = value.toPointF();
        emit nodePositionUpdated(nodeId);
        return true;
    }
    if (role == NodeRole::Size) {
        it->second.size = value.toSize();
        return true;
    }
    return false;
}

QVariant ShaderGraphModel::portData(NodeId nodeId, PortType portType,
                                    PortIndex portIndex, PortRole role) const
{
    auto it = nodes_.find(nodeId);
    if (it == nodes_.end()) return {};
    const NodeInfo &info = it->second;

    switch (role) {
    case PortRole::Data:
        return {};
    case PortRole::DataType:
        return QVariant::fromValue(QtNodes::NodeDataType{"shader", "shader"});
    case PortRole::ConnectionPolicyRole:
        return QVariant::fromValue(QtNodes::ConnectionPolicy::One);
    case PortRole::CaptionVisible:
        return true;
    case PortRole::Caption:
        if (portType == PortType::In &&
            portIndex < static_cast<PortIndex>(info.in_port_names.size()))
            return info.in_port_names[portIndex];
        if (portType == PortType::Out &&
            portIndex < static_cast<PortIndex>(info.out_port_names.size()))
            return info.out_port_names[portIndex];
        return QString("?");
    default:
        break;
    }
    return {};
}

bool ShaderGraphModel::setPortData(NodeId, PortType, PortIndex,
                                   QVariant const &, PortRole)
{
    return false;
}

std::string ShaderGraphModel::primPathForNode(NodeId nodeId) const
{
    auto it = nodes_.find(nodeId);
    return it != nodes_.end() ? it->second.prim_path : std::string();
}

bool ShaderGraphModel::isNodeGraph(NodeId nodeId) const
{
    auto it = nodes_.find(nodeId);
    return it != nodes_.end() && it->second.is_node_graph;
}

bool ShaderGraphModel::hasAuthoredPosition(NodeId nodeId) const
{
    auto it = nodes_.find(nodeId);
    return it != nodes_.end() && it->second.has_authored_position;
}

// ---------------------------------------------------------------------------
// NodeGraphWidget
// ---------------------------------------------------------------------------

NodeGraphWidget::NodeGraphWidget(QWidget *parent)
    : QWidget(parent)
{
    layout_ = new QVBoxLayout(this);
    layout_->setContentsMargins(0, 0, 0, 0);

    model_ = new ShaderGraphModel();
    scene_ = new BasicGraphicsScene(*model_, this);

    view_ = new GraphicsView(scene_);
    view_->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(view_, &QWidget::customContextMenuRequested,
            this, &NodeGraphWidget::showContextMenu);
    layout_->addWidget(view_);

    // Node selection → property panel
    selection_connection_ = connect(scene_, &QGraphicsScene::selectionChanged, this, [this]() {
        if (shutting_down_) return;
        for (auto nodeId : model_->allNodeIds()) {
            QtNodes::NodeGraphicsObject *ngo = scene_->nodeGraphicsObject(nodeId);
            if (ngo && ngo->isSelected()) {
                std::string path = model_->primPathForNode(nodeId);
                if (!path.empty()) {
                    emit nodeSelected(QString::fromStdString(path));
                }
                return;
            }
        }
    });

    setMinimumHeight(200);
}

NodeGraphWidget::~NodeGraphWidget()
{
    shutting_down_ = true;
    disconnect(selection_connection_);
}

void NodeGraphWidget::setMaterialGraph(const MaterialGraph &graph)
{
    current_graph_ = graph;
    expanded_node_graphs_.clear();
    saved_positions_.clear();
    ever_expanded_.clear();
    for (const auto &[ng_path, ng_data] : current_graph_.node_graphs) {
        if (ng_data.collapsed_node.node_graph_expanded) {
            expanded_node_graphs_.insert(ng_path);
        }
    }
    rebuildModel();
}

bool NodeGraphWidget::selectNodeByPrimPath(QString prim_path)
{
    const std::string path = prim_path.toStdString();

    auto selectVisibleNode = [this, &path, &prim_path]() -> bool {
        for (QtNodes::NodeId nodeId : model_->allNodeIds()) {
            if (model_->primPathForNode(nodeId) != path) {
                continue;
            }

            QtNodes::NodeGraphicsObject *ngo = scene_->nodeGraphicsObject(nodeId);
            if (!ngo) {
                return false;
            }

            if (ngo->isSelected()) {
                emit nodeSelected(prim_path);
            } else {
                scene_->clearSelection();
                ngo->setSelected(true);
            }
            view_->centerOn(ngo);
            return true;
        }
        return false;
    };

    if (selectVisibleNode()) {
        return true;
    }

    for (const auto &[ng_path, ng_data] : current_graph_.node_graphs) {
        for (const ShaderNode &inner : ng_data.internal_nodes) {
            if (inner.prim_path == path) {
                expanded_node_graphs_.insert(ng_path);
                rebuildModel();
                return selectVisibleNode();
            }
        }
    }

    return false;
}

void NodeGraphWidget::saveNodePositions()
{
    for (QtNodes::NodeId nodeId : model_->allNodeIds()) {
        std::string path = model_->primPathForNode(nodeId);
        if (path.empty()) continue;
        QPointF pos = model_->nodeData(nodeId, QtNodes::NodeRole::Position).toPointF();
        saved_positions_[path] = pos;
    }
}

void NodeGraphWidget::restoreOrLayoutNodes(
    const std::vector<GraphConnection> &connections,
    const std::string &material_path)
{
    // First pass: restore all nodes that have a saved position,
    // collect those that need layout.
    std::vector<QtNodes::NodeId> needs_layout;
    std::unordered_set<std::string> needs_layout_paths;

    for (QtNodes::NodeId nodeId : model_->allNodeIds()) {
        std::string path = model_->primPathForNode(nodeId);
        auto it = saved_positions_.find(path);
        if (it != saved_positions_.end()) {
            model_->setNodeData(nodeId, QtNodes::NodeRole::Position,
                                QVariant(it->second));
        } else if (model_->hasAuthoredPosition(nodeId)) {
            QPointF pos = model_->nodeData(nodeId, QtNodes::NodeRole::Position).toPointF();
            saved_positions_[path] = pos;
        } else {
            needs_layout.push_back(nodeId);
            needs_layout_paths.insert(path);
        }
    }

    if (needs_layout.empty()) return;

    // Find an anchor point: the saved position of the collapsed NodeGraph
    // these nodes came from, or (0,0) for the initial layout.
    QPointF anchor(0.0, 0.0);
    for (const auto &[ng_path, _] : current_graph_.node_graphs) {
        // If this NG is expanded and its collapsed position was saved, use that
        if (expanded_node_graphs_.count(ng_path) && saved_positions_.count(ng_path)) {
            // Check if any of the needs_layout nodes belong to this NG
            bool relevant = false;
            for (QtNodes::NodeId nodeId : needs_layout) {
                std::string path = model_->primPathForNode(nodeId);
                if (path.rfind(ng_path + "/", 0) == 0) {
                    relevant = true;
                    break;
                }
            }
            if (relevant) {
                anchor = saved_positions_[ng_path];
                break;
            }
        }
    }

    // Filter connections to only those between needs_layout nodes
    std::vector<const GraphConnection *> relevant_connections;
    for (const GraphConnection &conn : connections) {
        if (needs_layout_paths.count(conn.source_node_path) ||
            needs_layout_paths.count(conn.dest_node_path)) {
            relevant_connections.push_back(&conn);
        }
    }

    // Compute depth among the needs_layout nodes only.
    // Find sink nodes (nodes with no outgoing connections to other layout nodes)
    std::unordered_set<std::string> has_outgoing;
    for (const GraphConnection *conn : relevant_connections) {
        if (needs_layout_paths.count(conn->source_node_path)) {
            has_outgoing.insert(conn->source_node_path);
        }
    }

    // Sinks get depth 0, sources get negative depth
    std::unordered_map<std::string, int> depth;
    for (const std::string &path : needs_layout_paths) {
        if (!has_outgoing.count(path)) {
            depth[path] = 0; // sink
        }
    }
    // If no sinks found, just pick the first node as depth 0
    if (depth.empty() && !needs_layout_paths.empty()) {
        depth[*needs_layout_paths.begin()] = 0;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (const GraphConnection *conn : relevant_connections) {
            auto dest_it = depth.find(conn->dest_node_path);
            if (dest_it == depth.end()) continue;
            int new_depth = dest_it->second - 1;
            auto src_it = depth.find(conn->source_node_path);
            if (src_it == depth.end() || new_depth < src_it->second) {
                depth[conn->source_node_path] = new_depth;
                changed = true;
            }
        }
    }

    // Get node sizes for spacing. Query after scene has computed them.
    // Use a reasonable default if size isn't available yet.
    auto getNodeHeight = [&](QtNodes::NodeId nodeId) -> double {
        QSize sz = model_->nodeData(nodeId, QtNodes::NodeRole::Size).toSize();
        return sz.height() > 0 ? sz.height() + 20.0 : 100.0;
    };

    // Layout: columns by depth, rows within each column spaced by node height
    double col_width = 350.0;
    std::unordered_map<int, double> depth_y_cursor; // current Y offset per column

    for (QtNodes::NodeId nodeId : needs_layout) {
        std::string path = model_->primPathForNode(nodeId);
        int d = 0;
        auto dit = depth.find(path);
        if (dit != depth.end()) {
            d = dit->second;
        } else {
            d = -2;
        }

        double &y_cursor = depth_y_cursor[d];
        double x = anchor.x() + d * col_width;
        double y = anchor.y() + y_cursor;
        y_cursor += getNodeHeight(nodeId);

        QPointF pos(x, y);
        model_->setNodeData(nodeId, QtNodes::NodeRole::Position, QVariant(pos));
        saved_positions_[path] = pos;
    }
}

void NodeGraphWidget::rebuildModel()
{
    // Save current positions before tearing down
    saveNodePositions();

    // Assemble a flat node + connection list based on expand state
    std::vector<ShaderNode> flat_nodes;
    std::vector<GraphConnection> flat_connections;
    std::unordered_set<std::string> expanded = expanded_node_graphs_;

    for (const ShaderNode &shader : current_graph_.shaders) {
        if (!shader.is_node_graph) {
            flat_nodes.push_back(shader);
        } else {
            auto ng_it = current_graph_.node_graphs.find(shader.prim_path);
            if (ng_it == current_graph_.node_graphs.end()) {
                flat_nodes.push_back(shader);
                continue;
            }

            const NodeGraphData &ng = ng_it->second;

            if (expanded.count(shader.prim_path)) {
                for (const ShaderNode &inner : ng.internal_nodes) {
                    flat_nodes.push_back(inner);
                }
                for (const GraphConnection &ic : ng.internal_connections) {
                    flat_connections.push_back(ic);
                }
            } else {
                flat_nodes.push_back(ng.collapsed_node);
            }
        }
    }

    // Add top-level connections, re-routing through expanded NodeGraphs
    for (const GraphConnection &conn : current_graph_.connections) {
        GraphConnection routed = conn;

        auto src_ng = current_graph_.node_graphs.find(conn.source_node_path);
        if (src_ng != current_graph_.node_graphs.end() &&
            expanded.count(conn.source_node_path)) {
            auto route_it = src_ng->second.output_routing.find(conn.source_output);
            if (route_it != src_ng->second.output_routing.end()) {
                routed.source_node_path = route_it->second.first;
                routed.source_output = route_it->second.second;
            }
        }

        auto dst_ng = current_graph_.node_graphs.find(conn.dest_node_path);
        if (dst_ng != current_graph_.node_graphs.end() &&
            expanded.count(conn.dest_node_path)) {
            auto route_it = dst_ng->second.input_routing.find(conn.dest_input);
            if (route_it != dst_ng->second.input_routing.end()) {
                routed.dest_node_path = route_it->second.first;
                routed.dest_input = route_it->second.second;
            }
        }

        flat_connections.push_back(routed);
    }

    model_->setGraph(flat_nodes, flat_connections, current_graph_.material_path);

    // Disable cache on all nodes (large node clipping fix)
    for (QtNodes::NodeId nodeId : model_->allNodeIds()) {
        QtNodes::NodeGraphicsObject *ngo = scene_->nodeGraphicsObject(nodeId);
        if (ngo) {
            ngo->setCacheMode(QGraphicsItem::NoCache);
        }
    }

    // Restore saved positions or auto-layout new nodes
    restoreOrLayoutNodes(flat_connections, current_graph_.material_path);

    // Wrap expanded NodeGraph internals in QtNodes visual groups
    group_to_ng_path_.clear();
    scene_->setGroupingEnabled(true);
    for (const std::string &ng_path : expanded) {
        auto ng_it = current_graph_.node_graphs.find(ng_path);
        if (ng_it == current_graph_.node_graphs.end()) continue;

        std::vector<NodeGraphicsObject *> group_nodes;
        for (const ShaderNode &inner : ng_it->second.internal_nodes) {
            for (QtNodes::NodeId nodeId : model_->allNodeIds()) {
                if (model_->primPathForNode(nodeId) == inner.prim_path) {
                    QtNodes::NodeGraphicsObject *ngo = scene_->nodeGraphicsObject(nodeId);
                    if (ngo) group_nodes.push_back(ngo);
                    break;
                }
            }
        }

        if (!group_nodes.empty()) {
            std::string name = ng_path.substr(ng_path.rfind('/') + 1);
            std::weak_ptr<QtNodes::NodeGroup> group_weak =
                scene_->createGroup(group_nodes, QString::fromStdString(name));
            if (auto group = group_weak.lock()) {
                group_to_ng_path_[group->id()] = ng_path;

                // Add a name label as child of the group so it moves with it
                QtNodes::GroupGraphicsObject &ggo = group->groupGraphicsObject();
                QGraphicsSimpleTextItem *label = new QGraphicsSimpleTextItem(
                    QString::fromStdString(name), &ggo);
                QFont font;
                font.setBold(true);
                font.setPointSize(10);
                label->setFont(font);
                label->setBrush(QColor(220, 160, 50));
                // Position at top-left of group's local rect, offset up
                QRectF groupRect = ggo.boundingRect();
                label->setPos(groupRect.left() + 8,
                              groupRect.top() - label->boundingRect().height() - 4);
            }
        }
    }

    view_->centerScene();
}

void NodeGraphWidget::showContextMenu(const QPoint &pos)
{
    QPointF scene_pos = view_->mapToScene(pos);
    QGraphicsItem *item = scene_->itemAt(scene_pos, view_->transform());
    if (!item) return;

    // Check if we clicked on a GroupGraphicsObject (for Collapse)
    QGraphicsItem *walk = item;
    QtNodes::GroupGraphicsObject *ggo = nullptr;
    while (walk) {
        ggo = dynamic_cast<QtNodes::GroupGraphicsObject *>(walk);
        if (ggo) break;
        walk = walk->parentItem();
    }

    if (ggo) {
        // Find which group this belongs to
        for (const auto &[group_id, group_ptr] : scene_->groups()) {
            if (&group_ptr->groupGraphicsObject() == ggo) {
                auto ng_it = group_to_ng_path_.find(group_id);
                if (ng_it != group_to_ng_path_.end()) {
                    QMenu menu;
                    std::string ng_name = ng_it->second.substr(
                        ng_it->second.rfind('/') + 1);
                    QAction *collapse_action = menu.addAction(
                        QString("Collapse %1").arg(
                            QString::fromStdString(ng_name)));
                    if (menu.exec(view_->mapToGlobal(pos)) == collapse_action) {
                        expanded_node_graphs_.erase(ng_it->second);
                        rebuildModel();
                    }
                }
                return;
            }
        }
        return;
    }

    // Check if we clicked on a NodeGraphicsObject (for Expand)
    walk = item;
    QtNodes::NodeGraphicsObject *ngo = nullptr;
    while (walk) {
        ngo = dynamic_cast<QtNodes::NodeGraphicsObject *>(walk);
        if (ngo) break;
        walk = walk->parentItem();
    }

    if (!ngo) return;

    // Find the node ID
    QtNodes::NodeId clicked_id = QtNodes::InvalidNodeId;
    for (QtNodes::NodeId nodeId : model_->allNodeIds()) {
        if (scene_->nodeGraphicsObject(nodeId) == ngo) {
            clicked_id = nodeId;
            break;
        }
    }
    if (clicked_id == QtNodes::InvalidNodeId) return;

    std::string prim_path = model_->primPathForNode(clicked_id);

    // Check if it's a collapsed NodeGraph — offer Expand
    if (model_->isNodeGraph(clicked_id) &&
        current_graph_.node_graphs.count(prim_path)) {
        QMenu menu;
        QAction *expand_action = menu.addAction("Expand");
        if (menu.exec(view_->mapToGlobal(pos)) == expand_action) {
            expanded_node_graphs_.insert(prim_path);
            rebuildModel();
        }
    }
}
