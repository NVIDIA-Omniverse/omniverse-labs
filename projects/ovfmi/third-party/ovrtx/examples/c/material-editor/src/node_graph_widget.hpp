// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include "usd_material_graph.hpp"

#include <QtNodes/AbstractGraphModel>
#include <QtNodes/BasicGraphicsScene>
#include <QtNodes/GraphicsView>
#include <QtNodes/Definitions>

#include <QWidget>
#include <QVBoxLayout>
#include <QString>
#include <QSize>

#include <unordered_map>
#include <unordered_set>
#include <vector>

/// Graph model that represents a UsdShade material graph for QtNodes.
/// Each Material and Shader prim becomes a node; connections map to edges.
class ShaderGraphModel : public QtNodes::AbstractGraphModel {
    Q_OBJECT
public:
    ShaderGraphModel();

    /// Rebuild the graph from a flattened node/connection list.
    void setGraph(const std::vector<ShaderNode> &nodes,
                  const std::vector<GraphConnection> &connections,
                  const std::string &material_path);

    // -- AbstractGraphModel interface --
    std::unordered_set<QtNodes::NodeId> allNodeIds() const override;
    std::unordered_set<QtNodes::ConnectionId> allConnectionIds(QtNodes::NodeId nodeId) const override;
    std::unordered_set<QtNodes::ConnectionId> connections(QtNodes::NodeId nodeId,
                                                          QtNodes::PortType portType,
                                                          QtNodes::PortIndex portIndex) const override;
    bool connectionExists(QtNodes::ConnectionId connectionId) const override;
    QtNodes::NodeId newNodeId() override;
    QtNodes::NodeId addNode(QString const nodeType = QString()) override;
    bool connectionPossible(QtNodes::ConnectionId connectionId) const override;
    void addConnection(QtNodes::ConnectionId connectionId) override;
    bool nodeExists(QtNodes::NodeId nodeId) const override;
    QVariant nodeData(QtNodes::NodeId nodeId, QtNodes::NodeRole role) const override;
    bool setNodeData(QtNodes::NodeId nodeId, QtNodes::NodeRole role, QVariant value) override;
    QVariant portData(QtNodes::NodeId nodeId,
                      QtNodes::PortType portType,
                      QtNodes::PortIndex portIndex,
                      QtNodes::PortRole role) const override;
    bool setPortData(QtNodes::NodeId nodeId,
                     QtNodes::PortType portType,
                     QtNodes::PortIndex portIndex,
                     QVariant const &value,
                     QtNodes::PortRole role = QtNodes::PortRole::Data) override;
    bool deleteConnection(QtNodes::ConnectionId connectionId) override;
    bool deleteNode(QtNodes::NodeId nodeId) override;

    /// Get the prim path for a given node ID, or empty string.
    std::string primPathForNode(QtNodes::NodeId nodeId) const;

    /// Returns true if the node is a collapsed NodeGraph.
    bool isNodeGraph(QtNodes::NodeId nodeId) const;

    /// Returns true if this node has an authored USD node-graph position.
    bool hasAuthoredPosition(QtNodes::NodeId nodeId) const;

    /// Returns the prim paths of internal nodes for a given NodeGraph node.
    std::vector<QtNodes::NodeId> internalNodeIds(const std::string &ng_path) const;

private:
    struct NodeInfo {
        QtNodes::NodeId id;
        std::string prim_path;
        QString caption;
        QString subcaption;
        std::vector<QString> in_port_names;
        std::vector<QString> out_port_names;
        QPointF position;
        QSize size;
        bool is_node_graph = false;
        bool has_authored_position = false;
        std::string parent_node_graph; // prim path of containing NodeGraph (empty if top-level)
    };

    std::unordered_map<QtNodes::NodeId, NodeInfo> nodes_;
    std::unordered_set<QtNodes::ConnectionId> connections_;
    QtNodes::NodeId next_id_ = 0;
};

// ---------------------------------------------------------------------------

/// Widget wrapping a QtNodes GraphicsView that displays a shader graph.
/// Supports expanding/collapsing NodeGraph prims via right-click menu.
class NodeGraphWidget : public QWidget {
    Q_OBJECT
public:
    explicit NodeGraphWidget(QWidget *parent = nullptr);
    ~NodeGraphWidget() override;

    /// Display the shader graph for the given material.
    void setMaterialGraph(const MaterialGraph &graph);

    /// Select and center a visible node by prim path. Expands a containing
    /// NodeGraph when needed to make an internal shader visible.
    bool selectNodeByPrimPath(QString prim_path);

signals:
    /// Emitted when the user selects a node. Carries the prim path.
    void nodeSelected(QString prim_path);

private:
    void rebuildModel();
    void saveNodePositions();
    void restoreOrLayoutNodes(const std::vector<GraphConnection> &connections,
                              const std::string &material_path);
    void showContextMenu(const QPoint &pos);

    ShaderGraphModel *model_ = nullptr;
    QtNodes::BasicGraphicsScene *scene_ = nullptr;
    QtNodes::GraphicsView *view_ = nullptr;
    QVBoxLayout *layout_ = nullptr;

    MaterialGraph current_graph_;
    std::unordered_set<std::string> expanded_node_graphs_;
    /// Saved node positions keyed by prim path (persists across rebuilds).
    std::unordered_map<std::string, QPointF> saved_positions_;
    /// NodeGraphs that have been expanded at least once (for initial layout).
    std::unordered_set<std::string> ever_expanded_;
    /// Maps QtNodes GroupId → NodeGraph prim path (for collapse on right-click).
    std::unordered_map<unsigned int, std::string> group_to_ng_path_;
    QMetaObject::Connection selection_connection_;
    bool shutting_down_ = false;
};
