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

#include <QMainWindow>
#include <QVariant>
#include <string>
#include <unordered_map>
#include <vector>

class OvrtxEngine;
class ViewportWidget;
class MaterialListWidget;
class NodeGraphWidget;
class PropertyPanelWidget;

/// Main application window: owns the four-panel layout and wires signals
/// between the material list, viewport, node graph, and property panel.
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(const std::string &usda_path, QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void onMaterialSelected(QString material_path);
    void onNodeSelected(QString prim_path);
    void onAttributeChanged(QString prim_path, QString attr_name,
                            QString type_name, QVariant value);
    void onConnectedInputClicked(QString source_prim_path);

private:
    bool updateCachedAttributeValue(const std::string &prim_path,
                                    const std::string &attr_name,
                                    const QString &type_name,
                                    const QVariant &value);
    bool updateCachedShaderInput(ShaderNode &shader,
                                 const std::string &prim_path,
                                 const std::string &input_name,
                                 const QString &type_name,
                                 const QVariant &value);

    /// The hard-coded mesh prim whose material binding we manipulate.
    static constexpr const char *kTargetPrim = "/World/Sphere";

    OvrtxEngine *engine_ = nullptr;
    ViewportWidget *viewport_ = nullptr;
    MaterialListWidget *material_list_ = nullptr;
    NodeGraphWidget *node_graph_ = nullptr;
    PropertyPanelWidget *property_panel_ = nullptr;

    std::vector<MaterialGraph> material_graphs_;
    /// Lookup: material_path -> index into material_graphs_
    std::unordered_map<std::string, size_t> material_index_;
};
