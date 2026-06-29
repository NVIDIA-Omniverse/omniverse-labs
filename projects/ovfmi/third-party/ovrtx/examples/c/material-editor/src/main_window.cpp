// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "main_window.hpp"
#include "ovrtx_engine.hpp"
#include "viewport_widget.hpp"
#include "material_list_widget.hpp"
#include "node_graph_widget.hpp"
#include "property_panel_widget.hpp"

#include <QDockWidget>
#include <QSplitter>
#include <QMessageBox>
#include <QApplication>
#include <QSignalBlocker>

#include <array>
#include <cstddef>
#include <iostream>

MainWindow::MainWindow(const std::string &usda_path, QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("ovrtx Material Editor");

    // --- Create widgets ---
    engine_ = new OvrtxEngine(this);
    viewport_ = new ViewportWidget();
    material_list_ = new MaterialListWidget();
    node_graph_ = new NodeGraphWidget();
    property_panel_ = new PropertyPanelWidget();

    // --- Layout ---
    // Central: vertical splitter with viewport on top, node graph on bottom
    QSplitter *central = new QSplitter(Qt::Vertical);
    central->addWidget(viewport_);
    central->addWidget(node_graph_);
    central->setStretchFactor(0, 2);
    central->setStretchFactor(1, 1);
    setCentralWidget(central);

    // Left dock: material list
    QDockWidget *left_dock = new QDockWidget("Materials", this);
    left_dock->setWidget(material_list_);
    left_dock->setFeatures(QDockWidget::DockWidgetMovable |
                           QDockWidget::DockWidgetFloatable);
    addDockWidget(Qt::LeftDockWidgetArea, left_dock);

    // Right dock: property panel
    QDockWidget *right_dock = new QDockWidget("Properties", this);
    right_dock->setWidget(property_panel_);
    right_dock->setFeatures(QDockWidget::DockWidgetMovable |
                            QDockWidget::DockWidgetFloatable);
    addDockWidget(Qt::RightDockWidgetArea, right_dock);

    // --- Wire signals ---
    connect(material_list_, &MaterialListWidget::materialSelected,
            this, &MainWindow::onMaterialSelected);

    connect(node_graph_, &NodeGraphWidget::nodeSelected,
            this, &MainWindow::onNodeSelected);

    connect(property_panel_, &PropertyPanelWidget::attributeChanged,
            this, &MainWindow::onAttributeChanged);

    connect(property_panel_, &PropertyPanelWidget::connectedInputClicked,
            this, &MainWindow::onConnectedInputClicked);

    connect(engine_, &OvrtxEngine::frameReady,
            viewport_, &ViewportWidget::setImage);

    // --- Initialize renderer (must happen before USD introspection so that
    //     ovrtx's plugins and asset resolution are available) ---
    if (!engine_->initialize(usda_path)) {
        QMessageBox::critical(this, "Error",
                              "Failed to initialize ovrtx renderer.");
        return;
    }

    // --- Load material graphs from USD via OpenUSD C++ API ---
    material_graphs_ = loadMaterialGraphs(usda_path);
    for (size_t i = 0; i < material_graphs_.size(); ++i) {
        material_index_[material_graphs_[i].material_path] = i;
    }

    if (material_graphs_.empty()) {
        QMessageBox::warning(this, "Warning",
                             "No materials found in the USD stage.");
    }

    material_list_->setMaterials(material_graphs_);

    // Render initial frame
    engine_->startRender();

    // Auto-select the material already bound in the USD without rebinding it.
    if (!material_graphs_.empty()) {
        std::string bound_material = getBoundMaterialPath(usda_path, kTargetPrim);
        auto selected_it = material_index_.find(bound_material);

        size_t selected_index = 0;
        bool found_bound_material = selected_it != material_index_.end();
        if (found_bound_material) {
            selected_index = selected_it->second;
        }

        {
            QSignalBlocker blocker(material_list_);
            material_list_->setCurrentRow(static_cast<int>(selected_index));
        }
        node_graph_->setMaterialGraph(material_graphs_[selected_index]);
        material_list_->setActiveMaterial(
            found_bound_material ? QString::fromStdString(bound_material)
                                 : QString());
    }
}

MainWindow::~MainWindow()
{
    engine_->shutdown();
}

void MainWindow::onMaterialSelected(QString material_path)
{
    std::string path = material_path.toStdString();

    // Bind the material to the target prim
    bool changed = engine_->bindMaterial(kTargetPrim, path);
    if (changed) {
        material_list_->setActiveMaterial(material_path);
    }

    // Update node graph to show this material's shader graph
    auto it = material_index_.find(path);
    if (it != material_index_.end()) {
        node_graph_->setMaterialGraph(material_graphs_[it->second]);
    }

    // Clear property panel (no node selected yet)
    property_panel_->clear();

    // Re-render
    if (changed) {
        engine_->startRender();
    }
}

void MainWindow::onNodeSelected(QString prim_path)
{
    std::string path = prim_path.toStdString();

    // Find the shader node in the current material graphs
    for (const MaterialGraph &graph : material_graphs_) {
        for (const ShaderNode &shader : graph.shaders) {
            if (shader.prim_path == path) {
                property_panel_->showNodeProperties(shader);
                return;
            }
        }
        for (const auto &[node_graph_path, node_graph] : graph.node_graphs) {
            Q_UNUSED(node_graph_path);
            for (const ShaderNode &shader : node_graph.internal_nodes) {
                if (shader.prim_path == path) {
                    property_panel_->showNodeProperties(shader);
                    return;
                }
            }
        }
    }

    // If not found as a shader, it might be the material node itself
    property_panel_->clear();
}

void MainWindow::onAttributeChanged(QString prim_path, QString attr_name,
                                    QString type_name, QVariant value)
{
    std::string prim = prim_path.toStdString();
    std::string attr = attr_name.toStdString();

    bool changed = false;
    if (type_name == "float") {
        changed = engine_->writeFloatAttribute(prim, attr, value.toFloat());
    } else if (type_name == "color3f") {
        QVariantList rgb = value.toList();
        if (rgb.size() == 3) {
            changed = engine_->writeColor3fAttribute(prim, attr,
                                                     rgb[0].toFloat(),
                                                     rgb[1].toFloat(),
                                                     rgb[2].toFloat());
        }
    } else if (type_name == "int") {
        changed = engine_->writeIntAttribute(prim, attr, value.toInt());
    } else if (type_name == "bool") {
        changed = engine_->writeBoolAttribute(prim, attr, value.toBool());
    } else if (type_name == "token") {
        changed = engine_->writeTokenAttribute(prim, attr, value.toString().toStdString());
    }

    // Re-render to show the change
    if (changed) {
        updateCachedAttributeValue(prim, attr, type_name, value);
        engine_->startRender();
    }
}

void MainWindow::onConnectedInputClicked(QString source_prim_path)
{
    if (node_graph_->selectNodeByPrimPath(source_prim_path)) {
        return;
    }

    // Fall back to updating the property panel if the source exists in the
    // introspection data but is not represented as a visible graph node.
    onNodeSelected(source_prim_path);
}

bool MainWindow::updateCachedAttributeValue(const std::string &prim_path,
                                            const std::string &attr_name,
                                            const QString &type_name,
                                            const QVariant &value)
{
    static constexpr const char *kInputsPrefix = "inputs:";
    static constexpr std::size_t kInputsPrefixLength = 7;
    if (attr_name.rfind(kInputsPrefix, 0) != 0) {
        return false;
    }

    const std::string input_name = attr_name.substr(kInputsPrefixLength);
    bool updated = false;

    for (MaterialGraph &graph : material_graphs_) {
        for (ShaderNode &shader : graph.shaders) {
            updated = updateCachedShaderInput(shader, prim_path, input_name,
                                              type_name, value) || updated;
        }
        for (auto &[node_graph_path, node_graph] : graph.node_graphs) {
            Q_UNUSED(node_graph_path);
            updated = updateCachedShaderInput(node_graph.collapsed_node,
                                              prim_path, input_name,
                                              type_name, value) || updated;
            for (ShaderNode &shader : node_graph.internal_nodes) {
                updated = updateCachedShaderInput(shader, prim_path, input_name,
                                                  type_name, value) || updated;
            }
        }
    }

    return updated;
}

bool MainWindow::updateCachedShaderInput(ShaderNode &shader,
                                         const std::string &prim_path,
                                         const std::string &input_name,
                                         const QString &type_name,
                                         const QVariant &value)
{
    if (shader.prim_path != prim_path) {
        return false;
    }

    for (ShaderInput &input : shader.inputs) {
        if (input.name != input_name) {
            continue;
        }

        if (type_name == "float") {
            input.value = value.toFloat();
        } else if (type_name == "color3f") {
            QVariantList rgb = value.toList();
            if (rgb.size() != 3) {
                return false;
            }
            input.value = std::array<float, 3>{
                rgb[0].toFloat(),
                rgb[1].toFloat(),
                rgb[2].toFloat()};
        } else if (type_name == "int") {
            input.value = value.toInt();
        } else if (type_name == "bool") {
            input.value = value.toBool();
        } else if (type_name == "token") {
            input.value = value.toString().toStdString();
        } else {
            return false;
        }

        input.is_authored = true;
        return true;
    }

    return false;
}
