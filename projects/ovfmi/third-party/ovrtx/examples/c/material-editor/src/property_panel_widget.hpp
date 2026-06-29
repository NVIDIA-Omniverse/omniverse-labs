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

#include <QScrollArea>
#include <QWidget>
#include <QFormLayout>
#include <QVBoxLayout>
#include <QString>
#include <QVariant>

#include <map>
#include <string>
#include <vector>

/// Panel that shows the editable attributes of a selected shader node.
/// Inputs are grouped by Sdr page into collapsible accordion sections.
/// Authored properties have light blue labels; others show Sdr defaults.
class PropertyPanelWidget : public QScrollArea {
    Q_OBJECT
public:
    explicit PropertyPanelWidget(QWidget *parent = nullptr);

    /// Show properties for the given shader node.
    void showNodeProperties(const ShaderNode &node);

    /// Clear the panel.
    void clear();

signals:
    /// Emitted when the user changes an attribute value.
    void attributeChanged(QString prim_path, QString attr_name,
                          QString type_name, QVariant value);

    /// Emitted when the user clicks a connected input's source navigation button.
    void connectedInputClicked(QString source_prim_path);

private:
    QWidget *container_ = nullptr;
    QVBoxLayout *container_layout_ = nullptr;
    std::string current_prim_path_;

    /// Create a label widget with the right text, color, and tooltip.
    QWidget *makeLabel(const ShaderInput &input);

    /// Add one input row to the given form layout.
    void addRow(QFormLayout *layout, const ShaderInput &input);
};
