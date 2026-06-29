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

#include <QListWidget>
#include <QString>
#include <vector>

struct MaterialGraph;

/// Sidebar list showing all materials discovered in the USD stage.
/// Emits materialSelected when the user picks a material.
class MaterialListWidget : public QListWidget {
    Q_OBJECT
public:
    explicit MaterialListWidget(QWidget *parent = nullptr);

    /// Populate the list from parsed material graphs.
    void setMaterials(const std::vector<MaterialGraph> &materials);

    /// Highlight the material that is currently bound to the target prim.
    void setActiveMaterial(const QString &material_path);

signals:
    /// Emitted when the user selects a material.  Carries the full prim path.
    void materialSelected(QString material_path);

private slots:
    void onCurrentItemChanged(QListWidgetItem *current, QListWidgetItem *previous);

private:
    QString active_material_path_;
};
