// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "material_list_widget.hpp"
#include "usd_material_graph.hpp"

#include <QFont>

MaterialListWidget::MaterialListWidget(QWidget *parent)
    : QListWidget(parent)
{
    connect(this, &QListWidget::currentItemChanged,
            this, &MaterialListWidget::onCurrentItemChanged);
}

void MaterialListWidget::setMaterials(const std::vector<MaterialGraph> &materials)
{
    clear();
    for (const MaterialGraph &mat : materials) {
        QListWidgetItem *item = new QListWidgetItem(
            QString::fromStdString(mat.display_name));
        item->setData(Qt::UserRole,
                      QString::fromStdString(mat.material_path));
        addItem(item);
    }
}

void MaterialListWidget::setActiveMaterial(const QString &material_path)
{
    active_material_path_ = material_path;

    for (int i = 0; i < count(); ++i) {
        QListWidgetItem *item_ptr = item(i);
        QString path = item_ptr->data(Qt::UserRole).toString();
        QFont font = item_ptr->font();
        if (path == material_path) {
            font.setBold(true);
            item_ptr->setFont(font);
        } else {
            font.setBold(false);
            item_ptr->setFont(font);
        }
    }
}

void MaterialListWidget::onCurrentItemChanged(QListWidgetItem *current,
                                              QListWidgetItem * /*previous*/)
{
    if (!current) {
        return;
    }
    QString path = current->data(Qt::UserRole).toString();
    emit materialSelected(path);
}
