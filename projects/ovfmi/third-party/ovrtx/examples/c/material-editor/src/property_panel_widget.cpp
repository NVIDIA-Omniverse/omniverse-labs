// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "property_panel_widget.hpp"

#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>

PropertyPanelWidget::PropertyPanelWidget(QWidget *parent)
    : QScrollArea(parent)
{
    setWidgetResizable(true);
    setMinimumWidth(500);

    container_ = new QWidget();
    container_layout_ = new QVBoxLayout(container_);
    container_layout_->setContentsMargins(0, 0, 0, 0);
    container_layout_->setSpacing(0);
    setWidget(container_);
}

void PropertyPanelWidget::clear()
{
    // Delete all children of the container
    QLayoutItem *item;
    while ((item = container_layout_->takeAt(0)) != nullptr) {
        delete item->widget();
        delete item;
    }
    current_prim_path_.clear();
}

void PropertyPanelWidget::showNodeProperties(const ShaderNode &node)
{
    clear();
    current_prim_path_ = node.prim_path;

    // Header: node name and shader type
    QLabel *header = new QLabel(
        QString("<b>%1</b><br><i>%2</i>")
            .arg(QString::fromStdString(node.display_name))
            .arg(QString::fromStdString(
                node.mdl_sub_id.empty() ? node.mdl_asset : node.mdl_sub_id)));
    header->setContentsMargins(6, 6, 6, 6);
    container_layout_->addWidget(header);

    if (node.inputs.empty()) {
        QLabel *empty_label = new QLabel("(no inputs)");
        empty_label->setContentsMargins(6, 6, 6, 6);
        container_layout_->addWidget(empty_label);
        container_layout_->addStretch();
        return;
    }

    // Group inputs by page
    // Use a vector of pairs to preserve insertion order of pages
    std::vector<std::pair<std::string, std::vector<const ShaderInput *>>> pages;
    std::map<std::string, size_t> page_index;

    for (const ShaderInput &input : node.inputs) {
        std::string page = input.page.empty() ? "Other" : input.page;
        auto it = page_index.find(page);
        if (it == page_index.end()) {
            page_index[page] = pages.size();
            pages.push_back({page, {}});
            it = page_index.find(page);
        }
        pages[it->second].second.push_back(&input);
    }

    // Create collapsible sections for each page
    for (const auto &[page_name, inputs] : pages) {
        // --- Accordion header ---
        QToolButton *toggle_btn = new QToolButton();
        toggle_btn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        toggle_btn->setArrowType(Qt::DownArrow);
        toggle_btn->setText(QString::fromStdString(page_name));
        toggle_btn->setCheckable(true);
        toggle_btn->setChecked(true);
        toggle_btn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        toggle_btn->setStyleSheet(
            "QToolButton { font-weight: bold; background: #3a3a3a; color: #ddd;"
            "  border: none; padding: 4px 8px; text-align: left; }"
            "QToolButton:checked { background: #444; }");
        container_layout_->addWidget(toggle_btn);

        // --- Accordion body ---
        QWidget *body = new QWidget();
        QFormLayout *form = new QFormLayout(body);
        form->setContentsMargins(8, 4, 4, 4);
        form->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
        form->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
        container_layout_->addWidget(body);

        // Toggle visibility
        connect(toggle_btn, &QToolButton::toggled, body, [body, toggle_btn](bool checked) {
            body->setVisible(checked);
            toggle_btn->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        });

        // Add rows
        for (const ShaderInput *input : inputs) {
            addRow(form, *input);
        }
    }

    container_layout_->addStretch();
}

QWidget *PropertyPanelWidget::makeLabel(const ShaderInput &input)
{
    // Use Sdr label if available, otherwise the property name
    std::string display_text = input.label.empty() ? input.name : input.label;

    QLabel *label = new QLabel(QString::fromStdString(display_text));
    QString tooltip = QString("inputs:%1").arg(QString::fromStdString(input.name));
    if (input.is_connected) {
        tooltip += QString("\nConnected to %1:%2")
                       .arg(QString::fromStdString(input.connected_source_node_path))
                       .arg(QString::fromStdString(input.connected_source_output));
    }
    label->setToolTip(tooltip);

    if (input.is_authored) {
        label->setStyleSheet("QLabel { color: #6cb4ee; }"); // light blue
    }

    return label;
}

void PropertyPanelWidget::addRow(QFormLayout *layout, const ShaderInput &input)
{
    QWidget *label_widget = makeLabel(input);
    const std::string &type = input.type_name;

    if (input.is_connected) {
        QToolButton *button = new QToolButton();
        button->setArrowType(Qt::LeftArrow);
        button->setToolButtonStyle(Qt::ToolButtonIconOnly);
        button->setToolTip(
            QString("Go to %1:%2")
                .arg(QString::fromStdString(input.connected_source_node_path))
                .arg(QString::fromStdString(input.connected_source_output)));
        button->setAccessibleName("Go to connected source");
        button->setFixedSize(28, 24);
        button->setStyleSheet(
            "QToolButton { border: 1px solid #6cb4ee; border-radius: 3px; }"
            "QToolButton:hover { background: #3f5f7a; }");

        std::string source_path = input.connected_source_node_path;
        connect(button, &QToolButton::clicked, this,
                [this, source_path]() {
                    emit connectedInputClicked(
                        QString::fromStdString(source_path));
                });
        layout->addRow(label_widget, button);
        return;
    }

    // If the property has enum options, use a dropdown regardless of type
    if (!input.options.empty()) {
        QComboBox *combo = new QComboBox();
        for (const std::string &opt : input.options) {
            combo->addItem(QString::fromStdString(opt));
        }

        // Set current index from the int value
        int current = 0;
        if (std::holds_alternative<int>(input.value)) {
            current = std::get<int>(input.value);
        }
        if (current >= 0 && current < combo->count()) {
            combo->setCurrentIndex(current);
        }

        std::string attr_name = "inputs:" + input.name;
        connect(combo, &QComboBox::currentIndexChanged, this,
                [this, attr_name](int index) {
                    emit attributeChanged(
                        QString::fromStdString(current_prim_path_),
                        QString::fromStdString(attr_name),
                        "int", QVariant(index));
                });
        layout->addRow(label_widget, combo);
        return;
    }

    if (type == "float") {
        float val = std::get<float>(input.value);
        QDoubleSpinBox *spin = new QDoubleSpinBox();
        spin->setDecimals(4);
        spin->setSingleStep(0.01);

        if (input.has_range) {
            spin->setRange(static_cast<double>(input.range_min),
                           static_cast<double>(input.range_max));
        } else {
            spin->setRange(-1e6, 1e6);
        }
        spin->setValue(static_cast<double>(val));

        std::string attr_name = "inputs:" + input.name;
        connect(spin, &QDoubleSpinBox::valueChanged, this,
                [this, attr_name](double v) {
                    emit attributeChanged(
                        QString::fromStdString(current_prim_path_),
                        QString::fromStdString(attr_name),
                        "float", QVariant(static_cast<float>(v)));
                });

        if (input.has_range) {
            // Slider + spinbox side by side
            static constexpr int kSliderSteps = 1000;
            QSlider *slider = new QSlider(Qt::Horizontal);
            slider->setRange(0, kSliderSteps);
            double range_span = static_cast<double>(input.range_max - input.range_min);
            auto valToSlider = [=](double v) -> int {
                double t = (v - input.range_min) / range_span;
                return static_cast<int>(t * kSliderSteps + 0.5);
            };
            auto sliderToVal = [=](int s) -> double {
                return input.range_min + (static_cast<double>(s) / kSliderSteps) * range_span;
            };
            slider->setValue(valToSlider(val));

            // Slider → spinbox + emit attribute change
            connect(slider, &QSlider::valueChanged, this, [this, spin, sliderToVal, attr_name](int s) {
                double v = sliderToVal(s);
                QSignalBlocker blocker(spin);
                spin->setValue(v);
                emit attributeChanged(
                    QString::fromStdString(current_prim_path_),
                    QString::fromStdString(attr_name),
                    "float", QVariant(static_cast<float>(v)));
            });
            // Spinbox → slider
            connect(spin, &QDoubleSpinBox::valueChanged, slider, [slider, valToSlider](double v) {
                QSignalBlocker blocker(slider);
                slider->setValue(valToSlider(v));
            });

            QWidget *row = new QWidget();
            QHBoxLayout *hlayout = new QHBoxLayout(row);
            hlayout->setContentsMargins(0, 0, 0, 0);
            hlayout->addWidget(spin, 1);
            hlayout->addWidget(slider, 2);
            layout->addRow(label_widget, row);
        } else {
            layout->addRow(label_widget, spin);
        }

    } else if (type == "color3f" || type == "float3") {
        std::array<float, 3> val = std::get<std::array<float, 3>>(input.value);
        QPushButton *btn = new QPushButton();
        QColor initial = QColor::fromRgbF(
            static_cast<double>(val[0]),
            static_cast<double>(val[1]),
            static_cast<double>(val[2]));
        btn->setStyleSheet(
            QString("background-color: %1; border: 1px solid #888; min-height: 24px;")
                .arg(initial.name()));

        std::string attr_name = "inputs:" + input.name;
        connect(btn, &QPushButton::clicked, this,
                [this, btn, attr_name, initial]() mutable {
                    QColor color = QColorDialog::getColor(initial, this, "Select Color");
                    if (!color.isValid()) return;
                    initial = color;
                    btn->setStyleSheet(
                        QString("background-color: %1; border: 1px solid #888; min-height: 24px;")
                            .arg(color.name()));
                    QVariantList rgb;
                    rgb << static_cast<float>(color.redF())
                        << static_cast<float>(color.greenF())
                        << static_cast<float>(color.blueF());
                    emit attributeChanged(
                        QString::fromStdString(current_prim_path_),
                        QString::fromStdString(attr_name),
                        "color3f", QVariant(rgb));
                });
        layout->addRow(label_widget, btn);

    } else if (type == "bool") {
        bool val = std::get<bool>(input.value);
        QCheckBox *cb = new QCheckBox();
        cb->setChecked(val);

        std::string attr_name = "inputs:" + input.name;
        connect(cb, &QCheckBox::toggled, this,
                [this, attr_name](bool checked) {
                    emit attributeChanged(
                        QString::fromStdString(current_prim_path_),
                        QString::fromStdString(attr_name),
                        "bool", QVariant(checked));
                });
        layout->addRow(label_widget, cb);

    } else if (type == "int") {
        int val = std::get<int>(input.value);
        QSpinBox *spin = new QSpinBox();
        spin->setRange(-1000000, 1000000);
        spin->setValue(val);

        std::string attr_name = "inputs:" + input.name;
        connect(spin, &QSpinBox::valueChanged, this,
                [this, attr_name](int v) {
                    emit attributeChanged(
                        QString::fromStdString(current_prim_path_),
                        QString::fromStdString(attr_name),
                        "int", QVariant(v));
                });
        layout->addRow(label_widget, spin);

    } else if (type == "asset") {
        std::string val = std::get<std::string>(input.value);
        QLineEdit *edit = new QLineEdit(QString::fromStdString(val));
        edit->setReadOnly(true);
        layout->addRow(label_widget, edit);

    } else {
        // token, string, or unknown
        std::string val = std::get<std::string>(input.value);
        QLineEdit *edit = new QLineEdit(QString::fromStdString(val));

        std::string attr_name = "inputs:" + input.name;
        connect(edit, &QLineEdit::editingFinished, this,
                [this, edit, attr_name]() {
                    emit attributeChanged(
                        QString::fromStdString(current_prim_path_),
                        QString::fromStdString(attr_name),
                        "token", QVariant(edit->text()));
                });
        layout->addRow(label_widget, edit);
    }
}
