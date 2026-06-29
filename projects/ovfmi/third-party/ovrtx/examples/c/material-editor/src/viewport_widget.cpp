// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "viewport_widget.hpp"

#include <QPainter>

ViewportWidget::ViewportWidget(QWidget *parent)
    : QWidget(parent)
{
    setMinimumSize(256, 256);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void ViewportWidget::setImage(QImage image)
{
    image_ = std::move(image);
    update();
}

void ViewportWidget::paintEvent(QPaintEvent * /*event*/)
{
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    if (image_.isNull()) {
        return;
    }

    // Scale image to fit widget, preserving aspect ratio
    QSize scaled = image_.size().scaled(size(), Qt::KeepAspectRatio);
    int x = (width() - scaled.width()) / 2;
    int y = (height() - scaled.height()) / 2;
    painter.drawImage(QRect(x, y, scaled.width(), scaled.height()), image_);
}
