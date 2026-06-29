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

#include <QWidget>
#include <QImage>

/// Widget that displays a QImage rendered by ovrtx, scaled to fit with
/// preserved aspect ratio.
class ViewportWidget : public QWidget {
    Q_OBJECT
public:
    explicit ViewportWidget(QWidget *parent = nullptr);

public slots:
    void setImage(QImage image);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QImage image_;
};
