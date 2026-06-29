/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: LicenseRef-NvidiaProprietary */

/* Open AOV table images in a new tab when clicked. */
document.querySelectorAll(".aov-table a.image-reference, .compact-table a.image-reference").forEach(function (a) {
  a.setAttribute("target", "_blank");
});
