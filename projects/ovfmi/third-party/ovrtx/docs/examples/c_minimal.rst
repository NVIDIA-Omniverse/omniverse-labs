.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
..
.. NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
.. property and proprietary rights in and to this material, related
.. documentation and any modifications thereto. Any use, reproduction,
.. disclosure or distribution of this material and related documentation
.. without an express license agreement from NVIDIA CORPORATION or
.. its affiliates is strictly prohibited.

C: Minimal Example
==================

This example shows basic initialization of the renderer, rendering a single frame from an RGB camera, mapping the output and writing the result to disk as a PNG.

The example loads an example scene from S3 and writes the resulting image to ``out.png``.

The first time you run the example, the driver compiles and caches shaders. Subsequent runs are much faster.

.. pull-quote::

   *“Create the smallest useful C/C++ example that initializes ovrtx, loads a USD scene asynchronously, waits for it, renders one camera frame, fetches and CPU-maps the color output, writes it to an image file, and releases all ovrtx resources explicitly.”*

.. image:: ../../img/out.jpg
   :alt: Minimal example output
   :align: center

Build and Run
-------------

.. tab-set::

   .. tab-item:: Linux

      **Prerequisites**

      .. code-block:: bash

         sudo apt install build-essential cmake

      The ovrtx library is downloaded automatically at configure time. If ovrtx is already installed and available via ``CMAKE_PREFIX_PATH``, the local installation is used instead.

      **Building**

      .. code-block:: bash

         cmake -B build -DCMAKE_BUILD_TYPE=Release
         cmake --build build

      **Running**

      .. code-block:: bash

         ./build/minimal

   .. tab-item:: Windows

      **Prerequisites**

      - `Visual Studio 2017+ <https://visualstudio.microsoft.com/downloads/>`_

      The ovrtx library is downloaded automatically at configure time. If ovrtx is already installed and available via ``CMAKE_PREFIX_PATH``, the local installation is used instead.

      **Building**

      .. code-block:: pwsh

         cmake -B build
         cmake --build build --config Release

      **Running**

      .. code-block:: pwsh

         .\build\Release\minimal

Licensing
---------

This example contains ``stb_image_write.h``, © Sean Barrett, released under Public Domain.
