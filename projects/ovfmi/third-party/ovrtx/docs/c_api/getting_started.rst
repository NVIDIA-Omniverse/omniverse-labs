.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
..
.. NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
.. property and proprietary rights in and to this material, related
.. documentation and any modifications thereto. Any use, reproduction,
.. disclosure or distribution of this material and related documentation
.. without an express license agreement from NVIDIA CORPORATION or
.. its affiliates is strictly prohibited.

Getting Started in C
====================

The C/C++ examples require CMake and a development environment. Install the prerequisites for your platform:

.. tab-set::

   .. tab-item:: Windows

      Install `Visual Studio 2017 or newer <https://visualstudio.microsoft.com/>`_ (which provides CMake and a C++ toolchain).

   .. tab-item:: Linux (Ubuntu)

      .. code-block:: bash

         sudo apt-get install build-essential cmake

Next, clone `the repository <https://github.com/NVIDIA-Omniverse/ovrtx>`__ and configure the minimal example:

.. code-block:: bash

   git clone https://github.com/NVIDIA-Omniverse/ovrtx.git
   cd ovrtx/examples/c/minimal
   cmake -B build

Last, build and run the minimal example (choose your platform):

.. tab-set::

   .. tab-item:: Windows

      .. code-block:: text

         cmake --build build --config Release
         .\build\Release\minimal.exe

   .. tab-item:: Linux

      .. code-block:: bash

         cmake --build build --config Release
         ./build/minimal

The minimal example shows how to create the renderer, load an OpenUSD scene and render a single image, copying the results back to the CPU for writing out as a PNG.

.. image:: ../../img/example-minimal.jpg
   :alt: Minimal example output
   :align: center

The resulting image is written to ``./out.png`` and you can inspect it with any image viewer.

The first time you run a program built against ovrtx, it compiles and caches necessary shaders, which may take some time depending on your system. Subsequent runs use the cached shaders and are faster.

Installation
------------

CMake
^^^^^

ovrtx binary distributions can be found on the GitHub `Releases page <https://github.com/NVIDIA-Omniverse/ovrtx/releases>`__, and contain a CMake config.

Alternatively, download the appropriate package for your system from the `Releases page <https://github.com/NVIDIA-Omniverse/ovrtx/releases>`__ and point
``CMAKE_PREFIX_PATH`` to the directory where you extracted the archive and use ``find_package(ovrtx)``
from your ``CMakeLists.txt``.

For other build systems, download the appropriate package for your system from the `Releases page <https://github.com/NVIDIA-Omniverse/ovrtx/releases>`__. The headers are in the
``include`` directory and libraries are in ``lib`` and ``bin``, in either static or dynamic flavors.

The simplest way to add ovrtx as a dependency to your project is using CMake FetchContent:

.. literalinclude:: ../../examples/c/cmake/ovrtx.cmake
   :language: cmake
   :lines: 27-83

Note that the macro above is provided for convenience in ``ovrtx.cmake`` in the ``examples/c/cmake`` directory in `the repository <https://github.com/NVIDIA-Omniverse/ovrtx>`__.

Runtime Packaging and Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ovrtx requires several libraries and other runtime dependencies to be present and discoverable at runtime. These are all included in the ovrtx binary distribution under the ``bin`` directory:

.. code-block:: text

   bin/
   ├── libovrtx-dynamic.so / ovrtx-dynamic.dll
   ├── cache/
   ├── library/
   ├── libs/
   ├── mdl/
   ├── plugins/
   ├── rendering-data/
   └── usd_plugins/

The ovrtx dynamic library will automatically load the other dependencies at runtime if it is placed alongside them as in the binary distribution. If you need to deploy your application with a different layout, you can point ovrtx to the correct paths using the :c:func:`ovrtx_config_entry_binary_package_root_path` helper function when configuring the renderer:

.. code-block:: c

   ovrtx_config_entry_t config_entries[] = {
       ovrtx_config_entry_binary_package_root_path(ovx_string("/path/where/bin/contents/live"))
   };

   ovrtx_config_t config;
   config.entries = config_entries;
   config.entry_count = sizeof(config_entries) / sizeof(config_entries[0]);

   ovrtx_renderer_t* renderer;
   ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer);

Note that when static linking ovrtx, you MUST provide the binary package root path or ovrtx will not be able to find the required dependencies at runtime.

Sharing OpenUSD with Other Subsystems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ovrtx shares the OpenUSD runtime with other subsystems in the same process, every
subsystem must publish its USD schema and plugin discovery paths *before* the USD schema
registry is first populated; that registry is built only once per process. Use
:c:func:`ovrtx_register_schema_paths` early so the order of subsequent initialize calls
does not matter:

.. code-block:: c

   // Register schema paths up front, then initialize subsystems in any order.
   // Pass the same config you will later supply to ovrtx_initialize / ovrtx_create_renderer
   // so the binary package root used during registration matches the one used at init.
   ovphysx_prepare_usd_plugins();
   ovrtx_register_schema_paths(&config);

   ovrtx_create_renderer(&config, &renderer);

For default deployments where ovrtx lives next to the loader library, ``NULL`` is
acceptable and the loader-library directory is used as the root:

.. code-block:: c

   ovrtx_register_schema_paths(NULL);

For ovrtx-only applications this call is not required — :c:func:`ovrtx_initialize` /
:c:func:`ovrtx_create_renderer` register the same paths automatically.

Once schema paths have been registered against an effective binary package root, any
later :c:func:`ovrtx_register_schema_paths`, :c:func:`ovrtx_initialize`, or
:c:func:`ovrtx_create_renderer` call that resolves to a different root logs a warning
to stderr and is treated as a no-op against the first-registered root —
``PXR_PLUGINPATH_NAME`` is one-shot per process, so the first call wins. Use the same
``OVRTX_CONFIG_BINARY_PACKAGE_ROOT_PATH`` (or the same ``OMNI_USD_PLUGINS_BASE_PATH``
override) throughout the process. See the function's API documentation for the full
contract.

Minimal Example
---------------

.. filtered-literalinclude:: ../../examples/c/minimal/main.cpp
   :language: cpp
   :start-after: // its affiliates is strictly prohibited.
   :exclude-pattern: ^\s*//\s*\[/?snippet:

.. image:: ../../img/example-minimal.jpg
   :alt: Minimal example output
   :align: center

Next Steps
----------

* Explore more :doc:`../examples/index` including the :doc:`Vulkan Interop <../examples/c_vulkan_interop>` example with real-time GPU rendering, click picking, marquee selection, and styled selection outlines with translucent fill.
* See the :doc:`index` for the full C API reference.
