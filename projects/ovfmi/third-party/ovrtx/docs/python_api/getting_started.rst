.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
..
.. NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
.. property and proprietary rights in and to this material, related
.. documentation and any modifications thereto. Any use, reproduction,
.. disclosure or distribution of this material and related documentation
.. without an express license agreement from NVIDIA CORPORATION or
.. its affiliates is strictly prohibited.

Getting Started in Python
=========================

During Early Access, use the `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ Python package and project manager. All the examples in `the repository <https://github.com/NVIDIA-Omniverse/ovrtx>`__ contain ``pyproject.toml`` files that are tested with uv.

Python 3.10–3.13 is required.

To get started, first clone `the repository <https://github.com/NVIDIA-Omniverse/ovrtx>`__ and run the first example with uv:

.. code-block:: bash

   git clone https://github.com/NVIDIA-Omniverse/ovrtx.git
   cd ovrtx/examples/python/minimal
   uv run main.py

`The minimal example <https://github.com/NVIDIA-Omniverse/ovrtx/tree/main/examples/python/minimal>`__ shows how to create the renderer, load an OpenUSD scene and render a single image, copying the results back to the CPU for display.

.. image:: ../../img/example-minimal.jpg
   :alt: Minimal example output
   :align: center

The first time you run a program built against ovrtx, it compiles and caches necessary shaders, which may take some time depending on your system. Subsequent runs use the cached shaders and are faster.

Minimal Example
---------------

.. filtered-literalinclude:: ../../examples/python/minimal/main.py
   :language: python
   :start-after: # its affiliates is strictly prohibited.
   :exclude-pattern: ^\s*#\s*\[/?snippet:

The example above is provided as a Python project in the ``examples/python/minimal`` directory in `the repository <https://github.com/NVIDIA-Omniverse/ovrtx>`__.

Next Steps
----------

* Explore more :doc:`../examples/index` including the :doc:`Planet System <../examples/python_planet_system>` demo with GPU-accelerated animation.
* See the :doc:`index` for the full Python API.
