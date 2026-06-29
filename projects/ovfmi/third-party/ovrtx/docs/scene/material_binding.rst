.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
..
.. NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
.. property and proprietary rights in and to this material, related
.. documentation and any modifications thereto. Any use, reproduction,
.. disclosure or distribution of this material and related documentation
.. without an express license agreement from NVIDIA CORPORATION or
.. its affiliates is strictly prohibited.

Material Binding
================

Materials in USD are assigned to geometry prims via the ``material:binding`` relationship. ovrtx lets you change these bindings at runtime, so you can swap materials on prims without reloading the scene.

To bind a material, write the ``material:binding`` attribute on the target geometry prim with the absolute path of the material prim as a path string.

.. note::

   The material prim must already exist in the stage (loaded from USD). This operation changes which existing material is assigned to a prim -- it does not create new materials.

Binding a Material
------------------

.. tab-set::

   .. tab-item:: Python

      Use :py:meth:`~ovrtx.Renderer.write_array_attribute()` with the prim path, the ``material:binding`` attribute name, and the material path as a ``list[list[str]]``. The string list format is automatically detected as path/relationship data.

      .. literalinclude:: ../../tests/docs/python/test_base.py
         :language: python
         :start-after: # [snippet:doc-bind-material]
         :end-before: # [/snippet:doc-bind-material]
         :dedent:

   .. tab-item:: C

      Use the :c:func:`ovrtx_set_path_attributes()` convenience helper from ``<ovrtx/ovrtx_attributes.h>``. It wraps the path value into the single-element relationship array that USD requires.

      .. literalinclude:: ../../tests/docs/c/test_base.cpp
         :language: cpp
         :start-after: // [snippet:doc-bind-material-c]
         :end-before: // [/snippet:doc-bind-material-c]
         :dedent:
