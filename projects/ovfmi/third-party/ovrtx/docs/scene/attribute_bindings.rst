.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
..
.. NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
.. property and proprietary rights in and to this material, related
.. documentation and any modifications thereto. Any use, reproduction,
.. disclosure or distribution of this material and related documentation
.. without an express license agreement from NVIDIA CORPORATION or
.. its affiliates is strictly prohibited.

Attribute Bindings
==================

Attribute bindings are persistent descriptors for writing the same attribute on
the same set of prims repeatedly. They are useful for animation loops and other
hot paths where rebuilding the prim list, attribute name, dtype, and semantic on
every write would add overhead.

Use regular writes from :doc:`attributes` for one-shot edits. Use
:doc:`attribute_mapping` when the hot path needs zero-copy writes into
ovrtx-owned buffers.

Create and Write
----------------

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../tests/docs/python/test_attribute_bindings.py
         :language: python
         :start-after: # [snippet:doc-bind-attribute-write]
         :end-before: # [/snippet:doc-bind-attribute-write]
         :dedent:

   .. tab-item:: C

      .. literalinclude:: ../../tests/docs/c/test_attribute_bindings.cpp
         :language: cpp
         :start-after: // [snippet:doc-create-attribute-binding-c]
         :end-before: // [/snippet:doc-create-attribute-binding-c]
         :dedent:

      .. literalinclude:: ../../tests/docs/c/test_attribute_bindings.cpp
         :language: cpp
         :start-after: // [snippet:doc-write-bound-attribute-c]
         :end-before: // [/snippet:doc-write-bound-attribute-c]
         :dedent:

      .. literalinclude:: ../../tests/docs/c/test_attribute_bindings.cpp
         :language: cpp
         :start-after: // [snippet:doc-destroy-attribute-binding-c]
         :end-before: // [/snippet:doc-destroy-attribute-binding-c]
         :dedent:

Async Creation and Writes
-------------------------

Python exposes async binding creation and async writes for non-blocking update
pipelines.

.. literalinclude:: ../../tests/docs/python/test_attribute_bindings.py
   :language: python
   :start-after: # [snippet:doc-bind-attribute-async]
   :end-before: # [/snippet:doc-bind-attribute-async]
   :dedent:

.. literalinclude:: ../../tests/docs/python/test_attribute_bindings.py
   :language: python
   :start-after: # [snippet:doc-binding-write-async]
   :end-before: # [/snippet:doc-binding-write-async]
   :dedent:

Array Attributes
----------------

Use array bindings for variable-length USD array attributes such as mesh
``points``. Array writes take one tensor per prim.

.. literalinclude:: ../../tests/docs/python/test_attribute_bindings.py
   :language: python
   :start-after: # [snippet:doc-bind-array-attribute]
   :end-before: # [/snippet:doc-bind-array-attribute]
   :dedent:

Mapping Through a Binding
-------------------------

Persistent bindings can also be mapped, which avoids recreating the descriptor
before each map/unmap cycle.

.. literalinclude:: ../../tests/docs/python/test_attribute_bindings.py
   :language: python
   :start-after: # [snippet:doc-map-bound-attribute]
   :end-before: # [/snippet:doc-map-bound-attribute]
   :dedent:

Lifetime Rules
--------------

- Explicitly unbind or destroy bindings when the hot path is done.
- A binding fixes the prim list, attribute name, dtype, shape, and semantic.
- In C, keep strings and descriptor arrays alive until binding creation has
  completed.
- ``OVRTX_BINDING_FLAG_OPTIMIZE`` is intended for frequent high-volume writes.
  In Python, the equivalent is ``BindingFlag.OPTIMIZE``.
