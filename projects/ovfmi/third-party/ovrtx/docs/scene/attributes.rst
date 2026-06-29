.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
..
.. NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
.. property and proprietary rights in and to this material, related
.. documentation and any modifications thereto. Any use, reproduction,
.. disclosure or distribution of this material and related documentation
.. without an express license agreement from NVIDIA CORPORATION or
.. its affiliates is strictly prohibited.

Attribute Reads and Writes
==========================

ovrtx reads and writes runtime stage attributes using DLPack tensors. The dtype
and shape must match the USD attribute schema. Scalar APIs operate on one value
per prim. Array APIs operate on variable-length arrays such as mesh points or
relationships.

Tensor Layout
-------------

Python uses NumPy-style trailing dimensions for vectors and matrices. C uses
``DLDataType::lanes`` for multi-component values.

.. list-table::
   :header-rows: 1

   * - USD value
     - Python shape
     - C shape and dtype
   * - ``int`` for N prims
     - ``(N,)`` ``int32``
     - ``shape=[N]``, ``{kDLInt, 32, 1}``
   * - ``point3f`` for N prims
     - ``(N, 3)`` ``float32``
     - ``shape=[N]``, ``{kDLFloat, 32, 3}``
   * - ``matrix4d`` for N prims
     - ``(N, 16)`` ``float64``
     - ``shape=[N]``, ``{kDLFloat, 64, 16}``
   * - 4x4 transform semantic for N prims
     - ``(N, 4, 4)`` ``float64``
     - ``shape=[N]``, ``{kDLFloat, 64, 16}``
   * - ``point3f[]`` with M elements
     - ``(M, 3)`` ``float32``
     - ``shape=[M]``, ``{kDLFloat, 32, 3}``

Reading Attributes
------------------

.. tab-set::

   .. tab-item:: Python Scalar

      .. literalinclude:: ../../tests/docs/python/test_attribute_read.py
         :language: python
         :start-after: # [snippet:doc-read-attribute-scalar]
         :end-before: # [/snippet:doc-read-attribute-scalar]
         :dedent:

   .. tab-item:: Python Array

      .. literalinclude:: ../../tests/docs/python/test_attribute_read.py
         :language: python
         :start-after: # [snippet:doc-read-array-attribute]
         :end-before: # [/snippet:doc-read-array-attribute]
         :dedent:

   .. tab-item:: C Scalar

      .. literalinclude:: ../../tests/docs/c/test_attribute_read.cpp
         :language: cpp
         :start-after: // [snippet:doc-read-attribute-scalar-c]
         :end-before: // [/snippet:doc-read-attribute-scalar-c]
         :dedent:

   .. tab-item:: C Array

      .. literalinclude:: ../../tests/docs/c/test_attribute_read.cpp
         :language: cpp
         :start-after: // [snippet:doc-read-array-attribute-c]
         :end-before: // [/snippet:doc-read-array-attribute-c]
         :dedent:

Python reads can also write directly into a caller-provided DLPack destination,
including CUDA destinations:

.. tab-set::

   .. tab-item:: CPU destination

      .. literalinclude:: ../../tests/docs/python/test_attribute_read.py
         :language: python
         :start-after: # [snippet:doc-read-attribute-dest-tensor]
         :end-before: # [/snippet:doc-read-attribute-dest-tensor]
         :dedent:

   .. tab-item:: CUDA destination

      .. literalinclude:: ../../tests/docs/python/test_attribute_read.py
         :language: python
         :start-after: # [snippet:doc-read-attribute-cuda-dest]
         :end-before: # [/snippet:doc-read-attribute-cuda-dest]
         :dedent:

Writing Attributes
------------------

.. tab-set::

   .. tab-item:: Python Array

      .. literalinclude:: ../../tests/docs/python/test_attribute_shapes.py
         :language: python
         :start-after: # [snippet:doc-shape-float3-array]
         :end-before: # [/snippet:doc-shape-float3-array]
         :dedent:

   .. tab-item:: Python Token Array

      .. literalinclude:: ../../tests/docs/python/test_attribute_bindings.py
         :language: python
         :start-after: # [snippet:doc-write-token-array]
         :end-before: # [/snippet:doc-write-token-array]
         :dedent:

   .. tab-item:: C Scalar

      .. literalinclude:: ../../tests/docs/c/test_attribute_bindings.cpp
         :language: cpp
         :start-after: // [snippet:doc-write-bound-attribute-c]
         :end-before: // [/snippet:doc-write-bound-attribute-c]
         :dedent:

Data Access
-----------

Synchronous writes copy data before the call returns. Asynchronous writes may
access the caller's memory later during stream execution, so the source tensor
must remain alive until the operation completes. String data supports only
synchronous access.

Python exposes this through ``DataAccess.SYNC`` and ``DataAccess.ASYNC``. C uses
the access mode argument to :c:func:`ovrtx_write_attribute`.

Type Notes
----------

- Use ``read_array_attribute`` / ``write_array_attribute`` for USD array
  attributes and relationships.
- Semantics are write-side conversion hints. Attribute reads use raw storage
  layout and ``OVRTX_SEMANTIC_NONE``.
- Quaternion tensor order is ``(i, j, k, real)`` even though USDA authors values
  as ``(real, i, j, k)``.
- ``string`` attributes are represented as UTF-8 byte arrays. String arrays are
  not supported; use ``token[]`` for string-like arrays.
- Python can write token strings directly. C can create and resolve token ids
  through the path dictionary.
- Scalar ``asset`` values are supported as token pairs in C. Asset arrays and
  timecode attributes are not supported as runtime attributes.

C Convenience Helpers
---------------------

For path, token, and transform attributes, prefer helpers in
``<ovrtx/ovrtx_attributes.h>`` where available. For token strings:

.. literalinclude:: ../../tests/docs/c/test_attribute_helpers.cpp
   :language: cpp
   :start-after: // [snippet:doc-set-token-attributes-c]
   :end-before: // [/snippet:doc-set-token-attributes-c]
   :dedent:

Troubleshooting
---------------

- Match the runtime dtype, not the Python or C default numeric type.
- Array writes in Python take one tensor per prim.
- ``PrimMode.EXISTING_ONLY`` skips missing prims; use ``MUST_EXIST`` when a
  missing prim should be an error.
- In C, binding descriptors borrow path storage. Keep the strings and arrays
  alive until the operation that uses the descriptor has completed.
- Generic authored USD attributes require
  ``customLayerData.populateAllAuthoredAttributes = true`` on the root layer.

