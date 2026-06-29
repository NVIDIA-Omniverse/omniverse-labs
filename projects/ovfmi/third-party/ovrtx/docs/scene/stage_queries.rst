.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
..
.. NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
.. property and proprietary rights in and to this material, related
.. documentation and any modifications thereto. Any use, reproduction,
.. disclosure or distribution of this material and related documentation
.. without an express license agreement from NVIDIA CORPORATION or
.. its affiliates is strictly prohibited.

Stage Queries
=============

Stage queries discover prims on the runtime stage and optionally report
attribute schema metadata. A typical workflow is:

1. Query prims by type, attribute existence, or a filter combination.
2. Inspect returned paths and attribute descriptors.
3. Reuse the returned prim-list handles in later C reads or writes.

Filters combine as AND (``require_all``), OR (``require_any``), and NOT
(``exclude``). Attribute reporting can be disabled, enabled for all attributes,
or restricted to a requested name list.

Python Queries
--------------

.. tab-set::

   .. tab-item:: Basic

      .. literalinclude:: ../../tests/docs/python/test_stage_query.py
         :language: python
         :start-after: # [snippet:doc-query-prims-basic]
         :end-before: # [/snippet:doc-query-prims-basic]
         :dedent:

   .. tab-item:: By Type

      .. literalinclude:: ../../tests/docs/python/test_stage_query.py
         :language: python
         :start-after: # [snippet:doc-query-prims-by-type]
         :end-before: # [/snippet:doc-query-prims-by-type]
         :dedent:

   .. tab-item:: Attributes

      .. literalinclude:: ../../tests/docs/python/test_stage_query.py
         :language: python
         :start-after: # [snippet:doc-query-prims-with-attributes]
         :end-before: # [/snippet:doc-query-prims-with-attributes]
         :dedent:

   .. tab-item:: Combined

      .. literalinclude:: ../../tests/docs/python/test_stage_query.py
         :language: python
         :start-after: # [snippet:doc-query-require-any-exclude]
         :end-before: # [/snippet:doc-query-require-any-exclude]
         :dedent:

C Queries
---------

.. tab-set::

   .. tab-item:: Basic

      .. literalinclude:: ../../tests/docs/c/test_stage_query.cpp
         :language: cpp
         :start-after: // [snippet:doc-query-prims-basic-c]
         :end-before: // [/snippet:doc-query-prims-basic-c]
         :dedent:

   .. tab-item:: By Type

      .. literalinclude:: ../../tests/docs/c/test_stage_query.cpp
         :language: cpp
         :start-after: // [snippet:doc-query-prims-by-type-c]
         :end-before: // [/snippet:doc-query-prims-by-type-c]
         :dedent:

   .. tab-item:: Has Attribute

      .. literalinclude:: ../../tests/docs/c/test_stage_query.cpp
         :language: cpp
         :start-after: // [snippet:doc-query-has-attribute-c]
         :end-before: // [/snippet:doc-query-has-attribute-c]
         :dedent:

   .. tab-item:: Combined

      .. literalinclude:: ../../tests/docs/c/test_stage_query.cpp
         :language: cpp
         :start-after: // [snippet:doc-query-require-any-exclude-c]
         :end-before: // [/snippet:doc-query-require-any-exclude-c]
         :dedent:

Async Queries
-------------

Python async queries follow the same ``Operation`` / ``PendingFetch`` lifecycle
as other async APIs:

.. literalinclude:: ../../tests/docs/python/test_stage_query.py
   :language: python
   :start-after: # [snippet:doc-query-prims-async]
   :end-before: # [/snippet:doc-query-prims-async]
   :dedent:

Path Dictionary
---------------

C query results use token and prim-path ids. Resolve them through the renderer's
path dictionary while the query results are still valid:

.. literalinclude:: ../../tests/docs/c/test_stage_query.cpp
   :language: cpp
   :start-after: // [snippet:doc-path-dictionary-resolve-c]
   :end-before: // [/snippet:doc-path-dictionary-resolve-c]
   :dedent:

Python query results return strings directly for prim paths and attribute names.

Troubleshooting
---------------

- Release C query results only after copying any strings, descriptors, or ids you
  need to keep.
- ``AttributeFilterMode.SPECIFIC`` with an empty attribute-name list returns no
  descriptors. Use ``ALL`` to dump every descriptor or ``NONE`` for lightweight
  discovery.
- Relationship-valued attributes surface as path ids in C. Resolve them through
  the path dictionary before printing or storing string paths.

