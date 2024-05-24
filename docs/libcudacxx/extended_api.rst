.. _libcudacxx-extended-api:

Extended API
============

.. toctree::
   :hidden:
   :maxdepth: 2

   extended_api/memory_model
   extended_api/thread_groups
   extended_api/shapes
   extended_api/synchronization_primitives
   extended_api/asynchronous_operations
   extended_api/memory_access_properties
   extended_api/functional
   extended_api/stream_ref
   extended_api/memory_resource


Fundamentals
------------

| `Thread Scopes <./extended_api/memory_model.md#thread-scopes>`_ \|
  Defines the kind of threads that can synchronize using a primitive.
  ``(enum)`` 1.0.0 / CUDA 10.2 \|
| `Thread Groups <./extended_api/thread_groups.md>`_ \| Concepts for
  groups of cooperating threads. ``(concept)`` 1.2.0 / CUDA 11.1 \|

{% include_relative extended_api/shapes.md %}

{% include_relative extended_api/synchronization_primitives.md %}

{% include_relative extended_api/asynchronous_operations.md %}

{% include_relative extended_api/memory_access_properties.md %}

{% include_relative extended_api/functional.md %}

{% include_relative extended_api/memory_resource.md %}
