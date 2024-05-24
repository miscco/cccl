.. _libcudacxx-ptx-instructions-red-async:

red.async
=========

-  PTX ISA: `red.async <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async>`_

red.async
---------

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.*.u32
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .u32 }
   // .op        = { .add, .min, .max, .inc, .dec}
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_inc_t,
     uint32_t* dest,
     const uint32_t& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.*.s32
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .s32 }
   // .op        = { .add, .min, .max}
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_min_t,
     int32_t* dest,
     const int32_t& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.*.b32
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .b32 }
   // .op        = { .and, .or, xor }
   template <typename B32>
   __device__ static inline void red_async(
     cuda::ptx::op_and_op_t,
     B32* dest,
     const B32& value,
     uint64_t* remote_bar);

red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
   // .type      = { .u64 }
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_add_t,
     uint64_t* dest,
     const uint64_t& value,
     uint64_t* remote_bar);

red.async ``.s64`` emulation
----------------------------

PTX does not currently (CTK 12.3) expose ``red.async.add.s64``. This exposure is emulated in ``cuda::ptx`` using

.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}.u64  [dest], value, [remote_bar]; // .u64 intentional PTX ISA 81, SM_90
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_add_t,
     int64_t* dest,
     const int64_t& value,
     int64_t* remote_bar);
