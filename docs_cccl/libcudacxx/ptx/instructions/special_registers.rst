.. _libcudacxx-ptx-instructions-special-registers:

special registers
=================

-  PTX ISA: `Special Registers <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers>`_

tid.x
""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%tid.x; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_tid_x();

tid.y
""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%tid.y; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_tid_y();

tid.z
""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%tid.z; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_tid_z();

ntid.x
""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%ntid.x; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_ntid_x();

ntid.y
""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%ntid.y; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_ntid_y();

ntid.z
""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%ntid.z; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_ntid_z();

get_sreg_laneid
""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%laneid; // PTX ISA 13
   template <typename=void>
   __device__ static inline uint32_t get_sreg_laneid();

get_sreg_warpid
""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%warpid; // PTX ISA 13
   template <typename=void>
   __device__ static inline uint32_t get_sreg_warpid();

get_sreg_nwarpid
""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%nwarpid; // PTX ISA 20, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_nwarpid();

get_sreg_ctaid_x
""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%ctaid.x; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_ctaid_x();

get_sreg_ctaid_y
""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%ctaid.y; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_ctaid_y();

get_sreg_ctaid_z
""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%ctaid.z; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_ctaid_z();

get_sreg_nctaid_x
""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%nctaid.x; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_nctaid_x();

get_sreg_nctaid_y
""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%nctaid.y; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_nctaid_y();

get_sreg_nctaid_z
""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%nctaid.z; // PTX ISA 20
   template <typename=void>
   __device__ static inline uint32_t get_sreg_nctaid_z();

get_sreg_smid
""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%smid; // PTX ISA 13
   template <typename=void>
   __device__ static inline uint32_t get_sreg_smid();

get_sreg_nsmid
""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%nsmid; // PTX ISA 20, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_nsmid();

get_sreg_gridid
""""""""""""""""""""""""

.. code:: cuda

   // mov.u64 sreg_value, %%gridid; // PTX ISA 30
   template <typename=void>
   __device__ static inline uint64_t get_sreg_gridid();

get_sreg_is_explicit_cluster
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.pred sreg_value, %%is_explicit_cluster; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline bool get_sreg_is_explicit_cluster();

get_sreg_clusterid.x
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%clusterid.x; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_clusterid_x();

get_sreg_clusterid.y
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%clusterid.y; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_clusterid_y();

get_sreg_clusterid.z
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%clusterid.z; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_clusterid_z();

get_sreg_nclusterid.x
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%nclusterid.x; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_nclusterid_x();

get_sreg_nclusterid.y
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%nclusterid.y; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_nclusterid_y();

get_sreg_nclusterid.z
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%nclusterid.z; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_nclusterid_z();

get_sreg_cluster_ctaid.x
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%cluster_ctaid.x; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_cluster_ctaid_x();

get_sreg_cluster_ctaid.y
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%cluster_ctaid.y; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_cluster_ctaid_y();

get_sreg_cluster_ctaid.z
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%cluster_ctaid.z; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_cluster_ctaid_z();

get_sreg_cluster_nctaid.x
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%cluster_nctaid.x; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_cluster_nctaid_x();

get_sreg_cluster_nctaid.y
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%cluster_nctaid.y; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_cluster_nctaid_y();

get_sreg_cluster_nctaid.z
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%cluster_nctaid.z; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_cluster_nctaid_z();

get_sreg_cluster_ctarank
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%cluster_ctarank; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_cluster_ctarank();

get_sreg_cluster_nctarank
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%cluster_nctarank; // PTX ISA 78, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_cluster_nctarank();

get_sreg_lanemask_eq
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%lanemask_eq; // PTX ISA 20, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_lanemask_eq();

get_sreg_lanemask_le
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%lanemask_le; // PTX ISA 20, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_lanemask_le();

get_sreg_lanemask_lt
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%lanemask_lt; // PTX ISA 20, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_lanemask_lt();

get_sreg_lanemask_ge
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%lanemask_ge; // PTX ISA 20, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_lanemask_ge();

get_sreg_lanemask_gt
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%lanemask_gt; // PTX ISA 20, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_lanemask_gt();

get_sreg_clock
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%clock; // PTX ISA 10
   template <typename=void>
   __device__ static inline uint32_t get_sreg_clock();

get_sreg_clock_hi
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%clock_hi; // PTX ISA 50, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_clock_hi();

get_sreg_clock64
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u64 sreg_value, %%clock64; // PTX ISA 20, SM_35
   template <typename=void>
   __device__ static inline uint64_t get_sreg_clock64();

get_sreg_globaltimer
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u64 sreg_value, %%globaltimer; // PTX ISA 31, SM_35
   template <typename=void>
   __device__ static inline uint64_t get_sreg_globaltimer();

get_sreg_globaltimer_lo
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%globaltimer_lo; // PTX ISA 31, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_globaltimer_lo();

get_sreg_globaltimer_hi
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%globaltimer_hi; // PTX ISA 31, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_globaltimer_hi();

get_sreg_total_smem_size
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%total_smem_size; // PTX ISA 41, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_total_smem_size();

get_sreg_aggr_smem_size
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%aggr_smem_size; // PTX ISA 81, SM_90
   template <typename=void>
   __device__ static inline uint32_t get_sreg_aggr_smem_size();

get_sreg_dynamic_smem_size
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u32 sreg_value, %%dynamic_smem_size; // PTX ISA 41, SM_35
   template <typename=void>
   __device__ static inline uint32_t get_sreg_dynamic_smem_size();

get_sreg_current_graph_exec
"""""""""""""""""""""""""""""

.. code:: cuda

   // mov.u64 sreg_value, %%current_graph_exec; // PTX ISA 80, SM_50
   template <typename=void>
   __device__ static inline uint64_t get_sreg_current_graph_exec();
