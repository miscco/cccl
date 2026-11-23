// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! Operations for writing linear segments of data from the CUDA thread block

#pragma once

#include <cub/config.cuh>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_exchange.cuh>
#include <cub/detail/it_traits.cuh>
#include <cub/detail/null_type.cuh>
#include <cub/detail/numeric_traits.cuh>
#include <cub/detail/uninitialized.cuh>
#include <cub/util_ptx.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__memory/__vectorized_storage.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__memory/is_sufficiently_aligned.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__memory/runtime_assume_aligned.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/remove_reference.h>

#include <nv/target>

#if __cpp_lib_format >= 201907L
#  include <cuda/std/__concepts/same_as.h>
#  include <cuda/std/__fwd/format.h> // IWYU pragma: keep
#endif // __cpp_lib_format >= 201907L

CUB_NAMESPACE_BEGIN

//! @name Blocked arrangement I/O (direct)
//! @{

//! @rst
//! Store a blocked arrangement of items across a thread block into a linear segment of items
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @blocked
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., ``(threadIdx.y * blockDim.x) + linear_tid`` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[in] items
//!   Data to store
template <typename T, int ItemsPerThread, typename OutputIteratorT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
StoreDirectBlocked(int linear_tid, OutputIteratorT block_itr, T (&items)[ItemsPerThread])
{
  OutputIteratorT thread_itr = block_itr + (linear_tid * ItemsPerThread); // NOLINT(bugprone-misplaced-widening-cast)

  // Store directly in thread-blocked order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ItemsPerThread; ITEM++)
  {
    thread_itr[ITEM] = items[ITEM];
  }
}

//! @rst
//! Store a blocked arrangement of items across a
//! thread block into a linear segment of items, guarded by range
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @blocked
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[in] items
//!   Data to store
//!
//! @param[in] valid_items
//!   Number of valid items to write
template <typename T, int ItemsPerThread, typename OutputIteratorT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
StoreDirectBlocked(int linear_tid, OutputIteratorT block_itr, T (&items)[ItemsPerThread], int valid_items)
{
  OutputIteratorT thread_itr = block_itr + (linear_tid * ItemsPerThread); // NOLINT(bugprone-misplaced-widening-cast)

  // Store directly in thread-blocked order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ItemsPerThread; ITEM++)
  {
    if (ITEM + (linear_tid * ItemsPerThread) < valid_items)
    {
      thread_itr[ITEM] = items[ITEM];
    }
  }
}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{
//! @brief Whether the storage of ``OutputIteratorT`` can be stored with vectors of ``T``
// TODO(bgruber): remove the check for is_primitive<T> in CCCL 4.0
template <typename OutputIteratorT, typename T>
inline constexpr bool block_store_can_vectorize_v =
  THRUST_NS_QUALIFIER::is_contiguous_iterator_v<OutputIteratorT>
  && (::cuda::is_trivially_copyable_v<T> || is_primitive_v<T>)
  && ::cuda::std::is_same_v<it_value_t<OutputIteratorT>, T>
  // handles contiguous iterators wrapping volatile pointers, like span<volatile int>::iterator
  && !::cuda::std::is_volatile_v<::cuda::std::remove_reference_t<it_reference_t<OutputIteratorT>>>;

//! @brief The width in bytes of the widest vectors with which ``StoreDirectBlockedVectorized`` stores vectors of
//! ``VecSize`` consecutive items of ``T``, wider than 16 bytes only from sm_100
template <typename T, int VecSize>
inline constexpr ::cuda::std::size_t block_store_max_vector_access_v = ::cuda::std::min(
  ::cuda::std::size_t{1} << ::cuda::std::countr_zero(VecSize * sizeof(T)), ::cuda::__max_memory_access_width);

//! @brief Stores the ``NumVectors`` vectors of ``VecSize`` items of a thread with accesses of ``Alignment`` bytes,
//! which ``dst_ptr`` must be aligned to, ``VectorStride`` vectors apart from vector ``vector_index`` after ``dst_ptr``
//! on
template <::cuda::std::size_t Alignment, int VectorStride, int VecSize, typename T, typename VectorT, int NumVectors>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
store_direct_vectors(T* dst_ptr, unsigned int vector_index, const VectorT (&vectors)[NumVectors])
{
  using aligned_vector_t _CCCL_NODEBUG = ::cuda::__as_vectorized_storage_t<T, VecSize, Alignment>;
  aligned_vector_t* vec_ptr =
    reinterpret_cast<aligned_vector_t*>(::cuda::std::__runtime_assume_aligned(dst_ptr, Alignment)) + vector_index;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumVectors; i++)
  {
    // The vector is assigned word by word into a copy of the store alignment, which is then stored as a whole, so that
    // the words are stored with accesses of Alignment bytes
    aligned_vector_t aligned_vector;
    aligned_vector            = vectors[i];
    vec_ptr[i * VectorStride] = aligned_vector;
  }
}

//! @brief Stores the ``ItemsPerThread / VecSize`` vectors of ``VecSize`` items of a thread, ``VectorStride`` vectors
//! apart from vector ``vector_index`` after ``dst_ptr`` on, with the widest accesses that the alignment of
//! ``aligned_ptr`` allows, or item by item to ``thread_dst_ptr``, the first item of the thread. ``dst_ptr`` must be a
//! whole number of vectors after ``aligned_ptr``
template <int VectorStride, int VecSize, typename T, int ItemsPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void store_direct_vectorized(
  T* aligned_ptr, T* dst_ptr, unsigned int vector_index, T* thread_dst_ptr, const T (&items)[ItemsPerThread])
{
  constexpr int vectors_per_thread = ItemsPerThread / VecSize;
  constexpr auto max_vector_access = block_store_max_vector_access_v<T, VecSize>;
  if constexpr (block_store_can_vectorize_v<T*, T> && 4 <= max_vector_access && alignof(T) < max_vector_access)
  {
    using vector_t _CCCL_NODEBUG = ::cuda::__as_vectorized_storage_t<T, VecSize, max_vector_access>;

    // The items are copied into the vectors once for all widths. Every vector starts a whole number of vectors after
    // aligned_ptr, thus a single test selects the accesses of all vectors of the thread. Only sm_100 and newer store
    // vectors wider than 16 bytes with a single access
    vector_t vectors[vectors_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < vectors_per_thread; i++)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < VecSize; j++)
      {
        vectors[i].template __alias<T>(j) = items[(i * VecSize) + j];
      }
    }

    if constexpr (alignof(T) < 32 && 32 <= max_vector_access)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_100, ({
                     if (::cuda::std::is_sufficiently_aligned<32>(aligned_ptr))
                     {
                       _CUB::detail::store_direct_vectors<32, VectorStride, VecSize>(dst_ptr, vector_index, vectors);
                       return;
                     }
                   }))
    }
    if constexpr (alignof(T) < 16 && 16 <= max_vector_access)
    {
      if (::cuda::std::is_sufficiently_aligned<16>(aligned_ptr))
      {
        _CUB::detail::store_direct_vectors<16, VectorStride, VecSize>(dst_ptr, vector_index, vectors);
        return;
      }
    }
    if constexpr (alignof(T) < 8 && 8 <= max_vector_access)
    {
      if (::cuda::std::is_sufficiently_aligned<8>(aligned_ptr))
      {
        _CUB::detail::store_direct_vectors<8, VectorStride, VecSize>(dst_ptr, vector_index, vectors);
        return;
      }
    }
    if constexpr (alignof(T) < 4 && 4 <= max_vector_access)
    {
      if (::cuda::std::is_sufficiently_aligned<4>(aligned_ptr))
      {
        _CUB::detail::store_direct_vectors<4, VectorStride, VecSize>(dst_ptr, vector_index, vectors);
        return;
      }
    }
  }

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < vectors_per_thread; i++)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < VecSize; j++)
    {
      thread_dst_ptr[(i * VectorStride * VecSize) + j] = items[(i * VecSize) + j];
    }
  }
}
} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

//! @rst
//! Store a blocked arrangement of items across a
//! thread block into a linear segment of items.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @blocked
//!
//! Each thread stores its items with the widest vectors that the alignment of ``block_ptr`` allows, up to 16 bytes
//! wide. On sm_100 and newer, vectors are up to 32 bytes wide.
//!
//! The following conditions will prevent vectorization and storing will fall back to cub::BLOCK_STORE_DIRECT:
//!
//! - ``block_ptr`` or the items of a thread do not allow vectors of at least 4 bytes that are wider than the alignment
//!   of ``T``
//! - ``T`` is volatile or not trivially copyable
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., ``(threadIdx.y * blockDim.x) + linear_tid`` for 2D thread blocks)
//!
//! @param[out] block_ptr
//!   Output pointer for storing to
//!
//! @param[in] items
//!   Data to store
template <typename T, int ItemsPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
StoreDirectBlockedVectorized(int linear_tid, T* block_ptr, T (&items)[ItemsPerThread])
{
  // The blocked arrangement is the striped arrangement of a block of a single thread, whose single vector is vector
  // linear_tid. The offset of the vector is unsigned, because a signed one is sign-extended before the tests, while the
  // first item of the thread is offset in signed arithmetic like in StoreDirectBlocked
  _CUB::detail::store_direct_vectorized<1, ItemsPerThread>(
    block_ptr,
    block_ptr,
    static_cast<unsigned int>(linear_tid),
    block_ptr + (linear_tid * ItemsPerThread), // NOLINT(bugprone-misplaced-widening-cast)
    items);
}

//! @}
//! @name Striped arrangement I/O (direct)
//! @{

//! @rst
//! Store a striped arrangement of data across the thread block into a
//! linear segment of items.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @striped
//! @endrst
//!
//! @tparam BlockThreads
//!   The thread block size in threads
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[in] items
//!   Data to store
template <int BlockThreads, typename T, int ItemsPerThread, typename OutputIteratorT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
StoreDirectStriped(int linear_tid, OutputIteratorT block_itr, T (&items)[ItemsPerThread])
{
  OutputIteratorT thread_itr = block_itr + linear_tid;

  // Store directly in striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ItemsPerThread; ITEM++)
  {
    thread_itr[(ITEM * BlockThreads)] = items[ITEM];
  }
}

//! @rst
//! Store a striped arrangement of data across the thread block into
//! a linear segment of items, guarded by range
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @striped
//! @endrst
//!
//! @tparam BlockThreads
//!   The thread block size in threads
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[in] items
//!   Data to store
//!
//! @param[in] valid_items
//!   Number of valid items to write
template <int BlockThreads, typename T, int ItemsPerThread, typename OutputIteratorT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
StoreDirectStriped(int linear_tid, OutputIteratorT block_itr, T (&items)[ItemsPerThread], int valid_items)
{
  OutputIteratorT thread_itr = block_itr + linear_tid;

  // Store directly in striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ItemsPerThread; ITEM++)
  {
    if ((ITEM * BlockThreads) + linear_tid < valid_items)
    {
      thread_itr[(ITEM * BlockThreads)] = items[ITEM];
    }
  }
}

//! @}
//! @name Warp-striped arrangement I/O (direct)
//! @{

//! @rst
//! Store a warp-striped arrangement of data across the
//! thread block into a linear segment of items.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @warpstriped
//!
//! Usage Considerations
//! ++++++++++++++++++++
//!
//! The number of threads in the thread block must be a multiple of the architecture's warp size.
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[out] items
//!   Data to load
template <typename T, int ItemsPerThread, typename OutputIteratorT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
StoreDirectWarpStriped(int linear_tid, OutputIteratorT block_itr, T (&items)[ItemsPerThread])
{
  const int tid         = linear_tid & (detail::warp_threads - 1);
  const int wid         = linear_tid >> detail::log2_warp_threads;
  const int warp_offset = wid * detail::warp_threads * ItemsPerThread;

  OutputIteratorT thread_itr = block_itr + warp_offset + tid;

  // Store directly in warp-striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ItemsPerThread; ITEM++)
  {
    thread_itr[(ITEM * detail::warp_threads)] = items[ITEM]; // NOLINT(bugprone-misplaced-widening-cast)
  }
}

//! @rst
//! Store a warp-striped arrangement of data across the thread block into a
//! linear segment of items, guarded by range
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @warpstriped
//!
//! Usage Considerations
//! ++++++++++++++++++++
//!
//! The number of threads in the thread block must be a multiple of the architecture's warp size.
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[in] items
//!   Data to store
//!
//! @param[in] valid_items
//!   Number of valid items to write
template <typename T, int ItemsPerThread, typename OutputIteratorT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
StoreDirectWarpStriped(int linear_tid, OutputIteratorT block_itr, T (&items)[ItemsPerThread], int valid_items)
{
  const int tid         = linear_tid & (detail::warp_threads - 1);
  const int wid         = linear_tid >> detail::log2_warp_threads;
  const int warp_offset = wid * detail::warp_threads * ItemsPerThread;

  OutputIteratorT thread_itr = block_itr + warp_offset + tid;

  // Store directly in warp-striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ItemsPerThread; ITEM++)
  {
    if (warp_offset + tid + (ITEM * detail::warp_threads) < valid_items)
    {
      thread_itr[(ITEM * detail::warp_threads)] = items[ITEM]; // NOLINT(bugprone-misplaced-widening-cast)
    }
  }
}

//! @}

//-----------------------------------------------------------------------------
// Generic BlockStore abstraction
//-----------------------------------------------------------------------------

//! cub::BlockStoreAlgorithm enumerates alternative algorithms for cub::BlockStore to write a
//! blocked arrangement of items across a CUDA thread block to a linear segment of memory.
enum BlockStoreAlgorithm // NOLINT(cppcoreguidelines-use-enum-class)
{
  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is written directly to memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) decreases as the
  //!   access stride between threads increases (i.e., the number items per thread).
  //!
  //! @endrst
  BLOCK_STORE_DIRECT,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of data is written directly to memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! The utilization of memory transactions (coalescing) remains high regardless
  //! of items written per thread.
  //!
  //! @endrst
  BLOCK_STORE_STRIPED,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is written directly
  //! to memory using CUDA's built-in vectorized stores as a coalescing optimization.
  //! For example, ``st.global.v4.s32`` instructions will be generated
  //! when ``T = int`` and ``ItemsPerThread % 4 == 0``.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high until the the
  //!   access stride between threads (i.e., the number items per thread) exceeds the
  //!   maximum vector store width (typically 4 items or 64B, whichever is lower).
  //! - Only sm_100 and newer store vectors of 32 bytes with a single access.
  //! - The following conditions will prevent vectorization and writing will fall back to cub::BLOCK_STORE_DIRECT:
  //!
  //!   - The ``OutputIteratorT`` is not a contiguous iterator
  //!   - The block output offset is not aligned to more than ``T`` and to at least 4 bytes, otherwise the widest
  //!     vectors that its alignment allows are used
  //!   - The output is volatile or ``T`` is not trivially copyable
  //!
  //! @endrst
  BLOCK_STORE_VECTORIZE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` is locally
  //! transposed and then efficiently written to memory as a :ref:`striped arrangement <flexible-data-arrangement>`.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless
  //!   of items written per thread.
  //! - The local reordering incurs slightly longer latencies and throughput than the
  //!   direct cub::BLOCK_STORE_DIRECT and cub::BLOCK_STORE_VECTORIZE alternatives.
  //!
  //! @endrst
  BLOCK_STORE_TRANSPOSE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` is locally
  //! transposed and then efficiently written to memory as a
  //! :ref:`warp-striped arrangement <flexible-data-arrangement>`.
  //!
  //! Usage Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - BLOCK_THREADS must be a multiple of WARP_THREADS
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless
  //!   of items written per thread.
  //! - The local reordering incurs slightly longer latencies and throughput than the
  //!   direct cub::BLOCK_STORE_DIRECT and cub::BLOCK_STORE_VECTORIZE alternatives.
  //!
  //! @endrst
  BLOCK_STORE_WARP_TRANSPOSE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` is locally
  //! transposed and then efficiently written to memory as a
  //! :ref:`warp-striped arrangement <flexible-data-arrangement>`.
  //! To reduce the shared memory requirement, only one warp's worth of shared
  //! memory is provisioned and is subsequently time-sliced among warps.
  //!
  //! Usage Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - BLOCK_THREADS must be a multiple of WARP_THREADS
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless
  //!   of items written per thread.
  //! - Provisions less shared memory temporary storage, but incurs larger
  //!   latencies than the BLOCK_STORE_WARP_TRANSPOSE alternative.
  //!
  //! @endrst
  BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
};

#if _CCCL_HOSTED() && !defined(_CCCL_DOXYGEN_INVOKED)
namespace detail
{
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const char* to_string(BlockStoreAlgorithm algo) noexcept
{
  switch (algo)
  {
    case BLOCK_STORE_DIRECT:
      return "BLOCK_STORE_DIRECT";
    case BLOCK_STORE_STRIPED:
      return "BLOCK_STORE_STRIPED";
    case BLOCK_STORE_VECTORIZE:
      return "BLOCK_STORE_VECTORIZE";
    case BLOCK_STORE_TRANSPOSE:
      return "BLOCK_STORE_TRANSPOSE";
    case BLOCK_STORE_WARP_TRANSPOSE:
      return "BLOCK_STORE_WARP_TRANSPOSE";
    case BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED:
      return "BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED";
  }
  return "<unknown BlockStoreAlgorithm>";
}
} // namespace detail

_CCCL_HOST_API inline ::std::ostream& operator<<(::std::ostream& os, BlockStoreAlgorithm algo)
{
  return os << _CUB::detail::to_string(algo);
}
#endif // _CCCL_HOSTED() && !_CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END

#if __cpp_lib_format >= 201907L && !defined(_CCCL_DOXYGEN_INVOKED)
template <::cuda::std::same_as<char> CharT>
struct std::formatter<_CUB::BlockStoreAlgorithm, CharT> : formatter<const CharT*, CharT>
{
  template <class FmtCtx>
  _CCCL_HOST_API auto format(const _CUB::BlockStoreAlgorithm& algo, FmtCtx& ctx) const
  {
    return formatter<const CharT*, CharT>::format(_CUB::detail::to_string(algo), ctx);
  }
};
#endif // __cpp_lib_format >= 201907L && !defined(_CCCL_DOXYGEN_INVOKED)

CUB_NAMESPACE_BEGIN

//! @rst
//! The BlockStore class provides :ref:`collective <collective-primitives>` data movement
//! methods for writing a :ref:`blocked arrangement <flexible-data-arrangement>` of items
//! partitioned across a CUDA thread block to a linear segment of memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - The BlockStore class provides a single data movement abstraction that can be specialized
//!   to implement different cub::BlockStoreAlgorithm strategies. This facilitates different
//!   performance policies for different architectures, data types, granularity sizes, etc.
//! - BlockStore can be optionally specialized by different data movement strategies:
//!
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_DIRECT`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` of data is written directly to memory.
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_STRIPED`:
//!      A :ref:`striped arrangement <flexible-data-arrangement>` of data is written directly to memory.
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_VECTORIZE`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` of data is written directly to memory
//!      using CUDA's built-in vectorized stores as a coalescing optimization.
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_TRANSPOSE`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` is locally transposed into
//!      a :ref:`striped arrangement <flexible-data-arrangement>` which is then written to memory.
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_WARP_TRANSPOSE`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` is locally transposed into
//!      a :ref:`warp-striped arrangement <flexible-data-arrangement>` which is then written to memory.
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` is locally transposed into
//!      a :ref:`warp-striped arrangement <flexible-data-arrangement>` which is then written to memory.
//!      To reduce the shared memory requireent, only one warp's worth of shared memory is provisioned and is
//!      subsequently time-sliced among warps.
//!
//! - @rowmajor
//!
//! A Simple Example
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @blockcollective{BlockStore}
//!
//! The code snippet below illustrates the storing of a "blocked" arrangement
//! of 512 integers across 128 threads (where each thread owns 4 consecutive items)
//! into a linear segment of memory. The store is specialized for ``BLOCK_STORE_WARP_TRANSPOSE``,
//! meaning items are locally reordered among threads so that memory references will be
//! efficiently coalesced using a warp-striped access pattern.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_store.cuh>
//!
//!    __global__ void ExampleKernel(int *d_data, ...)
//!    {
//!        // Specialize BlockStore for a 1D block of 128 threads owning 4 integer items each
//!        using BlockStore = cub::BlockStore<int, 128, 4, BLOCK_STORE_WARP_TRANSPOSE>;
//!
//!        // Allocate shared memory for BlockStore
//!        __shared__ typename BlockStore::TempStorage temp_storage;
//!
//!        // Obtain a segment of consecutive items that are blocked across threads
//!        int thread_data[4];
//!        ...
//!
//!        // Store items to linear memory
//!        BlockStore(temp_storage).Store(d_data, thread_data);
//!    }
//!
//! Suppose the set of ``thread_data`` across the block of threads is
//! ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
//! The output ``d_data`` will be ``0, 1, 2, 3, 4, 5, ...``.
//!
//! Re-using dynamically allocating shared memory
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The ``block/example_block_reduce_dyn_smem.cu`` example illustrates usage of
//! dynamically shared memory with BlockReduce and how to re-purpose the same memory region.
//! This example can be easily adapted to the storage required by BlockStore.
//!
//! @endrst
//!
//! @tparam T
//!   The type of data to be written.
//!
//! @tparam BlockDimX
//!   The thread block length in threads along the X dimension
//!
//! @tparam ItemsPerThread
//!   The number of consecutive items partitioned onto each thread.
//!
//! @tparam Algorithm
//!   **[optional]** cub::BlockStoreAlgorithm tuning policy enumeration (default: cub::BLOCK_STORE_DIRECT)
//!
//! @tparam BlockDimY
//!   **[optional]** The thread block length in threads along the Y dimension (default: 1)
//!
//! @tparam BlockDimZ
//!   **[optional]** The thread block length in threads along the Z dimension (default: 1)
//!
template <typename T,
          int BlockDimX,
          int ItemsPerThread,
          BlockStoreAlgorithm Algorithm = BLOCK_STORE_DIRECT,
          int BlockDimY                 = 1,
          int BlockDimZ                 = 1>
class BlockStore
{
  /// The thread block size in threads
  static constexpr int BLOCK_THREADS = BlockDimX * BlockDimY * BlockDimZ;

  // transposing store algorithms need a BlockExchange
  using block_exchange =
    BlockExchange<T, BlockDimX, ItemsPerThread, Algorithm == BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED, BlockDimY, BlockDimZ>;

  static_assert((Algorithm != BLOCK_STORE_WARP_TRANSPOSE && Algorithm != BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED)
                  || (BLOCK_THREADS % detail::warp_threads == 0),
                "Threads per block must be a multiple of warp_threads for this BlockStoreAlgorithm");

  _CCCL_HOST_DEVICE_API static constexpr auto temp_storage_helper()
  {
    if constexpr (Algorithm == BLOCK_STORE_DIRECT || Algorithm == BLOCK_STORE_STRIPED
                  || Algorithm == BLOCK_STORE_VECTORIZE)
    {
      return NullType{};
    }
    else if constexpr (Algorithm == BLOCK_STORE_TRANSPOSE || Algorithm == BLOCK_STORE_WARP_TRANSPOSE
                       || Algorithm == BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED)
    {
      struct _TempStorage : block_exchange::TempStorage
      {
        volatile int valid_items; // Temporary storage for partially-full block guard
      };
      return _TempStorage{};
    }
  }

  using _TempStorage _CCCL_NODEBUG = decltype(temp_storage_helper());

  _CCCL_DEVICE_API _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  _TempStorage& temp_storage;

  int linear_tid;

public:
  //! @smemstorage{BlockStore}
  using TempStorage = Uninitialized<_TempStorage>;

  //! @name Collective constructors
  //! @{

  /**
   * @brief Collective constructor using a private static allocation of shared memory as temporary storage.
   *
   * @rst
   * .. versionadded:: 2.2.0
   *    First appears in CUDA Toolkit 12.3.
   * @endrst
   */
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockStore()
      : temp_storage(PrivateStorage())
      , linear_tid(_CUB::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  /**
   * @brief Collective constructor using the specified memory allocation as temporary storage.
   *
   * @rst
   * .. versionadded:: 2.2.0
   *    First appears in CUDA Toolkit 12.3.
   * @endrst
   *
   * @param[in] temp_storage
   *   Reference to memory allocation having layout type TempStorage
   */
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockStore(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(_CUB::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @}
  //! @name Data movement
  //! @{

  //! @rst
  //! Store items into a linear segment of memory
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the storing of a "blocked" arrangement
  //! of 512 integers across 128 threads (where each thread owns 4 consecutive items)
  //! into a linear segment of memory. The store is specialized for ``BLOCK_STORE_WARP_TRANSPOSE``,
  //! meaning items are locally reordered among threads so that memory references will be
  //! efficiently coalesced using a warp-striped access pattern.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_store.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, ...)
  //!    {
  //!        // Specialize BlockStore for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockStore = cub::BlockStore<int, 128, 4, BLOCK_STORE_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockStore
  //!        __shared__ typename BlockStore::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Store items to linear memory
  //!        BlockStore(temp_storage).Store(d_data, thread_data);
  //!    }
  //!
  //! Suppose the set of ``thread_data`` across the block of threads is
  //! ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
  //! The output ``d_data`` will be ``0, 1, 2, 3, 4, 5, ...``.
  //! @endrst
  //!
  //! @param[out] block_itr
  //!   The thread block's base output iterator for storing to
  //!
  //! @param[in] items
  //!   Data to store
  template <typename OutputIteratorT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ItemsPerThread])
  {
    if constexpr (Algorithm == BLOCK_STORE_DIRECT)
    {
      _CUB::StoreDirectBlocked(linear_tid, block_itr, items);
    }
    else if constexpr (Algorithm == BLOCK_STORE_STRIPED)
    {
      _CUB::StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items);
    }
    else if constexpr (Algorithm == BLOCK_STORE_VECTORIZE)
    {
      if constexpr (detail::block_store_can_vectorize_v<OutputIteratorT, T>)
      {
        _CUB::StoreDirectBlockedVectorized(linear_tid, ::cuda::std::to_address(block_itr), items);
      }
      else
      {
        _CUB::StoreDirectBlocked(linear_tid, block_itr, items);
      }
    }
    else if constexpr (Algorithm == BLOCK_STORE_TRANSPOSE)
    {
      block_exchange(temp_storage).BlockedToStriped(items);
      _CUB::StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items);
    }
    else if constexpr (Algorithm == BLOCK_STORE_WARP_TRANSPOSE || Algorithm == BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED)
    {
      block_exchange(temp_storage).BlockedToWarpStriped(items);
      _CUB::StoreDirectWarpStriped(linear_tid, block_itr, items);
    }
  }

  //! @rst
  //! Store items into a linear segment of memory, guarded by range.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the guarded storing of a "blocked" arrangement
  //! of 512 integers across 128 threads (where each thread owns 4 consecutive items)
  //! into a linear segment of memory. The store is specialized for ``BLOCK_STORE_WARP_TRANSPOSE``,
  //! meaning items are locally reordered among threads so that memory references will be
  //! efficiently coalesced using a warp-striped access pattern.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_store.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int valid_items, ...)
  //!    {
  //!        // Specialize BlockStore for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockStore = cub::BlockStore<int, 128, 4, BLOCK_STORE_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockStore
  //!        __shared__ typename BlockStore::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Store items to linear memory
  //!        BlockStore(temp_storage).Store(d_data, thread_data, valid_items);
  //!    }
  //!
  //! Suppose the set of ``thread_data`` across the block of threads is
  //! ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }`` and ``valid_items`` is ``5``.
  //! The output ``d_data`` will be ``0, 1, 2, 3, 4, ?, ?, ?, ...``, with
  //! only the first two threads being unmasked to store portions of valid data.
  //! @endrst
  //!
  //! @param[out] block_itr
  //!   The thread block's base output iterator for storing to
  //!
  //! @param[in] items
  //!   Data to store
  //!
  //! @param[in] valid_items
  //!   Number of valid items to write
  template <typename OutputIteratorT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ItemsPerThread], int valid_items)
  {
    if constexpr (Algorithm == BLOCK_STORE_DIRECT || Algorithm == BLOCK_STORE_VECTORIZE)
    {
      _CUB::StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
    }
    else if constexpr (Algorithm == BLOCK_STORE_STRIPED)
    {
      _CUB::StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items, valid_items);
    }
    else if constexpr (Algorithm == BLOCK_STORE_TRANSPOSE)
    {
      block_exchange(temp_storage).BlockedToStriped(items);
      if (linear_tid == 0)
      {
        // Move through volatile smem as a workaround to prevent RF spilling on subsequent loads
        temp_storage.valid_items = valid_items;
      }
      __syncthreads();
      _CUB::StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items, temp_storage.valid_items);
    }
    else if constexpr (Algorithm == BLOCK_STORE_WARP_TRANSPOSE || Algorithm == BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED)
    {
      block_exchange(temp_storage).BlockedToWarpStriped(items);
      if (linear_tid == 0)
      {
        // Move through volatile smem as a workaround to prevent RF spilling on subsequent loads
        temp_storage.valid_items = valid_items;
      }
      __syncthreads();
      _CUB::StoreDirectWarpStriped(linear_tid, block_itr, items, temp_storage.valid_items);
    }
  }

  //! @}
};

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
template <class Policy, class It, class T = cub::detail::it_value_t<It>>
struct BlockStoreType
{
  using type _CCCL_NODEBUG =
    cub::BlockStore<T, Policy::BLOCK_THREADS, Policy::ITEMS_PER_THREAD, Policy::STORE_ALGORITHM>;
};
#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
