// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/round_up.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstddef>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! Identifies the algorithm whose shared memory requirements are queried through `required_smem_layout`.
//!
//! Commented out enumerators do not have a fixed storage type: their shared memory is a function of a stage count, or
//! their internal shuffle and shared-memory strategies are selected by the collective that calls them.
enum class cub_algorithm
{
  thread_load,
  thread_store,
  thread_reduce,
  thread_scan,
  thread_sort,
  thread_search,

  // The warp-scope collectives below pick their storage from their own parameters, so their internal shfl/smem
  // strategies never need to be queried separately.
  // warp_exchange_smem,
  // warp_exchange_shfl,
  warp_exchange,
  warp_load,
  warp_store,
  // warp_reduce_smem,
  // warp_reduce_shfl,
  warp_reduce,
  // warp_scan_smem,
  // warp_scan_shfl,
  warp_scan,
  // warp_reduce_batched_wspro,
  // warp_reduce_batched,
  warp_merge_sort,
  // warp_bitonic_sort,

  block_raking_layout,
  block_exchange,
  block_load,
  block_store,
  block_shuffle,
  block_discontinuity,
  block_adjacent_difference,
  block_load_to_shared,
  block_scan_raking,
  block_scan_warp_scans,
  block_scan,
  block_reduce_raking,
  block_reduce_raking_commutative_only,
  block_reduce_warp_reductions,
  block_reduce,
  block_radix_rank_basic,
  block_radix_rank_match,
  block_radix_rank_match_early_counts,
  block_radix_rank,
  block_radix_sort,
  block_histogram_atomic,
  block_histogram_sort,
  block_histogram,
  block_run_length_decode,
  block_merge_sort_strategy,
  block_merge_sort,
  block_topk_air,
  block_topk,

  tile_prefix_callback,
  agent_scan,
  agent_reduce,
  agent_warp_reduce,
  agent_reduce_by_key,
  agent_scan_by_key,
  agent_select_if,
  agent_unique_by_key,
  agent_three_way_partition,
  agent_rle,
  agent_difference,
  agent_difference_init,
  agent_find,
  agent_find_bound_sorted_values,
  agent_histogram,
  agent_topk,
  agent_batched_topk,
  agent_batched_topk_cluster,
  agent_batch_memcpy,
  agent_merge,
  agent_block_sort,
  agent_merge_sort_partition,
  agent_merge_sort_merge,
  agent_radix_sort_upsweep,
  agent_radix_sort_downsweep,
  agent_radix_sort_histogram,
  agent_radix_sort_onesweep,
  agent_segmented_radix_sort,
  agent_sub_warp_merge_sort,
  agent_for,
  agent_segmented_scan,

  // Kernels assemble their shared memory from the agents they launch rather than from a storage type of their own, so
  // there is nothing to specialize for them yet.
  // kernel_transform,
  // kernel_rle_encode_lookahead,
  // kernel_scan_lookahead,
  // kernel_batched_topk_sort,
  // kernel_segmented_reduce,
  // kernel_segmented_sort,
};

struct smem_layout
{
  ::cuda::std::size_t size;
  ::cuda::std::size_t alignment;
};

inline constexpr smem_layout no_smem{1, 1};

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr smem_layout array_layout(smem_layout element, ::cuda::std::size_t count)
{
  return count == 0 ? no_smem
                    : smem_layout{::cuda::round_up(element.size, element.alignment) * count, element.alignment};
}

template <typename T>
inline constexpr smem_layout type_layout{sizeof(T), alignof(T)};

template <typename... Layouts>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr smem_layout union_layout(Layouts... layouts)
{
  ::cuda::std::size_t size      = 0;
  ::cuda::std::size_t alignment = 1;
  ((size      = size < layouts.size ? layouts.size : size,
    alignment = alignment < layouts.alignment ? layouts.alignment : alignment),
   ...);
  return {::cuda::round_up(size, alignment), alignment};
}

template <typename... Layouts>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr smem_layout struct_layout(Layouts... layouts)
{
  ::cuda::std::size_t size      = 0;
  ::cuda::std::size_t alignment = 1;
  ((size      = ::cuda::round_up(size, layouts.alignment) + layouts.size,
    alignment = alignment < layouts.alignment ? layouts.alignment : alignment),
   ...);
  return {::cuda::round_up(size, alignment), alignment};
}

template <::cuda::std::size_t Alignment>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr smem_layout aligned_layout(smem_layout layout)
{
  const auto alignment = layout.alignment < Alignment ? Alignment : layout.alignment;
  return {::cuda::round_up(layout.size, alignment), alignment};
}

// `Uninitialized<T>` preserves `sizeof(T)`, but its `UnitWord<T>::DeviceWord` backing storage is at most 16-byte
// aligned.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr smem_layout uninitialized_layout(smem_layout layout)
{
  constexpr ::cuda::std::size_t max_device_word_alignment = 16;
  return {layout.size, layout.alignment < max_device_word_alignment ? layout.alignment : max_device_word_alignment};
}

template <cub_algorithm Algorithm, typename... Args>
struct required_smem_layout
{
  static_assert(::cuda::std::__always_false_v<::cuda::std::integral_constant<cub_algorithm, Algorithm>>,
                "Unknown required_smem_layout requested. This needs to be specialized by the algorithm author");
  static constexpr detail::smem_layout value = no_smem;
};

template <cub_algorithm Algorithm, typename... Args>
inline constexpr smem_layout required_smem_layout_v = required_smem_layout<Algorithm, Args...>::value;

template <cub_algorithm Algorithm, typename... Args>
inline constexpr ::cuda::std::size_t required_smem_v = required_smem_layout_v<Algorithm, Args...>.size;
} // namespace detail

CUB_NAMESPACE_END
