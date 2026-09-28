// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_merge_sort.cuh>
#include <cub/detail/cc_dispatch.cuh>
#include <cub/detail/required_smem.cuh>
#include <cub/device/dispatch/tuning/tuning_merge_sort.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_vsmem.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::merge_sort
{
template <typename DefaultPolicyGetter>
struct fallback_policy_getter
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE constexpr auto operator()() const
  {
    MergeSortPolicy policy   = DefaultPolicyGetter{}();
    policy.threads_per_block = 64;
    policy.items_per_thread  = 1;
    return policy;
  }
};

//! @brief Helper class template for merge sort-specific virtual shared memory handling. The merge sort algorithm in
//! its current implementation relies on the fact that both the sorting as well as the merging kernels use the same tile
//! size. This circumstance needs to be respected when determining whether the fallback policy for large user types is
//! applicable: we must either use the fallback for both or for none of the two agents.
template <typename DefaultPolicyGetter,
          typename FallbackPolicyGetter,
          typename KeyInIt,
          typename ValInIt,
          typename KeyOutIt,
          typename ValOutIt,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
class merge_sort_vsmem_helper_impl
{
  // Use fallback if either (a) the default block sort or (b) the block merge agent exceed the maximum shared memory
  // available per block and both (1) the fallback block sort and (2) the fallback merge agent would not exceed the
  // available shared memory
  template <typename PolicyGetter>
  static constexpr auto required_smem =
    (::cuda::std::
       max) (required_smem_v<cub_algorithm::agent_block_sort, PolicyGetter, KeyInIt, ValInIt, KeyOutIt, ValOutIt, KeyT, ValueT>,
             required_smem_v<cub_algorithm::agent_merge_sort_merge, PolicyGetter, KeyOutIt, ValOutIt, KeyT, ValueT>);

  static constexpr auto max_default_size     = required_smem<DefaultPolicyGetter>;
  static constexpr auto max_fallback_size    = required_smem<FallbackPolicyGetter>;
  static constexpr bool uses_fallback_policy = use_fallback_smem(max_default_size, max_fallback_size);

public:
  using active_policy_getter_t = ::cuda::std::_If<uses_fallback_policy, FallbackPolicyGetter, DefaultPolicyGetter>;

  static constexpr MergeSortPolicy policy = uses_fallback_policy ? FallbackPolicyGetter{}() : DefaultPolicyGetter{}();

  static constexpr ::cuda::std::size_t block_sort_required_smem =
    required_smem_v<cub_algorithm::agent_block_sort,
                    active_policy_getter_t,
                    KeyInIt,
                    ValInIt,
                    KeyOutIt,
                    ValOutIt,
                    KeyT,
                    ValueT>;
  static constexpr ::cuda::std::size_t merge_required_smem =
    required_smem_v<cub_algorithm::agent_merge_sort_merge, active_policy_getter_t, KeyOutIt, ValOutIt, KeyT, ValueT>;

  // The selected agent only. Kernels alias these types so the function body does not respecialize them.
  using block_sort_agent_t =
    AgentBlockSort<active_policy_getter_t, KeyInIt, ValInIt, KeyOutIt, ValOutIt, OffsetT, CompareOpT, KeyT, ValueT>;
  using merge_agent_t = AgentMerge<active_policy_getter_t, KeyOutIt, ValOutIt, OffsetT, CompareOpT, KeyT, ValueT>;
};

template <typename PolicyGetter,
          typename KeyInIt,
          typename ValInIt,
          typename KeyOutIt,
          typename ValOutIt,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
using merge_sort_vsmem_helper_t = merge_sort_vsmem_helper_impl<
  PolicyGetter,
  fallback_policy_getter<PolicyGetter>,
  KeyInIt,
  ValInIt,
  KeyOutIt,
  ValOutIt,
  OffsetT,
  CompareOpT,
  KeyT,
  ValueT>;

template <typename PolicyGetter,
          typename KeyInIt,
          typename ValInIt,
          typename KeyOutIt,
          typename ValOutIt,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
using device_merge_sort_vsmem_helper_t = merge_sort_vsmem_helper_impl<
  PolicyGetter,
  fallback_policy_getter<PolicyGetter>,
  KeyInIt,
  ValInIt,
  KeyOutIt,
  ValOutIt,
  OffsetT,
  CompareOpT,
  KeyT,
  ValueT>;

template <typename PolicySelectorT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
__launch_bounds__(
  device_merge_sort_vsmem_helper_t<
    device_policy_getter<PolicySelectorT, current_tuning_cc().get()>,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>::policy.threads_per_block)
  _CCCL_KERNEL_ATTRIBUTES void DeviceMergeSortBlockSortKernel(
    const bool ping,
    const KeyInputIteratorT keys_in,
    const ValueInputIteratorT items_in,
    const KeyIteratorT keys_out,
    const ValueIteratorT items_out,
    const OffsetT keys_count,
    KeyT* const tmp_keys_out,
    ValueT* const tmp_items_out,
    CompareOpT compare_op,
    vsmem_t vsmem)
{
  using vsmem_adapted_agents = device_merge_sort_vsmem_helper_t<
    device_policy_getter<PolicySelectorT, current_tuning_cc().get()>,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;

  static constexpr MergeSortPolicy active_policy = vsmem_adapted_agents::policy;
  using agent_block_sort_t                       = typename vsmem_adapted_agents::block_sort_agent_t;
  using vsmem_helper_t =
    agent_block_smem<typename agent_block_sort_t::TempStorage, vsmem_adapted_agents::block_sort_required_smem>;

  // Static shared memory allocation
  __shared__ typename vsmem_helper_t::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename agent_block_sort_t::TempStorage& temp_storage = vsmem_helper_t::get_temp_storage(static_temp_storage, vsmem);

  agent_block_sort_t agent(
    ping,
    temp_storage,
    try_make_cache_modified_iterator<active_policy.load_modifier>(keys_in),
    try_make_cache_modified_iterator<active_policy.load_modifier>(items_in),
    keys_count,
    keys_out,
    items_out,
    tmp_keys_out,
    tmp_items_out,
    compare_op);

  agent.Process();

  // If applicable, hints to discard modified cache lines for vsmem
  vsmem_helper_t::discard_temp_storage(temp_storage);
}

template <typename KeyIteratorT, typename OffsetT, typename CompareOpT, typename KeyT>
_CCCL_KERNEL_ATTRIBUTES void DeviceMergeSortPartitionKernel(
  const bool ping,
  const KeyIteratorT keys_ping,
  KeyT* const keys_pong,
  const OffsetT keys_count,
  const OffsetT num_partitions,
  OffsetT* const merge_partitions,
  CompareOpT compare_op,
  const OffsetT target_merged_tiles_number,
  const int items_per_tile)
{
  const OffsetT partition_idx =
    static_cast<OffsetT>(blockDim.x * blockIdx.x + threadIdx.x); // NOLINT(bugprone-misplaced-widening-cast)
  if (partition_idx < num_partitions)
  {
    AgentPartition<KeyIteratorT, OffsetT, CompareOpT, KeyT>{
      ping,
      keys_ping,
      keys_pong,
      keys_count,
      partition_idx,
      merge_partitions,
      compare_op,
      target_merged_tiles_number,
      items_per_tile,
      num_partitions}
      .Process();
  }
}

template <typename PolicySelectorT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
__launch_bounds__(
  device_merge_sort_vsmem_helper_t<
    device_policy_getter<PolicySelectorT, current_tuning_cc().get()>,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>::policy.threads_per_block)
  _CCCL_KERNEL_ATTRIBUTES void DeviceMergeSortMergeKernel(
    const bool ping,
    const KeyIteratorT keys_ping,
    const ValueIteratorT items_ping,
    const OffsetT keys_count,
    KeyT* const keys_pong,
    ValueT* const items_pong,
    CompareOpT compare_op,
    OffsetT* const merge_partitions,
    const OffsetT target_merged_tiles_number,
    vsmem_t vsmem)
{
  using vsmem_adapted_agents = device_merge_sort_vsmem_helper_t<
    device_policy_getter<PolicySelectorT, current_tuning_cc().get()>,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;

  static constexpr MergeSortPolicy active_policy = vsmem_adapted_agents::policy;
  using agent_merge_t                            = typename vsmem_adapted_agents::merge_agent_t;
  using vsmem_helper_t =
    agent_block_smem<typename agent_merge_t::TempStorage, vsmem_adapted_agents::merge_required_smem>;

  // Static shared memory allocation
  __shared__ typename vsmem_helper_t::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename agent_merge_t::TempStorage& temp_storage = vsmem_helper_t::get_temp_storage(static_temp_storage, vsmem);

  agent_merge_t agent(
    ping,
    temp_storage,
    try_make_cache_modified_iterator<active_policy.load_modifier>(keys_ping),
    try_make_cache_modified_iterator<active_policy.load_modifier>(items_ping),
    try_make_cache_modified_iterator<active_policy.load_modifier>(keys_pong),
    try_make_cache_modified_iterator<active_policy.load_modifier>(items_pong),
    keys_count,
    keys_pong,
    items_pong,
    keys_ping,
    items_ping,
    compare_op,
    merge_partitions,
    target_merged_tiles_number);

  agent.Process();

  // If applicable, hints to discard modified cache lines for vsmem
  vsmem_helper_t::discard_temp_storage(temp_storage);
}
} // namespace detail::merge_sort

CUB_NAMESPACE_END
