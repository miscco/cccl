// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/agent/agent_adjacent_difference.cuh>
#include <cub/agent/agent_batched_topk.cuh>
#include <cub/agent/agent_batched_topk_cluster.cuh>
#include <cub/agent/agent_merge.cuh>
#include <cub/agent/agent_merge_sort.cuh>
#include <cub/agent/agent_radix_sort_histogram.cuh>
#include <cub/agent/agent_radix_sort_onesweep.cuh>
#include <cub/agent/agent_reduce.cuh>
#include <cub/agent/agent_reduce_by_key.cuh>
#include <cub/agent/agent_select_if.cuh>
#include <cub/agent/agent_unique_by_key.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>

#include <cuda/argument>
#include <cuda/std/functional>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>

namespace
{
using cub::detail::cub_algorithm;
using cub::detail::required_smem_layout_v;
using cub::detail::required_smem_v;

template <auto Value>
using constant_t = ::cuda::std::integral_constant<decltype(Value), Value>;

using reduce_policy = cub::detail::
  agent_reduce_policy<128, 4, int, 1, cub::BLOCK_REDUCE_RAKING, cub::LOAD_DEFAULT, cub::detail::NoScaling<128, 4, int>>;
using reduce_agent = cub::detail::reduce::AgentReduce<reduce_policy, int*, int, ::cuda::std::plus<>, int>;

using difference_policy = cub::detail::
  agent_adjacent_difference_policy<128, 4, cub::BLOCK_LOAD_TRANSPOSE, cub::LOAD_DEFAULT, cub::BLOCK_STORE_TRANSPOSE>;
using difference_agent = cub::detail::adjacent_difference::
  AgentDifference<difference_policy, int*, int*, ::cuda::std::minus<>, int, int, int, false, true>;

using merge_agent = cub::detail::merge::agent_t<
  128,
  4,
  cub::LOAD_DEFAULT,
  cub::BLOCK_STORE_TRANSPOSE,
  false,
  false,
  true,
  int*,
  int*,
  int*,
  int*,
  int*,
  int*,
  int,
  ::cuda::std::less<>>;

using unique_policy = cub::detail::
  agent_unique_by_key_policy<128, 4, cub::BLOCK_LOAD_TRANSPOSE, cub::LOAD_DEFAULT, cub::BLOCK_SCAN_WARP_SCANS>;
using unique_agent =
  cub::detail::unique_by_key::AgentUniqueByKey<unique_policy, int*, int*, int*, int*, ::cuda::std::equal_to<>, int>;

using select_if_policy =
  cub::detail::agent_select_if_policy<128, 4, cub::BLOCK_LOAD_TRANSPOSE, cub::LOAD_DEFAULT, cub::BLOCK_SCAN_WARP_SCANS>;
using select_if_agent = cub::detail::select::AgentSelectIf<
  select_if_policy,
  int*,
  cub::NullType*,
  int*,
  ::cuda::std::identity,
  cub::NullType,
  int,
  cub::NullType,
  cub::SelectImpl::Select>;

using reduce_by_key_policy = cub::detail::
  agent_reduce_by_key_policy<128, 4, cub::BLOCK_LOAD_TRANSPOSE, cub::LOAD_DEFAULT, cub::BLOCK_SCAN_WARP_SCANS>;
using reduce_by_key_agent = cub::detail::reduce_by_key::AgentReduceByKey<
  reduce_by_key_policy,
  int*,
  int*,
  int*,
  int*,
  int*,
  ::cuda::std::equal_to<>,
  ::cuda::std::plus<>,
  int,
  int,
  cub::NullType>;

template <cub::RadixRankAlgorithm RankAlgorithm>
using onesweep_policy = cub::detail::agent_radix_sort_onesweep_policy<
  0,
  0,
  void,
  1,
  RankAlgorithm,
  cub::BLOCK_SCAN_RAKING_MEMOIZE,
  cub::RADIX_SORT_STORE_DIRECT,
  8,
  cub::detail::NoScaling<128, 4>>;

template <cub::RadixRankAlgorithm RankAlgorithm>
using onesweep_agent =
  cub::detail::radix_sort::AgentRadixSortOnesweep<onesweep_policy<RankAlgorithm>, false, int, long long, int, int>;

using histogram_policy = cub::detail::agent_radix_sort_histogram_policy<128, 16, 1, void, 8>;
using histogram_agent  = cub::detail::radix_sort::AgentRadixSortHistogram<histogram_policy, false, int, int>;

struct merge_sort_policy_getter
{
  _CCCL_HOST_DEVICE_API constexpr cub::MergeSortPolicy operator()() const
  {
    return {128, 4, cub::BLOCK_LOAD_TRANSPOSE, cub::LOAD_DEFAULT, cub::BLOCK_STORE_TRANSPOSE, true};
  }
};

using block_sort_agent = cub::detail::merge_sort::
  AgentBlockSort<merge_sort_policy_getter, int*, long long*, int*, long long*, int, ::cuda::std::less<>, int, long long>;
using merge_sort_agent = cub::detail::merge_sort::
  AgentMerge<merge_sort_policy_getter, int*, long long*, int, ::cuda::std::less<>, int, long long>;

struct batched_topk_policy
{
  cub::detail::batched_topk::worker_policy worker_per_segment_policy;
  cub::detail::batched_topk::multi_worker_policy multi_worker_per_segment_policy;
};

struct batched_topk_policy_getter
{
  _CCCL_HOST_DEVICE_API constexpr batched_topk_policy operator()() const
  {
    return {{128,
             4,
             cub::BLOCK_LOAD_TRANSPOSE,
             cub::BLOCK_STORE_TRANSPOSE,
             {2, cub::BLOCK_LOAD_TRANSPOSE, cub::BLOCK_STORE_TRANSPOSE, cub::BLOCK_SCAN_WARP_SCANS}},
            {128, 4}};
  }
};

using segment_size_arg     = decltype(::cuda::args::constant<512>{});
using k_arg                = decltype(::cuda::args::constant<16>{});
using select_direction_arg = decltype(::cuda::args::constant<cub::detail::topk::select::max>{});
using num_segments_arg     = decltype(::cuda::args::constant<1>{});

using batched_topk_agent = cub::detail::batched_topk::agent_batched_topk_worker_per_segment<
  batched_topk_policy_getter,
  int**,
  int**,
  long long**,
  long long**,
  segment_size_arg,
  k_arg,
  select_direction_arg,
  num_segments_arg,
  int>;

struct cluster_topk_policy
{
  int threads_per_block;
  int min_blocks_per_sm;
  int min_chunks_per_block;
  int chunk_bytes;
  int load_align_bytes;
  int pipeline_stages;
  int single_block_max_seg_size;
  int bits_per_pass;
  int histogram_items_per_thread;
  int tie_break_items_per_thread;
  int copy_items_per_thread;
  int max_blocks_per_cluster;
  int max_chunk_slots_per_block;
};

struct cluster_topk_policy_getter
{
  _CCCL_HOST_DEVICE_API constexpr cluster_topk_policy operator()() const
  {
    return {128, 1, 1, 256, 16, 2, 2048, 4, 4, 4, 4, 0, 0};
  }
};

using cluster_topk_agent = cub::detail::batched_topk_cluster::agent_batched_topk_cluster<
  cluster_topk_policy_getter,
  ::cuda::execution::determinism::__determinism_t::__not_guaranteed,
  ::cuda::execution::tie_break::__tie_break_t::__unspecified,
  int**,
  int**,
  long long**,
  long long**,
  segment_size_arg,
  k_arg,
  select_direction_arg,
  num_segments_arg>;

// The queries below are spelled with the very template arguments of the agent they describe, never with the agent type
// itself: computing the required shared memory must never instantiate an agent.
inline constexpr auto reduce_layout = required_smem_layout_v<cub_algorithm::agent_reduce, reduce_policy, int>;

inline constexpr auto difference_layout =
  required_smem_layout_v<cub_algorithm::agent_difference, difference_policy, int, int>;

inline constexpr auto merge_layout = required_smem_layout_v<
  cub_algorithm::agent_merge,
  constant_t<128>,
  constant_t<4>,
  constant_t<cub::BLOCK_STORE_TRANSPOSE>,
  constant_t<false>,
  constant_t<false>,
  int*,
  int*>;

inline constexpr auto unique_layout =
  required_smem_layout_v<cub_algorithm::agent_unique_by_key, unique_policy, int*, int*, int>;

inline constexpr auto select_if_layout =
  required_smem_layout_v<cub_algorithm::agent_select_if, select_if_policy, int*, cub::NullType*, int>;

inline constexpr auto reduce_by_key_layout =
  required_smem_layout_v<cub_algorithm::agent_reduce_by_key, reduce_by_key_policy, int*, int*, int, int>;

template <cub::RadixRankAlgorithm RankAlgorithm>
inline constexpr auto onesweep_layout =
  required_smem_layout_v<cub_algorithm::agent_radix_sort_onesweep, onesweep_policy<RankAlgorithm>, int, long long, int, int>;

inline constexpr auto histogram_layout =
  required_smem_layout_v<cub_algorithm::agent_radix_sort_histogram, histogram_policy, int>;

inline constexpr auto block_sort_layout = required_smem_layout_v<
  cub_algorithm::agent_block_sort,
  merge_sort_policy_getter,
  int*,
  long long*,
  int*,
  long long*,
  int,
  long long>;

inline constexpr auto merge_sort_layout =
  required_smem_layout_v<cub_algorithm::agent_merge_sort_merge, merge_sort_policy_getter, int*, long long*, int, long long>;

inline constexpr auto batched_topk_layout =
  required_smem_layout_v<cub_algorithm::agent_batched_topk,
                         batched_topk_policy_getter,
                         int**,
                         long long**,
                         segment_size_arg>;

inline constexpr auto cluster_topk_layout =
  required_smem_layout_v<cub_algorithm::agent_batched_topk_cluster, cluster_topk_policy_getter, int**>;

template <typename AgentT>
[[nodiscard]] constexpr bool matches_temp_storage(cub::detail::smem_layout layout)
{
  return layout.size == sizeof(typename AgentT::TempStorage)
      && layout.alignment == alignof(typename AgentT::TempStorage);
}
} // namespace

CUB_TEST("required shared memory matches core agent storage", "[required_smem]", CUB_SMALL)
{
  STATIC_REQUIRE(matches_temp_storage<reduce_agent>(reduce_layout));
  STATIC_REQUIRE(matches_temp_storage<difference_agent>(difference_layout));
  STATIC_REQUIRE(matches_temp_storage<merge_agent>(merge_layout));
  STATIC_REQUIRE(matches_temp_storage<unique_agent>(unique_layout));
  STATIC_REQUIRE(matches_temp_storage<select_if_agent>(select_if_layout));
  STATIC_REQUIRE(matches_temp_storage<reduce_by_key_agent>(reduce_by_key_layout));
}

CUB_TEST("required shared memory matches sorting agent storage", "[required_smem]", CUB_SMALL)
{
  STATIC_REQUIRE(matches_temp_storage<block_sort_agent>(block_sort_layout));
  STATIC_REQUIRE(matches_temp_storage<merge_sort_agent>(merge_sort_layout));
  STATIC_REQUIRE(matches_temp_storage<batched_topk_agent>(batched_topk_layout));
  STATIC_REQUIRE(matches_temp_storage<cluster_topk_agent>(cluster_topk_layout));
}

CUB_TEST("required shared memory matches radix sort agent storage", "[required_smem]", CUB_SMALL)
{
  STATIC_REQUIRE(matches_temp_storage<onesweep_agent<cub::RADIX_RANK_MATCH>>(onesweep_layout<cub::RADIX_RANK_MATCH>));
  STATIC_REQUIRE(matches_temp_storage<onesweep_agent<cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY>>(
    onesweep_layout<cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY>));
  STATIC_REQUIRE(matches_temp_storage<onesweep_agent<cub::RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR>>(
    onesweep_layout<cub::RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR>));
  STATIC_REQUIRE(matches_temp_storage<histogram_agent>(histogram_layout));
}
