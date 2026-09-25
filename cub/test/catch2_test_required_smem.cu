// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_load_to_shared.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_raking_layout.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_shuffle.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_topk.cuh>
#include <cub/device/dispatch/kernels/kernel_batched_topk.cuh>
#include <cub/warp/warp_exchange.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/warp/warp_store.cuh>

#include <cuda/std/cstdint>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>

namespace
{
using cub::detail::cub_algorithm;
using cub::detail::required_smem_layout_v;
using cub::detail::required_smem_v;

template <typename T, int BlockThreads, int ItemsPerThread, bool WarpTimeSlicing>
inline constexpr auto block_exchange_smem =
  required_smem_v<cub_algorithm::block_exchange,
                  T,
                  ::cuda::std::integral_constant<int, BlockThreads>,
                  ::cuda::std::integral_constant<int, ItemsPerThread>,
                  ::cuda::std::bool_constant<WarpTimeSlicing>,
                  ::cuda::std::integral_constant<int, 1>,
                  ::cuda::std::integral_constant<int, 1>>;

template <typename T, int BlockThreads, int ItemsPerThread, cub::BlockLoadAlgorithm Algorithm>
inline constexpr auto block_load_smem =
  required_smem_v<cub_algorithm::block_load,
                  T,
                  ::cuda::std::integral_constant<int, BlockThreads>,
                  ::cuda::std::integral_constant<int, ItemsPerThread>,
                  ::cuda::std::integral_constant<cub::BlockLoadAlgorithm, Algorithm>,
                  ::cuda::std::integral_constant<int, 1>,
                  ::cuda::std::integral_constant<int, 1>>;

template <typename T, int BlockThreads, int ItemsPerThread, cub::BlockStoreAlgorithm Algorithm>
inline constexpr auto block_store_smem =
  required_smem_v<cub_algorithm::block_store,
                  T,
                  ::cuda::std::integral_constant<int, BlockThreads>,
                  ::cuda::std::integral_constant<int, ItemsPerThread>,
                  ::cuda::std::integral_constant<cub::BlockStoreAlgorithm, Algorithm>,
                  ::cuda::std::integral_constant<int, 1>,
                  ::cuda::std::integral_constant<int, 1>>;
} // namespace

CUB_TEST("required shared memory matches block data movement storage", "[required_smem]", CUB_SMALL)
{
  using exchange = cub::BlockExchange<int, 128, 4>;
  STATIC_REQUIRE(block_exchange_smem<int, 128, 4, false> == sizeof(typename exchange::TempStorage));

  using sliced_exchange = cub::BlockExchange<int, 128, 8, true>;
  STATIC_REQUIRE(block_exchange_smem<int, 128, 8, true> == sizeof(typename sliced_exchange::TempStorage));

  STATIC_REQUIRE(block_load_smem<int, 128, 4, cub::BLOCK_LOAD_DIRECT> == 1);
  STATIC_REQUIRE(block_load_smem<int, 128, 4, cub::BLOCK_LOAD_STRIPED> == 1);
  STATIC_REQUIRE(block_load_smem<int, 128, 4, cub::BLOCK_LOAD_VECTORIZE> == 1);

  using block_load = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  STATIC_REQUIRE(
    block_load_smem<int, 128, 4, cub::BLOCK_LOAD_WARP_TRANSPOSE> == sizeof(typename block_load::TempStorage));

  STATIC_REQUIRE(block_store_smem<int, 128, 4, cub::BLOCK_STORE_DIRECT> == 1);

  using block_store = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  STATIC_REQUIRE(
    block_store_smem<int, 128, 4, cub::BLOCK_STORE_WARP_TRANSPOSE> == sizeof(typename block_store::TempStorage));
}

CUB_TEST("required shared memory matches warp collective storage", "[required_smem]", CUB_SMALL)
{
  constexpr auto warp_exchange_smem =
    required_smem_v<cub_algorithm::warp_exchange,
                    int,
                    ::cuda::std::integral_constant<int, 4>,
                    ::cuda::std::integral_constant<int, 32>,
                    ::cuda::std::integral_constant<cub::WarpExchangeAlgorithm, cub::WARP_EXCHANGE_SMEM>>;
  using warp_exchange = cub::WarpExchange<int, 4, 32, cub::WARP_EXCHANGE_SMEM>;
  STATIC_REQUIRE(warp_exchange_smem == sizeof(typename warp_exchange::TempStorage));

  STATIC_REQUIRE(
    required_smem_v<cub_algorithm::warp_exchange,
                    int,
                    ::cuda::std::integral_constant<int, 32>,
                    ::cuda::std::integral_constant<int, 32>,
                    ::cuda::std::integral_constant<cub::WarpExchangeAlgorithm, cub::WARP_EXCHANGE_SHUFFLE>>
    == 1);
  STATIC_REQUIRE(
    required_smem_v<cub_algorithm::warp_load,
                    int,
                    ::cuda::std::integral_constant<int, 4>,
                    ::cuda::std::integral_constant<cub::WarpLoadAlgorithm, cub::WARP_LOAD_DIRECT>,
                    ::cuda::std::integral_constant<int, 32>>
    == 1);
  STATIC_REQUIRE(
    required_smem_v<cub_algorithm::warp_store,
                    int,
                    ::cuda::std::integral_constant<int, 4>,
                    ::cuda::std::integral_constant<cub::WarpStoreAlgorithm, cub::WARP_STORE_DIRECT>,
                    ::cuda::std::integral_constant<int, 32>>
    == 1);
  STATIC_REQUIRE(required_smem_v<cub_algorithm::warp_reduce, int, ::cuda::std::integral_constant<int, 32>> == 1);
  STATIC_REQUIRE(required_smem_v<cub_algorithm::warp_scan, int, ::cuda::std::integral_constant<int, 32>> == 1);

  using warp_reduce = cub::WarpReduce<int, 3>;
  STATIC_REQUIRE(required_smem_v<cub_algorithm::warp_reduce, int, ::cuda::std::integral_constant<int, 3>>
                 == sizeof(typename warp_reduce::TempStorage));

  using warp_scan = cub::WarpScan<int, 3>;
  STATIC_REQUIRE(required_smem_v<cub_algorithm::warp_scan, int, ::cuda::std::integral_constant<int, 3>>
                 == sizeof(typename warp_scan::TempStorage));
}

CUB_TEST("required shared memory matches block leaf storage", "[required_smem]", CUB_SMALL)
{
  using raking = cub::BlockRakingLayout<int, 65>;
  STATIC_REQUIRE(required_smem_v<cub_algorithm::block_raking_layout, int, ::cuda::std::integral_constant<int, 65>>
                 == sizeof(typename raking::TempStorage));

  using discontinuity = cub::BlockDiscontinuity<int, 32, 2, 2>;
  STATIC_REQUIRE(
    required_smem_v<cub_algorithm::block_discontinuity,
                    int,
                    ::cuda::std::integral_constant<int, 32>,
                    ::cuda::std::integral_constant<int, 2>,
                    ::cuda::std::integral_constant<int, 2>>
    == sizeof(typename discontinuity::TempStorage));

  using adjacent_difference = cub::BlockAdjacentDifference<int, 32, 2, 2>;
  STATIC_REQUIRE(
    required_smem_v<cub_algorithm::block_adjacent_difference,
                    int,
                    ::cuda::std::integral_constant<int, 32>,
                    ::cuda::std::integral_constant<int, 2>,
                    ::cuda::std::integral_constant<int, 2>>
    == sizeof(typename adjacent_difference::TempStorage));

  using shuffle = cub::BlockShuffle<int, 32, 2, 2>;
  STATIC_REQUIRE(
    required_smem_v<cub_algorithm::block_shuffle,
                    int,
                    ::cuda::std::integral_constant<int, 32>,
                    ::cuda::std::integral_constant<int, 2>,
                    ::cuda::std::integral_constant<int, 2>>
    == sizeof(typename shuffle::TempStorage));

  using load_to_shared = cub::detail::BlockLoadToShared<128>;
  STATIC_REQUIRE(required_smem_v<cub_algorithm::block_load_to_shared> == sizeof(typename load_to_shared::TempStorage));
}

template <typename T, int BlockThreads, cub::BlockScanAlgorithm Algorithm>
[[nodiscard]] constexpr bool block_scan_layout_matches()
{
  using storage = typename cub::BlockScan<T, BlockThreads, Algorithm>::TempStorage;
  constexpr auto l =
    required_smem_layout_v<cub_algorithm::block_scan,
                           T,
                           ::cuda::std::integral_constant<int, BlockThreads>,
                           ::cuda::std::integral_constant<cub::BlockScanAlgorithm, Algorithm>,
                           ::cuda::std::integral_constant<int, 1>,
                           ::cuda::std::integral_constant<int, 1>>;
  return l.size == sizeof(storage) && l.alignment == alignof(storage);
}

template <typename T, int BlockThreads, cub::BlockReduceAlgorithm Algorithm>
[[nodiscard]] constexpr bool block_reduce_layout_matches()
{
  using storage = typename cub::BlockReduce<T, BlockThreads, Algorithm>::TempStorage;
  constexpr auto l =
    required_smem_layout_v<cub_algorithm::block_reduce,
                           T,
                           ::cuda::std::integral_constant<int, BlockThreads>,
                           ::cuda::std::integral_constant<cub::BlockReduceAlgorithm, Algorithm>,
                           ::cuda::std::integral_constant<int, 1>,
                           ::cuda::std::integral_constant<int, 1>>;
  return l.size == sizeof(storage) && l.alignment == alignof(storage);
}

template <int BlockDimX,
          int RadixBits,
          bool IsDescending,
          bool MemoizeOuterScan,
          cub::BlockScanAlgorithm ScanAlgorithm,
          cudaSharedMemConfig SMemConfig,
          int BlockDimY = 1,
          int BlockDimZ = 1>
[[nodiscard]] constexpr bool block_radix_rank_basic_layout_matches()
{
  using storage = typename cub::
    BlockRadixRank<BlockDimX, RadixBits, IsDescending, MemoizeOuterScan, ScanAlgorithm, SMemConfig, BlockDimY, BlockDimZ>::
      TempStorage;
  constexpr auto l =
    required_smem_layout_v<cub_algorithm::block_radix_rank_basic,
                           ::cuda::std::integral_constant<int, BlockDimX>,
                           ::cuda::std::integral_constant<int, RadixBits>,
                           ::cuda::std::integral_constant<cub::BlockScanAlgorithm, ScanAlgorithm>,
                           ::cuda::std::integral_constant<cudaSharedMemConfig, SMemConfig>,
                           ::cuda::std::integral_constant<int, BlockDimY>,
                           ::cuda::std::integral_constant<int, BlockDimZ>>;
  return l.size == sizeof(storage) && l.alignment == alignof(storage);
}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int BlockThreads,
          int RadixBits,
          bool IsDescending,
          cub::BlockScanAlgorithm ScanAlgorithm>
[[nodiscard]] constexpr bool block_radix_rank_layout_matches()
{
  using storage = typename cub::detail::
    block_radix_rank_t<RankAlgorithm, BlockThreads, RadixBits, IsDescending, ScanAlgorithm>::TempStorage;
  constexpr auto l =
    required_smem_layout_v<cub_algorithm::block_radix_rank,
                           ::cuda::std::integral_constant<cub::RadixRankAlgorithm, RankAlgorithm>,
                           ::cuda::std::integral_constant<int, BlockThreads>,
                           ::cuda::std::integral_constant<int, RadixBits>,
                           ::cuda::std::integral_constant<cub::BlockScanAlgorithm, ScanAlgorithm>>;
  return l.size == sizeof(storage) && l.alignment == alignof(storage);
}

template <typename KeyT,
          int BlockDimX,
          int ItemsPerThread,
          typename ValueT,
          int RadixBits,
          bool MemoizeOuterScan,
          cub::BlockScanAlgorithm ScanAlgorithm,
          cudaSharedMemConfig SMemConfig,
          int BlockDimY = 1,
          int BlockDimZ = 1>
[[nodiscard]] constexpr bool block_radix_sort_layout_matches()
{
  using storage = typename cub::BlockRadixSort<
    KeyT,
    BlockDimX,
    ItemsPerThread,
    ValueT,
    RadixBits,
    MemoizeOuterScan,
    ScanAlgorithm,
    SMemConfig,
    BlockDimY,
    BlockDimZ>::TempStorage;
  constexpr auto l = required_smem_layout_v<
    cub_algorithm::block_radix_sort,
    KeyT,
    ::cuda::std::integral_constant<int, BlockDimX>,
    ::cuda::std::integral_constant<int, ItemsPerThread>,
    ValueT,
    ::cuda::std::integral_constant<int, RadixBits>,
    ::cuda::std::integral_constant<cub::BlockScanAlgorithm, ScanAlgorithm>,
    ::cuda::std::integral_constant<cudaSharedMemConfig, SMemConfig>,
    ::cuda::std::integral_constant<int, BlockDimY>,
    ::cuda::std::integral_constant<int, BlockDimZ>>;
  return l.size == sizeof(storage) && l.alignment == alignof(storage);
}

CUB_TEST("required shared memory matches block scan and reduce dispatch", "[required_smem]", CUB_SMALL)
{
  STATIC_REQUIRE(block_scan_layout_matches<int, 128, cub::BLOCK_SCAN_WARP_SCANS>());
  // Not a multiple of the warp size, so `BlockScan` silently falls back to raking
  STATIC_REQUIRE(block_scan_layout_matches<int, 100, cub::BLOCK_SCAN_WARP_SCANS>());
  STATIC_REQUIRE(block_scan_layout_matches<int, 32, cub::BLOCK_SCAN_WARP_SCANS>());
  STATIC_REQUIRE(block_scan_layout_matches<int, 128, cub::BLOCK_SCAN_RAKING>());
  STATIC_REQUIRE(block_scan_layout_matches<int, 128, cub::BLOCK_SCAN_RAKING_MEMOIZE>());
  STATIC_REQUIRE(block_scan_layout_matches<char, 33, cub::BLOCK_SCAN_RAKING>());
  STATIC_REQUIRE(block_scan_layout_matches<double, 96, cub::BLOCK_SCAN_RAKING_MEMOIZE>());
  STATIC_REQUIRE(block_scan_layout_matches<long long, 64, cub::BLOCK_SCAN_WARP_SCANS>());

  STATIC_REQUIRE(block_reduce_layout_matches<int, 128, cub::BLOCK_REDUCE_WARP_REDUCTIONS>());
  STATIC_REQUIRE(block_reduce_layout_matches<int, 100, cub::BLOCK_REDUCE_WARP_REDUCTIONS>());
  STATIC_REQUIRE(block_reduce_layout_matches<int, 16, cub::BLOCK_REDUCE_WARP_REDUCTIONS>());
  STATIC_REQUIRE(block_reduce_layout_matches<int, 128, cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC>());
  STATIC_REQUIRE(block_reduce_layout_matches<int, 128, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>());
  STATIC_REQUIRE(block_reduce_layout_matches<int, 32, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>());
  STATIC_REQUIRE(block_reduce_layout_matches<int, 128, cub::BLOCK_REDUCE_RAKING>());
  STATIC_REQUIRE(block_reduce_layout_matches<char, 33, cub::BLOCK_REDUCE_RAKING>());
  STATIC_REQUIRE(block_reduce_layout_matches<double, 96, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>());
  STATIC_REQUIRE(block_reduce_layout_matches<long long, 64, cub::BLOCK_REDUCE_WARP_REDUCTIONS>());
}

CUB_TEST("required shared memory matches block sorting storage", "[required_smem]", CUB_SMALL)
{
  STATIC_REQUIRE(
    block_radix_rank_basic_layout_matches<128,
                                          4,
                                          false,
                                          false,
                                          cub::BLOCK_SCAN_WARP_SCANS,
                                          cudaSharedMemBankSizeFourByte>());
  STATIC_REQUIRE(
    block_radix_rank_basic_layout_matches<64, 5, true, true, cub::BLOCK_SCAN_RAKING, cudaSharedMemBankSizeEightByte>());
  STATIC_REQUIRE(
    block_radix_rank_basic_layout_matches<
      32,
      1,
      false,
      true,
      cub::BLOCK_SCAN_RAKING_MEMOIZE,
      cudaSharedMemBankSizeFourByte,
      2,
      2>());

  STATIC_REQUIRE(block_radix_rank_layout_matches<cub::RADIX_RANK_BASIC, 128, 4, false, cub::BLOCK_SCAN_WARP_SCANS>());
  STATIC_REQUIRE(block_radix_rank_layout_matches<cub::RADIX_RANK_MEMOIZE, 128, 4, true, cub::BLOCK_SCAN_RAKING>());
  STATIC_REQUIRE(block_radix_rank_layout_matches<cub::RADIX_RANK_MATCH, 128, 5, false, cub::BLOCK_SCAN_WARP_SCANS>());
  STATIC_REQUIRE(
    block_radix_rank_layout_matches<cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY, 128, 4, false, cub::BLOCK_SCAN_WARP_SCANS>());
  STATIC_REQUIRE(
    block_radix_rank_layout_matches<cub::RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR,
                                    128,
                                    4,
                                    true,
                                    cub::BLOCK_SCAN_WARP_SCANS>());

  STATIC_REQUIRE(
    block_radix_sort_layout_matches<
      int,
      128,
      4,
      cub::NullType,
      4,
      true,
      cub::BLOCK_SCAN_WARP_SCANS,
      cudaSharedMemBankSizeFourByte>());
  STATIC_REQUIRE(
    block_radix_sort_layout_matches<
      long long,
      64,
      3,
      int,
      5,
      false,
      cub::BLOCK_SCAN_RAKING,
      cudaSharedMemBankSizeEightByte>());
  STATIC_REQUIRE(
    block_radix_sort_layout_matches<
      int,
      32,
      2,
      long long,
      3,
      true,
      cub::BLOCK_SCAN_RAKING_MEMOIZE,
      cudaSharedMemBankSizeFourByte,
      2,
      2>());

  using topk_air = cub::detail::block_topk_air<int, 128, 4, long long>;
  STATIC_REQUIRE(
    required_smem_v<cub_algorithm::block_topk_air,
                    int,
                    ::cuda::std::integral_constant<int, 128>,
                    ::cuda::std::integral_constant<int, 4>,
                    long long,
                    ::cuda::std::integral_constant<int, 8>>
    == sizeof(typename topk_air::TempStorage));

  using topk = cub::detail::block_topk<int, 128, 4, long long>;
  STATIC_REQUIRE(
    required_smem_v<cub_algorithm::block_topk,
                    int,
                    ::cuda::std::integral_constant<int, 128>,
                    ::cuda::std::integral_constant<int, 4>,
                    long long>
    == sizeof(typename topk::TempStorage));

  using merge_sort = cub::BlockMergeSort<int, 128, 4, long long>;
  STATIC_REQUIRE(
    required_smem_v<cub_algorithm::block_merge_sort,
                    int,
                    ::cuda::std::integral_constant<int, 128>,
                    ::cuda::std::integral_constant<int, 4>,
                    long long,
                    ::cuda::std::integral_constant<int, 1>,
                    ::cuda::std::integral_constant<int, 1>>
    == sizeof(typename merge_sort::TempStorage));
}

CUB_TEST("sorted-output temp storage layout matches BlockLoad and BlockRadixSort union", "[required_smem]", CUB_SMALL)
{
  using cub::detail::batched_topk::sort_covered_max;
  using cub::detail::batched_topk::sort_temp_storage;
  using cub::detail::batched_topk::sort_temp_storage_layout;

  STATIC_REQUIRE(sort_temp_storage_layout<int, int, 256, 64>.size == sizeof(sort_temp_storage<int, int, 256, 64>));
  STATIC_REQUIRE(
    sort_temp_storage_layout<int, int, 256, 64>.alignment == alignof(sort_temp_storage<int, int, 256, 64>));
  STATIC_REQUIRE(sort_temp_storage_layout<int, int, 32, 1>.size == sizeof(sort_temp_storage<int, int, 32, 1>));
  STATIC_REQUIRE(sort_temp_storage_layout<::cuda::std::int64_t, ::cuda::std::int64_t, 128, 2>.size
                 == sizeof(sort_temp_storage<::cuda::std::int64_t, ::cuda::std::int64_t, 128, 2>));
  STATIC_REQUIRE(sort_temp_storage_layout<float, cub::NullType, 256, 8>.size
                 == sizeof(sort_temp_storage<float, cub::NullType, 256, 8>));

  STATIC_REQUIRE(sort_covered_max<::cuda::std::int32_t, ::cuda::std::int32_t>() >= 2048);
  STATIC_REQUIRE(sort_covered_max<::cuda::std::int64_t, ::cuda::std::int32_t>() >= 2048);
  STATIC_REQUIRE(sort_covered_max<::cuda::std::int64_t, ::cuda::std::int64_t>() >= 2048);
  STATIC_REQUIRE(sort_covered_max<float, ::cuda::std::int32_t>() >= 2048);

  struct sixteen_byte_value
  {
    ::cuda::std::int64_t first;
    ::cuda::std::int64_t second;
  };
  STATIC_REQUIRE(sort_covered_max<::cuda::std::int64_t, sixteen_byte_value>() >= 2048);
}
