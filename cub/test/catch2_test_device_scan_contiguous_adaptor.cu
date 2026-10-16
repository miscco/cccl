// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Covers https://github.com/NVIDIA/cccl/issues/11840: InclusiveSum of a transform_iterator over char storage into a
// transform_output_iterator over uint8_t storage. The scan reads uint32_t reset flags and writes uint8_t segments,
// while the bulk copy uses the underlying char and uint8_t pointers.

#include <cub/device/device_scan.cuh>

#include <cuda/iterator>
#include <cuda/std/cstdint>

#include <cstdint>

#include "cub_test_macros.h"

struct newline_to_reset_segment
{
  _CCCL_HOST_DEVICE ::cuda::std::uint32_t operator()(char symbol) const
  {
    return symbol == '\n' ? 1u : 0u;
  }
};

struct mod_to_target_type
{
  _CCCL_HOST_DEVICE ::cuda::std::uint8_t operator()(::cuda::std::uint32_t value) const
  {
    return static_cast<::cuda::std::uint8_t>(value % 256u);
  }
};

CUB_TEST("Device scan issue 11840 uses contiguous adaptor storage", "[scan][device]", CUB_SMALL, c2h::type_list<int>)
{
  constexpr int num_items    = 1 << 20;
  constexpr int reset_period = 1024;

  c2h::host_vector<char> host_symbols(num_items);
  c2h::host_vector<::cuda::std::uint8_t> expected(num_items);
  for (int item = 0; item < num_items; ++item)
  {
    host_symbols[item] = item % reset_period == 0 ? '\n' : '{';
    expected[item]     = static_cast<::cuda::std::uint8_t>((item / reset_period + 1) % 256);
  }

  c2h::device_vector<char> symbols = host_symbols;
  c2h::device_vector<::cuda::std::uint8_t> segments(num_items);
  auto input = ::cuda::transform_iterator<newline_to_reset_segment, char*>{
    thrust::raw_pointer_cast(symbols.data()), newline_to_reset_segment{}};
  auto output = ::cuda::transform_output_iterator<mod_to_target_type, ::cuda::std::uint8_t*>{
    thrust::raw_pointer_cast(segments.data()), mod_to_target_type{}};

  void* temporary_storage       = nullptr;
  ::cuda::std::size_t num_bytes = 0;
  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSum(temporary_storage, num_bytes, input, output, num_items));

  c2h::device_vector<::cuda::std::uint8_t> temporary(num_bytes);
  temporary_storage = thrust::raw_pointer_cast(temporary.data());
  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSum(temporary_storage, num_bytes, input, output, num_items));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(c2h::host_vector<::cuda::std::uint8_t>(segments) == expected);
}

template <typename T>
struct zero_to_reset_segment
{
  _CCCL_HOST_DEVICE ::cuda::std::uint32_t operator()(T symbol) const
  {
    return symbol == T{} ? 1u : 0u;
  }
};

// Same segment scan as the issue, with the stored symbol type swept across the integer types.
CUB_TEST_LIST(
  "Device scan contiguous adaptor works for integer symbol types",
  "[scan][device]",
  CUB_SMALL,
  std::int8_t,
  std::int16_t,
  std::int32_t,
  std::int64_t,
  __int128_t)
{
  using symbol_t             = TestType;
  constexpr int num_items    = 1 << 16;
  constexpr int reset_period = 1024;

  c2h::host_vector<symbol_t> host_symbols(num_items);
  c2h::host_vector<::cuda::std::uint8_t> expected(num_items);
  for (int item = 0; item < num_items; ++item)
  {
    host_symbols[item] = item % reset_period == 0 ? symbol_t{} : symbol_t{1};
    expected[item]     = static_cast<::cuda::std::uint8_t>((item / reset_period + 1) % 256);
  }

  c2h::device_vector<symbol_t> symbols = host_symbols;
  c2h::device_vector<::cuda::std::uint8_t> segments(num_items);
  auto input = ::cuda::transform_iterator<zero_to_reset_segment<symbol_t>, symbol_t*>{
    thrust::raw_pointer_cast(symbols.data()), zero_to_reset_segment<symbol_t>{}};
  auto output = ::cuda::transform_output_iterator<mod_to_target_type, ::cuda::std::uint8_t*>{
    thrust::raw_pointer_cast(segments.data()), mod_to_target_type{}};

  void* temporary_storage       = nullptr;
  ::cuda::std::size_t num_bytes = 0;
  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSum(temporary_storage, num_bytes, input, output, num_items));

  c2h::device_vector<::cuda::std::uint8_t> temporary(num_bytes);
  temporary_storage = thrust::raw_pointer_cast(temporary.data());
  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSum(temporary_storage, num_bytes, input, output, num_items));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(c2h::host_vector<::cuda::std::uint8_t>(segments) == expected);
}
