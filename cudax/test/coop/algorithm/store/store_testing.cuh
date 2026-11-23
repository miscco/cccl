//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __COOP_STORE_TESTING_CUH__
#define __COOP_STORE_TESTING_CUH__

#include "../load/load_testing.cuh"

// Reads the items of each thread in the order of the ranks and stores the tile of every group
template <class Group, class Arrangement, int N, class... Args>
struct store_kernel
{
  template <class Config, class T, class OutputT>
  __device__ void operator()(Config config, const T* in, OutputT* dst, Args... args) const
  {
    Group{}(config, [&](const auto& group, unsigned tile) {
      const unsigned rank      = cuda::gpu_thread.template rank_as<unsigned>(group);
      const unsigned count     = cuda::gpu_thread.count_as<unsigned>(group);
      const unsigned tile_size = count * static_cast<unsigned>(N);
      T items[N];
      for (int i = 0; i < N; ++i)
      {
        items[i] = in[((tile * count + rank) * N) + i];
      }
      cudax::coop::store(group, Arrangement{}, dst + (tile * tile_size), items, args...);
    });
  }
};

// Output items placed at a byte offset into an allocation, so that the tile can start at any alignment
template <class OutputT>
struct offset_output
{
  c2h::device_vector<unsigned char> bytes;
  int num_items;
  int byte_offset;

  offset_output(int num_items, int byte_offset, OutputT initial)
      : bytes(static_cast<std::size_t>(num_items) * sizeof(OutputT) + 64)
      , num_items(num_items)
      , byte_offset(byte_offset)
  {
    std::vector<OutputT> items(static_cast<std::size_t>(num_items), initial);
    std::vector<unsigned char> host(bytes.size(), 0xcd);
    std::memcpy(host.data() + byte_offset, items.data(), items.size() * sizeof(OutputT));
    bytes = host;
  }

  OutputT* ptr()
  {
    return reinterpret_cast<OutputT*>(thrust::raw_pointer_cast(bytes.data()) + byte_offset);
  }

  std::vector<OutputT> items() const
  {
    const c2h::host_vector<unsigned char> host = bytes;
    std::vector<OutputT> result(static_cast<std::size_t>(num_items));
    std::memcpy(result.data(), thrust::raw_pointer_cast(host.data()) + byte_offset, result.size() * sizeof(OutputT));
    return result;
  }
};

// Expects the item at position pos of the tile of group g to be item i of the thread of rank r of g, or the initial
// value of the destination if pos is not before valid_items
template <class Group, class Arrangement, int N, class T, class OutputT>
void verify_store(const std::vector<T>& input, const std::vector<OutputT>& out, int valid_items, OutputT initial)
{
  const int p        = Group::threads;
  int mismatches     = 0;
  int first_mismatch = -1;
  for (int g = 0; g < Group::groups; ++g)
  {
    for (int r = 0; r < p; ++r)
    {
      for (int i = 0; i < N; ++i)
      {
        const int pos = item_position(Arrangement{}, p, N, r, i);
        const OutputT expected =
          pos < valid_items ? static_cast<OutputT>(input[static_cast<std::size_t>((((g * p) + r) * N) + i)]) : initial;
        const int index = (g * p * N) + pos;
        if (!items_equal(out[static_cast<std::size_t>(index)], expected))
        {
          first_mismatch = mismatches == 0 ? index : first_mismatch;
          ++mismatches;
        }
      }
    }
  }
  // The group and the position in its tile of the first mismatch
  CAPTURE(valid_items, first_mismatch / (p * N), first_mismatch % (p * N));
  REQUIRE(mismatches == 0);
}

// Stores full tiles of T into OutputT at every byte offset from 0 to 32 that OutputT allows
template <class Group, class Arrangement, int N, class T, class OutputT, bool StaticBlock = true, class... Args>
void test_store_offsets(Args... args)
{
  constexpr int num_items = block_threads * N;
  const cuda::stream stream{cuda::devices[0]};
  std::vector<T> input(static_cast<std::size_t>(num_items));
  for (int k = 0; k < num_items; ++k)
  {
    input[static_cast<std::size_t>(k)] = make_item<T>(k);
  }
  const c2h::device_vector<T> d_input(input.begin(), input.end());
  for (int byte_offset = 0; byte_offset <= 32; byte_offset += static_cast<int>(alignof(OutputT)))
  {
    CAPTURE(byte_offset);
    offset_output<OutputT> output(num_items, byte_offset, make_item<OutputT>(-1));
    cuda::launch(
      stream,
      make_load_config<StaticBlock>(),
      store_kernel<Group, Arrangement, N, Args...>{},
      thrust::raw_pointer_cast(d_input.data()),
      output.ptr(),
      args...);
    stream.sync();
    verify_store<Group, Arrangement, N>(input, output.items(), num_items, make_item<OutputT>(-1));
  }
}

#endif // __COOP_STORE_TESTING_CUH__
