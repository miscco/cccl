//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __COOP_LOAD_TESTING_CUH__
#define __COOP_LOAD_TESTING_CUH__

#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/coop/algorithm>
#include <cuda/experimental/coop/group>

#include <vector>

#include <testing.cuh>

#include <c2h/catch2_test_helper.h>

/***********************************************************************************************************************
 * Item types
 **********************************************************************************************************************/

// Padded: 1 byte of padding, aligned to 2 bytes
struct padded_i16_i8
{
  cuda::std::int16_t a;
  cuda::std::int8_t b;

  __host__ __device__ friend bool operator==(const padded_i16_i8& lhs, const padded_i16_i8& rhs)
  {
    return lhs.a == rhs.a && lhs.b == rhs.b;
  }
};

// Padded: 3 bytes of padding, aligned to 4 bytes
struct padded_i32_i8
{
  cuda::std::int32_t a;
  cuda::std::int8_t b;

  __host__ __device__ friend bool operator==(const padded_i32_i8& lhs, const padded_i32_i8& rhs)
  {
    return lhs.a == rhs.a && lhs.b == rhs.b;
  }
};

// 16 bytes without padding, aligned to 4 bytes
struct i32x4
{
  cuda::std::int32_t a;
  cuda::std::int32_t b;
  cuda::std::int32_t c;
  cuda::std::int32_t d;

  __host__ __device__ friend bool operator==(const i32x4& lhs, const i32x4& rhs)
  {
    return lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c && lhs.d == rhs.d;
  }
};

// Distinct values for nearby items, so that loading the wrong item fails
template <class T>
T make_item(int k)
{
  if constexpr (cuda::std::is_same_v<T, padded_i16_i8>)
  {
    return {static_cast<cuda::std::int16_t>(k % 30011), static_cast<cuda::std::int8_t>(k % 113)};
  }
  else if constexpr (cuda::std::is_same_v<T, padded_i32_i8>)
  {
    return {k * 3 + 1, static_cast<cuda::std::int8_t>(k % 113)};
  }
  else if constexpr (cuda::std::is_same_v<T, i32x4>)
  {
    return {k, -k, k * 7, k + 11};
  }
  else if constexpr (cuda::std::is_same_v<T, int4>)
  {
    return int4{k, -k, k * 7, k + 11};
  }
  else if constexpr (sizeof(T) == 1)
  {
    return static_cast<T>(k % 251);
  }
  else if constexpr (sizeof(T) == 2)
  {
    return static_cast<T>(k % 30011);
  }
  else
  {
    return static_cast<T>(k * 3 + 1);
  }
}

template <class T>
bool items_equal(const T& lhs, const T& rhs)
{
  if constexpr (cuda::std::is_same_v<T, int4>)
  {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
  }
  else
  {
    return lhs == rhs;
  }
}

/***********************************************************************************************************************
 * Arrangements
 **********************************************************************************************************************/

// The position of item i of the thread of rank r in a tile of a group of P threads with N items per thread, from the
// definitions of the arrangements
inline int item_position(cudax::coop::blocked_t, int, int n, int r, int i)
{
  return (r * n) + i;
}

template <cuda::std::size_t V>
int item_position(cudax::coop::striped_t<V>, int p, int, int r, int i)
{
  constexpr int vec = static_cast<int>(V);
  return ((((i / vec) * p) + r) * vec) + (i % vec);
}

/***********************************************************************************************************************
 * Groups
 **********************************************************************************************************************/

inline constexpr int block_threads = 128;

struct is_even_rank
{
  template <class MappingResult>
  __device__ bool operator()(const MappingResult& mapping_result) const
  {
    return mapping_result.unit_rank() % 2 == 0;
  }
};

// Each group factory makes the group of the calling thread and passes it with the index of its tile to a callable.
// threads is the number of threads of a group, groups the number of groups of a block
struct this_block_group
{
  static constexpr int threads = block_threads;
  static constexpr int groups  = 1;

  template <class Config, class Fn>
  __device__ void operator()(Config config, Fn fn) const
  {
    const cudax::coop::this_block block{config};
    fn(block, 0u);
  }
};

struct this_warp_group
{
  static constexpr int threads = 32;
  static constexpr int groups  = block_threads / 32;

  template <class Config, class Fn>
  __device__ void operator()(Config config, Fn fn) const
  {
    const cudax::coop::this_warp warp{config};
    fn(warp, warp.template rank_as<unsigned>(cuda::block));
  }
};

struct this_thread_group
{
  static constexpr int threads = 1;
  static constexpr int groups  = block_threads;

  template <class Config, class Fn>
  __device__ void operator()(Config config, Fn fn) const
  {
    const cudax::coop::this_thread thread{config};
    fn(thread, thread.template rank_as<unsigned>(cuda::block));
  }
};

// Groups of Threads threads within a warp, with a static or a dynamic unit count
template <int Threads, bool Static>
struct threads_in_warp_group
{
  static constexpr int threads = Threads;
  static constexpr int groups  = block_threads / Threads;

  template <class Config, class Fn>
  __device__ void operator()(Config config, Fn fn) const
  {
    const cudax::coop::this_warp warp{config};
    const auto mapping = [] {
      if constexpr (Static)
      {
        return cudax::coop::group_by{cuda::std::cw<static_cast<cuda::std::size_t>(Threads)>};
      }
      else
      {
        return cudax::coop::group_by{static_cast<unsigned>(Threads)};
      }
    }();
    const cudax::coop::virtual_group group{cuda::gpu_thread, warp, mapping};
    fn(group,
       (warp.template rank_as<unsigned>(cuda::block) * (32u / Threads)) + group.template rank_as<unsigned>(warp));
  }
};

// The even and the odd lanes of a warp, whose ranks are not contiguous
struct even_odd_lanes_group
{
  static constexpr int threads = 16;
  static constexpr int groups  = block_threads / 16;

  template <class Config, class Fn>
  __device__ void operator()(Config config, Fn fn) const
  {
    const cudax::coop::this_warp warp{config};
    const cudax::coop::virtual_group group{cuda::gpu_thread, warp, cudax::coop::binary_partition{is_even_rank{}}};
    fn(group, (warp.template rank_as<unsigned>(cuda::block) * 2u) + group.template rank_as<unsigned>(warp));
  }
};

// Groups of Warps warps within a block
template <int Warps>
struct warps_in_block_group
{
  static constexpr int threads = Warps * 32;
  static constexpr int groups  = block_threads / (Warps * 32);

  template <class Config, class Fn>
  __device__ void operator()(Config config, Fn fn) const
  {
    const cudax::coop::this_block block{config};
    const cudax::coop::virtual_group group{
      cuda::warp, block, cudax::coop::group_by{cuda::std::cw<static_cast<cuda::std::size_t>(Warps)>}};
    fn(group, group.template rank_as<unsigned>(block));
  }
};

/***********************************************************************************************************************
 * Kernels
 **********************************************************************************************************************/

// Loads the tile of every group and stores the items of each thread in the order of the ranks
template <class Group, class Arrangement, int N, class... Args>
struct load_kernel
{
  template <class Config, class InputIt, class T>
  __device__ void operator()(Config config, InputIt src, T* out, Args... args) const
  {
    Group{}(config, [&](const auto& group, unsigned tile) {
      const unsigned rank      = cuda::gpu_thread.template rank_as<unsigned>(group);
      const unsigned count     = cuda::gpu_thread.count_as<unsigned>(group);
      const unsigned tile_size = count * static_cast<unsigned>(N);
      T items[N];
      for (int i = 0; i < N; ++i)
      {
        items[i] = out[((tile * count + rank) * N) + i];
      }
      cudax::coop::load(group, Arrangement{}, src + (tile * tile_size), items, args...);
      for (int i = 0; i < N; ++i)
      {
        out[((tile * count + rank) * N) + i] = items[i];
      }
    });
  }
};

/***********************************************************************************************************************
 * Launch and verification
 **********************************************************************************************************************/

// Input items placed at a byte offset into an allocation, so that the tile can start at any alignment
template <class InputT>
struct offset_input
{
  c2h::device_vector<unsigned char> bytes;
  std::vector<InputT> items;

  offset_input(int num_items, int byte_offset)
      : bytes(static_cast<std::size_t>(num_items) * sizeof(InputT) + 64)
      , items(static_cast<std::size_t>(num_items))
  {
    for (int k = 0; k < num_items; ++k)
    {
      items[static_cast<std::size_t>(k)] = make_item<InputT>(k);
    }
    std::vector<unsigned char> host(bytes.size(), 0xcd);
    std::memcpy(host.data() + byte_offset, items.data(), items.size() * sizeof(InputT));
    bytes = host;
    ptr   = reinterpret_cast<const InputT*>(thrust::raw_pointer_cast(bytes.data()) + byte_offset);
  }

  const InputT* ptr;
};

template <bool StaticBlock = true>
auto make_load_config()
{
  if constexpr (StaticBlock)
  {
    return cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<block_threads>());
  }
  else
  {
    return cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(block_threads));
  }
}

// Expects item i of the thread of rank r of group g to be the item at its position in the tile of g, or the initial
// value of the destination if its position is not before valid_items
template <class Group, class Arrangement, int N, class InputT, class T>
void verify_load(const std::vector<InputT>& input, const c2h::device_vector<T>& d_out, int valid_items, T initial)
{
  const c2h::host_vector<T> out = d_out;
  const int p                   = Group::threads;
  int mismatches                = 0;
  int first_mismatch            = -1;
  for (int g = 0; g < Group::groups; ++g)
  {
    for (int r = 0; r < p; ++r)
    {
      for (int i = 0; i < N; ++i)
      {
        const int pos = item_position(Arrangement{}, p, N, r, i);
        const T expected =
          pos < valid_items ? static_cast<T>(input[static_cast<std::size_t>((g * p * N) + pos)]) : initial;
        const int index = (((g * p) + r) * N) + i;
        if (!items_equal(out[static_cast<std::size_t>(index)], expected))
        {
          first_mismatch = mismatches == 0 ? index : first_mismatch;
          ++mismatches;
        }
      }
    }
  }
  // The group, rank and item of the first mismatch
  CAPTURE(valid_items, first_mismatch / (p * N), first_mismatch / N % p, first_mismatch % N);
  REQUIRE(mismatches == 0);
}

// Loads full tiles of InputT from every byte offset from 0 to 32 that InputT allows, through make_it(pointer), into T
template <class Group, class Arrangement, int N, class T, class InputT, bool StaticBlock = true, class MakeIt, class... Args>
void test_load_offsets(MakeIt make_it, Args... args)
{
  constexpr int num_items = block_threads * N;
  const cuda::stream stream{cuda::devices[0]};
  for (int byte_offset = 0; byte_offset <= 32; byte_offset += static_cast<int>(alignof(InputT)))
  {
    CAPTURE(byte_offset);
    const offset_input<InputT> input(num_items, byte_offset);
    c2h::device_vector<T> d_out(static_cast<std::size_t>(num_items), make_item<T>(-1));
    cuda::launch(
      stream,
      make_load_config<StaticBlock>(),
      load_kernel<Group, Arrangement, N, Args...>{},
      make_it(input.ptr),
      thrust::raw_pointer_cast(d_out.data()),
      args...);
    stream.sync();
    verify_load<Group, Arrangement, N>(input.items, d_out, num_items, make_item<T>(-1));
  }
}

struct pointer_input
{
  template <class T>
  const T* operator()(const T* ptr) const
  {
    return ptr;
  }
};

template <cub::CacheLoadModifier Modifier>
struct cache_modified_input
{
  template <class T>
  cub::CacheModifiedInputIterator<Modifier, T> operator()(const T* ptr) const
  {
    return cub::CacheModifiedInputIterator<Modifier, T>{ptr};
  }
};

#endif // __COOP_LOAD_TESTING_CUH__
