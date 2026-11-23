// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_store.cuh>

#include <thrust/device_vector.h>

#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/utility>

#include <cuda_runtime_api.h>
#include <nvbench_helper.cuh>

// Pairs are aligned to less than their size, thus their vectors can be wider than their alignment. They also exercise
// padding, which the vectorized stores write along with the members
using pair_i16_i8  = cuda::std::pair<int16_t, int8_t>;
using pair_i32_i16 = cuda::std::pair<int32_t, int16_t>;
using pair_i64_i32 = cuda::std::pair<int64_t, int32_t>;
NVBENCH_DECLARE_TYPE_STRINGS(pair_i16_i8, "P_I16_I8", "pair<int16_t, int8_t>");
NVBENCH_DECLARE_TYPE_STRINGS(pair_i32_i16, "P_I32_I16", "pair<int32_t, int16_t>");
NVBENCH_DECLARE_TYPE_STRINGS(pair_i64_i32, "P_I64_I32", "pair<int64_t, int32_t>");

using value_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t,
#if _CCCL_HAS_INT128()
                     int128_t,
#endif // _CCCL_HAS_INT128()
                     pair_i16_i8,
                     pair_i32_i16,
                     pair_i64_i32>;

// The bytes that each thread stores, which bound the width of its vectors. A thread stores several vectors once its
// bytes exceed the widest vector
using bytes_per_thread = nvbench::enum_type_list<16, 32, 64>;

constexpr int threads_per_block = 256;

template <typename T>
inline constexpr bool is_pair_v = false;

template <typename T1, typename T2>
inline constexpr bool is_pair_v<cuda::std::pair<T1, T2>> = true;

// Derives an item from a seed, so that the stores are not constant
template <typename T>
__device__ __forceinline__ T make_item(uint32_t seed)
{
  if constexpr (is_pair_v<T>)
  {
    return T{static_cast<typename T::first_type>(seed), static_cast<typename T::second_type>(seed >> 1)};
  }
  else
  {
    return static_cast<T>(seed);
  }
}

template <int ItemsPerThread, typename T>
__launch_bounds__(threads_per_block) __global__ void store_kernel(T* output, cuda::std::size_t num_tiles)
{
  using block_store_t      = cub::BlockStore<T, threads_per_block, ItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;
  constexpr int tile_items = threads_per_block * ItemsPerThread;
  for (cuda::std::size_t tile = blockIdx.x; tile < num_tiles; tile += gridDim.x)
  {
    T items[ItemsPerThread];
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      items[i] = make_item<T>(static_cast<uint32_t>(tile) ^ ((threadIdx.x * ItemsPerThread) + i));
    }
    block_store_t{}.Store(output + (tile * tile_items), items);
  }
}

template <typename T, int BytesPerThread>
void direct_blocked_vectorized(nvbench::state& state, nvbench::type_list<T, nvbench::enum_type<BytesPerThread>>)
{
  if constexpr (BytesPerThread % sizeof(T) != 0)
  {
    state.skip("Skipping: the bytes per thread are not a whole number of items.");
  }
  else
  {
    constexpr int items_per_thread = BytesPerThread / sizeof(T);
    constexpr int tile_items       = threads_per_block * items_per_thread;

    // The output starts Alignment bytes after an allocation that is aligned to more than every vector, thus it is
    // aligned to exactly Alignment bytes
    const auto alignment = static_cast<cuda::std::size_t>(state.get_int64("Alignment"));
    if (alignment < alignof(T))
    {
      state.skip("Skipping: the alignment is less than the alignment of T.");
      return;
    }
    const auto num_tiles = static_cast<cuda::std::size_t>(state.get_int64("Elements{io}")) / tile_items;
    thrust::device_vector<nvbench::uint8_t> buffer((num_tiles * tile_items * sizeof(T)) + alignment);
    T* output = reinterpret_cast<T*>(thrust::raw_pointer_cast(buffer.data()) + alignment);

    state.add_element_count(num_tiles * tile_items);
    state.add_global_memory_writes<T>(num_tiles * tile_items, "Output");

    const auto& kernel = store_kernel<items_per_thread, T>;
    const int num_SMs  = state.get_device().value().get_number_of_sms(); // NOLINT(bugprone-unchecked-optional-access)
    int max_blocks_per_SM = 0;
    NVBENCH_CUDA_CALL_NOEXCEPT(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, threads_per_block, 0));
    // NVBENCH_CUDA_CALL_NOEXCEPT swallows a failed occupancy query, which would leave
    // max_blocks_per_SM at 0 and turn the launch into a bare cudaErrorInvalidConfiguration.
    if (max_blocks_per_SM == 0)
    {
      state.skip("Skipping: no resident blocks for this type on this device.");
      return;
    }
    const auto grid_size =
      static_cast<unsigned int>(cuda::std::min(num_tiles, static_cast<cuda::std::size_t>(max_blocks_per_SM * num_SMs)));
    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
      kernel<<<grid_size, threads_per_block, 0, launch.get_stream()>>>(output, num_tiles);
    });
  }
}

NVBENCH_BENCH_TYPES(direct_blocked_vectorized, NVBENCH_TYPE_AXES(value_types, bytes_per_thread))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "BytesPerThread{ct}"})
  .add_int64_power_of_two_axis("Alignment", nvbench::range(0, 5, 1))
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
