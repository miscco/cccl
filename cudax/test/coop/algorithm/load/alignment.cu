//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/iterator>

#include "load_testing.cuh"

// Loads the tile with vector_aligned if is_vector_aligned holds for it, and with element_aligned otherwise
template <class Arrangement, int N>
struct hoisted_load_kernel
{
  template <class Config, class InputIt, class T>
  __device__ void operator()(Config config, InputIt src, T* out, int* is_aligned) const
  {
    const cudax::coop::this_block block{config};
    const unsigned rank = cuda::gpu_thread.rank_as<unsigned>(block);
    T items[N];
    const bool vector_aligned = cudax::coop::is_vector_aligned(Arrangement{}, src, items);
    if (vector_aligned)
    {
      cudax::coop::load(block, Arrangement{}, src, items, cudax::coop::vector_aligned);
    }
    else
    {
      cudax::coop::load(block, Arrangement{}, src, items, cudax::coop::element_aligned);
    }
    for (int i = 0; i < N; ++i)
    {
      out[(rank * N) + i] = items[i];
    }
    if (rank == 0)
    {
      *is_aligned = vector_aligned;
    }
  }
};

// The widest vectors of a load in bytes, before sm_100 and from sm_100 on, which only sm_100 and newer access at once
// when they have 32 bytes. 0 if the load reads no vectors
template <class T, class Arrangement, int N, class MakeIt, int BeforeSm100, int FromSm100>
struct widest_vectors
{
  using item_t                          = T;
  using arrangement_t                   = Arrangement;
  using make_it_t                       = MakeIt;
  static constexpr int items_per_thread = N;
  static constexpr int before_sm100     = BeforeSm100;
  static constexpr int from_sm100       = FromSm100;
};

// The vectors hold the items of a thread (blocked) or VecSize items (striped), up to 16 bytes, or 32 bytes from sm_100
// on without a cache modifier or with LOAD_LDG; LOAD_VOLATILE up to 8 bytes. Vectors are at least 4 bytes and wider
// than the items
using widest_vectors_list = c2h::type_list<
  widest_vectors<cuda::std::int32_t, cudax::coop::striped_t<4>, 8, pointer_input, 16, 16>,
  widest_vectors<cuda::std::int32_t, cudax::coop::blocked_t, 8, pointer_input, 16, 32>,
  widest_vectors<cuda::std::int64_t, cudax::coop::striped_t<4>, 8, pointer_input, 16, 32>,
  widest_vectors<cuda::std::int8_t, cudax::coop::blocked_t, 4, pointer_input, 4, 4>,
  widest_vectors<cuda::std::int8_t, cudax::coop::blocked_t, 2, pointer_input, 0, 0>,
  widest_vectors<cuda::std::int32_t, cudax::coop::striped_t<1>, 4, pointer_input, 0, 0>,
  widest_vectors<cuda::std::int32_t, cudax::coop::blocked_t, 3, pointer_input, 0, 0>,
  widest_vectors<padded_i16_i8, cudax::coop::blocked_t, 2, pointer_input, 8, 8>,
  widest_vectors<padded_i16_i8, cudax::coop::blocked_t, 3, pointer_input, 4, 4>,
  widest_vectors<int4, cudax::coop::blocked_t, 2, pointer_input, 0, 32>,
  widest_vectors<cuda::std::int32_t, cudax::coop::blocked_t, 8, cache_modified_input<cub::LOAD_CS>, 16, 16>,
  widest_vectors<cuda::std::int32_t, cudax::coop::blocked_t, 8, cache_modified_input<cub::LOAD_LDG>, 16, 32>,
  widest_vectors<cuda::std::int32_t, cudax::coop::blocked_t, 8, cache_modified_input<cub::LOAD_VOLATILE>, 8, 8>,
  widest_vectors<cuda::std::int32_t, cudax::coop::blocked_t, 1, cache_modified_input<cub::LOAD_CG>, 0, 0>>;

C2H_TEST("is_vector_aligned tests the alignment of the widest vectors of a load",
         "[load][alignment]",
         widest_vectors_list)
{
  using case_t        = c2h::get<0, TestType>;
  using T             = typename case_t::item_t;
  using arrangement_t = typename case_t::arrangement_t;
  constexpr int items = case_t::items_per_thread;
  CAPTURE(c2h::type_name<case_t>());

  int major = 0;
  REQUIRE(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess);
  const int widest = major >= 10 ? case_t::from_sm100 : case_t::before_sm100;

  const cuda::stream stream{cuda::devices[0]};
  c2h::device_vector<int> d_is_aligned(1, -1);
  for (int byte_offset = 0; byte_offset <= 64; byte_offset += static_cast<int>(alignof(T)))
  {
    CAPTURE(byte_offset);
    const offset_input<T> input(block_threads * items, byte_offset);
    c2h::device_vector<T> d_out(static_cast<std::size_t>(block_threads * items), make_item<T>(-1));
    cuda::launch(
      stream,
      make_load_config(),
      hoisted_load_kernel<arrangement_t, items>{},
      typename case_t::make_it_t{}(input.ptr),
      thrust::raw_pointer_cast(d_out.data()),
      thrust::raw_pointer_cast(d_is_aligned.data()));
    stream.sync();
    verify_load<this_block_group, arrangement_t, items>(input.items, d_out, block_threads * items, make_item<T>(-1));
    const bool expected = widest != 0 && reinterpret_cast<cuda::std::uintptr_t>(input.ptr) % widest == 0;
    REQUIRE(d_is_aligned[0] == static_cast<int>(expected));
  }
}

C2H_TEST("is_vector_aligned is false for iterators that are not contiguous", "[load][alignment]")
{
  const cuda::stream stream{cuda::devices[0]};
  c2h::device_vector<int> d_out(block_threads * 4, -1);
  c2h::device_vector<int> d_is_aligned(1, -1);
  cuda::launch(
    stream,
    make_load_config(),
    hoisted_load_kernel<cudax::coop::blocked_t, 4>{},
    cuda::counting_iterator<int>{0},
    thrust::raw_pointer_cast(d_out.data()),
    thrust::raw_pointer_cast(d_is_aligned.data()));
  stream.sync();
  REQUIRE(d_is_aligned[0] == 0);
  std::vector<int> expected(block_threads * 4);
  for (int k = 0; k < block_threads * 4; ++k)
  {
    expected[static_cast<std::size_t>(k)] = k;
  }
  verify_load<this_block_group, cudax::coop::blocked_t, 4>(expected, d_out, block_threads * 4, -1);
}

template <cuda::std::size_t Alignment>
struct overaligned_flag
{
  static constexpr cuda::std::size_t alignment = Alignment;

  __device__ auto operator()() const
  {
    return cudax::coop::overaligned<Alignment>;
  }
};

struct element_aligned_flag
{
  static constexpr cuda::std::size_t alignment = 1;

  __device__ auto operator()() const
  {
    return cudax::coop::element_aligned;
  }
};

template <class Arrangement, int N, class Flag>
struct flag_load_kernel
{
  template <class Config, class InputIt, class T>
  __device__ void operator()(Config config, InputIt src, T* out) const
  {
    const cudax::coop::this_block block{config};
    const unsigned rank = cuda::gpu_thread.rank_as<unsigned>(block);
    T items[N];
    cudax::coop::load(block, Arrangement{}, src, items, Flag{}());
    for (int i = 0; i < N; ++i)
    {
      out[(rank * N) + i] = items[i];
    }
  }
};

using flag_list =
  c2h::type_list<element_aligned_flag,
                 overaligned_flag<2>,
                 overaligned_flag<4>,
                 overaligned_flag<8>,
                 overaligned_flag<16>,
                 overaligned_flag<32>,
                 overaligned_flag<256>>;

using flag_type_list =
  c2h::type_list<cuda::std::int8_t, cuda::std::int16_t, cuda::std::int32_t, double, padded_i16_i8, i32x4>;

using flag_arrangement_list = c2h::type_list<cudax::coop::blocked_t, cudax::coop::striped_t<2>>;

C2H_TEST("load relies on the alignment that a flag states",
         "[load][alignment]",
         flag_list,
         flag_type_list,
         flag_arrangement_list)
{
  using flag_t        = c2h::get<0, TestType>;
  using T             = c2h::get<1, TestType>;
  using arrangement_t = c2h::get<2, TestType>;
  constexpr int items = 8;
  CAPTURE(c2h::type_name<flag_t>(), c2h::type_name<T>(), c2h::type_name<arrangement_t>());
  const cuda::stream stream{cuda::devices[0]};
  for (int byte_offset = 0; byte_offset <= 32; byte_offset += static_cast<int>(alignof(T)))
  {
    const offset_input<T> input(block_threads * items, byte_offset);
    // The flag states the alignment of the tile, which must hold
    if (reinterpret_cast<cuda::std::uintptr_t>(input.ptr) % flag_t::alignment != 0)
    {
      continue;
    }
    CAPTURE(byte_offset);
    c2h::device_vector<T> d_out(static_cast<std::size_t>(block_threads * items), make_item<T>(-1));
    cuda::launch(stream,
                 make_load_config(),
                 flag_load_kernel<arrangement_t, items, flag_t>{},
                 input.ptr,
                 thrust::raw_pointer_cast(d_out.data()));
    stream.sync();
    verify_load<this_block_group, arrangement_t, items>(input.items, d_out, block_threads * items, make_item<T>(-1));
  }
}
