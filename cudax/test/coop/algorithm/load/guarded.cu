//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "load_testing.cuh"

// Loads the first valid_items items of every tile, with oob_default for the other items if WithDefault, and otherwise
// expects them to keep their initial values
template <class Group, class Arrangement, int N, class T, bool WithDefault, class MakeIt>
void test_load_guarded(MakeIt make_it, int byte_offset, int valid_items)
{
  constexpr int num_items = block_threads * N;
  const cuda::stream stream{cuda::devices[0]};
  const offset_input<T> input(num_items, byte_offset);
  const T initial     = make_item<T>(-1);
  const T oob_default = make_item<T>(-2);
  c2h::device_vector<T> d_out(static_cast<std::size_t>(num_items), initial);
  if constexpr (WithDefault)
  {
    cuda::launch(
      stream,
      make_load_config(),
      load_kernel<Group, Arrangement, N, int, T>{},
      make_it(input.ptr),
      thrust::raw_pointer_cast(d_out.data()),
      valid_items,
      oob_default);
  }
  else
  {
    cuda::launch(
      stream,
      make_load_config(),
      load_kernel<Group, Arrangement, N, int>{},
      make_it(input.ptr),
      thrust::raw_pointer_cast(d_out.data()),
      valid_items);
  }
  stream.sync();
  verify_load<Group, Arrangement, N>(input.items, d_out, valid_items, WithDefault ? oob_default : initial);
}

using arrangement_list = c2h::type_list<cudax::coop::blocked_t, cudax::coop::striped_t<1>, cudax::coop::striped_t<2>>;

using item_type_list = c2h::type_list<cuda::std::int8_t, cuda::std::int32_t, padded_i16_i8>;

using group_list = c2h::type_list<this_block_group, this_warp_group, even_odd_lanes_group>;

C2H_TEST("load loads the valid items of a tile", "[load][guarded]", arrangement_list, item_type_list, group_list)
{
  using arrangement_t     = c2h::get<0, TestType>;
  using T                 = c2h::get<1, TestType>;
  using group_t           = c2h::get<2, TestType>;
  constexpr int items     = 4;
  constexpr int tile_size = group_t::threads * items;
  CAPTURE(c2h::type_name<arrangement_t>(), c2h::type_name<T>(), c2h::type_name<group_t>());
  for (const int valid_items : {-1, 0, 1, 5, tile_size / 2, tile_size - 1, tile_size, tile_size + 3})
  {
    CAPTURE(valid_items);
    for (int byte_offset = 0; byte_offset <= 4; byte_offset += static_cast<int>(alignof(T)))
    {
      CAPTURE(byte_offset);
      test_load_guarded<group_t, arrangement_t, items, T, false>(pointer_input{}, byte_offset, valid_items);
      test_load_guarded<group_t, arrangement_t, items, T, true>(pointer_input{}, byte_offset, valid_items);
    }
  }
}

C2H_TEST("load loads the valid items of a cache modified tile", "[load][guarded][cache_modifier]", arrangement_list)
{
  using arrangement_t     = c2h::get<0, TestType>;
  constexpr int items     = 4;
  constexpr int tile_size = block_threads * items;
  CAPTURE(c2h::type_name<arrangement_t>());
  for (const int valid_items : {0, 1, tile_size - 1, tile_size})
  {
    CAPTURE(valid_items);
    test_load_guarded<this_block_group, arrangement_t, items, cuda::std::int32_t, true>(
      cache_modified_input<cub::LOAD_CS>{}, 0, valid_items);
    test_load_guarded<this_block_group, arrangement_t, items, cuda::std::int32_t, false>(
      cache_modified_input<cub::LOAD_LDG>{}, 4, valid_items);
  }
}

C2H_TEST("load accepts valid item counts of every integer type", "[load][guarded]")
{
  constexpr int items = 4;
  using arrangement_t = cudax::coop::striped_t<2>;
  const cuda::stream stream{cuda::devices[0]};
  const offset_input<cuda::std::int32_t> input(block_threads * items, 0);
  for (const int valid_items : {0, 100, block_threads * items})
  {
    CAPTURE(valid_items);
    c2h::device_vector<cuda::std::int32_t> d_out(block_threads * items, -1);
    cuda::launch(
      stream,
      make_load_config(),
      load_kernel<this_block_group, arrangement_t, items, cuda::std::uint64_t>{},
      input.ptr,
      thrust::raw_pointer_cast(d_out.data()),
      static_cast<cuda::std::uint64_t>(valid_items));
    stream.sync();
    verify_load<this_block_group, arrangement_t, items>(input.items, d_out, valid_items, -1);

    c2h::device_vector<cuda::std::int32_t> d_out16(block_threads * items, -1);
    cuda::launch(
      stream,
      make_load_config(),
      load_kernel<this_block_group, arrangement_t, items, cuda::std::int16_t>{},
      input.ptr,
      thrust::raw_pointer_cast(d_out16.data()),
      static_cast<cuda::std::int16_t>(valid_items));
    stream.sync();
    verify_load<this_block_group, arrangement_t, items>(input.items, d_out16, valid_items, -1);
  }
}
