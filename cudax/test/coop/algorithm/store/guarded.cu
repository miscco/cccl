//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "store_testing.cuh"

using arrangement_list = c2h::type_list<cudax::coop::blocked_t, cudax::coop::striped_t<1>, cudax::coop::striped_t<2>>;

using item_type_list = c2h::type_list<cuda::std::int8_t, cuda::std::int32_t, padded_i16_i8>;

using group_list = c2h::type_list<this_block_group, this_warp_group, even_odd_lanes_group>;

C2H_TEST("store stores the valid items of a tile", "[store][guarded]", arrangement_list, item_type_list, group_list)
{
  using arrangement_t     = c2h::get<0, TestType>;
  using T                 = c2h::get<1, TestType>;
  using group_t           = c2h::get<2, TestType>;
  constexpr int items     = 4;
  constexpr int num_items = block_threads * items;
  constexpr int tile_size = group_t::threads * items;
  CAPTURE(c2h::type_name<arrangement_t>(), c2h::type_name<T>(), c2h::type_name<group_t>());

  const cuda::stream stream{cuda::devices[0]};
  std::vector<T> input(num_items);
  for (int k = 0; k < num_items; ++k)
  {
    input[static_cast<std::size_t>(k)] = make_item<T>(k);
  }
  const c2h::device_vector<T> d_input(input.begin(), input.end());
  for (const int valid_items : {-1, 0, 1, 5, tile_size / 2, tile_size - 1, tile_size, tile_size + 3})
  {
    CAPTURE(valid_items);
    for (int byte_offset = 0; byte_offset <= 4; byte_offset += static_cast<int>(alignof(T)))
    {
      CAPTURE(byte_offset);
      offset_output<T> output(num_items, byte_offset, make_item<T>(-1));
      cuda::launch(
        stream,
        make_load_config(),
        store_kernel<group_t, arrangement_t, items, int>{},
        thrust::raw_pointer_cast(d_input.data()),
        output.ptr(),
        valid_items);
      stream.sync();
      verify_store<group_t, arrangement_t, items>(input, output.items(), valid_items, make_item<T>(-1));
    }
  }
}
