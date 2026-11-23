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

using item_type_list =
  c2h::type_list<cuda::std::int8_t,
                 cuda::std::int16_t,
                 cuda::std::int32_t,
                 cuda::std::int64_t,
                 double,
                 padded_i16_i8,
                 padded_i32_i8,
                 i32x4,
                 int4>;

// 3 and 7 items allow only narrow or no vectors, 16 items of 2 bytes or more allow 32 byte vectors
using items_per_thread_list = c2h::enum_type_list<int, 1, 3, 4, 7, 16>;

C2H_TEST("store/blocked stores the items of a thread to every alignment",
         "[store][blocked]",
         item_type_list,
         items_per_thread_list)
{
  using T             = c2h::get<0, TestType>;
  constexpr int items = c2h::get<1, TestType>::value;
  CAPTURE(c2h::type_name<T>(), items);
  test_store_offsets<this_block_group, cudax::coop::blocked_t, items, T, T>();
}

using striped_list = c2h::type_list<cudax::coop::striped_t<1>, cudax::coop::striped_t<2>, cudax::coop::striped_t<4>>;

C2H_TEST("store/striped stores the items of a thread to every alignment",
         "[store][striped]",
         c2h::type_list<cuda::std::int8_t, cuda::std::int16_t, float, padded_i32_i8>,
         striped_list)
{
  using T             = c2h::get<0, TestType>;
  using arrangement_t = c2h::get<1, TestType>;
  CAPTURE(c2h::type_name<T>(), c2h::type_name<arrangement_t>());
  test_store_offsets<this_block_group, arrangement_t, 8, T, T>();
  test_store_offsets<this_block_group, arrangement_t, 8, T, T, false>();
}

using group_list =
  c2h::type_list<this_warp_group,
                 this_thread_group,
                 threads_in_warp_group<8, true>,
                 threads_in_warp_group<8, false>,
                 even_odd_lanes_group,
                 warps_in_block_group<2>>;

C2H_TEST("store stores the tile of every group", "[store][groups]", group_list)
{
  using group_t = c2h::get<0, TestType>;
  CAPTURE(c2h::type_name<group_t>());
  test_store_offsets<group_t, cudax::coop::blocked_t, 4, cuda::std::int32_t, cuda::std::int32_t>();
  test_store_offsets<group_t, cudax::coop::striped_t<2>, 4, cuda::std::int16_t, cuda::std::int16_t>();
}

C2H_TEST("store converts the items to the value type of the destination", "[store][conversion]")
{
  test_store_offsets<this_block_group, cudax::coop::blocked_t, 4, float, double>();
  test_store_offsets<this_block_group, cudax::coop::blocked_t, 4, cuda::std::int32_t, cuda::std::int8_t>();
  test_store_offsets<this_warp_group, cudax::coop::striped_t<2>, 4, cuda::std::int64_t, cuda::std::int16_t>();
}

C2H_TEST("store relies on a stated alignment", "[store][alignment]")
{
  const cuda::stream stream{cuda::devices[0]};
  constexpr int items     = 8;
  constexpr int num_items = block_threads * items;
  std::vector<cuda::std::int16_t> input(num_items);
  for (int k = 0; k < num_items; ++k)
  {
    input[static_cast<std::size_t>(k)] = make_item<cuda::std::int16_t>(k);
  }
  const c2h::device_vector<cuda::std::int16_t> d_input(input.begin(), input.end());
  auto run = [&](int byte_offset, auto alignment) {
    CAPTURE(byte_offset);
    offset_output<cuda::std::int16_t> output(num_items, byte_offset, -1);
    cuda::launch(
      stream,
      make_load_config(),
      store_kernel<this_block_group, cudax::coop::blocked_t, items, decltype(alignment)>{},
      thrust::raw_pointer_cast(d_input.data()),
      output.ptr(),
      alignment);
    stream.sync();
    verify_store<this_block_group, cudax::coop::blocked_t, items>(
      input, output.items(), num_items, cuda::std::int16_t{-1});
  };
  run(0, cudax::coop::vector_aligned);
  run(0, cudax::coop::overaligned<8>);
  run(8, cudax::coop::overaligned<8>);
  run(4, cudax::coop::overaligned<4>);
  run(2, cudax::coop::element_aligned);
}
