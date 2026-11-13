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

// Groups of one thread, of threads within a warp with static and dynamic unit counts, of threads whose lanes are not
// contiguous, of warps within a block, and the warp and the block themselves
using group_list =
  c2h::type_list<this_thread_group,
                 this_warp_group,
                 threads_in_warp_group<8, true>,
                 threads_in_warp_group<8, false>,
                 even_odd_lanes_group,
                 warps_in_block_group<2>,
                 this_block_group>;

using arrangement_list = c2h::type_list<cudax::coop::blocked_t, cudax::coop::striped_t<1>, cudax::coop::striped_t<2>>;

using item_type_list = c2h::type_list<cuda::std::int8_t, cuda::std::int32_t>;

C2H_TEST(
  "load loads the tile of every group into its threads", "[load][groups]", group_list, arrangement_list, item_type_list)
{
  using group_t       = c2h::get<0, TestType>;
  using arrangement_t = c2h::get<1, TestType>;
  using T             = c2h::get<2, TestType>;
  CAPTURE(c2h::type_name<group_t>(), c2h::type_name<arrangement_t>(), c2h::type_name<T>());
  test_load_offsets<group_t, arrangement_t, 4, T, T>(pointer_input{});
}

C2H_TEST("load loads the tile of every group into its threads of a dynamically sized block",
         "[load][groups]",
         group_list,
         arrangement_list)
{
  using group_t       = c2h::get<0, TestType>;
  using arrangement_t = c2h::get<1, TestType>;
  CAPTURE(c2h::type_name<group_t>(), c2h::type_name<arrangement_t>());
  test_load_offsets<group_t, arrangement_t, 4, cuda::std::int16_t, cuda::std::int16_t, false>(pointer_input{});
}
