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

using item_type_list =
  c2h::type_list<cuda::std::int8_t,
                 cuda::std::int16_t,
                 cuda::std::int32_t,
                 cuda::std::int64_t,
                 float,
                 double,
                 padded_i16_i8,
                 padded_i32_i8,
                 i32x4,
                 int4>;

// 3 and 7 items allow only narrow or no vectors, 16 items of 2 bytes or more allow 32 byte vectors
using items_per_thread_list = c2h::enum_type_list<int, 1, 3, 4, 7, 16>;

C2H_TEST("load/blocked loads the items of a thread from every alignment",
         "[load][blocked]",
         item_type_list,
         items_per_thread_list)
{
  using T             = c2h::get<0, TestType>;
  constexpr int items = c2h::get<1, TestType>::value;
  using arrangement_t = cudax::coop::blocked_t;
  CAPTURE(c2h::type_name<T>(), items);
  test_load_offsets<this_block_group, arrangement_t, items, T, T>(pointer_input{});
}

C2H_TEST("load/blocked loads the items of a thread in a dynamically sized block", "[load][blocked]", item_type_list)
{
  using T             = c2h::get<0, TestType>;
  using arrangement_t = cudax::coop::blocked_t;
  CAPTURE(c2h::type_name<T>());
  test_load_offsets<this_block_group, arrangement_t, 4, T, T, false>(pointer_input{});
}
