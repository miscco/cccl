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

template <cuda::std::size_t VecSize, int ItemsPerThread>
struct striped_config
{
  using arrangement_t                   = cudax::coop::striped_t<VecSize>;
  static constexpr int items_per_thread = ItemsPerThread;
};

// Single items, three vectors of 2 items, two vectors of 4 items, and a single vector of 8 items per thread
using striped_config_list =
  c2h::type_list<striped_config<1, 4>, striped_config<2, 6>, striped_config<4, 8>, striped_config<8, 8>>;

C2H_TEST(
  "load/striped loads striped vectors from every alignment", "[load][striped]", item_type_list, striped_config_list)
{
  using T             = c2h::get<0, TestType>;
  using config_t      = c2h::get<1, TestType>;
  using arrangement_t = typename config_t::arrangement_t;
  constexpr int items = config_t::items_per_thread;
  CAPTURE(c2h::type_name<T>(), items, c2h::type_name<arrangement_t>());
  test_load_offsets<this_block_group, arrangement_t, items, T, T>(pointer_input{});
}

C2H_TEST("load/striped loads striped vectors in a dynamically sized block", "[load][striped]", item_type_list)
{
  using T             = c2h::get<0, TestType>;
  using arrangement_t = cudax::coop::striped_t<2>;
  CAPTURE(c2h::type_name<T>());
  test_load_offsets<this_block_group, arrangement_t, 4, T, T, false>(pointer_input{});
}

C2H_TEST("load/striped arrangement objects name their arrangements", "[load][striped]")
{
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(cudax::coop::striped), const cudax::coop::striped_t<1>>);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(cudax::coop::striped_vectors<4>), const cudax::coop::striped_t<4>>);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(cudax::coop::blocked), const cudax::coop::blocked_t>);
}
