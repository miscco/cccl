//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_ptr.h>

#include <cuda/iterator>

#include "load_testing.cuh"

template <class Arrangement, int ItemsPerThread>
struct load_config
{
  using arrangement_t                   = Arrangement;
  static constexpr int items_per_thread = ItemsPerThread;
};

// A single vector of 8 items and two vectors of 2 items per thread
using config_list = c2h::type_list<load_config<cudax::coop::blocked_t, 8>, load_config<cudax::coop::striped_t<2>, 4>>;

using modifier_list =
  c2h::enum_type_list<cub::CacheLoadModifier,
                      cub::LOAD_DEFAULT,
                      cub::LOAD_CA,
                      cub::LOAD_CG,
                      cub::LOAD_CS,
                      cub::LOAD_CV,
                      cub::LOAD_LDG,
                      cub::LOAD_VOLATILE>;

using modified_type_list =
  c2h::type_list<cuda::std::int8_t, cuda::std::int32_t, cuda::std::int64_t, padded_i16_i8, i32x4>;

C2H_TEST("load loads cache modified inputs from every alignment",
         "[load][cache_modifier]",
         modifier_list,
         modified_type_list,
         config_list)
{
  constexpr auto modifier = c2h::get<0, TestType>::value;
  using T                 = c2h::get<1, TestType>;
  using config_t          = c2h::get<2, TestType>;
  CAPTURE(modifier, c2h::type_name<T>(), c2h::type_name<config_t>());
  test_load_offsets<this_block_group, typename config_t::arrangement_t, config_t::items_per_thread, T, T>(
    cache_modified_input<modifier>{});
}

template <class InputT, class T>
struct conversion
{
  using input_t = InputT;
  using item_t  = T;
};

using conversion_list =
  c2h::type_list<conversion<cuda::std::int8_t, cuda::std::int32_t>,
                 conversion<cuda::std::uint8_t, cuda::std::int16_t>,
                 conversion<cuda::std::int16_t, cuda::std::int32_t>,
                 conversion<cuda::std::int32_t, cuda::std::int64_t>,
                 conversion<cuda::std::int32_t, cuda::std::uint32_t>,
                 conversion<cuda::std::int32_t, float>,
                 conversion<float, cuda::std::int32_t>,
                 conversion<cuda::std::int64_t, cuda::std::int32_t>,
                 conversion<float, double>,
                 conversion<double, float>>;

C2H_TEST("load converts the items from every alignment", "[load][conversion]", conversion_list, config_list)
{
  using input_t  = typename c2h::get<0, TestType>::input_t;
  using T        = typename c2h::get<0, TestType>::item_t;
  using config_t = c2h::get<1, TestType>;
  CAPTURE(c2h::type_name<input_t>(), c2h::type_name<T>(), c2h::type_name<config_t>());
  test_load_offsets<this_block_group, typename config_t::arrangement_t, config_t::items_per_thread, T, input_t>(
    pointer_input{});
}

using modified_conversion_modifier_list =
  c2h::enum_type_list<cub::CacheLoadModifier, cub::LOAD_CS, cub::LOAD_LDG, cub::LOAD_VOLATILE>;

// Conversions between items of the same size must not copy the words of the input
using modified_conversion_list =
  c2h::type_list<conversion<cuda::std::int8_t, cuda::std::int32_t>,
                 conversion<cuda::std::int32_t, float>,
                 conversion<cuda::std::int32_t, cuda::std::int64_t>,
                 conversion<cuda::std::int64_t, cuda::std::int32_t>>;

C2H_TEST("load converts cache modified items from every alignment",
         "[load][conversion][cache_modifier]",
         modified_conversion_modifier_list,
         modified_conversion_list,
         config_list)
{
  constexpr auto modifier = c2h::get<0, TestType>::value;
  using input_t           = typename c2h::get<1, TestType>::input_t;
  using T                 = typename c2h::get<1, TestType>::item_t;
  using config_t          = c2h::get<2, TestType>;
  CAPTURE(modifier, c2h::type_name<input_t>(), c2h::type_name<T>(), c2h::type_name<config_t>());
  test_load_offsets<this_block_group, typename config_t::arrangement_t, config_t::items_per_thread, T, input_t>(
    cache_modified_input<modifier>{});
}

struct volatile_input
{
  template <class T>
  const volatile T* operator()(const T* ptr) const
  {
    return ptr;
  }
};

struct device_ptr_input
{
  template <class T>
  thrust::device_ptr<const T> operator()(const T* ptr) const
  {
    return thrust::device_pointer_cast(ptr);
  }
};

C2H_TEST("load loads volatile items and contiguous iterators from every alignment", "[load][iterator]", config_list)
{
  using config_t      = c2h::get<0, TestType>;
  using arrangement_t = typename config_t::arrangement_t;
  constexpr int items = config_t::items_per_thread;
  CAPTURE(c2h::type_name<config_t>());
  test_load_offsets<this_block_group, arrangement_t, items, cuda::std::int16_t, cuda::std::int16_t>(volatile_input{});
  test_load_offsets<this_block_group, arrangement_t, items, cuda::std::int16_t, cuda::std::int16_t>(device_ptr_input{});
}

C2H_TEST("load loads the items of an iterator that is not contiguous", "[load][iterator]", config_list)
{
  using config_t      = c2h::get<0, TestType>;
  using arrangement_t = typename config_t::arrangement_t;
  constexpr int items = config_t::items_per_thread;
  constexpr int first = 7;
  CAPTURE(c2h::type_name<config_t>());

  std::vector<int> expected(block_threads * items);
  for (int k = 0; k < block_threads * items; ++k)
  {
    expected[static_cast<std::size_t>(k)] = first + k;
  }
  const cuda::stream stream{cuda::devices[0]};
  c2h::device_vector<cuda::std::int64_t> d_out(expected.size(), -1);
  cuda::launch(stream,
               make_load_config(),
               load_kernel<this_block_group, arrangement_t, items>{},
               cuda::counting_iterator<int>{first},
               thrust::raw_pointer_cast(d_out.data()));
  stream.sync();
  verify_load<this_block_group, arrangement_t, items>(expected, d_out, block_threads * items, cuda::std::int64_t{-1});
}
