//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>
#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cuda/experimental/container.cuh>

#include <stdexcept>

#include "helper.h"
#include "types.h"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("cudax::async_mdarray access",
                   "[container][async_mdarray]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env             = typename extract_properties<TestType>::env;
  using Resource        = typename extract_properties<TestType>::resource;
  using Array           = typename extract_properties<TestType>::async_mdarray;
  using T               = typename Array::value_type;
  using reference       = typename Array::reference;
  using const_reference = typename Array::const_reference;
  using pointer         = typename Array::pointer;
  using const_pointer   = typename Array::const_pointer;

  cudax::stream stream{};
  Env env{Resource{}, stream};

  SECTION("cudax::async_mdarray::operator[]")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>()[1ull]), reference>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const Array&>()[1ull]), const_reference>);

    {
      Array vec{env, cuda::std::dims<1>{4}, {T(1), T(42), T(1337), T(0)}};
      auto& res = vec[2];
      CHECK(compare_value<Array::__is_host_only>(res, T(1337)));
      CHECK(static_cast<size_t>(cuda::std::addressof(res) - vec.data()) == 2);
      assign_value<Array::__is_host_only>(res, T(4));

      auto& const_res = cuda::std::as_const(vec)[2];
      CHECK(compare_value<Array::__is_host_only>(const_res, T(4)));
      CHECK(static_cast<size_t>(cuda::std::addressof(const_res) - vec.data()) == 2);
    }
  }

  SECTION("cudax::async_mdarray::data")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>().data()), pointer>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const Array&>().data()), const_pointer>);

    { // Works without allocation
      Array vec{env};
      CHECK(vec.data() == nullptr);
      CHECK(cuda::std::as_const(vec).data() == nullptr);
    }

    { // Works with allocation
      Array vec{env, cuda::std::dims<1>{4}, {T(1), T(42), T(1337), T(0)}};
      CHECK(vec.data() != nullptr);
      CHECK(cuda::std::as_const(vec).data() != nullptr);
      CHECK(cuda::std::as_const(vec).data() == vec.data());
    }
  }
}
