//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>

// template<class ExecutionPolicy, class ForwardIterator, class Size, class Function>
//   ForwardIterator for_each_n(ExecutionPolicy&& exec, ForwardIterator first, Size n,
//                              Function f);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"

EXECUTION_POLICY_SFINAE_TEST(for_each_n);

static_assert(sfinae_test_for_each_n<int, int*, int, bool (*)(int)>);
static_assert(!sfinae_test_for_each_n<cuda::std::execution::parallel_policy_host, int*, int, bool (*)(int)>);

struct test_functor
{
  __host__ __device__ void operator()(Bool& called) const noexcept
  {
    assert(!called);
    called = true;
  }
};

struct convert_to_bool
{
  __host__ __device__ bool operator()(Bool& b) const noexcept
  {
    return b;
  }
};

STATIC_TEST_GLOBAL_VAR constexpr size_t num_tests     = 4;
STATIC_TEST_GLOBAL_VAR constexpr int sizes[num_tests] = {0, 1, 20, 1000};

STATIC_TEST_GLOBAL_VAR Bool data[1000];

template <class Iter>
struct Test
{
  template <class Policy>
  __host__ __device__ void operator()(Policy&& policy)
  {
    for (size_t i = 0; i < num_tests; ++i)
    {
      cuda::std::fill(cuda::std::begin(data), cuda::std::end(data), Bool{false});
      cuda::std::for_each_n(policy, Iter(data), sizes[i], test_functor{});
      assert(cuda::std::all_of(data, data + sizes[i], convert_to_bool{}));
    }
  }
};

int main(int, char**)
{
  types::for_each(types::forward_iterator_list<Bool*>{}, TestIteratorWithPolicies<Test>{});

  return 0;
}
