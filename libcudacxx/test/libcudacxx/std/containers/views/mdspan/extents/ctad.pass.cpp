//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++11, c++14

// <mdspan>

// template <class... Integrals>
// explicit extents(Integrals...) -> see below;
//   Constraints: (is_convertible_v<Integrals, size_t> && ...) is true.
//
// Remarks: The deduced type is dextents<size_t, sizeof...(Integrals)>.

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include "../ConvertibleToIntegral.h"
#include "test_macros.h"

template <class E, class Expected>
__host__ __device__ constexpr void test(E e, Expected expected)
{
  ASSERT_SAME_TYPE(E, Expected);
  assert(e == expected);
}

__host__ __device__ constexpr bool test()
{
  constexpr cuda::std::size_t D = cuda::std::dynamic_extent;

  test(cuda::std::extents(), cuda::std::extents<size_t>());
  test(cuda::std::extents(1), cuda::std::extents<cuda::std::size_t, D>(1));
  test(cuda::std::extents(1, 2u), cuda::std::extents<cuda::std::size_t, D, D>(1, 2u));
  test(cuda::std::extents(1, 2u, 3, 4, 5, 6, 7, 8, 9),
       cuda::std::extents<cuda::std::size_t, D, D, D, D, D, D, D, D, D>(1, 2u, 3, 4, 5, 6, 7, 8, 9));
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}