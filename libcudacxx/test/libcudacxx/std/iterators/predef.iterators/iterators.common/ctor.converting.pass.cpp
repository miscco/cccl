//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// template<class I2, class S2>
//   requires convertible_to<const I2&, I> && convertible_to<const S2&, S>
//     constexpr common_iterator(const common_iterator<I2, S2>& x);

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  struct Base
  {};
  struct Derived : Base
  {};

  using BaseIt    = cuda::std::common_iterator<Base*, const Base*>;
  using DerivedIt = cuda::std::common_iterator<Derived*, const Derived*>;
  static_assert(cuda::std::is_convertible_v<DerivedIt, BaseIt>); // Derived* to Base*
  static_assert(!cuda::std::is_constructible_v<DerivedIt, BaseIt>); // Base* to Derived*

  Derived a[10] = {};
  DerivedIt it  = DerivedIt(a); // the iterator type
  BaseIt jt     = BaseIt(it);
  assert(jt == BaseIt(a));

  it = DerivedIt((const Derived*) a); // the sentinel type
  jt = BaseIt(it);
  assert(jt == BaseIt(a));

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  return 0;
}
