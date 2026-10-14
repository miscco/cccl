//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define _LIBCUDACXX_TESTING_FALLBACK_IS_CONSTRUCTIBLE
#include <cuda/std/type_traits>

#include "test_macros.h"

struct Base
{};
struct Derived : Base
{};

struct RefDerived;
struct RefBase
{
  TEST_FUNC explicit operator RefDerived&() const;
};
struct RefDerived : RefBase
{};

struct ValueDerived;
struct ValueBase
{
  TEST_FUNC explicit operator ValueDerived() const;
};
struct ValueDerived : ValueBase
{};

struct ExplicitDerived : Base
{
  TEST_FUNC explicit ExplicitDerived(Base);
};
struct ImplicitDerived : Base
{
  TEST_FUNC ImplicitDerived(Base);
};

template <bool Expected, class To, class From>
TEST_FUNC constexpr void check()
{
  static_assert(cuda::std::is_constructible_v<To, From> == Expected);
  static_assert(cuda::std::__cccl_is_constructible<To, From>::value == Expected);
}

int main(int, char**)
{
#if !TEST_COMPILER(GCC) || TEST_STD_VER < 2020
  // GCC's C++20 reference initialization accepts aggregate construction here.
  check<false, Derived&, Base&>();
  check<false, const Derived&, const Base&>();
  check<false, const Derived&, Base>();
  check<false, Derived&&, Base&&>();
  check<false, Derived&&, Base>();
#endif
  check<true, RefDerived&, const RefBase&>();
  check<true, const RefDerived&, RefBase>();
#if !TEST_COMPILER(GCC) || TEST_STD_VER < 2020
  check<false, RefDerived&&, RefBase>();
#endif
  check<false, ValueDerived&, ValueBase>();
  check<true, const ValueDerived&, ValueBase>();
  check<true, ValueDerived&&, ValueBase>();
  check<false, const ExplicitDerived&, Base>();
  check<false, ExplicitDerived&&, Base>();
  check<true, const ImplicitDerived&, Base>();
  check<true, ImplicitDerived&&, Base>();
  return 0;
}
