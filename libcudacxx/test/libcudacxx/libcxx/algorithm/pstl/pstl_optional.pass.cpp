//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
//

// <cuda/std/algorithm>

#include <cuda/std/__execution_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using optional = cuda::std::__pstl_optional<int>;
  {
    static_assert(!cuda::std::is_default_constructible<optional>::value, "");
    static_assert(cuda::std::is_trivially_copy_constructible<optional>::value, "");
    static_assert(cuda::std::is_trivially_move_constructible<optional>::value, "");
    static_assert(cuda::std::is_trivially_copy_assignable<optional>::value, "");
    static_assert(cuda::std::is_trivially_move_assignable<optional>::value, "");

    // No conversions
    static_assert(cuda::std::is_constructible<optional, short>::value, "");
  }
  {
    const optional from_abort{cuda::std::__pstl_abort{}};
    assert(!from_abort);

    const int val = 42;
    const optional from_lvalue{val};
    assert(from_lvalue);
    assert(from_lvalue.__val_ == 42);

    const optional from_rvalue{1337};
    assert(from_rvalue);
    assert(from_rvalue.__val_ == 1337);
  }

  using empty_optional = cuda::std::__pstl_optional<cuda::std::__empty>;
  {
    static_assert(!cuda::std::is_default_constructible<empty_optional>::value, "");
    static_assert(cuda::std::is_trivially_copy_constructible<empty_optional>::value, "");
    static_assert(cuda::std::is_trivially_move_constructible<empty_optional>::value, "");
    static_assert(cuda::std::is_trivially_copy_assignable<empty_optional>::value, "");
    static_assert(cuda::std::is_trivially_move_assignable<empty_optional>::value, "");
    static_assert(cuda::std::is_trivially_destructible<empty_optional>::value, "");
  }
  {
    const empty_optional from_abort{cuda::std::__pstl_abort{}};
    assert(!from_abort);

    const cuda::std::__empty val{};
    const empty_optional from_lvalue{val};
    assert(from_lvalue);

    const empty_optional from_rvalue{cuda::std::__empty{}};
    assert(from_rvalue);
  }

  return 0;
}
