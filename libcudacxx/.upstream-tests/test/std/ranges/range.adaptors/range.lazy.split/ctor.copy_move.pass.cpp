//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Test the implicitly-generated copy and move constructors since `lazy_split_view` has non-trivial members.

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/utility>
#include "types.h"

constexpr bool test() {
  // Can copy `lazy_split_view`.
  {
    // Forward range.
    {
      cuda::std::ranges::lazy_split_view<cuda::std::string_view, cuda::std::string_view> v1("abc def", " ");
      auto v2 = v1;
      assert(v2.base() == v1.base());
    }

    // Input range.
    {
      SplitViewInput v1("abc def", ' ');
      auto v2 = v1;
      assert(v2.base() == v1.base());
    }
  }

  // Can move `lazy_split_view`.
  {
    // Forward range.
    {
      cuda::std::string_view base = "abc def";
      cuda::std::ranges::lazy_split_view<cuda::std::string_view, cuda::std::string_view> v1(base, " ");
      auto v2 = cuda::std::move(v1);
      assert(v2.base() == base);
    }

    // Input range.
    {
      InputView base("abc def");
      SplitViewInput v1(base, ' ');
      auto v2 = cuda::std::move(v1);
      assert(v2.base() == base);
    }
  }

  // `non-propagating-cache` is not copied.
  {
    SplitViewInput v1("abc def ghi", ' ');
    auto outer_iter1 = v1.begin();
    ++outer_iter1;
    auto val1 = *outer_iter1;
    auto i1 = val1.begin();
    assert(*i1 == 'd');
    ++i1;
    assert(*i1 == 'e');

    auto v2 = v1;
    auto val2 = *v2.begin();
    auto i2 = val2.begin();
    assert(*i2 == 'a');
    ++i2;
    assert(*i2 == 'b');
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
