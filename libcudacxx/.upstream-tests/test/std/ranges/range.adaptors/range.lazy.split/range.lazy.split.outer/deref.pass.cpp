//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr outer-iterator::value-type outer-iterator::operator*() const;

#include <cuda/std/ranges>

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/string>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include "../types.h"

template <class View, class Separator>
constexpr void test_one(Separator sep) {
  using namespace cuda::std::string_literals;
  using namespace cuda::std::string_view_literals;

  View v("abc def ghi"sv, sep);

  // Non-const iterator.
  {
    auto i = v.begin();
    static_assert(!cuda::std::is_reference_v<decltype(*i)>);
    assert(cuda::std::ranges::equal(*i, "abc"s));
    assert(cuda::std::ranges::equal(*(++i), "def"s));
    assert(cuda::std::ranges::equal(*(++i), "ghi"s));
  }

  // Const iterator.
  {
    const auto ci = v.begin();
    static_assert(!cuda::std::is_reference_v<decltype(*ci)>);
    assert(cuda::std::ranges::equal(*ci, "abc"s));
  }
}

constexpr bool test() {
  // `View` is a forward range.
  test_one<SplitViewDiff>(" ");
  test_one<SplitViewInput>(' ');

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
