//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// class cuda::std::ranges::lazy_split_view;

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include "types.h"

using V = SplitViewForward;

static_assert(cuda::std::is_base_of_v<cuda::std::ranges::view_interface<SplitViewForward>, SplitViewForward>);

constexpr bool test() {
  using namespace cuda::std::string_view_literals;

  // empty()
  {
    {
      cuda::std::ranges::lazy_split_view v("abc def", " ");
      assert(!v.empty());
    }

    {
      // Note: an empty string literal would still produce a non-empty output because the terminating zero is treated as
      // a separate character; hence the use of `string_view`.
      cuda::std::ranges::lazy_split_view v(""sv, "");
      assert(v.empty());
    }
  }

  // operator bool()
  {
    {
      cuda::std::ranges::lazy_split_view v("abc", "");
      assert(v);
    }

    {
      // Note: an empty string literal would still produce a non-empty output because the terminating zero is treated as
      // a separate character; hence the use of `string_view`.
      cuda::std::ranges::lazy_split_view v(""sv, "");
      assert(!v);
    }
  }

  // front()
  {
    SplitViewForward v("abc", "");
    assert(*(v.front()).begin() == 'a');
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
