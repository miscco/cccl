//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// class cuda::std::ranges::lazy_split_view::outer-iterator::value_type;

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include "../types.h"

using V = ValueTypeForward;
static_assert(cuda::std::ranges::forward_range<V>);
static_assert(cuda::std::ranges::view<V>);

static_assert(cuda::std::is_base_of_v<cuda::std::ranges::view_interface<ValueTypeForward>, ValueTypeForward>);

constexpr bool test() {
  // empty()
  {
    {
      SplitViewForward v("abc def", " ");
      auto val = *v.begin();
      assert(!val.empty());
    }

    {
      SplitViewForward v;
      auto val = *v.begin();
      assert(val.empty());
    }
  }

  // operator bool()
  {
    {
      SplitViewForward v("abc def", " ");
      auto val = *v.begin();
      assert(val);
    }

    {
      SplitViewForward v;
      auto val = *v.begin();
      assert(!val);
    }
  }

  // front()
  {
    SplitViewForward v("abc def", " ");
    auto val = *v.begin();
    assert(val.front() == 'a');
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
