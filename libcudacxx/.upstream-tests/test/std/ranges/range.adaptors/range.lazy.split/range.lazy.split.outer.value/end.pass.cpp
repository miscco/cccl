//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// cuda::std::ranges::lazy_split_view::outer-iterator::value_type::end()

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include "../types.h"

constexpr bool test() {
  // `View` is a forward range.
  {
    CopyableView input("a");

    // Non-const.
    {
      SplitViewCopyable v(input, "b");
      auto val = *v.begin();

      static_assert(cuda::std::same_as<decltype(val.end()), cuda::std::default_sentinel_t>);
      static_assert(noexcept(val.end()));
      [[maybe_unused]] auto e = val.end();
    }

    // Const.
    {
      SplitViewCopyable v(input, "b");
      const auto val = *v.begin();

      static_assert(cuda::std::same_as<decltype(val.end()), cuda::std::default_sentinel_t>);
      static_assert(noexcept(val.end()));
      [[maybe_unused]] auto e = val.end();
    }
  }

  // `View` is an input range.
  {
    InputView input("a");

    // Non-const.
    {
      SplitViewInput v(input, 'b');
      auto val = *v.begin();

      static_assert(cuda::std::same_as<decltype(val.end()), cuda::std::default_sentinel_t>);
      static_assert(noexcept(val.end()));
      [[maybe_unused]] auto e = val.end();
    }

    // Const.
    {
      SplitViewInput v(input, 'b');
      const auto val = *v.begin();

      static_assert(cuda::std::same_as<decltype(val.end()), cuda::std::default_sentinel_t>);
      static_assert(noexcept(val.end()));
      [[maybe_unused]] auto e = val.end();
    }
  }

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test(), "");

  return 0;
}
