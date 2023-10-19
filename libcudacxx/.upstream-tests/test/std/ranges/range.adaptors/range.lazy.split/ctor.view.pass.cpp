//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr lazy_split_view(View base, Pattern pattern);

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/utility>
#include "types.h"

struct ViewWithCounting : cuda::std::ranges::view_base {
  int* times_copied = nullptr;
  int* times_moved = nullptr;

  constexpr ViewWithCounting(int& copies_ctr, int& moves_ctr) : times_copied(&copies_ctr), times_moved(&moves_ctr) {}

  constexpr ViewWithCounting(const ViewWithCounting& rhs)
    : times_copied(rhs.times_copied)
    , times_moved(rhs.times_moved) {
    ++(*times_copied);
  }
  constexpr ViewWithCounting(ViewWithCounting&& rhs)
    : times_copied(rhs.times_copied)
    , times_moved(rhs.times_moved) {
    ++(*times_moved);
  }

  constexpr const char* begin() const { return nullptr; }
  constexpr const char* end() const { return nullptr; }

  constexpr ViewWithCounting& operator=(const ViewWithCounting&) = default;
  constexpr ViewWithCounting& operator=(ViewWithCounting&&) = default;
  constexpr bool operator==(const ViewWithCounting&) const { return true; }
};
static_assert(cuda::std::ranges::forward_range<ViewWithCounting>);
static_assert(cuda::std::ranges::view<ViewWithCounting>);

constexpr bool test() {
  // Calling the constructor with `(ForwardView, ForwardView)`.
  {
    CopyableView input = "abc def";
    cuda::std::ranges::lazy_split_view<CopyableView, CopyableView> v(input, " ");
    assert(v.base() == input);
  }

  // Calling the constructor with `(InputView, TinyView)`.
  {
    InputView input = "abc def";
    cuda::std::ranges::lazy_split_view<InputView, ForwardTinyView> v(input, ' ');
    // Note: `InputView` isn't equality comparable.
    (void)v;
  }

  // Make sure the arguments are moved, not copied.
  {
    using View = ViewWithCounting;
    using Pattern = ViewWithCounting;

    // Arguments are lvalues.
    {
      int view_copied = 0, view_moved = 0, pattern_copied = 0, pattern_moved = 0;
      View view(view_copied, view_moved);
      Pattern pattern(pattern_copied, pattern_moved);

      cuda::std::ranges::lazy_split_view<View, Pattern> v(view, pattern);
      assert(view_copied == 1); // The local variable is copied into the argument.
      assert(view_moved == 1);
      assert(pattern_copied == 1);
      assert(pattern_moved == 1);
    }

    // Arguments are rvalues.
    {
      int view_copied = 0, view_moved = 0, pattern_copied = 0, pattern_moved = 0;
      cuda::std::ranges::lazy_split_view<View, Pattern> v(
          View(view_copied, view_moved), Pattern(pattern_copied, pattern_moved));
      assert(view_copied == 0);
      assert(view_moved == 1);
      assert(pattern_copied == 0);
      assert(pattern_moved == 1);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
