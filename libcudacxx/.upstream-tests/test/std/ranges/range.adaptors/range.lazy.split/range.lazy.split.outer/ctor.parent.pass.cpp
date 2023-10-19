//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// explicit cuda::std::ranges::lazy_split_view::outer-iterator::outer-iterator(Parent& parent)
//   requires (!forward_range<Base>)

#include <cuda/std/ranges>

#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include "../types.h"

// Verify that the constructor is `explicit`.
static_assert(!cuda::std::is_convertible_v<SplitViewInput&, OuterIterInput>);

static_assert( cuda::std::ranges::forward_range<SplitViewForward>);
static_assert(!cuda::std::is_constructible_v<OuterIterForward, SplitViewForward&>);

constexpr bool test() {
  InputView input;
  SplitViewInput v(input, ForwardTinyView());
  [[maybe_unused]] OuterIterInput i(v);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
