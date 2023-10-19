//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr outer-iterator(Parent& parent, iterator_t<Base> current);
//   requires forward_range<Base>

#include <cuda/std/ranges>

#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include "../types.h"

static_assert(!cuda::std::ranges::forward_range<SplitViewInput>);
static_assert(!cuda::std::is_constructible_v<OuterIterInput, SplitViewInput&, cuda::std::ranges::iterator_t<InputView>>);

constexpr bool test() {
  ForwardView input("abc");
  SplitViewForward v(cuda::std::move(input), " ");
  [[maybe_unused]] OuterIterForward i(v, input.begin());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
