//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

//  lazy_split_view() requires default_initializable<V> && default_initializable<P> = default;

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include "types.h"

struct ThrowingDefaultCtorForwardView : cuda::std::ranges::view_base {
  ThrowingDefaultCtorForwardView() noexcept(false);
  forward_iterator<int*> begin() const;
  forward_iterator<int*> end() const;
};

struct NoDefaultCtorForwardView : cuda::std::ranges::view_base {
  NoDefaultCtorForwardView() = delete;
  forward_iterator<int*> begin() const;
  forward_iterator<int*> end() const;
};

static_assert( cuda::std::is_default_constructible_v<cuda::std::ranges::lazy_split_view<ForwardView, ForwardView>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::lazy_split_view<NoDefaultCtorForwardView, ForwardView>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::lazy_split_view<ForwardView, NoDefaultCtorForwardView>>);

static_assert( cuda::std::is_nothrow_default_constructible_v<cuda::std::ranges::lazy_split_view<ForwardView, ForwardView>>);
static_assert(!cuda::std::is_nothrow_default_constructible_v<ThrowingDefaultCtorForwardView>);
static_assert(!cuda::std::is_nothrow_default_constructible_v<
    cuda::std::ranges::lazy_split_view<ThrowingDefaultCtorForwardView, ForwardView>>);

constexpr bool test() {
  {
    cuda::std::ranges::lazy_split_view<CopyableView, ForwardView> v;
    assert(v.base() == CopyableView());
  }

  {
    cuda::std::ranges::lazy_split_view<CopyableView, ForwardView> v = {};
    assert(v.base() == CopyableView());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
