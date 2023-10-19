//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr auto begin();
// constexpr auto begin() const requires forward_range<View> && forward_range<const View>;

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include <cuda/std/utility>
#include "test_iterators.h"
#include "types.h"

template <class View>
concept ConstBeginDisabled = !requires (const View v) {
  { (*v.begin()) };
};

constexpr bool test() {
  // non-const: forward_range<View> && simple-view<View> -> outer-iterator<Const = true>
  // const: forward_range<View> && forward_range<const View> -> outer-iterator<Const = true>
  {
    using V = ForwardView;
    using P = V;

    static_assert(cuda::std::ranges::forward_range<V>);
    static_assert(cuda::std::ranges::forward_range<const V>);
    LIBCPP_STATIC_ASSERT(cuda::std::ranges::__simple_view<V>);
    LIBCPP_STATIC_ASSERT(cuda::std::ranges::__simple_view<P>);

    {
      cuda::std::ranges::lazy_split_view<V, P> v;
      auto it = v.begin();
      static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::forward_iterator_tag>);
      static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), const char&>);
    }

    {
      const cuda::std::ranges::lazy_split_view<V, P> cv;
      auto it = cv.begin();
      static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::forward_iterator_tag>);
      static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), const char&>);
    }
  }

  // non-const: forward_range<View> && !simple-view<View> -> outer-iterator<Const = false>
  // const: forward_range<View> && forward_range<const View> -> outer-iterator<Const = true>
  {
    using V = ForwardDiffView;
    using P = V;

    static_assert(cuda::std::ranges::forward_range<V>);
    static_assert(cuda::std::ranges::forward_range<const V>);
    LIBCPP_STATIC_ASSERT(!cuda::std::ranges::__simple_view<V>);
    LIBCPP_STATIC_ASSERT(!cuda::std::ranges::__simple_view<P>);

    {
      cuda::std::ranges::lazy_split_view<V, P> v;
      auto it = v.begin();
      static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::forward_iterator_tag>);
      static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), char&>);
    }

    {
      const cuda::std::ranges::lazy_split_view<V, P> cv;
      auto it = cv.begin();
      static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::forward_iterator_tag>);
      static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), const char&>);
    }
  }

  // non-const: forward_range<View> && !simple-view<View> -> outer-iterator<Const = false>
  // const: forward_range<View> && !forward_range<const View> -> disabled
  {
    using V = ForwardOnlyIfNonConstView;
    using P = V;
    static_assert(cuda::std::ranges::forward_range<V>);
    static_assert(!cuda::std::ranges::forward_range<const V>);
    LIBCPP_STATIC_ASSERT(!cuda::std::ranges::__simple_view<V>);
    LIBCPP_STATIC_ASSERT(!cuda::std::ranges::__simple_view<P>);

    cuda::std::ranges::lazy_split_view<V, P> v;
    auto it = v.begin();
    static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), const char&>);

    static_assert(ConstBeginDisabled<decltype(v)>);
  }

  // non-const: forward_range<View> && simple-view<View> && !simple-view<Pattern> -> outer-iterator<Const = false>
  // const: forward_range<View> && forward_range<const View> -> outer-iterator<Const = true>
  {
    using V = ForwardView;
    using P = ForwardOnlyIfNonConstView;

    static_assert(cuda::std::ranges::forward_range<V>);
    static_assert(cuda::std::ranges::forward_range<const V>);
    LIBCPP_STATIC_ASSERT(cuda::std::ranges::__simple_view<V>);
    LIBCPP_STATIC_ASSERT(!cuda::std::ranges::__simple_view<P>);

    {
      cuda::std::ranges::lazy_split_view<V, P> v;
      auto it = v.begin();
      static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::forward_iterator_tag>);
      static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), const char&>);
    }

    {
      const cuda::std::ranges::lazy_split_view<V, P> cv;
      auto it = cv.begin();
      static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::forward_iterator_tag>);
      static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), const char&>);
    }
  }

  // non-const: !forward_range<View> && tiny-range<Pattern> -> outer-iterator<Const = false>
  // const: !forward_range<View> -> disabled
  {
    using V = InputView;
    using P = ForwardTinyView;

    static_assert(!cuda::std::ranges::forward_range<V>);
    static_assert(cuda::std::ranges::forward_range<P>);

    cuda::std::ranges::lazy_split_view<V, P> v;
    auto it = v.begin();
    static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), char&>);

    static_assert(ConstBeginDisabled<decltype(v)>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
