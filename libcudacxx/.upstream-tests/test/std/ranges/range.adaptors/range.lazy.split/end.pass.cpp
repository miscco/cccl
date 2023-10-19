//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr auto end() requires forward_range<View> && common_range<View>;
// constexpr auto end() const;

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include <cuda/std/utility>
#include "test_iterators.h"
#include "types.h"

struct ForwardViewCommonIfConst : cuda::std::ranges::view_base {
  cuda::std::string_view view_;
  constexpr explicit ForwardViewCommonIfConst() = default;
  constexpr ForwardViewCommonIfConst(const char* ptr) : view_(ptr) {}
  constexpr ForwardViewCommonIfConst(cuda::std::string_view v) : view_(v) {}
  constexpr ForwardViewCommonIfConst(ForwardViewCommonIfConst&&) = default;
  constexpr ForwardViewCommonIfConst& operator=(ForwardViewCommonIfConst&&) = default;
  constexpr ForwardViewCommonIfConst(const ForwardViewCommonIfConst&) = default;
  constexpr ForwardViewCommonIfConst& operator=(const ForwardViewCommonIfConst&) = default;
  constexpr forward_iterator<char*> begin() { return forward_iterator<char*>(nullptr); }
  constexpr cuda::std::default_sentinel_t end()  { return cuda::std::default_sentinel; }
  constexpr forward_iterator<const char*> begin() const { return forward_iterator<const char*>(view_.begin()); }
  constexpr forward_iterator<const char*> end() const { return forward_iterator<const char*>(view_.end()); }
};
bool operator==(forward_iterator<char*>, cuda::std::default_sentinel_t) { return false; }

struct ForwardViewNonCommonRange : cuda::std::ranges::view_base {
  cuda::std::string_view view_;
  constexpr explicit ForwardViewNonCommonRange() = default;
  constexpr ForwardViewNonCommonRange(const char* ptr) : view_(ptr) {}
  constexpr ForwardViewNonCommonRange(cuda::std::string_view v) : view_(v) {}
  constexpr ForwardViewNonCommonRange(ForwardViewNonCommonRange&&) = default;
  constexpr ForwardViewNonCommonRange& operator=(ForwardViewNonCommonRange&&) = default;
  constexpr ForwardViewNonCommonRange(const ForwardViewNonCommonRange&) = default;
  constexpr ForwardViewNonCommonRange& operator=(const ForwardViewNonCommonRange&) = default;
  constexpr forward_iterator<char*> begin() { return forward_iterator<char*>(nullptr); }
  constexpr cuda::std::default_sentinel_t end()  { return cuda::std::default_sentinel; }
  constexpr forward_iterator<const char*> begin() const { return forward_iterator<const char*>(view_.begin()); }
  constexpr cuda::std::default_sentinel_t end() const { return cuda::std::default_sentinel; }
};
bool operator==(forward_iterator<const char*>, cuda::std::default_sentinel_t) { return false; }

constexpr bool test() {
  // non-const: forward_range<V> && simple_view<V> && simple_view<P> -> outer-iterator<Const = true>
  // const: forward_range<V> && common_range<V> -> outer-iterator<Const = true>
  {
    using V = ForwardView;
    using P = V;

    static_assert(cuda::std::ranges::forward_range<V>);
    static_assert(cuda::std::ranges::common_range<const V>);
    LIBCPP_STATIC_ASSERT(cuda::std::ranges::__simple_view<V>);
    LIBCPP_STATIC_ASSERT(cuda::std::ranges::__simple_view<P>);

    {
      cuda::std::ranges::lazy_split_view<V, P> v;
      auto it = v.end();
      static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::forward_iterator_tag>);
      static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), const char&>);
    }

    {
      const cuda::std::ranges::lazy_split_view<V, P> cv;
      auto it = cv.end();
      static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::forward_iterator_tag>);
      static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), const char&>);
    }
  }

  // non-const: forward_range<V> && common_range<V> && simple_view<V> && !simple_view<P> -> outer-iterator<Const=false>
  // const: forward_range<V> && forward_range<const V> && common_range<const V> -> outer-iterator<Const = false>
  {
    using V = ForwardView;
    using P = ForwardDiffView;

    static_assert(cuda::std::ranges::forward_range<V>);
    static_assert(cuda::std::ranges::common_range<V>);
    LIBCPP_STATIC_ASSERT(cuda::std::ranges::__simple_view<V>);
    LIBCPP_STATIC_ASSERT(!cuda::std::ranges::__simple_view<P>);
    static_assert(cuda::std::ranges::forward_range<const V>);
    static_assert(cuda::std::ranges::common_range<const V>);

    {
      cuda::std::ranges::lazy_split_view<V, P> v;
      auto it = v.end();
      static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::forward_iterator_tag>);
      static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), const char&>);
    }

    {
      const cuda::std::ranges::lazy_split_view<V, P> cv;
      auto it = cv.end();
      static_assert(cuda::std::is_same_v<decltype(it)::iterator_concept, cuda::std::forward_iterator_tag>);
      static_assert(cuda::std::is_same_v<decltype(*(*it).begin()), const char&>);
    }
  }

  // non-const: forward_range<V> && !common_range<V> -> disabled
  // const: forward_range<V> && forward_range<const V> && common_range<const V> -> outer-iterator<Const = true>
  {
    using V = ForwardViewCommonIfConst;
    using P = V;

    static_assert(cuda::std::ranges::forward_range<V>);
    static_assert(!cuda::std::ranges::common_range<V>);
    static_assert(cuda::std::ranges::forward_range<const V>);
    static_assert(cuda::std::ranges::common_range<const V>);

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

  // non-const: forward_range<V> && !common_range<V> -> disabled
  // const: forward_range<V> && forward_range<const V> && !common_range<const V> -> outer-iterator<Const = false>
  {
    using V = ForwardViewNonCommonRange;
    using P = V;

    static_assert(cuda::std::ranges::forward_range<V>);
    static_assert(!cuda::std::ranges::common_range<V>);
    static_assert(cuda::std::ranges::forward_range<const V>);
    static_assert(!cuda::std::ranges::common_range<const V>);

    {
      cuda::std::ranges::lazy_split_view<V, P> v;
      auto it = v.end();
      static_assert(cuda::std::same_as<decltype(it), cuda::std::default_sentinel_t>);
    }

    {
      const cuda::std::ranges::lazy_split_view<V, P> cv;
      auto it = cv.end();
      static_assert(cuda::std::same_as<decltype(it), cuda::std::default_sentinel_t>);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
