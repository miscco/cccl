//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// cuda::std::views::lazy_split

#include <cuda/std/ranges>

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/string_view>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "types.h"

template <class View, class T>
concept CanBePiped = requires (View&& view, T&& t) {
  { cuda::std::forward<View>(view) | cuda::std::forward<T>(t) };
};

struct SomeView : cuda::std::ranges::view_base {
  const cuda::std::string_view* v_;
  constexpr SomeView(const cuda::std::string_view& v) : v_(&v) {}
  constexpr auto begin() const { return v_->begin(); }
  constexpr auto end() const { return v_->end(); }
};

struct NotAView { };

static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::lazy_split)>);
static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::lazy_split), SomeView, NotAView>);
static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::lazy_split), NotAView, SomeView>);
static_assert( cuda::std::is_invocable_v<decltype(cuda::std::views::lazy_split), SomeView, SomeView>);

static_assert( CanBePiped<SomeView&,    decltype(cuda::std::views::lazy_split)>);
static_assert( CanBePiped<char(&)[10],  decltype(cuda::std::views::lazy_split)>);
static_assert(!CanBePiped<char(&&)[10], decltype(cuda::std::views::lazy_split)>);
static_assert(!CanBePiped<NotAView,     decltype(cuda::std::views::lazy_split)>);

static_assert(cuda::std::same_as<decltype(cuda::std::views::lazy_split), decltype(cuda::std::ranges::views::lazy_split)>);

constexpr bool test() {
  cuda::std::string_view input = "abc";
  cuda::std::string_view sep = "a";

  // Test that `cuda::std::views::lazy_split` is a range adaptor.

  // Test `views::lazy_split(input, sep)`.
  {
    SomeView view(input);

    using Result = cuda::std::ranges::lazy_split_view<SomeView, cuda::std::string_view>;
    cuda::std::same_as<Result> decltype(auto) result = cuda::std::views::lazy_split(view, sep);
    assert(result.base().begin() == input.begin());
    assert(result.base().end() == input.end());
  }

  // Test `views::lazy_split(sep)(input)`.
  {
    SomeView view(input);

    using Result = cuda::std::ranges::lazy_split_view<SomeView, cuda::std::string_view>;
    cuda::std::same_as<Result> decltype(auto) result = cuda::std::views::lazy_split(sep)(view);
    assert(result.base().begin() == input.begin());
    assert(result.base().end() == input.end());
  }

  // Test `view | views::lazy_split`.
  {
    SomeView view(input);

    using Result = cuda::std::ranges::lazy_split_view<SomeView, cuda::std::string_view>;
    cuda::std::same_as<Result> decltype(auto) result = view | cuda::std::views::lazy_split(sep);
    assert(result.base().begin() == input.begin());
    assert(result.base().end() == input.end());
  }

  // Test `adaptor | views::lazy_split`.
  {
    SomeView view(input);
    auto f = [](char c) { return c; };
    auto partial = cuda::std::views::transform(f) | cuda::std::views::lazy_split(sep);

    using Result = cuda::std::ranges::lazy_split_view<cuda::std::ranges::transform_view<SomeView, decltype(f)>, cuda::std::string_view>;
    cuda::std::same_as<Result> decltype(auto) result = partial(view);
    assert(result.base().base().begin() == input.begin());
    assert(result.base().base().end() == input.end());
  }

  // Test `views::lazy_split | adaptor`.
  {
    SomeView view(input);
    auto f = [](auto v) { return v; };
    auto partial = cuda::std::views::lazy_split(sep) | cuda::std::views::transform(f);

    using Result = cuda::std::ranges::transform_view<cuda::std::ranges::lazy_split_view<SomeView, cuda::std::string_view>, decltype(f)>;
    cuda::std::same_as<Result> decltype(auto) result = partial(view);
    assert(result.base().base().begin() == input.begin());
    assert(result.base().base().end() == input.end());
  }

  // Test that one can call `cuda::std::views::lazy_split` with arbitrary stuff, as long as we
  // don't try to actually complete the call by passing it a range.
  //
  // That makes no sense and we can't do anything with the result, but it's valid.
  {
    struct X { };
    [[maybe_unused]] auto partial = cuda::std::views::lazy_split(X{});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
