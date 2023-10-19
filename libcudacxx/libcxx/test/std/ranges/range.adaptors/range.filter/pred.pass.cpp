//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr Pred const& pred() const;

#include <ranges>

#include <cassert>
#include <concepts>

struct Range : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct Pred {
  bool operator()(int) const;
  int value;
};

constexpr bool test() {
  {
    Pred pred{42};
    std::ranges::filter_view<Range, Pred> const view(Range{}, pred);
    static_assert(std::same_as<decltype(view.pred()), const Pred&>);
    decltype(auto) result = view.pred();
    static_assert(std::same_as<decltype(result), const Pred&>);
    assert(result.value == 42);

    // Make sure we're really holding a reference to something inside the view
    decltype(auto) result2 = view.pred();
    static_assert(std::same_as<decltype(result2), const Pred&>);
    assert(&result == &result2);
  }

  // Same, but calling on a non-const view
  {
    Pred pred{42};
    std::ranges::filter_view<Range, Pred> view(Range{}, pred);
    decltype(auto) result = view.pred();
    static_assert(std::same_as<decltype(result), const Pred&>);
    assert(result.value == 42);

    // Make sure we're really holding a reference to something inside the view
    decltype(auto) result2 = view.pred();
    static_assert(std::same_as<decltype(result2), const Pred&>);
    assert(&result == &result2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
