//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16

// Throwing bad_variant_access is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}} && !no-exceptions

// <cuda/std/variant>
// template <class Visitor, class... Variants>
// constexpr see below visit(Visitor&& vis, Variants&&... vars);

#include <cuda/std/cassert>
// #include <cuda/std/memory>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

struct almost_string {
    const char * ptr;

    __host__ __device__
    almost_string(const char * ptr) : ptr(ptr) {}

    __host__ __device__
    friend bool operator==(const almost_string & lhs, const almost_string & rhs) {
        return lhs.ptr == rhs.ptr;
    }
};

__host__ __device__
void test_exceptions() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  ReturnArity obj{};
  auto test = [&](auto &&... args) {
    try {
      cuda::std::visit(obj, args...);
    } catch (const cuda::std::bad_variant_access &) {
      return true;
    } catch (...) {
    }
    return false;
  };
  {
    using V = cuda::std::variant<int, MakeEmptyT>;
    V v;
    makeEmpty(v);
    assert(test(v));
  }
  {
    using V = cuda::std::variant<int, MakeEmptyT>;
    using V2 = cuda::std::variant<long, almost_string, void *>;
    V v;
    makeEmpty(v);
    V2 v2("hello");
    assert(test(v, v2));
  }
  {
    using V = cuda::std::variant<int, MakeEmptyT>;
    using V2 = cuda::std::variant<long, almost_string, void *>;
    V v;
    makeEmpty(v);
    V2 v2("hello");
    assert(test(v2, v));
  }
  {
    using V = cuda::std::variant<int, MakeEmptyT>;
    using V2 = cuda::std::variant<long, almost_string, void *, MakeEmptyT>;
    V v;
    makeEmpty(v);
    V2 v2;
    makeEmpty(v2);
    assert(test(v, v2));
  }
  {
    using V = cuda::std::variant<int, long, double, MakeEmptyT>;
    V v1(42l), v2(101), v3(202), v4(1.1);
    makeEmpty(v1);
    assert(test(v1, v2, v3, v4));
  }
  {
    using V = cuda::std::variant<int, long, double, long long, MakeEmptyT>;
    V v1(42l), v2(101), v3(202), v4(1.1);
    makeEmpty(v1);
    makeEmpty(v2);
    makeEmpty(v3);
    makeEmpty(v4);
    assert(test(v1, v2, v3, v4));
  }
#endif
}

int main(int, char**) {
  test_exceptions();

  return 0;
}