//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__memory/__aligned_array.h>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif // _CCCL_HAS_NVFP16()

struct three_chars_t
{
  char data[3];
};

struct int_and_char_t
{
  int i;
  char c;
};

struct alignas(32) aligned_32_t
{
  long long data[4];
};

struct non_default_constructible_t
{
  int value;

  TEST_FUNC explicit non_default_constructible_t(int v)
      : value(v)
  {}
};

// The layout of T[Count], aligned to Alignment
template <class T, cuda::std::size_t Count, cuda::std::size_t Alignment>
TEST_FUNC constexpr void test_layout()
{
  using array_t = cuda::__aligned_array<T, Count, Alignment>;
  static_assert(sizeof(array_t) == sizeof(T) * Count);
  static_assert(alignof(array_t) == Alignment);
  static_assert(cuda::std::is_trivially_copyable_v<array_t>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const array_t&>()[0]), T>);
}

TEST_FUNC void test()
{
  test_layout<char, 1, 1>();
  test_layout<char, 3, 1>();
  test_layout<char, 16, 16>();
  test_layout<char, 32, 32>();
  test_layout<short, 8, 16>();
  test_layout<int, 1, 4>();
  test_layout<int, 3, 4>();
  test_layout<int, 4, 16>();
  test_layout<int, 8, 32>();
  test_layout<float, 4, 8>();
  test_layout<double, 2, 16>();
  test_layout<double, 4, 32>();
  test_layout<three_chars_t, 4, 4>();
  test_layout<int_and_char_t, 2, 16>();
  test_layout<float3, 4, 16>();
  test_layout<aligned_32_t, 1, 32>();
  test_layout<aligned_32_t, 2, 64>();
  test_layout<non_default_constructible_t, 4, 16>();
#if _CCCL_HAS_NVFP16()
  test_layout<__half, 8, 16>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_INT128()
  test_layout<__int128_t, 1, 16>();
  test_layout<__int128_t, 2, 32>();
  test_layout<__uint128_t, 1, 16>();
  test_layout<__uint128_t, 4, 32>();
#endif // _CCCL_HAS_INT128()
}

int main(int, char**)
{
  test();
  return 0;
}
