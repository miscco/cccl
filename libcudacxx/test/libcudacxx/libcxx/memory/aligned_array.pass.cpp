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
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstring>

#include "test_macros.h"

struct three_chars_t
{
  char data[3];
};

struct alignas(32) aligned_32_t
{
  long long data[4];
};

struct non_default_constructible_t
{
  int value;

  TEST_FUNC constexpr explicit non_default_constructible_t(int v)
      : value(v)
  {}
};

// Every item of the array is the corresponding item of the bytes it was copied from
template <class T, cuda::std::size_t Count, cuda::std::size_t Alignment>
TEST_FUNC void test_items()
{
  using array_t = cuda::__aligned_array<T, Count, Alignment>;

  // Unique bytes as long as the array has at most 256 bytes
  unsigned char bytes[sizeof(array_t)];
  for (cuda::std::size_t i = 0; i < sizeof(array_t); ++i)
  {
    bytes[i] = static_cast<unsigned char>(i);
  }

  array_t array;
  cuda::std::memcpy(&array, bytes, sizeof(array_t));
  for (cuda::std::size_t i = 0; i < Count; ++i)
  {
    T item = array[i];
    assert(cuda::std::memcmp(&item, bytes + i * sizeof(T), sizeof(T)) == 0);
  }
}

TEST_FUNC constexpr bool test_index_types()
{
  cuda::__aligned_array<int, 4, 16> array{{1, 2, 3, 4}};
  assert(array[0] == 1);
  assert(array[short{1}] == 2);
  assert(array[cuda::std::size_t{2}] == 3);
  assert(array[static_cast<unsigned char>(3)] == 4);
  assert(array[3ll] == 4);
  return true;
}

TEST_FUNC constexpr bool test_non_default_constructible()
{
  cuda::__aligned_array<non_default_constructible_t, 2, 8> array{
    {non_default_constructible_t{1}, non_default_constructible_t{2}}};
  assert(array[0].value == 1);
  assert(array[1].value == 2);
  return true;
}

TEST_FUNC void test()
{
  test_items<char, 1, 1>();
  test_items<char, 32, 32>();
  test_items<short, 8, 16>();
  test_items<int, 3, 4>();
  test_items<int, 8, 32>();
  test_items<float, 4, 16>();
  test_items<double, 4, 32>();
  test_items<three_chars_t, 4, 4>();
  test_items<float3, 4, 16>();
  test_items<aligned_32_t, 2, 64>();
#if _CCCL_HAS_INT128()
  test_items<__int128_t, 1, 16>();
  test_items<__int128_t, 2, 32>();
  test_items<__uint128_t, 4, 32>();
#endif // _CCCL_HAS_INT128()

  test_index_types();
  test_non_default_constructible();
}

int main(int, char**)
{
  test();
  static_assert(test_index_types());
  static_assert(test_non_default_constructible());
  return 0;
}
