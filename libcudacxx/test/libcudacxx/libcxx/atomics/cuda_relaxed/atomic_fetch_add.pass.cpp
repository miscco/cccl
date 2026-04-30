//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

// <cuda/std/atomic>

// template <class Integral>
//     Integral
//     atomic_fetch_add(volatile atomic<Integral>* obj, Integral op);
//
// template <class Integral>
//     Integral
//     atomic_fetch_add(atomic<Integral>* obj, Integral op);
//
// template <class T>
//     T*
//     atomic_fetch_add(volatile atomic<T*>* obj, ptrdiff_t op);
//
// template <class T>
//     T*
//     atomic_fetch_add(atomic<T*>* obj, ptrdiff_t op);

#include <cuda/__atomic/cuda_atomic_relaxed.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> class Selector, cuda::thread_scope _Scope>
struct TestFn
{
  TEST_DEVICE_FUNC void operator()() const
  {
    {
      Selector<T, constructor_initializer> sel;
      T& t = *sel.construct(1);
      assert(::cuda::__cccl_cuda_atomic_fetch_add_relaxed<_Scope>(&t, T(2)) == T(1));
      assert(t == T(3));
    }
  }
};

template <template <typename, typename> class Selector, cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test()
{
  TestFn<char, Selector, Scope>()();
  TestFn<signed char, Selector, Scope>()();
  TestFn<unsigned char, Selector, Scope>()();
  TestFn<short, Selector, Scope>()();
  TestFn<unsigned short, Selector, Scope>()();
  TestFn<int, Selector, Scope>()();
  TestFn<unsigned int, Selector, Scope>()();
  TestFn<long, Selector, Scope>()();
  TestFn<unsigned long, Selector, Scope>()();
  TestFn<long long, Selector, Scope>()();
  TestFn<unsigned long long, Selector, Scope>()();
  TestFn<wchar_t, Selector, Scope>();
  TestFn<char16_t, Selector, Scope>()();
  TestFn<char32_t, Selector, Scope>()();
  TestFn<int8_t, Selector, Scope>()();
  TestFn<uint8_t, Selector, Scope>()();
  TestFn<int16_t, Selector, Scope>()();
  TestFn<uint16_t, Selector, Scope>()();
  TestFn<int32_t, Selector, Scope>()();
  TestFn<uint32_t, Selector, Scope>()();
  TestFn<int64_t, Selector, Scope>()();
  TestFn<uint64_t, Selector, Scope>()();
  TestFn<float, Selector, Scope>()();
  TestFn<double, Selector, Scope>()();
}

template <template <typename, typename> class Selector>
TEST_DEVICE_FUNC void test()
{
  test<Selector, ::cuda::thread_scope_system>();
  test<Selector, ::cuda::thread_scope_device>();
  test<Selector, ::cuda::thread_scope_block>();
  test<Selector, ::cuda::thread_scope_thread>();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, ({
                 test<local_memory_selector>();
                 test<shared_memory_selector>();
                 test<global_memory_selector>();
               }))

  return 0;
}
