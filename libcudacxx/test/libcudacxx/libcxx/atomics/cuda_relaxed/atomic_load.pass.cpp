//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: asm statement is unsupported in tile code

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
//  ... assertion fails line 35

// <cuda/std/atomic>

// template <class T>
//     T
//     atomic_load(const volatile atomic<T>* obj);
//
// template <class T>
//     T
//     atomic_load(const atomic<T>* obj);

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
    Selector<T, constructor_initializer> sel;
    T& t = *sel.construct(1337);
    T t1;
    __syncthreads();
    ::cuda::__cccl_cuda_atomic_load_relaxed<_Scope>(&t, t1);
    assert(t1 == T(1337));
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
_CCCL_DEVICE void test()
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
