//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++11

// <mdspan>

// constexpr mdspan& operator=(const mdspan& rhs) = default;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "../CustomTestLayouts.h"
#include "../MinimalElementType.h"
#include "CustomTestAccessors.h"
#include "test_macros.h"

// The defaulted assignment operator seems to be deprecated because:
//   error: definition of implicit copy assignment operator for 'checked_accessor<const double>' is deprecated
//   because it has a user-provided copy constructor [-Werror,-Wdeprecated-copy-with-user-provided-copy]
template <class MDS,
          class A,
          cuda::std::enable_if_t<!cuda::std::is_same<A, checked_accessor<const double>>::value, int> = 0>
__host__ __device__ constexpr void test_implicit_copy_assignment(MDS& m, MDS& m_org)
{
  m = m_org;
}
template <class MDS,
          class A,
          cuda::std::enable_if_t<cuda::std::is_same<A, checked_accessor<const double>>::value, int> = 0>
__host__ __device__ constexpr void test_implicit_copy_assignment(MDS&, MDS&)
{}

template <class H, class M, class A>
__host__ __device__ constexpr void test_mdspan_types(const H& handle, const M& map, const A& acc)
{
  using MDS = cuda::std::mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  MDS m_org(handle, map, acc);
  MDS m(handle, map, acc);

  test_implicit_copy_assignment<MDS, A>(m, m_org);
  // even though the following checks out:
  static_assert(cuda::std::copyable<checked_accessor<const double>>, "");
  static_assert(cuda::std::is_assignable<checked_accessor<const double>, checked_accessor<const double>>::value, "");

  static_assert(noexcept(m = m_org), "");
  assert(m.extents() == map.extents());
  test_equality_handle(m, handle);
  test_equality_mapping(m, map);
  test_equality_accessor(m, acc);

  static_assert(cuda::std::is_trivially_assignable<MDS, const MDS&>::value
                  == ((!cuda::std::is_class<H>::value || cuda::std::is_trivially_assignable<H, const H&>::value)
                      && cuda::std::is_trivially_assignable<M, const M&>::value
                      && cuda::std::is_trivially_assignable<A, const A&>::value),
                "");
}

template <class H, class L, class A>
__host__ __device__ constexpr void mixin_extents(const H& handle, const L& layout, const A& acc)
{
  constexpr size_t D = cuda::std::dynamic_extent;
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<int>()), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<char, D>(7)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<unsigned, 7>()), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<size_t, D, 4, D>(2, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<char, D, 7, D>(0, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc);
}

template <class H, class A>
__host__ __device__ constexpr void mixin_layout(const H& handle, const A& acc)
{
  // make sure we test a trivially assignable mapping
  static_assert(cuda::std::is_trivially_assignable<
                  typename cuda::std::layout_left::template mapping<cuda::std::extents<int>>,
                  const typename cuda::std::layout_left::template mapping<cuda::std::extents<int>>&>::value,
                "");
  mixin_extents(handle, cuda::std::layout_left(), acc);
  mixin_extents(handle, cuda::std::layout_right(), acc);
  // make sure we test a not trivially assignable mapping
  static_assert(!cuda::std::is_trivially_assignable<
                  typename layout_wrapping_integral<4>::template mapping<cuda::std::extents<int>>,
                  const typename layout_wrapping_integral<4>::template mapping<cuda::std::extents<int>>&>::value,
                "");
  mixin_extents(handle, layout_wrapping_integral<4>(), acc);
}

template <class T, cuda::std::enable_if_t<cuda::std::is_default_constructible<T>::value, int> = 0>
__host__ __device__ constexpr void mixin_accessor()
{
  cuda::std::array<T, 1024> elements{42};
  // make sure we test trivially constructible accessor and data_handle
  static_assert(cuda::std::is_trivially_copyable<cuda::std::default_accessor<T>>::value, "");
  static_assert(cuda::std::is_trivially_copyable<typename cuda::std::default_accessor<T>::data_handle_type>::value, "");
  mixin_layout(elements.data(), cuda::std::default_accessor<T>());

  // Using weird accessor/data_handle
  // Make sure they actually got the properties we want to test
  // checked_accessor is noexcept copy constructible except for const double
  checked_accessor<T> acc(1024);
  static_assert(noexcept(checked_accessor<T>(acc)) != cuda::std::is_same<T, const double>::value, "");
  mixin_layout(typename checked_accessor<T>::data_handle_type(elements.data()), acc);
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_default_constructible<T>::value, int> = 0>
__host__ __device__ TEST_CONSTEXPR_CXX20 void mixin_accessor()
{
  ElementPool<T, 1024> elements;
  // make sure we test trivially constructible accessor and data_handle
  static_assert(cuda::std::is_trivially_copyable<cuda::std::default_accessor<T>>::value, "");
  static_assert(cuda::std::is_trivially_copyable<typename cuda::std::default_accessor<T>::data_handle_type>::value, "");
  mixin_layout(elements.get_ptr(), cuda::std::default_accessor<T>());

  // Using weird accessor/data_handle
  // Make sure they actually got the properties we want to test
  // checked_accessor is noexcept copy constructible except for const double
  checked_accessor<T> acc(1024);
  static_assert(noexcept(checked_accessor<T>(acc)) != cuda::std::is_same<T, const double>::value, "");
  mixin_layout(typename checked_accessor<T>::data_handle_type(elements.get_ptr()), acc);
}

__host__ __device__ constexpr bool test()
{
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();
  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_evil()
{
  mixin_accessor<MinimalElementType>();
  mixin_accessor<const MinimalElementType>();
  return true;
}

int main(int, char**)
{
  test();
  test_evil();

#if TEST_STD_VER >= 2020
  static_assert(test(), "");
  static_assert(test_evil(), "");
#endif // TEST_STD_VER >= 2020
  return 0;
}