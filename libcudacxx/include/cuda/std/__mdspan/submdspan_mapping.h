//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_SUBMDSPAN_MAPPING_H
#define _LIBCUDACXX___MDSPAN_SUBMDSPAN_MAPPING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/mdspan.h>
#include <cuda/std/__mdspan/concepts.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__mdspan/layout_left.h>
#include <cuda/std/__mdspan/layout_right.h>
#include <cuda/std/__mdspan/layout_stride.h>
#include <cuda/std/__mdspan/submdspan_extents.h>
#include <cuda/std/__mdspan/submdspan_helper.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/detail/libcxx/include/__assert>

#if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// [mdspan.sub.map]

// [mdspan.submdspan.submdspan.mapping.result]
template <class _LayoutMapping>
struct submdspan_mapping_result
{
  static_assert(true, // __is_layout_mapping<_LayoutMapping>,
                "[mdspan.submdspan.submdspan.mapping.result] shall meet the layout mapping requirements");

  _CCCL_NO_UNIQUE_ADDRESS _LayoutMapping mapping{};
  size_t offset{};
};

// [mdspan.sub.map.common]
_LIBCUDACXX_TEMPLATE(size_t _SliceIndex, class _LayoutMapping, class... _Slices)
_LIBCUDACXX_REQUIRES(__is_strided_slice<__get_slice_type<_SliceIndex, _Slices...>>)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
__get_submdspan_strides(const _LayoutMapping& __mapping, _Slices... __slices) noexcept
{
  using _SliceType    = __get_slice_type<_SliceIndex, _Slices...>;
  _SliceType& __slice = _CUDA_VSTD::__get_slice_at<_SliceIndex>(__slices...);
  return __mapping.stride(_SliceIndex) * (__slice.stride < __slice.extent ? _CUDA_VSTD::__de_ice(__slice.stride) : 1);
}

_LIBCUDACXX_TEMPLATE(size_t _SliceIndex, class _LayoutMapping, class... _Slices)
_LIBCUDACXX_REQUIRES((!__is_strided_slice<__get_slice_type<_SliceIndex, _Slices...>>) )
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
__get_submdspan_strides(const _LayoutMapping& __mapping, _Slices...) noexcept
{
  return __mapping.stride(_SliceIndex);
}

template <class _Extents, class _SubExtent, class _LayoutMapping, class... _Slices, size_t... _SliceIndexes>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
__submdspan_strides(index_sequence<_SliceIndexes...>, const _LayoutMapping& __mapping, _Slices... __slices) noexcept
{
  using _IndexType           = typename _Extents::index_type;
  constexpr auto __map_rank_ = _CUDA_VSTD::__map_rank<_IndexType, _Slices...>();

  const array<_IndexType, _Extents::rank()> __arr = {
    _CUDA_VSTD::__get_submdspan_strides<_SliceIndexes, _Extents>(__mapping, __slices...)...};

  array<_IndexType, _SubExtent::rank()> __res = {};

  for (size_t __index = 0; index < _SubExtent::rank(); ++__index)
  {
    if (__map_rank_[__index] != dynamic_extent)
    {
      __res[__map_rank_[__index]] = __arr[__index];
    }
  }
  return __res;
}

template <class _Extents, class _SubExtent, class _LayoutMapping, class... _Slices>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
__submdspan_strides(const _LayoutMapping& __mapping, _Slices... __slices)
{
  return __submdspan_strides(_CUDA_VSTD::index_sequence_for<_Slices...>(), __mapping, __slices...);
}

// [mdspan.sub.map]
_LIBCUDACXX_TEMPLATE(class _Extents, class _LayoutMapping, class... _Slices)
_LIBCUDACXX_REQUIRES((_Extents::rank() == 0))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
__submdspan_mapping_impl(const _LayoutMapping& __mapping, _Slices...)
{
  return submdspan_mapping_result{__mapping, 0};
}

_LIBCUDACXX_TEMPLATE(class _Extents, class _LayoutMapping, class... _Slices)
_LIBCUDACXX_REQUIRES((_Extents::rank() != 0) _LIBCUDACXX_AND(_Extents::rank() == sizeof...(_Slices)))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
__submdspan_mapping_impl(const _LayoutMapping& __mapping, _Slices... __slices)
{
  const auto __sub_ext = _CUDA_VSTD::submdspan_extents(__mapping.extents(), __slices...);
  using _SubExtents    = remove_const_t<decltype(__sub_ext)>;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_STD_VER >= 2014

#endif // _LIBCUDACXX___MDSPAN_SUBMDSPAN_MAPPING_H
