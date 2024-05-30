//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_SUBMDSPAN_HELPER_H
#define _LIBCUDACXX___MDSPAN_SUBMDSPAN_HELPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__fwd/mdspan.h>
#include <cuda/std/__mdspan/concepts.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/detail/libcxx/include/__assert>
#include <cuda/std/tuple>

#if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// [mdspan.sub.overview]-2.5
struct __map_rank_indexes
{
  template <class _IndexType, size_t... _FilteredIndexes>
  auto __get(index_sequence<>, index_sequence<_FilteredIndexes...>) -> index_sequence<_FilteredIndexes...>;

  template <class _IndexType,
            class _Slice,
            class... _Slices,
            size_t _Index,
            size_t... _SliceIndices,
            size_t... _FilteredIndexes>
  auto __get(index_sequence<_Index, _SliceIndices...>, index_sequence<_FilteredIndexes...>)
    -> _If<convertible_to<_Slice, _IndexType>,
           decltype(__get<_IndexType, _Slices...>(index_sequence<_SliceIndices...>{},
                                                  index_sequence<_FilteredIndexes...>{})),
           decltype(__get<_IndexType, _Slices...>(index_sequence<_SliceIndices...>{},
                                                  index_sequence<_FilteredIndexes..., _Index>{}))>;

  template <class _IndexType, class... _Slices>
  using type = decltype(__get<_IndexType, _Slices...>(index_sequence_for<_Slices...>{}, index_sequence<>{}));
};

template <class _IndexType, class... _Slices>
using __map_rank_indexes_t = typename __map_rank_indexes::template type<_IndexType, _Slices...>;

template <class _IndexType, class... _Slices>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr array<size_t, sizeof...(_Slices)> __map_rank() noexcept
{
  array<size_t, sizeof...(_Slices)> __arr = {convertible_to<_Slices, _IndexType>...};

  size_t __count = 0;
  for (size_t& elem : __arr)
  {
    elem = elem ? dynamic_extent : __count++;
  }
  return __arr;
}

// [mdspan.submdspan.strided.slice]
template <class _OffsetType, class _ExtentType, class _StrideType>
struct strided_slice
{
  using offset_type = _OffsetType;
  using extent_type = _ExtentType;
  using stride_type = _StrideType;

  static_assert(__index_like<offset_type>,
                "[mdspan.submdspan.strided.slice] cuda::std::strided_slice::offset_type must be signed or unsigned or "
                "integral-constant-like");
  static_assert(__index_like<extent_type>,
                "[mdspan.submdspan.strided.slice] cuda::std::strided_slice::extent_type must be signed or unsigned or "
                "integral-constant-like");
  static_assert(__index_like<stride_type>,
                "[mdspan.submdspan.strided.slice] cuda::std::strided_slice::stride_type must be signed or unsigned or "
                "integral-constant-like");

  _CCCL_NO_UNIQUE_ADDRESS offset_type offset{};
  _CCCL_NO_UNIQUE_ADDRESS extent_type extent{};
  _CCCL_NO_UNIQUE_ADDRESS stride_type stride{};
};

template <typename>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_strided_slice = false;

template <class _OffsetType, class _ExtentType, class _StrideType>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_strided_slice<strided_slice<_OffsetType, _ExtentType, _StrideType>> = true;

struct full_extent_t
{
  explicit full_extent_t() = default;
};
_CCCL_GLOBAL_CONSTANT full_extent_t full_extent{};

// [mdspan.submdspan.submdspan.mapping.result]
template <class _LayoutMapping>
struct submdspan_mapping_result
{
  static_assert(true, // __is_layout_mapping<_LayoutMapping>,
                "[mdspan.submdspan.submdspan.mapping.result] shall meet the layout mapping requirements");

  _CCCL_NO_UNIQUE_ADDRESS _LayoutMapping mapping{};
  size_t offset{};
};

// [mdspan.submdspan.helpers]
_LIBCUDACXX_TEMPLATE(class _Tp)
_LIBCUDACXX_REQUIRES((!__integral_constant_like<_Tp>) )
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __de_ice(_Tp __val)
{
  return __val;
}

_LIBCUDACXX_TEMPLATE(class _Tp)
_LIBCUDACXX_REQUIRES(__integral_constant_like<_Tp>)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __de_ice(_Tp)
{
  return _Tp::value;
}

_LIBCUDACXX_TEMPLATE(class _IndexType, class _From)
_LIBCUDACXX_REQUIRES(_CCCL_TRAIT(is_integral, _From) _LIBCUDACXX_AND(!_CCCL_TRAIT(is_same, _From, bool)))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto __index_cast(_From&& __from) noexcept
{
  return __from;
}
_LIBCUDACXX_TEMPLATE(class _IndexType, class _From)
_LIBCUDACXX_REQUIRES((!_CCCL_TRAIT(is_integral, _From)) || _CCCL_TRAIT(is_same, _From, bool))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto __index_cast(_From&& __from) noexcept
{
  return static_cast<_IndexType>(__from);
}

template <size_t _Index, class... _Slices>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) __get_slice_at(_Slices&&... __slices) noexcept
{
  return _CUDA_VSTD::get<_Index>(_CUDA_VSTD::forward_as_tuple(_CUDA_VSTD::forward<_Slices>(__slices)...));
}

struct __first_extent_from_slice_impl
{
  _LIBCUDACXX_TEMPLATE(class _IndexType, class _SliceType)
  _LIBCUDACXX_REQUIRES(convertible_to<_SliceType, _IndexType>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _IndexType __get(_SliceType& __slice) noexcept
  {
    return _CUDA_VSTD::__index_cast<_IndexType>(__slice);
  }
  _LIBCUDACXX_TEMPLATE(class _IndexType, class _SliceType)
  _LIBCUDACXX_REQUIRES(
    (!convertible_to<_SliceType, _IndexType>) _LIBCUDACXX_AND __index_pair_like<_SliceType, _IndexType>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _IndexType __get(_SliceType& __slice) noexcept
  {
    return _CUDA_VSTD::__index_cast<_IndexType>(_CUDA_VSTD::get<0>(__slice));
  }
  _LIBCUDACXX_TEMPLATE(class _IndexType, class _SliceType)
  _LIBCUDACXX_REQUIRES((!convertible_to<_SliceType, _IndexType>) _LIBCUDACXX_AND(
    !__index_pair_like<_SliceType, _IndexType>) _LIBCUDACXX_AND __is_strided_slice<_SliceType>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _IndexType __get(_SliceType& __slice) noexcept
  {
    return _CUDA_VSTD::__index_cast<_IndexType>(_CUDA_VSTD::__de_ice(__slice.offset));
  }
  _LIBCUDACXX_TEMPLATE(class _IndexType, class _SliceType)
  _LIBCUDACXX_REQUIRES((!convertible_to<_SliceType, _IndexType>) _LIBCUDACXX_AND(
    !__index_pair_like<_SliceType, _IndexType>) _LIBCUDACXX_AND(!__is_strided_slice<_SliceType>))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _IndexType __get(_SliceType&) noexcept
  {
    return 0;
  }
};

template <size_t _Index, class... _Slices>
using __get_slice_type = __tuple_element_t<_Index, __tuple_types<_Slices...>>;

template <class _IndexType, size_t _Index, class... _Slices>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _IndexType __first_extent_from_slice(_Slices... __slices) noexcept
{
  static_assert(_CCCL_TRAIT(is_signed, _IndexType) || _CCCL_TRAIT(is_unsigned, _IndexType),
                "[mdspan.sub.helpers] mandates IndexType to be a signed or unsigned integral");
  using _SliceType    = __get_slice_type<_Index, _Slices...>;
  _SliceType& __slice = _CUDA_VSTD::__get_slice_at<_Index>(__slices...);
  return __first_extent_from_slice_impl::template __get<_IndexType>(__slice);
}

template <size_t _Index>
struct __last_extent_from_slice_impl
{
  _LIBCUDACXX_TEMPLATE(class _IndexType, class _SliceType, class _Extents)
  _LIBCUDACXX_REQUIRES(convertible_to<_SliceType, _IndexType>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _IndexType
  __get(_SliceType& __slice, const _Extents&) noexcept
  {
    return _CUDA_VSTD::__index_cast<_IndexType>(_CUDA_VSTD::__de_ice(__slice) + 1);
  }
  _LIBCUDACXX_TEMPLATE(class _IndexType, class _SliceType, class _Extents)
  _LIBCUDACXX_REQUIRES(
    (!convertible_to<_SliceType, _IndexType>) _LIBCUDACXX_AND __index_pair_like<_SliceType, _IndexType>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _IndexType
  __get(_SliceType& __slice, const _Extents&) noexcept
  {
    return _CUDA_VSTD::__index_cast<_IndexType>(_CUDA_VSTD::get<1>(__slice));
  }
  _LIBCUDACXX_TEMPLATE(class _IndexType, class _SliceType, class _Extents)
  _LIBCUDACXX_REQUIRES((!convertible_to<_SliceType, _IndexType>) _LIBCUDACXX_AND(
    !__index_pair_like<_SliceType, _IndexType>) _LIBCUDACXX_AND __is_strided_slice<_SliceType>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _IndexType
  __get(_SliceType& __slice, const _Extents&) noexcept
  {
    return _CUDA_VSTD::__index_cast<_IndexType>(
      _CUDA_VSTD::__de_ice(__slice.offset) * _CUDA_VSTD::__de_ice(__slice.extent));
  }
  _LIBCUDACXX_TEMPLATE(class _IndexType, class _SliceType, class _Extents)
  _LIBCUDACXX_REQUIRES((!convertible_to<_SliceType, _IndexType>) _LIBCUDACXX_AND(
    !__index_pair_like<_SliceType, _IndexType>) _LIBCUDACXX_AND(!__is_strided_slice<_SliceType>))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _IndexType
  __get(_SliceType&, const _Extents& __src) noexcept
  {
    return _CUDA_VSTD::__index_cast<_IndexType>(__src.extent(_Index));
  }
};

template <size_t _Index, class _Extents, class... _Slices>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr typename _Extents::index_type
__last_extent_from_slice(const _Extents& __src, _Slices... __slices) noexcept
{
  static_assert(_CCCL_TRAIT(__mdspan_detail::__is_extents, _Extents),
                "[mdspan.sub.helpers] mandates Extents to be a specialization of extents");
  using _IndexType    = typename _Extents::index_type;
  using _SliceType    = __get_slice_type<_Index, _Slices...>;
  _SliceType& __slice = _CUDA_VSTD::__get_slice_at<_Index>(__slices...);
  return __last_extent_from_slice_impl<_Index>::template __get<_IndexType>(__slice, __src);
}

template <class _IndexType, class... _Slices, size_t... _SliceIndexes>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr array<_IndexType, sizeof...(_Slices)> __src_indices(
  index_sequence<_SliceIndexes...>, const array<_IndexType, sizeof...(_Slices)>& __indices, _Slices... __slices) noexcept
{
  constexpr array<size_t, sizeof...(_Slices)> __ranks = _CUDA_VSTD::__map_rank<_IndexType, _Slices...>();
  array<_IndexType, sizeof...(_Slices)> __arr         = {
    _CUDA_VSTD::__first_extent_from_slice<_IndexType, _SliceIndexes>(__slices)...};

  for (size_t __index = 0; __index < sizeof...(_Slices); ++__index)
  {
    if (__ranks[__index] != dynamic_extent)
    {
      __arr += __indices[__ranks[__index]];
    }
  }
  return __arr;
}

template <class _IndexType, class... _Slices>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr array<_IndexType, sizeof...(_Slices)>
__src_indices(const array<_IndexType, sizeof...(_Slices)>& __indices, _Slices... __slices) noexcept
{
  return _CUDA_VSTD::__src_indices(index_sequence_for<_Slices...>(), __indices, __slices...);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_STD_VER >= 2014

#endif // _LIBCUDACXX___MDSPAN_SUBMDSPAN_HELPER_H
