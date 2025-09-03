//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_LAYOUT_LEFT_PADDED_H
#define _LIBCUDACXX___MDSPAN_LAYOUT_LEFT_PADDED_H

#include <cuda/std/detail/__config>

#include "cuda/std/__fwd/span.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/mdspan.h>
#include <cuda/std/__mdspan/concepts.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__mdspan/submdspan_helper.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// [mdspan.layout.leftpad.expo] #1
template <class _Extents, size_t _PaddingValue>
[[nodiscard]] _CCCL_API static constexpr size_t __static_padding_stride() noexcept
{
  if constexpr (_Extents::rank() <= 1)
  {
    return 0;
  }
  else if constexpr (_Extents::static_extent(0) == dynamic_extent || _PaddingValue == dynamic_extent)
  {
    return dynamic_extent;
  }
  else
  {
    return __least_multiple_at_least(_PaddingValue, _Extents::static_extent(0));
  }
}

// [mdspan.layout.leftpad.expo] #2
template <size_t _FirstStride>
struct __cccl_stride_one
{
  _CCCL_HIDE_FROM_ABI constexpr __cccl_stride_one() noexcept = default;
  _CCCL_API constexpr __cccl_stride_one(size_t) noexcept {}

  [[nodiscard]] _CCCL_API static constexpr size_t __stride_one() noexcept
  {
    return _FirstStride;
  }
};

template <>
struct __cccl_stride_one<dynamic_extent>
{
  size_t __first_stride_ = dynamic_extent;

  _CCCL_HIDE_FROM_ABI constexpr __cccl_stride_one() noexcept = default;
  _CCCL_API constexpr __cccl_stride_one(const size_t __first_stride) noexcept
      : __first_stride_(__first_stride)
  {}

  [[nodiscard]] _CCCL_API constexpr size_t __stride_one() const noexcept
  {
    return __first_stride_;
  }
};

template <size_t _PaddingValue>
template <class _Extents>
class layout_left_padded<_PaddingValue>::mapping
    : private __mdspan_ebco<__cccl_stride_one<::cuda::std::__static_padding_stride<_Extents, _PaddingValue>(), _Extents>>
{
public:
  static constexpr size_t padding_value = _PaddingValue;

  using extents_type = _Extents;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = layout_left_padded<_PaddingValue>;
  using __base =
    __mdspan_ebco<__cccl_stride_one<::cuda::std::__static_padding_stride<_Extents, _PaddingValue>(), _Extents>>;

  static_assert(__mdspan_detail::__is_extents<_Extents>::value,
                "layout_left_padded::mapping template argument must be a specialization of extents.");

  // [mdspan.layout.leftpad.overview] #5.2
  static_assert(::cuda::std::in_range<index_type>(_PaddingValue),
                "layout_left_padded: padding_value must be representable as a value of type index_type.");

private:
  // [mdspan.layout.leftpad.expo]
  static constexpr size_t __first_static_extent = extents_type::static_extent(0);

  [[nodiscard]] _CCCL_API static constexpr bool __extents_is_representable() noexcept
  {
    // [mdspan.layout.leftpad.overview] #5.1
    if (extents_type::rank_dynamic() == 0)
    {
      rank_type __total_size = 1;
      for (auto __i = 0; __i != extents_type::rank(); ++__i)
      {
        __total_size *= extents_type::static_extent(__i);
      }
      return ::cuda::std::in_range<index_type>(__total_size);
    }
    // [mdspan.layout.leftpad.overview] #5.3
    else if constexpr (extents_type::rank() >= 1 && padding_value != dynamic_extent
                       && extents_type::static_extent(0) != dynamic_extent)
    {
      constexpr auto __mul = __least_multiple_at_least(padding_value, extents_type::static_extent(0));
      return ::cuda::std::in_range<size_t>(__mul) && ::cuda::std::in_range<index_type>(__mul);
    }
    // [mdspan.layout.leftpad.overview] #5.3
    else if constexpr (extents_type::rank() >= 1 && padding_value != dynamic_extent)
    {
      for (rank_type __i = 0; __i != extents_type::rank(); ++__i)
      {
        size_t __ext = extents_type::static_extent(__i);
        if (__ext ==)
        {
          return false;
        }
      }
    }
    return true;
  }
  static_assert(__extents_is_representable(),
                "layout_left_padded: The size of the multidimensional index space Extents() must be representable as a "
                "value of type index_type.");

  [[nodiscard]] _CCCL_API static constexpr bool
  __mul_overflow_or_negative(index_type __lhs, index_type __rhs, index_type* __res) noexcept
  {
    if constexpr (is_signed_v<index_type>)
    {
      if (__lhs < 0 || __rhs < 0)
      {
        return false;
      }
    }
    *__res = __lhs * __rhs;
    return __lhs && ((*__res / __lhs) != __rhs);
  }

  [[nodiscard]] _CCCL_API static constexpr bool __required_span_size_is_representable(const extents_type& __ext)
  {
    index_type __prod{1};
    for (rank_type __r = 0; __r != extents_type::rank(); __r++)
    {
      if (__mul_overflow(__prod, __ext.extent(__r), &__prod))
      {
        return false;
      }
    }
    return true;
  }

  static_assert((extents_type::rank_dynamic() > 0) || __required_span_size_is_representable(extents_type()),
                "layout_left_padded::mapping product of static extents must be representable as index_type.");

  [[nodiscard]] _CCCL_API static constexpr index_type __init_stride(const extents_type& __ext, const index_type __pad)
  {
    if constexpr (extents_type::rank() == 0)
    {
      return 0;
    }
    else if constexpr (_PaddingValue == dynamic_extent)
    {
      return __ext.extents(0);
    }
    else
    {
      return static_cast<index_type>(__least_multiple_at_least(__pad, __ext.extent(0)));
    }
  }

public:
  // [mdspan.layout.left.cons], constructors
  _CCCL_HIDE_FROM_ABI constexpr mapping() noexcept                          = default;
  _CCCL_HIDE_FROM_ABI constexpr mapping(const mapping&) noexcept            = default;
  _CCCL_HIDE_FROM_ABI constexpr mapping& operator=(const mapping&) noexcept = default;

  _CCCL_API constexpr mapping(const extents_type& __ext) noexcept
      : __base(__init_stride(__ext, _PaddingValue), __ext)
  {
    _CCCL_ASSERT(__required_span_size_is_representable(__ext),
                 "layout_left_padded::mapping: product of extents must be representable as index_type.");
  }

  template <class _OtherIndexType>
  _CCCL_API constexpr mapping(const extents_type& __ext, _OtherIndexType __pad) noexcept
      : __base(__init_stride(__ext, __pad), __ext)
  {
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__pad),
                 "layout_left_padded::mapping: pad must be representable as index_type.");
    _CCCL_ASSERT(::cuda::std::__index_cast<index_type>(__pad) > 0,
                 "layout_left_padded::mapping: extents_type::index-cast(pad) is greater than zero.");

    _CCCL_ASSERT(__required_span_size_is_representable(__ext),
                 "layout_left_padded::mapping: product of extents must be representable as index_type.");
  }

  _CCCL_TEMPLATE(class _OtherExtents)
  _CCCL_REQUIRES(is_convertible_v<_OtherExtents, extents_type>)
  _CCCL_API constexpr mapping(const layout_left::mapping<_OtherExtents>& __other) noexcept
      : __base(__init_stride(__other.extents(), _PaddingValue), __other.extents())
  {
    if constexpr (extents_type::rank() > 1 && _PaddingValue != dynamic_extent)
    {
      _CCCL_ASSERT(
        __other.stride(1)
          == static_cast<index_type>(
            __least_multiple_at_least(_PaddingValue, ::cuda::std::__index_cast<index_type>(__other.extents.extent(0)))),
        "layout_left_padded::mapping: other.stride(1) must equal "
        "least-multiple-at-least(padding_value,extents_type::index-cast(other.extents().extent(0)))");
    }
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
                 "layout_left_padded::mapping: other.required_span_size() must be representable as index_type.");
  }

  _CCCL_TEMPLATE(class _OtherExtents)
  _CCCL_REQUIRES(
    is_constructible_v<extents_type, _OtherExtents> _CCCL_AND(!is_convertible_v<_OtherExtents, extents_type>))
  _CCCL_API explicit constexpr mapping(const layout_left::mapping<_OtherExtents>& __other) noexcept
      : __base(__init_stride(__other.extents(), _PaddingValue), __other.extents())
  {
    if constexpr (extents_type::rank() > 1 && _PaddingValue != dynamic_extent)
    {
      _CCCL_ASSERT(
        __other.stride(1)
          == static_cast<index_type>(
            __least_multiple_at_least(_PaddingValue, ::cuda::std::__index_cast<index_type>(__other.extents.extent(0)))),
        "layout_left_padded::mapping: other.stride(1) must equal "
        "least-multiple-at-least(padding_value,extents_type::index-cast(other.extents().extent(0)))");
    }
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
                 "layout_left_padded::mapping: other.required_span_size() must be representable as index_type.");
  }

  _CCCL_TEMPLATE(class _OtherExtents, rank_type _Rank = extents_type::rank())
  _CCCL_REQUIRES((_Rank == 0))
  _CCCL_API constexpr mapping(const layout_stride::mapping<_OtherExtents>& __other) noexcept
      : __base(0, __other.extents())
  {
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
                 "layout_left_padded::mapping: other.required_span_size() must be representable as index_type.");
  }

  _CCCL_TEMPLATE(class _OtherExtents, rank_type _Rank = extents_type::rank())
  _CCCL_REQUIRES((_Rank > 0))
  _CCCL_API explicit constexpr mapping(const layout_stride::mapping<_OtherExtents>& __other) noexcept
      : __base(0, __other.extents())
  {
    _CCCL_ASSERT(__mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
                 "layout_left_padded::mapping: other.required_span_size() must be representable as index_type.");
  }

  template <class _OtherMappping>
  [[nodiscard]] _CCCL_API constexpr bool __check_strides(const _OtherMappping& __other) const noexcept
  {
    // avoid warning when comparing signed and unsigner integers and pick the wider of two types
    using _CommonType = common_type_t<index_type, typename _OtherMappping::index_type>;
    for (rank_type __r = 0; __r != extents_type::rank(); __r++)
    {
      if (static_cast<_CommonType>(stride(__r)) != static_cast<_CommonType>(__other.stride(__r)))
      {
        return false;
      }
    }
    return true;
  }

  _CCCL_TEMPLATE(class _OtherExtents)
  _CCCL_REQUIRES(is_constructible_v<extents_type, _OtherExtents> _CCCL_AND(extents_type::rank() > 0))
  _CCCL_API explicit constexpr mapping(const layout_stride::mapping<_OtherExtents>& __other) noexcept
      : __base(0, __other.extents())
  {
    _CCCL_ASSERT(__check_strides(__other),
                 "layout_left_padded::mapping from layout_stride ctor: strides are not compatible with "
                 "layout_left_padded.");
    _CCCL_ASSERT(
      __mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
      "layout_left_padded::mapping from layout_stride ctor: other.required_span_size() must be representable as "
      "index_type.");
  }

  _CCCL_TEMPLATE(class _OtherExtents)
  _CCCL_REQUIRES(is_constructible_v<extents_type, _OtherExtents> _CCCL_AND(extents_type::rank() == 0))
  _CCCL_API constexpr mapping(const layout_stride::mapping<_OtherExtents>& __other) noexcept
      : __base(0, __other.extents())
  {}

  // [mdspan.layout.left.obs], observers
  [[nodiscard]] _CCCL_API constexpr const extents_type& extents() const noexcept
  {
    return __extents_;
  }

  [[nodiscard]] _CCCL_API constexpr index_type required_span_size() const noexcept
  {
    if constexpr (extents_type::rank() == 0)
    {
      return 1;
    }
    else
    {
      index_type __size = 1;
      for (size_t __r = 0; __r != extents_type::rank(); __r++)
      {
        __size *= __extents_.extent(__r);
      }
      return __size;
    }
  }

  template <size_t... _Pos>
  [[nodiscard]] _CCCL_API constexpr index_type
  __op_index(const array<index_type, _Extents::rank()>& __idx_a, index_sequence<_Pos...>) const noexcept
  {
    if constexpr (sizeof...(_Pos) == 0)
    {
      return 0;
    }
    else
    {
      index_type __res = 0;
      ((__res = __idx_a[extents_type::rank() - 1 - _Pos] + __extents_.extent(extents_type::rank() - 1 - _Pos) * __res),
       ...);
      return __res;
    }
  }

  template <class... _Indices>
  static constexpr bool __can_operator_bracket =
    (is_convertible_v<_Indices, index_type> && ... && true)
    && (is_nothrow_constructible_v<index_type, _Indices> && ... && true);

  _CCCL_TEMPLATE(class... _Indices)
  _CCCL_REQUIRES((sizeof...(_Indices) == extents_type::rank()) _CCCL_AND __can_operator_bracket<_Indices...>)
  [[nodiscard]] _CCCL_API constexpr index_type operator()(_Indices... __idx) const noexcept
  {
    // Mappings are generally meant to be used for accessing allocations and are meant to guarantee to never
    // return a value exceeding required_span_size(), which is used to know how large an allocation one needs
    // Thus, this is a canonical point in multi-dimensional data structures to make invalid element access checks
    // However, mdspan does check this on its own, so for now we avoid double checking in hardened mode
    _CCCL_ASSERT(__mdspan_detail::__is_multidimensional_index_in(__extents_, __idx...),
                 "layout_left_padded::mapping: out of bounds indexing");

    const array<index_type, extents_type::rank()> __idx_a{static_cast<index_type>(__idx)...};
    return __op_index(__idx_a, make_index_sequence<sizeof...(_Indices)>());
  }

  [[nodiscard]] _CCCL_API static constexpr bool is_always_unique() noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_API static constexpr bool is_always_exhaustive() noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_API static constexpr bool is_always_strided() noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_API static constexpr bool is_unique() noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_API static constexpr bool is_exhaustive() noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_API static constexpr bool is_strided() noexcept
  {
    return true;
  }

  _CCCL_TEMPLATE(class _Extents2 = _Extents)
  _CCCL_REQUIRES((_Extents2::rank() > 0))
  [[nodiscard]] _CCCL_API constexpr index_type stride(rank_type __r) const noexcept
  {
    // While it would be caught by extents itself too, using a too large __r
    // is functionally an out of bounds access on the stored information needed to compute strides
    _CCCL_ASSERT(__r < extents_type::rank(), "layout_left_padded::mapping::stride(): invalid rank index");
    index_type __s = 1;
    for (rank_type __i = 0; __i < __r; __i++)
    {
      __s *= __extents_.extent(__i);
    }
    return __s;
  }

  _CCCL_TEMPLATE(class _OtherExtents, class _Extents2 = _Extents)
  _CCCL_REQUIRES((_OtherExtents::rank() == _Extents2::rank()))
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator==(const mapping& __lhs, const mapping<_OtherExtents>& __rhs) noexcept
  {
    return __lhs.extents() == __rhs.extents();
  }

#if _CCCL_STD_VER <= 2017
  _CCCL_TEMPLATE(class _OtherExtents, class _Extents2 = _Extents)
  _CCCL_REQUIRES((_OtherExtents::rank() == _Extents2::rank()))
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const mapping& __lhs, const mapping<_OtherExtents>& __rhs) noexcept
  {
    return __lhs.extents() != __rhs.extents();
  }
#endif // _CCCL_STD_VER <= 2017
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MDSPAN_LAYOUT_LEFT_PADDED_H
