// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_LAYOUT_STRIDE_H
#define _LIBCUDACXX___MDSPAN_LAYOUT_STRIDE_H

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
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/as_const.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/detail/libcxx/include/__assert>
#include <cuda/std/limits>

_CCCL_PUSH_MACROS

#if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __layout_stride_detail
{

struct __constraints
{
  template <class _StridedLayoutMapping, class _Extents>
  static constexpr bool __can_convert =
    __mdspan_detail::__layout_mapping_alike<_StridedLayoutMapping> //
    && _StridedLayoutMapping::is_always_unique() && _StridedLayoutMapping::is_always_strided()
    && _CCCL_TRAIT(is_constructible, _Extents, typename _StridedLayoutMapping::extents_type);

  template <class _StridedLayoutMapping, class _Extents>
  static constexpr bool __converts_implicit =
    _CCCL_TRAIT(is_convertible, typename _StridedLayoutMapping::extents_type, _Extents)
    && (__mdspan_detail::__is_mapping_of<layout_left, _StridedLayoutMapping>
        || __mdspan_detail::__is_mapping_of<layout_right, _StridedLayoutMapping>
        || __mdspan_detail::__is_mapping_of<layout_stride, _StridedLayoutMapping>);
};

} // namespace __layout_stride_detail

template <class _Extents>
class layout_stride::mapping
{
public:
  static_assert(__mdspan_detail::__is_extents<_Extents>::value,
                "layout_stride::mapping template argument must be a specialization of extents.");

  using extents_type = _Extents;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = layout_stride;

private:
  static constexpr rank_type __rank_ = extents_type::rank();

  using __stride_array = __mdspan_detail::__possibly_empty_array<index_type, extents_type::rank()>;

  // Used for default construction check and mandates
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool
  __mul_overflow(index_type __x, index_type __y, index_type* __res) noexcept
  {
    *__res = __x * __y;
    return __x && ((*__res / __x) != __y);
  }
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool
  __add_overflow(index_type __x, index_type __y, index_type* __res) noexcept
  {
    *__res = __x + __y;
    return *__res < __y;
  }

  _LIBCUDACXX_TEMPLATE(size_t _Rank = _Extents::rank())
  _LIBCUDACXX_REQUIRES((_Rank != 0))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool
  __required_span_size_is_representable(const extents_type& __ext) noexcept
  {
    index_type __prod = __ext.extent(0);
    for (rank_type __r = 1; __r < __rank_; __r++)
    {
      bool __overflowed = __mul_overflow(__prod, __ext.extent(__r), &__prod);
      if (__overflowed)
      {
        return false;
      }
    }
    return true;
  }
  _LIBCUDACXX_TEMPLATE(size_t _Rank = _Extents::rank())
  _LIBCUDACXX_REQUIRES((_Rank == 0))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool
  __required_span_size_is_representable(const extents_type& __ext) noexcept
  {
    return true;
  }

  _LIBCUDACXX_TEMPLATE(class _OtherIndexType)
  _LIBCUDACXX_REQUIRES(_CCCL_TRAIT(is_integral, _OtherIndexType))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool __conversion_may_overflow(_OtherIndexType __stride) noexcept
  {
    using _CommonType = common_type_t<index_type, _OtherIndexType>;
    return static_cast<_CommonType>(__stride) > static_cast<_CommonType>((numeric_limits<index_type>::max)());
  }
  _LIBCUDACXX_TEMPLATE(class _OtherIndexType)
  _LIBCUDACXX_REQUIRES((!_CCCL_TRAIT(is_integral, _OtherIndexType)))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool __conversion_may_overflow(_OtherIndexType) noexcept
  {
    return false;
  }

  _LIBCUDACXX_TEMPLATE(class _OtherIndexType)
  _LIBCUDACXX_REQUIRES((__rank_ != 0))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool __required_span_size_is_representable(
    const extents_type& __ext, span<_OtherIndexType, extents_type::rank()> __strides)
  {
    index_type __size = 1;
    for (rank_type __r = 0; __r < __rank_; __r++)
    {
      // We can only check correct conversion of _OtherIndexType if it is an integral
      if (__conversion_may_overflow(__strides[__r]))
      {
        return false;
      }
      if (__ext.extent(__r) == static_cast<index_type>(0))
      {
        return true;
      }

      index_type __prod = (__ext.extent(__r) - 1);
      if (__mul_overflow(__prod, static_cast<index_type>(__strides[__r]), &__prod))
      {
        return false;
      }
      if (__add_overflow(__size, __prod, &__size))
      {
        return false;
      }
    }
    return true;
  }

  _LIBCUDACXX_TEMPLATE(class _OtherIndexType)
  _LIBCUDACXX_REQUIRES((__rank_ == 0))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool __required_span_size_is_representable(
    const extents_type& __ext, span<_OtherIndexType, extents_type::rank()> __strides)
  {
    return true;
  }

  // compute offset of a strided layout mapping
  template <class _StridedMapping, size_t... _Pos>
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr auto
  __offset(const _StridedMapping& __mapping, index_sequence<_Pos...>) noexcept
  {
    using index_type = typename _StridedMapping::index_type;
    return static_cast<index_type>(__mapping((_Pos ? 0 : 0)...));
  }

  _LIBCUDACXX_TEMPLATE(class _StridedMapping)
  _LIBCUDACXX_REQUIRES((_StridedMapping::extents_type::rank() != 0))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr index_type __offset(const _StridedMapping& __mapping)
  {
    if (__mapping.required_span_size() == static_cast<typename _StridedMapping::index_type>(0))
    {
      return static_cast<index_type>(0);
    }
    else
    {
      return __offset(__mapping, _CUDA_VSTD::make_index_sequence<__rank_>());
    }
  }
  _LIBCUDACXX_TEMPLATE(class _StridedMapping)
  _LIBCUDACXX_REQUIRES((_StridedMapping::extents_type::rank() == 0))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr index_type __offset(const _StridedMapping& __mapping)
  {
    return static_cast<index_type>(__mapping());
  }

  static_assert((extents_type::rank_dynamic() > 0) || __required_span_size_is_representable(extents_type()),
                "layout_stride::mapping product of static extents must be representable as index_type.");

public:
  // [mdspan.layout.stride.cons], constructors
  _LIBCUDACXX_TEMPLATE(size_t _Rank = _Extents::rank())
  _LIBCUDACXX_REQUIRES((_Rank == 0))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr mapping() noexcept
      : __extents_(extents_type())
  {}

  _LIBCUDACXX_TEMPLATE(size_t _Rank = _Extents::rank())
  _LIBCUDACXX_REQUIRES((_Rank > 0))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr mapping() noexcept
      : __extents_(extents_type())
  {
    index_type __stride = 1;
    for (rank_type __r = __rank_ - 1; __r > static_cast<rank_type>(0); __r--)
    {
      __strides_[__r] = __stride;
      __stride *= __extents_.extent(__r);
    }
    __strides_[0] = __stride;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr mapping(const mapping&) noexcept = default;

  template <class _OtherIndexType, size_t... _Pos>
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr auto
  __to_strides_array(span<_OtherIndexType, extents_type::rank()> __strides, index_sequence<_Pos...>) noexcept
  {
    (void) __strides; // nvcc believes strides is unused here
    return __stride_array{static_cast<index_type>(_CUDA_VSTD::as_const(__strides[_Pos]))...};
  }

  _LIBCUDACXX_TEMPLATE(class _OtherIndexType, size_t... _Pos)
  _LIBCUDACXX_REQUIRES(_CCCL_TRAIT(is_integral, _OtherIndexType))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr auto
  __check_strides(span<_OtherIndexType, extents_type::rank()> __strides, index_sequence<_Pos...>) noexcept
  {
    return _CCCL_FOLD_AND((__strides[_Pos] > static_cast<_OtherIndexType>(0)));
  }

  _LIBCUDACXX_TEMPLATE(class _OtherIndexType, size_t... _Pos)
  _LIBCUDACXX_REQUIRES((!_CCCL_TRAIT(is_integral, _OtherIndexType)))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr auto
  __check_strides(span<_OtherIndexType, extents_type::rank()> __strides, index_sequence<_Pos...>) noexcept
  {
    return _CCCL_FOLD_AND((static_cast<index_type>(__strides[_Pos]) > static_cast<index_type>(0)));
  }

  // compute the permutation for sorting the stride array
  // we never actually sort the stride array
  template <class _OtherIndexType>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr void
  __bubble_sort_by_strides(array<rank_type, extents_type::rank()>& __permute) const noexcept
  {
    for (rank_type __i = __rank_ - 1; __i > 0; __i--)
    {
      for (rank_type __r = 0; __r < __i; __r++)
      {
        if (__strides_[__permute[__r]] > __strides_[__permute[__r + 1]])
        {
          swap(__permute[__r], __permute[__r + 1]);
        }
        else
        {
          // if two strides are the same then one of the associated extents must be 1 or 0
          // both could be, but you can't have one larger than 1 come first
          if ((__strides_[__permute[__r]] == __strides_[__permute[__r + 1]])
              && (__extents_.extent(__permute[__r]) > static_cast<index_type>(1)))
          {
            swap(__permute[__r], __permute[__r + 1]);
          }
        }
      }
    }
  }

  template <size_t... _Pos>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr bool __check_unique_mapping(index_sequence<_Pos...>) const noexcept
  {
    // basically sort the dimensions based on strides and extents, sorting is represented in permute array
    array<rank_type, __rank_> __permute{_Pos...};
    __bubble_sort_by_strides(__permute);

    // check that this permutations represents a growing set
    for (rank_type __i = 1; __i < __rank_; __i++)
    {
      if (static_cast<index_type>(__strides_[__permute[__i]])
          < static_cast<index_type>(__strides_[__permute[__i - 1]]) * __extents_.extent(__permute[__i - 1]))
      {
        return false;
      }
    }
    return true;
  }
  _LIBCUDACXX_INLINE_VISIBILITY constexpr bool __check_unique_mapping(index_sequence<>) const noexcept
  {
    return true;
  }

  // nvcc cannot deduce this constructor when using _LIBCUDACXX_REQUIRES
  template <class _OtherIndexType,
            enable_if_t<_CCCL_TRAIT(is_constructible, index_type, const _OtherIndexType&), int> = 0,
            enable_if_t<_CCCL_TRAIT(is_convertible, const _OtherIndexType&, index_type), int>   = 0>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr mapping(const extents_type& __ext,
                                                  span<_OtherIndexType, extents_type::rank()> __strides) noexcept
      : __extents_(__ext)
      , __strides_(__to_strides_array(__strides, _CUDA_VSTD::make_index_sequence<extents_type::rank()>()))
  {
    _LIBCUDACXX_ASSERT_VALID_ELEMENT_ACCESS(
      __check_strides(__strides), "layout_stride::mapping ctor: all strides must be greater than 0");
    _LIBCUDACXX_ASSERT_VALID_ELEMENT_ACCESS(
      __required_span_size_is_representable(__ext, __strides),
      "layout_stride::mapping ctor: required span size is not representable as index_type.");
    _LIBCUDACXX_ASSERT_UNCATEGORIZED(
      __check_unique_mapping(_CUDA_VSTD::make_index_sequence<__rank_>()),
      "layout_stride::mapping ctor: the provided extents and strides lead to a non-unique mapping");
  }

  // nvcc cannot deduce this constructor when using _LIBCUDACXX_REQUIRES
  template <class _OtherIndexType,
            enable_if_t<_CCCL_TRAIT(is_constructible, index_type, const _OtherIndexType&), int> = 0,
            enable_if_t<_CCCL_TRAIT(is_convertible, const _OtherIndexType&, index_type), int>   = 0>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr mapping(const extents_type& __ext,
                                                  const array<_OtherIndexType, extents_type::rank()> __strides) noexcept
      : mapping(__ext, span<const _OtherIndexType, extents_type::rank()>(__strides))
  {}

  template <class _StridedLayoutMapping, size_t... _Pos>
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr auto
  __to_strides_array(const _StridedLayoutMapping& __other, index_sequence<_Pos...>) noexcept
  {
    return __stride_array{static_cast<index_type>(__other.stride(_Pos))...};
  }

  // stride() only compiles for rank > 0
  template <class _StridedLayoutMapping, size_t... _Pos>
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr auto
  __check_strides(const _StridedLayoutMapping& __other, index_sequence<_Pos...>) noexcept
  {
    return _CCCL_FOLD_AND((static_cast<index_type>(__other.stride(_Pos)) > static_cast<index_type>(0)));
  }
  template <class _StridedLayoutMapping>
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr auto
  __check_strides(const _StridedLayoutMapping&, index_sequence<>) noexcept
  {
    return true;
  }

  _LIBCUDACXX_TEMPLATE(class _StridedLayoutMapping)
  _LIBCUDACXX_REQUIRES(
    __layout_stride_detail::__constraints::__can_convert<_StridedLayoutMapping, _Extents> _LIBCUDACXX_AND
      __layout_stride_detail::__constraints::__converts_implicit<_StridedLayoutMapping, _Extents>)
  _LIBCUDACXX_INLINE_VISIBILITY constexpr mapping(const _StridedLayoutMapping& __other) noexcept
      : __extents_(__other.extents())
      , __strides_(__to_strides_array(__other, _CUDA_VSTD::make_index_sequence<__rank_>()))
  {
    _LIBCUDACXX_ASSERT_VALID_ELEMENT_ACCESS(
      __check_strides(__other, _CUDA_VSTD::make_index_sequence<__rank_>()),
      "layout_stride::mapping converting ctor: all strides must be greater than 0");
    _LIBCUDACXX_ASSERT_VALID_ELEMENT_ACCESS(
      __mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
      "layout_stride::mapping converting ctor: other.required_span_size() must be representable as index_type.");
    _LIBCUDACXX_ASSERT_VALID_ELEMENT_ACCESS(
      static_cast<index_type>(0) == __offset(__other),
      "layout_stride::mapping converting ctor: base offset of mapping must be zero.");
  }
  _LIBCUDACXX_TEMPLATE(class _StridedLayoutMapping)
  _LIBCUDACXX_REQUIRES(
    __layout_stride_detail::__constraints::__can_convert<_StridedLayoutMapping, _Extents> _LIBCUDACXX_AND(
      !__layout_stride_detail::__constraints::__converts_implicit<_StridedLayoutMapping, _Extents>))
  _LIBCUDACXX_INLINE_VISIBILITY explicit constexpr mapping(const _StridedLayoutMapping& __other) noexcept
      : __extents_(__other.extents())
      , __strides_(__to_strides_array(__other, _CUDA_VSTD::make_index_sequence<__rank_>()))
  {
    _LIBCUDACXX_ASSERT_VALID_ELEMENT_ACCESS(
      __check_strides(__other, _CUDA_VSTD::make_index_sequence<__rank_>()),
      "layout_stride::mapping converting ctor: all strides must be greater than 0");
    _LIBCUDACXX_ASSERT_VALID_ELEMENT_ACCESS(
      __mdspan_detail::__is_representable_as<index_type>(__other.required_span_size()),
      "layout_stride::mapping converting ctor: other.required_span_size() must be representable as index_type.");
    _LIBCUDACXX_ASSERT_VALID_ELEMENT_ACCESS(
      static_cast<index_type>(0) == __offset(__other),
      "layout_stride::mapping converting ctor: base offset of mapping must be zero.");
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr mapping& operator=(const mapping&) noexcept = default;

  // [mdspan.layout.stride.obs], observers
  _LIBCUDACXX_INLINE_VISIBILITY constexpr const extents_type& extents() const noexcept
  {
    return __extents_;
  }

  template <size_t... _Pos>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr array<index_type, extents_type::rank()>
  __to_strides(index_sequence<_Pos...>) const noexcept
  {
    return array<index_type, extents_type::rank()>{__strides_[_Pos]...};
  }

  _LIBCUDACXX_INLINE_VISIBILITY constexpr array<index_type, extents_type::rank()> strides() const noexcept
  {
    return __to_strides(_CUDA_VSTD::make_index_sequence<__rank_>());
  }

  template <size_t... _Pos>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr index_type __required_span_size(index_sequence<_Pos...>) const noexcept
  {
    const index_type __product = _CCCL_FOLD_TIMES(index_type{1}, __extents_.extent(_Pos));
    if (__product == index_type{0})
    {
      return index_type{0};
    }
    else
    {
      return _CCCL_FOLD_PLUS(index_type{1}, ((__extents_.extent(_Pos) - static_cast<index_type>(1)) * __strides_[_Pos]));
    }
  }

  _LIBCUDACXX_TEMPLATE(size_t _Rank = _Extents::rank())
  _LIBCUDACXX_REQUIRES((_Rank > 0))
  _LIBCUDACXX_INLINE_VISIBILITY constexpr index_type required_span_size() const noexcept
  {
    return __required_span_size(_CUDA_VSTD::make_index_sequence<__rank_>());
  }

  _LIBCUDACXX_TEMPLATE(size_t _Rank = _Extents::rank())
  _LIBCUDACXX_REQUIRES((_Rank == 0))
  _LIBCUDACXX_INLINE_VISIBILITY constexpr index_type required_span_size() const noexcept
  {
    return static_cast<index_type>(1);
  }

  template <size_t... _Pos, class... _Indices>
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr index_type
  __op_index(const __stride_array& __strides, index_sequence<_Pos...>, _Indices... __idx) noexcept
  {
    return _CCCL_FOLD_PLUS(index_type{0}, (static_cast<index_type>(__idx) * __strides[_Pos]));
  }

  _LIBCUDACXX_TEMPLATE(class... _Indices)
  _LIBCUDACXX_REQUIRES((sizeof...(_Indices) == __rank_)
                         _LIBCUDACXX_AND _CCCL_FOLD_AND(_CCCL_TRAIT(is_convertible, _Indices, index_type))
                           _LIBCUDACXX_AND _CCCL_FOLD_AND(_CCCL_TRAIT(is_nothrow_constructible, index_type, _Indices)))
  _LIBCUDACXX_INLINE_VISIBILITY constexpr index_type operator()(_Indices... __idx) const noexcept
  {
    // Mappings are generally meant to be used for accessing allocations and are meant to guarantee to never
    // return a value exceeding required_span_size(), which is used to know how large an allocation one needs
    // Thus, this is a canonical point in multi-dimensional data structures to make invalid element access checks
    // However, mdspan does check this on its own, so for now we avoid double checking in hardened mode
    _LIBCUDACXX_ASSERT_UNCATEGORIZED(__mdspan_detail::__is_multidimensional_index_in(__extents_, __idx...),
                                     "layout_stride::mapping: out of bounds indexing");
    return __op_index(__strides_, _CUDA_VSTD::make_index_sequence<sizeof...(_Indices)>(), __idx...);
  }

  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool is_always_unique() noexcept
  {
    return true;
  }
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool is_always_exhaustive() noexcept
  {
    return false;
  }
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool is_always_strided() noexcept
  {
    return true;
  }

  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool is_unique() noexcept
  {
    return true;
  }

  // The answer of this function is fairly complex in the case where one or more
  // extents are zero.
  // Technically it is meaningless to query is_exhaustive() in that case, but unfortunately
  // the way the standard defines this function, we can't give a simple true or false then.
  template <size_t... _Pos>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr index_type __to_total_size(index_sequence<_Pos...>) const noexcept
  {
    return _CCCL_FOLD_TIMES(index_type{1}, (__extents_.extent(_Pos)));
  }

  _LIBCUDACXX_TEMPLATE(size_t _Rank = _Extents::rank())
  _LIBCUDACXX_REQUIRES((_Rank > 1))
  _LIBCUDACXX_INLINE_VISIBILITY constexpr bool is_exhaustive() const noexcept
  {
    const index_type __span_size = required_span_size();
    if (__span_size == static_cast<index_type>(0))
    {
      rank_type __r_largest = 0;
      for (rank_type __r = 1; __r < __rank_; __r++)
      {
        if (__strides_[__r] > __strides_[__r_largest])
        {
          __r_largest = __r;
        }
      }
      for (rank_type __r = 0; __r < __rank_; __r++)
      {
        if (__extents_.extent(__r) == 0 && __r != __r_largest)
        {
          return false;
        }
      }
      return true;
    }
    else
    {
      const index_type __total_size = __to_total_size(_CUDA_VSTD::make_index_sequence<__rank_>());
      return __span_size == __total_size;
    }
  }
  _LIBCUDACXX_TEMPLATE(size_t _Rank = _Extents::rank())
  _LIBCUDACXX_REQUIRES((_Rank == 1))
  _LIBCUDACXX_INLINE_VISIBILITY constexpr bool is_exhaustive() const noexcept
  {
    const index_type __span_size = required_span_size();
    if (__span_size == static_cast<index_type>(0))
    {
      return __strides_[0] == 1;
    }
    else
    {
      const index_type __total_size = __to_total_size(_CUDA_VSTD::make_index_sequence<__rank_>());
      return __span_size == __total_size;
    }
  }
  _LIBCUDACXX_TEMPLATE(size_t _Rank = _Extents::rank())
  _LIBCUDACXX_REQUIRES((_Rank == 0))
  _LIBCUDACXX_INLINE_VISIBILITY constexpr bool is_exhaustive() const noexcept
  {
    return true;
  }
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool is_strided() noexcept
  {
    return true;
  }

  // according to the standard layout_stride does not have a constraint on stride(r) for rank>0
  // it still has the precondition though
  _LIBCUDACXX_INLINE_VISIBILITY constexpr index_type stride(rank_type __r) const noexcept
  {
    _LIBCUDACXX_ASSERT_VALID_ELEMENT_ACCESS(__r < __rank_, "layout_stride::mapping::stride(): invalid rank index");
    return __strides_[__r];
  }

  template <class _OtherMapping, size_t... _Pos>
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool
  __op_eq(const mapping& __lhs, const _OtherMapping& __rhs, index_sequence<_Pos...>) noexcept
  {
    // avoid warning when comparing signed and unsigner integers and pick the wider of two types
    using _CommonType = common_type_t<index_type, typename _OtherMapping::index_type>;
    return _CCCL_FOLD_AND(
      (static_cast<_CommonType>(__lhs.stride(_Pos)) == static_cast<_CommonType>(__rhs.stride(_Pos))));
  }

  _LIBCUDACXX_TEMPLATE(class _OtherMapping, class _Extents2 = _Extents)
  _LIBCUDACXX_REQUIRES((_Extents2::rank() > 0))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool __op_eq(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
  {
    if (__offset(__rhs))
    {
      return false;
    }
    return __lhs.extents() == __rhs.extents() && __op_eq(__lhs, __rhs, _CUDA_VSTD::make_index_sequence<__rank_>());
  }

  _LIBCUDACXX_TEMPLATE(class _OtherMapping, class _Extents2 = _Extents)
  _LIBCUDACXX_REQUIRES((_Extents2::rank() == 0))
  _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool __op_eq(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
  {
    return (!__offset(__rhs));
  }

  template <class _OtherMapping>
  static constexpr bool __can_compare =
    __mdspan_detail::__layout_mapping_alike<_OtherMapping> && (_OtherMapping::extents_type::rank() == _Extents::rank())
    && _OtherMapping::is_always_strided();

  template <class _OtherMapping>
  _LIBCUDACXX_INLINE_VISIBILITY friend constexpr auto
  operator==(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__can_compare<_OtherMapping>)
  {
    return __op_eq(__lhs, __rhs);
  }

#  if _CCCL_STD_VER <= 2017
  template <class _OtherMapping>
  _LIBCUDACXX_INLINE_VISIBILITY friend constexpr auto
  operator==(const _OtherMapping& __lhs, const mapping& __rhs) noexcept _LIBCUDACXX_TRAILING_REQUIRES(bool)(
    (!__mdspan_detail::__is_mapping_of<layout_stride, _OtherMapping>) &&__can_compare<_OtherMapping>)
  {
    return __op_eq(__rhs, __lhs);
  }
  template <class _OtherMapping, class _Extents2 = _Extents>
  _LIBCUDACXX_INLINE_VISIBILITY friend constexpr auto
  operator!=(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__can_compare<_OtherMapping>)
  {
    return !__op_eq(__lhs, __rhs);
  }
  template <class _OtherMapping, class _Extents2 = _Extents>
  _LIBCUDACXX_INLINE_VISIBILITY friend constexpr auto
  operator!=(const _OtherMapping& __lhs, const mapping& __rhs) noexcept _LIBCUDACXX_TRAILING_REQUIRES(bool)(
    (!__mdspan_detail::__is_mapping_of<layout_stride, _OtherMapping>) &&__can_compare<_OtherMapping>)
  {
    return __op_eq(__rhs, __lhs);
  }
#  endif // _CCCL_STD_VER <= 2017

private:
  _CCCL_NO_UNIQUE_ADDRESS extents_type __extents_{};
  _CCCL_NO_UNIQUE_ADDRESS __stride_array __strides_{};
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_STD_VER >= 2014

_CCCL_POP_MACROS

#endif // _LIBCUDACXX___MDSPAN_LAYOUT_STRIDE_H
