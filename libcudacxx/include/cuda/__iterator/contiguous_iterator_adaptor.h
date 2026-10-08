//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_CONTIGUOUS_ITERATOR_ADAPTOR_H
#define _CUDA___ITERATOR_CONTIGUOUS_ITERATOR_ADAPTOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/iterator.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief True when `_Iter` provides `__rebase` for a pointer to `_Tp`.
template <class _Iter>
_CCCL_CONCEPT __iterator_can_unwrap = _CCCL_REQUIRES_EXPR((_Iter), const _Iter& __iter)((__iter.base()));

//! @brief Peel iterator adaptors until the base iterator remains.
//! @param[in] __iter The iterator to unwrap.
//! @return `__iter` when it is not an iterator adaptor. Otherwise the unwrapped base iterator.
template <class _Iter>
[[nodiscard]] _CCCL_API constexpr auto __unwrap_iterator_adaptor(const _Iter& __iter)
{
  if constexpr (__iterator_can_unwrap<_Iter>)
  {
    return ::cuda::__unwrap_iterator_adaptor(__iter.base());
  }
  else if constexpr (::cuda::std::__can_to_address<_Iter>)
  { // Support standard contiguous iterators
    return ::cuda::std::to_address(__iter);
  }
  else
  {
    return __iter;
  }
}

//! @brief The contiguous iterator that remains after peeling adaptors from `_Iter`.
template <class _Iter>
using __unwrapped_iterator_t = decltype(::cuda::__unwrap_iterator_adaptor(::cuda::std::declval<_Iter>()));

//! @brief The value type of @ref __unwrapped_iterator_t.
template <class _Iter>
using __unwrapped_iter_value_t = ::cuda::std::iter_value_t<__unwrapped_iterator_t<_Iter>>;

template <class _Iter, class _Tp = __unwrapped_iter_value_t<_Iter>>
_CCCL_CONCEPT __iterator_can_rebase =
  _CCCL_REQUIRES_EXPR((_Iter, _Tp), _Iter& __iter, _Tp* __ptr)((__iter.__rebase(__ptr)));

template <class _Iter, class _Tp>
_CCCL_API constexpr void __rebase_contiguous_iterator_adaptor(_Iter& __iter, _Tp* __ptr)
{
  if constexpr (__iterator_can_rebase<_Iter, _Tp>)
  {
    __iter.__rebase(__ptr);
  }
  else if constexpr (::cuda::std::is_pointer_v<_Iter>)
  {
    __iter = __ptr;
  }
}

//! @brief True when `_Iter` is contiguous storage, or an adaptor whose stored iterator is.
//! @tparam _Iter The iterator type to query. cv-qualifiers and references are ignored.
template <class _Iter>
inline constexpr bool __is_contiguous_iterator_or_adaptor_v =
  ::cuda::std::is_pointer_v<::cuda::std::remove_cvref_t<_Iter>>
  || ::cuda::std::__has_contiguous_traversal<::cuda::std::remove_cvref_t<_Iter>>;

template <class _Iter>
inline constexpr bool __is_contiguous_iterator_or_adaptor_v<const _Iter> = __is_contiguous_iterator_or_adaptor_v<_Iter>;

template <class _Iter>
inline constexpr bool __is_contiguous_iterator_or_adaptor_v<volatile _Iter> =
  __is_contiguous_iterator_or_adaptor_v<_Iter>;

template <class _Iter>
inline constexpr bool __is_contiguous_iterator_or_adaptor_v<const volatile _Iter> =
  __is_contiguous_iterator_or_adaptor_v<_Iter>;

template <class _Iter>
inline constexpr bool __is_contiguous_iterator_or_adaptor_v<_Iter&> = __is_contiguous_iterator_or_adaptor_v<_Iter>;

template <class _Iter>
inline constexpr bool __is_contiguous_iterator_or_adaptor_v<_Iter&&> = __is_contiguous_iterator_or_adaptor_v<_Iter>;

template <class _Fn, class _Iter>
inline constexpr bool __is_contiguous_iterator_or_adaptor_v<::cuda::transform_iterator<_Fn, _Iter>> =
  ::cuda::std::is_pointer_v<__unwrapped_iterator_t<::cuda::std::remove_cvref_t<_Iter>>>;

template <class _Fn, class _Iter>
inline constexpr bool __is_contiguous_iterator_or_adaptor_v<::cuda::transform_output_iterator<_Fn, _Iter>> =
  ::cuda::std::is_pointer_v<__unwrapped_iterator_t<::cuda::std::remove_cvref_t<_Iter>>>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_CONTIGUOUS_ITERATOR_ADAPTOR_H
