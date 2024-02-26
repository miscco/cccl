//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_ROTATE_H
#define _LIBCUDACXX___BIT_ROTATE_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__concepts/arithmetic.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_unsigned_integer.h"
#include "../limits"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp __rotl(_Tp __t, unsigned int __cnt) noexcept
{
  static_assert(__libcpp_is_unsigned_integer<_Tp>::value, "__rotl requires unsigned");
  using __nlt = numeric_limits<_Tp>;

  return ((__cnt % __nlt::digits) == 0)
         ? __t
         : (__t << (__cnt % __nlt::digits)) | (__t >> (__nlt::digits - (__cnt % __nlt::digits)));
}

template <class _Tp>
_LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp __rotr(_Tp __t, unsigned int __cnt) noexcept
{
  static_assert(__libcpp_is_unsigned_integer<_Tp>::value, "__rotr requires unsigned");
  using __nlt = numeric_limits<_Tp>;

  return ((__cnt % __nlt::digits) == 0)
         ? __t
         : (__t >> (__cnt % __nlt::digits)) | (__t << (__nlt::digits - (__cnt % __nlt::digits)));
}

template <class _Tp, __enable_if_t<__libcpp_is_unsigned_integer<_Tp>::value, int> = 0>
_LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp rotr(_Tp __t, int __cnt) noexcept
{
  return _CUDA_VSTD::__rotr(__t, __cnt);
}

template <class _Tp, __enable_if_t<__libcpp_is_unsigned_integer<_Tp>::value, int> = 0>
_LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp rotl(_Tp __t, int __cnt) noexcept
{
  return _CUDA_VSTD::__rotl(__t, __cnt);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_ROTATE_H
