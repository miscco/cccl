//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PSTL_OPTIONAL_H
#define _LIBCUDACXX___ALGORITHM_PSTL_OPTIONAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/empty.h>
#include <cuda/std/__utility/forward.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __pstl_abort
{};

template <class _Tp>
struct __pstl_optional
{
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __pstl_optional(__pstl_abort) noexcept
      : __dummy_()
      , __engaged_(false)
  {}
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __pstl_optional(const _Tp& __val) noexcept
      : __val_(__val)
      , __engaged_(true)
  {}
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __pstl_optional(_Tp&& __val) noexcept
      : __val_(_CUDA_VSTD::forward<_Tp>(__val))
      , __engaged_(true)
  {}
  _LIBCUDACXX_HIDE_FROM_ABI ~__pstl_optional()
  {
    if (__engaged_)
    {
      __val_.~_Tp();
    }
  }
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr operator bool() const noexcept
  {
    return __engaged_;
  }

  union
  {
    _Tp __val_;
    char __dummy_;
  };
  bool __engaged_;
};

template <>
struct __pstl_optional<__empty>
{
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __pstl_optional(__pstl_abort) noexcept
      : __engaged_(false)
  {}
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __pstl_optional(const __empty&) noexcept
      : __engaged_(true)
  {}
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __pstl_optional(__empty&&) noexcept
      : __engaged_(true)
  {}
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr operator bool() const noexcept
  {
    return __engaged_;
  }

  bool __engaged_;
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_PSTL_OPTIONAL_H
