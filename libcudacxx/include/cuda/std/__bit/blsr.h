//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_BLSR_H
#define _LIBCUDACXX___BIT_BLSR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr unsigned __cccl_blsr(unsigned __x) noexcept
{
  return __x ^ (__x & -__x);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr unsigned long __cccl_blsr(unsigned long __x) noexcept
{
  return __x ^ (__x & -__x);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr unsigned long long __cccl_blsr(unsigned long long __x) noexcept
{
  return __x ^ (__x & -__x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCPP___BIT_BLSR_H
