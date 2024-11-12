//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_CONDITIONAL_H
#define _CUDA_STD___TYPE_TRAITS_CONDITIONAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <bool _Bp, class _If, class _Then>
struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional
{
  using type = _If;
};
template <class _If, class _Then>
struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional<false, _If, _Then>
{
  using type = _Then;
};

#if defined(_CCCL_COMPILER_MSVC)

template <bool _Cond, class _If, class _Then>
using conditional_t _CCCL_NODEBUG_ALIAS = typename conditional<_Cond, _If, _Then>::type;

#else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv

// Optimized implementation of `conditional_t` instantiating only two classes
template <bool>
struct __conditional_impl;

template <>
struct __conditional_impl<true>
{
  template <class _If, class _Then>
  using type _CCCL_NODEBUG_ALIAS = _If;
};

template <>
struct __conditional_impl<false>
{
  template <class _If, class _Then>
  using type _CCCL_NODEBUG_ALIAS = _Then;
};

template <bool _Cond, class _If, class _Then>
using conditional_t _CCCL_NODEBUG_ALIAS = typename __conditional_impl<_Cond>::template type<_If, _Then>;

#endif // !_CCCL_COMPILER_MSVC

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_CONDITIONAL_H
