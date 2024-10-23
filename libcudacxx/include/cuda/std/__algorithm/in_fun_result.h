//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_IN_FUN_RESULT_H
#define _LIBCUDACXX___ALGORITHM_IN_FUN_RESULT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__utility/move.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

template <class _InIter1, class _Func1>
struct in_fun_result
{
  _CCCL_NO_UNIQUE_ADDRESS _InIter1 in;
  _CCCL_NO_UNIQUE_ADDRESS _Func1 fun;

  _LIBCUDACXX_TEMPLATE(class _InIter2, class _Func2)
  _LIBCUDACXX_REQUIRES(convertible_to<const _InIter1&, _InIter2> _LIBCUDACXX_AND convertible_to<const _Func1&, _Func2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr operator in_fun_result<_InIter2, _Func2>() const&
  {
    return {in, fun};
  }

  _LIBCUDACXX_TEMPLATE(class _InIter2, class _Func2)
  _LIBCUDACXX_REQUIRES(convertible_to<_InIter1, _InIter2> _LIBCUDACXX_AND convertible_to<_Func1, _Func2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr operator in_fun_result<_InIter2, _Func2>() &&
  {
    return {_CUDA_VSTD::move(in), _CUDA_VSTD::move(fun)};
  }
};

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___ALGORITHM_IN_FUN_RESULT_H
