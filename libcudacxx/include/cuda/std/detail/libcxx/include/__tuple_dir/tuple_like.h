//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_TUPLE_LIKE_H
#define _LIBCUDACXX___TUPLE_TUPLE_LIKE_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#include "../__fwd/array.h"
#include "../__fwd/pair.h"
#include "../__fwd/subrange.h"
#include "../__fwd/tuple.h"
#include "../__tuple_dir/tuple_types.h"
#include "../__type_traits/integral_constant.h"
#include "../cstddef"

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __tuple_like : false_type
{};

template <class _Tp>
struct __tuple_like<const _Tp> : public __tuple_like<_Tp>
{};
template <class _Tp>
struct __tuple_like<volatile _Tp> : public __tuple_like<_Tp>
{};
template <class _Tp>
struct __tuple_like<const volatile _Tp> : public __tuple_like<_Tp>
{};

template <class... _Tp>
struct __tuple_like<tuple<_Tp...> > : true_type
{};

template <class _T1, class _T2>
struct __tuple_like<pair<_T1, _T2> > : true_type {};

template <class _Tp, size_t _Size>
struct __tuple_like<array<_Tp, _Size> > : true_type {};

#if _LIBCUDACXX_STD_VER > 14
template <class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct __tuple_like<_CUDA_VRANGES::subrange<_Ip, _Sp, _Kp> > : true_type {};
#endif

template <class... _Tp>
struct __tuple_like<__tuple_types<_Tp...> > : true_type
{};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TUPLE_TUPLE_LIKE_H
