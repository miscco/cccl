//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PSTL_BACKEND_H
#define _LIBCUDACXX___ALGORITHM_PSTL_BACKEND_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/pstl_backends/cpu_backend.h>
#include <cuda/std/__execution/policy.h>

#if !defined(_LIBCUDACXX_HAS_NO_INCOMPLETE_PSTL) && _CCCL_STD_VER >= 2017

_LIBCUDACXX_BEGIN_NAMESPACE_STD

/*
TODO: Documentation of how backends work

A PSTL parallel backend is a tag type to which the following functions are associated, at minimum:

  template <class _ExecutionPolicy, class _Iterator, class _Func>
  __pstl_optional<__empty> __pstl_for_each(_Backend, _ExecutionPolicy&&, _Iterator __first, _Iterator __last, _Func
__f);

// TODO: Complete this list

The following functions are __pstl_optional but can be provided. If provided, they are used by the corresponding
algorithms, otherwise they are implemented in terms of other algorithms. If none of the __pstl_optional algorithms are
implemented, all the algorithms will eventually forward to the basis algorithms listed above:

  template <class _ExecutionPolicy, class _Iterator, class _Size, class _Func>
  __pstl_optional<__empty> __pstl_for_each_n(_Backend, _Iterator __first, _Size __n, _Func __f);

// TODO: Complete this list

Exception handling
==================

PSTL backends are expected to report errors (i.e. failure to allocate) by returning a disengaged `__pstl_optional` from
their implementation. Exceptions shouldn't be used to report an internal failure-to-allocate, since all exceptions are
turned into a program termination at the front-end level. When a backend returns a disengaged `__pstl_optional` to the
frontend, the frontend will turn that into a call to `std::__throw_bad_alloc();` to report the internal failure to the
user.
*/

template <class _ExecutionPolicy>
struct __select_backend;

template <>
struct __select_backend<_CUDA_VEXEC::sequenced_policy>
{
  using type = __cpu_backend_tag;
};

template <>
struct __select_backend<_CUDA_VEXEC::unsequenced_policy_host>
{
  using type = __cpu_backend_tag;
};

template <>
struct __select_backend<_CUDA_VEXEC::unsequenced_policy_device>
{
  // TODO: fixme
  using type = __cpu_backend_tag;
};

#  if defined(_LIBCUDACXX_PSTL_CPU_BACKEND_SERIAL)
template <>
struct __select_backend<_CUDA_VEXEC::parallel_policy_host>
{
  using type = __cpu_backend_tag;
};

template <>
struct __select_backend<_CUDA_VEXEC::parallel_policy_device>
{
  // TODO: fixme
  using type = __cpu_backend_tag;
};

template <>
struct __select_backend<_CUDA_VEXEC::parallel_unsequenced_policy_host>
{
  using type = __cpu_backend_tag;
};

template <>
struct __select_backend<_CUDA_VEXEC::parallel_unsequenced_policy_device>
{
  // TODO: fixme
  using type = __cpu_backend_tag;
};

#  else

// ...New vendors can add parallel backends here...

#    error "Invalid choice of a PSTL parallel backend"
#  endif

template <class _ExecutionPolicy>
using __select_backend_t = typename __select_backend<_ExecutionPolicy>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // !defined(_LIBCUDACXX_HAS_NO_INCOMPLETE_PSTL) && _CCCL_STD_VER >= 2017

#endif // _LIBCUDACXX___ALGORITHM_PSTL_BACKEND_H
