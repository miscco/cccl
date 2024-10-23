//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_SERIAL_H
#define _LIBCUDACXX___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_SERIAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/pstl_backends/optional.h>
#include <cuda/std/__utility/empty.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstddef>

#if !defined(_LIBCUDACXX_HAS_NO_INCOMPLETE_PSTL) && _CCCL_STD_VER >= 2017

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __par_backend
{
inline namespace __serial_cpu_backend
{

template <class _RandomAccessIterator, class _Fp>
_LIBCUDACXX_HIDE_FROM_ABI __pstl_optional<__empty>
__parallel_for(_RandomAccessIterator __first, _RandomAccessIterator __last, _Fp __f)
{
  __f(__first, __last);
  return __pstl_optional<__empty>{__empty{}};
}

// TODO: Complete this list

} // namespace __serial_cpu_backend
} // namespace __par_backend

_LIBCUDACXX_END_NAMESPACE_STD

#endif // !defined(_LIBCUDACXX_HAS_NO_INCOMPLETE_PSTL) && _CCCL_STD_VER >= 2017

#endif // _LIBCUDACXX___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_SERIAL_H
