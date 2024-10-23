//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PSTL_BACKENDS_CPU_BACKEND_BACKEND_H
#define _LIBCUDACXX___ALGORITHM_PSTL_BACKENDS_CPU_BACKEND_BACKEND_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>

#if defined(_LIBCUDACXX_PSTL_CPU_BACKEND_SERIAL)
#  include <cuda/std/__algorithm/pstl_backends/cpu_backends/serial.h>
#else
#  error "Invalid CPU backend choice"
#endif

#if _CCCL_STD_VER >= 2017

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __cpu_backend_tag
{};

inline constexpr size_t __lane_size = 64;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_STD_VER >= 2017

#endif // _LIBCUDACXX___ALGORITHM_PSTL_BACKENDS_CPU_BACKEND_BACKEND_H
