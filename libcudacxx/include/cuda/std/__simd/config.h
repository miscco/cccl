//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___SIMD_CONFIG_H
#define _LIBCUDACXX___SIMD_CONFIG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// TODO: support more targets
#if defined(__AVX__)
#  define _LIBCUDACXX_NATIVE_SIMD_WIDTH_IN_BYTES 32
#else
#  define _LIBCUDACXX_NATIVE_SIMD_WIDTH_IN_BYTES 16
#endif

#endif // _LIBCUDACXX___SIMD_CONFIG_H
