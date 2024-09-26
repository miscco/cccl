//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_IS_IMPLICITLY_CONVERTIBLE_H
#define __CCCL_IS_IMPLICITLY_CONVERTIBLE_H

#include <cuda/std/__cccl/compiler.h>

//! There is compiler bug that results in incorrect results for the below `__is_implicitly_convertible` check.
//! This breaks some common functionality, so this *must* be included outside of a system header. See nvbug4867473.
#if defined(_CCCL_FORCE_SYSTEM_HEADER_GCC) || defined(_CCCL_FORCE_SYSTEM_HEADER_CLANG) \
  || defined(_CCCL_FORCE_SYSTEM_HEADER_MSVC)
#  error \
    "This header must be included only within the <cuda/std/__cccl/system_header>. This most likely means a mix and match of different versions of CCCL."
#endif // system header detected

namespace __cccl_internal
{

#if defined(_CCCL_CUDA_COMPILER) && (defined(__CUDACC__) || defined(_NVHPC_CUDA))
template <class _Tp>
__host__ __device__ _Tp&& __cccl_declval(int);
template <class _Tp>
__host__ __device__ _Tp __cccl_declval(long);
template <class _Tp>
__host__ __device__ decltype(__cccl_internal::__cccl_declval<_Tp>(0)) __cccl_declval() noexcept;
#else // ^^^ CUDA compilation ^^^ / vvv no CUDA compilation
template <class _Tp>
_Tp&& __cccl_declval(int);
template <class _Tp>
_Tp __cccl_declval(long);
template <class _Tp>
decltype(__cccl_internal::__cccl_declval<_Tp>(0)) __cccl_declval() noexcept;
#endif // no CUDA compilation

template <class...>
using __cccl_void_t = void;

template <class _Dest, class _Source, class = void>
struct __is_implicitly_convertible
{
  static constexpr bool value = false;
};

template <class _Dest, class _Source>
struct __is_implicitly_convertible<_Dest,
                                   _Source,
                                   __cccl_void_t<decltype(_Dest{__cccl_internal::__cccl_declval<_Source>()})>>
{
  static constexpr bool value = true;
};

} // namespace __cccl_internal

#endif // __CCCL_IS_IMPLICITLY_CONVERTIBLE_H
