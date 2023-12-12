//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___SIMD_SIMD_SIZE_H
#define _LIBCUDACXX___SIMD_SIMD_SIZE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__simd/simd_abi.h>
#include <cuda/std/__simd/simd_storage.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_const.h>

_LIBCUDACXX_BEGIN_NAMESPACE_SIMD_ABI

template <class _Tp, class _Abi = simd_abi::compatible<_Tp>>
struct simd_size;

template <class _Tp, _StorageKind __kind, int _Np>
struct simd_size<_Tp, __simd_abi<__kind, _Np>> : integral_constant<size_t, _Np>
{
  static_assert(_CCCL_TRAIT(is_arithmetic, _Tp) && !_CCCL_TRAIT(is_same, remove_const_t<_Tp>, bool),
                "Element type should be vectorizable");
};

_LIBCUDACXX_END_NAMESPACE_SIMD_ABI

#endif // _LIBCUDACXX___SIMD_SIMD_SIZE_H
