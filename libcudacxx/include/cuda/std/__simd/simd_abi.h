//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___SIMD_SIMD_ABI_H
#define _LIBCUDACXX___SIMD_SIMD_ABI_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__simd/config.h>
#include <cuda/std/__type_traits/integral_constant.h>

_LIBCUDACXX_BEGIN_NAMESPACE_SIMD_ABI

enum class _StorageKind
{
  _Scalar,
  _Array,
  _VecExt,
};

template <_StorageKind __kind, int _Np>
struct __simd_abi
{};

using scalar = __simd_abi<_StorageKind::_Scalar, 1>;

template <int _Np>
using fixed_size = __simd_abi<_StorageKind::_Array, _Np>;

template <class _Tp>
_CCCL_INLINE_VAR constexpr size_t max_fixed_size = 32;

template <class _Tp>
using compatible = fixed_size<16 / sizeof(_Tp)>;

#ifndef _LIBCUDACXX_HAS_NO_VECTOR_EXTENSION
template <class _Tp>
using native = __simd_abi<_StorageKind::_VecExt, _LIBCUDACXX_NATIVE_SIMD_WIDTH_IN_BYTES / sizeof(_Tp)>;
#else
template <class _Tp>
using native = __simd_abi<_StorageKind::_Array, _LIBCUDACXX_NATIVE_SIMD_WIDTH_IN_BYTES / sizeof(_Tp)>;
#endif // _LIBCUDACXX_HAS_NO_VECTOR_EXTENSION

// traits [simd.traits]
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_abi_tag_v = false;

template <_StorageKind __kind, int _Np>
_CCCL_INLINE_VAR constexpr bool is_abi_tag_v<__simd_abi<__kind, _Np>> = true;

template <class _Tp>
struct is_abi_tag : integral_constant<bool, is_abi_tag_v<_Tp>>
{};

struct element_aligned_tag
{};
_CCCL_GLOBAL_CONSTANT element_aligned_tag element_aligned{};

struct vector_aligned_tag
{};
_CCCL_GLOBAL_CONSTANT vector_aligned_tag vector_aligned{};

template <size_t>
struct overaligned_tag
{};

template <size_t _Np>
_CCCL_GLOBAL_CONSTANT overaligned_tag<_Np> overaligned{};

template <class _Tp, class _Abi = _CUDA_SIMD::compatible<_Tp>>
class simd;

template <class _Tp, class _Abi = _CUDA_SIMD::compatible<_Tp>>
class simd_mask;

_LIBCUDACXX_END_NAMESPACE_SIMD_ABI

#endif // _LIBCUDACXX___SIMD_SIMD_ABI_H
