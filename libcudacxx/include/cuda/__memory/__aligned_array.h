//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY___ALIGNED_ARRAY_H
#define _CUDA___MEMORY___ALIGNED_ARRAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// 32-byte accesses are supported since PTX ISA 9.0, and only for global memory on sm_100 and newer
inline constexpr size_t __max_memory_access_width = (__cccl_ptx_isa >= 900ULL) ? 32ull : 16ull;

//! `_Count` objects of type `_Tp`, aligned to `_Alignment`
//!
//! The compiler loads a copy of the array with accesses of up to `_Alignment` bytes. The alignment must divide the size
//! of the array, so that an array of aligned arrays has the layout of the objects it holds.
template <class _Tp, size_t _Count, size_t _Alignment>
struct __aligned_array
{
  static_assert(_Count > 0, "An aligned array must hold at least one object");
  static_assert(alignof(_Tp) <= _Alignment, "An aligned array must be at least as aligned as its objects");
  static_assert(sizeof(_Tp) * _Count % _Alignment == 0,
                "The size of an aligned array must be a multiple of its alignment");

  alignas(_Alignment) _Tp __storage_[_Count];

  //! Returns the `__index`-th object of the array
  template <class _Index>
  [[nodiscard]] _CCCL_API constexpr _Tp operator[](const _Index __index) const noexcept
  {
    static_assert(::cuda::std::__cccl_is_integer_v<_Index>, "The index must be an integer");
    return __storage_[__index];
  }
};

#if _CCCL_HAS_INT128()
// The compiler does not merge the copies of 128-bit integers into accesses wider than 16 bytes, thus they are stored as
// 64-bit words
template <size_t _Count, size_t _Alignment>
struct __aligned_array<__int128_t, _Count, _Alignment>
{
  static_assert(_Count > 0, "An aligned array must hold at least one object");
  static_assert(alignof(__int128_t) <= _Alignment, "An aligned array must be at least as aligned as its objects");
  static_assert(sizeof(__int128_t) * _Count % _Alignment == 0,
                "The size of an aligned array must be a multiple of its alignment");

  alignas(_Alignment)::cuda::std::uint64_t __storage_[2 * _Count];

  //! Returns the `__index`-th object of the array
  template <class _Index>
  [[nodiscard]] _CCCL_API __int128_t operator[](const _Index __index) const noexcept
  {
    static_assert(::cuda::std::__cccl_is_integer_v<_Index>, "The index must be an integer");
    return ::cuda::std::launder(reinterpret_cast<const __int128_t*>(__storage_))[__index];
  }
};

template <size_t _Count, size_t _Alignment>
struct __aligned_array<__uint128_t, _Count, _Alignment>
{
  static_assert(_Count > 0, "An aligned array must hold at least one object");
  static_assert(alignof(__uint128_t) <= _Alignment, "An aligned array must be at least as aligned as its objects");
  static_assert(sizeof(__uint128_t) * _Count % _Alignment == 0,
                "The size of an aligned array must be a multiple of its alignment");

  alignas(_Alignment)::cuda::std::uint64_t __storage_[2 * _Count];

  //! Returns the `__index`-th object of the array
  template <class _Index>
  [[nodiscard]] _CCCL_API __uint128_t operator[](const _Index __index) const noexcept
  {
    static_assert(::cuda::std::__cccl_is_integer_v<_Index>, "The index must be an integer");
    return ::cuda::std::launder(reinterpret_cast<const __uint128_t*>(__storage_))[__index];
  }
};
#endif // _CCCL_HAS_INT128()

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY___ALIGNED_ARRAY_H
