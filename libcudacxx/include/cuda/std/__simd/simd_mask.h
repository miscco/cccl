//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___SIMD_SIMD_MASK_H
#define _LIBCUDACXX___SIMD_SIMD_MASK_H

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

_LIBCUDACXX_BEGIN_NAMESPACE_SIMD_ABI

// [simd.mask.class]
template <class _Tp, class _Abi>
class simd_mask
{
public:
  using value_type = bool;
  using reference  = bool&;
  using simd_type  = simd<_Tp, _Abi>;
  using abi_type   = _Abi;

  static constexpr size_t size() noexcept;

  _CCCL_HIDE_FROM_ABI simd_mask() = default;

  // broadcast constructor
  _LIBCUDACXX_HIDE_FROM_ABI explicit simd_mask(value_type) noexcept;

  // implicit type conversion constructor
  template <class _Up>
  _LIBCUDACXX_HIDE_FROM_ABI simd_mask(const simd_mask<_Up, simd_abi::fixed_size<size()>>&) noexcept;

  // load constructor
  template <class _Flags>
  _LIBCUDACXX_HIDE_FROM_ABI simd_mask(const value_type*, _Flags);

  // loads [simd.mask.copy]
  template <class _Flags>
  _LIBCUDACXX_HIDE_FROM_ABI void copy_from(const value_type*, _Flags);
  template <class _Flags>
  _LIBCUDACXX_HIDE_FROM_ABI void copy_to(value_type*, _Flags) const;

  // scalar access [simd.mask.subscr]
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI reference operator[](size_t);
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI value_type operator[](size_t) const;

  // unary operators [simd.mask.unary]
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI simd_mask operator!() const noexcept;

  // simd_mask binary operators [simd.mask.binary]
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI simd_mask operator&&(const simd_mask&, const simd_mask&) noexcept;
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI simd_mask operator||(const simd_mask&, const simd_mask&) noexcept;
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI simd_mask operator&(const simd_mask&, const simd_mask&) noexcept;
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI simd_mask operator|(const simd_mask&, const simd_mask&) noexcept;
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI simd_mask operator^(const simd_mask&, const simd_mask&) noexcept;

  // simd_mask compound assignment [simd.mask.cassign]
  _LIBCUDACXX_HIDE_FROM_ABI friend simd_mask& operator&=(simd_mask&, const simd_mask&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI friend simd_mask& operator|=(simd_mask&, const simd_mask&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI friend simd_mask& operator^=(simd_mask&, const simd_mask&) noexcept;

  // simd_mask compares [simd.mask.comparison]
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI simd_mask operator==(const simd_mask&, const simd_mask&) noexcept;
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI simd_mask operator!=(const simd_mask&, const simd_mask&) noexcept;
};

// traits [simd.traits]
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_simd_mask_v = false;

template <class _Tp, class _Abi>
_CCCL_INLINE_VAR constexpr bool is_simd_mask_v<simd_mask<_Tp, _Abi>> = true;

template <class _Tp>
struct is_simd_mask : integral_constant<bool, is_simd_mask_v<_Tp>>
{};

_LIBCUDACXX_END_NAMESPACE_SIMD_ABI

#endif // _LIBCUDACXX___SIMD_SIMD_MASK_H
