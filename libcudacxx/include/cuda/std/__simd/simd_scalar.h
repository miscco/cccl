//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___SIMD_SIMD_SCALAR_H
#define _LIBCUDACXX___SIMD_SIMD_SCALAR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__simd/simd.h>
#include <cuda/std/__simd/simd_abi.h>
#include <cuda/std/__simd/simd_mask.h>
#include <cuda/std/__simd/simd_reference.h>
#include <cuda/std/__simd/simd_storage.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>

_LIBCUDACXX_BEGIN_NAMESPACE_SIMD_ABI

// [simd.class]
template <class _Tp>
class simd<_Tp, __simd_abi<_StorageKind::_Scalar, 1>>
{
private:
  __simd_storage<_Tp, __simd_abi<_StorageKind::_Scalar, 1>> __s_;

public:
  using value_type = _Tp;
  using reference  = __simd_reference<_Tp, __simd_abi<_StorageKind::_Scalar, 1>>;
  using mask_type  = simd_mask<_Tp, __simd_abi<_StorageKind::_Scalar, 1>>;
  using abi_type   = __simd_abi<_StorageKind::_Scalar, 1>;

  simd()                       = default;
  simd(const simd&)            = default;
  simd& operator=(const simd&) = default;

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t size() noexcept
  {
    return 1;
  }

  // implicit broadcast constructor
  template <class _Up, enable_if_t<__can_broadcast<_Up>, int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd(_Up&& __rv) noexcept
      : __s_(_CUDA_VSTD::forward(__v))
  {}

  // generator constructor
  template <
    class _Generator,
    enable_if_t<__can_broadcast<decltype(_CUDA_VSTD::declval<_Generator>()(integral_constant<size_t, 0>{}))>, int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 explicit simd(_Generator&& __g) noexcept
      : __s_(__g(integral_constant<size_t, 0>{}))
  {}

  // load constructor
  template <class _Up,
            class _Flags,
            enable_if_t<__vectorizable<_Up>, int>              = 0,
            enable_if_t<is_simd_flag_type<_Flags>::value, int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd(const _Up* __buffer, _Flags) noexcept
      : __s_(static_cast<_Tp>(__buffer[0]))
  {}

  // loads [simd.load]
  template <class _Up,
            class _Flags,
            enable_if_t<__vectorizable<_Up>, int>              = 0,
            enable_if_t<is_simd_flag_type<_Flags>::value, int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 void copy_from(const _Up* __buffer, _Flags) noexcept
  {
    __s_ = static_cast<_Tp>(__buffer[0]);
  }

  // stores [simd.store]
  template <class _Up, class _Flags enable_if_t<__vectorizable<_Up>() && is_simd_flag_type<_Flags>::value, int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 void copy_to(_Up* __buffer, _Flags) const noexcept
  {
    __buffer[0] = static_cast<_Up>(__s_.__storage);
  }

  // scalar access [simd.subscr]
  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference
  operator[](const size_t __index) noexcept
  {
    return reference(&__s_, __index_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr value_type operator[](const size_t __index) const noexcept
  {
    return __s_.__storage;
  }

  // unary operators [simd.unary]
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd& operator++() noexcept
  {
    ++__s_.__storage;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd operator++(int) noexcept
  {
    simd __temp = *this;
    ++__s_.__storage;
    return __temp;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd& operator--() noexcept
  {
    --__s_.__storage;
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd operator--(int) noexcept
  {
    simd __temp = *this;
    --__s_.__storage;
    return __temp;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 mask_type operator!() const noexcept
  {
    return mask_type{!__s_.__storage};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd operator~() const noexcept
  {
    return simd{static_cast<_Tp>(~__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd operator+() const noexcept
  {
    return simd{static_cast<_Tp>(+__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd operator-() const noexcept
  {
    return simd{static_cast<_Tp>(-__s_.__storage)};
  }

  // binary operators [simd.binary]
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator+(const simd& __lhs, const simd& __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage + __rhs.__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator-(const simd& __lhs, const simd& __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage - __rhs.__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator*(const simd& __lhs, const simd& __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage * __rhs.__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator/(const simd& __lhs, const simd& __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage / __rhs.__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator%(const simd& __lhs, const simd& __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage % __rhs.__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator&(const simd& __lhs, const simd& __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage & __rhs.__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator|(const simd& __lhs, const simd& __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage | __rhs.__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator^(const simd& __lhs, const simd& __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage ^ __rhs.__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator<<(const simd& __lhs, const simd& __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage << __rhs.__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator>>(const simd& __lhs, const simd& __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage >> __rhs.__s_.__storage)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator<<(const simd& __lhs, int __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage << __rhs)};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator>>(const simd& __lhs, int __rhs) noexcept
  {
    return simd{static_cast<_Tp>(__lhs.__s_.__storage >> __rhs)};
  }

  // compound assignment [simd.cassign]
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator+=(simd& __lhs, const simd& __rhs) noexcept
  {
    __lhs.__s_.__storage += __rhs.__s_.__storage;
    return __lhs;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator-=(simd& __lhs, const simd& __rhs) noexcept
  {
    __lhs.__s_.__storage -= __rhs.__s_.__storage;
    return __lhs;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator*=(simd& __lhs, const simd& __rhs) noexcept
  {
    __lhs.__s_.__storage *= __rhs.__s_.__storage;
    return __lhs;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator/=(simd& __lhs, const simd& __rhs) noexcept
  {
    __lhs.__s_.__storage /= __rhs.__s_.__storage;
    return __lhs;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator%=(simd& __lhs, const simd& __rhs) noexcept
  {
    __lhs.__s_.__storage %= __rhs.__s_.__storage;
    return __lhs;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator&=(simd& __lhs, const simd& __rhs) noexcept
  {
    __lhs.__s_.__storage &= __rhs.__s_.__storage;
    return __lhs;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator|=(simd& __lhs, const simd& __rhs) noexcept
  {
    __lhs.__s_.__storage |= __rhs.__s_.__storage;
    return __lhs;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator^=(simd& __lhs, const simd& __rhs) noexcept
  {
    __lhs.__s_.__storage ^= __rhs.__s_.__storage;
    return __lhs;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator<<=(simd& __lhs, const simd& __rhs) noexcept
  {
    __lhs.__s_.__storage <<= __rhs.__s_.__storage;
    return __lhs;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator>>=(simd& __lhs, const simd& __rhs) noexcept
  {
    __lhs.__s_.__storage >>= __rhs.__s_.__storage;
    return __lhs;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator<<=(simd& __lhs, int __rhs) noexcept
  {
    __lhs.__s_.__storage <<= __rhs;
    return __lhs;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator>>=(simd& __lhs, int __rhs) noexcept
  {
    __lhs.__s_.__storage >>= __rhs;
    return __lhs;
  }

  // compares [simd.comparison]
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type
  operator==(const simd& __lhs, const simd& __rhs) noexcept
  {
    return mask_type{__lhs.__s_.__storage == __rhs.__s_.__storage};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type
  operator!=(const simd& __lhs, const simd& __rhs) noexcept
  {
    return mask_type{__lhs.__s_.__storage != __rhs.__s_.__storage};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type
  operator>=(const simd& __lhs, const simd& __rhs) noexcept
  {
    return mask_type{__lhs.__s_.__storage >= __rhs.__s_.__storage};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type
  operator<=(const simd& __lhs, const simd& __rhs) noexcept
  {
    return mask_type{__lhs.__s_.__storage <= __rhs.__s_.__storage};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type
  operator>(const simd& __lhs, const simd& __rhs) noexcept
  {
    return mask_type{__lhs.__s_.__storage > __rhs.__s_.__storage};
  }
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type
  operator<(const simd& __lhs, const simd& __rhs) noexcept
  {
    return mask_type{__lhs.__s_.__storage < __rhs.__s_.__storage};
  }
};

_LIBCUDACXX_END_NAMESPACE_SIMD_ABI

#endif // _LIBCUDACXX___SIMD_SIMD_SCALAR_H
