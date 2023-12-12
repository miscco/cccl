//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___SIMD_SIMD_REFERENCE_H
#define _LIBCUDACXX___SIMD_SIMD_REFERENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__simd/simd_storage.h>

_LIBCUDACXX_BEGIN_NAMESPACE_SIMD_ABI

// [simd.reference]
template <class _Tp, class _Abi>
class __simd_reference
{
private:
  template <class, class>
  friend struct simd;

  template <class, class>
  friend struct simd_mask;

  __simd_storage<_Tp, _Abi>* __ptr_;
  size_t __index_;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __simd_reference(__simd_storage<_Tp, _Abi>* __ptr, const size_t __index) noexcept
      : __ptr_(__ptr)
      , __index_(__index)
  {}

  _CCCL_HIDE_FROM_ABI __simd_reference(const __simd_reference&) = default;

public:
  __simd_reference()                                   = delete;
  __simd_reference& operator=(const __simd_reference&) = delete;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 operator _Tp() const noexcept
  {
    return __ptr_->__get(__index_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator=(_Tp __value) && noexcept
  {
    __ptr_->__set(__index_, __value);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator++() && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) + 1);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Tp operator++(int) && noexcept
  {
    auto __val = __ptr_->__get(__index_);
    __ptr_->__set(__index_, __val + 1);
    return __val;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator--() && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) - 1);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Tp operator--(int) && noexcept
  {
    auto __val = __ptr_->__get(__index_);
    __ptr_->__set(__index_, __val - 1);
    return __val;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator+=(_Tp __value) && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) + __value);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator-=(_Tp __value) && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) - __value);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator*=(_Tp __value) && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) * __value);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator/=(_Tp __value) && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) / __value);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI __simd_reference operator%=(_Tp __value) && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) % __value);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator>>=(_Tp __value) && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) >> __value);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator<<=(_Tp __value) && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) << __value);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator&=(_Tp __value) && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) & __value);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator|=(_Tp __value) && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) | __value);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 __simd_reference operator^=(_Tp __value) && noexcept
  {
    __ptr_->__set(__index_, __ptr_->__get(__index_) ^ __value);
    return *this;
  }
};

_LIBCUDACXX_END_NAMESPACE_SIMD_ABI

#endif // _LIBCUDACXX___SIMD_SIMD_REFERENCE_H
