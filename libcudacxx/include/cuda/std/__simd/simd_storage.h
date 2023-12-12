//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___SIMD_SIMD_STORAGE_H
#define _LIBCUDACXX___SIMD_SIMD_STORAGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__simd/simd_abi.h>

_LIBCUDACXX_BEGIN_NAMESPACE_SIMD_ABI

template <class _Tp, class _Abi>
struct __simd_storage
{};

template <class _Tp>
struct __simd_storage<_Tp, __simd_abi<_StorageKind::_Scalar, 1>>
{
  _Tp __storage_;

  template <class, class>
  friend struct simd;

  template <class, class>
  friend struct simd_mask;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __get(const size_t __index) const noexcept
  {
    return (&__storage_)[__index];
  };

  _LIBCUDACXX_HIDE_FROM_ABI constexpr void __set(const size_t __index, _Tp __val) noexcept
  {
    (&__storage_)[__index] = __val;
  }
};

template <class _Tp, int __num_element>
struct __simd_storage<_Tp, __simd_abi<_StorageKind::_Array, __num_element>>
{
  _Tp __storage_[__num_element];

  template <class, class>
  friend struct simd;

  template <class, class>
  friend struct simd_mask;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __get(const size_t __index) const noexcept
  {
    return __storage_[__index];
  };

  _LIBCUDACXX_HIDE_FROM_ABI constexpr void __set(const size_t __index, const _Tp __val) noexcept
  {
    __storage_[__index] = __val;
  }
};

#ifndef _LIBCUDACXX_HAS_NO_VECTOR_EXTENSION
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr size_t __floor_pow_of_2(size_t __val) noexcept
{
  return ((__val - 1) & __val) == 0 ? __val : __floor_pow_of_2((__val - 1) & __val);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr size_t __ceil_pow_of_2(size_t __val) noexcept
{
  return __val == 1 ? 1 : __floor_pow_of_2(__val - 1) << 1;
}

template <class _Tp, size_t __bytes>
struct __vec_ext_traits
{
#  if !defined(_LIBCUDACXX_COMPILER_CLANG)
  typedef _Tp type __attribute__((vector_size(__ceil_pow_of_2(__bytes))));
#  endif
};

#  if defined(_LIBCUDACXX_COMPILER_CLANG)
#    define _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, _NUM_ELEMENT)                        \
      template <>                                                                      \
      struct __vec_ext_traits<_TYPE, sizeof(_TYPE) * _NUM_ELEMENT>                     \
      {                                                                                \
        using type = _TYPE __attribute__((vector_size(sizeof(_TYPE) * _NUM_ELEMENT))); \
      }

#    define _LIBCUDACXX_SPECIALIZE_VEC_EXT_32(_TYPE) \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 1);      \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 2);      \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 3);      \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 4);      \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 5);      \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 6);      \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 7);      \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 8);      \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 9);      \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 10);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 11);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 12);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 13);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 14);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 15);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 16);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 17);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 18);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 19);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 20);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 21);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 22);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 23);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 24);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 25);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 26);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 27);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 28);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 29);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 30);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 31);     \
      _LIBCUDACXX_SPECIALIZE_VEC_EXT(_TYPE, 32);

_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(char);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(char16_t);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(char32_t);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(wchar_t);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(signed char);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(signed short);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(signed int);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(signed long);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(signed long long);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(unsigned char);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(unsigned short);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(unsigned int);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(unsigned long);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(unsigned long long);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(float);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(double);
_LIBCUDACXX_SPECIALIZE_VEC_EXT_32(long double);

#    undef _LIBCUDACXX_SPECIALIZE_VEC_EXT_32
#    undef _LIBCUDACXX_SPECIALIZE_VEC_EXT
#  endif

template <class _Tp, int __num_element>
struct __simd_storage<_Tp, __simd_abi<_StorageKind::_VecExt, __num_element>>
{
  using _StorageType = typename __vec_ext_traits<_Tp, sizeof(_Tp) * __num_element>::type;

  _StorageType __storage_;

  template <class, class>
  friend struct simd;

  template <class, class>
  friend struct simd_mask;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __get(const size_t __index) const noexcept
  {
    return __storage_[__index];
  };

  _LIBCUDACXX_HIDE_FROM_ABI constexpr void __set(const size_t __index, const _Tp __val) noexcept
  {
    __storage_[__index] = __val;
  }
};

#endif // _LIBCUDACXX_HAS_NO_VECTOR_EXTENSION

_LIBCUDACXX_END_NAMESPACE_SIMD_ABI

#endif // _LIBCUDACXX___SIMD_SIMD_STORAGE_H
