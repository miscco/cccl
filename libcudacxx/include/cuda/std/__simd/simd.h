//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___SIMD_SIMD_H
#define _LIBCUDACXX___SIMD_SIMD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__simd/simd_abi.h>
#include <cuda/std/__simd/simd_mask.h>
#include <cuda/std/__simd/simd_reference.h>
#include <cuda/std/__simd/simd_size.h>
#include <cuda/std/__simd/simd_storage.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/tuple>

_LIBCUDACXX_BEGIN_NAMESPACE_SIMD_ABI

template <class _From, class _To, class = void>
_CCCL_INLINE_VAR constexpr bool __is_non_narrowing_arithmetic_convertible =
  __cccl_internal::__is_non_narrowing_convertible<_From, _To>::value && _CCCL_TRAIT(is_arithmetic, _From)
  && _CCCL_TRAIT(is_arithmetic, _To);

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __vectorizable =
  _CCCL_TRAIT(is_arithmetic, _Tp) && !_CCCL_TRAIT(is_const, _Tp) && !_CCCL_TRAIT(is_volatile, _Tp)
  && !_CCCL_TRAIT(is_same, _Tp, bool);

template <class _Tp, class _Up>
_CCCL_INLINE_VAR constexpr bool __can_broadcast =
  (_CCCL_TRAIT(is_arithmetic, _Up) && __is_non_narrowing_arithmetic_convertible<_Up, _Tp>)
  || (!_CCCL_TRAIT(is_arithmetic, _Up) && _CCCL_TRAIT(is_convertible, _Up, _Tp))
  || (_CCCL_TRAIT(is_same, remove_const_t<_Up>, int))
  || (_CCCL_TRAIT(is_same, remove_const_t<_Up>, unsigned int) && _CCCL_TRAIT(is_unsigned, _Tp));

// [simd.class]
template <class _Tp, class _Abi>
class simd
{
public:
  using value_type = _Tp;
  using reference  = __simd_reference<_Tp, _Abi>;
  using mask_type  = simd_mask<_Tp, _Abi>;
  using abi_type   = _Abi;

  _CCCL_HIDE_FROM_ABI simd()                       = default;
  _CCCL_HIDE_FROM_ABI simd(const simd&)            = default;
  _CCCL_HIDE_FROM_ABI simd& operator=(const simd&) = default;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t size() noexcept
  {
    return simd_size<_Tp, _Abi>::value;
  }

private:
  __simd_storage<_Tp, _Abi> __s_;

  template <class _Generator, size_t... __indicies>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr decltype(_CUDA_VSTD::forward_as_tuple(_CUDA_VSTD::declval<_Generator>()(
                                                        integral_constant<size_t, __indicies>())...),
                                                      bool())
  __can_generate(index_sequence<__indicies...>)
  {
    return !__variadic_sum<bool>(
      !__can_broadcast<decltype(_CUDA_VSTD::declval<_Generator>()(integral_constant<size_t, __indicies>()))>()...);
  }

  template <class _Generator>
  _LIBCUDACXX_HIDE_FROM_ABI static bool __can_generate(...)
  {
    return false;
  }

  template <class _Generator, size_t... __indicies>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd(_Generator&& __g, index_sequence<__indicies...>) noexcept
      : __s_({__g(integral_constant<size_t, __indicies>()...)})
  {}

public:
  // implicit type conversion constructor
  template <class _Up,
            enable_if_t<is_same<_Abi, simd_abi::fixed_size<simd_size<_Tp, _Abi>::value>>::value, int> = 0,
            enable_if_t<__is_non_narrowing_arithmetic_convertible<_Up, _Tp>, int>                     = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14
  simd(const simd<_Up, simd_abi::fixed_size<simd_size<_Tp, _Abi>::value>>& __v) noexcept
  {
    for (size_t __i = 0; __i < simd_size<_Tp, _Abi>::value; __i++)
    {
      __s_.__set(__i, static_cast<_Tp>(__v[__i]));
    }
  }

  // implicit broadcast constructor
  template <class _Up, enable_if_t<__can_broadcast<_Up, _Tp>, int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd(_Up&& __rv) noexcept
  {
    const auto __v = static_cast<_Tp>(__rv);
    for (size_t __i = 0; __i < simd_size<_Tp, _Abi>::value; __i++)
    {
      __s_.__set(__i, __v);
    }
  }

  // generator constructor
  template <
    class _Generator,
    enable_if_t<__can_generate<_Generator>(_CUDA_VSTD::make_index_sequence<simd_size<_Tp, _Abi>::value>()), int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 explicit simd(_Generator&& __g) noexcept
      : simd(_CUDA_VSTD::forward<_Generator>(__g), _CUDA_VSTD::make_index_sequence<simd_size<_Tp, _Abi>::value>())
  {}

  // load constructor
  template <class _Up,
            class _Flags,
            enable_if_t<__vectorizable<_Up>, int>              = 0,
            enable_if_t<is_simd_flag_type<_Flags>::value, int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd(const _Up* __buffer, _Flags) noexcept
  {
    for (size_t __i = 0; __i < simd_size<_Tp, _Abi>::value; __i++)
    {
      __s_.__set(__i, static_cast<_Tp>(__buffer[__i]));
    }
  }

  // loads [simd.load]
  template <class _Up,
            class _Flags,
            enable_if_t<__vectorizable<_Up>, int>              = 0,
            enable_if_t<is_simd_flag_type<_Flags>::value, int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 void copy_from(const _Up* __buffer, _Flags) noexcept
  {
    for (size_t __i = 0; __i < simd_size<_Tp, _Abi>::value; __i++)
    {
      __s_.__set(__i, static_cast<_Tp>(__buffer[__i]));
    }
  }

  // stores [simd.store]
  template <class _Up, class _Flags enable_if_t<__vectorizable<_Up>() && is_simd_flag_type<_Flags>::value, int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 void copy_to(_Up* __buffer, _Flags) const noexcept
  {
    for (size_t __i = 0; __i < simd_size<_Tp, _Abi>::value; __i++)
    {
      __buffer[__i] = static_cast<_Up>(__s_.__get(__i));
    }
  }

  // scalar access [simd.subscr]
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr reference
  operator[](const size_t __index) noexcept
  {
    return reference(&__s_, __index_);
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr value_type operator[](const size_t __index) const noexcept
  {
    return __s_.__get(__index_);
  }

  // unary operators [simd.unary]
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd& operator++() noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd operator++(int) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd& operator--() noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd operator--(int) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 mask_type operator!() const noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd operator~() const noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd operator+() const noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 simd operator-() const noexcept;

  // binary operators [simd.binary]
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator+(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator-(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator*(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator/(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator%(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator&(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator|(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator^(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator<<(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator>>(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator<<(const simd&, int) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd operator>>(const simd&, int) noexcept;

  // compound assignment [simd.cassign]
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator+=(simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator-=(simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator*=(simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator/=(simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator%=(simd&, const simd&) noexcept;

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator&=(simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator|=(simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator^=(simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator<<=(simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator>>=(simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator<<=(simd&, int) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend simd& operator>>=(simd&, int) noexcept;

  // compares [simd.comparison]
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type operator==(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type operator!=(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type operator>=(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type operator<=(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type operator>(const simd&, const simd&) noexcept;
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 friend mask_type operator<(const simd&, const simd&) noexcept;
};

// traits [simd.traits]
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_simd_v = false;

template <class _Tp, class _Abi>
_CCCL_INLINE_VAR constexpr bool is_simd_v<simd<_Tp, _Abi>> = true;

template <class _Tp>
struct is_simd : integral_constant<bool, is_simd_v<_Tp>>
{};

_LIBCUDACXX_END_NAMESPACE_SIMD_ABI

#endif // _LIBCUDACXX___SIMD_SIMD_H
