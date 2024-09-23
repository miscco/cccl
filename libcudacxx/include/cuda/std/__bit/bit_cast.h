//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_BIT_CAST_H
#define _CUDA_STD___BIT_BIT_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/is_trivially_default_constructible.h>
#include <cuda/std/__type_traits/remove_extent.h>
#include <cuda/std/cstring>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_BIT_CAST)
#  define _CCCL_CONSTEXPR_BIT_CAST       constexpr
#  define _CCCL_HAS_CONSTEXPR_BIT_CAST() 1
#else // ^^^ _CCCL_BUILTIN_BIT_CAST ^^^ / vvv !_CCCL_BUILTIN_BIT_CAST vvv
#  define _CCCL_CONSTEXPR_BIT_CAST
#  define _CCCL_HAS_CONSTEXPR_BIT_CAST() 0
#  if _CCCL_COMPILER(GCC, >=, 8)
// GCC starting with GCC8 warns about our extended floating point types having protected data members
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wclass-memaccess")
#  endif // _CCCL_COMPILER(GCC, >=, 8)
#endif // !_CCCL_BUILTIN_BIT_CAST

// Also allow arrays / tuples
template <class _Tp>
struct __is_bit_castable
{
  static constexpr bool value = is_trivially_copyable_v<_Tp> || __is_extended_floating_point_v<_Tp>;
};

template <class _Tp, size_t _Np>
struct __is_bit_castable<_Tp[_Np]>
{
  static constexpr bool value = __is_bit_castable<remove_extent_t<_Tp>>::value;
};

template <class _Tp, size_t _Np>
struct __is_bit_castable<array<_Tp, _Np>>
{
  static constexpr bool value = __is_bit_castable<_Tp>::value;
};

template <class _Tp, class _Up>
struct __is_bit_castable<pair<_Tp, _Up>>
{
  static constexpr bool value = __is_bit_castable<_Tp>::value && __is_bit_castable<_Up>::value;
};

template <class... _Tp>
struct __is_bit_castable<tuple<_Tp...>>
{
  static constexpr bool value = __all<__is_bit_castable<_Tp>::value...>::value;
};

template <class _Tp>
inline constexpr bool __is_bit_castable_v = is_trivially_copyable_v<_Tp> || __is_extended_floating_point_v<_Tp>;

template <class _Tp, size_t _Np>
inline constexpr bool __is_bit_castable_v<_Tp[_Np]> = __is_bit_castable_v<remove_extent_t<_Tp>>;

template <class _Tp, size_t _Np>
inline constexpr bool __is_bit_castable_v<array<_Tp, _Np>> = __is_bit_castable_v<_Tp>;

template <class _Tp, class _Up>
inline constexpr bool __is_bit_castable_v<pair<_Tp, _Up>> = __is_bit_castable_v<_Tp> && __is_bit_castable_v<_Up>;

template <class... _Tp>
inline constexpr bool __is_bit_castable_v<tuple<_Tp...>> = __all<__is_bit_castable_v<_Tp>...>::value;

template <class _To,
          class _From,
          enable_if_t<(sizeof(_To) == sizeof(_From)), int>        = 0,
          enable_if_t<_CCCL_TRAIT(__is_bit_castable, _To), int>   = 0,
          enable_if_t<_CCCL_TRAIT(__is_bit_castable, _From), int> = 0>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_BIT_CAST _To bit_cast(const _From& __from) noexcept
{
#if defined(_CCCL_BUILTIN_BIT_CAST)
  return _CCCL_BUILTIN_BIT_CAST(_To, __from);
#else // ^^^ _CCCL_BUILTIN_BIT_CAST ^^^ / vvv !_CCCL_BUILTIN_BIT_CAST vvv
  static_assert(is_trivially_default_constructible_v<_To>,
                "The compiler does not support __builtin_bit_cast, so bit_cast additionally requires the destination "
                "type to be trivially constructible");
  _To __temp;
  ::cuda::std::memcpy(&__temp, &__from, sizeof(_To));
  return __temp;
#endif // !_CCCL_BUILTIN_BIT_CAST
}

#if !defined(_CCCL_BUILTIN_BIT_CAST)
#  if _CCCL_COMPILER(GCC, >=, 8)
_CCCL_DIAG_POP
#  endif // _CCCL_COMPILER(GCC, >=, 8)
#endif // !_CCCL_BUILTIN_BIT_CAST

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_BIT_CAST_H
