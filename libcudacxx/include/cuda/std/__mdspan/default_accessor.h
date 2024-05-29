// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_DEFAULT_ACCESSOR_HPP
#define _LIBCUDACXX___MDSPAN_DEFAULT_ACCESSOR_HPP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__type_traits/is_abstract.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/cstddef>

#if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _ElementType>
struct default_accessor
{
  static_assert(!_CCCL_TRAIT(is_array, _ElementType), "default_accessor: template argument may not be an array type");
  static_assert(!_CCCL_TRAIT(is_abstract, _ElementType),
                "default_accessor: template argument may not be an abstract class");

  using offset_policy    = default_accessor;
  using element_type     = _ElementType;
  using reference        = _ElementType&;
  using data_handle_type = _ElementType*;

  constexpr default_accessor() noexcept = default;

  _LIBCUDACXX_TEMPLATE(class _OtherElementType)
  _LIBCUDACXX_REQUIRES(_CCCL_TRAIT(is_convertible, _OtherElementType (*)[], element_type (*)[]))
  _LIBCUDACXX_INLINE_VISIBILITY constexpr default_accessor(default_accessor<_OtherElementType>) noexcept {}

  _LIBCUDACXX_INLINE_VISIBILITY constexpr reference access(data_handle_type __p, size_t __i) const noexcept
  {
    return __p[__i];
  }
  _LIBCUDACXX_INLINE_VISIBILITY constexpr data_handle_type offset(data_handle_type __p, size_t __i) const noexcept
  {
    return __p + __i;
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_STD_VER >= 2014

#endif // _LIBCUDACXX___MDSPAN_DEFAULT_ACCESSOR_H
