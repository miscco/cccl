// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NEW_THROW_BAD_ALLOC_H
#define _LIBCUDACXX___NEW_THROW_BAD_ALLOC_H

#ifndef __cuda_std__
#  include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../cstdlib"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_NORETURN _LIBCUDACXX_FUNC_VIS void __throw_bad_alloc(); // not in C++ spec

#ifdef __cuda_std__

_LIBCUDACXX_NORETURN _LIBCUDACXX_FUNC_VIS void __throw_bad_alloc()
{
#  if defined(__CUDA_ARCH__)
  __trap();
#  else // ^^^ __CUDA_ARCH__ ^^^ / vvv !__CUDA_ARCH__ vvv
  ::abort();
#  endif // !__CUDA_ARCH__
}

#endif // __cuda_std__

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NEW_THROW_BAD_ALLOC_H
