//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MIN_H
#define _LIBCUDACXX___ALGORITHM_MIN_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__algorithm/comp.h"
#include "../__algorithm/comp_ref_type.h"
#include "../__algorithm/min_element.h"
#include "../initializer_list"

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC
#ifndef __cuda_std__
#include <__pragma_push>
#endif // __cuda_std__

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, class _Compare>
_LIBCUDACXX_NODISCARD_EXT inline
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
const _Tp&
min(const _Tp& __a, const _Tp& __b, _Compare __comp)
{
    return __comp(__b, __a) ? __b : __a;
}

template <class _Tp>
_LIBCUDACXX_NODISCARD_EXT inline
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
const _Tp&
min(const _Tp& __a, const _Tp& __b)
{
    return _CUDA_VSTD::min(__a, __b, __less<_Tp>());
}

#ifndef _LIBCUDACXX_CXX03_LANG

template<class _Tp, class _Compare>
_LIBCUDACXX_NODISCARD_EXT inline
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
_Tp
min(initializer_list<_Tp> __t, _Compare __comp)
{
    return *_CUDA_VSTD::__min_element<__comp_ref_type<_Compare> >(__t.begin(), __t.end(), __comp);
}

template<class _Tp>
_LIBCUDACXX_NODISCARD_EXT inline
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
_Tp
min(initializer_list<_Tp> __t)
{
    return *_CUDA_VSTD::min_element(__t.begin(), __t.end(), __less<_Tp>());
}

#endif // _LIBCUDACXX_CXX03_LANG

_LIBCUDACXX_END_NAMESPACE_STD

#ifndef __cuda_std__
#include <__pragma_pop>
#endif // __cuda_std__

#endif // _LIBCUDACXX___ALGORITHM_MIN_H
