// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_INSERT_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_INSERT_ITERATOR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/iterator_traits.h"
#include "../__iterator/iterator.h"
#include "../__memory/addressof.h"
#include "../__utility/move.h"
#include "../cstddef"

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Container>
using __insert_iterator_iter_t = typename _Container::iterator;

_LIBCUDACXX_SUPPRESS_DEPRECATED_PUSH
template <class _Container>
class _LIBCUDACXX_TEMPLATE_VIS insert_iterator
#if _LIBCUDACXX_STD_VER <= 14 || !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<output_iterator_tag, void, void, void, void>
#endif
{
_LIBCUDACXX_SUPPRESS_DEPRECATED_POP
protected:
    _Container* container;
    __insert_iterator_iter_t<_Container> iter;
public:
    typedef output_iterator_tag iterator_category;
    typedef void value_type;
#if _LIBCUDACXX_STD_VER > 17
    typedef ptrdiff_t difference_type;
#else
    typedef void difference_type;
#endif
    typedef void pointer;
    typedef void reference;
    typedef _Container container_type;

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 insert_iterator(_Container& __x, __insert_iterator_iter_t<_Container> __i)
        : container(_CUDA_VSTD::addressof(__x)), iter(__i) {}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 insert_iterator& operator=(const typename _Container::value_type& __value)
        {iter = container->insert(iter, __value); ++iter; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 insert_iterator& operator=(typename _Container::value_type&& __value)
        {iter = container->insert(iter, _CUDA_VSTD::move(__value)); ++iter; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 insert_iterator& operator*()        {return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 insert_iterator& operator++()       {return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 insert_iterator& operator++(int)    {return *this;}
};

template <class _Container>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
insert_iterator<_Container>
inserter(_Container& __x, __insert_iterator_iter_t<_Container> __i)
{
    return insert_iterator<_Container>(__x, __i);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_INSERT_ITERATOR_H
