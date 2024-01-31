//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_RANGES_FOR_EACH_N_H
#define _LIBCUDACXX___ALGORITHM_RANGES_FOR_EACH_N_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__algorithm/in_fun_result.h"
#include "../__functional/identity.h"
#include "../__functional/invoke.h"
#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iterator_traits.h"
#include "../__iterator/projected.h"
#include "../__ranges/concepts.h"
#include "../__utility/move.h"

#if _CCCL_STD_VER >= 2017

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

template <class _Iter, class _Func>
using for_each_n_result = in_fun_result<_Iter, _Func>;

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__for_each_n)

struct __fn
{
  _LIBCUDACXX_TEMPLATE(class _Iter, class _Func, class _Proj = identity)
  _LIBCUDACXX_REQUIRES(input_iterator<_Iter> _LIBCUDACXX_AND indirectly_unary_invocable<_Func, projected<_Iter, _Proj>>)
  _LIBCUDACXX_INLINE_VISIBILITY constexpr for_each_n_result<_Iter, _Func>
  operator()(_Iter __first, iter_difference_t<_Iter> __count, _Func __func, _Proj __proj = {}) const
  {
    while (__count-- > 0)
    {
      _CUDA_VSTD::invoke(__func, _CUDA_VSTD::invoke(__proj, *__first));
      ++__first;
    }
    return {_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__func)};
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_LIBCUDACXX_CPO_ACCESSIBILITY auto for_each_n = __for_each_n::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _CCCL_STD_VER >= 2017

#endif // _LIBCUDACXX___ALGORITHM_RANGES_FOR_EACH_N_H
