//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_RANGES_FOR_EACH_N_H
#define _LIBCUDACXX___ALGORITHM_RANGES_FOR_EACH_N_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/in_fun_result.h>
#include <cuda/std/__algorithm/pstl_backend.h>
#include <cuda/std/__algorithm/pstl_backends/optional.h>
#include <cuda/std/__algorithm/pstl_frontend_dispatch.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/projected.h>
#include <cuda/std/__new/bad_alloc.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/dangling.h>
#include <cuda/std/__type_traits/is_execution_policy.h>
#include <cuda/std/__utility/move.h>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

template <class _Iter, class _Func>
using for_each_n_result = in_fun_result<_Iter, _Func>;

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__for_each_n)

struct __fn
{
  _LIBCUDACXX_TEMPLATE(class _Iter, class _Func, class _Proj = identity)
  _LIBCUDACXX_REQUIRES(input_iterator<_Iter> _LIBCUDACXX_AND indirectly_unary_invocable<_Func, projected<_Iter, _Proj>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr for_each_n_result<_Iter, _Func>
  operator()(_Iter __first, iter_difference_t<_Iter> __count, _Func __func, _Proj __proj = {}) const
  {
    while (__count-- > 0)
    {
      _CUDA_VSTD::invoke(__func, _CUDA_VSTD::invoke(__proj, *__first));
      ++__first;
    }
    return {_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__func)};
  }

  _LIBCUDACXX_TEMPLATE(class _ExecutionPolicy,
                       class _Iter,
                       class _Func,
                       class _Proj      = identity,
                       class _RawPolicy = __remove_cvref_t<_ExecutionPolicy>)
  _LIBCUDACXX_REQUIRES(is_execution_policy_v<_RawPolicy> _LIBCUDACXX_AND forward_iterator<_Iter> _LIBCUDACXX_AND
                         indirectly_unary_invocable<_Func, projected<_Iter, _Proj>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr for_each_n_result<_Iter, _Func> operator()(
    _ExecutionPolicy&& __policy, _Iter __first, iter_difference_t<_Iter> __count, _Func __func, _Proj __proj = {}) const
  {
    using _Backend = _CUDA_VSTD::__select_backend_t<_RawPolicy>;
    auto __res     = _CUDA_VSTD::__pstl_ranges_for_each_n<_RawPolicy>(
      _Backend{}, _CUDA_VSTD::move(__first), __count, _CUDA_VSTD::move(__func), _CUDA_VSTD::move(__proj));
    if (!__res)
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }
    return __res.__val_;
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto for_each_n = __for_each_n::__fn{};
} // namespace __cpo

template <class _Iter, class _Func, class _Proj>
_LIBCUDACXX_HIDE_FROM_ABI constexpr in_fun_result<_Iter, _Func>
__ranges_for_each_n_indirection(_Iter __first, iter_difference_t<_Iter> __count, _Func __func, _Proj __proj)
{
  return for_each_n(_CUDA_VSTD::move(__first), __count, _CUDA_VSTD::move(__func), _CUDA_VSTD::move(__proj));
}

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

#endif // _LIBCUDACXX___ALGORITHM_RANGES_FOR_EACH_N_H
