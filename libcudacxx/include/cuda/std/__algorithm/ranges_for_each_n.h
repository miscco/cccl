//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_RANGES_FOR_EACH_N_H
#define _CUDA_STD___ALGORITHM_RANGES_FOR_EACH_N_H

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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_RANGES

template <class _Iter, class _Func>
using for_each_n_result = in_fun_result<_Iter, _Func>;

_CCCL_BEGIN_NAMESPACE_CPO(__for_each_n)

struct __fn
{
  _CCCL_TEMPLATE(class _Iter, class _Func, class _Proj = identity)
  _CCCL_REQUIRES(input_iterator<_Iter> _CCCL_AND indirectly_unary_invocable<_Func, projected<_Iter, _Proj>>)
  _CCCL_API constexpr for_each_n_result<_Iter, _Func>
  operator()(_Iter __first, iter_difference_t<_Iter> __count, _Func __func, _Proj __proj = {}) const
  {
    while (__count-- > 0)
    {
      ::cuda::std::invoke(__func, ::cuda::std::invoke(__proj, *__first));
      ++__first;
    }
    return {::cuda::std::move(__first), ::cuda::std::move(__func)};
  }

  _CCCL_TEMPLATE(class _ExecutionPolicy,
                 class _Iter,
                 class _Func,
                 class _Proj      = identity,
                 class _RawPolicy = remove_cvref_t<_ExecutionPolicy>)
  _CCCL_REQUIRES(is_execution_policy_v<_RawPolicy> _CCCL_AND forward_iterator<_Iter> _CCCL_AND
                   indirectly_unary_invocable<_Func, projected<_Iter, _Proj>>)
  _CCCL_API constexpr for_each_n_result<_Iter, _Func> operator()(
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
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto for_each_n = __for_each_n::__fn{};
} // namespace __cpo

template <class _Iter, class _Func, class _Proj>
_CCCL_API constexpr in_fun_result<_Iter, _Func>
__ranges_for_each_n_indirection(_Iter __first, iter_difference_t<_Iter> __count, _Func __func, _Proj __proj)
{
  return for_each_n(_CUDA_VSTD::move(__first), __count, _CUDA_VSTD::move(__func), _CUDA_VSTD::move(__proj));
}

_CCCL_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_RANGES_FOR_EACH_N_H
