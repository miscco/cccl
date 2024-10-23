//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_FOR_EACH_H
#define _LIBCUDACXX___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_FOR_EACH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/for_each.h>
#include <cuda/std/__algorithm/in_fun_result.h>
#include <cuda/std/__algorithm/pstl_backends/cpu_backends/backend.h>
#include <cuda/std/__algorithm/pstl_backends/optional.h>
#include <cuda/std/__cccl/simd.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__type_traits/is_execution_policy.h>
#include <cuda/std/__utility/empty.h>

#if !defined(_LIBCUDACXX_HAS_NO_INCOMPLETE_PSTL) && _CCCL_STD_VER >= 2017

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Iterator, class _DifferenceType, class _Function>
_LIBCUDACXX_HIDE_FROM_ABI _Iterator __simd_walk(_Iterator __first, _DifferenceType __n, _Function __f) noexcept
{
  _CCCL_PRAGMA_SIMD
  for (_DifferenceType __i = 0; __i < __n; ++__i)
  {
    __f(__first[__i]);
  }

  return __first + __n;
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Functor>
_LIBCUDACXX_HIDE_FROM_ABI __pstl_optional<__empty>
__pstl_for_each(__cpu_backend_tag, _ForwardIterator __first, _ForwardIterator __last, _Functor __func)
{
  if constexpr (false /* __is_parallel_execution_policy_v<_ExecutionPolicy> */
                && random_access_iterator<_ForwardIterator>)
  {
    /* return _CUDA_VSTD::__par_backend::__parallel_for(
      __first, __last, [__func](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
        [[maybe_unused]] auto __res = _CUDA_VSTD::__pstl_for_each<__remove_parallel_policy_t<_ExecutionPolicy>>(
          __cpu_backend_tag{}, __brick_first, __brick_last, __func);
        _CCCL_VERIFY(__res, "unseq/seq should never try to allocate!");
      });
    */
    return __pstl_optional<__empty>{__empty{}};
  }
  else if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> && random_access_iterator<_ForwardIterator>)
  {
    _CUDA_VSTD::__simd_walk(__first, __last - __first, __func);
    return __pstl_optional<__empty>{__empty{}};
  }
  else
  {
    _CUDA_VSTD::for_each(__first, __last, __func);
    return __pstl_optional<__empty>{__empty{}};
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#  if !defined(_CCCL_COMPILER_MSVC_2017)

// We need to go through `ranges::for_each_n` as the CPO for the initial parallel algorithm call, so we cannot directly
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

template <class _Iter, class _Sent, class _Func, class _Proj>
_LIBCUDACXX_HIDE_FROM_ABI constexpr in_fun_result<_Iter, _Func>
  __ranges_for_each_indirection(_Iter, _Sent, _Func, _Proj);

template <class _Iter, class _Func, class _Proj>
_LIBCUDACXX_HIDE_FROM_ABI constexpr in_fun_result<_Iter, _Func>
  __ranges_for_each_n_indirection(_Iter, iter_difference_t<_Iter>, _Func, _Proj);

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Iter, class _Size, class _Func, class _Proj>
_LIBCUDACXX_HIDE_FROM_ABI __pstl_optional<_CUDA_VRANGES::in_fun_result<_Iter, _Func>>
__simd_walk(_Iter __first, _Size __n, _Func& __func, _Proj& __proj) noexcept
{
  _CCCL_PRAGMA_SIMD
  for (_Size __i = 0; __i < __n; ++__i)
  {
    _CUDA_VSTD::invoke(__func, _CUDA_VSTD::invoke(__proj, __first[__i]));
  }
  return __pstl_optional<_CUDA_VRANGES::in_fun_result<_Iter, _Func>>{
    {_CUDA_VSTD::move(__first) + __n, _CUDA_VSTD::move(__func)}};
}

template <class _ExecutionPolicy, class _Iter, class _Sent, class _Func, class _Proj>
_LIBCUDACXX_HIDE_FROM_ABI __pstl_optional<_CUDA_VRANGES::in_fun_result<_Iter, _Func>>
__pstl_ranges_for_each(__cpu_backend_tag, _Iter __first, _Sent __last, _Func __func, _Proj __proj)
{
  if constexpr (false /* __is_parallel_execution_policy_v<_ExecutionPolicy> */
                && random_access_iterator<_Iter>)
  {
    /* return _CUDA_VSTD::__par_backend::__parallel_for(
      __first, __last, [__func](_Iter __brick_first, _Iter __brick_last) {
        [[maybe_unused]] auto __res = _CUDA_VSTD::__pstl_for_each<__remove_parallel_policy_t<_ExecutionPolicy>>(
          __cpu_backend_tag{}, __brick_first, __brick_last, __func);
        _LIBCUDACXX_ASSERT_INTERNAL(__res, "unseq/seq should never try to allocate!");
      });
    */
    return __pstl_optional<_CUDA_VRANGES::in_fun_result<_Iter, _Func>>{
      _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__func)};
  }
  else if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> && random_access_iterator<_Iter>
                     && sized_sentinel_for<_Sent, _Iter>)
  {
    const auto __n = _CUDA_VRANGES::distance(__first, __last);
    return __pstl_optional<_CUDA_VRANGES::in_fun_result<_Iter, _Func>>{
      _CUDA_VSTD::__simd_walk(_CUDA_VSTD::move(__first), __n, __func, __proj)};
  }
  else
  {
    return __pstl_optional<_CUDA_VRANGES::in_fun_result<_Iter, _Func>>{_CUDA_VRANGES::__ranges_for_each_indirection(
      _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), _CUDA_VSTD::move(__func), _CUDA_VSTD::move(__proj))};
  }
}

template <class _ExecutionPolicy, class _Iter, class _Size, class _Func, class _Proj>
_LIBCUDACXX_HIDE_FROM_ABI __pstl_optional<_CUDA_VRANGES::in_fun_result<_Iter, _Func>>
__pstl_ranges_for_each_n(__cpu_backend_tag, _Iter __first, _Size __n, _Func __func, _Proj __proj)
{
  if constexpr (false /* __is_parallel_execution_policy_v<_ExecutionPolicy> */
                && random_access_iterator<_Iter>)
  {
    /* return _CUDA_VSTD::__par_backend::__parallel_for(
      __first, __last, [__func](_Iter __brick_first, _Iter __brick_last) {
        [[maybe_unused]] auto __res = _CUDA_VSTD::__pstl_for_each<__remove_parallel_policy_t<_ExecutionPolicy>>(
          __cpu_backend_tag{}, __brick_first, __brick_last, __func);
        _LIBCUDACXX_ASSERT_INTERNAL(__res, "unseq/seq should never try to allocate!");
      });
    */
    return __pstl_optional<_CUDA_VRANGES::in_fun_result<_Iter, _Func>>{
      _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__func)};
  }
  else if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> && random_access_iterator<_Iter>)
  {
    return __pstl_optional<_CUDA_VRANGES::in_fun_result<_Iter, _Func>>{
      _CUDA_VSTD::__simd_walk(_CUDA_VSTD::move(__first), __n, __func, __proj)};
  }
  else
  {
    return __pstl_optional<_CUDA_VRANGES::in_fun_result<_Iter, _Func>>{_CUDA_VRANGES::__ranges_for_each_n_indirection(
      _CUDA_VSTD::move(__first), __n, _CUDA_VSTD::move(__func), _CUDA_VSTD::move(__proj))};
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#  endif // !_CCCL_COMPILER_MSVC_2017

#endif // !defined(_LIBCUDACXX_HAS_NO_INCOMPLETE_PSTL) && _CCCL_STD_VER >= 2017

#endif // _LIBCUDACXX___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_FOR_EACH_H
