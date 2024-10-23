//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PSTL_FOR_EACH_N_H
#define _LIBCUDACXX___ALGORITHM_PSTL_FOR_EACH_N_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/for_each.h>
#include <cuda/std/__algorithm/for_each_n.h>
#include <cuda/std/__algorithm/pstl_backend.h>
#include <cuda/std/__algorithm/pstl_backends/optional.h>
#include <cuda/std/__algorithm/pstl_for_each.h>
#include <cuda/std/__algorithm/pstl_frontend_dispatch.h>
#include <cuda/std/__execution/policy.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__new/bad_alloc.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_execution_policy.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/empty.h>
#include <cuda/std/__utility/move.h>

#if !defined(_LIBCUDACXX_HAS_NO_INCOMPLETE_PSTL) && _CCCL_STD_VER >= 2017

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class>
void __pstl_for_each_n(); // declaration needed for the frontend dispatch below

template <class _RawPolicy>
struct __pstl_specialized_for_each_n
{
  template <class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Args&&... __args) noexcept
    -> decltype(_CUDA_VSTD::for_each_n<_RawPolicy>(__select_backend_t<_RawPolicy>{},
                                                   _CUDA_VSTD::forward<decltype(__args)>(__args)...))
  {
    return _CUDA_VSTD::for_each_n<_RawPolicy>(
      __select_backend_t<_RawPolicy>{}, _CUDA_VSTD::forward<decltype(__args)>(__args)...);
  }
};

template <class _RawPolicy>
struct __pstl_generic_for_each_n
{
  template <class _ForwardIterator, class _Size, class _Function>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __pstl_optional<__empty>
  operator()(_ForwardIterator __g_first, _Size __g_size, _Function __g_func) noexcept
  {
    if constexpr (random_access_iterator<_ForwardIterator>)
    {
      _RawPolicy __policy{_CUDA_VEXEC::__disable_user_instantiations_tag{}};
      _CUDA_VSTD::for_each(__policy, _CUDA_VSTD::move(__g_first), __g_first + __g_size, _CUDA_VSTD::move(__g_func));
      return __pstl_optional<__empty>{__empty{}};
    }
    else
    {
      _CUDA_VSTD::for_each_n(_CUDA_VSTD::move(__g_first), __g_size, _CUDA_VSTD::move(__g_func));
      return __pstl_optional<__empty>{__empty{}};
    }
  }
};

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Size,
          class _Function,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI __pstl_optional<__empty>
__for_each_n(_ExecutionPolicy&& __policy, _ForwardIterator&& __first, _Size&& __size, _Function&& __func) noexcept
{
  return _CUDA_VSTD::__pstl_frontend_dispatch(
    __pstl_specialized_for_each_n<_RawPolicy>{},
    __pstl_generic_for_each_n<_RawPolicy>{},
    _CUDA_VSTD::move(__first),
    _CUDA_VSTD::move(__size),
    _CUDA_VSTD::move(__func));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Size,
          class _Function,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCUDACXX_HIDE_FROM_ABI void
for_each_n(_ExecutionPolicy&& __policy, _ForwardIterator __first, _Size __size, _Function __func)
{
  static_assert(__iterator_traits_detail::__cpp17_forward_iterator<_ForwardIterator>,
                "[alg.foreach] parallel for_each_n requires at least forward iterators");
  auto __res =
    _CUDA_VSTD::__for_each_n(__policy, _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__size), _CUDA_VSTD::move(__func));
  if (!__res)
  {
    _CUDA_VSTD::__throw_bad_alloc();
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // !defined(_LIBCUDACXX_HAS_NO_INCOMPLETE_PSTL) && _CCCL_STD_VER >= 2017

#endif // _LIBCUDACXX___ALGORITHM_PSTL_FOR_EACH_N_H
