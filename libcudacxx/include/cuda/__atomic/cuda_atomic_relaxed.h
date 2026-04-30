// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the CUDA C++ Standard Library,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ATOMIC_CUDA_ATOMIC_RELAXED_H
#define _CUDA___ATOMIC_CUDA_ATOMIC_RELAXED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/cuda_ptx_derived.h>
#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/scopes.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CUDA_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA

template <::cuda::thread_scope _Scope>
[[nodiscard]] _CCCL_API constexpr auto __cccl_select_scope()
{
  if constexpr (_Scope == ::cuda::thread_scope_system)
  {
    return ::cuda::std::__thread_scope_system_tag{};
  }
  else if constexpr (_Scope == ::cuda::thread_scope_device)
  {
    return ::cuda::std::__thread_scope_device_tag{};
  }
  else if constexpr (_Scope == ::cuda::thread_scope_block)
  {
    return ::cuda::std::__thread_scope_block_tag{};
  }
  else if constexpr (_Scope == ::cuda::thread_scope_thread)
  {
    return ::cuda::std::__thread_scope_block_tag{};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<::cuda::std::integral_constant<::cuda::thread_scope, _Scope>>,
                  "Invalid thread scope");
    _CCCL_UNREACHABLE();
  }
}

//! Loads `*__ptr` atomically into `__dst` using `__nv_atomic_load` when available, otherwise uses ptx helper
template <::cuda::thread_scope _Scope, typename _Tp>
_CCCL_DEVICE_API void __cccl_cuda_atomic_load_relaxed(const _Tp* __ptr, _Tp& __dst)
{
  using __proxy_t              = typename ::cuda::std::__atomic_cuda_deduce_bitwise<_Tp>::__type;
  using __proxy_tag            = typename ::cuda::std::__atomic_cuda_deduce_bitwise<_Tp>::__tag;
  const __proxy_t* __ptr_proxy = reinterpret_cast<const __proxy_t*>(__ptr);
  __proxy_t* __dst_proxy       = reinterpret_cast<__proxy_t*>(&__dst);
  if (::cuda::std::__cuda_load_weak_if_local(__ptr_proxy, __dst_proxy, sizeof(__proxy_t)))
  {
    return;
  }

  auto __scope = __cccl_select_scope<_Scope>();
  ::cuda::std::__cuda_atomic_bind_load<__proxy_t, __proxy_tag, decltype(__scope), ::cuda::std::__atomic_cuda_mmio_disable>
    __bound_load{__ptr_proxy, __dst_proxy};
  ::cuda::std::__cuda_atomic_load_memory_order_dispatch(__bound_load, ::cuda::memory_order_relaxed, __scope);
}

//! Loads `*__ptr` atomically into `__dst` using `__nv_atomic_load` when available, otherwise uses ptx helper
template <::cuda::thread_scope _Scope, typename _Tp>
_CCCL_DEVICE_API void __cccl_cuda_atomic_store_relaxed(_Tp* __ptr, _Tp __val)
{
  using __proxy_t        = typename ::cuda::std::__atomic_cuda_deduce_bitwise<_Tp>::__type;
  using __proxy_tag      = typename ::cuda::std::__atomic_cuda_deduce_bitwise<_Tp>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __val_proxy = reinterpret_cast<__proxy_t*>(&__val);
  if (::cuda::std::__cuda_store_weak_if_local(__ptr_proxy, __val_proxy, sizeof(__proxy_t)))
  {
    return;
  }

  auto __scope = __cccl_select_scope<_Scope>();
  ::cuda::std::__cuda_atomic_bind_store<__proxy_t, __proxy_tag, decltype(__scope), ::cuda::std::__atomic_cuda_mmio_disable>
    __bound_store{__ptr_proxy, __val_proxy};
  ::cuda::std::__cuda_atomic_store_memory_order_dispatch(__bound_store, ::cuda::memory_order_relaxed, __scope);
}

//! Adds `__val` to `*__ptr` and returns the previous value. Uses `__nv_atomic_fetch_add` when available, PTX otherwise
template <::cuda::thread_scope _Scope, typename _Tp, typename _Up>
_CCCL_DEVICE_API _Tp __cccl_cuda_atomic_fetch_add_relaxed(_Tp* __ptr, _Up __op)
{
  constexpr auto __skip_v = ::cuda::std::__atomic_ptr_skip_t<_Tp>::__skip;
  __op                    = __op * __skip_v;
  using __proxy_t         = typename ::cuda::std::__atomic_cuda_deduce_arithmetic<_Tp>::__type;
  using __proxy_tag       = typename ::cuda::std::__atomic_cuda_deduce_arithmetic<_Tp>::__tag;
  _Tp __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (::cuda::std::__cuda_fetch_add_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }

  auto __scope = __cccl_select_scope<_Scope>();
  ::cuda::std::__cuda_atomic_bind_fetch_add<__proxy_t, __proxy_tag, decltype(__scope)> __bound_add{
    __ptr_proxy, __dst_proxy, __op_proxy};
  ::cuda::std::__cuda_atomic_fetch_memory_order_dispatch(__bound_add, ::cuda::memory_order_relaxed, __scope);
  return __dst;
}

_CCCL_END_NAMESPACE_CUDA

#endif // _CCCL_CUDA_COMPILATION()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ATOMIC_CUDA_ATOMIC_RELAXED_H
