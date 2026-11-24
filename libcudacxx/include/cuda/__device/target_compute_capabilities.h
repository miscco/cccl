//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_TARGET_COMPUTE_CAPABILITIES_H
#define _CUDA___DEVICE_TARGET_COMPUTE_CAPABILITIES_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__type_traits/always_false.h> // IWYU pragma: keep
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <int... _Vs>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL auto __make_all_compute_capabilities() noexcept
{
  return ::cuda::std::array{compute_capability{_Vs}...};
}

[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL auto __all_compute_capabilities() noexcept
{
  return ::cuda::__make_all_compute_capabilities<_CCCL_KNOWN_CUDA_ARCH_LIST>();
}

#if _CCCL_CUDA_COMPILATION()
template <int... _Vs>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL auto __make_cc_list() noexcept
{
#  ifdef __CUDA_ARCH_LIST__
  return ::cuda::std::array{compute_capability{_Vs / 10}...};
#  elif defined(NV_TARGET_SM_INTEGER_LIST)
  return ::cuda::std::array{compute_capability{_Vs}...};
#  else // ^^^ has arch list ^^^ / vvv no arch list vvv
  static_assert(::cuda::std::__always_false_v<decltype(sizeof...(_Vs))>,
                "This function can be instantiated only when __CUDA_ARCH_LIST__ or NV_TARGET_SM_INTEGER_LIST are "
                "defined");
#  endif // ^^^ no arch list ^^^
}

[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL auto __target_compute_capabilities() noexcept
{
#  ifdef __CUDA_ARCH_LIST__
  return ::cuda::__make_cc_list<__CUDA_ARCH_LIST__>();
#  elif defined(NV_TARGET_SM_INTEGER_LIST)
  return ::cuda::__make_cc_list<NV_TARGET_SM_INTEGER_LIST>();
#  else // ^^^ has arch list ^^^ / vvv no arch list vvv
  // Fallback to a list of all compute capabilities.
  return ::cuda::__all_compute_capabilities();
#  endif // ^^^ no arch list ^^^
}
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___DEVICE_TARGET_COMPUTE_CAPABILITIES_H
