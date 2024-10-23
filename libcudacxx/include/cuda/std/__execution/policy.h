//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___EXECUTION_POLICY_H
#define _LIBCUDACXX___EXECUTION_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

_LIBCUDACXX_BEGIN_NAMESPACE_EXECUTION

struct __disable_user_instantiations_tag
{
  explicit __disable_user_instantiations_tag() = default;
};

struct sequenced_policy
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit sequenced_policy(__disable_user_instantiations_tag) noexcept {}
  sequenced_policy(const sequenced_policy&)            = delete;
  sequenced_policy& operator=(const sequenced_policy&) = delete;
};

_CCCL_GLOBAL_CONSTANT sequenced_policy seq{__disable_user_instantiations_tag{}};

struct parallel_policy_host
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit parallel_policy_host(__disable_user_instantiations_tag) noexcept {}
  parallel_policy_host(const parallel_policy_host&)            = delete;
  parallel_policy_host& operator=(const parallel_policy_host&) = delete;
};
_CCCL_GLOBAL_CONSTANT parallel_policy_host par_host{__disable_user_instantiations_tag{}};

struct parallel_policy_device
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit parallel_policy_device(__disable_user_instantiations_tag) noexcept {}
  parallel_policy_device(const parallel_policy_device&)            = delete;
  parallel_policy_device& operator=(const parallel_policy_device&) = delete;
};

_CCCL_GLOBAL_CONSTANT parallel_policy_device par_device{__disable_user_instantiations_tag{}};

struct parallel_unsequenced_policy_host
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit parallel_unsequenced_policy_host(
    __disable_user_instantiations_tag) noexcept
  {}
  parallel_unsequenced_policy_host(const parallel_unsequenced_policy_host&)            = delete;
  parallel_unsequenced_policy_host& operator=(const parallel_unsequenced_policy_host&) = delete;
};
_CCCL_GLOBAL_CONSTANT parallel_unsequenced_policy_host par_unseq_host{__disable_user_instantiations_tag{}};

struct parallel_unsequenced_policy_device
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit parallel_unsequenced_policy_device(
    __disable_user_instantiations_tag) noexcept
  {}
  parallel_unsequenced_policy_device(const parallel_unsequenced_policy_device&)            = delete;
  parallel_unsequenced_policy_device& operator=(const parallel_unsequenced_policy_device&) = delete;
};
_CCCL_GLOBAL_CONSTANT parallel_unsequenced_policy_device par_unseq_device{__disable_user_instantiations_tag{}};

struct unsequenced_policy_host
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit unsequenced_policy_host(__disable_user_instantiations_tag) noexcept {}
  unsequenced_policy_host(const unsequenced_policy_host&)            = delete;
  unsequenced_policy_host& operator=(const unsequenced_policy_host&) = delete;
};
_CCCL_GLOBAL_CONSTANT unsequenced_policy_host unseq_host{__disable_user_instantiations_tag{}};

struct unsequenced_policy_device
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit unsequenced_policy_device(__disable_user_instantiations_tag) noexcept {}
  unsequenced_policy_device(const unsequenced_policy_device&)            = delete;
  unsequenced_policy_device& operator=(const unsequenced_policy_device&) = delete;
};
_CCCL_GLOBAL_CONSTANT unsequenced_policy_device unseq_device{__disable_user_instantiations_tag{}};

_LIBCUDACXX_END_NAMESPACE_EXECUTION

#endif // _LIBCUDACXX___EXECUTION_POLICY_H
