// SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * \file
 * This file contains facilities that help to prevent exceeding the available shared memory per thread block
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/required_smem.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_policy_wrapper_t.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/round_up.h>
#include <cuda/__memory/discard_memory.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{
/**
 * @brief Helper struct to wrap all the information needed to implement virtual shared memory that's passed to a kernel.
 *
 */
struct vsmem_t
{
  void* gmem_ptr;
};

[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL bool needs_vsmem(::cuda::std::size_t required_smem)
{
  return required_smem > max_smem_per_block;
}

// Bytes of global-memory-backed virtual shared memory to allocate per block for an agent that requires `required_smem`
// bytes of temporary storage.
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL ::cuda::std::size_t
vsmem_bytes_per_block(::cuda::std::size_t required_smem)
{
  constexpr ::cuda::std::size_t line_size = 128;
  return needs_vsmem(required_smem) ? ::cuda::round_up(required_smem, line_size) : 0;
}

[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL bool
use_fallback_smem(::cuda::std::size_t default_smem, ::cuda::std::size_t fallback_smem)
{
  return needs_vsmem(default_smem) && !needs_vsmem(fallback_smem);
}

/**
 * @brief Device-side adapter that maps static shared memory or global-backed virtual shared memory onto an agent's
 * temporary storage.
 *
 * @tparam TempStorage The agent's `TempStorage` type. Named only from within the selected kernel.
 * @tparam RequiredSmem Byte size of that storage, from `required_smem_v` (or an equivalent cheap query).
 */
template <typename TempStorage, ::cuda::std::size_t RequiredSmem>
class agent_block_smem
{
private:
  static constexpr ::cuda::std::size_t line_size = 128;
  static constexpr bool uses_vsmem               = needs_vsmem(RequiredSmem);

public:
  using static_temp_storage_t = ::cuda::std::conditional_t<uses_vsmem, cub::NullType, TempStorage>;

  static constexpr ::cuda::std::size_t vsmem_per_block = vsmem_bytes_per_block(RequiredSmem);

  static _CCCL_DEVICE _CCCL_FORCEINLINE TempStorage& get_temp_storage(TempStorage& static_temp_storage, vsmem_t&)
  {
    return static_temp_storage;
  }

  static _CCCL_DEVICE _CCCL_FORCEINLINE TempStorage&
  get_temp_storage(TempStorage& static_temp_storage, vsmem_t&, ::cuda::std::size_t)
  {
    return static_temp_storage;
  }

  static _CCCL_DEVICE _CCCL_FORCEINLINE TempStorage& get_temp_storage(cub::NullType&, vsmem_t& vsmem)
  {
    return *reinterpret_cast<TempStorage*>(static_cast<char*>(vsmem.gmem_ptr) + (vsmem_per_block * blockIdx.x));
  }

  static _CCCL_DEVICE _CCCL_FORCEINLINE TempStorage&
  get_temp_storage(cub::NullType&, vsmem_t& vsmem, ::cuda::std::size_t linear_block_id)
  {
    return *reinterpret_cast<TempStorage*>(static_cast<char*>(vsmem.gmem_ptr) + (vsmem_per_block * linear_block_id));
  }

  template <bool UsesVsmem = uses_vsmem, ::cuda::std::enable_if_t<!UsesVsmem, int> = 0>
  static _CCCL_DEVICE _CCCL_FORCEINLINE bool discard_temp_storage(TempStorage&)
  {
    return false;
  }

  template <bool UsesVsmem = uses_vsmem, ::cuda::std::enable_if_t<UsesVsmem, int> = 0>
  static _CCCL_DEVICE _CCCL_FORCEINLINE bool discard_temp_storage(TempStorage& temp_storage)
  {
    __syncthreads();

    const ::cuda::std::size_t linear_tid   = threadIdx.x;
    const ::cuda::std::size_t block_stride = line_size * blockDim.x;

    char* ptr    = reinterpret_cast<char*>(&temp_storage);
    auto ptr_end = ptr + vsmem_per_block;

    for (auto thread_ptr = ptr + (linear_tid * line_size); thread_ptr < ptr_end; thread_ptr += block_stride)
    {
      ::cuda::discard_memory(thread_ptr, line_size);
    }
    return true;
  }
};

/**
 * @brief Chooses the default or fallback policy from two cheap shared-memory sizes. Does not instantiate an agent.
 */
template <typename DefaultPolicyT,
          typename FallbackPolicyT,
          ::cuda::std::size_t DefaultSmem,
          ::cuda::std::size_t FallbackSmem,
          bool UseFallbackPolicy = use_fallback_smem(DefaultSmem, FallbackSmem)>
struct smem_policy_choice
{
  using agent_policy_t                                 = DefaultPolicyT;
  static constexpr ::cuda::std::size_t required_smem   = DefaultSmem;
  static constexpr ::cuda::std::size_t vsmem_per_block = vsmem_bytes_per_block(DefaultSmem);
};

template <typename DefaultPolicyT,
          typename FallbackPolicyT,
          ::cuda::std::size_t DefaultSmem,
          ::cuda::std::size_t FallbackSmem>
struct smem_policy_choice<DefaultPolicyT, FallbackPolicyT, DefaultSmem, FallbackSmem, true>
{
  using agent_policy_t                                 = FallbackPolicyT;
  static constexpr ::cuda::std::size_t required_smem   = FallbackSmem;
  static constexpr ::cuda::std::size_t vsmem_per_block = vsmem_bytes_per_block(FallbackSmem);
};

/**
 * @brief Fallback selection for an algorithm that answers `required_smem_layout`. Host dispatch should use this
 * instead of instantiating default and fallback agents.
 */
template <cub_algorithm Algorithm, typename DefaultPolicyT, typename FallbackPolicyT, typename... Args>
using smem_fallback_traits =
  smem_policy_choice<DefaultPolicyT,
                     FallbackPolicyT,
                     required_smem_v<Algorithm, DefaultPolicyT, Args...>,
                     required_smem_v<Algorithm, FallbackPolicyT, Args...>>;

template <cub_algorithm Algorithm, typename DefaultPolicyT, typename... Args>
using smem_default_fallback_traits =
  smem_fallback_traits<Algorithm, DefaultPolicyT, policy_wrapper_t<DefaultPolicyT, 64, 1>, Args...>;
} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
