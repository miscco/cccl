//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_MINMAX_ELEMENT_H
#define _CUDA_STD___PSTL_CUDA_MINMAX_ELEMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_BACKEND_CUDA()

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wshadow")
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-local-typedef")
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

#  include <cub/device/device_reduce.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__iterator/zip_transform_iterator.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/minmax_element.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_callable.h>
#  include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#  include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/__utility/pair.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

struct __minmax_fake_init
{};

template <class _Tp>
struct __minmax_result_type
{
  size_t __min_index_;
  size_t __max_index_;
  _Tp __min_value_;
  _Tp __max_value_;

  _CCCL_API constexpr __minmax_result_type& operator=(__minmax_fake_init) noexcept
  { // Needed to satisfy CUB implementation details
    _CCCL_ASSERT(false, "Should not assign __minmax_result_type from __minmax_fake_init");
    return *this;
  }
};

template <class _Tp, class _BinaryPredicate>
struct __minmax_functor
{
  _BinaryPredicate __pred_;

  _CCCL_API constexpr __minmax_functor(_BinaryPredicate __pred) noexcept(
    is_nothrow_move_constructible_v<_BinaryPredicate>)
      : __pred_(::cuda::std::move(__pred))
  {}

  [[nodiscard]] _CCCL_DEVICE_API constexpr __minmax_result_type<_Tp>
  operator()(const __minmax_result_type<_Tp>& __lhs, const __minmax_result_type<_Tp>& __rhs) const
    noexcept(is_nothrow_copy_constructible_v<_Tp> && __is_nothrow_callable_v<_BinaryPredicate, const _Tp&, const _Tp&>)
  {
    __minmax_result_type __result = __lhs;
    if (__pred_(__rhs.__min_value_, __result.__min_value_))
    { // __rhs.__min_value_ is smaller, update both
      __result.__min_index_ = __rhs.__min_index_;
      __result.__min_value_ = __rhs.__min_value_;
    }
    else if (!__pred_(__result.__min_value_, __rhs.__min_value_))
    { // __rhs.__min_value_ is equal, update index to the smaller one
      __result.__min_index_ = __result.__min_index_ < __rhs.__min_index_ ? __result.__min_index_ : __rhs.__min_index_;
    }

    if (__pred_(__result.__max_value_, __rhs.__max_value_))
    { // __rhs.__max_value_ is greater, update both
      __result.__max_index_ = __rhs.__max_index_;
      __result.__max_value_ = __rhs.__max_value_;
    }
    else if (!__pred_(__rhs.__max_value_, __result.__max_value_))
    { // __rhs.__max_value_ is equal, update index to the smaller one
      __result.__max_index_ = __result.__max_index_ < __rhs.__max_index_ ? __result.__max_index_ : __rhs.__max_index_;
    }
    return __result;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr __minmax_result_type<_Tp>
  operator()(__minmax_fake_init __lhs, const __minmax_result_type<_Tp>& __rhs) const
    noexcept(is_nothrow_copy_constructible_v<_Tp>)
  {
    return __rhs;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr __minmax_result_type<_Tp>
  operator()(const __minmax_result_type<_Tp>& __lhs, __minmax_fake_init __rhs) const
    noexcept(is_nothrow_copy_constructible_v<_Tp>)
  {
    return __lhs;
  }
};

// Note: This needs to be on the struct not the call operator.
// The issue is that we need the raw value type and not device_reference<value_type>
template <class _Tp>
struct __minmax_expand
{
  [[nodiscard]] _CCCL_API constexpr __minmax_result_type<_Tp> operator()(const size_t __index, const _Tp& __value) const
    noexcept(is_nothrow_copy_constructible_v<_Tp>)
  {
    return {__index, __index, __value, __value};
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__minmax_element, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator, class _BinaryPred>
  [[nodiscard]] _CCCL_HOST_API static pair<_InputIterator, _InputIterator> __par_impl(
    [[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, _BinaryPred __pred)
  {
    using value_type = iter_value_t<_InputIterator>;
    __minmax_result_type<value_type> __ret;
    const auto __count = static_cast<int64_t>(::cuda::std::distance(__first, __last));
    auto __stream      = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, __policy);

    // We need to zip_transform __first into __minmax_result
    auto __zip_first =
      ::cuda::zip_transform_iterator{__minmax_expand<value_type>{}, ::cuda::counting_iterator<size_t>{0}, __first};
    auto __zip_pred = __minmax_functor<value_type, _BinaryPred>{::cuda::std::move(__pred)};

    // Determine temporary device storage requirements for minmax_element
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceReduce::Reduce,
      "__pstl_cuda_minmax_element: determination of device storage for cub::DeviceReduce::Reduce failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      __zip_first,
      static_cast<__minmax_result_type<value_type>*>(nullptr),
      __count,
      __zip_pred,
      CUB_NS_QUALIFIER::detail::reduce::empty_problem_init_t<__minmax_fake_init>{},
      __stream.get());

    {
      __temporary_storage<__minmax_result_type<value_type>> __storage{__policy, __num_bytes, 1};

      // Run the reduction
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DeviceReduce::Reduce,
        "__pstl_cuda_minmax_element: kernel launch of cub::DeviceReduce::Reduce failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        __zip_first,
        __storage.template __get_raw_ptr<0>(),
        __count,
        ::cuda::std::move(__zip_pred),
        CUB_NS_QUALIFIER::detail::reduce::empty_problem_init_t<__minmax_fake_init>{},
        __stream.get());

      // Copy the result back from storage
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "__pstl_cuda_minmax_element: copy of result from device to host failed",
        ::cuda::std::addressof(__ret),
        __storage.template __get_raw_ptr<0>(),
        sizeof(__minmax_result_type<value_type>),
        ::cudaMemcpyDefault,
        __stream.get());
    }

    __stream.sync();
    return {__first + static_cast<iter_difference_t<_InputIterator>>(__ret.__min_index_),
            __first + static_cast<iter_difference_t<_InputIterator>>(__ret.__max_index_)};
  }

  template <class _Policy, class _InputIterator, class _BinaryPred>
  [[nodiscard]] _CCCL_HOST_API pair<_InputIterator, _InputIterator> operator()(
    [[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, _BinaryPred __pred) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>)
    {
      _CCCL_TRY
      {
        return __par_impl(__policy, ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
      }
      _CCCL_CATCH (const ::cuda::cuda_error& __err)
      {
        if (__err.status() == cudaErrorMemoryAllocation)
        {
          _CCCL_THROW(::std::bad_alloc);
        }
        else
        {
          _CCCL_RETHROW;
        }
      }
      _CCCL_CATCH_FALLTHROUGH
    }
    else
    {
      static_assert(__always_false_v<_Policy>,
                    "__pstl_dispatch: CUDA backend of cuda::std::minmax_element requires at least random access "
                    "iterators");
      return ::cuda::std::minmax_element(
        ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_MINMAX_ELEMENT_H
