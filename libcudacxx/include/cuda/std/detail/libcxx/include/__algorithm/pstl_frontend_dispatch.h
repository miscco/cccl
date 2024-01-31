//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PSTL_FRONTEND_DISPATCH
#define _LIBCUDACXX___ALGORITHM_PSTL_FRONTEND_DISPATCH

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

#include "../__type_traits/is_callable.h"
#include "../__utility/forward.h"

#if _CCCL_STD_VER >= 2017

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _SpecializedImpl, class _GenericImpl, class... _Args>
_LIBCUDACXX_INLINE_VISIBILITY decltype(auto)
__pstl_frontend_dispatch(_SpecializedImpl __specialized_impl, _GenericImpl __generic_impl, _Args&&... __args)
{
  if constexpr (__is_callable<_SpecializedImpl, _Args...>::value)
  {
    return __specialized_impl(_CUDA_VSTD::forward<_Args>(__args)...);
  }
  else
  {
    return __generic_impl(_CUDA_VSTD::forward<_Args>(__args)...);
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_STD_VER >= 2017

#endif // _LIBCUDACXX___ALGORITHM_PSTL_FRONTEND_DISPATCH
