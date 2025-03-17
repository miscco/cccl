//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___SEMAPHORE_FAIR_SEMAPHORE_H
#define _LIBCUDACXX___SEMAPHORE_FAIR_SEMAPHORE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/atomic>
#include <cuda/std/chrono>
#include <cuda/std/cstdint>

_CCCL_PUSH_MACROS

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <thread_scope _Sco>
class __fair_sempahore
{
private:
  union
  {
    __atomic_impl<uint64_t, _Sco> __tickets{0};
    struct
    {
      __atomic_impl<uint32_t, _Sco> __current_ticket{0};
      __atomic_impl<uint32_t, _Sco> __next_ticket{0};
    } __store;
  };

public:
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __fair_sempahore(ptrdiff_t __desired) noexcept
      : __store{static_cast<uint32_t>(__desired), static_cast<uint32_t>(__desired)}
  {
    _CCCL_ASSERT(__desired == 0 || __desired == 1, "Invalid desired count passed to __fair_sempahore constructor");
  }

  __fair_sempahore(__fair_sempahore const&)            = delete;
  __fair_sempahore& operator=(__fair_sempahore const&) = delete;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr ptrdiff_t max() noexcept
  {
    return 1;
  }

  _LIBCUDACXX_HIDE_FROM_ABI void release([[maybe_unused]] ptrdiff_t __update) noexcept
  {
    _CCCL_ASSERT(__update == 1, "Invalid update passed to __fair_sempahore::release()");
    __store.__current_ticket.store(
      __store.__current_ticket.load(__update, memory_order_relaxed) + 1, memory_order_release);
    __store.__count.notify_one();
  }

  _LIBCUDACXX_HIDE_FROM_ABI void acquire()
  {
    const uint32_t __ticket = __store.__current_ticket.fetch_add(1, memory_order_relaxed);
    __store.__current_ticket.wait(__ticket);
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool try_acquire() noexcept
  {
    const uint32_t __my_ticket = __store.__next_ticket.fetch_add(1, memory_order_release);
    return __store.__current_ticket.load(memory_order_acquire) != __my_ticket;
  }

  template <class Clock, class Duration>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool try_acquire_until(chrono::time_point<Clock, Duration> const& __abs_time)
  {
    if (try_acquire())
    {
      return true;
    }
    else
    {
      return __acquire_slow_timed(__abs_time - Clock::now());
    }
  }

  template <class Rep, class Period>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool try_acquire_for(chrono::duration<Rep, Period> const& __rel_time)
  {
    if (try_acquire())
    {
      return true;
    }
    else
    {
      return __acquire_slow_timed(__rel_time);
    }
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_POP_MACROS

#endif // _LIBCUDACXX___SEMAPHORE_FAIR_SEMAPHORE_H
