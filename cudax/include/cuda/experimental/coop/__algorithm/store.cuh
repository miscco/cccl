//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___ALGORITHM_STORE_CUH
#define _CUDA_EXPERIMENTAL_COOP___ALGORITHM_STORE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__memory/is_aligned.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/hierarchy>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__memory/runtime_assume_aligned.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/cmp.h>

#include <cuda/experimental/coop/__algorithm/load.cuh>
#include <cuda/experimental/coop/__group/concepts.cuh>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
// Vectors are only stored to contiguous ranges, whose items are not accessed through volatile references
template <class _OutputIt>
struct __store_target
{
  using __reference _CCCL_NODEBUG = ::cuda::std::iter_reference_t<_OutputIt>;
  using value_type _CCCL_NODEBUG  = ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<__reference>>;

  static constexpr bool __is_contiguous =
    THRUST_NS_QUALIFIER::is_contiguous_iterator_v<_OutputIt>
    && !::cuda::std::is_volatile_v<::cuda::std::remove_reference_t<__reference>>;

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static value_type* __pointer(const _OutputIt& __it)
  {
    return ::cuda::std::to_address(__it);
  }
};

// Stores use the access widths of loads without a cache modifier
template <class _OutputIt, ::cuda::std::size_t _VecSize>
using __store_widths_t _CCCL_NODEBUG =
  __load_widths<typename __store_target<_OutputIt>::value_type, _VecSize, ::cub::LOAD_DEFAULT>;

template <class _OutputIt, ::cuda::std::size_t _VecSize>
inline constexpr bool __store_can_vectorize_v =
  __store_target<_OutputIt>::__is_contiguous
  && ::cuda::is_trivially_copyable_v<typename __store_target<_OutputIt>::value_type>
  && __store_widths_t<_OutputIt, _VecSize>::__any;

// Converts the items of the thread to _OutputT and copies them into __words. Each word is copied with memcpy into a
// copy of it, because the items may be aligned to less than a word
template <class _OutputT, class _Word, ::cuda::std::size_t _NumWords, class _Tp, ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void __store_copy_words(const _Tp (&__src)[_Np], _Word (&__words)[_NumWords])
{
  auto __copy = [&](const _OutputT(&__items)[_Np]) {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < static_cast<int>(_NumWords); ++__i)
    {
      _Word __word;
      ::cuda::std::memcpy(
        &__word, reinterpret_cast<const unsigned char*>(__items) + (__i * int{sizeof(_Word)}), sizeof(_Word));
      __words[__i] = __word;
    }
  };
  if constexpr (::cuda::std::is_same_v<_Tp, _OutputT>)
  {
    __copy(__src);
  }
  else
  {
    _OutputT __items[_Np];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < static_cast<int>(_Np); ++__i)
    {
      __items[__i] = static_cast<_OutputT>(__src[__i]);
    }
    __copy(__items);
  }
}

// Stores the vectors of a thread with accesses of _Width bytes, which __block_dst must be aligned to. The first vector
// of the thread is vector __rank of the tile, and its vectors are __stride vectors apart
template <::cuda::std::size_t _Width,
          ::cuda::std::size_t _VecSize,
          class _OutputT,
          class _Index,
          class _Tp,
          ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
__store_vectors(_OutputT* __block_dst, _Index __rank, _Index __stride, const _Tp (&__src)[_Np])
{
  using _Word _CCCL_NODEBUG          = __load_word_t<_OutputT>;
  constexpr int __num_words          = static_cast<int>((_Np * sizeof(_OutputT)) / sizeof(_Word));
  constexpr int __vectors_per_thread = static_cast<int>(_Np / _VecSize);
  constexpr int __words_per_vector   = __num_words / __vectors_per_thread;
  using _Vector _CCCL_NODEBUG        = __load_aligned_words<_Word, __words_per_vector, _Width>;
  static_assert((_Np * sizeof(_OutputT)) % sizeof(_Word) == 0);

  _Word __words[__num_words];
  ::cuda::experimental::coop::__store_copy_words<_OutputT>(__src, __words);

  _Vector* __vec_ptr = reinterpret_cast<_Vector*>(::cuda::std::__runtime_assume_aligned(__block_dst, _Width)) + __rank;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __i = 0; __i < __vectors_per_thread; ++__i)
  {
    _Vector __vec;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __j = 0; __j < __words_per_vector; ++__j)
    {
      __vec.__words_[__j] = __words[(__i * __words_per_vector) + __j];
    }
    __vec_ptr[static_cast<_Index>(__i) * __stride] = __vec;
  }
}

// Stores the items of the vectors of a thread one by one, from __thread_dst, the first item of the thread, on. Its
// vectors are __stride vectors apart
template <::cuda::std::size_t _VecSize, class _OutputIt, class _Index, class _Tp, ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void __store_items(_OutputIt __thread_dst, _Index __stride, const _Tp (&__src)[_Np])
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __i = 0; __i < static_cast<int>(_Np / _VecSize); ++__i)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __j = 0; __j < static_cast<int>(_VecSize); ++__j)
    {
      const _Index __offset =
        ((static_cast<_Index>(__i) * __stride) * static_cast<_Index>(_VecSize)) + static_cast<_Index>(__j);
      __thread_dst[__offset] = __src[(__i * static_cast<int>(_VecSize)) + __j];
    }
  }
}

// Like __store_items, but only stores the items before item __valid_items of the tile. __thread_pos is the position of
// the first item of the thread in the tile
template <::cuda::std::size_t _VecSize, class _OutputIt, class _Index, class _Size, class _Tp, ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void __store_items_guarded(
  _OutputIt __thread_dst, _Index __thread_pos, _Index __stride, _Size __valid_items, const _Tp (&__src)[_Np])
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __i = 0; __i < static_cast<int>(_Np / _VecSize); ++__i)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __j = 0; __j < static_cast<int>(_VecSize); ++__j)
    {
      const _Index __offset =
        ((static_cast<_Index>(__i) * __stride) * static_cast<_Index>(_VecSize)) + static_cast<_Index>(__j);
      if (::cuda::std::cmp_less(__thread_pos + __offset, __valid_items))
      {
        __thread_dst[__offset] = __src[(__i * static_cast<int>(_VecSize)) + __j];
      }
    }
  }
}

// Stores the vectors of a thread with the widest accesses that the alignment of __block_dst allows, or item by item
template <::cuda::std::size_t _VecSize, class _OutputT, class _Index, class _Tp, ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
__store_dispatch(_OutputT* __block_dst, _Index __rank, _Index __stride, const _Tp (&__src)[_Np])
{
  using _Widths _CCCL_NODEBUG = __load_widths<_OutputT, _VecSize, ::cub::LOAD_DEFAULT>;

  // Every vector starts a whole number of vectors after __block_dst, thus a single test selects the accesses of all
  // vectors of the thread
  if constexpr (_Widths::__has(32))
  {
    NV_IF_TARGET(NV_PROVIDES_SM_100, ({
                   if (::cuda::is_aligned(__block_dst, 32))
                   {
                     ::cuda::experimental::coop::__store_vectors<32, _VecSize>(__block_dst, __rank, __stride, __src);
                     return;
                   }
                 }))
  }
  if constexpr (_Widths::__has(16))
  {
    if (::cuda::is_aligned(__block_dst, 16))
    {
      ::cuda::experimental::coop::__store_vectors<16, _VecSize>(__block_dst, __rank, __stride, __src);
      return;
    }
  }
  if constexpr (_Widths::__has(8))
  {
    if (::cuda::is_aligned(__block_dst, 8))
    {
      ::cuda::experimental::coop::__store_vectors<8, _VecSize>(__block_dst, __rank, __stride, __src);
      return;
    }
  }
  if constexpr (_Widths::__has(4))
  {
    if (::cuda::is_aligned(__block_dst, 4))
    {
      ::cuda::experimental::coop::__store_vectors<4, _VecSize>(__block_dst, __rank, __stride, __src);
      return;
    }
  }
  ::cuda::experimental::coop::__store_items<_VecSize>(
    __block_dst + (__rank * static_cast<_Index>(_VecSize)), __stride, __src);
}

// Stores the vectors of a thread with accesses of _Width bytes, which __block_dst must be aligned to, or item by item
// if _Width is 0
template <::cuda::std::size_t _Width,
          ::cuda::std::size_t _VecSize,
          class _OutputT,
          class _Index,
          class _Tp,
          ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
__store_static(_OutputT* __block_dst, _Index __rank, _Index __stride, const _Tp (&__src)[_Np])
{
  if constexpr (_Width == 0)
  {
    ::cuda::experimental::coop::__store_items<_VecSize>(
      __block_dst + (__rank * static_cast<_Index>(_VecSize)), __stride, __src);
  }
  else
  {
    ::cuda::experimental::coop::__store_vectors<_Width, _VecSize>(__block_dst, __rank, __stride, __src);
  }
}

template <::cuda::std::size_t _KnownAlignment,
          ::cuda::std::size_t _VecSize,
          class _Group,
          class _OutputIt,
          class _Tp,
          ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void __store(const _Group& __group, _OutputIt __dst, const _Tp (&__src)[_Np])
{
  static_assert(_Np % _VecSize == 0, "the vector size must evenly divide the number of items of a thread");

  using _Target _CCCL_NODEBUG = __store_target<_OutputIt>;
  using _Index _CCCL_NODEBUG  = __load_index_t<_Group>;

  const _Index __rank   = ::cuda::gpu_thread.rank(__group);
  const _Index __stride = ::cuda::experimental::coop::__load_stride<_VecSize, _Np>(__group);

  if constexpr (__store_can_vectorize_v<_OutputIt, _VecSize>)
  {
    using _Widths _CCCL_NODEBUG = __store_widths_t<_OutputIt, _VecSize>;
    auto* __block_dst           = _Target::__pointer(__dst);
    if constexpr (_KnownAlignment == 0)
    {
      ::cuda::experimental::coop::__store_dispatch<_VecSize>(__block_dst, __rank, __stride, __src);
    }
    else
    {
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_100,
        (::cuda::experimental::coop::__store_static<_Widths::__widest(_KnownAlignment), _VecSize>(
           __block_dst, __rank, __stride, __src);),
        (::cuda::experimental::coop::
           __store_static<_Widths::__widest(::cuda::std::min(_KnownAlignment, ::cuda::std::size_t{16})), _VecSize>(
             __block_dst, __rank, __stride, __src);))
    }
  }
  else
  {
    ::cuda::experimental::coop::__store_items<_VecSize>(
      __dst + (__rank * static_cast<_Index>(_VecSize)), __stride, __src);
  }
}

//! @brief Stores the ``_Np`` items of each of the ``P`` threads of a group into a tile of ``P * _Np`` consecutive
//! items, in the given arrangement.
//!
//! The tile is stored with the widest vectors that the alignment of ``__dst`` allows, which the store tests at
//! runtime, or item by item. Pass an alignment flag to rely on a known alignment instead.
//!
//! Vectors are stored to contiguous ranges of trivially copyable items that are not accessed through volatile
//! references, when the vectors of the arrangement allow accesses of at least 4 bytes that are wider than the items.
//! The items are converted to the value type of ``__dst``.
//!
//! @param[in] __group The group, whose threads each call the store.
//! @param[in] __arrangement ``blocked``, ``striped`` or ``striped_vectors<V>``.
//! @param[out] __dst The first item of the tile.
//! @param[in] __src The items of the thread.
_CCCL_TEMPLATE(class _Group, class _Arrangement, class _OutputIt, class _Tp, ::cuda::std::size_t _Np)
_CCCL_REQUIRES(group<_Group> _CCCL_AND __is_load_arrangement_v<_Arrangement>)
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
store(const _Group& __group, [[maybe_unused]] _Arrangement __arrangement, _OutputIt __dst, const _Tp (&__src)[_Np])
{
  ::cuda::experimental::coop::__store<0, __load_vec_size_v<_Arrangement, _Np>>(__group, __dst, __src);
}

//! @brief Stores a tile like ``store(__group, __arrangement, __dst, __src)``, relying on the alignment that
//! ``__alignment`` states instead of testing it.
//!
//! @param[in] __alignment ``element_aligned``, ``vector_aligned`` or ``overaligned<A>``. ``__dst`` must be aligned as
//! stated.
_CCCL_TEMPLATE(class _Group, class _Arrangement, class _OutputIt, class _Tp, ::cuda::std::size_t _Np, class _Alignment)
_CCCL_REQUIRES(group<_Group> _CCCL_AND __is_load_arrangement_v<_Arrangement> _CCCL_AND __is_load_alignment_v<_Alignment>)
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
store(const _Group& __group,
      [[maybe_unused]] _Arrangement __arrangement,
      _OutputIt __dst,
      const _Tp (&__src)[_Np],
      [[maybe_unused]] _Alignment __alignment)
{
  ::cuda::experimental::coop::__store<__load_known_alignment_v<_Alignment>, __load_vec_size_v<_Arrangement, _Np>>(
    __group, __dst, __src);
}

//! @brief Stores the items of a tile before item ``__valid_items`` in the given arrangement, item by item through
//! ``__dst``. The other items of the tile are not written.
_CCCL_TEMPLATE(class _Group, class _Arrangement, class _OutputIt, class _Tp, ::cuda::std::size_t _Np, class _Size)
_CCCL_REQUIRES(
  group<_Group> _CCCL_AND __is_load_arrangement_v<_Arrangement> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Size>)
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
store(const _Group& __group,
      [[maybe_unused]] _Arrangement __arrangement,
      _OutputIt __dst,
      const _Tp (&__src)[_Np],
      _Size __valid_items)
{
  constexpr auto __vec_size  = __load_vec_size_v<_Arrangement, _Np>;
  using _Index _CCCL_NODEBUG = __load_index_t<_Group>;
  static_assert(_Np % __vec_size == 0, "the vector size must evenly divide the number of items of a thread");

  const _Index __thread_pos = ::cuda::gpu_thread.rank(__group) * static_cast<_Index>(__vec_size);
  ::cuda::experimental::coop::__store_items_guarded<__vec_size>(
    __dst + __thread_pos,
    __thread_pos,
    ::cuda::experimental::coop::__load_stride<__vec_size, _Np>(__group),
    __valid_items,
    __src);
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___ALGORITHM_STORE_CUH
