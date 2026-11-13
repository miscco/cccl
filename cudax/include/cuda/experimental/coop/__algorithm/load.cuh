//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___ALGORITHM_LOAD_CUH
#define _CUDA_EXPERIMENTAL_COOP___ALGORITHM_LOAD_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_load.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__memory/__vectorized_storage.h>
#include <cuda/__memory/is_aligned.h>
#include <cuda/__memory/is_valid_alignment.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/__type_traits/is_vector_type.h>
#include <cuda/hierarchy>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__memory/runtime_assume_aligned.h>
#include <cuda/std/__type_traits/has_unique_object_representation.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/climits>

#include <cuda/experimental/coop/__group/concepts.cuh>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
//! @brief The blocked arrangement: thread ``r`` of a group owns the ``N`` consecutive items from item ``r * N`` on.
struct blocked_t
{};

//! @brief The striped arrangement of vectors of ``_VecSize`` consecutive items: vector ``i`` of thread ``r`` of a group
//! of ``P`` threads holds the items from item ``(i * P + r) * _VecSize`` on.
template <::cuda::std::size_t _VecSize>
struct striped_t
{
  static_assert(_VecSize > 0, "a vector holds at least one item");
};

_CCCL_GLOBAL_CONSTANT blocked_t blocked{};

//! @brief The striped arrangement of single items: item ``i`` of thread ``r`` of a group of ``P`` threads is item
//! ``i * P + r``.
_CCCL_GLOBAL_CONSTANT striped_t<1> striped{};

//! @brief The striped arrangement of vectors of ``_VecSize`` consecutive items.
template <::cuda::std::size_t _VecSize>
_CCCL_GLOBAL_CONSTANT striped_t<_VecSize> striped_vectors{};

//! @brief The tile is only known to be aligned to its items: the load reads items and tests no alignment.
struct element_aligned_t
{};

//! @brief The tile is aligned to the widest vectors of the load, see ``is_vector_aligned``: the load reads these
//! vectors and tests no alignment.
struct vector_aligned_t
{};

//! @brief The tile is aligned to ``_Alignment`` bytes: the load reads the widest vectors that this alignment allows and
//! tests no alignment.
template <::cuda::std::size_t _Alignment>
struct overaligned_t
{
  static_assert(::cuda::__is_valid_alignment(_Alignment), "the alignment must be a power of two");
};

_CCCL_GLOBAL_CONSTANT element_aligned_t element_aligned{};
_CCCL_GLOBAL_CONSTANT vector_aligned_t vector_aligned{};
template <::cuda::std::size_t _Alignment>
_CCCL_GLOBAL_CONSTANT overaligned_t<_Alignment> overaligned{};

template <class _Tp>
inline constexpr bool __is_load_arrangement_v = false;
template <>
inline constexpr bool __is_load_arrangement_v<blocked_t> = true;
template <::cuda::std::size_t _VecSize>
inline constexpr bool __is_load_arrangement_v<striped_t<_VecSize>> = true;

// The number of consecutive items of a vector of the arrangement, when a thread owns _Np items
template <class _Arrangement, ::cuda::std::size_t _Np>
inline constexpr ::cuda::std::size_t __load_vec_size_v = _Np;
template <::cuda::std::size_t _VecSize, ::cuda::std::size_t _Np>
inline constexpr ::cuda::std::size_t __load_vec_size_v<striped_t<_VecSize>, _Np> = _VecSize;

// The alignment of the tile in bytes that the load relies on without testing it, 0 if it tests the alignment
template <class _Tp>
inline constexpr bool __is_load_alignment_v = false;
template <>
inline constexpr bool __is_load_alignment_v<element_aligned_t> = true;
template <>
inline constexpr bool __is_load_alignment_v<vector_aligned_t> = true;
template <::cuda::std::size_t _Alignment>
inline constexpr bool __is_load_alignment_v<overaligned_t<_Alignment>> = true;

template <class _Tp>
inline constexpr ::cuda::std::size_t __load_known_alignment_v = 0;
template <>
inline constexpr ::cuda::std::size_t __load_known_alignment_v<element_aligned_t> = 1;
template <>
inline constexpr ::cuda::std::size_t __load_known_alignment_v<vector_aligned_t> = ~::cuda::std::size_t{0};
template <::cuda::std::size_t _Alignment>
inline constexpr ::cuda::std::size_t __load_known_alignment_v<overaligned_t<_Alignment>> = _Alignment;

// Vectors are only loaded from contiguous ranges, whose items are not accessed through volatile references
template <class _InputIt>
struct __load_source
{
  using value_type _CCCL_NODEBUG = ::cuda::std::iter_value_t<_InputIt>;

  static constexpr ::cub::CacheLoadModifier __modifier = ::cub::LOAD_DEFAULT;

  static constexpr bool __is_contiguous =
    THRUST_NS_QUALIFIER::is_contiguous_iterator_v<_InputIt>
    && !::cuda::std::is_volatile_v<::cuda::std::remove_reference_t<::cuda::std::iter_reference_t<_InputIt>>>;

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static const value_type* __pointer(const _InputIt& __it)
  {
    return ::cuda::std::to_address(__it);
  }
};

template <::cub::CacheLoadModifier _Modifier, class _ValueType, class _Offset>
struct __load_source<::cub::CacheModifiedInputIterator<_Modifier, _ValueType, _Offset>>
{
  using value_type _CCCL_NODEBUG = ::cuda::std::remove_cv_t<_ValueType>;

  static constexpr ::cub::CacheLoadModifier __modifier = _Modifier;

  static constexpr bool __is_contiguous = !::cuda::std::is_volatile_v<_ValueType>;

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static const value_type*
  __pointer(const ::cub::CacheModifiedInputIterator<_Modifier, _ValueType, _Offset>& __it)
  {
    return __it.ptr;
  }
};

// The widths in bytes of the accesses with which a load reads vectors of _VecSize consecutive items of _InputT
template <class _InputT, ::cuda::std::size_t _VecSize, ::cub::CacheLoadModifier _Modifier>
struct __load_widths
{
  // An access is no wider than a vector. Only loads without a cache modifier or with LOAD_LDG access 32 bytes at once,
  // and ThreadLoad copies volatile vectors in 8 byte words
  static constexpr ::cuda::std::size_t __max = ::cuda::std::min(
    ::cuda::std::size_t{1} << ::cuda::std::countr_zero(_VecSize * sizeof(_InputT)),
    (_Modifier == ::cub::LOAD_DEFAULT || _Modifier == ::cub::LOAD_LDG) ? ::cuda::__max_memory_access_width
    : (_Modifier == ::cub::LOAD_VOLATILE)
      ? ::cuda::std::size_t{8}
      : ::cuda::std::size_t{16});

  // LOAD_CG, LOAD_CS, LOAD_CV and LOAD_VOLATILE bypass L1 or evict from it first, thus their narrower accesses load
  // slower than the items through L1
  static constexpr ::cuda::std::size_t __min =
    (_Modifier == ::cub::LOAD_CG || _Modifier == ::cub::LOAD_CS || _Modifier == ::cub::LOAD_CV
     || _Modifier == ::cub::LOAD_VOLATILE)
      ? __max
      : ::cuda::std::size_t{4};

  // Accesses no wider than the alignment of the items load like the items
  [[nodiscard]] _CCCL_API static constexpr bool __has(::cuda::std::size_t __width) noexcept
  {
    return alignof(_InputT) < __width && __min <= __width && __width <= __max;
  }

  // The widest access of at most __alignment bytes, or 0 if there is none
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t __widest(::cuda::std::size_t __alignment) noexcept
  {
    for (::cuda::std::size_t __width = 32; __width >= 4; __width /= 2)
    {
      if (__width <= __alignment && __has(__width))
      {
        return __width;
      }
    }
    return 0;
  }

  static constexpr bool __any = __widest(~::cuda::std::size_t{0}) != 0;
};

template <class _InputIt, ::cuda::std::size_t _VecSize>
using __load_widths_t _CCCL_NODEBUG =
  __load_widths<typename __load_source<_InputIt>::value_type, _VecSize, __load_source<_InputIt>::__modifier>;

template <class _InputIt, ::cuda::std::size_t _VecSize>
inline constexpr bool __load_can_vectorize_v =
  __load_source<_InputIt>::__is_contiguous
  && ::cuda::is_trivially_copyable_v<typename __load_source<_InputIt>::value_type>
  && __load_widths_t<_InputIt, _VecSize>::__any;

// Every byte holds a part of the value: arithmetic and extended floating point types, CUDA vector types, and types with
// unique object representations
template <class _Tp>
inline constexpr bool __load_is_padding_free_v =
  ::cuda::std::__is_extended_arithmetic_v<_Tp> || ::cuda::is_vector_type_v<_Tp>
  || ::cuda::std::has_unique_object_representations_v<_Tp>;

// The words into which the vectors of every width are loaded. They divide the items of a thread and of every vector,
// which allow accesses of at least 4 bytes
template <class _InputT>
using __load_word_t _CCCL_NODEBUG = ::cuda::std::__make_nbit_uint_t<
  ::cuda::std::clamp(alignof(_InputT), ::cuda::std::size_t{4}, ::cuda::std::size_t{8}) * CHAR_BIT>;

// _NumWords words aligned to _Alignment bytes, which must divide their size, thus a copy of them is loaded with
// accesses of _Alignment bytes
template <class _Word, ::cuda::std::size_t _NumWords, ::cuda::std::size_t _Alignment>
struct alignas(_Alignment) __load_aligned_words
{
  _Word __words_[_NumWords];
};

// Every path of a load that reads vectors ends with this copy of the words into __dst, which the compiler merges.
// Padded items loaded without a cache modifier are copied one by one, because copying their words narrows the vector
// accesses to the members. __dst may be aligned to less than a word, thus each word is copied with memcpy from a copy
// of it: a memcpy straight from the words is split into bytes
template <class _InputT,
          ::cub::CacheLoadModifier _Modifier,
          class _Word,
          ::cuda::std::size_t _NumWords,
          class _Tp,
          ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void __load_copy_words(const _Word (&__words)[_NumWords], _Tp (&__dst)[_Np])
{
  if constexpr (::cuda::std::is_same_v<_Tp, _InputT>
                && (__load_is_padding_free_v<_InputT> || _Modifier != ::cub::LOAD_DEFAULT))
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < static_cast<int>(_NumWords); ++__i)
    {
      const _Word __word = __words[__i];
      ::cuda::std::memcpy(reinterpret_cast<unsigned char*>(__dst) + (__i * int{sizeof(_Word)}), &__word, sizeof(_Word));
    }
  }
  else
  {
    const _InputT* __items = reinterpret_cast<const _InputT*>(__words);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < static_cast<int>(_Np); ++__i)
    {
      __dst[__i] = static_cast<_Tp>(__items[__i]);
    }
  }
}

// Loads the vectors of a thread with accesses of _Width bytes, which __block_src must be aligned to, and copies them
// into __dst through the shared __words. The first vector of the thread is vector __rank of the tile, and its vectors
// are __stride vectors apart
template <::cuda::std::size_t _Width,
          ::cuda::std::size_t _VecSize,
          ::cub::CacheLoadModifier _Modifier,
          class _InputT,
          class _Index,
          class _Word,
          ::cuda::std::size_t _NumWords,
          class _Tp,
          ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void __load_vectors(
  const _InputT* __block_src, _Index __rank, _Index __stride, _Word (&__words)[_NumWords], _Tp (&__dst)[_Np])
{
  constexpr int __vectors_per_thread = static_cast<int>(_Np / _VecSize);
  constexpr int __words_per_vector   = static_cast<int>(_NumWords) / __vectors_per_thread;
  using _Vector _CCCL_NODEBUG        = __load_aligned_words<_Word, __words_per_vector, _Width>;

  const _Vector* __vec_ptr =
    reinterpret_cast<const _Vector*>(::cuda::std::__runtime_assume_aligned(__block_src, _Width)) + __rank;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __i = 0; __i < __vectors_per_thread; ++__i)
  {
    const _Vector __vec = ::cub::ThreadLoad<_Modifier>(__vec_ptr + (static_cast<_Index>(__i) * __stride));
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __j = 0; __j < __words_per_vector; ++__j)
    {
      __words[(__i * __words_per_vector) + __j] = __vec.__words_[__j];
    }
  }
  ::cuda::experimental::coop::__load_copy_words<_InputT, _Modifier>(__words, __dst);
}

// Loads the items of the vectors of a thread one by one, from __thread_src, the first item of the thread, on. Its
// vectors are __stride vectors apart
template <::cuda::std::size_t _VecSize, class _InputIt, class _Index, class _Tp, ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void __load_items(_InputIt __thread_src, _Index __stride, _Tp (&__dst)[_Np])
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __i = 0; __i < static_cast<int>(_Np / _VecSize); ++__i)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __j = 0; __j < static_cast<int>(_VecSize); ++__j)
    {
      const _Index __offset =
        ((static_cast<_Index>(__i) * __stride) * static_cast<_Index>(_VecSize)) + static_cast<_Index>(__j);
      __dst[(__i * static_cast<int>(_VecSize)) + __j] = __thread_src[__offset];
    }
  }
}

// Like __load_items, but only loads the items before item __valid_items of the tile. __thread_pos is the position of
// the first item of the thread in the tile
template <::cuda::std::size_t _VecSize, class _InputIt, class _Index, class _Size, class _Tp, ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void __load_items_guarded(
  _InputIt __thread_src, _Index __thread_pos, _Index __stride, _Size __valid_items, _Tp (&__dst)[_Np])
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
        __dst[(__i * static_cast<int>(_VecSize)) + __j] = __thread_src[__offset];
      }
    }
  }
}

template <class _Group>
using __load_index_t _CCCL_NODEBUG = decltype(::cuda::gpu_thread.rank(::cuda::std::declval<const _Group&>()));

// The number of vectors between two consecutive vectors of a thread: the number of threads of the group. Threads that
// own a single vector do not query it
template <::cuda::std::size_t _VecSize, ::cuda::std::size_t _Np, class _Group>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE __load_index_t<_Group> __load_stride(const _Group& __group)
{
  using _Index _CCCL_NODEBUG = __load_index_t<_Group>;
  if constexpr (_Np == _VecSize)
  {
    return _Index{0};
  }
  else
  {
    return ::cuda::gpu_thread.count_as<_Index>(__group);
  }
}

template <::cuda::std::size_t _Width>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __load_is_aligned(const void* __ptr)
{
  if constexpr (_Width == 0)
  {
    return false;
  }
  else
  {
    return ::cuda::is_aligned(__ptr, _Width);
  }
}

// Loads the vectors of a thread with the widest accesses that the alignment of __block_src allows, or item by item
template <::cuda::std::size_t _VecSize,
          ::cub::CacheLoadModifier _Modifier,
          class _InputT,
          class _Index,
          class _Tp,
          ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
__load_dispatch(const _InputT* __block_src, _Index __rank, _Index __stride, _Tp (&__dst)[_Np])
{
  using _Widths _CCCL_NODEBUG = __load_widths<_InputT, _VecSize, _Modifier>;
  using _Word _CCCL_NODEBUG   = __load_word_t<_InputT>;
  static_assert((_Np * sizeof(_InputT)) % sizeof(_Word) == 0);
  // Aligned to the items, which the item by item path stores into the words
  alignas(_InputT) alignas(_Word) _Word __words[(_Np * sizeof(_InputT)) / sizeof(_Word)];

  // The thread offset is computed before the tests, because computing it in the item by item path costs registers
  const _InputT* __thread_src = __block_src + (__rank * static_cast<_Index>(_VecSize));

  // Every vector starts a whole number of vectors after __block_src, thus a single test selects the accesses of all
  // vectors of the thread
  if constexpr (_Widths::__has(32))
  {
    NV_IF_TARGET(NV_PROVIDES_SM_100, ({
                   if (::cuda::is_aligned(__block_src, 32))
                   {
                     ::cuda::experimental::coop::__load_vectors<32, _VecSize, _Modifier>(
                       __block_src, __rank, __stride, __words, __dst);
                     return;
                   }
                 }))
  }
  if constexpr (_Widths::__has(16))
  {
    if (::cuda::is_aligned(__block_src, 16))
    {
      ::cuda::experimental::coop::__load_vectors<16, _VecSize, _Modifier>(__block_src, __rank, __stride, __words, __dst);
      return;
    }
  }
  if constexpr (_Widths::__has(8))
  {
    if (::cuda::is_aligned(__block_src, 8))
    {
      ::cuda::experimental::coop::__load_vectors<8, _VecSize, _Modifier>(__block_src, __rank, __stride, __words, __dst);
      return;
    }
  }
  if constexpr (_Widths::__has(4))
  {
    if (::cuda::is_aligned(__block_src, 4))
    {
      ::cuda::experimental::coop::__load_vectors<4, _VecSize, _Modifier>(__block_src, __rank, __stride, __words, __dst);
      return;
    }
  }

  // The items are loaded without the cache modifier
  ::cuda::experimental::coop::__load_items<_VecSize>(__thread_src, __stride, reinterpret_cast<_InputT(&)[_Np]>(__words));
  ::cuda::experimental::coop::__load_copy_words<_InputT, ::cub::LOAD_DEFAULT>(__words, __dst);
}

// Loads the vectors of a thread with accesses of _Width bytes, which __block_src must be aligned to, or item by item
// without the cache modifier if _Width is 0
template <::cuda::std::size_t _Width,
          ::cuda::std::size_t _VecSize,
          ::cub::CacheLoadModifier _Modifier,
          class _InputT,
          class _Index,
          class _Tp,
          ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
__load_static(const _InputT* __block_src, _Index __rank, _Index __stride, _Tp (&__dst)[_Np])
{
  if constexpr (_Width == 0)
  {
    ::cuda::experimental::coop::__load_items<_VecSize>(
      __block_src + (__rank * static_cast<_Index>(_VecSize)), __stride, __dst);
  }
  else
  {
    using _Word _CCCL_NODEBUG = __load_word_t<_InputT>;
    _Word __words[(_Np * sizeof(_InputT)) / sizeof(_Word)];
    ::cuda::experimental::coop::__load_vectors<_Width, _VecSize, _Modifier>(
      __block_src, __rank, __stride, __words, __dst);
  }
}

template <::cuda::std::size_t _KnownAlignment,
          ::cuda::std::size_t _VecSize,
          class _Group,
          class _InputIt,
          class _Tp,
          ::cuda::std::size_t _Np>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void __load(const _Group& __group, _InputIt __src, _Tp (&__dst)[_Np])
{
  static_assert(_Np % _VecSize == 0, "the vector size must evenly divide the number of items of a thread");

  using _Source _CCCL_NODEBUG = __load_source<_InputIt>;
  using _InputT _CCCL_NODEBUG = typename _Source::value_type;
  using _Index _CCCL_NODEBUG  = __load_index_t<_Group>;

  const _Index __rank   = ::cuda::gpu_thread.rank(__group);
  const _Index __stride = ::cuda::experimental::coop::__load_stride<_VecSize, _Np>(__group);

  if constexpr (__load_can_vectorize_v<_InputIt, _VecSize>)
  {
    using _Widths _CCCL_NODEBUG = __load_widths<_InputT, _VecSize, _Source::__modifier>;
    const _InputT* __block_src  = _Source::__pointer(__src);
    if constexpr (_KnownAlignment == 0)
    {
      ::cuda::experimental::coop::__load_dispatch<_VecSize, _Source::__modifier>(__block_src, __rank, __stride, __dst);
    }
    else
    {
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_100,
        (::cuda::experimental::coop::__load_static<_Widths::__widest(_KnownAlignment), _VecSize, _Source::__modifier>(
           __block_src, __rank, __stride, __dst);),
        (::cuda::experimental::coop::__load_static<
           _Widths::__widest(::cuda::std::min(_KnownAlignment, ::cuda::std::size_t{16})),
           _VecSize,
           _Source::__modifier>(__block_src, __rank, __stride, __dst);))
    }
  }
  else
  {
    ::cuda::experimental::coop::__load_items<_VecSize>(
      __src + (__rank * static_cast<_Index>(_VecSize)), __stride, __dst);
  }
}

//! @brief Loads a tile of ``P * _Np`` consecutive items into the ``P`` threads of a group, ``_Np`` items per thread,
//! in the given arrangement.
//!
//! The tile is loaded with the widest vectors that the alignment of ``__src`` allows, which the load tests at runtime,
//! or item by item. Pass an alignment flag to rely on a known alignment instead.
//!
//! Vectors are loaded from contiguous ranges of trivially copyable items that are not accessed through volatile
//! references, when the vectors of the arrangement allow accesses of at least 4 bytes that are wider than the items.
//! The items of ``__src`` are converted to ``_Tp``. A ``cub::CacheModifiedInputIterator`` loads its vectors with its
//! cache modifier, and its items without it, thus items are not loaded volatile even with ``LOAD_VOLATILE``.
//! ``LOAD_CG``, ``LOAD_CS``, ``LOAD_CV`` and ``LOAD_VOLATILE`` only use the widest accesses that the vectors allow.
//!
//! @param[in] __group The group, whose threads each call the load.
//! @param[in] __arrangement ``blocked``, ``striped`` or ``striped_vectors<V>``.
//! @param[in] __src The first item of the tile.
//! @param[out] __dst The items of the thread.
_CCCL_TEMPLATE(class _Group, class _Arrangement, class _InputIt, class _Tp, ::cuda::std::size_t _Np)
_CCCL_REQUIRES(group<_Group> _CCCL_AND __is_load_arrangement_v<_Arrangement>)
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
load(const _Group& __group, [[maybe_unused]] _Arrangement __arrangement, _InputIt __src, _Tp (&__dst)[_Np])
{
  ::cuda::experimental::coop::__load<0, __load_vec_size_v<_Arrangement, _Np>>(__group, __src, __dst);
}

//! @brief Loads a tile like ``load(__group, __arrangement, __src, __dst)``, relying on the alignment that
//! ``__alignment`` states instead of testing it.
//!
//! @param[in] __alignment ``element_aligned``, ``vector_aligned`` or ``overaligned<A>``. ``__src`` must be aligned as
//! stated.
_CCCL_TEMPLATE(class _Group, class _Arrangement, class _InputIt, class _Tp, ::cuda::std::size_t _Np, class _Alignment)
_CCCL_REQUIRES(group<_Group> _CCCL_AND __is_load_arrangement_v<_Arrangement> _CCCL_AND __is_load_alignment_v<_Alignment>)
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
load(const _Group& __group,
     [[maybe_unused]] _Arrangement __arrangement,
     _InputIt __src,
     _Tp (&__dst)[_Np],
     [[maybe_unused]] _Alignment __alignment)
{
  ::cuda::experimental::coop::__load<__load_known_alignment_v<_Alignment>, __load_vec_size_v<_Arrangement, _Np>>(
    __group, __src, __dst);
}

//! @brief Loads the first ``__valid_items`` items of a tile in the given arrangement, item by item through ``__src``.
//! The items of the thread at or after item ``__valid_items`` of the tile are not assigned.
_CCCL_TEMPLATE(class _Group, class _Arrangement, class _InputIt, class _Tp, ::cuda::std::size_t _Np, class _Size)
_CCCL_REQUIRES(
  group<_Group> _CCCL_AND __is_load_arrangement_v<_Arrangement> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Size>)
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
load(const _Group& __group,
     [[maybe_unused]] _Arrangement __arrangement,
     _InputIt __src,
     _Tp (&__dst)[_Np],
     _Size __valid_items)
{
  constexpr auto __vec_size  = __load_vec_size_v<_Arrangement, _Np>;
  using _Index _CCCL_NODEBUG = __load_index_t<_Group>;
  static_assert(_Np % __vec_size == 0, "the vector size must evenly divide the number of items of a thread");

  const _Index __thread_pos = ::cuda::gpu_thread.rank(__group) * static_cast<_Index>(__vec_size);
  ::cuda::experimental::coop::__load_items_guarded<__vec_size>(
    __src + __thread_pos,
    __thread_pos,
    ::cuda::experimental::coop::__load_stride<__vec_size, _Np>(__group),
    __valid_items,
    __dst);
}

//! @brief Loads the first ``__valid_items`` items of a tile like ``load(__group, __arrangement, __src, __dst,
//! __valid_items)``, and assigns ``__oob_default`` to the other items of the thread.
_CCCL_TEMPLATE(
  class _Group, class _Arrangement, class _InputIt, class _Tp, ::cuda::std::size_t _Np, class _Size, class _Default)
_CCCL_REQUIRES(
  group<_Group> _CCCL_AND __is_load_arrangement_v<_Arrangement> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Size>)
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
load(const _Group& __group,
     _Arrangement __arrangement,
     _InputIt __src,
     _Tp (&__dst)[_Np],
     _Size __valid_items,
     _Default __oob_default)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __i = 0; __i < static_cast<int>(_Np); ++__i)
  {
    __dst[__i] = __oob_default;
  }
  ::cuda::experimental::coop::load(__group, __arrangement, __src, __dst, __valid_items);
}

//! @brief Whether a load of ``_Np`` items per thread in the given arrangement reads vectors from ``__src`` with
//! ``vector_aligned``, because ``__src`` is aligned to the widest vectors that the load uses on the current target.
//!
//! Tiles that start a whole number of tiles after an aligned tile are aligned as well, thus a single test serves every
//! tile of a range.
_CCCL_TEMPLATE(class _Arrangement, class _InputIt, class _Tp, ::cuda::std::size_t _Np)
_CCCL_REQUIRES(__is_load_arrangement_v<_Arrangement>)
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
is_vector_aligned([[maybe_unused]] _Arrangement __arrangement, [[maybe_unused]] _InputIt __src, _Tp (&)[_Np])
{
  constexpr auto __vec_size = __load_vec_size_v<_Arrangement, _Np>;
  if constexpr (__load_can_vectorize_v<_InputIt, __vec_size>)
  {
    using _Widths _CCCL_NODEBUG = __load_widths_t<_InputIt, __vec_size>;
    const void* __ptr           = __load_source<_InputIt>::__pointer(__src);
    // Only sm_100 and newer access 32 bytes at once
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_100,
      (return ::cuda::experimental::coop::__load_is_aligned<_Widths::__widest(~::cuda::std::size_t{0})>(__ptr);),
      (return ::cuda::experimental::coop::__load_is_aligned<_Widths::__widest(16)>(__ptr);))
  }
  else
  {
    return false;
  }
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___ALGORITHM_LOAD_CUH
