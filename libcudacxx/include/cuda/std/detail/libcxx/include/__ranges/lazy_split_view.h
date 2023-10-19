// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_LAZY_SPLIT_VIEW_H
#define _LIBCUDACXX___RANGES_LAZY_SPLIT_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__algorithm/ranges_find.h"
#include "../__algorithm/ranges_mismatch.h"
#include "../__concepts/constructible.h"
#include "../__concepts/convertible_to.h"
#include "../__concepts/derived_from.h"
#include "../__functional/bind_back.h"
#include "../__functional/ranges_operations.h"
#include "../__iterator/concepts.h"
#include "../__iterator/default_sentinel.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/indirectly_comparable.h"
#include "../__iterator/iter_move.h"
#include "../__iterator/iter_swap.h"
#include "../__iterator/iterator_traits.h"
#include "../__memory/addressof.h"
#include "../__ranges/access.h"
#include "../__ranges/all.h"
#include "../__ranges/concepts.h"
#include "../__ranges/non_propagating_cache.h"
#include "../__ranges/range_adaptor.h"
#include "../__ranges/single_view.h"
#include "../__ranges/subrange.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/maybe_const.h"
#include "../__type_traits/remove_reference.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC
#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

template <auto> struct __require_constant;

#if _LIBCUDACXX_STD_VER > 17
template <class _Range>
concept __tiny_range =
  sized_range<_Range> &&
  requires { typename __require_constant<remove_reference_t<_Range>::size()>; } &&
  (remove_reference_t<_Range>::size() <= 1);
#else
  template<class _Range>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __tiny_range_,
    requires()(
      requires(sized_range<_Range>),
      typename(typename __require_constant<remove_reference_t<_Range>::size()>)
      requires(remove_reference_t<_Range>::size() <= 1)
    ));

  template<class _Range>
  _LIBCUDACXX_CONCEPT __tiny_range = _LIBCUDACXX_FRAGMENT(__tiny_range_, _Range);
#endif

#if _LIBCUDACXX_STD_VER > 17
template <input_range _View, forward_range _Pattern>
  requires view<_View> && view<_Pattern> &&
           indirectly_comparable<iterator_t<_View>, iterator_t<_Pattern>, _CUDA_VRANGES::equal_to> &&
           (forward_range<_View> || __tiny_range<_Pattern>)
#else
template <class _View, class _Pattern, enable_if_t<input_range<_View>, int> = 0
                                     , enable_if_t<forward_range<_Pattern>, int> = 0
                                     , enable_if_t<view<_View>, int> = 0
                                     , enable_if_t<view<_Pattern>, int> = 0
                                     , enable_if_t<indirectly_comparable<iterator_t<_View>, iterator_t<_Pattern>, _CUDA_VRANGES::equal_to>, int> = 0
                                     , enable_if_t<(forward_range<_View> || __tiny_range<_Pattern>), int> = 0>

#endif
class lazy_split_view : public view_interface<lazy_split_view<_View, _Pattern>> {

  _LIBCUDACXX_NO_UNIQUE_ADDRESS _View __base_ = _View();
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Pattern __pattern_ = _Pattern();

  using _MaybeCurrent = _If<!forward_range<_View>, __non_propagating_cache<iterator_t<_View>>, __empty_cache>;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _MaybeCurrent __current_ = _MaybeCurrent();

  template <bool> struct __outer_iterator;
  template <bool> struct __inner_iterator;

public:
#if _LIBCUDACXX_STD_VER > 17
  _LIBCUDACXX_HIDE_FROM_ABI
  lazy_split_view() requires default_initializable<_View> && default_initializable<_Pattern> = default;
#else
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    _LIBCUDACXX_REQUIRES(default_initializable<_View2> _LIBCUDACXX_AND default_initializable<_Pattern>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr lazy_split_view() noexcept(is_nothrow_default_constructible_v<_View2> && is_nothrow_default_constructible_v<_Pattern>)
    : view_interface<lazy_split_view<_View, _Pattern>>() {}
#endif

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr lazy_split_view(_View __base, _Pattern __pattern)
    : view_interface<lazy_split_view<_View, _Pattern>>()
    , __base_(_CUDA_VSTD::move(__base)), __pattern_(_CUDA_VSTD::move(__pattern)) {}

  _LIBCUDACXX_TEMPLATE(class _Range)
    _LIBCUDACXX_REQUIRES(input_range<_Range> _LIBCUDACXX_AND
              constructible_from<_View, views::all_t<_Range>> _LIBCUDACXX_AND
              constructible_from<_Pattern, single_view<range_value_t<_Range>>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr lazy_split_view(_Range&& __r, range_value_t<_Range> __e)
    : view_interface<lazy_split_view<_View, _Pattern>>()
    , __base_(views::all(_CUDA_VSTD::forward<_Range>(__r)))
    , __pattern_(views::single(_CUDA_VSTD::move(__e))) {}

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    _LIBCUDACXX_REQUIRES(copy_constructible<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() const& noexcept(is_nothrow_copy_constructible_v<_View2>) { return __base_; }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() && noexcept(is_nothrow_move_constructible_v<_View>) { return _CUDA_VSTD::move(__base_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto begin() {
    if constexpr (forward_range<_View>) {
      return __outer_iterator<__simple_view<_View> && __simple_view<_Pattern>>{*this, _CUDA_VRANGES::begin(__base_)};
    } else {
      __current_.__emplace(_CUDA_VRANGES::begin(__base_));
      return __outer_iterator<false>{*this};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    _LIBCUDACXX_REQUIRES(forward_range<_View2> && forward_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto begin() const {
    return __outer_iterator<true>{*this, _CUDA_VRANGES::begin(__base_)};
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    _LIBCUDACXX_REQUIRES(forward_range<_View2> && common_range<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto end() {
    return __outer_iterator<__simple_view<_View> && __simple_view<_Pattern>>{*this, _CUDA_VRANGES::end(__base_)};
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto end() const {
    if constexpr (forward_range<_View> && forward_range<const _View> && common_range<const _View>) {
      return __outer_iterator<true>{*this, _CUDA_VRANGES::end(__base_)};
    } else {
      return default_sentinel;
    }
  }

private:

  template <class, class = void>
  struct __outer_iterator_category {};

  template <class _Tp>
  struct __outer_iterator_category<_Tp, enable_if_t<forward_range<_Tp>>> {
    using iterator_category = input_iterator_tag;
  };

  template <bool _Const>
  struct __outer_iterator : __outer_iterator_category<__maybe_const<_Const, _View>> {
  private:
    template <bool>
    friend struct __inner_iterator;
    friend __outer_iterator<true>;

    using _Parent = __maybe_const<_Const, lazy_split_view>;
    using _Base = __maybe_const<_Const, _View>;

    _Parent* __parent_ = nullptr;
    using _MaybeCurrent = _If<forward_range<_View>, iterator_t<_Base>, __empty_cache>;
    _LIBCUDACXX_NO_UNIQUE_ADDRESS _MaybeCurrent __current_ = _MaybeCurrent();
    bool __trailing_empty_ = false;

    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto& __current() noexcept {
      if constexpr (forward_range<_View>) {
        return __current_;
      } else {
        return *__parent_->__current_;
      }
    }

    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr const auto& __current() const noexcept {
      if constexpr (forward_range<_View>) {
        return __current_;
      } else {
        return *__parent_->__current_;
      }
    }

    // Workaround for the GCC issue that doesn't allow calling `__parent_->__base_` from friend functions (because
    // `__base_` is private).
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto& __parent_base() const noexcept {
      return __parent_->__base_;
    }

  public:
    // using iterator_category = inherited;
    using iterator_concept = conditional_t<forward_range<_Base>, forward_iterator_tag, input_iterator_tag>;
    using difference_type = range_difference_t<_Base>;

    struct value_type : view_interface<value_type> {
    private:
      __outer_iterator __i_ = __outer_iterator();

    public:
      _LIBCUDACXX_HIDE_FROM_ABI
      value_type() = default;
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr explicit value_type(__outer_iterator __i)
        : __i_(_CUDA_VSTD::move(__i)) {}

      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr __inner_iterator<_Const> begin() const { return __inner_iterator<_Const>{__i_}; }
      _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
      constexpr default_sentinel_t end() const noexcept { return default_sentinel; }
    };

    _LIBCUDACXX_HIDE_FROM_ABI
    __outer_iterator() = default;

    _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
      _LIBCUDACXX_REQUIRES((!forward_range<__maybe_const<_OtherConst, _View>>))
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr explicit __outer_iterator(_Parent& __parent)
      : __parent_(_CUDA_VSTD::addressof(__parent)) {}

    _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
      _LIBCUDACXX_REQUIRES(forward_range<__maybe_const<_OtherConst, _View>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __outer_iterator(_Parent& __parent, iterator_t<_Base> __current)
      : __parent_(_CUDA_VSTD::addressof(__parent)), __current_(_CUDA_VSTD::move(__current)) {}

    _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
      _LIBCUDACXX_REQUIRES(_OtherConst _LIBCUDACXX_AND convertible_to<iterator_t<_View>, iterator_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __outer_iterator(__outer_iterator<!_OtherConst> __i)
      : __parent_(__i.__parent_), __current_(_CUDA_VSTD::move(__i.__current_)) {}

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr value_type operator*() const { return value_type{*this}; }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __outer_iterator& operator++() {
      const auto __end = _CUDA_VRANGES::end(__parent_->__base_);
      if (__current() == __end) {
        __trailing_empty_ = false;
        return *this;
      }

      const auto [__pbegin, __pend] = _CUDA_VRANGES::subrange{__parent_->__pattern_};
      if (__pbegin == __pend) {
        // Empty pattern: split on every element in the input range
        ++__current();

      } else if constexpr (__tiny_range<_Pattern>) {
        // One-element pattern: we can use `_CUDA_VRANGES::find`.
        __current() = _CUDA_VRANGES::find(_CUDA_VSTD::move(__current()), __end, *__pbegin);
        if (__current() != __end) {
          // Make sure we point to after the separator we just found.
          ++__current();
          if (__current() == __end)
            __trailing_empty_ = true;
        }

      } else {
        // General case for n-element pattern.
        do {
          const auto [__b, __p] = _CUDA_VRANGES::mismatch(__current(), __end, __pbegin, __pend);
          if (__p == __pend) {
            __current() = __b;
            if (__current() == __end) {
              __trailing_empty_ = true;
            }
            break; // The pattern matched; skip it.
          }
        } while (++__current() != __end);
      }

      return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr decltype(auto) operator++(int) {
      if constexpr (forward_range<_Base>) {
        auto __tmp = *this;
        ++*this;
        return __tmp;

      } else {
        ++*this;
      }
    }

    template<bool _OtherConst = _Const>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr bool operator==(const __outer_iterator& __x, const __outer_iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(forward_range<__maybe_const<_OtherConst, _View>>) {
      return __x.__current_ == __y.__current_ && __x.__trailing_empty_ == __y.__trailing_empty_;
    }
#if _LIBCUDACXX_STD_VER < 20
    template<bool _OtherConst = _Const>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr bool operator!=(const __outer_iterator& __x, const __outer_iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(forward_range<__maybe_const<_OtherConst, _View>>) {
      return __x.__current_ != __y.__current_ || __x.__trailing_empty_ != __y.__trailing_empty_;
    }
#endif

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr bool operator==(const __outer_iterator& __x, default_sentinel_t)
      _LIBCUDACXX_ASSERT(__x.__parent_, "Cannot call comparison on a default-constructed iterator.");
      return __x.__current() == _CUDA_VRANGES::end(__x.__parent_base()) && !__x.__trailing_empty_;
    }
#if _LIBCUDACXX_STD_VER < 20
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr bool operator==(default_sentinel_t, const __outer_iterator& __x)
      _LIBCUDACXX_ASSERT(__x.__parent_, "Cannot call comparison on a default-constructed iterator.");
      return __x.__current() == _CUDA_VRANGES::end(__x.__parent_base()) && !__x.__trailing_empty_;
    }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr bool operator!=(const __outer_iterator& __x, default_sentinel_t)
      _LIBCUDACXX_ASSERT(__x.__parent_, "Cannot call comparison on a default-constructed iterator.");
      return __x.__current() != _CUDA_VRANGES::end(__x.__parent_base()) || __x.__trailing_empty_;
    }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr bool operator!=(default_sentinel_t, const __outer_iterator& __x)
      _LIBCUDACXX_ASSERT(__x.__parent_, "Cannot call comparison on a default-constructed iterator.");
      return __x.__current() != _CUDA_VRANGES::end(__x.__parent_base()) || __x.__trailing_empty_;
    }
#endif
  };

  template <class, class = void>
  struct __inner_iterator_category {};

  template <class _Tp>
  struct __inner_iterator_category<_Tp, enable_if_t<forward_range<_Tp>>> {
    using iterator_category = _If<
      derived_from<typename iterator_traits<iterator_t<_Tp>>::iterator_category, forward_iterator_tag>,
      forward_iterator_tag,
      typename iterator_traits<iterator_t<_Tp>>::iterator_category
    >;
  };

  template <bool _Const>
  struct __inner_iterator : __inner_iterator_category<__maybe_const<_Const, _View>> {
  private:
    using _Base = __maybe_const<_Const, _View>;
    // Workaround for a GCC issue.
    static constexpr bool _OuterConst = _Const;
    __outer_iterator<_Const> __i_ = __outer_iterator<_OuterConst>();
    bool __incremented_ = false;

    // Note: these private functions are necessary because GCC doesn't allow calls to private members of `__i_` from
    // free functions that are friends of `inner-iterator`.

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr bool __is_done() const {
      _LIBCUDACXX_ASSERT(__i_.__parent_, "Cannot call comparison on a default-constructed iterator.");

      auto [__pcur, __pend] = _CUDA_VRANGES::subrange{__i_.__parent_->__pattern_};
      auto __end = _CUDA_VRANGES::end(__i_.__parent_->__base_);

      if constexpr (__tiny_range<_Pattern>) {
        const auto& __cur = __i_.__current();
        if (__cur == __end)
          return true;
        if (__pcur == __pend)
          return __incremented_;

        return *__cur == *__pcur;

      } else {
        auto __cur = __i_.__current();
        if (__cur == __end)
          return true;
        if (__pcur == __pend)
          return __incremented_;

        do {
          if (*__cur != *__pcur)
            return false;
          if (++__pcur == __pend)
            return true;
        } while (++__cur != __end);

        return false;
      }
    }

    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto& __outer_current() noexcept {
      return __i_.__current();
    }

    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr const auto& __outer_current() const noexcept {
      return __i_.__current();
    }

  public:
    // using iterator_category = inherited;
    using iterator_concept = typename __outer_iterator<_Const>::iterator_concept;
    using value_type = range_value_t<_Base>;
    using difference_type = range_difference_t<_Base>;

    _LIBCUDACXX_HIDE_FROM_ABI
    __inner_iterator() = default;

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr explicit __inner_iterator(__outer_iterator<_Const> __i)
      : __i_(_CUDA_VSTD::move(__i)) {}

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr const iterator_t<_Base>& base() const& noexcept { return __i_.__current(); }
    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
      _LIBCUDACXX_REQUIRES(forward_range<_View2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr iterator_t<_Base> base() && noexcept(is_nothrow_move_constructible_v<iterator_t<_Base>>)
    { return _CUDA_VSTD::move(__i_.__current()); }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr decltype(auto) operator*() const { return *__i_.__current(); }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __inner_iterator& operator++() {
      __incremented_ = true;

      if constexpr (!forward_range<_Base>) {
        if constexpr (_Pattern::size() == 0) {
          return *this;
        }
      }

      ++__i_.__current();
      return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr decltype(auto) operator++(int) {
      if constexpr (forward_range<_Base>) {
        auto __tmp = *this;
        ++*this;
        return __tmp;

      } else {
        ++*this;
      }
    }

    template <bool _OtherConst = _Const>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator==(const __inner_iterator& __x, const __inner_iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(forward_range<__maybe_const<_OtherConst, _View>>) {
      return __x.__outer_current() == __y.__outer_current();
    }
#if _LIBCUDACXX_STD_VER < 20
    template <bool _OtherConst = _Const>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator!=(const __inner_iterator& __x, const __inner_iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(forward_range<__maybe_const<_OtherConst, _View>>) {
      return __x.__outer_current() != __y.__outer_current();
    }
#endif

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr bool operator==(const __inner_iterator& __x, default_sentinel_t) {
      return __x.__is_done();
    }
#if _LIBCUDACXX_STD_VER < 20
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr bool operator==(default_sentinel_t, const __inner_iterator& __x) {
      return __x.__is_done();
    }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr bool operator!=(const __inner_iterator& __x, default_sentinel_t) {
      return !__x.__is_done();
    }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr bool operator!=(default_sentinel_t, const __inner_iterator& __x) {
      return !__x.__is_done();
    }
#endif

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr decltype(auto) iter_move(const __inner_iterator& __i)
        noexcept(noexcept(_CUDA_VRANGES::iter_move(__i.__outer_current()))) {
      return _CUDA_VRANGES::iter_move(__i.__outer_current());
    }

    template <bool _OtherConst = _Const>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto iter_swap(const __inner_iterator& __x, const __inner_iterator& __y)
      noexcept(noexcept(_CUDA_VRANGES::iter_swap(__x.__outer_current(), __y.__outer_current())))
      _LIBCUDACXX_TRAILING_REQUIRES(void)(indirectly_swappable<iterator_t<__maybe_const<_OtherConst, _View>>>) {
      _CUDA_VRANGES::iter_swap(__x.__outer_current(), __y.__outer_current());
    }
  };
};

template <class _Range, class _Pattern>
lazy_split_view(_Range&&, _Pattern&&) -> lazy_split_view<views::all_t<_Range>, views::all_t<_Pattern>>;

template <class _Range, enable_if_t<input_range<_Range>, int> = 0>
lazy_split_view(_Range&&, range_value_t<_Range>)
  -> lazy_split_view<views::all_t<_Range>, single_view<range_value_t<_Range>>>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__lazy_split_view)
struct __fn : __range_adaptor_closure<__fn> {
  template <class _Range, class _Pattern>
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Range&& __range, _Pattern&& __pattern) const
    noexcept(noexcept(lazy_split_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pattern>(__pattern))))
    -> decltype(      lazy_split_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pattern>(__pattern)))
    { return          lazy_split_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pattern>(__pattern)); }

  _LIBCUDACXX_TEMPLATE(class _Pattern)
    _LIBCUDACXX_REQUIRES(constructible_from<decay_t<_Pattern>, _Pattern>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Pattern&& __pattern) const
      noexcept(is_nothrow_constructible_v<decay_t<_Pattern>, _Pattern>) {
    return __range_adaptor_closure_t(_CUDA_VSTD::__bind_back(*this, _CUDA_VSTD::forward<_Pattern>(__pattern)));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto lazy_split = __lazy_split_view::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___RANGES_LAZY_SPLIT_VIEW_H
