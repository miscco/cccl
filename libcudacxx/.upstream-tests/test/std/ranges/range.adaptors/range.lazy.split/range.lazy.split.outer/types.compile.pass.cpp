//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// using iterator_category = input_iterator_tag; // Only defined if `View` is a forward range.
// using iterator_concept = conditional_t<forward_range<Base>, forward_iterator_tag, input_iterator_tag>;
// using difference_type = range_difference_t<Base>;

#include <cuda/std/ranges>

#include <cuda/std/concepts>
#include <cuda/std/iterator>
#include "../types.h"

template <class Range, class Pattern>
using OuterIter = decltype(cuda::std::declval<cuda::std::ranges::lazy_split_view<Range, Pattern>>().begin());

// iterator_category

static_assert(cuda::std::same_as<typename OuterIter<ForwardView, ForwardView>::iterator_category, cuda::std::input_iterator_tag>);

template <class Range, class Pattern>
concept NoIteratorCategory = !requires { typename OuterIter<Range, Pattern>::iterator_category; };
static_assert(NoIteratorCategory<InputView, ForwardTinyView>);

// iterator_concept

static_assert(cuda::std::same_as<typename OuterIter<ForwardView, ForwardView>::iterator_concept, cuda::std::forward_iterator_tag>);
static_assert(cuda::std::same_as<typename OuterIter<InputView, ForwardTinyView>::iterator_concept, cuda::std::input_iterator_tag>);

// difference_type

static_assert(cuda::std::same_as<typename OuterIter<ForwardView, ForwardView>::difference_type,
    cuda::std::ranges::range_difference_t<ForwardView>>);
