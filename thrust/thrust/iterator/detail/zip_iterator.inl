/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/tuple_transform.h>
#include <thrust/iterator/zip_iterator.h>

THRUST_NAMESPACE_BEGIN

template <typename IteratorTuple>
_CCCL_HOST_DEVICE zip_iterator<IteratorTuple>::zip_iterator(IteratorTuple iterator_tuple)
    : m_iterator_tuple(iterator_tuple)
{} // end zip_iterator::zip_iterator()

template <typename IteratorTuple>
_CCCL_HOST_DEVICE const IteratorTuple& zip_iterator<IteratorTuple>::get_iterator_tuple() const
{
  return m_iterator_tuple;
} // end zip_iterator::get_iterator_tuple()

template <typename IteratorTuple>
typename zip_iterator<IteratorTuple>::super_t::reference _CCCL_HOST_DEVICE
zip_iterator<IteratorTuple>::dereference() const
{
  using namespace detail::tuple_impl_specific;

  return thrust::detail::tuple_host_device_transform<detail::dereference_iterator::template apply>(
    get_iterator_tuple(), detail::dereference_iterator());
} // end zip_iterator::dereference()

_CCCL_EXEC_CHECK_DISABLE
template <typename IteratorTuple>
template <typename OtherIteratorTuple>
_CCCL_HOST_DEVICE bool zip_iterator<IteratorTuple>::equal(const zip_iterator<OtherIteratorTuple>& other) const
{
  return get<0>(get_iterator_tuple()) == get<0>(other.get_iterator_tuple());
} // end zip_iterator::equal()

template <typename IteratorTuple>
_CCCL_HOST_DEVICE void zip_iterator<IteratorTuple>::advance(typename super_t::difference_type n)
{
  using namespace detail::tuple_impl_specific;
  tuple_for_each(m_iterator_tuple, detail::advance_iterator<typename super_t::difference_type>(n));
} // end zip_iterator::advance()

template <typename IteratorTuple>
_CCCL_HOST_DEVICE void zip_iterator<IteratorTuple>::increment()
{
  using namespace detail::tuple_impl_specific;
  tuple_for_each(m_iterator_tuple, detail::increment_iterator());
} // end zip_iterator::increment()

template <typename IteratorTuple>
_CCCL_HOST_DEVICE void zip_iterator<IteratorTuple>::decrement()
{
  using namespace detail::tuple_impl_specific;
  tuple_for_each(m_iterator_tuple, detail::decrement_iterator());
} // end zip_iterator::decrement()

_CCCL_EXEC_CHECK_DISABLE
template <typename IteratorTuple>
template <typename OtherIteratorTuple>
_CCCL_HOST_DEVICE typename zip_iterator<IteratorTuple>::super_t::difference_type
zip_iterator<IteratorTuple>::distance_to(const zip_iterator<OtherIteratorTuple>& other) const
{
  return get<0>(other.get_iterator_tuple()) - get<0>(get_iterator_tuple());
} // end zip_iterator::distance_to()

template <typename... Iterators>
_CCCL_HOST_DEVICE zip_iterator<_CUDA_VSTD::tuple<Iterators...>> make_zip_iterator(_CUDA_VSTD::tuple<Iterators...> t)
{
  return zip_iterator<_CUDA_VSTD::tuple<Iterators...>>(t);
} // end make_zip_iterator()

template <typename... Iterators>
_CCCL_HOST_DEVICE zip_iterator<_CUDA_VSTD::tuple<Iterators...>> make_zip_iterator(Iterators... its)
{
  return make_zip_iterator(_CUDA_VSTD::make_tuple(its...));
} // end make_zip_iterator()

THRUST_NAMESPACE_END
