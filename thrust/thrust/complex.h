/*
 *  Copyright 2008-2019 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
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

/*! \file complex.h
 *  \brief Complex numbers
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>

#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/std/complex>


THRUST_NAMESPACE_BEGIN

/*! \addtogroup numerics
 *  \{
 */

/*! \addtogroup complex_numbers Complex Numbers
 *  \{
 */

/*! \p complex is the Thrust equivalent to <tt>std::complex</tt>. It is
 *  functionally identical to it, but can also be used in device code which
 *  <tt>std::complex</tt> currently cannot.
 *
 *  \tparam T The type used to hold the real and imaginary parts. Should be
 *  <tt>float</tt> or <tt>double</tt>. Others types are not supported.
 *
 */
//template<class T>
//using complex = ::cuda::std::complex<T>;

template <class T>
struct complex: public ::cuda::std::complex<T> {
    using ::cuda::std::complex<T>::complex; // inherit constructors
};

// Enable arithmetic operators for different underlying types
// via type promoting.
template <typename T0, typename T1>
__host__ __device__
typename std::enable_if<!std::is_same<T0, T1>::value, 
                        complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator+(const complex<T0>& x, const complex<T1>& y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(T(x.real()) + T(y.real()), T(x.imag()) + T(y.imag()));
}

template <typename T0, typename T1>
__host__ __device__
typename std::enable_if<!std::is_same<T0, T1>::value, 
                        complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator-(const complex<T0>& x, const complex<T1>& y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(T(x.real()) - T(y.real()), T(x.imag()) - T(y.imag()));
}

template <typename T0, typename T1>
__host__ __device__
typename std::enable_if<!std::is_same<T0, T1>::value, 
                        complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator*(const complex<T0>& x, const complex<T1>& y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return complex<T>(T(x.real()) * T(y.real()) - T(x.imag()) * T(y.imag()), T(x.real()) * T(y.imag()) + T(x.imag()) * T(y.real()));
}

template <typename T0, typename T1>
__host__ __device__
typename std::enable_if<!std::is_same<T0, T1>::value, 
                        complex<typename detail::promoted_numerical_type<T0, T1>::type>>::type
operator/(const complex<T0>& x, const complex<T1>& y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;

  // Find `abs` by ADL.
  using std::abs;

  T s = abs(T(y.real())) + abs(T(y.imag()));

  T oos = T(1.0) / s;

  T ars = x.real() * oos;
  T ais = x.imag() * oos;
  T brs = y.real() * oos;
  T bis = y.imag() * oos;

  s = (brs * brs) + (bis * bis);

  oos = T(1.0) / s;

  return complex<T>(((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos);
}

template <typename T0, typename T1>
__host__ __device__
typename std::enable_if<!std::is_same<T0, T1>::value, bool>::type
operator==(const complex<T0>& x, const complex<T1>& y)
{
  return x.real() == y.real() && x.imag() == y.imag();
}

template <typename T0, typename T1>
__host__ __device__
typename std::enable_if<!std::is_same<T0, T1>::value, bool>::type
operator!=(const complex<T0>& x, const complex<T1>& y)
{
  return !(x == y);
}

// The using declarations allows imports all necessary functions for thurst::complex.
// However, they also lead to thrust::abs(1.0F) being valid code after include <thurst/complex.h>.
using ::cuda::std::abs;
using ::cuda::std::arg;
using ::cuda::std::norm;
using ::cuda::std::conj;
using ::cuda::std::polar;

using ::cuda::std::exp;
using ::cuda::std::log;
using ::cuda::std::log10;
using ::cuda::std::pow;
using ::cuda::std::sqrt;
using ::cuda::std::sin;
using ::cuda::std::cos;
using ::cuda::std::sinh;
using ::cuda::std::cosh;
using ::cuda::std::tan;
using ::cuda::std::tanh;
using ::cuda::std::asin;
using ::cuda::std::acos;
using ::cuda::std::atan;
using ::cuda::std::atanh;

using ::cuda::std::proj;
using ::cuda::std::polar;

template <typename T>
struct proclaim_trivially_relocatable<complex<T> > : thrust::true_type {};

THRUST_NAMESPACE_END

/*! \} // complex_numbers
 */

/*! \} // numerics
 */
